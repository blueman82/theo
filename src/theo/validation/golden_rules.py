"""Golden rules management for validation loop.

This module provides the GoldenRules class for managing high-confidence
patterns that should never decay. Golden rules represent validated
knowledge that has consistently proven useful.

A document becomes a golden rule when:
1. Confidence reaches 0.9 or higher
2. Type is promotable (preference, decision, or pattern)

Golden rules are special because:
- They never decay (always maintain high confidence)
- They appear in context injection regardless of recency
- They can be demoted if they start failing
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from theo.constants import DEMOTION_FAILURE_THRESHOLD, GOLDEN_RULE_THRESHOLD
from theo.storage import SQLiteStore, SQLiteStoreError

logger = logging.getLogger(__name__)

# Types that can be promoted to golden rules
PROMOTABLE_TYPES = {"preference", "decision", "pattern"}


@dataclass
class GoldenRule:
    """Representation of a golden rule document.

    Attributes:
        doc_id: Document ID
        content: Document content
        original_type: Original type before promotion
        promoted_at: When the document was promoted
        confidence: Current confidence score
        validation_count: Number of successful validations
        failure_count: Number of failures since promotion
        metadata: Additional metadata
    """

    doc_id: str
    content: str
    original_type: str
    promoted_at: datetime = field(default_factory=datetime.now)
    confidence: float = GOLDEN_RULE_THRESHOLD
    validation_count: int = 0
    failure_count: int = 0
    metadata: Optional[dict[str, Any]] = None


@dataclass
class DemotionResult:
    """Result of attempting to demote a golden rule.

    Attributes:
        success: Whether the operation succeeded
        doc_id: Document ID
        demoted: Whether the document was actually demoted
        new_confidence: Confidence after demotion
        restored_type: Type after demotion (original type)
        reason: Reason for demotion/non-demotion
        error: Error message if operation failed
    """

    success: bool
    doc_id: str
    demoted: bool = False
    new_confidence: float = 0.0
    restored_type: Optional[str] = None
    reason: str = ""
    error: Optional[str] = None


class GoldenRules:
    """Manager for golden rule documents.

    Provides operations for:
    - Listing current golden rules
    - Checking if a document is a golden rule
    - Demoting golden rules that have started failing
    - Tracking golden rule statistics

    Golden rules are identified by:
    - memory_type == 'golden_rule', OR
    - confidence >= 0.9

    Args:
        store: SQLiteStore instance for persistence

    Example:
        >>> store = SQLiteStore()
        >>> rules = GoldenRules(store)
        >>> # Get all golden rules
        >>> all_rules = rules.get_all()
        >>> # Check if specific doc is golden
        >>> is_golden = rules.is_golden_rule("doc_123")
        >>> # Demote a failing golden rule
        >>> result = await rules.demote("doc_123", reason="Contradicted by user")
    """

    def __init__(self, store: SQLiteStore):
        """Initialize GoldenRules manager.

        Args:
            store: SQLiteStore instance for persistence
        """
        self._store = store
        # In-memory tracking of failure counts (could be persisted)
        self._failure_counts: dict[str, int] = {}

    def get_all(self, namespace: Optional[str] = None) -> list[GoldenRule]:
        """Get all golden rules.

        Returns documents that are either:
        - Type == 'golden_rule'
        - Confidence >= 0.9

        Args:
            namespace: Optional namespace filter

        Returns:
            List of GoldenRule objects
        """
        golden_rules: list[GoldenRule] = []

        try:
            # Get golden rules by type
            type_memories = self._store.list_memories(
                namespace=namespace,
                memory_type="golden_rule",
                limit=10000,
            )

            seen_ids: set[str] = set()

            # Process type-based golden rules
            for memory in type_memories:
                doc_id = memory["id"]
                seen_ids.add(doc_id)
                content = memory.get("content", "")
                golden_rules.append(self._to_golden_rule(doc_id, content, memory))

            # Also get high-confidence documents
            all_memories = self._store.list_memories(namespace=namespace, limit=10000)

            for memory in all_memories:
                doc_id = memory["id"]
                if doc_id in seen_ids:
                    continue

                confidence = memory.get("confidence", 0.3)

                if confidence >= GOLDEN_RULE_THRESHOLD:
                    content = memory.get("content", "")
                    golden_rules.append(self._to_golden_rule(doc_id, content, memory))

            return golden_rules

        except SQLiteStoreError as e:
            logger.error(f"Error getting golden rules: {e}")
            return []

    def get_count(self, namespace: Optional[str] = None) -> int:
        """Get count of golden rules.

        Args:
            namespace: Optional namespace filter

        Returns:
            Number of golden rules
        """
        return len(self.get_all(namespace))

    def is_golden_rule(self, doc_id: str) -> bool:
        """Check if a document is a golden rule.

        A document is a golden rule if:
        - confidence >= 0.9 OR
        - memory_type == 'golden_rule'

        Args:
            doc_id: Document ID to check

        Returns:
            True if document is a golden rule
        """
        try:
            memory = self._store.get_memory(doc_id)

            if memory is None:
                return False

            memory_type = memory.get("memory_type", "document")
            confidence = memory.get("confidence", 0.3)

            return memory_type == "golden_rule" or confidence >= GOLDEN_RULE_THRESHOLD

        except SQLiteStoreError:
            return False

    def record_failure(self, doc_id: str) -> int:
        """Record a failure for a golden rule.

        Tracks failures to determine if demotion is warranted.

        Args:
            doc_id: Document ID that failed

        Returns:
            New failure count
        """
        self._failure_counts[doc_id] = self._failure_counts.get(doc_id, 0) + 1
        return self._failure_counts[doc_id]

    def reset_failures(self, doc_id: str) -> None:
        """Reset failure count for a document.

        Call when a golden rule succeeds to reset the failure counter.

        Args:
            doc_id: Document ID to reset
        """
        if doc_id in self._failure_counts:
            del self._failure_counts[doc_id]

    def should_demote(self, doc_id: str) -> bool:
        """Check if a golden rule should be demoted.

        Args:
            doc_id: Document ID to check

        Returns:
            True if failures exceed threshold
        """
        return self._failure_counts.get(doc_id, 0) >= DEMOTION_FAILURE_THRESHOLD

    async def demote(
        self,
        doc_id: str,
        reason: str,
        new_confidence: float = 0.7,
    ) -> DemotionResult:
        """Demote a golden rule back to its original type.

        Reduces confidence and (conceptually) restores the original type.
        The document will need to re-earn golden rule status.

        Args:
            doc_id: Document ID to demote
            reason: Reason for demotion
            new_confidence: Confidence to set after demotion (default: 0.7)

        Returns:
            DemotionResult with operation outcome
        """
        try:
            # Verify document exists and is a golden rule
            memory = self._store.get_memory(doc_id)

            if memory is None:
                return DemotionResult(
                    success=False,
                    doc_id=doc_id,
                    error=f"Document '{doc_id}' not found",
                )

            memory_type = memory.get("memory_type", "document")
            confidence = memory.get("confidence", 0.3)

            # Check if actually a golden rule
            if memory_type != "golden_rule" and confidence < GOLDEN_RULE_THRESHOLD:
                return DemotionResult(
                    success=True,
                    doc_id=doc_id,
                    demoted=False,
                    reason="Document is not a golden rule",
                )

            # Get original type if stored in tags/metadata
            tags = memory.get("tags") or {}
            original_type = tags.get("promoted_from", "document")

            # Update confidence (demotion) using SQLiteStore.update_memory()
            updated = self._store.update_memory(doc_id, confidence=new_confidence)

            if not updated:
                return DemotionResult(
                    success=False,
                    doc_id=doc_id,
                    error="Failed to update confidence",
                )

            # Reset failure count
            self.reset_failures(doc_id)

            logger.info(
                f"Demoted golden rule {doc_id}: {confidence:.2f} -> {new_confidence:.2f}, "
                f"reason: {reason}"
            )

            return DemotionResult(
                success=True,
                doc_id=doc_id,
                demoted=True,
                new_confidence=new_confidence,
                restored_type=original_type,
                reason=reason,
            )

        except SQLiteStoreError as e:
            logger.error(f"Storage error demoting {doc_id}: {e}")
            return DemotionResult(
                success=False,
                doc_id=doc_id,
                error=f"Storage error: {e}",
            )
        except Exception as e:
            logger.error(f"Error demoting {doc_id}: {e}")
            return DemotionResult(
                success=False,
                doc_id=doc_id,
                error=f"Unexpected error: {e}",
            )

    def get_stats(self, namespace: Optional[str] = None) -> dict[str, Any]:
        """Get golden rule statistics.

        Args:
            namespace: Optional namespace filter

        Returns:
            Dictionary with statistics
        """
        rules = self.get_all(namespace)

        if not rules:
            return {
                "total_golden_rules": 0,
                "by_original_type": {},
                "avg_confidence": 0.0,
                "at_risk_count": 0,  # Rules with failures
            }

        by_type: dict[str, int] = {}
        total_confidence = 0.0
        at_risk = 0

        for rule in rules:
            by_type[rule.original_type] = by_type.get(rule.original_type, 0) + 1
            total_confidence += rule.confidence

            if self._failure_counts.get(rule.doc_id, 0) > 0:
                at_risk += 1

        return {
            "total_golden_rules": len(rules),
            "by_original_type": by_type,
            "avg_confidence": total_confidence / len(rules) if rules else 0.0,
            "at_risk_count": at_risk,
        }

    def _to_golden_rule(
        self,
        doc_id: str,
        content: str,
        memory: dict[str, Any],
    ) -> GoldenRule:
        """Convert storage data to GoldenRule object.

        Args:
            doc_id: Document ID
            content: Document content
            memory: Memory dictionary from SQLiteStore

        Returns:
            GoldenRule instance
        """
        # Parse promoted_at timestamp from created_at (Unix timestamp)
        promoted_at = datetime.now()
        created_at = memory.get("created_at")
        if created_at is not None:
            try:
                promoted_at = datetime.fromtimestamp(created_at)
            except (ValueError, TypeError, OSError):
                pass

        # Get original type from tags if available
        tags = memory.get("tags") or {}
        original_type = tags.get("promoted_from", memory.get("memory_type", "document"))

        return GoldenRule(
            doc_id=doc_id,
            content=content,
            original_type=original_type,
            promoted_at=promoted_at,
            confidence=memory.get("confidence", GOLDEN_RULE_THRESHOLD),
            validation_count=memory.get("access_count", 0),
            failure_count=self._failure_counts.get(doc_id, 0),
            metadata=memory,
        )
