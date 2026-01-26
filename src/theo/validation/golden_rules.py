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
from theo.storage import ChromaStore, StorageError

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
    - doc_type == 'golden_rule', OR
    - confidence >= 0.9

    Args:
        store: ChromaStore instance for persistence

    Example:
        >>> store = ChromaStore(ephemeral=True)
        >>> rules = GoldenRules(store)
        >>> # Get all golden rules
        >>> all_rules = rules.get_all()
        >>> # Check if specific doc is golden
        >>> is_golden = rules.is_golden_rule("doc_123")
        >>> # Demote a failing golden rule
        >>> result = await rules.demote("doc_123", reason="Contradicted by user")
    """

    def __init__(self, store: ChromaStore):
        """Initialize GoldenRules manager.

        Args:
            store: ChromaStore instance for persistence
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
            # Get by type
            where_filter: dict[str, Any] = {"doc_type": "golden_rule"}
            if namespace:
                where_filter["namespace"] = namespace

            type_results = self._store.get(where=where_filter)

            seen_ids: set[str] = set()

            # Process type-based golden rules
            for i, doc_id in enumerate(type_results["ids"]):
                seen_ids.add(doc_id)
                content = type_results["documents"][i] if type_results["documents"] else ""
                metadata = type_results["metadatas"][i] if type_results["metadatas"] else {}

                golden_rules.append(self._to_golden_rule(doc_id, content, metadata))

            # Also get high-confidence documents
            # Note: ChromaDB doesn't support >= in where filters well,
            # so we fetch all and filter
            all_filter: dict[str, Any] = {}
            if namespace:
                all_filter["namespace"] = namespace

            all_results = self._store.get(where=all_filter if all_filter else None)

            for i, doc_id in enumerate(all_results["ids"]):
                if doc_id in seen_ids:
                    continue

                metadata = all_results["metadatas"][i] if all_results["metadatas"] else {}
                confidence = metadata.get("confidence", 0.3)

                if confidence >= GOLDEN_RULE_THRESHOLD:
                    content = all_results["documents"][i] if all_results["documents"] else ""
                    golden_rules.append(self._to_golden_rule(doc_id, content, metadata))

            return golden_rules

        except StorageError as e:
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

        Args:
            doc_id: Document ID to check

        Returns:
            True if document is a golden rule
        """
        try:
            result = self._store.get(ids=[doc_id])

            if not result["ids"]:
                return False

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            doc_type = metadata.get("doc_type", "document")
            confidence = metadata.get("confidence", 0.3)

            return doc_type == "golden_rule" or confidence >= GOLDEN_RULE_THRESHOLD

        except StorageError:
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
            result = self._store.get(ids=[doc_id])

            if not result["ids"]:
                return DemotionResult(
                    success=False,
                    doc_id=doc_id,
                    error=f"Document '{doc_id}' not found",
                )

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            doc_type = metadata.get("doc_type", "document")
            confidence = metadata.get("confidence", 0.3)

            # Check if actually a golden rule
            if doc_type != "golden_rule" and confidence < GOLDEN_RULE_THRESHOLD:
                return DemotionResult(
                    success=True,
                    doc_id=doc_id,
                    demoted=False,
                    reason="Document is not a golden rule",
                )

            # Get original type if stored in metadata
            original_type = metadata.get("meta_promoted_from", "document")

            # Update confidence (demotion)
            updated = self._store.update_confidence(doc_id, new_confidence)

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

        except StorageError as e:
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
        metadata: dict[str, Any],
    ) -> GoldenRule:
        """Convert storage data to GoldenRule object.

        Args:
            doc_id: Document ID
            content: Document content
            metadata: Metadata dictionary

        Returns:
            GoldenRule instance
        """
        # Parse promoted_at timestamp
        promoted_at = datetime.now()
        if created_str := metadata.get("created_at"):
            try:
                promoted_at = datetime.fromisoformat(created_str)
            except (ValueError, TypeError):
                pass

        return GoldenRule(
            doc_id=doc_id,
            content=content,
            original_type=metadata.get("meta_promoted_from", metadata.get("doc_type", "document")),
            promoted_at=promoted_at,
            confidence=metadata.get("confidence", GOLDEN_RULE_THRESHOLD),
            validation_count=metadata.get("access_count", 0),
            failure_count=self._failure_counts.get(doc_id, 0),
            metadata=metadata,
        )
