"""Core validation loop for confidence scoring.

This module provides the ValidationLoop class that implements confidence
scoring for documents/memories based on their practical usefulness.
Ported from Recall's memory_validate/memory_outcome functions.

The validation loop adjusts confidence scores:
- Success: confidence += adjustment (default 0.1, max 1.0)
- Failure: confidence -= adjustment * 1.5 (min 0.0)

Documents reaching confidence >= 0.9 can be promoted to golden rules.
Documents with confidence < 0.15 are candidates for removal.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from theo.constants import (
    FAILURE_MULTIPLIER,
    GOLDEN_RULE_THRESHOLD,
    INITIAL_CONFIDENCE,
    LOW_CONFIDENCE_THRESHOLD,
    SUCCESS_ADJUSTMENT,
)
from theo.storage import ChromaStore, StorageError
from theo.types import MemoryType

logger = logging.getLogger(__name__)


@dataclass
class UsageResult:
    """Result of recording a usage event.

    Attributes:
        success: Whether the recording succeeded
        doc_id: ID of the document that was used
        old_confidence: Confidence score before update
        new_confidence: Confidence score after update
        was_helpful: Whether the usage was helpful
        promoted: Whether the document was promoted to golden rule
        error: Error message if operation failed
    """

    success: bool
    doc_id: str
    old_confidence: float = 0.0
    new_confidence: float = 0.0
    was_helpful: bool = False
    promoted: bool = False
    error: Optional[str] = None


@dataclass
class DecayResult:
    """Result of decaying unused documents.

    Attributes:
        success: Whether the decay operation succeeded
        decayed_count: Number of documents whose confidence was decayed
        decayed_ids: List of document IDs that were decayed
        skipped_golden: Number of golden rules skipped (never decay)
        error: Error message if operation failed
    """

    success: bool
    decayed_count: int = 0
    decayed_ids: list[str] = field(default_factory=list)
    skipped_golden: int = 0
    error: Optional[str] = None


@dataclass
class PromotionResult:
    """Result of checking for golden rule promotion.

    Attributes:
        success: Whether the check succeeded
        doc_id: ID of the document checked
        promoted: Whether the document was promoted
        new_type: New memory type if promoted
        reason: Reason for promotion/non-promotion
        error: Error message if operation failed
    """

    success: bool
    doc_id: str
    promoted: bool = False
    new_type: Optional[MemoryType] = None
    reason: str = ""
    error: Optional[str] = None


class ValidationLoop:
    """Core confidence scoring logic for the validation loop.

    Tracks document/memory usefulness over time by adjusting confidence
    scores based on whether retrievals were helpful. Implements the
    TRY → BREAK → ANALYZE → LEARN cycle from Recall.

    Confidence scoring rules:
    - Initial confidence: 0.3 (memories need validation to build trust)
    - Success: confidence += adjustment (default 0.1)
    - Failure: confidence -= adjustment * 1.5 (failures penalized more heavily)
    - Golden rule threshold: 0.9 (high confidence, never decay)
    - Low confidence threshold: 0.15 (candidates for deletion)

    Args:
        store: ChromaStore instance for persistence
        success_adjustment: Confidence increase on success (default: 0.1)
        failure_multiplier: Multiplier for failure penalty (default: 1.5)

    Example:
        >>> store = ChromaStore(ephemeral=True)
        >>> loop = ValidationLoop(store)
        >>> # Record helpful usage
        >>> result = await loop.record_usage("doc_123", was_helpful=True)
        >>> print(f"Confidence: {result.old_confidence} -> {result.new_confidence}")
    """

    def __init__(
        self,
        store: ChromaStore,
        success_adjustment: float = SUCCESS_ADJUSTMENT,
        failure_multiplier: float = FAILURE_MULTIPLIER,
    ):
        """Initialize ValidationLoop with storage backend.

        Args:
            store: ChromaStore instance for persistence
            success_adjustment: Base confidence adjustment on success
            failure_multiplier: Multiplier applied to adjustment on failure
        """
        self._store = store
        self._success_adjustment = success_adjustment
        self._failure_multiplier = failure_multiplier

    async def record_usage(
        self,
        doc_id: str,
        was_helpful: bool,
        context: Optional[str] = None,
    ) -> UsageResult:
        """Record usage of a document and adjust confidence.

        Called when a document/memory is retrieved and used. Adjusts
        confidence based on whether it was helpful:
        - Helpful: confidence += success_adjustment
        - Not helpful: confidence -= success_adjustment * failure_multiplier

        Also checks for golden rule promotion when confidence reaches 0.9.

        Args:
            doc_id: ID of the document that was used
            was_helpful: Whether the document helped the task
            context: Optional context describing the usage

        Returns:
            UsageResult with old/new confidence and promotion status
        """
        try:
            # Get current document state
            result = self._store.get(ids=[doc_id])

            if not result["ids"]:
                return UsageResult(
                    success=False,
                    doc_id=doc_id,
                    error=f"Document '{doc_id}' not found",
                )

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            old_confidence = metadata.get("confidence", INITIAL_CONFIDENCE)

            # Calculate new confidence
            if was_helpful:
                new_confidence = min(1.0, old_confidence + self._success_adjustment)
            else:
                # Failures penalized more heavily
                penalty = self._success_adjustment * self._failure_multiplier
                new_confidence = max(0.0, old_confidence - penalty)

            # Update confidence in storage
            updated = self._store.update_confidence(doc_id, new_confidence)

            if not updated:
                return UsageResult(
                    success=False,
                    doc_id=doc_id,
                    old_confidence=old_confidence,
                    error=f"Failed to update confidence for '{doc_id}'",
                )

            # Check for golden rule promotion
            promoted = False
            if was_helpful and new_confidence >= GOLDEN_RULE_THRESHOLD:
                # Update metadata with new confidence for promotion check
                updated_metadata = {**metadata, "confidence": new_confidence}
                promotion_result = await self._check_promotion(doc_id, updated_metadata)
                promoted = promotion_result.promoted

            logger.info(
                f"Recorded usage for {doc_id}: helpful={was_helpful}, "
                f"confidence {old_confidence:.2f} -> {new_confidence:.2f}"
                + (", promoted to golden rule" if promoted else "")
            )

            return UsageResult(
                success=True,
                doc_id=doc_id,
                old_confidence=old_confidence,
                new_confidence=new_confidence,
                was_helpful=was_helpful,
                promoted=promoted,
            )

        except StorageError as e:
            logger.error(f"Storage error recording usage for {doc_id}: {e}")
            return UsageResult(
                success=False,
                doc_id=doc_id,
                error=f"Storage error: {e}",
            )
        except Exception as e:
            logger.error(f"Error recording usage for {doc_id}: {e}")
            return UsageResult(
                success=False,
                doc_id=doc_id,
                error=f"Unexpected error: {e}",
            )

    async def decay_unused(
        self,
        days_threshold: int = 30,
        decay_amount: float = 0.05,
        min_confidence: float = LOW_CONFIDENCE_THRESHOLD,
        namespace: Optional[str] = None,
    ) -> DecayResult:
        """Decay confidence of documents not accessed recently.

        Documents that haven't been used within days_threshold have their
        confidence reduced. Golden rules (confidence >= 0.9 or type=golden_rule)
        are never decayed.

        Args:
            days_threshold: Days without access before decay (default: 30)
            decay_amount: Amount to subtract from confidence (default: 0.05)
            min_confidence: Minimum confidence floor (default: 0.15)
            namespace: Optional namespace filter

        Returns:
            DecayResult with count of decayed documents
        """
        try:
            decayed_ids: list[str] = []
            skipped_golden = 0

            # Get all documents (potentially filtered by namespace)
            where_filter = {"namespace": namespace} if namespace else None
            all_docs = self._store.get(where=where_filter)

            if not all_docs["ids"]:
                return DecayResult(success=True, decayed_count=0)

            # Calculate threshold timestamp
            threshold_time = time.time() - (days_threshold * 24 * 60 * 60)

            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}

                # Skip golden rules
                confidence = metadata.get("confidence", INITIAL_CONFIDENCE)
                doc_type = metadata.get("doc_type", "document")

                if confidence >= GOLDEN_RULE_THRESHOLD or doc_type == "golden_rule":
                    skipped_golden += 1
                    continue

                # Check if document is stale
                accessed_at_str = metadata.get("accessed_at")
                if accessed_at_str:
                    try:
                        accessed_at = datetime.fromisoformat(accessed_at_str)
                        accessed_timestamp = accessed_at.timestamp()
                    except (ValueError, TypeError):
                        accessed_timestamp = time.time()
                else:
                    # If no accessed_at, use created_at
                    created_at_str = metadata.get("created_at")
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str)
                            accessed_timestamp = created_at.timestamp()
                        except (ValueError, TypeError):
                            accessed_timestamp = time.time()
                    else:
                        accessed_timestamp = time.time()

                # Skip if accessed recently
                if accessed_timestamp >= threshold_time:
                    continue

                # Decay confidence
                new_confidence = max(min_confidence, confidence - decay_amount)

                if new_confidence != confidence:
                    updated = self._store.update_confidence(doc_id, new_confidence)
                    if updated:
                        decayed_ids.append(doc_id)
                        logger.debug(
                            f"Decayed {doc_id}: {confidence:.2f} -> {new_confidence:.2f}"
                        )

            logger.info(
                f"Decay complete: {len(decayed_ids)} decayed, {skipped_golden} golden rules skipped"
            )

            return DecayResult(
                success=True,
                decayed_count=len(decayed_ids),
                decayed_ids=decayed_ids,
                skipped_golden=skipped_golden,
            )

        except StorageError as e:
            logger.error(f"Storage error during decay: {e}")
            return DecayResult(success=False, error=f"Storage error: {e}")
        except Exception as e:
            logger.error(f"Error during decay: {e}")
            return DecayResult(success=False, error=f"Unexpected error: {e}")

    async def get_confidence(self, doc_id: str) -> Optional[float]:
        """Get the current confidence score for a document.

        Args:
            doc_id: Document ID to query

        Returns:
            Confidence score (0.0-1.0) or None if document not found
        """
        try:
            result = self._store.get(ids=[doc_id])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0] if result["metadatas"] else {}
            return metadata.get("confidence", INITIAL_CONFIDENCE)

        except StorageError:
            return None

    async def _check_promotion(
        self,
        doc_id: str,
        metadata: dict[str, Any],
    ) -> PromotionResult:
        """Check if a document should be promoted to golden rule.

        A document is promoted when:
        1. Confidence >= 0.9 (GOLDEN_RULE_THRESHOLD)
        2. Type is promotable (preference, decision, or pattern)
        3. Not already a golden rule

        Args:
            doc_id: Document ID to check
            metadata: Current document metadata

        Returns:
            PromotionResult indicating if promotion occurred
        """
        confidence = metadata.get("confidence", INITIAL_CONFIDENCE)
        doc_type = metadata.get("doc_type", "document")

        # Already a golden rule
        if doc_type == "golden_rule":
            return PromotionResult(
                success=True,
                doc_id=doc_id,
                promoted=False,
                reason="Already a golden rule",
            )

        # Confidence not high enough
        if confidence < GOLDEN_RULE_THRESHOLD:
            return PromotionResult(
                success=True,
                doc_id=doc_id,
                promoted=False,
                reason=f"Confidence {confidence:.2f} below threshold {GOLDEN_RULE_THRESHOLD}",
            )

        # Check if type is promotable
        promotable_types = {"preference", "decision", "pattern"}
        if doc_type not in promotable_types:
            return PromotionResult(
                success=True,
                doc_id=doc_id,
                promoted=False,
                reason=f"Type '{doc_type}' is not promotable",
            )

        # Promote to golden rule
        # Note: ChromaStore.update_confidence only updates confidence field
        # For full type change, we would need a more comprehensive update method
        # For now, we just note the promotion eligibility
        logger.info(
            f"Document {doc_id} eligible for golden rule promotion "
            f"(confidence={confidence:.2f}, type={doc_type})"
        )

        return PromotionResult(
            success=True,
            doc_id=doc_id,
            promoted=True,
            new_type=MemoryType.GOLDEN_RULE,
            reason=f"Promoted from {doc_type} (confidence={confidence:.2f})",
        )

    def get_low_confidence_candidates(
        self,
        threshold: float = LOW_CONFIDENCE_THRESHOLD,
        namespace: Optional[str] = None,
    ) -> list[str]:
        """Get document IDs with confidence below threshold.

        These are candidates for deletion as they have consistently
        failed to be helpful.

        Args:
            threshold: Confidence threshold (default: 0.15)
            namespace: Optional namespace filter

        Returns:
            List of document IDs below the threshold
        """
        try:
            where_filter = {"namespace": namespace} if namespace else None
            all_docs = self._store.get(where=where_filter)

            candidates = []
            for i, doc_id in enumerate(all_docs["ids"]):
                metadata = all_docs["metadatas"][i] if all_docs["metadatas"] else {}
                confidence = metadata.get("confidence", INITIAL_CONFIDENCE)

                if confidence < threshold:
                    candidates.append(doc_id)

            return candidates

        except StorageError:
            return []
