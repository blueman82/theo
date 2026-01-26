"""Feedback collection for validation loop.

This module provides the FeedbackCollector class that aggregates usage
signals from query responses to determine if retrieved documents were
helpful. This implements the BREAK phase of the TRY → BREAK → ANALYZE → LEARN
cycle.

Usage signals can be:
- Explicit: User provides direct feedback
- Implicit: Inferred from actions (e.g., document was used in response)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of feedback signals.

    Categorizes how feedback was gathered:
    - EXPLICIT_POSITIVE: User explicitly marked as helpful
    - EXPLICIT_NEGATIVE: User explicitly marked as unhelpful
    - IMPLICIT_USED: Document content was used in response
    - IMPLICIT_IGNORED: Document was retrieved but not used
    - IMPLICIT_REJECTED: Document was explicitly not used (contradiction, outdated)
    """

    EXPLICIT_POSITIVE = "explicit_positive"
    EXPLICIT_NEGATIVE = "explicit_negative"
    IMPLICIT_USED = "implicit_used"
    IMPLICIT_IGNORED = "implicit_ignored"
    IMPLICIT_REJECTED = "implicit_rejected"


@dataclass
class UsageFeedback:
    """Feedback signal for a single document usage.

    Records whether a retrieved document was helpful in a specific context.

    Attributes:
        doc_id: ID of the document that was retrieved
        was_helpful: Whether the document helped the task
        feedback_type: How the feedback was gathered
        context: Query or context where document was used
        session_id: Optional session identifier
        timestamp: When the feedback was recorded
        confidence_delta: Suggested confidence change (-1.0 to 1.0)
        metadata: Additional feedback metadata
    """

    doc_id: str
    was_helpful: bool
    feedback_type: FeedbackType = FeedbackType.IMPLICIT_USED
    context: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_delta: float = 0.0
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        """Validate feedback and set default confidence_delta."""
        if self.confidence_delta == 0.0:
            # Set default delta based on feedback type
            if self.was_helpful:
                if self.feedback_type == FeedbackType.EXPLICIT_POSITIVE:
                    self.confidence_delta = 0.15  # Explicit feedback weighted more
                else:
                    self.confidence_delta = 0.1  # Implicit positive
            else:
                if self.feedback_type == FeedbackType.EXPLICIT_NEGATIVE:
                    self.confidence_delta = -0.2  # Explicit negative weighted more
                elif self.feedback_type == FeedbackType.IMPLICIT_REJECTED:
                    self.confidence_delta = -0.15  # Active rejection
                else:
                    self.confidence_delta = -0.05  # Just ignored


@dataclass
class AggregatedFeedback:
    """Aggregated feedback for a document over a time period.

    Combines multiple feedback signals into a summary.

    Attributes:
        doc_id: ID of the document
        total_uses: Total number of times document was retrieved
        helpful_count: Number of times marked helpful
        unhelpful_count: Number of times marked unhelpful
        net_confidence_delta: Sum of all confidence deltas
        last_feedback: Timestamp of most recent feedback
        feedback_sources: Breakdown by feedback type
    """

    doc_id: str
    total_uses: int = 0
    helpful_count: int = 0
    unhelpful_count: int = 0
    net_confidence_delta: float = 0.0
    last_feedback: Optional[datetime] = None
    feedback_sources: dict[str, int] = field(default_factory=dict)


class FeedbackCollector:
    """Collects and aggregates usage feedback for documents.

    Provides an in-memory buffer for feedback signals before they are
    applied to the validation loop. This allows batching updates and
    filtering noise from individual signals.

    Usage pattern:
    1. Record feedback as documents are used: collect(feedback)
    2. Periodically flush aggregated feedback to ValidationLoop
    3. Clear processed feedback

    Args:
        batch_size: Number of feedback items before auto-flush (default: 100)
        max_age_seconds: Maximum age of buffered feedback (default: 3600)

    Example:
        >>> collector = FeedbackCollector()
        >>> collector.collect(UsageFeedback(
        ...     doc_id="doc_123",
        ...     was_helpful=True,
        ...     feedback_type=FeedbackType.IMPLICIT_USED,
        ...     context="answering user question about Python",
        ... ))
        >>> # Later, flush to validation loop
        >>> aggregated = collector.get_aggregated()
        >>> for feedback in aggregated.values():
        ...     helpful = feedback.helpful_count > feedback.unhelpful_count
        ...     await loop.record_usage(feedback.doc_id, helpful)
        >>> collector.clear()
    """

    def __init__(
        self,
        batch_size: int = 100,
        max_age_seconds: int = 3600,
    ):
        """Initialize FeedbackCollector.

        Args:
            batch_size: Threshold for auto-flush suggestion
            max_age_seconds: Maximum age before stale warning
        """
        self._feedback_buffer: list[UsageFeedback] = []
        self._batch_size = batch_size
        self._max_age_seconds = max_age_seconds
        self._oldest_timestamp: Optional[datetime] = None

    def collect(self, feedback: UsageFeedback) -> None:
        """Add a feedback signal to the buffer.

        Args:
            feedback: UsageFeedback instance to record
        """
        self._feedback_buffer.append(feedback)

        if self._oldest_timestamp is None:
            self._oldest_timestamp = feedback.timestamp

        logger.debug(
            f"Collected feedback for {feedback.doc_id}: "
            f"helpful={feedback.was_helpful}, type={feedback.feedback_type.value}"
        )

    def collect_many(self, feedbacks: list[UsageFeedback]) -> None:
        """Add multiple feedback signals at once.

        Args:
            feedbacks: List of UsageFeedback instances
        """
        for feedback in feedbacks:
            self.collect(feedback)

    def should_flush(self) -> bool:
        """Check if buffer should be flushed.

        Returns True if:
        - Buffer size exceeds batch_size
        - Oldest feedback exceeds max_age_seconds

        Returns:
            True if flush is recommended
        """
        if len(self._feedback_buffer) >= self._batch_size:
            return True

        if self._oldest_timestamp is not None:
            age = (datetime.now() - self._oldest_timestamp).total_seconds()
            if age >= self._max_age_seconds:
                return True

        return False

    def get_aggregated(self) -> dict[str, AggregatedFeedback]:
        """Aggregate buffered feedback by document ID.

        Returns:
            Dictionary mapping doc_id to AggregatedFeedback
        """
        aggregated: dict[str, AggregatedFeedback] = {}

        for feedback in self._feedback_buffer:
            doc_id = feedback.doc_id

            if doc_id not in aggregated:
                aggregated[doc_id] = AggregatedFeedback(doc_id=doc_id)

            agg = aggregated[doc_id]
            agg.total_uses += 1
            agg.net_confidence_delta += feedback.confidence_delta

            if feedback.was_helpful:
                agg.helpful_count += 1
            else:
                agg.unhelpful_count += 1

            # Track feedback sources
            source = feedback.feedback_type.value
            agg.feedback_sources[source] = agg.feedback_sources.get(source, 0) + 1

            # Update last feedback timestamp
            if agg.last_feedback is None or feedback.timestamp > agg.last_feedback:
                agg.last_feedback = feedback.timestamp

        return aggregated

    def get_pending_count(self) -> int:
        """Get number of pending feedback items.

        Returns:
            Number of items in buffer
        """
        return len(self._feedback_buffer)

    def get_pending_for_doc(self, doc_id: str) -> list[UsageFeedback]:
        """Get pending feedback for a specific document.

        Args:
            doc_id: Document ID to filter by

        Returns:
            List of pending feedback for the document
        """
        return [f for f in self._feedback_buffer if f.doc_id == doc_id]

    def clear(self) -> int:
        """Clear the feedback buffer.

        Returns:
            Number of items cleared
        """
        count = len(self._feedback_buffer)
        self._feedback_buffer.clear()
        self._oldest_timestamp = None
        logger.debug(f"Cleared {count} feedback items from buffer")
        return count

    def get_stats(self) -> dict:
        """Get buffer statistics.

        Returns:
            Dictionary with buffer stats
        """
        helpful_count = sum(1 for f in self._feedback_buffer if f.was_helpful)
        unique_docs = len(set(f.doc_id for f in self._feedback_buffer))

        return {
            "total_pending": len(self._feedback_buffer),
            "unique_documents": unique_docs,
            "helpful_count": helpful_count,
            "unhelpful_count": len(self._feedback_buffer) - helpful_count,
            "oldest_age_seconds": (
                (datetime.now() - self._oldest_timestamp).total_seconds()
                if self._oldest_timestamp
                else 0
            ),
            "should_flush": self.should_flush(),
        }


def create_implicit_feedback(
    doc_id: str,
    was_used_in_response: bool,
    query_text: Optional[str] = None,
    session_id: Optional[str] = None,
) -> UsageFeedback:
    """Create implicit feedback from response generation.

    Helper function to create feedback based on whether a retrieved
    document was actually used in generating a response.

    Args:
        doc_id: ID of the retrieved document
        was_used_in_response: Whether content was used
        query_text: The original query
        session_id: Optional session identifier

    Returns:
        UsageFeedback with appropriate implicit type
    """
    if was_used_in_response:
        return UsageFeedback(
            doc_id=doc_id,
            was_helpful=True,
            feedback_type=FeedbackType.IMPLICIT_USED,
            context=query_text,
            session_id=session_id,
        )
    else:
        return UsageFeedback(
            doc_id=doc_id,
            was_helpful=False,
            feedback_type=FeedbackType.IMPLICIT_IGNORED,
            context=query_text,
            session_id=session_id,
        )
