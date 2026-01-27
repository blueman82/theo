"""Tests for theo.validation module.

Tests the validation loop system including:
- ValidationLoop confidence scoring
- FeedbackCollector aggregation
- GoldenRules management
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from theo.validation import (
    FeedbackCollector,
    GoldenRules,
    UsageFeedback,
    ValidationLoop,
)
from theo.validation.feedback import (
    FeedbackType,
    create_implicit_feedback,
)
from theo.validation.golden_rules import (
    DEMOTION_FAILURE_THRESHOLD,
)
from theo.constants import (
    FAILURE_MULTIPLIER,
    SUCCESS_ADJUSTMENT,
    LOW_CONFIDENCE_THRESHOLD,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_store():
    """Create a mock ChromaStore for testing."""
    store = MagicMock()
    # Default get behavior returns empty
    store.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    store.update_confidence.return_value = True
    return store


@pytest.fixture
def sample_doc_metadata():
    """Sample document metadata for testing."""
    return {
        "confidence": 0.5,
        "doc_type": "pattern",
        "namespace": "test",
        "created_at": datetime.now().isoformat(),
        "accessed_at": datetime.now().isoformat(),
    }


@pytest.fixture
def validation_loop(mock_store):
    """Create ValidationLoop with mock store."""
    return ValidationLoop(mock_store)


@pytest.fixture
def feedback_collector():
    """Create FeedbackCollector for testing."""
    return FeedbackCollector(batch_size=10, max_age_seconds=60)


@pytest.fixture
def golden_rules(mock_store):
    """Create GoldenRules manager with mock store."""
    return GoldenRules(mock_store)


# ============================================================================
# ValidationLoop Tests
# ============================================================================


class TestValidationLoop:
    """Tests for ValidationLoop class."""

    @pytest.mark.asyncio
    async def test_record_usage_helpful_increases_confidence(
        self, validation_loop, mock_store, sample_doc_metadata
    ):
        """Helpful usage should increase confidence."""
        # Setup mock to return document with confidence 0.5
        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test content"],
            "metadatas": [sample_doc_metadata],
        }

        result = await validation_loop.record_usage("doc_123", was_helpful=True)

        assert result.success is True
        assert result.doc_id == "doc_123"
        assert result.old_confidence == 0.5
        assert result.new_confidence == 0.5 + SUCCESS_ADJUSTMENT
        assert result.was_helpful is True
        mock_store.update_confidence.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_usage_not_helpful_decreases_confidence(
        self, validation_loop, mock_store, sample_doc_metadata
    ):
        """Unhelpful usage should decrease confidence with penalty multiplier."""
        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test content"],
            "metadatas": [sample_doc_metadata],
        }

        result = await validation_loop.record_usage("doc_123", was_helpful=False)

        expected_new_confidence = (
            0.5 - SUCCESS_ADJUSTMENT * DEFAULT_FAILURE_MULTIPLIER
        )
        assert result.success is True
        assert result.new_confidence == expected_new_confidence
        assert result.was_helpful is False

    @pytest.mark.asyncio
    async def test_record_usage_document_not_found(self, validation_loop, mock_store):
        """Should return error for non-existent document."""
        mock_store.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        result = await validation_loop.record_usage("nonexistent", was_helpful=True)

        assert result.success is False
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_record_usage_confidence_capped_at_max(
        self, validation_loop, mock_store
    ):
        """Confidence should not exceed 1.0."""
        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test"],
            "metadatas": [{"confidence": 0.95}],
        }

        result = await validation_loop.record_usage("doc_123", was_helpful=True)

        assert result.new_confidence == 1.0

    @pytest.mark.asyncio
    async def test_record_usage_confidence_capped_at_min(
        self, validation_loop, mock_store
    ):
        """Confidence should not go below 0.0."""
        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test"],
            "metadatas": [{"confidence": 0.1}],
        }

        result = await validation_loop.record_usage("doc_123", was_helpful=False)

        assert result.new_confidence == 0.0

    @pytest.mark.asyncio
    async def test_record_usage_promotes_to_golden_rule(
        self, validation_loop, mock_store
    ):
        """Should detect promotion eligibility when confidence reaches threshold."""
        # Start at 0.85, will go to 0.95 with success
        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test"],
            "metadatas": [{"confidence": 0.85, "doc_type": "pattern"}],
        }

        result = await validation_loop.record_usage("doc_123", was_helpful=True)

        assert result.new_confidence == 0.95
        assert result.promoted is True

    @pytest.mark.asyncio
    async def test_decay_unused_skips_golden_rules(self, validation_loop, mock_store):
        """Decay should skip golden rules."""
        old_time = (datetime.now() - timedelta(days=60)).isoformat()
        mock_store.get.return_value = {
            "ids": ["golden_1", "normal_1"],
            "documents": ["golden content", "normal content"],
            "metadatas": [
                {"confidence": 0.95, "doc_type": "golden_rule", "accessed_at": old_time},
                {"confidence": 0.5, "doc_type": "pattern", "accessed_at": old_time},
            ],
        }

        result = await validation_loop.decay_unused(days_threshold=30)

        assert result.success is True
        assert result.skipped_golden == 1
        assert len(result.decayed_ids) == 1
        assert "normal_1" in result.decayed_ids

    @pytest.mark.asyncio
    async def test_decay_unused_respects_threshold(self, validation_loop, mock_store):
        """Decay should skip recently accessed documents."""
        recent_time = datetime.now().isoformat()
        old_time = (datetime.now() - timedelta(days=60)).isoformat()

        mock_store.get.return_value = {
            "ids": ["recent_1", "old_1"],
            "documents": ["recent content", "old content"],
            "metadatas": [
                {"confidence": 0.5, "doc_type": "pattern", "accessed_at": recent_time},
                {"confidence": 0.5, "doc_type": "pattern", "accessed_at": old_time},
            ],
        }

        result = await validation_loop.decay_unused(days_threshold=30)

        assert result.success is True
        assert "old_1" in result.decayed_ids
        assert "recent_1" not in result.decayed_ids

    @pytest.mark.asyncio
    async def test_get_confidence(self, validation_loop, mock_store):
        """Should return confidence for existing document."""
        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test"],
            "metadatas": [{"confidence": 0.75}],
        }

        confidence = await validation_loop.get_confidence("doc_123")

        assert confidence == 0.75

    @pytest.mark.asyncio
    async def test_get_confidence_not_found(self, validation_loop, mock_store):
        """Should return None for non-existent document."""
        mock_store.get.return_value = {"ids": [], "documents": [], "metadatas": []}

        confidence = await validation_loop.get_confidence("nonexistent")

        assert confidence is None

    def test_get_low_confidence_candidates(self, validation_loop, mock_store):
        """Should return documents below confidence threshold."""
        mock_store.get.return_value = {
            "ids": ["low_1", "high_1", "low_2"],
            "documents": ["", "", ""],
            "metadatas": [
                {"confidence": 0.1},
                {"confidence": 0.8},
                {"confidence": 0.05},
            ],
        }

        candidates = validation_loop.get_low_confidence_candidates(
            threshold=LOW_CONFIDENCE_THRESHOLD
        )

        assert "low_1" in candidates
        assert "low_2" in candidates
        assert "high_1" not in candidates


# ============================================================================
# FeedbackCollector Tests
# ============================================================================


class TestFeedbackCollector:
    """Tests for FeedbackCollector class."""

    def test_collect_single_feedback(self, feedback_collector):
        """Should add feedback to buffer."""
        feedback = UsageFeedback(
            doc_id="doc_123", was_helpful=True, feedback_type=FeedbackType.IMPLICIT_USED
        )

        feedback_collector.collect(feedback)

        assert feedback_collector.get_pending_count() == 1

    def test_collect_many_feedbacks(self, feedback_collector):
        """Should add multiple feedbacks at once."""
        feedbacks = [
            UsageFeedback(doc_id=f"doc_{i}", was_helpful=True)
            for i in range(5)
        ]

        feedback_collector.collect_many(feedbacks)

        assert feedback_collector.get_pending_count() == 5

    def test_get_aggregated_combines_by_doc_id(self, feedback_collector):
        """Should aggregate feedback by document ID."""
        # Add multiple feedbacks for same document
        feedbacks = [
            UsageFeedback(doc_id="doc_123", was_helpful=True),
            UsageFeedback(doc_id="doc_123", was_helpful=True),
            UsageFeedback(doc_id="doc_123", was_helpful=False),
            UsageFeedback(doc_id="doc_456", was_helpful=True),
        ]
        feedback_collector.collect_many(feedbacks)

        aggregated = feedback_collector.get_aggregated()

        assert "doc_123" in aggregated
        assert aggregated["doc_123"].total_uses == 3
        assert aggregated["doc_123"].helpful_count == 2
        assert aggregated["doc_123"].unhelpful_count == 1
        assert "doc_456" in aggregated

    def test_should_flush_by_batch_size(self, feedback_collector):
        """Should recommend flush when batch size exceeded."""
        # Collector has batch_size=10
        for i in range(15):
            feedback_collector.collect(
                UsageFeedback(doc_id=f"doc_{i}", was_helpful=True)
            )

        assert feedback_collector.should_flush() is True

    def test_should_flush_by_age(self):
        """Should recommend flush when max age exceeded."""
        collector = FeedbackCollector(batch_size=100, max_age_seconds=1)
        collector.collect(UsageFeedback(doc_id="doc_1", was_helpful=True))

        # Manually set old timestamp
        collector._oldest_timestamp = datetime.now() - timedelta(seconds=5)

        assert collector.should_flush() is True

    def test_clear_buffer(self, feedback_collector):
        """Should clear all buffered feedback."""
        feedbacks = [
            UsageFeedback(doc_id=f"doc_{i}", was_helpful=True) for i in range(5)
        ]
        feedback_collector.collect_many(feedbacks)

        count = feedback_collector.clear()

        assert count == 5
        assert feedback_collector.get_pending_count() == 0

    def test_get_stats(self, feedback_collector):
        """Should return buffer statistics."""
        feedbacks = [
            UsageFeedback(doc_id="doc_1", was_helpful=True),
            UsageFeedback(doc_id="doc_2", was_helpful=False),
            UsageFeedback(doc_id="doc_1", was_helpful=True),
        ]
        feedback_collector.collect_many(feedbacks)

        stats = feedback_collector.get_stats()

        assert stats["total_pending"] == 3
        assert stats["unique_documents"] == 2
        assert stats["helpful_count"] == 2
        assert stats["unhelpful_count"] == 1

    def test_get_pending_for_doc(self, feedback_collector):
        """Should filter feedback by document ID."""
        feedbacks = [
            UsageFeedback(doc_id="doc_1", was_helpful=True),
            UsageFeedback(doc_id="doc_2", was_helpful=False),
            UsageFeedback(doc_id="doc_1", was_helpful=False),
        ]
        feedback_collector.collect_many(feedbacks)

        pending = feedback_collector.get_pending_for_doc("doc_1")

        assert len(pending) == 2
        assert all(f.doc_id == "doc_1" for f in pending)


class TestUsageFeedback:
    """Tests for UsageFeedback dataclass."""

    def test_default_confidence_delta_helpful(self):
        """Should set positive delta for helpful implicit feedback."""
        feedback = UsageFeedback(
            doc_id="doc_1",
            was_helpful=True,
            feedback_type=FeedbackType.IMPLICIT_USED,
        )

        assert feedback.confidence_delta > 0

    def test_default_confidence_delta_unhelpful(self):
        """Should set negative delta for unhelpful feedback."""
        feedback = UsageFeedback(
            doc_id="doc_1",
            was_helpful=False,
            feedback_type=FeedbackType.IMPLICIT_IGNORED,
        )

        assert feedback.confidence_delta < 0

    def test_explicit_feedback_weighted_more(self):
        """Explicit feedback should have larger delta than implicit."""
        explicit = UsageFeedback(
            doc_id="doc_1",
            was_helpful=True,
            feedback_type=FeedbackType.EXPLICIT_POSITIVE,
        )
        implicit = UsageFeedback(
            doc_id="doc_1",
            was_helpful=True,
            feedback_type=FeedbackType.IMPLICIT_USED,
        )

        assert explicit.confidence_delta > implicit.confidence_delta


class TestCreateImplicitFeedback:
    """Tests for create_implicit_feedback helper."""

    def test_used_creates_positive_feedback(self):
        """Should create positive feedback when content was used."""
        feedback = create_implicit_feedback(
            doc_id="doc_1",
            was_used_in_response=True,
            query_text="test query",
        )

        assert feedback.was_helpful is True
        assert feedback.feedback_type == FeedbackType.IMPLICIT_USED
        assert feedback.context == "test query"

    def test_not_used_creates_negative_feedback(self):
        """Should create negative feedback when content was ignored."""
        feedback = create_implicit_feedback(
            doc_id="doc_1",
            was_used_in_response=False,
        )

        assert feedback.was_helpful is False
        assert feedback.feedback_type == FeedbackType.IMPLICIT_IGNORED


# ============================================================================
# GoldenRules Tests
# ============================================================================


class TestGoldenRules:
    """Tests for GoldenRules class."""

    def test_is_golden_rule_by_type(self, golden_rules, mock_store):
        """Should identify golden rule by type."""
        mock_store.get.return_value = {
            "ids": ["doc_1"],
            "documents": ["test"],
            "metadatas": [{"doc_type": "golden_rule", "confidence": 0.5}],
        }

        assert golden_rules.is_golden_rule("doc_1") is True

    def test_is_golden_rule_by_confidence(self, golden_rules, mock_store):
        """Should identify golden rule by high confidence."""
        mock_store.get.return_value = {
            "ids": ["doc_1"],
            "documents": ["test"],
            "metadatas": [{"doc_type": "pattern", "confidence": 0.95}],
        }

        assert golden_rules.is_golden_rule("doc_1") is True

    def test_is_not_golden_rule(self, golden_rules, mock_store):
        """Should return False for non-golden rule."""
        mock_store.get.return_value = {
            "ids": ["doc_1"],
            "documents": ["test"],
            "metadatas": [{"doc_type": "pattern", "confidence": 0.5}],
        }

        assert golden_rules.is_golden_rule("doc_1") is False

    def test_record_failure_increments_count(self, golden_rules):
        """Should track failures per document."""
        assert golden_rules.record_failure("doc_1") == 1
        assert golden_rules.record_failure("doc_1") == 2
        assert golden_rules.record_failure("doc_1") == 3

    def test_reset_failures_clears_count(self, golden_rules):
        """Should reset failure count."""
        golden_rules.record_failure("doc_1")
        golden_rules.record_failure("doc_1")

        golden_rules.reset_failures("doc_1")

        assert golden_rules._failure_counts.get("doc_1", 0) == 0

    def test_should_demote_after_threshold(self, golden_rules):
        """Should recommend demotion after failure threshold."""
        for _ in range(DEMOTION_FAILURE_THRESHOLD):
            golden_rules.record_failure("doc_1")

        assert golden_rules.should_demote("doc_1") is True

    def test_should_not_demote_below_threshold(self, golden_rules):
        """Should not recommend demotion below threshold."""
        for _ in range(DEMOTION_FAILURE_THRESHOLD - 1):
            golden_rules.record_failure("doc_1")

        assert golden_rules.should_demote("doc_1") is False

    @pytest.mark.asyncio
    async def test_demote_reduces_confidence(self, golden_rules, mock_store):
        """Demotion should reduce confidence."""
        mock_store.get.return_value = {
            "ids": ["doc_1"],
            "documents": ["test"],
            "metadatas": [{"doc_type": "golden_rule", "confidence": 0.95}],
        }

        result = await golden_rules.demote("doc_1", reason="Failed multiple times")

        assert result.success is True
        assert result.demoted is True
        assert result.new_confidence == 0.7  # Default demotion confidence
        mock_store.update_confidence.assert_called_once_with("doc_1", 0.7)

    @pytest.mark.asyncio
    async def test_demote_non_golden_rule(self, golden_rules, mock_store):
        """Should not demote non-golden rule."""
        mock_store.get.return_value = {
            "ids": ["doc_1"],
            "documents": ["test"],
            "metadatas": [{"doc_type": "pattern", "confidence": 0.5}],
        }

        result = await golden_rules.demote("doc_1", reason="test")

        assert result.demoted is False
        assert "not a golden rule" in result.reason.lower()

    def test_get_all_returns_golden_rules(self, golden_rules, mock_store):
        """Should return all golden rules."""
        # First call for type filter
        mock_store.get.side_effect = [
            {
                "ids": ["golden_1"],
                "documents": ["golden content"],
                "metadatas": [{"doc_type": "golden_rule", "confidence": 0.9}],
            },
            {
                "ids": ["golden_1", "high_conf_1"],
                "documents": ["golden content", "high conf content"],
                "metadatas": [
                    {"doc_type": "golden_rule", "confidence": 0.9},
                    {"doc_type": "pattern", "confidence": 0.95},
                ],
            },
        ]

        rules = golden_rules.get_all()

        assert len(rules) == 2
        assert any(r.doc_id == "golden_1" for r in rules)
        assert any(r.doc_id == "high_conf_1" for r in rules)

    def test_get_stats(self, golden_rules, mock_store):
        """Should return statistics about golden rules."""
        mock_store.get.side_effect = [
            {
                "ids": ["g1"],
                "documents": ["test"],
                "metadatas": [{"doc_type": "golden_rule", "confidence": 0.95}],
            },
            {
                "ids": ["g1", "g2"],
                "documents": ["test", "test2"],
                "metadatas": [
                    {"doc_type": "golden_rule", "confidence": 0.95},
                    {"doc_type": "pattern", "confidence": 0.92},
                ],
            },
        ]

        # Record a failure for g1
        golden_rules.record_failure("g1")

        stats = golden_rules.get_stats()

        assert stats["total_golden_rules"] == 2
        assert stats["at_risk_count"] == 1  # g1 has failures


# ============================================================================
# Integration Tests
# ============================================================================


class TestValidationIntegration:
    """Integration tests for validation components."""

    @pytest.mark.asyncio
    async def test_feedback_to_validation_flow(self, mock_store):
        """Test flow from feedback collection to validation update."""
        # Setup
        collector = FeedbackCollector()
        loop = ValidationLoop(mock_store)

        mock_store.get.return_value = {
            "ids": ["doc_123"],
            "documents": ["test"],
            "metadatas": [{"confidence": 0.5, "doc_type": "pattern"}],
        }

        # Collect feedback
        for _ in range(3):
            collector.collect(
                UsageFeedback(doc_id="doc_123", was_helpful=True)
            )
        collector.collect(UsageFeedback(doc_id="doc_123", was_helpful=False))

        # Aggregate
        aggregated = collector.get_aggregated()
        assert "doc_123" in aggregated
        assert aggregated["doc_123"].helpful_count == 3
        assert aggregated["doc_123"].unhelpful_count == 1

        # Apply to validation loop (helpful majority)
        result = await loop.record_usage(
            "doc_123",
            was_helpful=aggregated["doc_123"].helpful_count
            > aggregated["doc_123"].unhelpful_count,
        )

        assert result.success is True
        assert result.new_confidence > 0.5

    @pytest.mark.asyncio
    async def test_golden_rule_failure_tracking_flow(self, mock_store):
        """Test flow for tracking golden rule failures and demotion."""
        rules = GoldenRules(mock_store)
        loop = ValidationLoop(mock_store)

        # Setup golden rule document
        mock_store.get.return_value = {
            "ids": ["golden_1"],
            "documents": ["golden content"],
            "metadatas": [{"doc_type": "golden_rule", "confidence": 0.92}],
        }

        # Simulate failures
        for _ in range(DEMOTION_FAILURE_THRESHOLD):
            # Record failure in rules tracker
            rules.record_failure("golden_1")

            # Also record negative usage
            await loop.record_usage("golden_1", was_helpful=False)

        # Check demotion recommendation
        assert rules.should_demote("golden_1") is True

        # Demote
        result = await rules.demote("golden_1", reason="Multiple failures")
        assert result.demoted is True
