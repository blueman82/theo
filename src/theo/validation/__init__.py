"""Validation loop for confidence scoring in Theo.

This module provides the validation loop system that tracks document/memory
usefulness over time, adjusting confidence scores based on practical use.
Ported from Recall's validation system.

The validation loop follows the TRY → BREAK → ANALYZE → LEARN cycle:
1. TRY: Apply a memory/document in a context
2. BREAK: Observe if it helped or failed
3. ANALYZE: Update confidence scores based on outcome
4. LEARN: Promote to golden rules or decay unused items

Key components:
- ValidationLoop: Core confidence scoring logic with record_usage() and decay_unused()
- FeedbackCollector: Aggregates usage signals (was_helpful) from query responses
- GoldenRules: Manages patterns that maintain high confidence (never decay)

Example:
    >>> from theo.validation import ValidationLoop, FeedbackCollector
    >>> from theo.storage import ChromaStore
    >>> store = ChromaStore(ephemeral=True)
    >>> loop = ValidationLoop(store)
    >>> # Record positive usage
    >>> result = await loop.record_usage("doc_123", was_helpful=True)
    >>> # Decay unused documents periodically
    >>> decayed = await loop.decay_unused(days_threshold=30)
"""

from theo.validation.feedback import FeedbackCollector, UsageFeedback
from theo.validation.golden_rules import GoldenRules
from theo.validation.loop import (
    DecayResult,
    PromotionResult,
    UsageResult,
    ValidationLoop,
)

__all__ = [
    # Core validation
    "ValidationLoop",
    "UsageResult",
    "DecayResult",
    "PromotionResult",
    # Feedback collection
    "FeedbackCollector",
    "UsageFeedback",
    # Golden rules
    "GoldenRules",
]
