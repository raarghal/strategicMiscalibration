"""
Strategic Uncertainty Quantification package.

This package provides tools for studying strategic behavior in LLM confidence reporting.
"""

from src.strategicuncertainty.llm_interface import (
    ConfidenceMode,
    ModelResponse,
    UserResponse,
    load_template,
    query_llm,
)
from src.strategicuncertainty.single_player import SinglePlayerConfig
from src.strategicuncertainty.two_player import TwoPlayerConfig
from src.strategicuncertainty.utils import (
    BaseGameConfig,
    HistoryEntry,
    QuestionData,
    RoundResult,
    TrialStatistics,
    evaluate_answer,
    extract_question_from_dataset,
)

__all__ = [
    # Config classes
    "BaseGameConfig",
    "SinglePlayerConfig",
    "TwoPlayerConfig",
    # Enums
    "ConfidenceMode",
    # Type definitions
    "HistoryEntry",
    "QuestionData",
    "RoundResult",
    "TrialStatistics",
    # Response schemas
    "ModelResponse",
    "UserResponse",
    # Utility functions
    "evaluate_answer",
    "extract_question_from_dataset",
    "load_template",
    "query_llm",
]
