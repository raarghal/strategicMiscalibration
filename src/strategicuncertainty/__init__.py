"""
Strategic Uncertainty Quantification package.

This package provides tools for studying strategic behavior in LLM confidence reporting.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BaseGameConfig",
    "SinglePlayerConfig",
    "TwoPlayerConfig",
    "ConfidenceMode",
    "HistoryEntry",
    "QuestionData",
    "RoundResult",
    "TrialStatistics",
    "ModelResponse",
    "UserResponse",
    "evaluate_answer",
    "extract_question_from_dataset",
    "load_template",
    "query_llm",
]

if TYPE_CHECKING:
    from .llm_interface import (
        ConfidenceMode,
        ModelResponse,
        UserResponse,
        load_template,
        query_llm,
    )
    from .single_player import SinglePlayerConfig
    from .two_player import TwoPlayerConfig
    from .utils import (
        BaseGameConfig,
        HistoryEntry,
        QuestionData,
        RoundResult,
        TrialStatistics,
        evaluate_answer,
        extract_question_from_dataset,
    )


def __getattr__(name: str):
    if name in {
        "SinglePlayerConfig",
        "TwoPlayerConfig",
        "BaseGameConfig",
        "HistoryEntry",
        "QuestionData",
        "RoundResult",
        "TrialStatistics",
        "evaluate_answer",
        "extract_question_from_dataset",
    }:
        module = import_module(".utils", __name__)
        return getattr(module, name)

    if name in {
        "ConfidenceMode",
        "ModelResponse",
        "UserResponse",
        "load_template",
        "query_llm",
    }:
        module = import_module(".llm_interface", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
