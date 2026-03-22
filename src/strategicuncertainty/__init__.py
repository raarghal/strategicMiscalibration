"""
Strategic Uncertainty Quantification package.

This package provides tools for studying strategic behavior in LLM confidence reporting.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "BaseGameConfig",
    "ConfidenceMode",
    "HistoryEntry",
    "TaskData",
    "RoundResult",
    "TrialStatistics",
    "AgentBaselineResponse",
    "AgentGameResponse",
    "UserDecisionResponse",
    "UserPosteriorResponse",
    "evaluate_solution",
    "extract_task_from_dataset",
    "load_template",
    "query_llm",
]

if TYPE_CHECKING:
    from .llm_interface import (
        AgentBaselineResponse,
        AgentGameResponse,
        ConfidenceMode,
        UserDecisionResponse,
        UserPosteriorResponse,
        load_template,
        query_llm,
    )
    from .datatypes import (
        BaseGameConfig,
        HistoryEntry,
        RoundResult,
        TaskData,
        TrialStatistics,
    )
    from .utils import (
        evaluate_solution,
        extract_task_from_dataset,
    )


def __getattr__(name: str):
    if name in {
        "BaseGameConfig",
        "HistoryEntry",
        "TaskData",
        "RoundResult",
        "TrialStatistics",
    }:
        module = import_module(".datatypes", __name__)
        return getattr(module, name)

    if name in {
        "evaluate_solution",
        "extract_task_from_dataset",
    }:
        module = import_module(".utils", __name__)
        return getattr(module, name)

    if name in {
        "ConfidenceMode",
        "AgentBaselineResponse",
        "AgentGameResponse",
        "UserDecisionResponse",
        "UserPosteriorResponse",
        "load_template",
        "query_llm",
    }:
        module = import_module(".llm_interface", __name__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
