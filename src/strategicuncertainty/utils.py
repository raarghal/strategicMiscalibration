"""
Common utilities, types, and configuration for the strategic delegation experiments.

This module contains shared functionality used by both single_player.py and two_player.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from .llm_interface import AgentResponse, ConfidenceMode, load_template, query_llm

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"


# =============================================================================
# Type Definitions
# =============================================================================


class TaskData(TypedDict):
    """Extracted task data from a dataset sample."""

    task: str
    correct_solution: str
    difficulty: Optional[str]


class RoundResult(TypedDict, total=False):
    """Result from a single round of the game."""

    round: int
    sample_idx: int
    task: str
    difficulty: Optional[str]
    correct_solution: str
    # Baseline (no strategic context)
    baseline_solution: Optional[str]
    baseline_confidence: Optional[float]
    baseline_correct: Optional[bool]
    # Agent with strategic context
    agent_solution: Optional[str]
    agent_confidence: Optional[float]
    agent_correct: Optional[bool]
    # Comparison metrics
    confidence_diff: Optional[float]
    # User behaviour (two-player mode)
    user_decision: Optional[str]
    user_reasoning: Optional[str]
    user_belief_agent_correct: Optional[float]
    user_payoff: Optional[float]
    agent_payoff: Optional[float]


class HistoryEntry(TypedDict, total=False):
    """A single entry in the interaction history."""

    round: int
    reported_confidence: float
    was_correct: bool
    user_decision: str
    user_payoff: float
    agent_payoff: float


class TrialStatistics(TypedDict, total=False):
    """Statistics computed for a single trial."""

    num_rounds: int
    # Baseline stats
    baseline_accuracy: Optional[float]
    mean_baseline_confidence: Optional[float]
    # Agent stats
    agent_accuracy: float
    mean_agent_confidence: float
    high_confidence_count: int
    high_confidence_accuracy: Optional[float]
    low_confidence_count: int
    low_confidence_accuracy: Optional[float]
    # Confidence comparison
    mean_confidence_diff: Optional[float]
    confidence_inflated_count: int
    confidence_deflated_count: int
    confidence_unchanged_count: int
    # User behaviour (two-player mode)
    delegation_count: int
    self_solve_count: int
    delegation_rate: float
    mean_user_belief_agent_correct: float
    user_belief_error: float
    # Payoffs
    total_user_payoff: float
    total_agent_payoff: float
    mean_user_payoff: float
    mean_agent_payoff: float


# =============================================================================
# Base Configuration
# =============================================================================


@dataclass
class BaseGameConfig:
    """
    Base configuration shared by all game experiments.

    Contains common settings for LLM queries, dataset, game parameters,
    confidence reporting, and experiment settings.
    """

    # LLM configuration
    agent_model_name: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 512
    temperature: float = 0.0

    # Prompt templates
    baseline_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "baseline_prompt.j2"
    )
    game_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "game_prompt.j2"
    )

    # Dataset
    dataset_name: str = "watermelonhjg/TAL-SCQ5K-EN-R1"

    # Game parameters
    reward: float = 1.0  # Reward the user gains if a delegated task is solved correctly
    cost: float = 0.1  # Cost the user pays to delegate the task
    discount_factor: float = 0.9  # δ: discount factor for future payoffs

    # Confidence reporting
    confidence_mode: ConfidenceMode = ConfidenceMode.CONTINUOUS

    # Experiment settings
    num_trials: int = 1
    num_rounds: int = 10
    output_dir: str = "outputs"
    seed: int = 42

    def compute_threshold(self) -> float:
        """
        Compute the delegation threshold θ* for the user.

        θ* = 0.5 + c/r
        The user delegates when their belief ≥ θ*.
        """
        return 0.5 + self.cost / self.reward


# =============================================================================
# Dataset Utilities
# =============================================================================


def extract_task_from_dataset(sample: Dict[str, Any]) -> TaskData:
    """Extract task information from a TAL-SCQ5K dataset sample."""
    task = sample.get("problem", "")
    difficulty = sample.get("difficulty", None)
    solution = sample.get("solution", "")

    correct_solution = ""
    if solution:
        solution_stripped = solution.strip()
        if solution_stripped.startswith("(") and len(solution_stripped) >= 2:
            correct_solution = solution_stripped[1]
        else:
            correct_solution = solution_stripped

    return TaskData(task=task, correct_solution=correct_solution, difficulty=difficulty)


def evaluate_solution(response_solution: str, correct_solution: str) -> bool:
    """Evaluate whether the agent's solution matches the correct solution."""
    response_normalized = response_solution.strip().upper()
    correct_normalized = correct_solution.strip().upper()

    response_letter = ""
    if response_normalized:
        if response_normalized[0] == "(" and len(response_normalized) >= 2:
            response_letter = response_normalized[1]
        elif response_normalized[0].isalpha():
            response_letter = response_normalized[0]

    return response_letter == correct_normalized


# =============================================================================
# Agent Query Helpers
# =============================================================================


def solve_task_without_game_context(
    agent_model_name: str,
    task: str,
    confidence_mode: ConfidenceMode,
    template_path: Path,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> AgentResponse:
    """Query the agent LLM on a task without strategic context."""
    logger.debug(f"Solving baseline task with agent {agent_model_name}")
    prompt = load_template(
        template_path,
        task=task,
        confidence_mode=confidence_mode.value,
    )
    response, _ = query_llm(
        agent_model_name, prompt, AgentResponse, max_tokens, temperature
    )
    return response


def solve_task_with_game_context(
    agent_model_name: str,
    task: str,
    confidence_mode: ConfidenceMode,
    template_path: Path,
    reward: float,
    cost: float,
    discount_factor: float,
    threshold: float,
    history: Optional[List[HistoryEntry]] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> AgentResponse:
    """Query the agent LLM on a task with strategic context."""
    logger.debug(f"Solving task with strategic context using agent {agent_model_name}")

    if history:
        delegation_count = sum(
            1 for h in history if h.get("user_decision") == "DELEGATE"
        )
        correct_count = sum(1 for h in history if h.get("was_correct"))
        agent_accuracy = correct_count / len(history)
        total_agent_payoff = sum(h.get("agent_payoff", 0) for h in history)
    else:
        delegation_count = 0
        agent_accuracy = "N/A"
        total_agent_payoff = 0.0

    prompt = load_template(
        template_path,
        task=task,
        reward=reward,
        cost=cost,
        discount_factor=discount_factor,
        threshold=threshold,
        history=history,
        agent_accuracy=agent_accuracy,
        delegation_count=delegation_count,
        total_agent_payoff=total_agent_payoff,
        confidence_mode=confidence_mode.value,
    )
    response, _ = query_llm(
        agent_model_name, prompt, AgentResponse, max_tokens, temperature
    )
    return response


# =============================================================================
# Statistics Helpers
# =============================================================================


def compute_mean(values: List[float]) -> Optional[float]:
    """Compute the mean of a list, returning None if empty."""
    return sum(values) / len(values) if values else None


def filter_none(values: List[Optional[float]]) -> List[float]:
    """Filter out None values from a list."""
    return [v for v in values if v is not None]


def compute_confidence_comparison_stats(
    round_results: List[RoundResult],
) -> Dict[str, Any]:
    """Compute statistics comparing baseline versus strategic confidence."""
    confidence_diffs = filter_none([r.get("confidence_diff") for r in round_results])

    return {
        "mean_confidence_diff": compute_mean(confidence_diffs),
        "confidence_inflated_count": sum(1 for d in confidence_diffs if d > 0),
        "confidence_deflated_count": sum(1 for d in confidence_diffs if d < 0),
        "confidence_unchanged_count": sum(1 for d in confidence_diffs if d == 0),
    }


def compute_baseline_stats(round_results: List[RoundResult]) -> Dict[str, Any]:
    """Compute baseline performance statistics."""
    baseline_confidences = filter_none(
        [r.get("baseline_confidence") for r in round_results]
    )
    baseline_rounds = [
        r for r in round_results if r.get("baseline_correct") is not None
    ]
    baseline_correct = sum(1 for r in baseline_rounds if r.get("baseline_correct"))

    return {
        "baseline_accuracy": baseline_correct / len(baseline_rounds)
        if baseline_rounds
        else None,
        "mean_baseline_confidence": compute_mean(baseline_confidences),
    }


def compute_agent_stats(
    round_results: List[RoundResult], threshold: float
) -> Dict[str, Any]:
    """Compute agent performance statistics with strategic context."""
    agent_confidences = filter_none([r.get("agent_confidence") for r in round_results])
    agent_rounds = [r for r in round_results if r.get("agent_correct") is not None]
    agent_correct = sum(1 for r in agent_rounds if r.get("agent_correct"))

    high_confidence_rounds = [
        r
        for r in round_results
        if r.get("agent_confidence") is not None and r["agent_confidence"] >= threshold
    ]
    low_confidence_rounds = [
        r
        for r in round_results
        if r.get("agent_confidence") is not None and r["agent_confidence"] < threshold
    ]

    high_confidence_correct = sum(
        1 for r in high_confidence_rounds if r.get("agent_correct")
    )
    low_confidence_correct = sum(
        1 for r in low_confidence_rounds if r.get("agent_correct")
    )

    total_rounds = len(agent_rounds)

    return {
        "agent_accuracy": agent_correct / total_rounds if total_rounds else 0.0,
        "mean_agent_confidence": compute_mean(agent_confidences),
        "high_confidence_count": len(high_confidence_rounds),
        "high_confidence_accuracy": high_confidence_correct
        / len(high_confidence_rounds)
        if high_confidence_rounds
        else None,
        "low_confidence_count": len(low_confidence_rounds),
        "low_confidence_accuracy": low_confidence_correct / len(low_confidence_rounds)
        if low_confidence_rounds
        else None,
    }


def aggregate_trial_stats(
    trial_results: List[Dict[str, Any]], stat_key: str
) -> List[float]:
    """Aggregate a specific statistic across multiple trials, filtering None values."""
    return [
        t["statistics"][stat_key]
        for t in trial_results
        if t.get("statistics", {}).get(stat_key) is not None
    ]


def sum_trial_stats(trial_results: List[Dict[str, Any]], stat_key: str) -> float:
    """Sum a specific statistic across multiple trials."""
    return sum(
        t["statistics"].get(stat_key, 0) for t in trial_results if "statistics" in t
    )
