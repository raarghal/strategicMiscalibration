"""
Common utilities, types, and configuration for the strategic delegation experiments.

This module contains shared functionality used by both single_player.py and two_player.py.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .datatypes import (
    BaseGameConfig,
    HistoryEntry,
    RoundResult,
    TaskData,
)
from .llm_interface import (
    AgentBaselineResponse,
    AgentGameResponse,
    UserDecisionResponse,
    UserPosteriorResponse,
    load_template,
    query_llm,
)

logger = logging.getLogger(__name__)


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
    cfg: BaseGameConfig,
    task: str,
) -> AgentBaselineResponse:
    """Query the agent LLM on a task without strategic context."""
    logger.debug(f"Solving baseline task with agent {cfg.agent_model_name}")
    prompt = load_template(
        cfg.baseline_template_path,
        task=task,
        confidence_mode=cfg.confidence_mode.value,
    )
    response, _ = query_llm(
        cfg.agent_model_name,
        prompt,
        AgentBaselineResponse,
        cfg.max_tokens,
        cfg.temperature,
    )
    return response


def solve_task_with_game_context(
    cfg: BaseGameConfig,
    task: str,
    history: Optional[List[HistoryEntry]] = None,
) -> AgentGameResponse:
    """Query the agent LLM on a task with strategic context."""
    logger.debug(
        f"Solving task with strategic context using agent {cfg.agent_model_name}"
    )

    if history:
        delegation_count = sum(
            1 for h in history if h.get("user_decision") == "DELEGATE"
        )
        correct_count = sum(1 for h in history if h.get("was_correct"))
        agent_accuracy = correct_count / len(history)
        total_agent_payoff = sum(h.get("agent_payoff", 0) for h in history)
    else:
        delegation_count = 0
        agent_accuracy = None
        total_agent_payoff = 0.0

    # Use a safe default so that history can be None (e.g., in single-player mode)
    history_list = history or []

    prompt = load_template(
        cfg.game_template_path,
        task=task,
        reward=cfg.reward,
        cost=cfg.cost,
        effort=cfg.effort,
        discount_factor=cfg.discount_factor,
        num_rounds=cfg.num_rounds,
        threshold=cfg.compute_threshold(),
        history=history_list,
        round=len(history_list) + 1,
        agent_accuracy=agent_accuracy,
        delegation_count=delegation_count,
        total_agent_payoff=total_agent_payoff,
        confidence_mode=cfg.confidence_mode.value,
        priors=cfg.priors,
        prior_honesty=cfg.h_0,
        prior_capability=cfg.mu_0,
    )
    # print(prompt)
    response, _ = query_llm(
        cfg.agent_model_name, prompt, AgentGameResponse, cfg.max_tokens, cfg.temperature
    )
    return response


def query_user_delegation(
    cfg: BaseGameConfig,
    reported_confidence: float,
    history: List[HistoryEntry],
    h_t: Optional[float] = None,
    mu_t: Optional[float] = None,
) -> UserDecisionResponse:
    """Query the user LLM with the reported confidence and interaction history."""
    logger.debug(f"Asking user LLM with reported confidence {reported_confidence}")
    threshold = cfg.compute_threshold()

    delegation_count = 0
    delegated_correct = 0
    high_confidence_delegation_count = 0
    high_confidence_delegated_correct = 0
    total_payoff = 0.0

    for record in history:
        total_payoff += record.get("user_payoff", 0)
        if record.get("user_decision") == "DELEGATE":
            delegation_count += 1
            if record.get("was_correct"):
                delegated_correct += 1
            if record.get("reported_confidence", 0) >= threshold:
                high_confidence_delegation_count += 1
                if record.get("was_correct"):
                    high_confidence_delegated_correct += 1

    delegated_accuracy = (
        delegated_correct / delegation_count if delegation_count > 0 else "N/A"
    )
    high_confidence_delegated_accuracy = (
        high_confidence_delegated_correct / high_confidence_delegation_count
        if high_confidence_delegation_count > 0
        else "N/A"
    )

    prompt = load_template(
        cfg.user_decision_template_path,
        reported_confidence=reported_confidence,
        history=history,
        threshold=threshold,
        reward=cfg.reward,
        cost=cfg.cost,
        effort=cfg.effort,
        delegation_count=delegation_count,
        delegated_accuracy=delegated_accuracy,
        high_confidence_delegation_count=high_confidence_delegation_count,
        high_confidence_delegated_accuracy=high_confidence_delegated_accuracy,
        total_payoff=total_payoff,
        confidence_mode=cfg.confidence_mode.value,
        priors=cfg.priors,
        prior_ability=mu_t,
        prior_honesty=h_t,
    )

    response, _ = query_llm(
        cfg.user_model_name,
        prompt,
        UserDecisionResponse,
        cfg.max_tokens,
        cfg.temperature,
    )
    return response


def query_user_posterior(
    cfg: BaseGameConfig,
    reported_confidence: float,
    agent_correct: bool,
    prior_beliefs: Dict[str, Optional[float]],
    history: List[HistoryEntry],
) -> UserPosteriorResponse:
    """Query the user LLM to update beliefs after observing the delegated outcome."""
    logger.debug(
        f"Asking user LLM for posterior update (reported_confidence={reported_confidence}, agent_correct={agent_correct})"
    )
    threshold = cfg.compute_threshold()

    prompt = load_template(
        cfg.user_posterior_template_path,
        reported_confidence=reported_confidence,
        agent_correct=agent_correct,
        prior_belief_agent_correct=prior_beliefs.get("belief_agent_correct"),
        prior_belief_agent_ability=prior_beliefs.get("belief_agent_ability"),
        prior_belief_honesty=prior_beliefs.get("belief_honesty"),
        history=history,
        threshold=threshold,
        reward=cfg.reward,
        cost=cfg.cost,
        effort=cfg.effort,
        confidence_mode=cfg.confidence_mode.value,
    )

    response, _ = query_llm(
        cfg.user_model_name,
        prompt,
        UserPosteriorResponse,
        cfg.max_tokens,
        cfg.temperature,
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


def load_two_player_results_to_df(results_path: Path | str):
    """
    Load a two_player results.json file and return a pandas DataFrame
    where each row corresponds to one round of one trial, merged with config.

    Columns include all keys from the top-level "config" object plus:
    - trial_idx
    - num_rounds_completed
    - all fields from each round_results entry (e.g., round, sample_idx, task, etc.)

    Args:
        results_path: Path to the results.json file produced by two_player.py

    Returns:
        pandas.DataFrame with per-round rows and config merged.
    """
    import json

    import pandas as pd

    # Read results JSON
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data.get("config", {}) or {}
    trials = data.get("trial_results", []) or []

    rows = []
    for trial in trials:
        trial_idx = trial.get("trial_idx")
        num_rounds_completed = trial.get("num_rounds_completed")
        round_results = trial.get("round_results", []) or []

        for rr in round_results:
            # Merge config with trial-level identifiers and the per-round result
            row = {**config}
            row["trial_idx"] = trial_idx
            row["num_rounds_completed"] = num_rounds_completed
            # Include all round result fields verbatim
            row.update(rr or {})
            rows.append(row)

    df = pd.DataFrame(rows)
    return df
