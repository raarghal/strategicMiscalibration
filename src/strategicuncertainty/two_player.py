"""
Two-player strategic delegation experiment.

This module simulates both the agent and the user with LLMs:
1. Agent LLM: Solves tasks and reports confidence scores.
2. User LLM: Sees the reported confidence, decides whether to delegate (paying cost c)
   or self-solve, and observes outcomes when delegation occurs.

The repeated interaction creates reputation dynamics where both players can adapt.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from tqdm import tqdm

from .llm_interface import UserResponse, load_template, query_llm
from .utils import (
    TEMPLATE_DIR,
    BaseGameConfig,
    HistoryEntry,
    RoundResult,
    TaskData,
    TrialStatistics,
    aggregate_trial_stats,
    compute_agent_stats,
    compute_baseline_stats,
    compute_confidence_comparison_stats,
    compute_mean,
    evaluate_solution,
    extract_task_from_dataset,
    solve_task_with_game_context,
    solve_task_without_game_context,
    sum_trial_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class TwoPlayerConfig(BaseGameConfig):
    """
    Configuration for running the two-player strategic delegation experiment.

    Extends BaseGameConfig with two-player specific settings.
    """

    user_model_name: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"

    # Prompt template for the user (agent templates inherited from BaseGameConfig)
    user_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "user_prompt.j2"
    )


def query_user_delegation(
    cfg: TwoPlayerConfig,
    reported_confidence: float,
    history: List[HistoryEntry],
) -> UserResponse:
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
        cfg.user_template_path,
        reported_confidence=reported_confidence,
        history=history,
        threshold=threshold,
        reward=cfg.reward,
        cost=cfg.cost,
        delegation_count=delegation_count,
        delegated_accuracy=delegated_accuracy,
        high_confidence_delegation_count=high_confidence_delegation_count,
        high_confidence_delegated_accuracy=high_confidence_delegated_accuracy,
        total_payoff=total_payoff,
        confidence_mode=cfg.confidence_mode.value,
    )

    response, _ = query_llm(
        cfg.user_model_name, prompt, UserResponse, cfg.max_tokens, cfg.temperature
    )
    return response


def compute_payoffs(
    user_decision: str,
    agent_correct: bool,
    reward: float,
    cost: float,
) -> Dict[str, float]:
    """Compute payoffs for both players based on the user's decision and agent correctness."""
    if user_decision == "DELEGATE":
        agent_payoff = cost
        user_payoff = reward - cost if agent_correct else -cost
    else:
        agent_payoff = 0.0
        user_payoff = reward if random.random() < 0.5 else 0.0

    return {"user_payoff": user_payoff, "agent_payoff": agent_payoff}


def run_one_trial(
    cfg: TwoPlayerConfig,
    dataset: datasets.Dataset,
    trial_idx: int,
    progress: Optional[tqdm] = None,
) -> Dict[str, Any]:
    """Run a single trial of the two-player game."""
    history: List[HistoryEntry] = []
    round_results: List[RoundResult] = []
    dataset_size = len(dataset)

    for round_idx in range(cfg.num_rounds):
        sample_idx = random.randint(0, dataset_size - 1)
        sample = dataset[sample_idx]
        task_data: TaskData = extract_task_from_dataset(sample)
        task = task_data["task"]
        correct_solution = task_data["correct_solution"]
        difficulty = task_data["difficulty"]

        agent_history: List[HistoryEntry] = [
            {
                "round": record["round"],
                "reported_confidence": record["reported_confidence"],
                "was_correct": record["was_correct"],
                "user_decision": record["user_decision"],
                "agent_payoff": record["agent_payoff"],
            }
            for record in history
        ]

        try:
            baseline_response = solve_task_without_game_context(
                agent_model_name=cfg.agent_model_name,
                task=task,
                confidence_mode=cfg.confidence_mode,
                template_path=cfg.baseline_template_path,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            baseline_solution = baseline_response.solution
            baseline_confidence = baseline_response.confidence
            baseline_correct = evaluate_solution(baseline_solution, correct_solution)
        except Exception as exc:
            logger.error(
                f"Error in baseline query (trial {trial_idx}, round {round_idx}): {exc}"
            )
            print(
                f"Error in baseline query (trial {trial_idx}, round {round_idx}): {exc}"
            )
            baseline_solution = None
            baseline_confidence = None
            baseline_correct = None

        try:
            agent_response = solve_task_with_game_context(
                agent_model_name=cfg.agent_model_name,
                task=task,
                confidence_mode=cfg.confidence_mode,
                template_path=cfg.game_template_path,
                reward=cfg.reward,
                cost=cfg.cost,
                discount_factor=cfg.discount_factor,
                threshold=cfg.compute_threshold(),
                history=agent_history,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            agent_solution = agent_response.solution
            agent_confidence = agent_response.confidence
            agent_correct = evaluate_solution(agent_solution, correct_solution)
        except Exception as exc:
            logger.error(
                f"Error in agent query (trial {trial_idx}, round {round_idx}): {exc}"
            )
            print(f"Error in agent query (trial {trial_idx}, round {round_idx}): {exc}")
            agent_solution = None
            agent_confidence = None
            agent_correct = None

        if agent_confidence is None:
            if progress:
                progress.update(1)
            continue

        try:
            user_response = query_user_delegation(cfg, agent_confidence, history)
            user_decision = user_response.decision.upper()
            user_reasoning = user_response.reasoning
            user_belief = user_response.belief_agent_correct

            if user_decision not in {"DELEGATE", "SELF_SOLVE"}:
                user_decision = (
                    "DELEGATE" if "DELEGATE" in user_decision else "SELF_SOLVE"
                )
        except Exception as exc:
            logger.error(
                f"Error in user query (trial {trial_idx}, round {round_idx}): {exc}"
            )
            print(f"Error in user query (trial {trial_idx}, round {round_idx}): {exc}")
            user_decision = None
            user_reasoning = None
            user_belief = None

        if user_decision is None:
            if progress:
                progress.update(1)
            continue

        payoffs = compute_payoffs(
            user_decision, bool(agent_correct), cfg.reward, cfg.cost
        )

        confidence_diff = None
        if baseline_confidence is not None and agent_confidence is not None:
            confidence_diff = agent_confidence - baseline_confidence

        round_result: RoundResult = {
            "round": round_idx,
            "sample_idx": sample_idx,
            "task": task,
            "difficulty": difficulty,
            "correct_solution": correct_solution,
            "baseline_solution": baseline_solution,
            "baseline_confidence": baseline_confidence,
            "baseline_correct": baseline_correct,
            "agent_solution": agent_solution,
            "agent_confidence": agent_confidence,
            "agent_correct": agent_correct,
            "confidence_diff": confidence_diff,
            "user_decision": user_decision,
            "user_reasoning": user_reasoning,
            "user_belief_agent_correct": user_belief,
            "user_payoff": payoffs["user_payoff"],
            "agent_payoff": payoffs["agent_payoff"],
        }
        round_results.append(round_result)

        history_entry: HistoryEntry = {
            "round": round_idx,
            "reported_confidence": agent_confidence,
            "user_decision": user_decision,
            "was_correct": bool(agent_correct),
            "user_payoff": payoffs["user_payoff"],
            "agent_payoff": payoffs["agent_payoff"],
        }
        history.append(history_entry)

        if progress:
            progress.update(1)

    trial_stats = compute_trial_statistics(round_results, cfg)

    return {
        "trial_idx": trial_idx,
        "num_rounds_completed": len(round_results),
        "round_results": round_results,
        "statistics": trial_stats,
    }


def compute_trial_statistics(
    round_results: List[RoundResult], cfg: TwoPlayerConfig
) -> TrialStatistics:
    """Compute summary statistics for a single trial."""
    if not round_results:
        return {"error": "No valid rounds in trial"}  # type: ignore[return-value]

    threshold = cfg.compute_threshold()

    baseline_stats = compute_baseline_stats(round_results)
    agent_stats = compute_agent_stats(round_results, threshold)
    comparison_stats = compute_confidence_comparison_stats(round_results)

    delegation_count = sum(
        1 for record in round_results if record.get("user_decision") == "DELEGATE"
    )
    self_solve_count = len(round_results) - delegation_count
    delegation_rate = delegation_count / len(round_results) if round_results else 0.0

    user_beliefs = [
        record["user_belief_agent_correct"]
        for record in round_results
        if record.get("user_belief_agent_correct") is not None
    ]
    mean_user_belief = compute_mean(user_beliefs)
    agent_accuracy = agent_stats.get("agent_accuracy", 0)
    belief_error = (
        abs(mean_user_belief - agent_accuracy) if mean_user_belief is not None else None
    )

    total_user_payoff = sum(record.get("user_payoff", 0) for record in round_results)
    total_agent_payoff = sum(record.get("agent_payoff", 0) for record in round_results)
    mean_user_payoff = total_user_payoff / len(round_results) if round_results else 0.0
    mean_agent_payoff = (
        total_agent_payoff / len(round_results) if round_results else 0.0
    )

    stats: TrialStatistics = {
        "num_rounds": len(round_results),
        **baseline_stats,
        **agent_stats,
        **comparison_stats,
        "delegation_count": delegation_count,
        "self_solve_count": self_solve_count,
        "delegation_rate": delegation_rate,
        "mean_user_belief_agent_correct": mean_user_belief,
        "user_belief_error": belief_error,
        "total_user_payoff": total_user_payoff,
        "total_agent_payoff": total_agent_payoff,
        "mean_user_payoff": mean_user_payoff,
        "mean_agent_payoff": mean_agent_payoff,
    }

    return stats


def run_trials(cfg: TwoPlayerConfig) -> Dict[str, Any]:
    """Run multiple trials of the two-player experiment and save results."""
    random.seed(cfg.seed)
    logger.info(f"Starting two-player experiment with seed {cfg.seed}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent.parent.parent
        / Path(cfg.output_dir)
        / f"two_player_{timestamp}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    print(f"Loading dataset: {cfg.dataset_name}")
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    dataset = datasets.load_dataset(cfg.dataset_name, split="test")
    print(f"Dataset loaded with {len(dataset)} samples")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    print("\n" + "=" * 70)
    print("Two-Player Strategic Delegation Game")
    print("=" * 70)
    print(f"\nAgent model: {cfg.agent_model_name}")
    print(f"User model: {cfg.user_model_name}")
    print("\nInteraction Parameters:")
    print(f"  Reward (r): {cfg.reward}")
    print(f"  Delegation cost (c): {cfg.cost}")
    print(f"  Discount factor (δ): {cfg.discount_factor}")
    print(f"  User delegation threshold (θ*): {cfg.compute_threshold():.4f}")
    print(f"  Confidence mode: {cfg.confidence_mode.value}")
    print(f"\nExperiment: {cfg.num_trials} trials × {cfg.num_rounds} rounds\n")

    all_trial_results: List[Dict[str, Any]] = []
    progress = tqdm(
        total=cfg.num_trials * cfg.num_rounds, desc="Running two-player game"
    )

    for trial_idx in range(cfg.num_trials):
        logger.debug(f"Starting trial {trial_idx}")
        trial_result = run_one_trial(cfg, dataset, trial_idx, progress)
        all_trial_results.append(trial_result)

    progress.close()

    valid_trials = [
        trial
        for trial in all_trial_results
        if "error" not in trial.get("statistics", {})
    ]
    logger.info(f"Completed {len(valid_trials)} valid trials out of {cfg.num_trials}")

    if valid_trials:
        overall_stats = compute_overall_statistics(valid_trials)
    else:
        overall_stats = {"error": "No valid trials completed"}
        logger.warning("No valid trials completed")

    results = {
        "timestamp": timestamp,
        "config": {
            "agent_model_name": cfg.agent_model_name,
            "user_model_name": cfg.user_model_name,
            "dataset_name": cfg.dataset_name,
            "num_trials": cfg.num_trials,
            "num_rounds": cfg.num_rounds,
            "temperature": cfg.temperature,
            "confidence_mode": cfg.confidence_mode.value,
            "reward": cfg.reward,
            "cost": cfg.cost,
            "discount_factor": cfg.discount_factor,
            "threshold": cfg.compute_threshold(),
            "seed": cfg.seed,
        },
        "overall_statistics": overall_stats,
        "trial_results": all_trial_results,
    }

    with open(output_path / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, default=str)

    summary = generate_summary_report(cfg, timestamp, overall_stats)
    with open(output_path / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write(summary)

    print(summary)
    print(f"\nResults saved to: {output_path}")
    logger.info(f"Results saved to: {output_path}")

    return results


def compute_overall_statistics(valid_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall statistics aggregated across all valid trials."""
    baseline_accuracies = aggregate_trial_stats(valid_trials, "baseline_accuracy")
    baseline_confidences = aggregate_trial_stats(
        valid_trials, "mean_baseline_confidence"
    )
    agent_accuracies = aggregate_trial_stats(valid_trials, "agent_accuracy")
    agent_confidences = aggregate_trial_stats(valid_trials, "mean_agent_confidence")
    confidence_diffs = aggregate_trial_stats(valid_trials, "mean_confidence_diff")
    delegation_rates = aggregate_trial_stats(valid_trials, "delegation_rate")
    user_beliefs = aggregate_trial_stats(valid_trials, "mean_user_belief_agent_correct")
    belief_errors = aggregate_trial_stats(valid_trials, "user_belief_error")

    return {
        "num_trials": len(valid_trials),
        "total_rounds": sum(trial["num_rounds_completed"] for trial in valid_trials),
        "mean_baseline_accuracy": compute_mean(baseline_accuracies),
        "mean_baseline_confidence": compute_mean(baseline_confidences),
        "mean_agent_accuracy": compute_mean(agent_accuracies),
        "mean_agent_confidence": compute_mean(agent_confidences),
        "mean_confidence_diff": compute_mean(confidence_diffs),
        "total_confidence_inflated": sum_trial_stats(
            valid_trials, "confidence_inflated_count"
        ),
        "total_confidence_deflated": sum_trial_stats(
            valid_trials, "confidence_deflated_count"
        ),
        "total_confidence_unchanged": sum_trial_stats(
            valid_trials, "confidence_unchanged_count"
        ),
        "mean_delegation_rate": compute_mean(delegation_rates),
        "mean_user_belief_agent_correct": compute_mean(user_beliefs),
        "mean_user_belief_error": compute_mean(belief_errors),
        "total_user_payoff": sum_trial_stats(valid_trials, "total_user_payoff"),
        "total_agent_payoff": sum_trial_stats(valid_trials, "total_agent_payoff"),
        "mean_user_payoff_per_round": compute_mean(
            aggregate_trial_stats(valid_trials, "mean_user_payoff")
        ),
        "mean_agent_payoff_per_round": compute_mean(
            aggregate_trial_stats(valid_trials, "mean_agent_payoff")
        ),
    }


def generate_summary_report(
    cfg: TwoPlayerConfig,
    timestamp: str,
    overall_stats: Dict[str, Any],
) -> str:
    """Generate a human-readable summary report."""
    lines = [
        "=" * 70,
        "Two-Player Strategic Delegation Game Results",
        "=" * 70,
        f"Timestamp: {timestamp}",
        f"Agent model: {cfg.agent_model_name}",
        f"User model: {cfg.user_model_name}",
        f"Dataset: {cfg.dataset_name}",
        f"Trials: {cfg.num_trials}, Rounds per trial: {cfg.num_rounds}",
        "-" * 70,
        "Interaction Parameters:",
        f"  Reward (r): {cfg.reward}",
        f"  Delegation cost (c): {cfg.cost}",
        f"  Discount factor (δ): {cfg.discount_factor}",
        f"  User delegation threshold (θ*): {cfg.compute_threshold():.4f}",
        f"  Confidence mode: {cfg.confidence_mode.value}",
        "-" * 70,
        "Overall Statistics:",
    ]

    if "error" not in overall_stats:
        lines.extend(
            [
                "",
                "Baseline Performance (no strategic context):",
            ]
        )
        if overall_stats.get("mean_baseline_accuracy") is not None:
            lines.append(
                f"  Mean accuracy: {overall_stats['mean_baseline_accuracy']:.4f}"
            )
        if overall_stats.get("mean_baseline_confidence") is not None:
            lines.append(
                f"  Mean confidence: {overall_stats['mean_baseline_confidence']:.4f}"
            )

        lines.extend(
            [
                "",
                "Agent Performance (strategic context):",
            ]
        )
        if overall_stats.get("mean_agent_accuracy") is not None:
            lines.append(f"  Mean accuracy: {overall_stats['mean_agent_accuracy']:.4f}")
        if overall_stats.get("mean_agent_confidence") is not None:
            lines.append(
                f"  Mean reported confidence: {overall_stats['mean_agent_confidence']:.4f}"
            )

        lines.extend(
            [
                "",
                "Confidence Comparison (strategic - baseline):",
            ]
        )
        if overall_stats.get("mean_confidence_diff") is not None:
            lines.append(
                f"  Mean difference: {overall_stats['mean_confidence_diff']:.4f}"
            )
        lines.append(
            f"  Inflated (strategic > baseline): {overall_stats['total_confidence_inflated']} times"
        )
        lines.append(
            f"  Deflated (strategic < baseline): {overall_stats['total_confidence_deflated']} times"
        )
        lines.append(
            f"  Unchanged: {overall_stats['total_confidence_unchanged']} times"
        )

        lines.extend(
            [
                "",
                "User Behaviour:",
            ]
        )
        if overall_stats.get("mean_delegation_rate") is not None:
            lines.append(
                f"  Delegation rate: {overall_stats['mean_delegation_rate']:.4f}"
            )
        if overall_stats.get("mean_user_belief_agent_correct") is not None:
            lines.append(
                f"  Mean belief agent is correct: {overall_stats['mean_user_belief_agent_correct']:.4f}"
            )
        if overall_stats.get("mean_user_belief_error") is not None:
            lines.append(
                f"  User belief calibration error: {overall_stats['mean_user_belief_error']:.4f}"
            )

        lines.extend(
            [
                "",
                "Payoffs:",
            ]
        )
        if overall_stats.get("total_user_payoff") is not None:
            lines.append(
                f"  Total user payoff: {overall_stats['total_user_payoff']:.2f}"
            )
        if overall_stats.get("total_agent_payoff") is not None:
            lines.append(
                f"  Total agent payoff: {overall_stats['total_agent_payoff']:.2f}"
            )
        if overall_stats.get("mean_user_payoff_per_round") is not None:
            lines.append(
                f"  Mean user payoff per round: {overall_stats['mean_user_payoff_per_round']:.4f}"
            )
        if overall_stats.get("mean_agent_payoff_per_round") is not None:
            lines.append(
                f"  Mean agent payoff per round: {overall_stats['mean_agent_payoff_per_round']:.4f}"
            )
    else:
        lines.append(f"  Error: {overall_stats['error']}")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    config = TwoPlayerConfig(
        agent_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        user_model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        num_trials=1,
        num_rounds=5,
    )

    run_trials(config)
