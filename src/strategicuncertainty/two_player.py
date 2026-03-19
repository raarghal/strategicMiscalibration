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
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
from tqdm import tqdm

from .datatypes import (
    BaseGameConfig,
    HistoryEntry,
    RoundResult,
    TaskData,
    TrialStatistics,
)
from .utils import (
    aggregate_trial_stats,
    compute_agent_stats,
    compute_baseline_stats,
    compute_confidence_comparison_stats,
    compute_mean,
    evaluate_solution,
    extract_task_from_dataset,
    query_user_delegation,
    query_user_posterior,
    solve_task_with_game_context,
    solve_task_without_game_context,
    sum_trial_stats,
)

logger = logging.getLogger(__name__)


def compute_payoffs(
    user_decision: str,
    agent_correct: bool,
    reward: float,
    cost: float,
    effort: float,
) -> Dict[str, float]:
    """Compute payoffs for both players based on the user's decision, agent correctness, and self-solve effort."""
    if user_decision == "DELEGATE":
        agent_payoff = cost
        user_payoff = reward - cost if agent_correct else -cost
    else:
        # User self-solves:
        # Deterministic outcome: earns reward minus effort cost
        agent_payoff = 0.0
        user_payoff = reward - effort

    return {"user_payoff": user_payoff, "agent_payoff": agent_payoff}


def run_one_trial(
    cfg: BaseGameConfig,
    dataset: datasets.Dataset,
    trial_idx: int,
    progress: Optional[tqdm] = None,
) -> Dict[str, Any]:
    """Run a single trial of the two-player game."""
    history: List[HistoryEntry] = []
    round_results: List[RoundResult] = []
    dataset_size = len(dataset)

    h_t = cfg.h_0
    mu_t = cfg.mu_0

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
                cfg=cfg,
                task=task,
            )
            baseline_solution = baseline_response.solution
            baseline_confidence = baseline_response.confidence
            baseline_correct = evaluate_solution(baseline_solution, correct_solution)
        except Exception:
            logger.exception(
                "Error in baseline query (trial=%s, round=%s)",
                trial_idx,
                round_idx,
            )
            baseline_solution = None
            baseline_confidence = None
            baseline_correct = None

        try:
            agent_response = solve_task_with_game_context(
                cfg=cfg,
                task=task,
                history=agent_history,
            )
            agent_solution = agent_response.solution
            agent_confidence = agent_response.confidence
            agent_reasoning = agent_response.reasoning
            agent_correct = evaluate_solution(agent_solution, correct_solution)
        except Exception:
            logger.exception(
                "Error in agent query (trial=%s, round=%s)",
                trial_idx,
                round_idx,
            )
            agent_solution = None
            agent_confidence = None
            agent_reasoning = None
            agent_correct = None

        if agent_confidence is None:
            if progress:
                progress.update(1)
            continue

        try:
            user_response = query_user_delegation(
                cfg, agent_confidence, history, h_t, mu_t
            )
            user_decision = user_response.decision.upper()
            user_reasoning = user_response.reasoning
            user_belief_agent_correct = user_response.belief_agent_correct
            user_belief_agent_ability = user_response.belief_agent_ability
            user_belief_honesty = user_response.belief_honesty

            if user_decision not in {"DELEGATE", "SELF_SOLVE"}:
                user_decision = (
                    "DELEGATE" if "DELEGATE" in user_decision else "SELF_SOLVE"
                )
        except Exception:
            logger.exception(
                "Error in user query (trial=%s, round=%s)",
                trial_idx,
                round_idx,
            )
            user_decision = None
            user_reasoning = None
            user_belief_agent_correct = None
            user_belief_agent_ability = None
            user_belief_honesty = None

        if user_decision is None:
            if progress:
                progress.update(1)
            continue

        payoffs = compute_payoffs(
            user_decision, bool(agent_correct), cfg.reward, cfg.cost, cfg.effort
        )

        confidence_diff = None
        if baseline_confidence is not None and agent_confidence is not None:
            confidence_diff = agent_confidence - baseline_confidence

        # If delegated, ask user to update posterior beliefs after observing outcome
        posterior_user_belief_agent_ability = None
        posterior_user_belief_honesty = None
        if user_decision == "DELEGATE":
            try:
                prior_beliefs = {
                    "belief_agent_correct": user_belief_agent_correct,
                    "belief_agent_ability": user_belief_agent_ability,
                    "belief_honesty": user_belief_honesty,
                }
                posterior_response = query_user_posterior(
                    cfg=cfg,
                    reported_confidence=agent_confidence,
                    agent_correct=bool(agent_correct),
                    prior_beliefs=prior_beliefs,
                    history=history,
                )
                # Updated beliefs (posterior)
                posterior_user_belief_agent_ability = (
                    posterior_response.belief_agent_ability
                )
                posterior_user_belief_honesty = posterior_response.belief_honesty
            except Exception:
                logger.exception(
                    "Error in posterior user query (trial=%s, round=%s)",
                    trial_idx,
                    round_idx,
                )
        else:
            # No delegated outcome observed; posterior equals prior beliefs
            posterior_user_belief_agent_ability = user_belief_agent_ability
            posterior_user_belief_honesty = user_belief_honesty

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
            "agent_reasoning": agent_reasoning,
            "confidence_diff": confidence_diff,
            "prior_agent_honesty": h_t,
            "prior_agent_ability": mu_t,
            "user_decision": user_decision,
            "user_reasoning": user_reasoning,
            "user_belief_agent_correct": user_belief_agent_correct,
            "user_belief_agent_ability": user_belief_agent_ability,
            "user_belief_honesty": user_belief_honesty,
            "posterior_user_belief_agent_ability": posterior_user_belief_agent_ability,
            "posterior_user_belief_honesty": posterior_user_belief_honesty,
            "user_payoff": payoffs["user_payoff"],
            "agent_payoff": payoffs["agent_payoff"],
        }
        h_t = posterior_user_belief_honesty
        mu_t = posterior_user_belief_agent_ability
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
    round_results: List[RoundResult], cfg: BaseGameConfig
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
    user_beliefs_agent_ability = [
        record["user_belief_agent_ability"]
        for record in round_results
        if record.get("user_belief_agent_ability") is not None
    ]
    user_beliefs_honesty = [
        record["user_belief_honesty"]
        for record in round_results
        if record.get("user_belief_honesty") is not None
    ]
    mean_user_belief = compute_mean(user_beliefs)
    mean_user_belief_agent_ability = compute_mean(user_beliefs_agent_ability)
    mean_user_belief_honesty = compute_mean(user_beliefs_honesty)
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
        "mean_user_belief_agent_ability": mean_user_belief_agent_ability,
        "mean_user_belief_honesty": mean_user_belief_honesty,
        "mean_posterior_user_belief_agent_ability": compute_mean(
            [
                r["posterior_user_belief_agent_ability"]
                for r in round_results
                if r.get("posterior_user_belief_agent_ability") is not None
            ]
        ),
        "mean_posterior_user_belief_honesty": compute_mean(
            [
                r["posterior_user_belief_honesty"]
                for r in round_results
                if r.get("posterior_user_belief_honesty") is not None
            ]
        ),
        "user_belief_error": belief_error,
        "total_user_payoff": total_user_payoff,
        "total_agent_payoff": total_agent_payoff,
        "mean_user_payoff": mean_user_payoff,
        "mean_agent_payoff": mean_agent_payoff,
    }

    return stats


def run_trials(cfg: BaseGameConfig, progress: Optional[tqdm] = None) -> Dict[str, Any]:
    """Run multiple trials of the two-player experiment and save results."""
    random.seed(cfg.seed)
    logger.info("Starting two-player experiment with seed %s", cfg.seed)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent.parent.parent
        / Path(cfg.output_dir)
        / "trials"
        / f"two_player_{timestamp}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_path)

    logger.info("Loading dataset: %s", cfg.dataset_name)
    dataset = datasets.load_dataset(cfg.dataset_name, split="test")
    logger.info("Dataset loaded with %s samples", len(dataset))

    print("\n" + "=" * 70)
    print("Two-Player Strategic Delegation Game")
    print("=" * 70)
    print(f"\nAgent model: {cfg.agent_model_name}")
    print(f"User model: {cfg.user_model_name}")
    print("\nInteraction Parameters:")
    print(f"  Reward (r): {cfg.reward}")
    print(f"  Delegation cost (c): {cfg.cost}")
    print(f"  Effort (e): {cfg.effort}")
    print(f"  Discount factor (δ): {cfg.discount_factor}")
    print(f"  User delegation threshold (θ*): {cfg.compute_threshold():.4f}")
    print(f"  Confidence mode: {cfg.confidence_mode.value}")
    print(f"\nExperiment: {cfg.num_trials} trials × {cfg.num_rounds} rounds\n")

    all_trial_results: List[Dict[str, Any]] = []
    _progress_local = None
    if progress is None:
        _progress_local = tqdm(
            total=cfg.num_trials * cfg.num_rounds, desc="Running two-player game"
        )
        progress = _progress_local

    for trial_idx in range(cfg.num_trials):
        logger.debug("Starting trial %s", trial_idx)
        trial_result = run_one_trial(cfg, dataset, trial_idx, progress)
        all_trial_results.append(trial_result)

    if _progress_local is not None:
        _progress_local.close()

    valid_trials = [
        trial
        for trial in all_trial_results
        if "error" not in trial.get("statistics", {})
    ]
    logger.info(
        "Completed %s valid trials out of %s", len(valid_trials), cfg.num_trials
    )

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

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(
        output_path / f"results_{timestamp}.json", "w", encoding="utf-8"
    ) as handle:
        json.dump(results, handle, indent=2, default=str)

    summary = generate_summary_report(cfg, timestamp, overall_stats)
    with open(
        output_path / f"summary_{timestamp}.txt", "w", encoding="utf-8"
    ) as handle:
        handle.write(summary)

    print(summary)
    print(f"\nResults saved to: {output_path}")
    logger.info("Results saved to: %s", output_path)

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
    user_beliefs_agent_ability = aggregate_trial_stats(
        valid_trials, "mean_user_belief_agent_ability"
    )
    user_beliefs_honesty = aggregate_trial_stats(
        valid_trials, "mean_user_belief_honesty"
    )
    posterior_user_beliefs_agent_ability = aggregate_trial_stats(
        valid_trials, "mean_posterior_user_belief_agent_ability"
    )
    posterior_user_beliefs_honesty = aggregate_trial_stats(
        valid_trials, "mean_posterior_user_belief_honesty"
    )
    belief_errors = aggregate_trial_stats(valid_trials, "user_belief_error")

    # Per-round aggregated statistics across trials
    per_round_stats: Dict[int, Dict[str, Optional[float]]] = {}
    max_rounds = max((len(t["round_results"]) for t in valid_trials), default=0)
    for r in range(max_rounds):
        rounds = [
            t["round_results"][r] for t in valid_trials if len(t["round_results"]) > r
        ]
        if not rounds:
            continue

        def to_float_bool(v: Optional[bool]) -> Optional[float]:
            if v is None:
                return None
            return 1.0 if bool(v) else 0.0

        baseline_acc_vals = [
            to_float_bool(rnd.get("baseline_correct"))
            for rnd in rounds
            if rnd.get("baseline_correct") is not None
        ]
        baseline_conf_vals = [
            rnd.get("baseline_confidence")
            for rnd in rounds
            if rnd.get("baseline_confidence") is not None
        ]
        agent_acc_vals = [
            to_float_bool(rnd.get("agent_correct"))
            for rnd in rounds
            if rnd.get("agent_correct") is not None
        ]
        agent_conf_vals = [
            rnd.get("agent_confidence")
            for rnd in rounds
            if rnd.get("agent_confidence") is not None
        ]
        conf_diff_vals = [
            rnd.get("confidence_diff")
            for rnd in rounds
            if rnd.get("confidence_diff") is not None
        ]
        delegation_vals = [
            1.0 if rnd.get("user_decision") == "DELEGATE" else 0.0
            for rnd in rounds
            if rnd.get("user_decision") in {"DELEGATE", "SELF_SOLVE"}
        ]
        belief_vals = [
            rnd.get("user_belief_agent_correct")
            for rnd in rounds
            if rnd.get("user_belief_agent_correct") is not None
        ]
        user_payoff_vals = [
            rnd.get("user_payoff")
            for rnd in rounds
            if rnd.get("user_payoff") is not None
        ]
        agent_payoff_vals = [
            rnd.get("agent_payoff")
            for rnd in rounds
            if rnd.get("agent_payoff") is not None
        ]

        # Count confidence change categories by round
        inflated_count = sum(1 for v in conf_diff_vals if v is not None and v > 0)
        deflated_count = sum(1 for v in conf_diff_vals if v is not None and v < 0)
        unchanged_count = sum(1 for v in conf_diff_vals if v is not None and v == 0)
        per_round_stats[r] = {
            "baseline_accuracy": compute_mean(baseline_acc_vals),
            "baseline_confidence": compute_mean(baseline_conf_vals),
            "agent_accuracy": compute_mean(agent_acc_vals),
            "agent_confidence": compute_mean(agent_conf_vals),
            "confidence_diff": compute_mean(conf_diff_vals),
            "delegation_rate": compute_mean(delegation_vals),
            "user_belief_agent_correct": compute_mean(belief_vals),
            "mean_user_payoff": compute_mean(user_payoff_vals),
            "mean_agent_payoff": compute_mean(agent_payoff_vals),
            "confidence_inflated_count": float(inflated_count),
            "confidence_deflated_count": float(deflated_count),
            "confidence_unchanged_count": float(unchanged_count),
        }

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
        "mean_user_belief_agent_ability": compute_mean(user_beliefs_agent_ability),
        "mean_user_belief_honesty": compute_mean(user_beliefs_honesty),
        "mean_posterior_user_belief_agent_ability": compute_mean(
            posterior_user_beliefs_agent_ability
        ),
        "mean_posterior_user_belief_honesty": compute_mean(
            posterior_user_beliefs_honesty
        ),
        "mean_user_belief_error": compute_mean(belief_errors),
        "total_user_payoff": sum_trial_stats(valid_trials, "total_user_payoff"),
        "total_agent_payoff": sum_trial_stats(valid_trials, "total_agent_payoff"),
        "mean_user_payoff_per_round": compute_mean(
            aggregate_trial_stats(valid_trials, "mean_user_payoff")
        ),
        "mean_agent_payoff_per_round": compute_mean(
            aggregate_trial_stats(valid_trials, "mean_agent_payoff")
        ),
        "per_round_statistics": per_round_stats,
    }


def generate_summary_report(
    cfg: BaseGameConfig,
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

        # Per-round statistics section
        per_round = overall_stats.get("per_round_statistics") or {}
        if per_round:
            lines.extend(["", "Per-Round Statistics:"])
            for r in sorted(per_round.keys()):
                rs = per_round[r]
                lines.append(f"  Round {r}:")
                if (
                    rs.get("baseline_accuracy") is not None
                    and rs.get("baseline_confidence") is not None
                ):
                    lines.append(
                        f"    Baseline - accuracy: {rs['baseline_accuracy']:.4f}, confidence: {rs['baseline_confidence']:.4f}"
                    )
                elif rs.get("baseline_accuracy") is not None:
                    lines.append(
                        f"    Baseline - accuracy: {rs['baseline_accuracy']:.4f}"
                    )
                elif rs.get("baseline_confidence") is not None:
                    lines.append(
                        f"    Baseline - confidence: {rs['baseline_confidence']:.4f}"
                    )

                if (
                    rs.get("agent_accuracy") is not None
                    and rs.get("agent_confidence") is not None
                ):
                    lines.append(
                        f"    Agent - accuracy: {rs['agent_accuracy']:.4f}, reported confidence: {rs['agent_confidence']:.4f}"
                    )
                elif rs.get("agent_accuracy") is not None:
                    lines.append(f"    Agent - accuracy: {rs['agent_accuracy']:.4f}")
                elif rs.get("agent_confidence") is not None:
                    lines.append(
                        f"    Agent - reported confidence: {rs['agent_confidence']:.4f}"
                    )

                if rs.get("confidence_diff") is not None:
                    lines.append(
                        f"    Confidence diff (strategic - baseline): {rs['confidence_diff']:.4f}"
                    )
                # Per-round confidence change counts
                if rs.get("confidence_inflated_count") is not None:
                    lines.append(
                        f"    Confidence inflated rounds: {int(rs['confidence_inflated_count'])}"
                    )
                if rs.get("confidence_deflated_count") is not None:
                    lines.append(
                        f"    Confidence deflated rounds: {int(rs['confidence_deflated_count'])}"
                    )
                if rs.get("confidence_unchanged_count") is not None:
                    lines.append(
                        f"    Confidence unchanged rounds: {int(rs['confidence_unchanged_count'])}"
                    )
                if rs.get("delegation_rate") is not None:
                    lines.append(f"    Delegation rate: {rs['delegation_rate']:.4f}")
                if rs.get("user_belief_agent_correct") is not None:
                    lines.append(
                        f"    Mean user belief agent is correct: {rs['user_belief_agent_correct']:.4f}"
                    )
                if rs.get("mean_user_payoff") is not None:
                    lines.append(f"    Mean user payoff: {rs['mean_user_payoff']:.4f}")
                if rs.get("mean_agent_payoff") is not None:
                    lines.append(
                        f"    Mean agent payoff: {rs['mean_agent_payoff']:.4f}"
                    )
    else:
        lines.append(f"  Error: {overall_stats['error']}")

    lines.append("=" * 70)
    return "\n".join(lines)


def run_experiments(
    configs: List[BaseGameConfig], output_path: Path | str
) -> pd.DataFrame:
    """
    Run experiments for multiple configurations and export the results.

    Saves two files in the output_path directory:
    - results.csv: Contains all per-round rows, with constant fields removed.
    - config.json: Contains all fields that were constant across every row.

    Returns:
        pandas.DataFrame containing all per-round rows.
    """

    # Prepare rows buffer
    rows_buffer: List[Dict[str, Any]] = []

    # Create a tqdm progress bar across all configs and trials/rounds
    total_steps = sum(c.num_trials * c.num_rounds for c in configs)
    sweep_progress = tqdm(total=total_steps, desc="Config sweep progress")

    for cfg in configs:
        # Run the full experiment for this config, passing the shared progress bar
        results = run_trials(cfg, progress=sweep_progress)

        timestamp = results.get("timestamp")
        config = results.get("config", {}) or {}
        trials = results.get("trial_results", []) or []

        for trial in trials:
            trial_idx = trial.get("trial_idx")
            num_rounds_completed = trial.get("num_rounds_completed")
            round_results = trial.get("round_results", []) or []

            for rr in round_results:
                row: Dict[str, Any] = {**config}
                row["timestamp"] = timestamp
                row["trial_idx"] = trial_idx
                row["num_rounds_completed"] = num_rounds_completed
                # Include all round result fields verbatim
                row.update(rr or {})
                if "task" in row:
                    del row["task"]
                rows_buffer.append(row)

    # Close progress bar
    sweep_progress.close()

    if not rows_buffer:
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(rows_buffer)

    # Identify constant columns
    constant_config = {}
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            val = df[col].iloc[0] if not df.empty else None
            constant_config[col] = val
            constant_cols.append(col)

    # Drop constant columns from DataFrame
    df_trimmed = df.drop(columns=constant_cols)

    # Ensure output directory exists
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Write constant config to json
    with open(output_dir / f"config_{timestamp}.json", "w") as f:
        json.dump(constant_config, f, indent=4, default=str)

    # Write trimmed DataFrame to CSV
    df_trimmed.to_csv(output_dir / f"results_{timestamp}.csv", index=False)

    # Return the full DataFrame (or df_trimmed based on preference,
    # but the request implies returning the dataframe after processing)
    return df_trimmed


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = (
        Path(__file__).parent.parent.parent
        / "outputs"
        / "experiments"
        / f"sweep_two_player_{timestamp}"
    )
    deltas = [
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
    ]  # [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    h_s = [0.1, 0.3, 0.5, 0.7, 0.9]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mu_s = [0.1, 0.3, 0.5, 0.7, 0.9]  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sweep_configs = [
        BaseGameConfig(
            # model_name="together_ai/openai/gpt-oss-20b",
            num_trials=1,
            num_rounds=2,
            discount_factor=delta,
            priors=True,
            h_0=h,
            mu_0=mu,
            seed=int(time.time()),
        )
        for delta in deltas
        for h in h_s
        for mu in mu_s
    ]
    # print(len(sweep_configs))
    run_experiments(sweep_configs, output_filename)
