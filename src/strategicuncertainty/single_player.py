"""
Single-player strategic delegation experiment.

This module compares agent confidence scores in two settings:
1. Baseline: the agent solves tasks without any strategic context.
2. Game: the agent knows a user may delegate the task (at cost c) based on the
   reported confidence score.

We measure whether the strategic framing changes the agent's reported confidence.
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import datasets
from tqdm import tqdm

from .datatypes import (
    BaseGameConfig,
    RoundResult,
    TaskData,
    TrialStatistics,
)
from .utils import (
    aggregate_trial_stats,
    build_round_result,
    compute_agent_stats,
    compute_baseline_stats,
    compute_confidence_comparison_stats,
    compute_confidence_diff,
    compute_mean,
    extract_task_from_dataset,
    query_and_sanitize_agent_game_response,
    query_and_sanitize_baseline_response,
    sum_trial_stats,
)

logger = logging.getLogger(__name__)


def _sample_round_context(
    dataset: datasets.Dataset, round_idx: int, dataset_size: int
) -> tuple[int, str, str, str | None]:
    sample_idx = random.randint(0, dataset_size - 1)
    sample = dataset[sample_idx]
    task_data: TaskData = extract_task_from_dataset(sample)
    return (
        sample_idx,
        task_data["task"],
        task_data["correct_solution"],
        task_data["difficulty"],
    )


def _run_single_round(
    cfg: BaseGameConfig,
    round_idx: int,
    sample_idx: int,
    task: str,
    correct_solution: str,
    difficulty: str | None,
) -> RoundResult:
    baseline = query_and_sanitize_baseline_response(cfg, task, correct_solution)
    agent = query_and_sanitize_agent_game_response(
        cfg, task, correct_solution, history=None
    )
    confidence_diff = compute_confidence_diff(baseline.confidence, agent.confidence)

    return build_round_result(
        round_idx=round_idx,
        sample_idx=sample_idx,
        task=task,
        difficulty=difficulty,
        correct_solution=correct_solution,
        baseline_solution=baseline.solution,
        baseline_confidence=baseline.confidence,
        baseline_correct=baseline.correct,
        agent_solution=agent.solution,
        agent_confidence=agent.confidence,
        agent_correct=agent.correct,
        confidence_diff=confidence_diff,
    )


def run_one_trial(
    cfg: BaseGameConfig,
    dataset: datasets.Dataset,
    trial_idx: int,
    progress: tqdm,
) -> Dict[str, Any]:
    """
    Run a single trial of the experiment.

    For each round:
    1. Sample a task from the dataset
    2. Solve the task without strategic context (baseline)
    3. Solve the same task with strategic context
    4. Record both responses and compare confidence scores

    Note: This single-player version does not simulate actual game dynamics.
    It only measures whether the strategic framing affects confidence reporting.

    Args:
        cfg: Game configuration
        dataset: The dataset to sample tasks from
        trial_idx: Index of this trial
        progress: Progress bar

    Returns:
        Dictionary containing trial results
    """
    round_results: List[RoundResult] = []
    dataset_size = len(dataset)

    for round_idx in range(cfg.num_rounds):
        sample_idx, task, correct_solution, difficulty = _sample_round_context(
            dataset, round_idx, dataset_size
        )
        round_result = _run_single_round(
            cfg,
            round_idx,
            sample_idx,
            task,
            correct_solution,
            difficulty,
        )
        round_results.append(round_result)
        progress.update(1)

    # Compute trial statistics
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
    """
    Compute summary statistics for a single trial.

    Args:
        round_results: List of round results from run_one_trial
        cfg: Game configuration

    Returns:
        Dictionary with summary statistics
    """
    if not round_results:
        return {"error": "No valid rounds in trial"}

    threshold = cfg.compute_threshold()

    # Get component statistics
    baseline_stats = compute_baseline_stats(round_results)
    agent_stats = compute_agent_stats(round_results, threshold)
    comparison_stats = compute_confidence_comparison_stats(round_results)

    # Combine all statistics
    stats: TrialStatistics = {
        "num_rounds": len(round_results),
        **baseline_stats,
        **agent_stats,
        **comparison_stats,
    }

    return stats


def run_trials(cfg: BaseGameConfig) -> Dict[str, Any]:
    """
    Run multiple trials of the experiment and save results.

    This function:
    1. Loads the dataset
    2. Runs multiple trials, each consisting of multiple rounds
    3. Computes statistics for each trial and overall
    4. Saves results and metadata to the output directory

    Args:
        cfg: Game configuration

    Returns:
        Dictionary with all results and statistics
    """
    # Set random seed for reproducibility
    random.seed(cfg.seed)
    logger.info("Starting single-player experiment with seed %s", cfg.seed)

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent.parent.parent
        / Path(cfg.output_dir)
        / f"single_player_{timestamp}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_path)

    # Load dataset
    logger.info("Loading dataset: %s", cfg.dataset_name)
    dataset = datasets.load_dataset(cfg.dataset_name, split="test")
    logger.info("Dataset loaded with %s samples", len(dataset))

    # Print game parameters
    print("\nInteraction Parameters:")
    print(f"  Reward (r): {cfg.reward}")
    print(f"  Delegation cost (c): {cfg.cost}")
    print(f"  Effort (e): {cfg.effort}")
    print(f"  Discount factor (δ): {cfg.discount_factor}")
    print(f"  User delegation threshold (θ*): {cfg.compute_threshold():.4f}")
    print(f"  Confidence mode: {cfg.confidence_mode.value}")
    print()

    # Run trials
    all_trial_results = []
    progress = tqdm(total=cfg.num_trials * cfg.num_rounds, desc="Running trials")

    for trial_idx in range(cfg.num_trials):
        logger.debug("Starting trial %s", trial_idx)
        trial_result = run_one_trial(cfg, dataset, trial_idx, progress)
        all_trial_results.append(trial_result)
        # time.sleep(2)

    progress.close()

    # Compute overall statistics
    valid_trials = [
        t for t in all_trial_results if "error" not in t.get("statistics", {})
    ]
    logger.info(
        "Completed %s valid trials out of %s", len(valid_trials), cfg.num_trials
    )

    if valid_trials:
        # Aggregate statistics across trials
        baseline_accuracies = aggregate_trial_stats(valid_trials, "baseline_accuracy")
        baseline_confidences = aggregate_trial_stats(
            valid_trials, "mean_baseline_confidence"
        )
        agent_accuracies = aggregate_trial_stats(valid_trials, "agent_accuracy")
        agent_confidences = aggregate_trial_stats(valid_trials, "mean_agent_confidence")
        confidence_diffs = aggregate_trial_stats(valid_trials, "mean_confidence_diff")

        overall_stats = {
            "num_trials": len(valid_trials),
            "total_rounds": sum(t["num_rounds_completed"] for t in valid_trials),
            # Baseline
            "mean_baseline_confidence": compute_mean(baseline_confidences),
            "mean_baseline_accuracy": compute_mean(baseline_accuracies),
            # Model (game context)
            "mean_agent_confidence": compute_mean(agent_confidences),
            "mean_agent_accuracy": compute_mean(agent_accuracies),
            # Confidence comparison
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
        }
    else:
        overall_stats = {"error": "No valid trials completed"}
        logger.warning("No valid trials completed")

    # Prepare final results
    results = {
        "timestamp": timestamp,
        "config": {
            "model_name": cfg.agent_model_name,
            "dataset_name": cfg.dataset_name,
            "num_trials": cfg.num_trials,
            "num_rounds": cfg.num_rounds,
            "temperature": cfg.temperature,
            "confidence_mode": cfg.confidence_mode.value,
            "reward": cfg.reward,
            "cost": cfg.cost,
            "discount_factor": cfg.discount_factor,
            "effort": cfg.effort,
            "threshold": cfg.compute_threshold(),
            "seed": cfg.seed,
        },
        "overall_statistics": overall_stats,
        "trial_results": all_trial_results,
    }

    # Save results
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save summary report
    summary = generate_summary_report(cfg, timestamp, overall_stats)
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)

    print(summary)
    print(f"\nResults saved to: {output_path}")
    logger.info("Results saved to: %s", output_path)

    return results


def generate_summary_report(
    cfg: BaseGameConfig, timestamp: str, overall_stats: Dict[str, Any]
) -> str:
    """
    Generate a human-readable summary report.

    Args:
        cfg: Game configuration
        timestamp: Experiment timestamp
        overall_stats: Overall statistics dictionary

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "Single-Player Strategic Delegation Experiment Results",
        "=" * 70,
        f"Timestamp: {timestamp}",
        f"Agent model: {cfg.agent_model_name}",
        f"Dataset: {cfg.dataset_name}",
        f"Trials: {cfg.num_trials}, Rounds per trial: {cfg.num_rounds}",
        "-" * 70,
        "Interaction Parameters:",
        f"  Reward (r): {cfg.reward}",
        f"  Delegation cost (c): {cfg.cost}",
        f"  Effort (e): {cfg.effort}",
        f"  Discount factor (δ): {cfg.discount_factor}",
        f"  User delegation threshold (θ*): {cfg.compute_threshold():.4f}",
        f"  Confidence mode: {cfg.confidence_mode.value}",
        "-" * 70,
        "Overall Statistics:",
    ]

    if "error" not in overall_stats:
        lines.append("")
        lines.append("Baseline Performance (no game context):")
        if overall_stats.get("mean_baseline_confidence") is not None:
            lines.append(
                f"  Mean confidence: {overall_stats['mean_baseline_confidence']:.4f}"
            )
        if overall_stats.get("mean_baseline_accuracy") is not None:
            lines.append(
                f"  Mean accuracy: {overall_stats['mean_baseline_accuracy']:.4f}"
            )

        lines.append("")
        lines.append("Agent Performance (strategic context):")
        if overall_stats.get("mean_agent_confidence") is not None:
            lines.append(
                f"  Mean confidence: {overall_stats['mean_agent_confidence']:.4f}"
            )
        if overall_stats.get("mean_agent_accuracy") is not None:
            lines.append(f"  Mean accuracy: {overall_stats['mean_agent_accuracy']:.4f}")

        lines.append("")
        lines.append("Confidence Comparison (game - baseline):")
        if overall_stats.get("mean_confidence_diff") is not None:
            lines.append(
                f"  Mean difference: {overall_stats['mean_confidence_diff']:.4f}"
            )
        lines.append(
            f"  Inflated (game > baseline): {overall_stats['total_confidence_inflated']} times"
        )
        lines.append(
            f"  Deflated (game < baseline): {overall_stats['total_confidence_deflated']} times"
        )
        lines.append(
            f"  Unchanged: {overall_stats['total_confidence_unchanged']} times"
        )
    else:
        lines.append(f"  Error: {overall_stats['error']}")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    # Default configuration for testing
    config = BaseGameConfig(
        num_trials=5,
        num_rounds=2,
    )

    results = run_trials(config)
