"""
Single-player strategic uncertainty experiment.

This module compares LLM confidence scores in two settings:
1. Baseline: LLM answers questions without any strategic context
2. Game: LLM is informed about a user who decides whether to use its answer
   based on the reported confidence score

We are interested in seeing whether the LLM changes its confidence score
reports when given the game context.
"""

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import datasets
from tqdm import tqdm

from src.strategicuncertainty.utils import (
    BaseGameConfig,
    QuestionData,
    RoundResult,
    TrialStatistics,
    aggregate_trial_stats,
    ask_baseline,
    ask_with_game_context,
    compute_baseline_stats,
    compute_confidence_comparison_stats,
    compute_mean,
    compute_model_stats,
    evaluate_answer,
    extract_question_from_dataset,
    sum_trial_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class SinglePlayerConfig(BaseGameConfig):
    """
    Configuration for running the single-player strategic uncertainty experiment.

    Extends BaseGameConfig with single-player specific settings.
    """

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"


def run_one_trial(
    cfg: SinglePlayerConfig,
    dataset: datasets.Dataset,
    trial_idx: int,
    progress: tqdm,
) -> Dict[str, Any]:
    """
    Run a single trial of the experiment.

    For each round:
    1. Sample a question from the dataset
    2. Ask the question in baseline mode (no game context)
    3. Ask the same question in game mode (with strategic context)
    4. Record both responses and compare confidence scores

    Note: This single-player version does not simulate actual game dynamics.
    It only measures whether the game framing affects confidence reporting.

    Args:
        cfg: Game configuration
        dataset: The dataset to sample questions from
        trial_idx: Index of this trial
        progress: Progress bar

    Returns:
        Dictionary containing trial results
    """
    round_results: List[RoundResult] = []
    dataset_size = len(dataset)

    for round_idx in range(cfg.num_rounds):
        # Sample a random question from the dataset
        sample_idx = random.randint(0, dataset_size - 1)
        sample = dataset[sample_idx]

        # Extract question, correct answer, and difficulty
        question_data: QuestionData = extract_question_from_dataset(sample)
        question = question_data["question"]
        correct_answer = question_data["correct_answer"]
        difficulty = question_data["difficulty"]

        # Get baseline response (no game context)
        try:
            baseline_response = ask_baseline(
                model_name=cfg.model_name,
                question=question,
                confidence_mode=cfg.confidence_mode,
                template_path=cfg.baseline_template_path,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            baseline_answer = baseline_response.answer
            baseline_confidence = baseline_response.confidence
            baseline_correct = evaluate_answer(baseline_answer, correct_answer)
        except Exception as e:
            logger.error(
                f"Error in baseline query (trial {trial_idx}, round {round_idx}): {e}"
            )
            print(
                f"Error in baseline query (trial {trial_idx}, round {round_idx}): {e}"
            )
            baseline_answer = None
            baseline_confidence = None
            baseline_correct = None

        # Get game context response (no history in single-player mode)
        try:
            game_response = ask_with_game_context(
                model_name=cfg.model_name,
                question=question,
                confidence_mode=cfg.confidence_mode,
                template_path=cfg.game_template_path,
                reward=cfg.reward,
                cost=cfg.cost,
                discount_factor=cfg.discount_factor,
                threshold=cfg.compute_threshold(),
                history=None,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            game_answer = game_response.answer
            game_confidence = game_response.confidence
            game_correct = evaluate_answer(game_answer, correct_answer)
        except Exception as e:
            logger.error(
                f"Error in game query (trial {trial_idx}, round {round_idx}): {e}"
            )
            print(f"Error in game query (trial {trial_idx}, round {round_idx}): {e}")
            game_answer = None
            game_confidence = None
            game_correct = None

        # Calculate confidence difference (key metric for strategic behavior)
        confidence_diff = None
        if baseline_confidence is not None and game_confidence is not None:
            confidence_diff = game_confidence - baseline_confidence

        # Record the results
        round_result: RoundResult = {
            "round": round_idx,
            "sample_idx": sample_idx,
            "question": question,
            "correct_answer": correct_answer,
            "difficulty": difficulty,
            # Baseline results
            "baseline_answer": baseline_answer,
            "baseline_confidence": baseline_confidence,
            "baseline_correct": baseline_correct,
            # Game context results
            "model_answer": game_answer,
            "model_confidence": game_confidence,
            "model_correct": game_correct,
            # Comparison metrics
            "confidence_diff": confidence_diff,
        }
        round_results.append(round_result)

        progress.update(1)
        progress.refresh()

    # Compute trial statistics
    trial_stats = compute_trial_statistics(round_results, cfg)

    return {
        "trial_idx": trial_idx,
        "num_rounds_completed": len(round_results),
        "round_results": round_results,
        "statistics": trial_stats,
    }


def compute_trial_statistics(
    round_results: List[RoundResult], cfg: SinglePlayerConfig
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
    model_stats = compute_model_stats(round_results, threshold)
    comparison_stats = compute_confidence_comparison_stats(round_results)

    # Combine all statistics
    stats: TrialStatistics = {
        "num_rounds": len(round_results),
        **baseline_stats,
        **model_stats,
        **comparison_stats,
    }

    return stats


def run_trials(cfg: SinglePlayerConfig) -> Dict[str, Any]:
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
    logger.info(f"Starting single-player experiment with seed {cfg.seed}")

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent.parent.parent
        / Path(cfg.output_dir)
        / f"single_player_{timestamp}"
    )
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")

    # Load dataset
    print(f"Loading dataset: {cfg.dataset_name}")
    logger.info(f"Loading dataset: {cfg.dataset_name}")
    dataset = datasets.load_dataset(cfg.dataset_name, split="test")
    print(f"Dataset loaded with {len(dataset)} samples")
    logger.info(f"Dataset loaded with {len(dataset)} samples")

    # Print game parameters
    print("\nGame Parameters:")
    print(f"  Reward (r): {cfg.reward}")
    print(f"  Cost (c): {cfg.cost}")
    print(f"  Discount factor (δ): {cfg.discount_factor}")
    print(f"  User purchasing threshold (θ*): {cfg.compute_threshold():.4f}")
    print(f"  Confidence mode: {cfg.confidence_mode.value}")
    print()

    # Run trials
    all_trial_results = []
    progress = tqdm(total=cfg.num_trials * cfg.num_rounds, desc="Running trials")

    for trial_idx in range(cfg.num_trials):
        logger.debug(f"Starting trial {trial_idx}")
        trial_result = run_one_trial(cfg, dataset, trial_idx, progress)
        all_trial_results.append(trial_result)

    progress.close()

    # Compute overall statistics
    valid_trials = [
        t for t in all_trial_results if "error" not in t.get("statistics", {})
    ]
    logger.info(f"Completed {len(valid_trials)} valid trials out of {cfg.num_trials}")

    if valid_trials:
        # Aggregate statistics across trials
        baseline_accuracies = aggregate_trial_stats(valid_trials, "baseline_accuracy")
        baseline_confidences = aggregate_trial_stats(
            valid_trials, "mean_baseline_confidence"
        )
        model_accuracies = aggregate_trial_stats(valid_trials, "model_accuracy")
        model_confidences = aggregate_trial_stats(valid_trials, "mean_model_confidence")
        confidence_diffs = aggregate_trial_stats(valid_trials, "mean_confidence_diff")

        overall_stats = {
            "num_trials": len(valid_trials),
            "total_rounds": sum(t["num_rounds_completed"] for t in valid_trials),
            # Baseline
            "mean_baseline_confidence": compute_mean(baseline_confidences),
            "mean_baseline_accuracy": compute_mean(baseline_accuracies),
            # Model (game context)
            "mean_model_confidence": compute_mean(model_confidences),
            "mean_model_accuracy": compute_mean(model_accuracies),
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
            "model_name": cfg.model_name,
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

    # Save results
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate and save summary report
    summary = generate_summary_report(cfg, timestamp, overall_stats)
    with open(output_path / "summary.txt", "w") as f:
        f.write(summary)

    print(summary)
    print(f"\nResults saved to: {output_path}")
    logger.info(f"Results saved to: {output_path}")

    return results


def generate_summary_report(
    cfg: SinglePlayerConfig, timestamp: str, overall_stats: Dict[str, Any]
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
        "Single-Player Strategic Uncertainty Experiment Results",
        "=" * 70,
        f"Timestamp: {timestamp}",
        f"Model: {cfg.model_name}",
        f"Dataset: {cfg.dataset_name}",
        f"Trials: {cfg.num_trials}, Rounds per trial: {cfg.num_rounds}",
        "-" * 70,
        "Game Parameters:",
        f"  Reward (r): {cfg.reward}",
        f"  Cost (c): {cfg.cost}",
        f"  Discount factor (δ): {cfg.discount_factor}",
        f"  User purchasing threshold (θ*): {cfg.compute_threshold():.4f}",
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
        lines.append("Model Performance (with game context):")
        if overall_stats.get("mean_model_confidence") is not None:
            lines.append(
                f"  Mean confidence: {overall_stats['mean_model_confidence']:.4f}"
            )
        if overall_stats.get("mean_model_accuracy") is not None:
            lines.append(f"  Mean accuracy: {overall_stats['mean_model_accuracy']:.4f}")

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
    # Default configuration for testing
    config = SinglePlayerConfig(
        model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        num_trials=1,
        num_rounds=5,
    )

    results = run_trials(config)
