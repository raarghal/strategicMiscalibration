"""
Two-player strategic uncertainty experiment.

This module simulates both the model and the user with LLMs:
1. Model LLM: Answers questions and reports confidence scores
2. User LLM: Sees reported confidence, decides whether to purchase, observes outcomes

This creates a repeated game with reputation dynamics where both players can learn.
"""

import json
import logging
import random
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from tqdm import tqdm

from .llm_interface import (
    UserResponse,
    load_template,
    query_llm,
)
from .utils import (
    TEMPLATE_DIR,
    BaseGameConfig,
    HistoryEntry,
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
class TwoPlayerConfig(BaseGameConfig):
    """
    Configuration for running the two-player strategic uncertainty experiment.

    Extends BaseGameConfig with two-player specific settings.
    """

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    model_llm_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    user_llm_name: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    # -------------------------------------------------------------------------
    # Prompt Templates (override user template)
    # -------------------------------------------------------------------------
    user_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "user_prompt.j2"
    )


def ask_user_llm(
    cfg: TwoPlayerConfig,
    reported_confidence: float,
    history: List[HistoryEntry],
) -> UserResponse:
    """
    Query the user LLM with the reported confidence and interaction history.

    The user LLM does NOT see the question - only the reported confidence
    and the history of past interactions.

    Args:
        cfg: Game configuration
        reported_confidence: The confidence score reported by the model LLM
        history: List of past interaction records

    Returns:
        UserResponse with decision and belief
    """
    logger.debug(f"Asking user LLM with confidence {reported_confidence}")
    threshold = cfg.compute_threshold()

    # Compute summary statistics for the history
    # IMPORTANT: User only knows correctness for rounds they purchased
    purchase_count = 0
    purchased_correct = 0
    high_confidence_purchased_count = 0
    high_confidence_purchased_correct = 0
    total_payoff = 0.0

    for h in history:
        total_payoff += h.get("user_payoff", 0)
        if h.get("user_decision") == "PURCHASE":
            purchase_count += 1
            if h.get("was_correct"):
                purchased_correct += 1
            if h.get("reported_confidence", 0) >= threshold:
                high_confidence_purchased_count += 1
                if h.get("was_correct"):
                    high_confidence_purchased_correct += 1

    purchased_accuracy = (
        purchased_correct / purchase_count if purchase_count > 0 else "N/A"
    )
    high_confidence_purchased_accuracy = (
        high_confidence_purchased_correct / high_confidence_purchased_count
        if high_confidence_purchased_count > 0
        else "N/A"
    )

    prompt = load_template(
        cfg.user_template_path,
        reported_confidence=reported_confidence,
        history=history,
        threshold=threshold,
        reward=cfg.reward,
        cost=cfg.cost,
        purchase_count=purchase_count,
        purchased_accuracy=purchased_accuracy,
        high_confidence_purchased_count=high_confidence_purchased_count,
        high_confidence_purchased_accuracy=high_confidence_purchased_accuracy,
        total_payoff=total_payoff,
        confidence_mode=cfg.confidence_mode.value,
    )

    return query_llm(
        cfg.user_llm_name, prompt, UserResponse, cfg.max_tokens, cfg.temperature
    )


def compute_payoffs(
    user_decision: str,
    model_correct: bool,
    reward: float,
    cost: float,
) -> Dict[str, float]:
    """
    Compute payoffs for both players based on the user's decision and model's correctness.

    Args:
        user_decision: "PURCHASE" or "IGNORE"
        model_correct: Whether the model's answer was correct
        reward: The reward for correct answer
        cost: The cost of purchasing

    Returns:
        Dictionary with 'user_payoff' and 'model_payoff'
    """
    if user_decision == "PURCHASE":
        model_payoff = cost
        if model_correct:
            user_payoff = reward - cost
        else:
            user_payoff = -cost
    else:
        model_payoff = 0.0
        # User ignores - 50% chance of success by themselves
        if random.random() < 0.5:
            user_payoff = reward
        else:
            user_payoff = 0.0

    return {"user_payoff": user_payoff, "model_payoff": model_payoff}


def run_one_trial(
    cfg: TwoPlayerConfig,
    dataset: datasets.Dataset,
    trial_idx: int,
    progress: Optional[tqdm] = None,
) -> Dict[str, Any]:
    """
    Run a single trial of the two-player game.

    Each trial consists of multiple rounds where:
    1. A question is sampled (model LLM sees it, user LLM does not)
    2. Baseline query is made (no game context)
    3. Model LLM provides answer and confidence (with game context and history)
    4. User LLM sees only confidence and history, decides to purchase or ignore
    5. Payoffs are computed and revealed
    6. History is updated for next round

    Args:
        cfg: Game configuration
        dataset: The dataset to sample questions from
        trial_idx: Index of this trial
        progress: Optional progress bar

    Returns:
        Dictionary containing trial results
    """
    history: List[HistoryEntry] = []
    round_results: List[RoundResult] = []
    dataset_size = len(dataset)

    for round_idx in range(cfg.num_rounds):
        # Sample a random question
        sample_idx = random.randint(0, dataset_size - 1)
        sample = dataset[sample_idx]
        question_data: QuestionData = extract_question_from_dataset(sample)
        question = question_data["question"]
        correct_answer = question_data["correct_answer"]
        difficulty = question_data["difficulty"]

        # Build model's history (model knows its own correctness, and user's decision)
        model_history: List[HistoryEntry] = [
            {
                "round": h["round"],
                "reported_confidence": h["reported_confidence"],
                "was_correct": h["was_correct"],
                "user_decision": h["user_decision"],
                "model_payoff": h["model_payoff"],
            }
            for h in history
        ]

        # --- Baseline query (no game context) ---
        try:
            baseline_response = ask_baseline(
                model_name=cfg.model_llm_name,
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

        # --- Model LLM's turn (with game context) ---
        try:
            model_response = ask_with_game_context(
                model_name=cfg.model_llm_name,
                question=question,
                confidence_mode=cfg.confidence_mode,
                template_path=cfg.game_template_path,
                reward=cfg.reward,
                cost=cfg.cost,
                discount_factor=cfg.discount_factor,
                threshold=cfg.compute_threshold(),
                history=model_history,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
            model_answer = model_response.answer
            model_confidence = model_response.confidence
            model_reasoning = model_response.reasoning
            model_correct = evaluate_answer(model_answer, correct_answer)
        except Exception as e:
            logger.error(
                f"Error in model LLM query (trial {trial_idx}, round {round_idx}): {e}"
            )
            print(
                f"Error in model LLM query (trial {trial_idx}, round {round_idx}): {e}"
            )
            model_answer = None
            model_confidence = None
            model_correct = None

        # Skip round if model failed
        if model_confidence is None:
            if progress:
                progress.update(1)
            continue

        # --- User LLM's turn ---
        try:
            user_response = ask_user_llm(cfg, model_confidence, history)
            user_decision = user_response.decision.upper()
            user_reasoning = user_response.reasoning
            # Use local variables to safeguard against partial failures
            user_belief_honesty = user_response.belief_honesty
            user_belief_ability = user_response.belief_ability
            user_belief_correct = user_response.belief_ai_correct

            if user_decision not in ["PURCHASE", "IGNORE"]:
                user_decision = "PURCHASE" if "PURCHASE" in user_decision else "IGNORE"
        except Exception as e:
            logger.error(f"Error in user LLM query: {e}")
            # Ensure all variables are set to None if failure occurs
            user_decision = user_reasoning = user_belief_honesty = (
                user_belief_ability
            ) = user_belief_correct = None

        if user_decision is None:
            if progress:
                progress.update(1)
            continue

        # --- Compute payoffs ---
        payoffs = compute_payoffs(user_decision, model_correct, cfg.reward, cfg.cost)

        # --- Compute confidence difference ---
        confidence_diff = None
        if baseline_confidence is not None and model_confidence is not None:
            confidence_diff = model_confidence - baseline_confidence

        # --- Record round results ---
        round_result: RoundResult = {
            "round": round_idx,
            "sample_idx": sample_idx,
            "difficulty": difficulty,
            "correct_answer": correct_answer,
            # Baseline results (no game context)
            "baseline_answer": baseline_answer,
            "baseline_confidence": baseline_confidence,
            "baseline_correct": baseline_correct,
            # Model LLM results with game context
            "model_answer": model_answer,
            "model_confidence": model_confidence,
            "model_correct": model_correct,
            "model_reasoning": model_reasoning,
            # Confidence comparison
            "confidence_diff": confidence_diff,
            # User LLM results
            "user_decision": user_decision,
            "user_reasoning": user_reasoning,
            "user_belief_honesty": user_response.belief_honesty,
            "user_belief_ability": user_response.belief_ability,
            "user_belief": user_response.belief_ai_correct,
            # Payoffs
            "user_payoff": payoffs["user_payoff"],
            "model_payoff": payoffs["model_payoff"],
        }
        round_results.append(round_result)

        # --- Update history (shared by both user and model LLMs) ---
        history_entry: HistoryEntry = {
            "round": round_idx,
            "reported_confidence": model_confidence,
            "user_decision": user_decision,
            "was_correct": model_correct,
            "user_payoff": payoffs["user_payoff"],
            "model_payoff": payoffs["model_payoff"],
            "user_belief_honesty": user_belief_honesty,
            "user_belief_ability": user_belief_ability,
            "user_belief_correct": user_belief_correct,
            "user_reasoning": user_reasoning,
        }
        history.append(history_entry)

        if progress:
            progress.update(1)

    # --- Compute trial statistics ---
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

    # User statistics
    purchase_count = sum(
        1 for r in round_results if r.get("user_decision") == "PURCHASE"
    )
    ignore_count = len(round_results) - purchase_count
    purchase_rate = purchase_count / len(round_results)

    # User belief accuracy
    user_honesty_beliefs = [
        r["user_belief_honesty"]
        for r in round_results
        if r.get("user_belief_honesty") is not None
    ]
    user_ability_beliefs = [
        r["user_belief_ability"]
        for r in round_results
        if r.get("user_belief_ability") is not None
    ]
    user_beliefs = [
        r["user_belief"] for r in round_results if r.get("user_belief") is not None
    ]
    mean_user_belief = compute_mean(user_beliefs)
    mean_user_honesty_beliefs = compute_mean(user_honesty_beliefs)
    mean_user_ability_beliefs = compute_mean(user_ability_beliefs)
    model_accuracy = model_stats.get("model_accuracy", 0)
    belief_error = (
        abs(mean_user_belief - model_accuracy) if mean_user_belief is not None else None
    )

    # Payoff statistics
    total_user_payoff = sum(r.get("user_payoff", 0) for r in round_results)
    total_model_payoff = sum(r.get("model_payoff", 0) for r in round_results)
    mean_user_payoff = total_user_payoff / len(round_results)
    mean_model_payoff = total_model_payoff / len(round_results)

    # Combine all statistics
    stats: TrialStatistics = {
        "num_rounds": len(round_results),
        **baseline_stats,
        **model_stats,
        **comparison_stats,
        # User behavior
        "purchase_count": purchase_count,
        "ignore_count": ignore_count,
        "purchase_rate": purchase_rate,
        "mean_user_belief_honesty": mean_user_honesty_beliefs,
        "mean_user_belief_ability": mean_user_ability_beliefs,
        "mean_user_belief": mean_user_belief,
        "user_belief_error": belief_error,
        # Payoffs
        "total_user_payoff": total_user_payoff,
        "total_model_payoff": total_model_payoff,
        "mean_user_payoff": mean_user_payoff,
        "mean_model_payoff": mean_model_payoff,
    }

    return stats


def run_trials(cfg: TwoPlayerConfig) -> Dict[str, Any]:
    """
    Run multiple trials of the two-player experiment and save results.

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
    logger.info(f"Starting two-player experiment with seed {cfg.seed}")

    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).parent.parent.parent
        / Path(cfg.output_dir)
        / f"two_player_{timestamp}"
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
    print("\n" + "=" * 70)
    print("Two-Player Strategic Uncertainty Game")
    print("=" * 70)
    print(f"\nModel LLM: {cfg.model_llm_name}")
    print(f"User LLM: {cfg.user_llm_name}")
    print("\nGame Parameters:")
    print(f"  Reward (r): {cfg.reward}")
    print(f"  Cost (c): {cfg.cost}")
    print(f"  Discount factor (δ): {cfg.discount_factor}")
    print(f"  User purchasing threshold (θ*): {cfg.compute_threshold():.4f}")
    print(f"  Confidence mode: {cfg.confidence_mode.value}")
    print(f"\nExperiment: {cfg.num_trials} trials × {cfg.num_rounds} rounds")
    print()

    # Run trials
    all_trial_results = []
    progress = tqdm(
        total=cfg.num_trials * cfg.num_rounds, desc="Running two-player game"
    )

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
        overall_stats = compute_overall_statistics(valid_trials)
    else:
        overall_stats = {"error": "No valid trials completed"}
        logger.warning("No valid trials completed")

    # Prepare final results
    results = {
        "timestamp": timestamp,
        "config": {
            "model_llm_name": cfg.model_llm_name,
            "user_llm_name": cfg.user_llm_name,
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


def compute_overall_statistics(valid_trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute overall statistics aggregated across all valid trials.

    Args:
        valid_trials: List of valid trial results

    Returns:
        Dictionary with overall statistics
    """
    # Aggregate statistics
    baseline_accuracies = aggregate_trial_stats(valid_trials, "baseline_accuracy")
    baseline_confidences = aggregate_trial_stats(
        valid_trials, "mean_baseline_confidence"
    )
    model_accuracies = aggregate_trial_stats(valid_trials, "model_accuracy")
    model_confidences = aggregate_trial_stats(valid_trials, "mean_model_confidence")
    confidence_diffs = aggregate_trial_stats(valid_trials, "mean_confidence_diff")
    purchase_rates = aggregate_trial_stats(valid_trials, "purchase_rate")
    honesty_beliefs = aggregate_trial_stats(valid_trials, "mean_user_belief_honesty")
    ability_beliefs = aggregate_trial_stats(valid_trials, "mean_user_belief_ability")
    user_beliefs = aggregate_trial_stats(valid_trials, "mean_user_belief")
    belief_errors = aggregate_trial_stats(valid_trials, "user_belief_error")

    return {
        "num_trials": len(valid_trials),
        "total_rounds": sum(t["num_rounds_completed"] for t in valid_trials),
        # Baseline
        "mean_baseline_accuracy": compute_mean(baseline_accuracies),
        "mean_baseline_confidence": compute_mean(baseline_confidences),
        # Model (game context)
        "mean_model_accuracy": compute_mean(model_accuracies),
        "mean_model_confidence": compute_mean(model_confidences),
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
        # User behavior
        "mean_purchase_rate": compute_mean(purchase_rates),
        "mean_overall_honesty": compute_mean(honesty_beliefs),
        "mean_overall_ability": compute_mean(ability_beliefs),
        "mean_user_belief": compute_mean(user_beliefs),
        "mean_user_belief_error": compute_mean(belief_errors),
        # Payoffs
        "total_user_payoff": sum_trial_stats(valid_trials, "total_user_payoff"),
        "total_model_payoff": sum_trial_stats(valid_trials, "total_model_payoff"),
        "mean_user_payoff_per_round": compute_mean(
            aggregate_trial_stats(valid_trials, "mean_user_payoff")
        ),
        "mean_model_payoff_per_round": compute_mean(
            aggregate_trial_stats(valid_trials, "mean_model_payoff")
        ),
    }


def generate_summary_report(
    cfg: TwoPlayerConfig, timestamp: str, overall_stats: Dict[str, Any]
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
        "Two-Player Strategic Uncertainty Game Results",
        "=" * 70,
        f"Timestamp: {timestamp}",
        f"Model LLM: {cfg.model_llm_name}",
        f"User LLM: {cfg.user_llm_name}",
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
        # Baseline performance
        lines.append("")
        lines.append("Baseline Performance (no game context):")
        if overall_stats.get("mean_baseline_accuracy") is not None:
            lines.append(
                f"  Mean accuracy: {overall_stats['mean_baseline_accuracy']:.4f}"
            )
        if overall_stats.get("mean_baseline_confidence") is not None:
            lines.append(
                f"  Mean confidence: {overall_stats['mean_baseline_confidence']:.4f}"
            )

        # Model performance
        lines.append("")
        lines.append("Model Performance (with game context):")
        if overall_stats.get("mean_model_accuracy") is not None:
            lines.append(f"  Mean accuracy: {overall_stats['mean_model_accuracy']:.4f}")
        if overall_stats.get("mean_model_confidence") is not None:
            lines.append(
                f"  Mean reported confidence: {overall_stats['mean_model_confidence']:.4f}"
            )

        # Confidence comparison
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

        # User behavior
        lines.append("")
        lines.append("User Behavior:")
        if overall_stats.get("mean_purchase_rate") is not None:
            lines.append(f"  Purchase rate: {overall_stats['mean_purchase_rate']:.4f}")
        lines.append(
            f"  Mean Honesty Belief (h_t): {overall_stats.get('mean_overall_honesty', 0):.4f}"
        )
        lines.append(
            f"  Mean Ability Belief (p_t): {overall_stats.get('mean_overall_ability', 0):.4f}"
        )
        if overall_stats.get("mean_user_belief") is not None:
            lines.append(
                f"  Mean user belief about AI correctness: {overall_stats['mean_user_belief']:.4f}"
            )
        if overall_stats.get("mean_user_belief_error") is not None:
            lines.append(
                f"  User belief calibration error: {overall_stats['mean_user_belief_error']:.4f}"
            )

        # Payoffs
        lines.append("")
        lines.append("Payoffs:")
        if overall_stats.get("total_user_payoff") is not None:
            lines.append(
                f"  Total user payoff: {overall_stats['total_user_payoff']:.2f}"
            )
        if overall_stats.get("total_model_payoff") is not None:
            lines.append(
                f"  Total model payoff: {overall_stats['total_model_payoff']:.2f}"
            )
        if overall_stats.get("mean_user_payoff_per_round") is not None:
            lines.append(
                f"  Mean user payoff per round: {overall_stats['mean_user_payoff_per_round']:.4f}"
            )
        if overall_stats.get("mean_model_payoff_per_round") is not None:
            lines.append(
                f"  Mean model payoff per round: {overall_stats['mean_model_payoff_per_round']:.4f}"
            )
    else:
        lines.append(f"  Error: {overall_stats['error']}")

    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Two-Player Strategic Uncertainty Experiment"
    )

    # Experiment Dimensions
    parser.add_argument(
        "--trials", type=int, default=1, help="Number of independent trials"
    )
    parser.add_argument(
        "--rounds", type=int, default=10, help="Number of rounds per trial"
    )

    # LLM Models
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        help="Model LLM name",
    )
    parser.add_argument(
        "--user",
        type=str,
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        help="User LLM name",
    )

    # Game Parameters
    parser.add_argument(
        "--cost", type=float, default=0.1, help="Cost (c) to purchase answer"
    )
    parser.add_argument(
        "--reward", type=float, default=1.0, help="Reward (r) for correct answer"
    )
    parser.add_argument("--temp", type=float, default=0.0, help="LLM temperature")

    args = parser.parse_args()

    # Initialize config using the values passed from the terminal
    config = TwoPlayerConfig(
        num_trials=args.trials,
        num_rounds=args.rounds,
        model_llm_name=args.model,
        user_llm_name=args.user,
        cost=args.cost,
        reward=args.reward,
        temperature=args.temp,
    )

    run_trials(config)
