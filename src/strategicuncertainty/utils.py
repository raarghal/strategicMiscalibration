"""
Common utilities, types, and base configuration for the strategic uncertainty experiments.

This module contains shared functionality used by both singlePlayer.py and twoPlayer.py.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from .llm_interface import (
    ConfidenceMode,
    ModelResponse,
    load_template,
    query_llm,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"


# =============================================================================
# Type Definitions
# =============================================================================


class QuestionData(TypedDict):
    """Extracted question data from a dataset sample."""

    question: str
    correct_answer: str
    difficulty: Optional[str]


class RoundResult(TypedDict, total=False):
    """Result from a single round of the game."""

    round: int
    sample_idx: int
    difficulty: Optional[str]
    correct_answer: str
    # Baseline results
    baseline_answer: Optional[str]
    baseline_confidence: Optional[float]
    baseline_correct: Optional[bool]
    # Game context results
    model_answer: Optional[str]
    model_confidence: Optional[float]
    model_correct: Optional[bool]
    # Confidence comparison
    confidence_diff: Optional[float]
    # User results (twoPlayer only)
    user_decision: Optional[str]
    user_reasoning: Optional[str]
    user_belief: Optional[float]
    user_payoff: Optional[float]
    model_payoff: Optional[float]


class HistoryEntry(TypedDict, total=False):
    """A single entry in the interaction history."""

    round: int
    reported_confidence: float
    was_correct: bool
    user_decision: str
    user_payoff: float
    model_payoff: float


class TrialStatistics(TypedDict, total=False):
    """Statistics computed for a single trial."""

    num_rounds: int
    # Baseline stats
    baseline_accuracy: Optional[float]
    mean_baseline_confidence: Optional[float]
    # Model stats
    model_accuracy: float
    mean_model_confidence: float
    # Confidence comparison
    mean_confidence_diff: Optional[float]
    confidence_inflated_count: int
    confidence_deflated_count: int
    confidence_unchanged_count: int
    # Game-specific stats
    high_confidence_count: int
    high_confidence_accuracy: Optional[float]
    low_confidence_count: int
    low_confidence_accuracy: Optional[float]
    # User stats (twoPlayer only)
    purchase_count: int
    ignore_count: int
    purchase_rate: float
    mean_user_belief: float
    user_belief_error: float
    # Payoffs
    total_user_payoff: float
    total_model_payoff: float
    mean_user_payoff: float
    mean_model_payoff: float


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

    # -------------------------------------------------------------------------
    # LLM Configuration
    # -------------------------------------------------------------------------
    max_tokens: int = 512
    temperature: float = 0.0

    # -------------------------------------------------------------------------
    # Prompt Templates
    # -------------------------------------------------------------------------
    baseline_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "baseline_prompt.j2"
    )
    game_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "game_prompt.j2"
    )

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    dataset_name: str = "watermelonhjg/TAL-SCQ5K-EN-R1"

    # -------------------------------------------------------------------------
    # Game Parameters
    # -------------------------------------------------------------------------
    reward: float = 1.0  # r: reward user gets for correct answer
    cost: float = 0.1  # c: cost user pays to use model's answer
    discount_factor: float = 0.9  # δ: discount factor for future payoffs

    # -------------------------------------------------------------------------
    # Confidence Reporting
    # -------------------------------------------------------------------------
    confidence_mode: ConfidenceMode = ConfidenceMode.CONTINUOUS

    # -------------------------------------------------------------------------
    # Experiment Settings
    # -------------------------------------------------------------------------
    num_trials: int = 1
    num_rounds: int = 10  # Number of rounds per trial
    output_dir: str = "outputs"
    seed: int = 42

    def compute_threshold(self) -> float:
        """
        Compute the threshold θ* for user purchasing decision.

        θ* = 0.5 + c/r
        The user will purchase if their belief >= θ*.

        Returns:
            The purchasing threshold θ*
        """
        return 0.5 + self.cost / self.reward


# =============================================================================
# Dataset Utilities
# =============================================================================


def extract_question_from_dataset(sample: Dict[str, Any]) -> QuestionData:
    """
    Extract question from a TAL-SCQ5K dataset sample.

    The TAL-SCQ5K dataset format:
    - 'problem': Contains the question text with options embedded
    - 'solution': The correct answer (e.g., "(A) $$0$$")
    - 'difficulty': Difficulty rating (0-4)

    Args:
        sample: A sample from the dataset

    Returns:
        QuestionData with question, correct_answer, and difficulty
    """
    question = sample.get("problem", "")
    difficulty = sample.get("difficulty", None)
    solution = sample.get("solution", "")

    correct_answer = ""
    if solution:
        solution_stripped = solution.strip()
        if solution_stripped.startswith("(") and len(solution_stripped) >= 2:
            correct_answer = solution_stripped[1]
        else:
            correct_answer = solution_stripped

    return QuestionData(
        question=question,
        correct_answer=correct_answer,
        difficulty=difficulty,
    )


def evaluate_answer(response_answer: str, correct_answer: str) -> bool:
    """
    Evaluate whether the model's answer matches the correct answer.

    The correct_answer is typically a single letter (A, B, C, D, E).
    The model's response may be just the letter, or include additional text.

    Args:
        response_answer: The model's answer (e.g., "A", "(A)", "A. 24", etc.)
        correct_answer: The correct answer letter (e.g., "A")

    Returns:
        True if the answer is correct, False otherwise
    """
    response_normalized = response_answer.strip().upper()
    correct_normalized = correct_answer.strip().upper()

    response_letter = ""
    if response_normalized:
        if response_normalized[0] == "(":
            if len(response_normalized) >= 2:
                response_letter = response_normalized[1]
        elif response_normalized[0].isalpha():
            response_letter = response_normalized[0]

    return response_letter == correct_normalized


# =============================================================================
# LLM Query Functions
# =============================================================================


def ask_baseline(
    model_name: str,
    question: str,
    confidence_mode: ConfidenceMode,
    template_path: Path,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> ModelResponse:
    """
    Query the LLM with a question WITHOUT game context.

    Args:
        model_name: Name of the LLM model to use
        question: The question to ask
        confidence_mode: How confidence should be reported
        template_path: Path to the baseline prompt template
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        ModelResponse with answer and confidence
    """
    logger.debug(f"Asking baseline question with model {model_name}")
    prompt = load_template(
        template_path,
        question=question,
        confidence_mode=confidence_mode.value,
    )
    response, cost = query_llm(
        model_name, prompt, ModelResponse, max_tokens, temperature
    )
    logger.debug(f"LLM cost: ${cost:.6f}")
    return response


def ask_with_game_context(
    model_name: str,
    question: str,
    confidence_mode: ConfidenceMode,
    template_path: Path,
    reward: float,
    cost: float,
    discount_factor: float,
    threshold: float,
    history: Optional[List[HistoryEntry]] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> ModelResponse:
    """
    Query the LLM with a question WITH game context.

    Args:
        model_name: Name of the LLM model to use
        question: The question to ask
        confidence_mode: How confidence should be reported
        template_path: Path to the game prompt template
        reward: Reward for correct answer
        cost: Cost to user for purchasing
        discount_factor: Discount factor for future payoffs
        threshold: Purchasing threshold
        history: Optional interaction history
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        ModelResponse with answer and confidence
    """
    logger.debug(f"Asking game context question with model {model_name}")

    # Compute summary statistics for history if provided
    if history:
        purchase_count = sum(1 for h in history if h.get("user_decision") == "PURCHASE")
        correct_count = sum(1 for h in history if h.get("was_correct"))
        model_accuracy = correct_count / len(history)
        total_model_payoff = sum(h.get("model_payoff", 0) for h in history)
    else:
        purchase_count = 0
        model_accuracy = "N/A"
        total_model_payoff = 0.0

    prompt = load_template(
        template_path,
        question=question,
        reward=reward,
        cost=cost,
        discount_factor=discount_factor,
        threshold=threshold,
        history=history,
        model_accuracy=model_accuracy,
        purchase_count=purchase_count,
        total_model_payoff=total_model_payoff,
        confidence_mode=confidence_mode.value,
    )
    response, llm_cost = query_llm(
        model_name, prompt, ModelResponse, max_tokens, temperature
    )
    logger.debug(f"LLM cost: ${llm_cost:.6f}")
    return response


# =============================================================================
# Statistics Helpers
# =============================================================================


def compute_mean(values: List[float]) -> Optional[float]:
    """Compute mean of a list, returning None if empty."""
    return sum(values) / len(values) if values else None


def filter_none(values: List[Optional[float]]) -> List[float]:
    """Filter out None values from a list."""
    return [v for v in values if v is not None]


def compute_confidence_comparison_stats(
    round_results: List[RoundResult],
) -> Dict[str, Any]:
    """
    Compute statistics comparing baseline vs game confidence.

    Args:
        round_results: List of round results

    Returns:
        Dictionary with confidence comparison statistics
    """
    confidence_diffs = filter_none([r.get("confidence_diff") for r in round_results])

    return {
        "mean_confidence_diff": compute_mean(confidence_diffs),
        "confidence_inflated_count": sum(1 for d in confidence_diffs if d > 0),
        "confidence_deflated_count": sum(1 for d in confidence_diffs if d < 0),
        "confidence_unchanged_count": sum(1 for d in confidence_diffs if d == 0),
    }


def compute_baseline_stats(round_results: List[RoundResult]) -> Dict[str, Any]:
    """
    Compute baseline performance statistics.

    Args:
        round_results: List of round results

    Returns:
        Dictionary with baseline statistics
    """
    baseline_confidences = filter_none(
        [r.get("baseline_confidence") for r in round_results]
    )
    baseline_correct_count = sum(
        1 for r in round_results if r.get("baseline_correct") is True
    )

    return {
        "baseline_accuracy": baseline_correct_count / len(baseline_confidences)
        if baseline_confidences
        else None,
        "mean_baseline_confidence": compute_mean(baseline_confidences),
    }


def compute_model_stats(
    round_results: List[RoundResult], threshold: float
) -> Dict[str, Any]:
    """
    Compute model performance statistics with game context.

    Args:
        round_results: List of round results
        threshold: Confidence threshold for high/low classification

    Returns:
        Dictionary with model statistics
    """
    model_confidences = [
        r["model_confidence"]
        for r in round_results
        if r.get("model_confidence") is not None
    ]
    model_correct_count = sum(
        1 for r in round_results if r.get("model_correct") is True
    )

    high_confidence_rounds = [
        r
        for r in round_results
        if r.get("model_confidence") is not None and r["model_confidence"] >= threshold
    ]
    low_confidence_rounds = [
        r
        for r in round_results
        if r.get("model_confidence") is not None and r["model_confidence"] < threshold
    ]

    high_conf_correct = sum(
        1 for r in high_confidence_rounds if r.get("model_correct") is True
    )
    low_conf_correct = sum(
        1 for r in low_confidence_rounds if r.get("model_correct") is True
    )

    return {
        "model_accuracy": model_correct_count / len(round_results)
        if round_results
        else 0,
        "mean_model_confidence": compute_mean(model_confidences),
        "high_confidence_count": len(high_confidence_rounds),
        "high_confidence_accuracy": high_conf_correct / len(high_confidence_rounds)
        if high_confidence_rounds
        else None,
        "low_confidence_count": len(low_confidence_rounds),
        "low_confidence_accuracy": low_conf_correct / len(low_confidence_rounds)
        if low_confidence_rounds
        else None,
    }


def aggregate_trial_stats(
    trial_results: List[Dict[str, Any]], stat_key: str
) -> List[float]:
    """
    Aggregate a specific statistic across multiple trials, filtering None values.

    Args:
        trial_results: List of trial result dictionaries
        stat_key: Key of the statistic to aggregate

    Returns:
        List of non-None values for the statistic
    """
    return [
        t["statistics"][stat_key]
        for t in trial_results
        if t.get("statistics", {}).get(stat_key) is not None
    ]


def sum_trial_stats(trial_results: List[Dict[str, Any]], stat_key: str) -> float:
    """
    Sum a specific statistic across multiple trials.

    Args:
        trial_results: List of trial result dictionaries
        stat_key: Key of the statistic to sum

    Returns:
        Sum of the statistic values
    """
    return sum(
        t["statistics"].get(stat_key, 0) for t in trial_results if "statistics" in t
    )
