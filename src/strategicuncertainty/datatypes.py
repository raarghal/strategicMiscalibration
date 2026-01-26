from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict

from .llm_interface import (
    ConfidenceMode,
)

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
    # User priors
    prior_agent_honesty: Optional[float]
    prior_agent_ability: Optional[float]
    # User behaviour (two-player mode)
    user_decision: Optional[str]
    user_reasoning: Optional[str]
    user_belief_agent_correct: Optional[float]
    # Additional user beliefs
    user_belief_agent_ability: Optional[float]
    user_belief_honesty: Optional[float]
    # Posterior beliefs (after observing delegated outcome)
    posterior_user_belief_agent_ability: Optional[float]
    posterior_user_belief_honesty: Optional[float]
    # Payoffs
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
    # User beliefs (prior to posterior update)
    mean_user_belief_agent_correct: Optional[float]
    mean_user_belief_agent_ability: Optional[float]
    mean_user_belief_honesty: Optional[float]
    # Posterior user beliefs (after observing delegated outcome)
    mean_posterior_user_belief_agent_ability: Optional[float]
    mean_posterior_user_belief_honesty: Optional[float]
    # Calibration/error
    user_belief_error: Optional[float]
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
    user_model_name: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
    agent_model_name: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
    max_tokens: int = 512
    temperature: float = 0.0

    # Prompt templates
    baseline_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "baseline_agent_prompt.j2"
    )
    game_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "game_agent_prompt.j2"
    )
    # Prompt template for the user (agent templates inherited from BaseGameConfig)
    user_decision_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "decision_user_prompt.j2"
    )
    user_posterior_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "posterior_user_prompt.j2"
    )

    # Dataset
    dataset_name: str = "watermelonhjg/TAL-SCQ5K-EN-R1"

    # Game parameters
    reward: float = 1.0  # Reward the user gains if a delegated task is solved correctly
    cost: float = 0.2  # Cost the user pays to delegate the task
    discount_factor: float = 0.9  # δ: discount factor for future payoffs
    effort: float = 0.5  # Effort cost the user pays to self-solve the task
    priors: bool = False  # Whether to supply agent with prior beliefs
    h_0: float = 0.5  # Honesty prior belief
    mu_0: float = 0.5  # Ability prior belief

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
