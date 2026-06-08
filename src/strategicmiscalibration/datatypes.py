"""Shared types and configuration objects for the strategic-uncertainty package.

This module is the canonical home for data contracts used across:
- game execution (`single_player`, `two_player`)
- sanitization payloads (`utils`)
- LLM/query configuration (`llm_interface` integration)
"""

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
    agent_reasoning: Optional[str]
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
    agent_accuracy: Optional[float]
    mean_agent_confidence: Optional[float]
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


@dataclass(frozen=True)
class SanitizedBaselineResponse:
    """Normalized baseline-response payload returned by sanitizer helpers."""

    is_valid: bool
    solution: Optional[str]
    confidence: Optional[float]
    correct: Optional[bool]


@dataclass(frozen=True)
class SanitizedAgentGameResponse:
    """Normalized strategic-agent payload returned by sanitizer helpers."""

    is_valid: bool
    solution: Optional[str]
    confidence: Optional[float]
    reasoning: Optional[str]
    correct: Optional[bool]


@dataclass(frozen=True)
class SanitizedUserDecisionResponse:
    """Normalized user-decision payload returned by sanitizer helpers."""

    is_valid: bool
    decision: Optional[str]
    reasoning: Optional[str]
    belief_agent_correct: Optional[float]
    belief_agent_ability: Optional[float]
    belief_honesty: Optional[float]


@dataclass(frozen=True)
class SanitizedUserPosteriorResponse:
    """Normalized user-posterior payload returned by sanitizer helpers."""

    is_valid: bool
    reasoning: Optional[str]
    belief_agent_correct: Optional[float]
    belief_agent_ability: Optional[float]
    belief_honesty: Optional[float]


class ToyRoundResult(TypedDict, total=False):
    """Result from a single round of the toy signaling game."""

    round: int
    # Synthetic task
    rho_t: float
    is_easy_task: bool
    # Agent signal (agent_confidence used for compatibility with analysis.py)
    agent_confidence: Optional[float]
    agent_reasoning: Optional[str]
    report_type: Optional[str]  # CORRECT_REPORTING | OVERREPORTING | SANDBAGGING
    # User priors entering this round (same names as RoundResult)
    prior_agent_honesty: float
    prior_agent_ability: float
    # User decision (same names as RoundResult)
    user_decision: Optional[
        str
    ]  # DELEGATE | SELF_SOLVE (normalized from SELF_COMPLETE)
    user_reasoning: Optional[str]
    # Post-signal beliefs (same names as RoundResult)
    user_belief_honesty: Optional[float]
    user_belief_agent_ability: Optional[float]
    # Outcome and posterior beliefs (same names as RoundResult)
    outcome: Optional[bool]
    posterior_user_belief_honesty: Optional[float]
    posterior_user_belief_agent_ability: Optional[float]
    posterior_reasoning: Optional[str]
    # Payoffs (same names as RoundResult)
    user_payoff: Optional[float]
    agent_payoff: Optional[float]
    is_valid: bool


# =============================================================================
# Game Configuration Classes
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

    # Game parameters
    reward: float = 1.0  # Reward the user gains if a delegated task is solved correctly
    cost: float = 0.1  # Cost the user pays to delegate the task
    discount_factor: float = 0.9  # δ: weight for first round payoffs
    effort: float = 0.5  # Effort cost the user pays to self-solve the task
    priors: bool = False  # Whether to supply agent with prior beliefs
    h_0: float = 0.5  # Honesty prior belief
    mu_0: float = 0.5  # Ability prior belief

    # Experiment settings
    num_trials: int = 1
    num_rounds: int = 10
    output_dir: str = "outputs"
    seed: int = 42

    def compute_threshold(self) -> float:
        """
        Compute the delegation threshold for the user.
        """
        return 1 - (self.effort - self.cost) / self.reward


@dataclass
class ToyGameConfig(BaseGameConfig):
    """Configuration for the toy signaling game (monopolistic setting)."""

    # Toy game parameters
    rho_plus: float = 0.85
    rho_minus: float = 0.15
    theta_H: float = 0.8
    theta_L: float = 0.2
    first_round_task: str = "EASY"  # "EASY" | "HARD" | "RANDOM"
    agent_eta: int = 0  # 0=strategic, 1=honest
    agent_theta_kind: str = "H"  # "H" | "L"

    # Prompt templates
    game_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "toy/game_agent_prompt.j2"
    )
    game_final_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "toy/game_agent_final_prompt.j2"
    )
    user_decision_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "toy/decision_user_prompt.j2"
    )
    user_posterior_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "toy/posterior_user_prompt.j2"
    )

    @property
    def agent_type_desc(self) -> str:
        eta_label = "HONEST" if self.agent_eta == 1 else "STRATEGIC"
        ability_label = "HIGH" if self.agent_theta_kind == "H" else "LOW"
        return f"{eta_label} and {ability_label}-ABILITY"


@dataclass
class MathQAGameConfig(BaseGameConfig):
    """Configuration specific to Math QA dataset experiments."""

    # Prompt templates
    baseline_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "math_qa/baseline_agent_prompt.j2"
    )
    game_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "math_qa/game_agent_prompt.j2"
    )
    user_decision_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "math_qa/decision_user_prompt.j2"
    )
    user_posterior_template_path: Path = field(
        default_factory=lambda: TEMPLATE_DIR / "math_qa/posterior_user_prompt.j2"
    )

    # Dataset
    dataset_name: str = "watermelonhjg/TAL-SCQ5K-EN-R1"

    # Confidence reporting
    confidence_mode: ConfidenceMode = ConfidenceMode.CONTINUOUS
