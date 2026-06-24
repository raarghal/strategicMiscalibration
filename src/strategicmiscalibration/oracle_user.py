"""Bayes-rational oracle user: a drop-in replacement for the LLM user.

Why this exists
---------------
The strategic agent's "best response" is only well defined against a *known*
opponent. When the counterpart user is itself an unconstrained LLM whose belief
updates are free-form (not Bayes), three different users float around — the
theory's ``sigma^U``, the user the agent *imagines*, and the user that actually
moves — and any observed miscalibration is partly a response to a non-equilibrium
opponent. Swapping in this oracle removes that confound: the agent faces the
exact Bayes-rational threshold user, turning "is the agent best-responding to the
equilibrium user?" into a sharp test.

The oracle's decision and belief updates are exact (see ``toy_model``) given a
``conjecture`` about the strategic type's reporting rule. The conjecture is the
one substantive modelling choice (default: the trusting ``STANDARD`` user); set
it to the agent's actual ``sigma^A`` to close the equilibrium fixed point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from . import toy_model
from .toy_model import ToyModelParams


@dataclass(frozen=True)
class OracleDecision:
    """Mirrors the fields the LLM user decision would populate."""

    decision: str  # "DELEGATE" | "SELF_SOLVE"
    belief_honesty: float  # post-signal h'
    belief_agent_ability: float  # post-signal mu'
    expected_success: float  # E[success | signal] used for the threshold
    reasoning: str


@dataclass(frozen=True)
class OraclePosterior:
    """Mirrors the fields the LLM user posterior would populate."""

    belief_honesty: float  # post-outcome h''
    belief_agent_ability: float  # post-outcome mu''
    reasoning: str


def decide(
    h: float,
    mu: float,
    signal: float,
    conjecture: Tuple[float, float],
    P: ToyModelParams,
) -> OracleDecision:
    """Bayes-rational delegation decision and post-signal belief update."""
    signal_high = abs(signal - P.rho_plus) < abs(signal - P.rho_minus)
    e_succ = toy_model.expected_success_given_signal(h, mu, signal_high, conjecture, P)
    h_post, mu_post = toy_model.posterior_after_signal(h, mu, signal_high, conjecture, P)
    delegate = e_succ >= P.rho_star
    reasoning = (
        f"E[success|signal]={e_succ:.4f} {'>=' if delegate else '<'} "
        f"rho_star={P.rho_star:.4f} -> {'DELEGATE' if delegate else 'SELF_SOLVE'}; "
        f"Bayes post-signal h={h_post:.4f}, mu={mu_post:.4f}."
    )
    return OracleDecision(
        decision="DELEGATE" if delegate else "SELF_SOLVE",
        belief_honesty=h_post,
        belief_agent_ability=mu_post,
        expected_success=e_succ,
        reasoning=reasoning,
    )


def update_posterior(
    h: float,
    mu: float,
    signal: float,
    success: bool,
    conjecture: Tuple[float, float],
    P: ToyModelParams,
) -> OraclePosterior:
    """Exact Bayes belief update after a delegated outcome is observed."""
    signal_high = abs(signal - P.rho_plus) < abs(signal - P.rho_minus)
    h_post, mu_post = toy_model.posterior_after_outcome(h, mu, signal_high, success, conjecture, P)
    reasoning = (
        f"Observed {'SUCCESS' if success else 'FAILURE'} after "
        f"{'high' if signal_high else 'low'} signal; "
        f"Bayes post-outcome h={h_post:.4f}, mu={mu_post:.4f}."
    )
    return OraclePosterior(
        belief_honesty=h_post,
        belief_agent_ability=mu_post,
        reasoning=reasoning,
    )
