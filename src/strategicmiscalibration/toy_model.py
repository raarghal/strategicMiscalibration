"""Exact discrete Bayes model for the two-player toy signaling game.

This module is the closed-form ground truth for the toy game's information
structure. It is pure math (no LLM, no I/O) and is shared by:

- ``oracle_user`` — a drop-in Bayes-rational user that replaces the LLM user,
  so the strategic agent can be probed against the *actual* equilibrium-style
  opponent rather than a free-form LLM (addresses the "actions vs. equilibria"
  gap: the agent's best response is only well defined against a known user).
- ``toy_theory`` — derived phase-diagram thresholds and per-state predicted
  behaviour, used to overlay theory on the empirical sweep.

Model
-----
An agent has two independent binary attributes:

* ability ``A in {H, L}`` with user belief ``mu = P(A = H)``. A type-``A`` agent
  draws an EASY task with probability ``theta_A`` (``theta_H`` or ``theta_L``).
* honesty ``eta in {honest, strategic}`` with user belief ``h = P(honest)``.

A task is EASY (success prob ``rho_plus``) or HARD (success prob ``rho_minus``).
The agent observes its task difficulty and emits a binary signal
``s in {rho_plus, rho_minus}`` ("high"/"low").

* An HONEST agent reports ``high`` iff the task is easy.
* A STRATEGIC agent reports ``high`` with probability ``a_plus`` on an easy task
  and ``a_minus`` on a hard task. The pair ``conjecture = (a_plus, a_minus)`` is
  the user's *belief about the strategic type's reporting rule*.

The conjecture is the single modelling choice that makes "exact Bayes" well
defined off the equilibrium path. Two natural presets:

* ``STANDARD = (1.0, 0.0)`` — the strategic type is conjectured to report
  truthfully; the user trusts the signal (the "standard user" of the theory).
* ``BABBLE_UP = (1.0, 1.0)`` — the strategic type always claims ``high``; a high
  signal is then discounted by ``(1 - h)`` and failures are informative about
  honesty (gives the user a reason to punish, i.e. a reputation channel).

Closing the fixed point (true MPBE) amounts to setting ``conjecture`` equal to
the agent's *actual* strategy ``sigma^A``; exposing it as a parameter lets the
caller decide rather than baking in circular beliefs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Conjecture presets: (a_plus, a_minus) = P(report high | easy), P(report high | hard).
STANDARD: Tuple[float, float] = (1.0, 0.0)
BABBLE_UP: Tuple[float, float] = (1.0, 1.0)


@dataclass(frozen=True)
class ToyModelParams:
    """Primitive parameters of the toy game's information/payoff structure."""

    rho_plus: float = 0.85
    rho_minus: float = 0.15
    theta_H: float = 0.8
    theta_L: float = 0.2
    reward: float = 1.0
    cost: float = 0.1
    effort: float = 0.5

    @property
    def rho_star(self) -> float:
        """Delegation threshold on posterior success probability.

        The user delegates iff ``E[success | signal] >= rho_star``.
        """
        return 1.0 - (self.effort - self.cost) / self.reward

    @classmethod
    def from_config(cls, cfg) -> "ToyModelParams":
        """Build params from a ``ToyGameConfig``/``ToyExperimentConfig``."""
        return cls(
            rho_plus=cfg.rho_plus,
            rho_minus=cfg.rho_minus,
            theta_H=cfg.theta_H,
            theta_L=cfg.theta_L,
            reward=cfg.reward,
            cost=cfg.cost,
            effort=cfg.effort,
        )


# A "cell" is one of the four (ability, honesty) types. We carry, per cell, its
# prior mass, its easy-task probability theta_A, and whether it is honest.
def _cells(h: float, mu: float, P: ToyModelParams):
    """Yield (prior_mass, theta_A, is_honest) for the four type cells."""
    return (
        (mu * h, P.theta_H, True),  # high ability, honest
        (mu * (1.0 - h), P.theta_H, False),  # high ability, strategic
        ((1.0 - mu) * h, P.theta_L, True),  # low ability, honest
        ((1.0 - mu) * (1.0 - h), P.theta_L, False),  # low ability, strategic
    )


def _joint_difficulty_signal(theta_A, is_honest, conjecture):
    """Return P(easy & s_high), P(hard & s_high), P(easy & s_low), P(hard & s_low)
    for a single cell.
    """
    a_plus, a_minus = conjecture
    if is_honest:
        # honest reports high iff easy
        return (theta_A, 0.0, 0.0, 1.0 - theta_A)
    return (
        theta_A * a_plus,  # easy & high
        (1.0 - theta_A) * a_minus,  # hard & high
        theta_A * (1.0 - a_plus),  # easy & low
        (1.0 - theta_A) * (1.0 - a_minus),  # hard & low
    )


def p_easy(mu: float, P: ToyModelParams) -> float:
    """Prior probability the task is easy given ability belief ``mu``."""
    return mu * P.theta_H + (1.0 - mu) * P.theta_L


def signal_high_prob(h: float, mu: float, conjecture, P: ToyModelParams) -> float:
    """Marginal probability the agent emits a HIGH signal."""
    tot = 0.0
    for prior, theta_A, is_honest in _cells(h, mu, P):
        e_hi, h_hi, _, _ = _joint_difficulty_signal(theta_A, is_honest, conjecture)
        tot += prior * (e_hi + h_hi)
    return tot


def expected_success_given_signal(h: float, mu: float, signal_high: bool, conjecture, P: ToyModelParams) -> float:
    """Posterior E[success | signal] under the conjecture about strategic agents.

    Returns ``rho_star`` neutrally (no delegation incentive either way) if the
    signal has zero probability under the current beliefs/conjecture.
    """
    num = 0.0  # sum prior * P(difficulty & signal) * success_prob
    den = 0.0  # sum prior * P(signal)
    for prior, theta_A, is_honest in _cells(h, mu, P):
        e_hi, h_hi, e_lo, h_lo = _joint_difficulty_signal(theta_A, is_honest, conjecture)
        if signal_high:
            p_easy_s, p_hard_s = e_hi, h_hi
        else:
            p_easy_s, p_hard_s = e_lo, h_lo
        num += prior * (p_easy_s * P.rho_plus + p_hard_s * P.rho_minus)
        den += prior * (p_easy_s + p_hard_s)
    if den <= 0.0:
        return P.rho_star
    return num / den


def posterior_after_signal(
    h: float, mu: float, signal_high: bool, conjecture, P: ToyModelParams
) -> Tuple[float, float]:
    """Exact Bayes update of ``(h, mu)`` after observing the signal only."""
    masses = []
    for prior, theta_A, is_honest in _cells(h, mu, P):
        e_hi, h_hi, e_lo, h_lo = _joint_difficulty_signal(theta_A, is_honest, conjecture)
        p_sig = (e_hi + h_hi) if signal_high else (e_lo + h_lo)
        masses.append(prior * p_sig)
    return _marginals(masses, h, mu)


def posterior_after_outcome(
    h: float,
    mu: float,
    signal_high: bool,
    success: bool,
    conjecture,
    P: ToyModelParams,
) -> Tuple[float, float]:
    """Exact Bayes update of ``(h, mu)`` after signal AND a delegated outcome."""
    masses = []
    for prior, theta_A, is_honest in _cells(h, mu, P):
        e_hi, h_hi, e_lo, h_lo = _joint_difficulty_signal(theta_A, is_honest, conjecture)
        if signal_high:
            p_easy_s, p_hard_s = e_hi, h_hi
        else:
            p_easy_s, p_hard_s = e_lo, h_lo
        # Outcome likelihood folds in task difficulty's success probability.
        if success:
            like = p_easy_s * P.rho_plus + p_hard_s * P.rho_minus
        else:
            like = p_easy_s * (1.0 - P.rho_plus) + p_hard_s * (1.0 - P.rho_minus)
        masses.append(prior * like)
    return _marginals(masses, h, mu)


def _marginals(masses, h_fallback: float, mu_fallback: float) -> Tuple[float, float]:
    """Collapse the four ordered cell masses back into (h', mu') marginals.

    Cell order matches ``_cells``: (H,hon), (H,str), (L,hon), (L,str).
    """
    total = sum(masses)
    if total <= 0.0:
        return h_fallback, mu_fallback
    p = [m / total for m in masses]
    mu_post = p[0] + p[1]  # high-ability cells
    h_post = p[0] + p[2]  # honest cells
    return h_post, mu_post


def delegates(h: float, mu: float, signal_high: bool, conjecture, P: ToyModelParams) -> bool:
    """Bayes-rational threshold decision: delegate iff E[success|s] >= rho_star."""
    return expected_success_given_signal(h, mu, signal_high, conjecture, P) >= P.rho_star
