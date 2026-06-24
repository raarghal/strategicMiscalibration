"""Theory-side predictions for the toy game's two-threshold phase diagram.

Pure functions (no LLM, no I/O) implementing the closed-form predictions of the
phase-diagram law (``thm:phase`` in ``revised_main.tex``) so they can be overlaid
on the empirical sweep. The central object is the myopia ratio

    kappa = delta / (1 - delta)

(``delta`` weights the CURRENT-period fee, so large ``kappa`` = myopic). With
fragile post-failure reputation the strategic agent's first-period preference is

    Delta(rho) = c (1 - delta) [ kappa - (1 - rho) ],

so it prefers to report HIGH iff ``kappa > 1 - rho``. Evaluated at the two task
states this yields two thresholds:

* ``kappa = 1 - rho_plus``  -> onset of genuine deflation (sandbagging)
* ``kappa = 1 - rho_minus`` -> onset of inflation (over-reporting)

Because ``rho_plus > rho_minus`` we have ``1 - rho_plus < 1 - rho_minus``, giving
three regimes in increasing patience-to-myopia. Mapping a threshold in ``kappa``
back to ``delta`` uses ``delta = x / (1 + x)`` for ``x = 1 - rho``.

These thresholds are belief-independent (they do not move with ``h, mu``), so a
sweep that averages a binary action over a ``(h, mu)`` grid can still recover the
threshold *locations*; only the *level* of the curve is a grid-mix artifact.
"""

from __future__ import annotations

from enum import Enum

from .toy_model import ToyModelParams


class Regime(str, Enum):
    """Phase-diagram regime as a function of patience (myopia ratio kappa)."""

    SANDBAGGING = "SANDBAGGING"  # kappa < 1 - rho_plus: agent prefers to deflate
    TRANSITION = "TRANSITION"  # 1 - rho_plus < kappa < 1 - rho_minus: near-honest
    INFLATION = "INFLATION"  # kappa > 1 - rho_minus: report high regardless


def kappa(delta: float) -> float:
    """Myopia ratio kappa = delta / (1 - delta)."""
    if delta >= 1.0:
        return float("inf")
    return delta / (1.0 - delta)


def kappa_to_delta(x: float) -> float:
    """Invert kappa = delta/(1-delta): the delta at which kappa equals ``x``."""
    return x / (1.0 + x)


def delta_deflation_onset(P: ToyModelParams) -> float:
    """delta at which sandbagging turns on: kappa = 1 - rho_plus."""
    return kappa_to_delta(1.0 - P.rho_plus)


def delta_inflation_onset(P: ToyModelParams) -> float:
    """delta at which inflation turns on: kappa = 1 - rho_minus."""
    return kappa_to_delta(1.0 - P.rho_minus)


def regime(delta: float, P: ToyModelParams) -> Regime:
    """Classify the patience regime at discount factor ``delta``."""
    k = kappa(delta)
    if k > 1.0 - P.rho_minus:
        return Regime.INFLATION
    if k < 1.0 - P.rho_plus:
        return Regime.SANDBAGGING
    return Regime.TRANSITION


def prefers_high(rho_t_is_high: bool, delta: float, P: ToyModelParams) -> bool:
    """First-period preference: does the agent prefer HIGH at this task state?

    Implements ``sign Delta(rho) = sign(kappa - (1 - rho))`` for the relevant
    rho (rho_plus when the task is easy, rho_minus when hard).
    """
    rho = P.rho_plus if rho_t_is_high else P.rho_minus
    return kappa(delta) > (1.0 - rho)


def predicted_report_type(rho_t_is_high: bool, delta: float, P: ToyModelParams) -> str:
    """Predicted report direction relative to the true rho at this state.

    Returns one of ``CORRECT_REPORTING``, ``OVERREPORTING``, ``SANDBAGGING`` to
    match ``toy_game.classify_report_type``. The agent reports high iff
    ``prefers_high``; comparing that to the true state gives the direction.

    Note the structural ceiling: on an easy task the agent cannot over-report
    (no higher signal exists), and on a hard task it cannot sandbag — so a sweep
    must include BOTH task states to exercise both directions.
    """
    report_high = prefers_high(rho_t_is_high, delta, P)
    if rho_t_is_high:
        return "CORRECT_REPORTING" if report_high else "SANDBAGGING"
    return "OVERREPORTING" if report_high else "CORRECT_REPORTING"


def thresholds_summary(P: ToyModelParams) -> dict:
    """Convenience bundle of the derived thresholds for plotting/printing."""
    return {
        "rho_star": P.rho_star,
        "kappa_deflation_onset": 1.0 - P.rho_plus,
        "kappa_inflation_onset": 1.0 - P.rho_minus,
        "delta_deflation_onset": delta_deflation_onset(P),
        "delta_inflation_onset": delta_inflation_onset(P),
    }
