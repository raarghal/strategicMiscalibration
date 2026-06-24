"""Tests for the toy Bayes model, oracle user, and theory predictions.

These are pure-math checks (no LLM): they pin down the exact-Bayes ground truth
that the oracle user and the phase-diagram overlay rely on.
"""

from __future__ import annotations


import pytest

from strategicmiscalibration import oracle_user, toy_model, toy_theory
from strategicmiscalibration.toy_model import BABBLE_UP, STANDARD, ToyModelParams
from strategicmiscalibration.toy_theory import Regime

P = ToyModelParams(
    rho_plus=0.85,
    rho_minus=0.15,
    theta_H=0.8,
    theta_L=0.2,
    reward=1.0,
    cost=0.1,
    effort=0.5,
)


def test_rho_star():
    # rho_star = 1 - (e - c)/r = 1 - 0.4 = 0.6
    assert P.rho_star == pytest.approx(0.6)


def test_p_easy_monotone_in_mu():
    assert toy_model.p_easy(0.0, P) == pytest.approx(P.theta_L)
    assert toy_model.p_easy(1.0, P) == pytest.approx(P.theta_H)
    assert toy_model.p_easy(0.5, P) == pytest.approx(0.5)


def test_signal_high_prob_in_unit_interval():
    for h in (0.1, 0.5, 0.9):
        for mu in (0.1, 0.5, 0.9):
            p = toy_model.signal_high_prob(h, mu, STANDARD, P)
            assert 0.0 <= p <= 1.0


def test_standard_conjecture_high_signal_means_easy():
    # Under STANDARD (strategic also truthful), a high signal is fully informative:
    # success prob equals rho_plus regardless of beliefs.
    for h in (0.2, 0.8):
        for mu in (0.2, 0.8):
            e = toy_model.expected_success_given_signal(h, mu, True, STANDARD, P)
            assert e == pytest.approx(P.rho_plus)
            e_low = toy_model.expected_success_given_signal(h, mu, False, STANDARD, P)
            assert e_low == pytest.approx(P.rho_minus)


def test_standard_user_delegates_only_on_high():
    d_hi = oracle_user.decide(0.5, 0.5, P.rho_plus, STANDARD, P)
    d_lo = oracle_user.decide(0.5, 0.5, P.rho_minus, STANDARD, P)
    assert d_hi.decision == "DELEGATE"  # rho_plus=0.85 >= rho_star=0.6
    assert d_lo.decision == "SELF_SOLVE"  # rho_minus=0.15 < 0.6


def test_babble_up_discounts_high_signal():
    # Under BABBLE_UP the strategic type always claims high, so a high signal is
    # discounted toward the prior; success prob strictly below rho_plus when h<1.
    e = toy_model.expected_success_given_signal(0.5, 0.5, True, BABBLE_UP, P)
    assert P.rho_minus < e < P.rho_plus


def test_posteriors_are_probabilities():
    for s_high in (True, False):
        h2, mu2 = toy_model.posterior_after_signal(0.5, 0.5, s_high, BABBLE_UP, P)
        assert 0.0 <= h2 <= 1.0 and 0.0 <= mu2 <= 1.0
        for succ in (True, False):
            h3, mu3 = toy_model.posterior_after_outcome(0.5, 0.5, s_high, succ, BABBLE_UP, P)
            assert 0.0 <= h3 <= 1.0 and 0.0 <= mu3 <= 1.0


def test_failure_after_high_signal_lowers_honesty_under_babble():
    # Reputation channel: under BABBLE_UP, failing after claiming high is evidence
    # of dishonesty (or low ability), so posterior honesty should drop.
    h0 = 0.5
    op = oracle_user.update_posterior(h0, 0.5, P.rho_plus, False, BABBLE_UP, P)
    assert op.belief_honesty < h0


def test_outcome_posterior_consistent_with_signal_posterior():
    # Averaging the post-outcome honesty over the two outcomes, weighted by their
    # conditional probabilities, must return the post-signal honesty (tower rule).
    h, mu, s_high = 0.5, 0.5, True
    e_succ = toy_model.expected_success_given_signal(h, mu, s_high, BABBLE_UP, P)
    h_sig, _ = toy_model.posterior_after_signal(h, mu, s_high, BABBLE_UP, P)
    h_succ, _ = toy_model.posterior_after_outcome(h, mu, s_high, True, BABBLE_UP, P)
    h_fail, _ = toy_model.posterior_after_outcome(h, mu, s_high, False, BABBLE_UP, P)
    assert e_succ * h_succ + (1 - e_succ) * h_fail == pytest.approx(h_sig)


def test_threshold_locations():
    # delta solving kappa = 1 - rho: delta = (1-rho)/(2-rho).
    assert toy_theory.delta_deflation_onset(P) == pytest.approx((1 - 0.85) / (2 - 0.85))
    assert toy_theory.delta_inflation_onset(P) == pytest.approx((1 - 0.15) / (2 - 0.15))
    # ordering: deflation onset is at lower delta than inflation onset
    assert toy_theory.delta_deflation_onset(P) < toy_theory.delta_inflation_onset(P)


def test_regime_classification():
    assert toy_theory.regime(0.05, P) == Regime.SANDBAGGING
    assert toy_theory.regime(0.30, P) == Regime.TRANSITION
    assert toy_theory.regime(0.95, P) == Regime.INFLATION


def test_predicted_report_type_directionality():
    # Myopic (high delta): report high regardless -> over-report on a hard task.
    assert toy_theory.predicted_report_type(False, 0.95, P) == "OVERREPORTING"
    # Patient (low delta): deflate -> sandbag on an easy task.
    assert toy_theory.predicted_report_type(True, 0.05, P) == "SANDBAGGING"
    # Structural ceiling: never over-report on an easy task.
    assert toy_theory.predicted_report_type(True, 0.95, P) != "OVERREPORTING"
