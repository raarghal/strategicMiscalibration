from __future__ import annotations

import strategicuncertainty as su
from strategicuncertainty.two_player import compute_payoffs
from strategicuncertainty.utils import (
    build_round_result,
    compute_confidence_diff,
    evaluate_solution,
    extract_task_from_dataset,
    normalize_probability,
    normalize_user_decision,
)


def test_package_exports_are_valid_symbols() -> None:
    assert su.BaseGameConfig is not None
    assert su.ConfidenceMode is not None
    assert su.AgentBaselineResponse is not None
    assert su.AgentGameResponse is not None
    assert su.UserDecisionResponse is not None
    assert su.UserPosteriorResponse is not None
    assert su.extract_task_from_dataset is not None
    assert su.evaluate_solution is not None


def test_compute_payoffs_rejects_unknown_delegated_outcome() -> None:
    try:
        compute_payoffs("DELEGATE", None, reward=1.0, cost=0.1, effort=0.5)
        assert False, "Expected ValueError for unknown delegated outcome"
    except ValueError:
        pass


def test_compute_payoffs_self_solve_ignores_agent_correct() -> None:
    payoffs = compute_payoffs("SELF_SOLVE", None, reward=1.0, cost=0.1, effort=0.5)
    assert payoffs["user_payoff"] == 0.5
    assert payoffs["agent_payoff"] == 0.0


def test_extract_task_from_dataset_handles_parenthesized_solution() -> None:
    sample = {"problem": "Q?", "solution": "(B) because", "difficulty": "easy"}
    task = extract_task_from_dataset(sample)
    assert task["correct_solution"] == "B"


def test_extract_task_from_dataset_handles_plain_solution() -> None:
    sample = {"problem": "Q?", "solution": "c is correct", "difficulty": "easy"}
    task = extract_task_from_dataset(sample)
    assert task["correct_solution"] == "c"


def test_evaluate_solution_normalizes_parenthesized_formats() -> None:
    assert evaluate_solution("(a)", "A")
    assert evaluate_solution("A. final", "(A)")
    assert not evaluate_solution("B", "(A)")


def test_normalize_user_decision_handles_variants() -> None:
    assert normalize_user_decision("delegate") == "DELEGATE"
    assert normalize_user_decision("self solve") == "SELF_SOLVE"
    assert normalize_user_decision("I will self-solve this") == "SELF_SOLVE"
    assert normalize_user_decision("unknown") is None


def test_normalize_probability_bounds_and_types() -> None:
    assert normalize_probability(0.5) == 0.5
    assert normalize_probability("0.0") == 0.0
    assert normalize_probability(-0.1) is None
    assert normalize_probability(1.2) is None
    assert normalize_probability("nan") is None


def test_compute_confidence_diff_requires_both_values() -> None:
    assert compute_confidence_diff(0.2, 0.4) == 0.2
    assert compute_confidence_diff(None, 0.4) is None
    assert compute_confidence_diff(0.2, None) is None


def test_build_round_result_contains_expected_fields() -> None:
    rr = build_round_result(
        round_idx=2,
        sample_idx=3,
        task="task",
        difficulty=None,
        correct_solution="A",
        baseline_solution=None,
        baseline_confidence=None,
        baseline_correct=None,
        agent_solution=None,
        agent_confidence=None,
        agent_correct=None,
        confidence_diff=None,
    )
    assert rr["round"] == 2
    assert rr["sample_idx"] == 3
    assert "user_payoff" in rr


def test_build_round_result_handles_failed_round_shape() -> None:
    rr = build_round_result(
        round_idx=0,
        sample_idx=1,
        task="q",
        difficulty="easy",
        correct_solution="A",
        baseline_solution="A",
        baseline_confidence=0.5,
        baseline_correct=True,
        agent_solution="A",
        agent_confidence=None,
        agent_correct=True,
        agent_reasoning="r",
        confidence_diff=None,
        prior_agent_honesty=0.5,
        prior_agent_ability=0.5,
    )
    assert rr["user_decision"] is None
    assert rr["agent_confidence"] is None
