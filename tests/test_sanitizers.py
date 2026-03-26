from __future__ import annotations

from strategicuncertainty.llm_interface import (
    AgentBaselineResponse,
    AgentGameResponse,
    UserDecisionResponse,
    UserPosteriorResponse,
)
from strategicuncertainty.utils import (
    sanitize_agent_game_response,
    sanitize_baseline_response,
    sanitize_user_decision_response,
    sanitize_user_posterior_response,
)


def test_sanitize_baseline_response_none_input() -> None:
    result = sanitize_baseline_response(None, "A")
    assert result.is_valid is False
    assert result.solution is None
    assert result.confidence is None
    assert result.correct is None


def test_sanitize_baseline_response_invalid_confidence() -> None:
    response = AgentBaselineResponse(solution="A", confidence=1.5)
    result = sanitize_baseline_response(response, "A")
    assert result.is_valid is False
    assert result.solution == "A"
    assert result.confidence is None
    assert result.correct is True


def test_sanitize_agent_game_response_none_input() -> None:
    result = sanitize_agent_game_response(None, "A")
    assert result.is_valid is False
    assert result.solution is None
    assert result.confidence is None
    assert result.reasoning is None
    assert result.correct is None


def test_sanitize_agent_game_response_invalid_payload_field() -> None:
    response = AgentGameResponse(solution="A", confidence=0.8, reasoning="")
    result = sanitize_agent_game_response(response, "A")
    assert result.is_valid is False
    assert result.solution == "A"
    assert result.confidence == 0.8
    assert result.reasoning is None
    assert result.correct is True


def test_sanitize_user_decision_response_none_input() -> None:
    result = sanitize_user_decision_response(None)
    assert result.is_valid is False
    assert result.decision is None
    assert result.reasoning is None
    assert result.belief_agent_correct is None
    assert result.belief_agent_ability is None
    assert result.belief_honesty is None


def test_sanitize_user_decision_response_partial_invalid() -> None:
    response = UserDecisionResponse(
        decision="DELEGATE",
        reasoning="ok",
        belief_agent_correct=0.7,
        belief_agent_ability=2.0,
        belief_honesty=0.6,
    )
    result = sanitize_user_decision_response(response)
    assert result.is_valid is False
    assert result.decision == "DELEGATE"
    assert result.reasoning == "ok"
    assert result.belief_agent_correct == 0.7
    assert result.belief_agent_ability is None
    assert result.belief_honesty == 0.6


def test_sanitize_user_posterior_response_none_input() -> None:
    result = sanitize_user_posterior_response(None)
    assert result.is_valid is False
    assert result.reasoning is None
    assert result.belief_agent_correct is None
    assert result.belief_agent_ability is None
    assert result.belief_honesty is None


def test_sanitize_user_posterior_response_valid_input() -> None:
    response = UserPosteriorResponse(
        reasoning="updated",
        belief_agent_correct=0.9,
        belief_agent_ability=0.8,
        belief_honesty=0.7,
    )
    result = sanitize_user_posterior_response(response)
    assert result.is_valid is True
    assert result.reasoning == "updated"
    assert result.belief_agent_correct == 0.9
    assert result.belief_agent_ability == 0.8
    assert result.belief_honesty == 0.7
