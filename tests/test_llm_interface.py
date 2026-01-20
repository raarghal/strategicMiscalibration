"""
Tests for LLM interface cost tracking functionality.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from strategicuncertainty.llm_interface import _make_llm_request, query_llm


class MockResponseModel(BaseModel):
    """Mock response schema for testing."""

    answer: str = Field(description="Test answer")
    value: float = Field(description="Test value")


class TestCostTracking:
    """Test suite for LLM cost tracking functionality."""

    @patch("strategicuncertainty.llm_interface.completion")
    @patch("strategicuncertainty.llm_interface.completion_cost")
    def test_make_llm_request_returns_cost(self, mock_cost, mock_completion):
        """Test that _make_llm_request returns both response and cost."""
        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": "test", "value": 1.0}'
        mock_completion.return_value = mock_response

        # Mock the cost calculation
        mock_cost.return_value = 0.001234

        # Call the function
        response, cost = _make_llm_request(
            model="test-model",
            prompt="test prompt",
            response_template=MockResponseModel,
            max_tokens=100,
            temperature=0.0,
        )

        # Verify return types
        assert isinstance(response, str)
        assert isinstance(cost, float)

        # Verify values
        assert response == '{"answer": "test", "value": 1.0}'
        assert cost == 0.001234

        # Verify completion was called correctly
        mock_completion.assert_called_once()
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "test-model"
        assert call_args.kwargs["max_tokens"] == 100
        assert call_args.kwargs["temperature"] == 0.0

        # Verify cost calculation was called
        mock_cost.assert_called_once_with(completion_response=mock_response)

    @patch("strategicuncertainty.llm_interface.completion")
    @patch("strategicuncertainty.llm_interface.completion_cost")
    def test_make_llm_request_cost_fallback(self, mock_cost, mock_completion):
        """Test that _make_llm_request falls back to 0.0 cost on error."""
        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": "test", "value": 1.0}'
        mock_completion.return_value = mock_response

        # Mock cost calculation to raise an exception
        mock_cost.side_effect = Exception("Cost calculation failed")

        # Call the function
        response, cost = _make_llm_request(
            model="test-model",
            prompt="test prompt",
            response_template=MockResponseModel,
            max_tokens=100,
            temperature=0.0,
        )

        # Verify that cost falls back to 0.0
        assert cost == 0.0
        assert isinstance(response, str)

    @patch("strategicuncertainty.llm_interface.completion")
    @patch("strategicuncertainty.llm_interface.completion_cost")
    def test_query_llm_returns_parsed_and_cost(self, mock_cost, mock_completion):
        """Test that query_llm returns both parsed response and cost."""
        # Mock the completion response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"answer": "test answer", "value": 42.5}'
        mock_completion.return_value = mock_response

        # Mock the cost calculation
        mock_cost.return_value = 0.002468

        # Call the function
        parsed_response, cost = query_llm(
            model="test-model",
            prompt="test prompt",
            response_template=MockResponseModel,
            max_tokens=100,
            temperature=0.0,
        )

        # Verify return types
        assert isinstance(parsed_response, MockResponseModel)
        assert isinstance(cost, float)

        # Verify values
        assert parsed_response.answer == "test answer"
        assert parsed_response.value == 42.5
        assert cost == 0.002468

    @patch("strategicuncertainty.llm_interface.completion")
    @patch("strategicuncertainty.llm_interface.completion_cost")
    def test_query_llm_cost_with_parsing_error(self, mock_cost, mock_completion):
        """Test that cost is still tracked even when parsing fails."""
        # Mock the completion response with invalid JSON
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "invalid json"
        mock_completion.return_value = mock_response

        # Mock the cost calculation
        mock_cost.return_value = 0.001

        # Call the function and expect it to raise an error
        with pytest.raises(ValueError, match="Failed to parse LLM response"):
            query_llm(
                model="test-model",
                prompt="test prompt",
                response_template=MockResponseModel,
                max_tokens=100,
                temperature=0.0,
            )

        # Verify cost calculation was still called
        mock_cost.assert_called_once()

    @patch("strategicuncertainty.llm_interface.completion")
    @patch("strategicuncertainty.llm_interface.completion_cost")
    def test_query_llm_empty_response(self, mock_cost, mock_completion):
        """Test that query_llm handles empty responses correctly."""
        # Mock the completion response with empty content
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_completion.return_value = mock_response

        # Mock the cost calculation
        mock_cost.return_value = 0.001

        # Call the function and expect it to raise an error
        with pytest.raises(ValueError, match="Empty response from LLM"):
            query_llm(
                model="test-model",
                prompt="test prompt",
                response_template=MockResponseModel,
                max_tokens=100,
                temperature=0.0,
            )

        # Verify cost calculation was still called
        mock_cost.assert_called_once()
