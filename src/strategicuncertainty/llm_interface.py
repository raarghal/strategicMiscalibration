"""LLM interface layer for structured prompting and response parsing.

Responsibilities:
- Define response schemas used by the LLM calls.
- Render Jinja prompt templates with runtime values.
- Send requests through LiteLLM with retries and structured JSON schema output.
- Return parsed Pydantic payloads plus estimated request cost.
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple

import litellm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from litellm import completion, completion_cost
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_FILE)

# Enable JSON schema validation globally for litellm.
# This is configured at module import time to ensure all LLM requests use JSON schema validation.
# Note: This affects all litellm usage in the application, not just this module.
litellm.enable_json_schema_validation = True


# =============================================================================
# Confidence Mode
# =============================================================================


class ConfidenceMode(str, Enum):
    """Mode for how confidence scores are reported and discretized."""

    CONTINUOUS = "continuous"  # Full precision (0.0 to 1.0)
    BINARY = "binary"  # Only 0.0 or 1.0
    TERCILE = "tercile"  # 0.0 (Low), 0.5 (Medium), 1.0 (High)
    QUARTILE = "quartile"  # 0.0, 0.25, 0.5, 0.75, 1.0
    DECILE = "decile"  # Rounded to nearest 0.1


# =============================================================================
# Response Schemas
# =============================================================================


class AgentBaselineResponse(BaseModel):
    """Schema for the agent LLM's solution and confidence response."""

    solution: str = Field(description="The agent's proposed solution to the task")
    confidence: float = Field(
        description="Agent's confidence that the solution is correct (0.0 to 1.0)"
    )


class AgentGameResponse(BaseModel):
    """Schema for strategic-agent output (solution, confidence, reasoning)."""

    solution: str = Field(description="Agent's proposed solution to the task")
    confidence: float = Field(description="Agent's confidence signal to the user")
    reasoning: str = Field(description="Agent's reasoning behind the confidence signal")


class UserDecisionResponse(BaseModel):
    """Schema for the user's delegation decision response."""

    reasoning: str = Field(description="Brief explanation of the decision")
    decision: str = Field(description="DELEGATE or SELF_SOLVE")
    belief_agent_correct: float = Field(
        description="User's belief about the agent's probability of being correct (0.0 to 1.0)"
    )
    belief_agent_ability: float = Field(
        description="User's belief about the agent's general ability/success probability on similar tasks (0.0 to 1.0)"
    )
    belief_honesty: float = Field(
        description="User's belief that the agent is honest/truthful when reporting confidence (0.0 to 1.0)"
    )


class UserPosteriorResponse(BaseModel):
    """Schema for the user's posterior belief update after observing delegated outcome."""

    reasoning: str = Field(
        description="Brief explanation of how the observed outcome changed beliefs"
    )
    belief_agent_correct: Optional[float] = Field(
        default=None,
        description="Updated belief about the agent's probability of being correct (0.0 to 1.0). Optional if not updated.",
    )
    belief_agent_ability: float = Field(
        description="Updated belief about the agent's general ability/success probability on similar tasks (0.0 to 1.0)"
    )
    belief_honesty: float = Field(
        description="Updated belief that the agent is honest/truthful when reporting confidence (0.0 to 1.0)"
    )


# =============================================================================
# Template Loading
# =============================================================================


def load_template(template_path: Path, **kwargs) -> str:
    """Load a Jinja2 template from file and render with given arguments."""
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
    )
    return env.get_template(template_path.name).render(**kwargs)


# =============================================================================
# LLM Query Functions
# =============================================================================


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=5, max=120),
    reraise=True,
)
def _make_llm_request(
    model: str,
    prompt: str,
    response_template: type[BaseModel],
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.01,
) -> Tuple[str, float]:
    """
    Make a request to the LLM API with retry logic.

    Args:
        model: Model name/identifier
        prompt: The prompt text to send
        response_template: Pydantic model class defining the response schema
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature

    Returns:
        Tuple containing the raw response content and the estimated cost
    """
    try:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "schema": response_template.model_json_schema(),
            },
        )
    except Exception as e:
        logger.error(f"LLM REQUEST ERROR: {e}")
        raise e

    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.warning(f"Failed to calculate cost: {e}")
        cost = 0.0
    content = response.choices[0].message.content or ""
    if not content:
        # This triggers tenacity's retry with exponential backoff
        logger.error("Empty response from LLM, retrying...")
        raise ValueError("Empty response from LLM")

    return content, cost


def query_llm(
    model: str,
    prompt: str,
    response_template: type[BaseModel],
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.01,
) -> Tuple[BaseModel, float]:
    """
    Query the LLM and parse JSON into the supplied response schema.

    Args:
        model: Model name/identifier
        prompt: The prompt text to send
        response_template: Pydantic model class defining the response schema
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature

    Returns:
        Tuple containing `(parsed_response_model, estimated_cost)`.
    """
    try:
        raw_response, cost = _make_llm_request(
            model, prompt, response_template, max_tokens, temperature
        )

        if not raw_response:
            logger.error("LLM RESPONSE ERROR: Empty response")
            raise ValueError("Empty response from LLM")

        try:
            parsed_response = response_template.model_validate_json(raw_response)
            return parsed_response, cost
        except Exception as e:
            logger.error(f"LLM RESPONSE PARSING ERROR: {e}.")
            logger.error(f"Raw response: {raw_response}")
            raise ValueError("Failed to parse LLM response")

    except Exception as e:
        logger.error(f"LLM QUERY FAILED: {e}")
        raise e
