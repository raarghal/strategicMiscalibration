import logging
from enum import Enum
from pathlib import Path
from typing import Optional

import together
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_FILE)


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


class ModelResponse(BaseModel):
    """Schema for model LLM's answer and confidence response."""

    answer: str = Field(description="The model's answer to the question")
    confidence: float = Field(
        description="Confidence score that the answer is correct (0.0 to 1.0)"
    )


class UserResponse(BaseModel):
    """Schema for user LLM's purchase decision response."""

    reasoning: str = Field(description="Brief explanation of the decision")
    decision: str = Field(description="PURCHASE or IGNORE")
    belief_ai_correct: float = Field(
        description="User's belief about AI's probability of being correct (0.0 to 1.0)"
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
) -> str:
    """
    Make a request to the LLM API with retry logic.

    Args:
        model: Model name/identifier
        prompt: The prompt text to send
        response_template: Pydantic model class defining the response schema
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature

    Returns:
        Raw response content as string
    """
    client = together.Together()

    try:
        response = client.chat.completions.create(
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
        print(f"LLM REQUEST ERROR: {e}")
        raise e

    return response.choices[0].message.content


def query_llm(
    model: str,
    prompt: str,
    response_template: type[BaseModel],
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.01,
) -> BaseModel:
    """
    Query the LLM with a prompt and parse the JSON response.

    Args:
        model: Model name/identifier
        prompt: The prompt text to send
        response_template: Pydantic model class defining the response schema
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature

    Returns:
        Parsed response as an instance of response_template
    """
    try:
        raw_response = _make_llm_request(
            model, prompt, response_template, max_tokens, temperature
        )

        if not raw_response:
            print("LLM RESPONSE ERROR: Empty response")
            raise ValueError("Empty response from LLM")

        try:
            parsed_response = response_template.model_validate_json(raw_response)
            return parsed_response
        except Exception as e:
            print(f"LLM RESPONSE PARSING ERROR: {e}.")
            print(f"Raw response: {raw_response}")
            raise ValueError("Failed to parse LLM response")

    except Exception as e:
        print(f"LLM QUERY FAILED: {e}")
        raise e
