import logging
from pathlib import Path
from typing import Optional

import together
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"


def load_template(template_path: Path, **kwargs) -> str:
    """Load a Jinja2 template from file."""
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
    )
    return env.get_template(template_path.name).render(**kwargs)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(multiplier=1, min=5, max=120),
    reraise=True,
)
def _make_llm_request(
    self,
    model: str,
    prompt: str,
    response_template,
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.01,
) -> dict:
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
    self,
    model: str,
    prompt: str,
    response_template: BaseModel,
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.01,
) -> dict:
    """
    Query the LLM with a prompt and parse the JSON response.

    Args:
        model (str): Model name
        prompt (str): Prompt text
        response_template (BaseModel): Pydantic model for response schema
        max_tokens (int, optional): Max tokens for response
        temperature (float, optional): Sampling temperature

    Returns:
        dict: Parsed JSON response
    """
    try:
        raw_response = self._make_llm_request(
            model, prompt, response_template, max_tokens, temperature
        )

        if not raw_response:
            print("LLM RESPONSE ERROR: Empty response")
            raise ValueError("Empty response from LLM")

        try:
            parsed_response = response_template.model_validate_json(raw_response)
            return parsed_response
        except Exception as e:
            print(f"🔴 LLM RESPONSE PARSING ERROR: {e}.")
            print(f"Raw response: {raw_response}")
            raise ValueError("Failed to parse LLM response")

    except Exception as e:
        print(f"🔴 LLM QUERY FAILED: {e}")
        raise e


class llm_response(BaseModel):
    answer: str = Field(description="Answer")
    confidence: float = Field(description="Confidence score (0.0 to 1.0)")
