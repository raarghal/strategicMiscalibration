"""LLM interface layer for structured prompting and response parsing.

Responsibilities:
- Define response schemas used by the LLM calls.
- Render Jinja prompt templates with runtime values.
- Send requests through LiteLLM with retries and structured output.
- Return parsed Pydantic payloads plus estimated request cost.

Two output modes are supported (see ``OutputMode``):
- ``JSON_SCHEMA``: rely on the provider's native ``response_format`` JSON-schema
  enforcement. Best for models that support it (e.g. Llama-3.3 via Together).
- ``TEXT``: ask for plain text, inject human-readable JSON instructions into the
  prompt (via the ``{{ json_instructions }}`` template variable), and parse the
  response with ``extract_json``. Use this for models that handle structured
  output poorly (e.g. some GPT-OSS deployments).
"""

import logging
import re
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import litellm
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from litellm import completion, completion_cost
from pydantic import BaseModel, Field, create_model
from tenacity import retry, stop_after_attempt, wait_random_exponential

logger = logging.getLogger(__name__)

ENV_FILE = Path(__file__).parent.parent.parent / ".env"
load_dotenv(ENV_FILE)

# Enable JSON schema validation globally for litellm.
# This is configured at module import time to ensure all LLM requests use JSON schema validation.
# Note: This affects all litellm usage in the application, not just this module.
litellm.enable_json_schema_validation = True


def configure_logging() -> None:
    """Configure application logging and suppress noisy third-party output."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("strategicmiscalibration").setLevel(logging.INFO)
    logging.getLogger("litellm").setLevel(logging.WARNING)


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
# Output Mode
# =============================================================================


class OutputMode(str, Enum):
    """Controls how the LLM is asked to produce structured output."""

    JSON_SCHEMA = "json_schema"  # Native JSON-schema enforcement via response_format
    TEXT = "text"  # Plain text; JSON instructions injected into the prompt


def reasoning_native(model: str) -> bool:
    """Whether ``model`` emits its chain-of-thought in a dedicated API channel
    (``reasoning_content``) rather than (only) in the response body. For such models
    we can request just the answer field(s) and read the reasoning from the channel,
    avoiding the truncation that a long in-body ``reasoning`` field causes."""
    return "gpt-oss" in model.lower()


@lru_cache(maxsize=None)
def answer_only_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """A copy of ``schema`` with the free-text ``reasoning`` field removed.

    Used for reasoning-native models: the model's chain-of-thought is captured from
    the API ``reasoning_content`` channel, so the JSON answer needs only the
    decision field(s) (e.g. ``{"signal": 0.85}``), which cannot be truncated by the
    token cap. Field descriptions/required-ness are preserved for the JSON
    instructions. Returns ``schema`` unchanged if it has no ``reasoning`` field."""
    if "reasoning" not in schema.model_fields:
        return schema
    fields = {name: (f.annotation, f) for name, f in schema.model_fields.items() if name != "reasoning"}
    return create_model(f"{schema.__name__}Answer", **fields)  # type: ignore[call-overload]


# =============================================================================
# Response Schemas
# =============================================================================


class AgentBaselineResponse(BaseModel):
    """Schema for the agent LLM's solution and confidence response."""

    solution: str = Field(description="The agent's proposed solution to the task")
    confidence: float = Field(description="Agent's confidence that the solution is correct (0.0 to 1.0)")


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

    reasoning: str = Field(description="Brief explanation of how the observed outcome changed beliefs")
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


class ToyAgentSignalResponse(BaseModel):
    """Schema for the toy-game agent's binary signal choice."""

    reasoning: str = Field(description="Strategic reasoning about signal choice")
    signal: float = Field(description="The reported signal: either rho_plus or rho_minus")


class ToyAgentStrategyResponse(BaseModel):
    """Schema for eliciting the agent's reporting probability (mixing weight) for its
    CURRENT, already-observed task state rho_t, rather than a single sampled action:
    the probability of reporting HIGH given the observed rho_t. The agent knows rho_t
    when it moves, matching the game (cf. notes/strategy_elicitation_scope.md)."""

    reasoning: str = Field(description="Strategic reasoning about the reporting probability")
    prob_high: float = Field(
        description="Probability in [0,1] of reporting the HIGH signal given the observed true success probability rho_t"
    )


class ToyUserDecisionResponse(BaseModel):
    """Schema for the toy-game user's delegation decision with two-belief update."""

    reasoning: str = Field(description="Bayesian reasoning for delegation decision")
    decision: str = Field(description="DELEGATE or SELF_COMPLETE")
    belief_honesty: float = Field(description="Updated honesty belief h after observing signal (0.0 to 1.0)")
    belief_agent_ability: float = Field(description="Updated ability belief mu after observing signal (0.0 to 1.0)")


class ToyUserPosteriorResponse(BaseModel):
    """Schema for the toy-game user's posterior belief update after observing outcome."""

    reasoning: str = Field(description="Reasoning for belief update after outcome")
    belief_honesty: float = Field(description="Final honesty belief after outcome (0.0 to 1.0)")
    belief_agent_ability: float = Field(description="Final ability belief after outcome (0.0 to 1.0)")


# =============================================================================
# JSON Instruction Generation & Parsing (for TEXT output mode)
# =============================================================================


# Maps JSON-schema primitive types to readable placeholders shown to the model.
_JSON_TYPE_PLACEHOLDERS = {
    "string": '"<string>"',
    "number": "<float>",
    "integer": "<integer>",
    "boolean": "<true/false>",
}


def generate_json_instructions(schema: type[BaseModel]) -> str:
    """Build a human-readable JSON output instruction block from a Pydantic schema.

    This is the single canonical source for the JSON instructions appended to
    prompts in ``OutputMode.TEXT``. It can also be called directly by researchers
    constructing prompts outside the standard pipeline.

    Args:
        schema: Pydantic model class describing the expected response.

    Returns:
        A string instructing the model to emit a JSON object with the schema's
        fields, e.g.::

            Respond with ONLY the following JSON object (no markdown, no extra text):
            {
              "reasoning": "<string>",
              "confidence": <float>
            }
    """
    properties = schema.model_json_schema().get("properties", {})
    lines = []
    for name, spec in properties.items():
        placeholder = _JSON_TYPE_PLACEHOLDERS.get(spec.get("type"), "<value>")
        lines.append(f'  "{name}": {placeholder}')
    body = ",\n".join(lines)
    return f"Respond with ONLY the following JSON object (no markdown, no extra text):\n{{\n{body}\n}}"


def _strip_reasoning_wrappers(text: str) -> str:
    """Remove reasoning-model scaffolding that can surround the JSON answer.

    Handles ``<think>...</think>`` blocks and the harmony-style
    ``<|channel|>analysis<|message|>...`` preambles emitted by some open models
    (e.g. GPT-OSS), which otherwise confuse brace extraction."""
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<\|channel\|>.*?<\|message\|>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<\|(?:end|start|return)\|>", " ", text)
    return text


def _iter_brace_objects(text: str) -> list[str]:
    """Yield every top-level ``{...}`` substring, respecting strings/escapes.

    Unlike a regex, this tracks brace depth and string state, so it returns
    *complete* objects even when several appear or when prose contains stray
    braces. A final unbalanced (truncated) object is returned as-is so the
    field-salvage fallback can still recover from it."""
    objs: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, in_str, esc, j = 0, False, False, i
        while j < n:
            ch = text[j]
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(text[i : j + 1])
                        break
            j += 1
        else:  # ran off the end without closing -> truncated tail
            objs.append(text[i:])
        i = j + 1
    return objs


def _salvage_fields(text: str, schema: type[BaseModel]) -> dict:
    """Best-effort per-field extraction for malformed/truncated JSON.

    Scans the raw text for each schema field's ``"name": value`` pair, tolerating
    a missing closing quote on a truncated trailing string (e.g. a ``reasoning``
    field cut off by ``max_tokens``). Returns a dict of recovered values for
    ``model_validate`` to coerce/complete."""
    out: dict = {}
    for name in schema.model_fields:
        m = re.search(
            rf'"{re.escape(name)}"\s*:\s*'
            r'("(?:[^"\\]|\\.)*"?|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)',
            text,
            re.DOTALL,
        )
        if not m:
            continue
        raw = m.group(1)
        if raw.startswith('"'):
            val = raw[1:-1] if raw.endswith('"') and len(raw) > 1 else raw[1:]
            if "\\" in val:
                try:
                    val = val.encode("utf-8").decode("unicode_escape")
                except Exception:
                    pass
            out[name] = val
        elif raw in ("true", "false"):
            out[name] = raw == "true"
        elif raw == "null":
            out[name] = None
        else:
            out[name] = float(raw) if ("." in raw or "e" in raw or "E" in raw) else int(raw)
    return out


def extract_json(text: str, schema: type[BaseModel]) -> BaseModel:
    """Robustly parse a raw text response into a Pydantic model.

    Tries, in order: (1) the whole stripped string, (2) a fenced ```json block,
    (3) each balanced ``{...}`` object found by a string-aware scan, and finally
    (4) a per-field salvage that tolerates truncated/malformed JSON (e.g. a
    response cut off mid-``reasoning`` by ``max_tokens``, which surfaces as a JSON
    "EOF" error). Reasoning-model wrappers are stripped first.

    Args:
        text: Raw text returned by the model.
        schema: Pydantic model class to validate against.

    Returns:
        The parsed Pydantic model instance.

    Raises:
        ValueError: If no strategy recovers a valid instance.
    """
    cleaned = _strip_reasoning_wrappers(text)

    candidates: list[str] = []
    stripped = cleaned.strip()
    if stripped:
        candidates.append(stripped)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fenced:
        candidates.append(fenced.group(1))
    candidates.extend(_iter_brace_objects(cleaned))

    for candidate in candidates:
        try:
            return schema.model_validate_json(candidate)
        except Exception:
            continue

    # Fallback: recover whatever fields are present (handles truncation).
    salvaged = _salvage_fields(cleaned, schema)
    if salvaged:
        try:
            return schema.model_validate(salvaged)
        except Exception:
            pass

    raise ValueError(f"Could not extract valid {schema.__name__} JSON from response: {text[:300]!r}")


# =============================================================================
# Template Loading
# =============================================================================


def load_template(
    template_path: Path,
    output_mode: OutputMode = OutputMode.JSON_SCHEMA,
    response_schema: Optional[type[BaseModel]] = None,
    **kwargs,
) -> str:
    """Load a Jinja2 template from file and render it with the given arguments.

    The template variable ``json_instructions`` is always provided so templates
    can place ``{{ json_instructions }}`` wherever the output-format block belongs:
    - In ``OutputMode.TEXT`` (with ``response_schema`` given) it renders the JSON
      instruction block from ``generate_json_instructions``.
    - Otherwise it renders an empty string (native schema enforcement is used).

    Args:
        template_path: Path to the Jinja2 template file.
        output_mode: Whether structured output is enforced natively or via text.
        response_schema: Pydantic schema used to build the JSON instructions in
            TEXT mode. Ignored in JSON_SCHEMA mode.
        **kwargs: Variables passed through to the template.

    Returns:
        The rendered prompt string.
    """
    if output_mode == OutputMode.TEXT and response_schema is not None:
        json_instructions = generate_json_instructions(response_schema)
    else:
        json_instructions = ""

    env = Environment(
        loader=FileSystemLoader(template_path.parent),
    )
    return env.get_template(template_path.name).render(json_instructions=json_instructions, **kwargs)


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
    output_mode: OutputMode = OutputMode.JSON_SCHEMA,
) -> Tuple[str, float, str]:
    """
    Make a request to the LLM API with retry logic.

    Args:
        model: Model name/identifier
        prompt: The prompt text to send
        response_template: Pydantic model class defining the response schema
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature
        output_mode: Whether to enforce JSON schema natively (JSON_SCHEMA) or
            request plain text (TEXT). In TEXT mode no ``response_format`` is sent.

    Returns:
        Tuple of (raw response content, estimated cost, reasoning-channel text).
        ``reasoning`` is the model's native chain-of-thought (``reasoning_content``)
        when present, else an empty string.
    """
    request_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if output_mode == OutputMode.JSON_SCHEMA:
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "schema": response_template.model_json_schema(),
        }
    # GPT-OSS-style reasoning models on Together emit very long chain-of-thought into
    # a separate ``reasoning_content`` channel; with default effort it exhausts
    # ``max_tokens`` before emitting the final answer, leaving ``content`` empty. Cap
    # the reasoning so the answer is actually produced (litellm blocks this param for
    # together_ai by default, so we allow-list it).
    if "gpt-oss" in model.lower():
        request_kwargs["reasoning_effort"] = "low"
        request_kwargs["allowed_openai_params"] = ["reasoning_effort"]

    try:
        response = completion(**request_kwargs)
    except Exception as e:
        logger.error(f"LLM REQUEST ERROR: {e}")
        raise e

    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        logger.warning(f"Failed to calculate cost: {e}")
        cost = 0.0
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", "") or ""
    content = message.content or ""
    if not content:
        # Reasoning models may leave ``content`` empty and place the answer (or a
        # recoverable tail) in ``reasoning_content``; fall back to it before retrying.
        content = reasoning
    if not content:
        # This triggers tenacity's retry with exponential backoff
        logger.error("Empty response from LLM, retrying...")
        raise ValueError("Empty response from LLM")

    return content, cost, reasoning


def query_llm(
    model: str,
    prompt: str,
    response_template: type[BaseModel],
    max_tokens: Optional[int] = 256,
    temperature: Optional[float] = 0.01,
    output_mode: OutputMode = OutputMode.JSON_SCHEMA,
    return_reasoning: bool = False,
):
    """
    Query the LLM and parse the response into the supplied response schema.

    Args:
        model: Model name/identifier
        prompt: The prompt text to send. In TEXT mode the prompt is expected to
            already contain JSON instructions (see ``load_template``).
        response_template: Pydantic model class defining the response schema
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature
        output_mode: JSON_SCHEMA uses ``model_validate_json`` on a natively
            enforced response; TEXT uses ``extract_json`` to robustly parse a
            free-text response.

    Returns:
        Tuple containing `(parsed_response_model, estimated_cost)`.
    """
    try:
        raw_response, cost, reasoning = _make_llm_request(
            model, prompt, response_template, max_tokens, temperature, output_mode
        )

        if not raw_response:
            logger.error("LLM RESPONSE ERROR: Empty response")
            raise ValueError("Empty response from LLM")

        try:
            if output_mode == OutputMode.TEXT:
                parsed_response = extract_json(raw_response, response_template)
            else:
                parsed_response = response_template.model_validate_json(raw_response)
            if return_reasoning:
                return parsed_response, cost, reasoning
            return parsed_response, cost
        except Exception as e:
            logger.error(f"LLM RESPONSE PARSING ERROR: {e}.")
            logger.error(f"Raw response: {raw_response}")
            raise ValueError("Failed to parse LLM response")

    except Exception as e:
        logger.error(f"LLM QUERY FAILED: {e}")
        raise e
