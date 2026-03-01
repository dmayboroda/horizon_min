"""Prompt building, tool schema, and output parsing for Duck Hunt GRPO training.

The tool schema and prompt format use Mistral's **native function-calling
protocol** so the trained model can be served via any OpenAI-compatible API
(vLLM, TGI, etc.) and called with the standard ``tools`` / ``tool_choice``
parameters from the OpenAI SDK.

Mistral tool-call token flow
-----------------------------
Prompt:
    [AVAILABLE_TOOLS] [<tool JSON>][/AVAILABLE_TOOLS]
    [INST] <user message> [/INST]

Model output:
    [TOOL_CALLS] [{"name": "shoot", "arguments": {...}, "id": "<9-char>"}]</s>

The ``tools`` list is passed to ``processor.apply_chat_template(tools=…)``
so the tokenizer inserts the correct special tokens automatically.
"""

from __future__ import annotations

import json
import logging
import re
import string
import random
from dataclasses import dataclass

from PIL import Image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 4.2  Tool schema  (OpenAI / Mistral format)
# ---------------------------------------------------------------------------
SHOOT_TOOL = {
    "type": "function",
    "function": {
        "name": "shoot",
        "description": (
            "Fire at predicted duck position. "
            "Analyze the frame sequence to estimate duck velocity, "
            "then predict where the duck will be after "
            "processing_latency_frames + horizon frames."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "x": {
                    "type": "number",
                    "description": (
                        "Predicted horizontal position, normalised 0.0–1.0. "
                        "0.0 = left edge, 1.0 = right edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": (
                        "Predicted vertical position, normalised 0.0–1.0. "
                        "0.0 = top edge, 1.0 = bottom edge."
                    ),
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "horizon": {
                    "type": "integer",
                    "description": (
                        "Additional frames to wait before shooting (0-30). "
                        "Total prediction = processing_latency_frames + horizon."
                    ),
                    "minimum": 0,
                    "maximum": 30,
                },
            },
            "required": ["x", "y", "horizon"],
        },
    },
}

# Convenience list expected by processor.apply_chat_template(tools=…)
TOOLS = [SHOOT_TOOL]


# ---------------------------------------------------------------------------
# 4.1  System prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATE = """\
You are a Duck Hunt AI. Shoot flying ducks by calling the shoot tool.

You see {num_frames} frames. Latency: {processing_latency_frames} frames.
Coordinates: x (0=left, 1=right), y (0=top, 1=bottom).
Predict where the duck will be after latency + horizon frames.

IMPORTANT: Respond ONLY with the tool call. Do NOT explain your reasoning."""


def format_system_prompt(
    *,
    num_frames: int = 4,
    processing_latency_frames: int = 6,
) -> str:
    """Return the system prompt with placeholders filled in."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        num_frames=num_frames,
        processing_latency_frames=processing_latency_frames,
    )


# ---------------------------------------------------------------------------
# 4.3  Prompt builder  (chat messages for the processor)
# ---------------------------------------------------------------------------
def build_prompt(
    frames: list[Image.Image],
    state: dict,
    num_frames: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Build chat messages **and** the tools list for ``apply_chat_template``.

    Parameters
    ----------
    frames : list[Image.Image]
        Observation frames (PIL, RGB).
    state : dict
        Game state from ``DuckHuntEnvWrapper.get_state()``.
    num_frames : int, optional
        Override for the ``{num_frames}`` placeholder.  Defaults to
        ``len(frames)``.

    Returns
    -------
    messages : list[dict]
        Chat messages in Mistral-3 multimodal format.
    tools : list[dict]
        Tool definitions to pass as ``tools=`` to ``apply_chat_template``.

    Usage::

        messages, tools = build_prompt(frames, state)
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    """
    if num_frames is None:
        num_frames = len(frames)

    latency_frames = state.get("simulated_latency_frames", 6)

    system_prompt = format_system_prompt(
        num_frames=num_frames,
        processing_latency_frames=latency_frames,
    )

    # --- user content: interleaved images + text ---
    user_content: list[dict] = []

    for img in frames:
        user_content.append({"type": "image", "image": img})

    state_text = (
        f"{num_frames} frames, {state.get('ducks_flying', '?')} ducks flying, "
        f"latency {latency_frames} frames. Shoot now."
    )
    user_content.append({"type": "text", "text": state_text})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        # Few-shot example so the model knows the expected output format
        {
            "role": "user",
            "content": [{"type": "text", "text": (
                "Frame sequence: 4 frames. Ducks flying: 2. "
                "Latency: 6 frames. Call the shoot tool now."
            )}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": '[TOOL_CALLS] [{"name": "shoot", "arguments": {"x": 0.45, "y": 0.25, "horizon": 8}, "id": "abc123xyz"}]'}],
        },
        {"role": "user", "content": user_content},
    ]

    return messages, TOOLS


# ---------------------------------------------------------------------------
# Helper: generate a Mistral-compatible 9-char tool-call ID
# ---------------------------------------------------------------------------
_CALL_ID_CHARS = string.ascii_letters + string.digits


def generate_call_id() -> str:
    """Return a random 9-character alphanumeric ID (Mistral requirement)."""
    return "".join(random.choices(_CALL_ID_CHARS, k=9))


# ---------------------------------------------------------------------------
# 4.4  Parse tool call from model output
# ---------------------------------------------------------------------------
@dataclass
class Action:
    """Parsed shoot action from the model's completion."""

    x: float
    y: float
    horizon: int


def parse_tool_call(
    output_text: str,
    max_horizon: int = 30,
) -> Action | None:
    """Extract an ``Action`` from the model's raw decoded output.

    Handles multiple formats the model may produce:

    1. **Mistral native** (preferred — what the model is trained for)::

           [TOOL_CALLS] [{"name":"shoot","arguments":{"x":0.3,"y":0.2,"horizon":5},"id":"AbC123xYz"}]

    2. **Plain JSON object** (fallback)::

           {"name":"shoot","arguments":{"x":0.3,"y":0.2,"horizon":5}}
           or just {"x":0.3,"y":0.2,"horizon":5}

    3. **Key=value** (last resort)::

           x=0.3, y=0.2, horizon=5

    Returns ``None`` if parsing fails entirely.
    """
    # --- attempt 1: Mistral [TOOL_CALLS] format ---
    #   [TOOL_CALLS] [{"name": "shoot", "arguments": {...}, "id": "..."}]
    tc_match = re.search(
        r"\[TOOL_CALLS\]\s*(\[.*\])", output_text, re.DOTALL
    )
    if tc_match:
        try:
            calls = json.loads(tc_match.group(1))
            if isinstance(calls, list) and len(calls) > 0:
                call = calls[0]
                args = call.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if "x" in args and "y" in args:
                    return _build_action(
                        args["x"],
                        args["y"],
                        args.get("horizon", 0),
                        max_horizon,
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # --- attempt 1b: Ministral [TOOL_CALLS]name[ARGS]{...} format ---
    args_match = re.search(
        r"\[TOOL_CALLS\]\s*\w+\s*\[ARGS\]\s*(\{.*?\})", output_text, re.DOTALL
    )
    if args_match:
        try:
            args = json.loads(args_match.group(1))
            if "x" in args and "y" in args:
                return _build_action(
                    args["x"], args["y"], args.get("horizon", 0), max_horizon
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # --- attempt 2: find any JSON object with x/y keys ---
    try:
        start = output_text.index("{")
        # Find matching closing brace (handle nested)
        depth = 0
        end = start
        for i, ch in enumerate(output_text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        blob = json.loads(output_text[start:end])
        # Nested: {"name": "shoot", "arguments": {…}}
        if "arguments" in blob and isinstance(blob["arguments"], dict):
            blob = blob["arguments"]
        elif "arguments" in blob and isinstance(blob["arguments"], str):
            blob = json.loads(blob["arguments"])
        if "x" in blob and "y" in blob:
            return _build_action(
                blob["x"], blob["y"], blob.get("horizon", 0), max_horizon
            )
    except (ValueError, json.JSONDecodeError, KeyError, TypeError):
        pass

    # --- attempt 3: key=value fallback ---
    kv_match = re.search(
        r"x\s*[=:]\s*([0-9.eE+-]+).*?"
        r"y\s*[=:]\s*([0-9.eE+-]+).*?"
        r"horizon\s*[=:]\s*([0-9]+)",
        output_text,
        re.DOTALL | re.IGNORECASE,
    )
    if kv_match:
        return _build_action(
            kv_match.group(1),
            kv_match.group(2),
            kv_match.group(3),
            max_horizon,
        )

    logger.warning("Failed to parse tool call from: %s", output_text[:200])
    return None


def _build_action(
    raw_x: str | float,
    raw_y: str | float,
    raw_horizon: str | int,
    max_horizon: int,
) -> Action:
    """Clamp values and return an ``Action``."""
    x = max(0.0, min(1.0, float(raw_x)))
    y = max(0.0, min(1.0, float(raw_y)))
    horizon = max(0, min(max_horizon, int(float(raw_horizon))))
    return Action(x=x, y=y, horizon=horizon)
