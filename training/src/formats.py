"""Model-family-specific tool schemas, prompt builders, and output parsers.

Each model family (Mistral, LiquidAI, …) has a different:
  - Tool-call token format
  - Tool schema wrapping
  - Few-shot example
  - Primary parser regex

The shared pieces (system prompt, Action, _build_action, JSON/kv fallbacks)
live in ``utils.py``.  This module provides a ``ModelFormat`` interface and
a ``get_format(model_name)`` factory that auto-detects the right one.
"""

from __future__ import annotations

import json
import logging
import re
import string
import random
from abc import ABC, abstractmethod

from PIL import Image

from .utils import (
    Action,
    _build_action,
    format_system_prompt,
    SYSTEM_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


# ===================================================================
#  Base class
# ===================================================================
class ModelFormat(ABC):
    """Interface that each model family must implement."""

    @abstractmethod
    def get_tools(self) -> list[dict]:
        """Tool definitions for ``processor.apply_chat_template(tools=…)``."""

    @abstractmethod
    def build_prompt(
        self,
        frames: list[Image.Image],
        state: dict,
        num_frames: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Build (messages, tools) for the processor."""

    @abstractmethod
    def parse_tool_call(
        self,
        output_text: str,
        max_horizon: int = 30,
    ) -> Action | None:
        """Parse model output into an Action."""

    # ---- shared building blocks (subclasses can call these) ----

    def _build_user_content(
        self, frames: list[Image.Image], state: dict, latency_frames: int, num_frames: int,
    ) -> list[dict]:
        """Image list + state text — identical across formats."""
        user_content: list[dict] = [
            {"type": "image", "image": img} for img in frames
        ]
        user_content.append({
            "type": "text",
            "text": (
                f"{num_frames} frames, {state.get('ducks_flying', '?')} ducks flying, "
                f"latency {latency_frames} frames. Shoot now."
            ),
        })
        return user_content

    def _try_json_fallback(self, output_text: str, max_horizon: int) -> Action | None:
        """Shared fallback: find any JSON object with x/y keys."""
        try:
            start = output_text.index("{")
            depth, end = 0, start
            for i, ch in enumerate(output_text[start:], start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            blob = json.loads(output_text[start:end])
            if "arguments" in blob and isinstance(blob["arguments"], dict):
                blob = blob["arguments"]
            elif "arguments" in blob and isinstance(blob["arguments"], str):
                blob = json.loads(blob["arguments"])
            if "x" in blob and "y" in blob:
                return _build_action(blob["x"], blob["y"], blob.get("horizon", 0), max_horizon)
        except (ValueError, json.JSONDecodeError, KeyError, TypeError):
            pass
        return None

    def _try_kv_fallback(self, output_text: str, max_horizon: int) -> Action | None:
        """Shared fallback: x=0.3, y=0.2, horizon=5."""
        kv_match = re.search(
            r"x\s*[=:]\s*([0-9.eE+-]+).*?"
            r"y\s*[=:]\s*([0-9.eE+-]+).*?"
            r"horizon\s*[=:]\s*([0-9]+)",
            output_text, re.DOTALL | re.IGNORECASE,
        )
        if kv_match:
            return _build_action(
                kv_match.group(1), kv_match.group(2), kv_match.group(3), max_horizon,
            )
        return None


# ===================================================================
#  Mistral format
# ===================================================================
_CALL_ID_CHARS = string.ascii_letters + string.digits


def _generate_call_id() -> str:
    """Random 9-char alphanumeric ID (Mistral requirement)."""
    return "".join(random.choices(_CALL_ID_CHARS, k=9))


class MistralFormat(ModelFormat):
    """Mistral native: ``[TOOL_CALLS] [{"name":"shoot","arguments":{…},"id":"…"}]``."""

    TOOL_SCHEMA = {
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
                        "description": "Predicted horizontal position, normalised 0.0–1.0. 0.0 = left edge, 1.0 = right edge.",
                        "minimum": 0.0, "maximum": 1.0,
                    },
                    "y": {
                        "type": "number",
                        "description": "Predicted vertical position, normalised 0.0–1.0. 0.0 = top edge, 1.0 = bottom edge.",
                        "minimum": 0.0, "maximum": 1.0,
                    },
                    "horizon": {
                        "type": "integer",
                        "description": "Additional frames to wait before shooting (0-30). Total prediction = processing_latency_frames + horizon.",
                        "minimum": 0, "maximum": 30,
                    },
                },
                "required": ["x", "y", "horizon"],
            },
        },
    }

    FEW_SHOT_EXAMPLE = (
        '[TOOL_CALLS] [{"name": "shoot", "arguments": '
        '{"x": 0.45, "y": 0.25, "horizon": 8}, "id": "abc123xyz"}]'
    )

    def get_tools(self) -> list[dict]:
        return [self.TOOL_SCHEMA]

    def build_prompt(self, frames, state, num_frames=None):
        if num_frames is None:
            num_frames = len(frames)
        latency_frames = state.get("simulated_latency_frames", 6)
        system_prompt = format_system_prompt(
            num_frames=num_frames, processing_latency_frames=latency_frames,
        )
        user_content = self._build_user_content(frames, state, latency_frames, num_frames)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": (
                "Frame sequence: 4 frames. Ducks flying: 2. "
                "Latency: 6 frames. Call the shoot tool now."
            )}]},
            {"role": "assistant", "content": [{"type": "text", "text": self.FEW_SHOT_EXAMPLE}]},
            {"role": "user", "content": user_content},
        ]
        return messages, self.get_tools()

    def parse_tool_call(self, output_text, max_horizon=30):
        # --- Mistral [TOOL_CALLS] [{"name": "shoot", ...}] ---
        tc_match = re.search(r"\[TOOL_CALLS\]\s*(\[.*\])", output_text, re.DOTALL)
        if tc_match:
            try:
                calls = json.loads(tc_match.group(1))
                if isinstance(calls, list) and len(calls) > 0:
                    args = calls[0].get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    if "x" in args and "y" in args:
                        return _build_action(args["x"], args["y"], args.get("horizon", 0), max_horizon)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # --- Ministral [TOOL_CALLS]name[ARGS]{...} variant ---
        args_match = re.search(
            r"\[TOOL_CALLS\]\s*\w+\s*\[ARGS\]\s*(\{.*?\})", output_text, re.DOTALL,
        )
        if args_match:
            try:
                args = json.loads(args_match.group(1))
                if "x" in args and "y" in args:
                    return _build_action(args["x"], args["y"], args.get("horizon", 0), max_horizon)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # --- shared fallbacks ---
        action = self._try_json_fallback(output_text, max_horizon)
        if action:
            return action
        action = self._try_kv_fallback(output_text, max_horizon)
        if action:
            return action

        logger.warning("Failed to parse Mistral tool call from: %s", output_text[:200])
        return None


# ===================================================================
#  LiquidAI format
# ===================================================================
class LiquidAIFormat(ModelFormat):
    """LiquidAI Pythonic: ``<|tool_call_start|>[shoot(x=0.3, y=0.2, horizon=5)]<|tool_call_end|>``."""

    TOOL_SCHEMA = {
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
                    "description": "Predicted horizontal position, normalised 0.0-1.0. 0.0 = left edge, 1.0 = right edge.",
                    "minimum": 0.0, "maximum": 1.0,
                },
                "y": {
                    "type": "number",
                    "description": "Predicted vertical position, normalised 0.0-1.0. 0.0 = top edge, 1.0 = bottom edge.",
                    "minimum": 0.0, "maximum": 1.0,
                },
                "horizon": {
                    "type": "integer",
                    "description": "Additional frames to wait before shooting (0-30). Total prediction = processing_latency_frames + horizon.",
                    "minimum": 0, "maximum": 30,
                },
            },
            "required": ["x", "y", "horizon"],
        },
    }

    FEW_SHOT_EXAMPLE = "<|tool_call_start|>[shoot(x=0.45, y=0.25, horizon=8)]<|tool_call_end|>"

    def get_tools(self) -> list[dict]:
        return [self.TOOL_SCHEMA]

    def build_prompt(self, frames, state, num_frames=None):
        if num_frames is None:
            num_frames = len(frames)
        latency_frames = state.get("simulated_latency_frames", 5)
        system_prompt = format_system_prompt(
            num_frames=num_frames, processing_latency_frames=latency_frames,
        )
        user_content = self._build_user_content(frames, state, latency_frames, num_frames)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": (
                "Frame sequence: 4 frames. Ducks flying: 2. "
                "Latency: 5 frames. Call the shoot tool now."
            )}]},
            {"role": "assistant", "content": [{"type": "text", "text": self.FEW_SHOT_EXAMPLE}]},
            {"role": "user", "content": user_content},
        ]
        return messages, self.get_tools()

    def parse_tool_call(self, output_text, max_horizon=30):
        # --- LiquidAI <|tool_call_start|>[shoot(...)]<|tool_call_end|> ---
        liq_match = re.search(
            r"(?:<\|tool_call_start\|>|tool_call_start)"
            r".*?shoot\s*\((.*?)\)"
            r".*?(?:<\|tool_call_end\|>|tool_call_end)?",
            output_text, re.DOTALL | re.IGNORECASE,
        )
        if liq_match:
            return self._parse_kwargs(liq_match.group(1), max_horizon)

        # --- Plain pythonic shoot(...) ---
        py_match = re.search(
            r"shoot\s*\((.*?)\)", output_text, re.DOTALL | re.IGNORECASE,
        )
        if py_match:
            return self._parse_kwargs(py_match.group(1), max_horizon)

        # --- shared fallbacks ---
        action = self._try_json_fallback(output_text, max_horizon)
        if action:
            return action
        action = self._try_kv_fallback(output_text, max_horizon)
        if action:
            return action

        logger.warning("Failed to parse LiquidAI tool call from: %s", output_text[:200])
        return None

    @staticmethod
    def _parse_kwargs(args_str: str, max_horizon: int) -> Action | None:
        # Try keyword args first: shoot(x=0.5, y=0.3, horizon=8)
        vals = {}
        for match in re.finditer(r"(\w+)\s*=\s*([0-9.eE+-]+)", args_str):
            vals[match.group(1)] = match.group(2)
        if "x" in vals and "y" in vals:
            return _build_action(vals["x"], vals["y"], vals.get("horizon", "0"), max_horizon)

        # Fallback: positional args: shoot(0.5, 0.3, 8)
        nums = re.findall(r"\d+\.?\d*(?:[eE][+-]?\d+)?", args_str)
        if len(nums) >= 2:
            horizon = nums[2] if len(nums) >= 3 else "0"
            try:
                return _build_action(nums[0], nums[1], horizon, max_horizon)
            except (ValueError, TypeError):
                pass

        return None


# ===================================================================
#  Registry / factory
# ===================================================================
_FORMATS: dict[str, type[ModelFormat]] = {
    "mistral": MistralFormat,
    "liquidai": LiquidAIFormat,
}

# Model name prefixes → format key
_MODEL_PREFIX_MAP: list[tuple[str, str]] = [
    ("mistralai/", "mistral"),
    ("liquidai/", "liquidai"),
    ("lfm", "liquidai"),
]


def get_format(model_name: str) -> ModelFormat:
    """Auto-detect and return the right ``ModelFormat`` for *model_name*.

    Raises ``ValueError`` if the model family cannot be determined.
    """
    lower = model_name.lower()
    for prefix, key in _MODEL_PREFIX_MAP:
        if lower.startswith(prefix):
            logger.info("Detected model family '%s' for %s", key, model_name)
            return _FORMATS[key]()

    raise ValueError(
        f"Cannot determine model format for '{model_name}'. "
        f"Known prefixes: {[p for p, _ in _MODEL_PREFIX_MAP]}"
    )


def get_format_by_name(name: str) -> ModelFormat:
    """Look up a format by explicit name ('mistral' or 'liquidai')."""
    if name not in _FORMATS:
        raise ValueError(f"Unknown format '{name}'. Known: {list(_FORMATS.keys())}")
    return _FORMATS[name]()
