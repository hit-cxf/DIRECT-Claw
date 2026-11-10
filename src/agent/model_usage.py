"""Write per-attempt DashScope usage without persisting prompts or responses."""

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4


def _tokens(usage, field):
    value = usage.get(field)
    return value if isinstance(value, int) and value >= 0 else None


def _estimated_cost(model, input_tokens, output_tokens, cached_tokens):
    if model != "qwen3.7-plus" or input_tokens is None or output_tokens is None:
        return None
    # China (Beijing) public on-demand rates, CNY per million tokens.
    if input_tokens <= 256_000:
        input_rate, cached_rate, output_rate = 2.0, 0.4, 8.0
    elif input_tokens <= 1_000_000:
        input_rate, cached_rate, output_rate = 6.0, 1.2, 24.0
    else:
        return None
    cached = cached_tokens or 0
    if cached > input_tokens:
        return None
    return ((input_tokens - cached) * input_rate + cached * cached_rate
            + output_tokens * output_rate) / 1_000_000


class ModelUsageLog:
    def __init__(self, model):
        self.model = model
        self.path = os.environ.get("DIRECT_MODEL_USAGE_PATH")
        self.call_id = uuid4().hex if self.path else None

    def write(self, attempt, event, *, response=None, error=None):
        if not self.path:
            return
        usage = getattr(response, "usage", None) or {}
        input_tokens = _tokens(usage, "input_tokens")
        output_tokens = _tokens(usage, "output_tokens")
        total = _tokens(usage, "total_tokens")
        if total is None and input_tokens is not None and output_tokens is not None:
            total = input_tokens + output_tokens
        details = usage.get("prompt_tokens_details") or {}
        cached = _tokens(details, "cached_tokens")
        if cached is None:
            cached = _tokens(usage, "cached_tokens")
        row = {
            "timestamp": datetime.now().astimezone().isoformat(),
            "call_id": self.call_id,
            "attempt": attempt,
            "event": event,
            "model": self.model,
            "provider_request_id": getattr(response, "request_id", None),
            "http_status": getattr(response, "status_code", None),
            "error_type": type(error).__name__ if error is not None else None,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
            "cached_input_tokens": cached,
            "estimated_cost_cny": _estimated_cost(self.model, input_tokens, output_tokens, cached),
        }
        with Path(self.path).open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
