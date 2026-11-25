"""Model backends behind one conversation interface.

The agent loop only talks to `Conversation`: get the next response, add tool
results, add a user message. Each backend keeps its own wire format, so every
model runs through exactly the same agent logic, tools and prompts.

- `ClaudeModel`: Claude via the Anthropic Messages API (official SDK).
- `OpenAICompatModel`: any OpenAI-compatible chat endpoint, which is how local
  servers (Ollama, MLX, llama.cpp) expose models.
- `ScriptedModel`: replays a fixed list of steps, for tests without API calls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Set when the model's arguments were not valid JSON; the call is not run and
    # the error goes back to the model. A common failure mode for local models.
    parse_error: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_tokens += other.cache_read_tokens
        self.cache_write_tokens += other.cache_write_tokens


@dataclass
class ModelResponse:
    text: str
    tool_calls: list[ToolCall]
    stop_reason: str  # "tool_use", "end_turn", "max_tokens", "refusal", "error"
    usage: Usage
    cost_usd: float
    latency_s: float
    refusal_category: str | None = None
    error: str | None = None


class Conversation(Protocol):
    def next(self) -> ModelResponse: ...

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        """(tool call id, content, is_error) for every call in the last response."""
        ...

    def add_user_message(self, text: str) -> None: ...


class Model(Protocol):
    name: str  # stable label used in results, e.g. "claude-haiku-5-5"

    def start(self, system: str, tools: list[ToolSpec], task: str) -> Conversation: ...


# --------------------------------------------------------------------------- Claude

# USD per million tokens: (input, output, cache read, cache write at 5-minute TTL).
# From Anthropic's published pricing as of November 2026; verify before quoting
# results. Claude Haiku 5.5 bills prompts above 100K tokens at $0.50 / $2.50.
CLAUDE_PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
HAIKU_LONG_PROMPT = (100_000, 0.50, 2.50)


def claude_cost(model: str, usage: Usage) -> float:
    if model not in CLAUDE_PRICES:
        raise ValueError(f"no price for {model!r}; add it to CLAUDE_PRICES")
    price_in, price_out, price_read, price_write = CLAUDE_PRICES[model]
    prompt = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
    if model == "claude-haiku-5-5" and prompt > HAIKU_LONG_PROMPT[0]:
        scale = HAIKU_LONG_PROMPT[1] / price_in
        price_in, price_out = HAIKU_LONG_PROMPT[1], HAIKU_LONG_PROMPT[2]
        price_read, price_write = price_read * scale, price_write * scale
    return (
        usage.input_tokens * price_in
        + usage.output_tokens * price_out
        + usage.cache_read_tokens * price_read
        + usage.cache_write_tokens * price_write
    ) / 1_000_000


class ClaudeModel:
    """Claude through the Messages API, with a loop this project controls.

    Deliberately not enabled: server-side refusal fallbacks. A fallback would
    silently answer with a different model, which would corrupt a comparison
    between models; refusals are recorded as their own outcome instead.
    """

    def __init__(
        self,
        model: str,
        effort: str = "medium",
        max_tokens: int = 16000,
        client: Any | None = None,
    ) -> None:
        if client is None:
            import anthropic  # optional dependency: pip install "coding-agent-reliability[claude]"

            # Reads the API key from the environment (ANTHROPIC_API_KEY or an
            # `ant auth login` profile). Never pass keys in code.
            client = anthropic.Anthropic(max_retries=5)
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"{model}@{effort}"
        claude_cost(model, Usage())  # fail fast on unknown models

    def start(self, system: str, tools: list[ToolSpec], task: str) -> ClaudeConversation:
        return ClaudeConversation(self, system, tools, task)


class ClaudeConversation:
    def __init__(self, model: ClaudeModel, system: str, tools: list[ToolSpec], task: str):
        self.m = model
        # Static prefix: explicit cache breakpoint on the system prompt; the growing
        # conversation is cached via top-level automatic caching.
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        # Arguments are validated by the tool layer for every backend alike, so
        # optional parameters are allowed and strict mode is not used.
        self.tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters}
            for t in tools
        ]
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": task}]

    def next(self) -> ModelResponse:
        start = time.monotonic()
        try:
            response = self.m.client.messages.create(
                model=self.m.model,
                max_tokens=self.m.max_tokens,
                system=self.system,
                tools=self.tools,
                messages=self.messages,
                output_config={"effort": self.m.effort},
                cache_control={"type": "ephemeral"},
            )
        except Exception as exc:  # the SDK already retried transient errors
            return ModelResponse("", [], "error", Usage(), 0.0, time.monotonic() - start,
                                 error=f"{type(exc).__name__}: {exc}")  # fmt: skip
        latency = time.monotonic() - start
        u = response.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        cost = claude_cost(self.m.model, usage)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            return ModelResponse("", [], "refusal", usage, cost, latency,
                                 refusal_category=getattr(details, "category", None))  # fmt: skip
        # Pass the content back unchanged on the next request: it includes
        # thinking blocks, which must not be edited or dropped.
        self.messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [
            ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
            for b in response.content
            if b.type == "tool_use"
        ]
        return ModelResponse(text, calls, response.stop_reason or "end_turn", usage, cost, latency)

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        # All results for one assistant turn go back in a single user message.
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": cid,
                        "content": content,
                        **({"is_error": True} if is_error else {}),
                    }  # fmt: skip
                    for cid, content, is_error in results
                ],
            }
        )

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# ------------------------------------------------------- OpenAI-compatible (local)


class OpenAICompatModel:
    """A chat-completions endpoint, e.g. Ollama at http://localhost:11434/v1.

    Uses only the standard library. Cost is recorded as zero; local runs are
    compared on time instead.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout_s: float = 600,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.api_key = api_key  # most local servers need none
        self.name = f"local:{model}"

    def start(self, system: str, tools: list[ToolSpec], task: str) -> OpenAICompatConversation:
        return OpenAICompatConversation(self, system, tools, task)

    def post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            return json.loads(resp.read())


class OpenAICompatConversation:
    def __init__(self, model: OpenAICompatModel, system: str, tools: list[ToolSpec], task: str):
        self.m = model
        self.tools = [
            {"type": "function",
             "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]  # fmt: skip
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

    def next(self) -> ModelResponse:
        start = time.monotonic()
        try:
            data = self.m.post(
                {
                    "model": self.m.model,
                    "messages": self.messages,
                    "tools": self.tools,
                    "temperature": self.m.temperature,
                    "max_tokens": self.m.max_tokens,
                }
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return ModelResponse("", [], "error", Usage(), 0.0, time.monotonic() - start,
                                 error=f"{type(exc).__name__}: {exc}")  # fmt: skip
        latency = time.monotonic() - start
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        u = data.get("usage") or {}
        usage = Usage(u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0)
        calls: list[ToolCall] = []
        for i, raw in enumerate(message.get("tool_calls") or []):
            fn = raw.get("function") or {}
            args_text = fn.get("arguments") or "{}"
            call = ToolCall(
                id=raw.get("id") or f"call_{i}", name=fn.get("name") or "", arguments={}
            )
            try:
                parsed = json.loads(args_text) if isinstance(args_text, str) else args_text
                if not isinstance(parsed, dict):
                    raise ValueError("arguments must be a JSON object")
                call.arguments = parsed
            except (json.JSONDecodeError, ValueError) as exc:
                call.parse_error = f"invalid tool arguments ({exc}): {str(args_text)[:200]}"
            calls.append(call)
        # Keep the assistant turn exactly as the server sent it.
        self.messages.append(
            {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}
            | {"role": "assistant"}
        )
        finish = choice.get("finish_reason")
        stop = "tool_use" if calls else ("max_tokens" if finish == "length" else "end_turn")
        return ModelResponse(message.get("content") or "", calls, stop, usage, 0.0, latency)

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        for cid, content, is_error in results:
            self.messages.append(
                {"role": "tool", "tool_call_id": cid,
                 "content": f"ERROR: {content}" if is_error else content}
            )  # fmt: skip

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


# ------------------------------------------------------------------- scripted fake


@dataclass
class ScriptedModel:
    """Returns pre-written steps in order: for testing the harness for free.

    Each step is a list of (tool name, arguments) pairs, or a string for a final
    text answer without tool calls.
    """

    steps: list[list[tuple[str, dict[str, Any]]] | str]
    name: str = "scripted"
    seen: list[list[tuple[str, str, bool]]] = field(default_factory=list)

    def start(self, system: str, tools: list[ToolSpec], task: str) -> ScriptedConversation:
        return ScriptedConversation(self, list(self.steps))


class ScriptedConversation:
    def __init__(self, model: ScriptedModel, steps: list) -> None:
        self.model = model
        self.steps = steps
        self.counter = 0

    def next(self) -> ModelResponse:
        usage = Usage(input_tokens=100, output_tokens=20)
        if not self.steps:
            return ModelResponse("done", [], "end_turn", usage, 0.0, 0.0)
        step = self.steps.pop(0)
        if isinstance(step, str):
            return ModelResponse(step, [], "end_turn", usage, 0.0, 0.0)
        calls = []
        for name, args in step:
            self.counter += 1
            calls.append(ToolCall(id=f"t{self.counter}", name=name, arguments=args))
        return ModelResponse("", calls, "tool_use", usage, 0.0, 0.0)

    def add_tool_results(self, results: list[tuple[str, str, bool]]) -> None:
        self.model.seen.append(results)

    def add_user_message(self, text: str) -> None:
        self.model.seen.append([("user", text, False)])
