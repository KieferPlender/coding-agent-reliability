"""The agent loop: one model, one task, one sandbox, with limits and a full trace."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from car.models import Model, Usage
from car.sandbox import Sandbox
from car.tasks import Task
from car.tools import TOOL_SPECS, ToolBox, ToolError

SYSTEM_PROMPT = """\
You are an experienced software engineer fixing a bug in a Python repository.

The project's test suite currently fails because of a bug in the source code.
Find the bug and fix it with the smallest correct change.

Rules:
- Fix the source code. Do not modify, delete or skip tests.
- Use the tools to explore: run_tests, list_files, read_file, search, edit_file.
- Run the tests after your change. When they pass, call finish.
"""

TASK_PROMPT = """\
Repository: {target}
The test suite fails. Run the tests to see which ones, find the bug, and fix it.
"""

NUDGE = "Continue using the tools. Run the tests to check your fix, and call finish once they pass."


@dataclass
class AgentConfig:
    max_steps: int = 30  # model responses per attempt
    max_cost_usd: float = 1.00  # per attempt; the harness has its own overall budget
    max_seconds: float = 900
    test_output_chars: int = 6000
    max_nudges: int = 1  # replies without tool calls before giving up


@dataclass
class AgentResult:
    stop: str  # finished, max_steps, max_cost, max_time, no_tool_calls, refusal, model_error
    steps: int
    usage: Usage
    cost_usd: float
    seconds: float
    model_seconds: float
    tool_calls: int
    tool_errors: int
    malformed_tool_calls: int
    test_runs: int
    edited_files: list[str]
    blocked_test_edits: int
    refusal_category: str | None = None
    error: str | None = None
    tool_counts: dict[str, int] = field(default_factory=dict)


class Trace:
    """Append-only JSONL record of everything the agent saw and did."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.start = time.monotonic()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._f = path.open("w", encoding="utf-8")

    def event(self, kind: str, **data: Any) -> None:
        if self.path:
            record = {"t": round(time.monotonic() - self.start, 3), "event": kind, **data}
            self._f.write(json.dumps(record, default=str) + "\n")
            self._f.flush()

    def close(self) -> None:
        if self.path:
            self._f.close()


def run_agent(
    task: Task,
    sandbox: Sandbox,
    model: Model,
    test_command: list[str],
    config: AgentConfig | None = None,
    trace_path: Path | None = None,
) -> AgentResult:
    config = config or AgentConfig()
    tools = ToolBox(sandbox, test_command, test_output_chars=config.test_output_chars)
    trace = Trace(trace_path)
    task_text = TASK_PROMPT.format(target=task.target)
    trace.event("start", task=task.id, model=model.name, system=SYSTEM_PROMPT, prompt=task_text,
                config=asdict(config))  # fmt: skip
    conversation = model.start(SYSTEM_PROMPT, TOOL_SPECS, task_text)

    usage = Usage()
    cost = model_seconds = 0.0
    steps = tool_calls = tool_errors = malformed = nudges = 0
    counts: dict[str, int] = {}
    stop, refusal, error = "max_steps", None, None
    started = time.monotonic()

    try:
        while steps < config.max_steps:
            if cost >= config.max_cost_usd:
                stop = "max_cost"
                break
            if time.monotonic() - started >= config.max_seconds:
                stop = "max_time"
                break
            response = conversation.next()
            steps += 1
            usage.add(response.usage)
            cost += response.cost_usd
            model_seconds += response.latency_s
            trace.event("model", step=steps, stop_reason=response.stop_reason, text=response.text,
                        tool_calls=[asdict(c) for c in response.tool_calls],
                        usage=asdict(response.usage), cost_usd=response.cost_usd,
                        latency_s=round(response.latency_s, 3), error=response.error)  # fmt: skip
            if response.stop_reason == "error":
                stop, error = "model_error", response.error
                break
            if response.stop_reason == "refusal":
                stop, refusal = "refusal", response.refusal_category
                break
            if not response.tool_calls:
                if nudges >= config.max_nudges:
                    stop = "no_tool_calls"
                    break
                nudges += 1
                conversation.add_user_message(NUDGE)
                trace.event("nudge", text=NUDGE)
                continue

            results: list[tuple[str, str, bool]] = []
            for call in response.tool_calls:
                tool_calls += 1
                counts[call.name] = counts.get(call.name, 0) + 1
                if call.parse_error:
                    malformed += 1
                    output, is_error = call.parse_error, True
                else:
                    try:
                        output, is_error = tools.execute(call.name, dict(call.arguments)), False
                    except ToolError as exc:
                        output, is_error = str(exc), True
                    except (OSError, UnicodeDecodeError) as exc:
                        output, is_error = f"{type(exc).__name__}: {exc}", True
                tool_errors += is_error
                trace.event("tool", id=call.id, name=call.name, arguments=call.arguments,
                            output=output, is_error=is_error)  # fmt: skip
                results.append((call.id, output, is_error))
            conversation.add_tool_results(results)
            if tools.finished:
                stop = "finished"
                break
    finally:
        result = AgentResult(
            stop=stop,
            steps=steps,
            usage=usage,
            cost_usd=round(cost, 6),
            seconds=round(time.monotonic() - started, 3),
            model_seconds=round(model_seconds, 3),
            tool_calls=tool_calls,
            tool_errors=tool_errors,
            malformed_tool_calls=malformed,
            test_runs=tools.test_runs,
            edited_files=sorted(set(tools.edits)),
            blocked_test_edits=tools.blocked_test_edits,
            refusal_category=refusal,
            error=error,
            tool_counts=counts,
        )
        trace.event("end", result=asdict(result))
        trace.close()
    return result
