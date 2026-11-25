"""The agent's tools: run the tests, list, read and search files, edit, finish.

File tools work directly on the sandbox's copy of the repository; only
`run_tests` executes code, and it does so inside the sandbox. There is
deliberately no general shell tool: every action the agent takes is typed,
checked and logged, which is what makes failures classifiable afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from car.models import ToolSpec
from car.sandbox import Sandbox

MAX_READ_LINES = 400
MAX_SEARCH_RESULTS = 50
MAX_LIST_ENTRIES = 300
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".tox", ".venv", "node_modules"}

TOOL_SPECS = [
    ToolSpec(
        "run_tests",
        "Run the project's test suite and return the result: which tests failed "
        "and short tracebacks. Takes no arguments.",
        {"type": "object", "properties": {}},
    ),
    ToolSpec(
        "list_files",
        "List files and directories under a path in the repository (recursive, "
        "skips caches). Use '.' for the repository root.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path."}},
            "required": ["path"],
        },
    ),
    ToolSpec(
        "read_file",
        f"Read a file with line numbers. Returns at most {MAX_READ_LINES} lines per "
        "call; use start_line to read further.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "1-based, default 1."},
            },
            "required": ["path"],
        },
    ),
    ToolSpec(
        "search",
        "Search file contents for a regular expression (Python syntax). Returns "
        "matching lines as path:line: text.",
        {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Directory or file; default '.'."},
            },
            "required": ["pattern"],
        },
    ),
    ToolSpec(
        "edit_file",
        "Replace one exact occurrence of old_text with new_text in a file. old_text "
        "must match the file exactly, including whitespace, and occur exactly once. "
        "Test files cannot be edited.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    ),
    ToolSpec(
        "finish",
        "Call when the bug is fixed and the tests pass. Ends the task.",
        {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "What you changed."}},
            "required": ["summary"],
        },
    ),
]


class ToolError(Exception):
    """An error reported back to the model, which can then correct itself."""


def is_test_file(rel_path: str) -> bool:
    parts = rel_path.split("/")
    name = parts[-1]
    return (
        any(p in ("test", "tests", "testing") for p in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


@dataclass
class ToolBox:
    """Executes tool calls against one sandbox and records what happened."""

    sandbox: Sandbox
    test_command: list[str]
    test_output_chars: int = 6000  # tail of pytest output shown to the model
    test_timeout_s: float = 300
    finished: bool = False
    finish_summary: str = ""
    edits: list[str] = field(default_factory=list)  # paths edited successfully
    blocked_test_edits: int = 0
    test_runs: int = 0
    last_tests_passed: bool | None = None

    def _resolve(self, rel: str) -> Path:
        root = self.sandbox.workdir.resolve()
        path = (root / (rel or ".")).resolve()
        if path != root and root not in path.parents:
            raise ToolError(f"path is outside the repository: {rel}")
        return path

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.sandbox.workdir.resolve()).as_posix() or "."

    def execute(self, name: str, args: dict[str, Any]) -> str:
        method = getattr(self, f"tool_{name}", None)
        if method is None:
            raise ToolError(f"unknown tool {name!r}; available: {[t.name for t in TOOL_SPECS]}")
        return method(**_check_args(name, args))

    # -- tools ---------------------------------------------------------------

    def tool_run_tests(self) -> str:
        self.test_runs += 1
        result = self.sandbox.run(
            [*self.test_command, "-q", "--tb=short", "--color=no", "-p", "no:cacheprovider"],
            timeout_s=self.test_timeout_s,
        )
        self.last_tests_passed = result.ok
        if result.timed_out:
            return f"Tests timed out after {self.test_timeout_s:.0f}s."
        output = (result.stdout + result.stderr).strip()
        if len(output) > self.test_output_chars:
            output = "[... earlier output truncated ...]\n" + output[-self.test_output_chars :]
        status = "ALL TESTS PASSED" if result.ok else f"TESTS FAILED (exit {result.exit_code})"
        return f"{status}\n{output}"

    def tool_list_files(self, path: str) -> str:
        base = self._resolve(path)
        if not base.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = []
        for p in sorted(base.rglob("*")):
            if any(part in SKIP_DIRS for part in p.relative_to(base).parts):
                continue
            entries.append(self._rel(p) + ("/" if p.is_dir() else ""))
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"[... truncated at {MAX_LIST_ENTRIES} entries ...]")
                break
        return "\n".join(entries) or "(empty)"

    def tool_read_file(self, path: str, start_line: int = 1) -> str:
        file = self._resolve(path)
        if not file.is_file():
            raise ToolError(f"no such file: {path}")
        lines = file.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(1, int(start_line))
        if start > len(lines) and lines:
            raise ToolError(f"start_line {start} is past the end ({len(lines)} lines)")
        chunk = lines[start - 1 : start - 1 + MAX_READ_LINES]
        body = "\n".join(f"{start + i:>5}  {line}" for i, line in enumerate(chunk))
        end = start + len(chunk) - 1
        if end < len(lines):
            body += f"\n[... {len(lines) - end} more lines; call again with start_line={end + 1}]"
        return body or "(empty file)"

    def tool_search(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        base = self._resolve(path)
        files = [base] if base.is_file() else sorted(base.rglob("*.py"))
        hits = []
        for file in files:
            if any(part in SKIP_DIRS for part in file.parts):
                continue
            text = file.read_text(encoding="utf-8", errors="replace")
            for n, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{self._rel(file)}:{n}: {line.strip()[:200]}")
                    if len(hits) >= MAX_SEARCH_RESULTS:
                        return "\n".join(hits) + f"\n[... stopped at {MAX_SEARCH_RESULTS} matches]"
        return "\n".join(hits) or "No matches."

    def tool_edit_file(self, path: str, old_text: str, new_text: str) -> str:
        file = self._resolve(path)
        rel = self._rel(file)
        if is_test_file(rel):
            self.blocked_test_edits += 1
            raise ToolError("test files cannot be edited; fix the source code instead")
        if not file.is_file():
            raise ToolError(f"no such file: {path}")
        text = file.read_text(encoding="utf-8")
        count = text.count(old_text)
        if not old_text or count == 0:
            raise ToolError("old_text not found in the file; read the file and copy it exactly")
        if count > 1:
            raise ToolError(
                f"old_text occurs {count} times; include more context to make it unique"
            )
        file.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        self.edits.append(rel)
        return f"Edited {rel}."

    def tool_finish(self, summary: str) -> str:
        self.finished = True
        self.finish_summary = summary
        return "Finished."


def _check_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate arguments against the tool's schema (types, required, unknown keys)."""
    spec = next(t for t in TOOL_SPECS if t.name == name)
    props = spec.parameters.get("properties", {})
    unknown = set(args) - set(props)
    if unknown:
        raise ToolError(f"unknown argument(s) for {name}: {sorted(unknown)}")
    missing = [k for k in spec.parameters.get("required", []) if k not in args]
    if missing:
        raise ToolError(f"missing argument(s) for {name}: {missing}")
    types = {"string": str, "integer": int}
    for key, value in args.items():
        expected = types.get(props[key].get("type", ""))
        if expected is int and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            args[key] = int(value)  # small models often quote numbers
        elif expected and not isinstance(args[key], expected) or isinstance(args[key], bool):
            raise ToolError(f"argument {key!r} of {name} must be {props[key]['type']}")
    return args
