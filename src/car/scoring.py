"""Decide whether an attempt fixed the bug, independently of what the agent claims."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from car.mutate import apply_mutation
from car.tasks import SandboxFactory, Task, failing_tests, make_diff
from car.tools import is_test_file

_IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox"}


@dataclass
class Score:
    solved: bool  # full suite passes and no test file was changed
    tests_passed: bool
    still_failing: list[str]
    changed_files: list[str]
    changed_test_files: list[str]
    exact_revert: bool  # the change is exactly the original code
    touched_bug_file: bool
    patch: str  # the agent's change relative to the buggy code
    eval_seconds: float


def _repo_files(root: Path) -> list[str]:
    files = []
    for path in root.rglob("*"):
        rel = path.relative_to(root)
        skipped = any(part in _IGNORED_DIRS for part in rel.parts) or path.suffix == ".pyc"
        if path.is_file() and not skipped:
            files.append(rel.as_posix())
    return sorted(files)


def score_attempt(
    task: Task,
    repo: Path,
    workdir: Path,
    make_sandbox: SandboxFactory,
    test_command: list[str],
    timeout_s: float = 600,
) -> Score:
    """Compare the agent's workspace with the buggy code, then run the full suite.

    The suite runs in a fresh sandbox copied from the agent's workspace, so nothing
    the agent left running or cached can influence the verdict.
    """
    changed, patches = [], []
    for rel in _repo_files(repo):
        original = (repo / rel).read_bytes()
        buggy = original
        if rel == task.mutation.path:
            buggy = apply_mutation(original.decode("utf-8"), task.mutation).encode("utf-8")
        agent_file = workdir / rel
        current = agent_file.read_bytes() if agent_file.exists() else b""
        if current != buggy:
            changed.append(rel)
            try:
                patches.append(make_diff(rel, buggy.decode("utf-8"), current.decode("utf-8")))
            except UnicodeDecodeError:
                patches.append(f"Binary file {rel} changed\n")

    changed_tests = [p for p in changed if is_test_file(p)]
    original_bug_file = (repo / task.mutation.path).read_bytes()
    exact_revert = changed == [task.mutation.path] and (
        (workdir / task.mutation.path).read_bytes() == original_bug_file
    )

    box = make_sandbox(workdir)
    try:
        result = box.run(
            [*test_command, "-q", "-rfE", "--tb=no", "--color=no", "-p", "no:cacheprovider"],
            timeout_s=timeout_s,
        )
    finally:
        box.close()
    return Score(
        solved=result.ok and not changed_tests,
        tests_passed=result.ok,
        still_failing=sorted(failing_tests(result)),
        changed_files=changed,
        changed_test_files=changed_tests,
        exact_revert=exact_revert,
        touched_bug_file=task.mutation.path in changed,
        patch="".join(patches),
        eval_seconds=round(result.duration_s, 3),
    )
