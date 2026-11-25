"""Turn candidate mutations into validated bug-fix tasks.

A candidate becomes a task only if the clean repository passes its tests and the
mutated one fails at least one test. That rules out syntax-only changes and
"equivalent" mutations that change nothing observable, and gives every task an
automatic definition of success: the full test suite passes again.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from car.mutate import Mutation, apply_mutation, find_mutations
from car.sandbox import CommandResult, LocalSandbox, Sandbox
from car.targets import Target

# pytest's short test summary lines, enabled with -rfE.
# Matches `FAILED <id>`, `ERROR <id>` and subtest failures, which pytest reports as
# `SUBFAILED(<params>) <id>`; params may contain spaces, so the id is the first
# token after them that looks like a test path.
_SUMMARY_RE = re.compile(r"^(?:FAILED|ERROR|SUBFAILED(?:\(.*?\))?) (?P<id>\S+)", re.MULTILINE)
# Terminal colour codes: some projects force `--color=yes` in their pytest config.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

SandboxFactory = Callable[[Path], Sandbox]


@dataclass
class Task:
    id: str
    target: str
    commit: str | None
    mutation: Mutation
    diff: str  # the injected bug, as a unified diff against the clean repository
    failing_tests: list[str]
    test_seconds: float

    @property
    def operator(self) -> str:
        return self.mutation.operator

    def to_json(self) -> str:
        data = asdict(self)
        data["operator"] = self.operator
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> Task:
        data = json.loads(line)
        data.pop("operator", None)
        data["mutation"] = Mutation(**data["mutation"])
        return cls(**data)


@dataclass
class GenerationReport:
    candidates: int = 0
    tested: int = 0
    accepted: int = 0
    survived: int = 0  # tests still pass: the mutation is invisible to the suite
    timed_out: int = 0
    # Tests failed but no failing test could be identified from the output. Never
    # counted as "survived": a parsing problem must not silently discard bugs.
    unclear: int = 0
    baseline_failures: list[str] = field(default_factory=list)
    # One entry per unclear candidate, so the cause can be diagnosed afterwards.
    unclear_details: list[dict] = field(default_factory=list)


def failing_tests(result: CommandResult) -> set[str]:
    """Test ids reported as failed or errored by pytest (`-rfE`)."""
    output = _ANSI_RE.sub("", result.stdout)
    return {match.group("id") for match in _SUMMARY_RE.finditer(output)}


def pytest_args(target: Target) -> list[str]:
    """The target's test command plus flags that make the output parseable."""
    return [
        *target.test_command,
        "-q", "-rfE", "--tb=no", "--color=no", "-p", "no:cacheprovider",
    ]  # fmt: skip


def task_id(target: Target, mutation: Mutation) -> str:
    digest = hashlib.sha256(f"{target.name}|{target.commit}|{mutation.id}".encode()).hexdigest()
    return f"{target.name}-{digest[:10]}"


def make_diff(path: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def iter_candidates(target: Target, repo: Path) -> Iterator[Mutation]:
    for file in sorted(repo.rglob("*.py")):
        rel = file.relative_to(repo).as_posix()
        if target.is_mutable(rel):
            yield from find_mutations(file.read_text(encoding="utf-8"), rel)


def _round_robin(
    groups: dict[str, list[Mutation]], rng: random.Random, shuffle_groups: bool = True
) -> list[Mutation]:
    """Interleave groups one item at a time, visiting groups in a shuffled order each round."""
    if shuffle_groups:
        for group in groups.values():
            rng.shuffle(group)
    ordered: list[Mutation] = []
    keys = sorted(groups)
    while any(groups[k] for k in keys):
        rng.shuffle(keys)
        for k in keys:
            if groups[k]:
                ordered.append(groups[k].pop())
    return ordered


def sample_candidates(candidates: list[Mutation], limit: int | None, seed: int) -> list[Mutation]:
    """Deterministic random order, balanced over bug types and then over files.

    Without balancing, "change a number by one" makes up ~40% of candidates and
    one large module can dominate a target.
    """
    rng = random.Random(seed)
    by_operator: dict[str, list[Mutation]] = {}
    for m in candidates:
        by_operator.setdefault(m.operator, []).append(m)
    spread: dict[str, list[Mutation]] = {}
    for operator in sorted(by_operator):
        by_file: dict[str, list[Mutation]] = {}
        for m in by_operator[operator]:
            by_file.setdefault(m.path, []).append(m)
        spread[operator] = _round_robin(by_file, rng)[::-1]  # reversed: pop() takes from the end
    ordered = _round_robin(spread, rng, shuffle_groups=False)
    return ordered if limit is None else ordered[:limit]


def generate_tasks(
    target: Target,
    repo: Path,
    make_sandbox: SandboxFactory | None = None,
    max_candidates: int | None = 200,
    seed: int = 0,
    workers: int = 4,
    timeout_s: float = 300,
    on_task: Callable[[Task], None] | None = None,
) -> tuple[list[Task], GenerationReport]:
    """Validate sampled mutations of `repo` and return the ones that make tests fail."""
    make_sandbox = make_sandbox or (lambda src: LocalSandbox(src, pythonpath=target.pythonpath))
    report = GenerationReport()
    command = pytest_args(target)

    # The clean repository must pass; anything failing twice in a row is excluded
    # from the comparison so flaky or environment-specific tests can't create tasks.
    baseline_failed: set[str] = set()
    with _closing(make_sandbox(repo)) as box:
        for _ in range(2):
            result = box.run(command, timeout_s=timeout_s)
            if result.timed_out:
                raise RuntimeError(f"{target.name}: baseline test run timed out")
            failed = failing_tests(result)
            if result.exit_code != 0 and not failed:
                raise RuntimeError(
                    f"{target.name}: baseline tests could not run "
                    f"(exit {result.exit_code}):\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
                )
            baseline_failed |= failed
    report.baseline_failures = sorted(baseline_failed)

    candidates = list(iter_candidates(target, repo))
    report.candidates = len(candidates)
    sampled = sample_candidates(candidates, max_candidates, seed)

    def check(mutation: Mutation) -> tuple[str, Task | dict | None]:
        source = (repo / mutation.path).read_text(encoding="utf-8")
        mutated = apply_mutation(source, mutation)
        with _closing(make_sandbox(repo)) as box:
            (box.workdir / mutation.path).write_text(mutated, encoding="utf-8")
            result = box.run(command, timeout_s=timeout_s)
        if result.timed_out:
            return "timed_out", None
        failed = failing_tests(result)
        if result.exit_code != 0 and not failed:
            return "unclear", {
                "mutation": mutation.id,
                "exit_code": result.exit_code,
                "stdout_tail": _ANSI_RE.sub("", result.stdout)[-1500:],
                "stderr_tail": result.stderr[-1500:],
            }
        newly_failing = failed - baseline_failed
        if result.exit_code == 0 or not newly_failing:
            return "survived", None
        return "accepted", Task(
            id=task_id(target, mutation),
            target=target.name,
            commit=target.commit,
            mutation=mutation,
            diff=make_diff(mutation.path, source, mutated),
            failing_tests=sorted(newly_failing),
            test_seconds=round(result.duration_s, 2),
        )

    # Counting happens here, in the calling thread, not inside the workers.
    tasks: list[Task] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for status, task in pool.map(check, sampled):
            report.tested += 1
            if status == "timed_out":
                report.timed_out += 1
            elif status == "survived":
                report.survived += 1
            elif status == "unclear":
                report.unclear += 1
                report.unclear_details.append(task)  # type: ignore[arg-type]
                continue
            if task is not None:
                tasks.append(task)
                if on_task:
                    on_task(task)
    report.accepted = len(tasks)
    return tasks, report


class _closing:
    """Context manager for any sandbox, whether or not it implements one."""

    def __init__(self, box: Sandbox) -> None:
        self.box = box

    def __enter__(self) -> Sandbox:
        return self.box

    def __exit__(self, *exc: object) -> None:
        self.box.close()


def write_tasks(tasks: list[Task], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(task.to_json() + "\n")


def read_tasks(path: Path) -> list[Task]:
    with path.open(encoding="utf-8") as f:
        return [Task.from_json(line) for line in f if line.strip()]


def materialize(task: Task, repo: Path, make_sandbox: SandboxFactory) -> Sandbox:
    """A fresh sandbox containing `repo` with the task's bug applied."""
    box = make_sandbox(repo)
    file = box.workdir / task.mutation.path
    file.write_text(
        apply_mutation(file.read_text(encoding="utf-8"), task.mutation), encoding="utf-8"
    )
    return box
