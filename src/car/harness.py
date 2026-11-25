"""Run many attempts (tasks x models x repeats) with a budget, resumably.

Each attempt gets a fresh sandbox with the task's bug applied, runs the agent,
scores the result in a second fresh sandbox, and appends one JSON line to
`results.jsonl`. Traces go to `traces/<model>/<task>-<attempt>.jsonl`.
Re-running the same command skips attempts that already have a result.
"""

from __future__ import annotations

import json
import re
import threading
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from car.agent import AgentConfig, run_agent
from car.models import Model
from car.scoring import score_attempt
from car.targets import Target
from car.tasks import SandboxFactory, Task, materialize


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_suite(
    tasks: list[Task],
    model: Model,
    targets: dict[str, Target],
    repos: dict[str, Path],
    factories: dict[str, SandboxFactory],
    out_dir: Path,
    attempts: int = 3,
    budget_usd: float = 5.0,
    workers: int = 2,
    config: AgentConfig | None = None,
    on_result: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run every task `attempts` times. Stops starting new attempts once the
    budget is spent; attempts already running finish, so the total can exceed
    the budget by at most `workers` x `config.max_cost_usd`."""
    config = config or AgentConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    existing = load_results(results_path)
    done = {(r["task"], r["model"], r["attempt"]) for r in existing}
    spent = sum(r["cost_usd"] for r in existing if r["model"] == model.name)

    jobs = [
        (task, attempt)
        for attempt in range(1, attempts + 1)  # attempt 1 of every task first
        for task in tasks
        if (task.id, model.name, attempt) not in done
    ]
    lock = threading.Lock()
    new_results: list[dict] = []

    skipped = 0

    def run_one(job: tuple[Task, int]) -> None:
        try:
            attempt_one(job)
        except Exception:  # an infrastructure failure, not a model failure
            task, attempt = job
            with lock, (out_dir / "errors.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"task": task.id, "model": model.name, "attempt": attempt,
                                    "error": traceback.format_exc()}) + "\n")  # fmt: skip

    def attempt_one(job: tuple[Task, int]) -> None:
        nonlocal spent, skipped
        task, attempt = job
        with lock:
            if spent >= budget_usd:
                skipped += 1
                return
        target = targets[task.target]
        repo, factory = repos[task.target], factories[task.target]
        test_command = list(target.test_command)
        trace = out_dir / "traces" / _slug(model.name) / f"{task.id}-{attempt}.jsonl"
        box = materialize(task, repo, factory)
        try:
            agent = run_agent(task, box, model, test_command, config, trace_path=trace)
            score = score_attempt(task, repo, box.workdir, factory, test_command)
        finally:
            box.close()
        row = {
            "task": task.id,
            "target": task.target,
            "operator": task.operator,
            "n_failing_tests": len(task.failing_tests),
            "model": model.name,
            "attempt": attempt,
            **{k: v for k, v in asdict(agent).items() if k != "usage"},
            "usage": asdict(agent.usage),
            **{f"score_{k}": v for k, v in asdict(score).items() if k != "patch"},
            "patch": score.patch,
            "trace": str(trace.relative_to(out_dir)),
        }
        with lock:
            spent += agent.cost_usd
            new_results.append(row)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        if on_result:
            on_result(row)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run_one, jobs))
    if skipped:
        print(f"budget of ${budget_usd:.2f} reached: {skipped} attempts not started; "
              "raise --budget and re-run to continue")  # fmt: skip
    return new_results
