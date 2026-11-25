"""Summaries of a results file: reliability, cost and why attempts failed."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass
class ModelSummary:
    model: str
    tasks: int
    attempts: int
    solve_rate: float  # mean over tasks of the per-task success rate (pass@1)
    solve_rate_ci: tuple[float, float]  # 95% bootstrap interval over tasks
    pass_all_k: float | None  # share of tasks solved on every one of k attempts
    k: int
    cost_usd: float
    cost_per_solved: float | None
    seconds_per_attempt: float
    seconds_per_solved: float | None
    exact_revert_share: float | None  # of solved attempts, how many restored the original
    stop_reasons: dict[str, int]
    failure_kinds: dict[str, int]
    malformed_tool_call_rate: float  # per tool call


def failure_kind(row: dict) -> str | None:
    """A coarse, automatic label for why an attempt did not solve its task."""
    if row["score_solved"]:
        return None
    stop = row["stop"]
    if stop in ("refusal", "model_error", "max_cost", "max_time"):
        return stop
    if row["blocked_test_edits"] or row["score_changed_test_files"]:
        return "tried_to_edit_tests"
    if row["malformed_tool_calls"] and row["malformed_tool_calls"] >= row["tool_calls"] / 2:
        return "malformed_tool_calls"
    if not row["score_changed_files"]:
        return "no_change" if stop != "max_steps" else "ran_out_of_steps_without_change"
    if not row["score_touched_bug_file"]:
        return "edited_wrong_file"
    if stop == "finished":
        return "finished_but_tests_fail"
    if stop == "max_steps":
        return "ran_out_of_steps"
    return "gave_up"


def _bootstrap(values: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(n))
    return (means[int(0.025 * n)], means[int(0.975 * n) - 1])


def summarize(rows: list[dict]) -> list[ModelSummary]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)
    summaries = []
    for model, model_rows in sorted(by_model.items()):
        per_task: dict[str, list[bool]] = defaultdict(list)
        for r in model_rows:
            per_task[r["task"]].append(bool(r["score_solved"]))
        task_rates = [mean(v) for v in per_task.values()]
        k = min(len(v) for v in per_task.values())
        full = [v for v in per_task.values() if len(v) >= k]
        solved = [r for r in model_rows if r["score_solved"]]
        cost = sum(r["cost_usd"] for r in model_rows)
        seconds = sum(r["seconds"] for r in model_rows)
        tool_calls = sum(r["tool_calls"] for r in model_rows)
        summaries.append(
            ModelSummary(
                model=model,
                tasks=len(per_task),
                attempts=len(model_rows),
                solve_rate=mean(task_rates),
                solve_rate_ci=_bootstrap(task_rates),
                pass_all_k=mean(all(v[:k]) for v in full) if k > 1 else None,
                k=k,
                cost_usd=round(cost, 4),
                cost_per_solved=round(cost / len(solved), 4) if solved else None,
                seconds_per_attempt=round(seconds / len(model_rows), 1),
                seconds_per_solved=round(seconds / len(solved), 1) if solved else None,
                exact_revert_share=(
                    mean(r["score_exact_revert"] for r in solved) if solved else None
                ),
                stop_reasons=dict(Counter(r["stop"] for r in model_rows)),
                failure_kinds=dict(
                    Counter(k for r in model_rows if (k := failure_kind(r)) is not None)
                ),
                malformed_tool_call_rate=(
                    sum(r["malformed_tool_calls"] for r in model_rows) / tool_calls
                    if tool_calls
                    else 0.0
                ),
            )
        )
    return summaries


def breakdown(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    """Solve rate per model for each value of `key` (e.g. target, operator)."""
    table: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        table[str(r[key])][r["model"]].append(bool(r["score_solved"]))
    return {group: {m: mean(v) for m, v in models.items()} for group, models in table.items()}


def difficulty_bucket(n_failing_tests: int) -> str:
    return "1 test" if n_failing_tests == 1 else "2-5 tests" if n_failing_tests <= 5 else "6+ tests"


def format_report(rows: list[dict]) -> str:
    lines = []
    for s in summarize(rows):
        lo, hi = s.solve_rate_ci
        lines += [
            f"== {s.model}",
            f"  tasks {s.tasks}, attempts {s.attempts} (k={s.k})",
            f"  solve rate      {s.solve_rate:6.1%}   (95% CI {lo:.1%}-{hi:.1%})",
        ]
        if s.pass_all_k is not None:
            lines.append(f"  solved all {s.k}x   {s.pass_all_k:6.1%}")
        lines += [
            f"  cost            ${s.cost_usd:.4f} total, "
            + (f"${s.cost_per_solved:.4f} per solved" if s.cost_per_solved is not None else "-"),
            f"  time            {s.seconds_per_attempt}s per attempt, "
            + (f"{s.seconds_per_solved}s per solved" if s.seconds_per_solved is not None else "-"),
            f"  stop reasons    {s.stop_reasons}",
            f"  failure kinds   {s.failure_kinds}",
            f"  malformed tool calls {s.malformed_tool_call_rate:.1%} of calls",
        ]
        if s.exact_revert_share is not None:
            lines.append(f"  exact reverts   {s.exact_revert_share:.0%} of solved attempts")
    for key, label in (("target", "by target"), ("operator", "by bug type")):
        lines.append(f"\n{label}:")
        for group, models in sorted(breakdown(rows, key).items()):
            cells = ", ".join(f"{m} {v:.0%}" for m, v in sorted(models.items()))
            lines.append(f"  {group:<16} {cells}")
    bucketed = [{**r, "difficulty": difficulty_bucket(r["n_failing_tests"])} for r in rows]
    lines.append("\nby failing tests:")
    for group, models in sorted(breakdown(bucketed, "difficulty").items()):
        cells = ", ".join(f"{m} {v:.0%}" for m, v in sorted(models.items()))
        lines.append(f"  {group:<16} {cells}")
    return "\n".join(lines)
