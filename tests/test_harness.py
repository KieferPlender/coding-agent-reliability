from car.harness import load_results, run_suite
from car.metrics import format_report, summarize
from car.models import ModelResponse, ScriptedModel, Usage

from .conftest import TOY, TOY_TARGET, toy_factory

FIX = [("edit_file", {"path": "toyproj/calendar_utils.py", "old_text": "month <= 1",
                      "new_text": "month < 1"})]  # fmt: skip


class OracleModel:
    """Fixes the one known bug, at a fixed cost per response."""

    name = "oracle"

    def __init__(self, cost=0.01):
        self.cost = cost

    def start(self, system, tools, task):
        conv = ScriptedModel([FIX, [("finish", {"summary": "fixed"})]]).start(system, tools, task)
        original = conv.next

        def next_with_cost():
            r = original()
            return ModelResponse(r.text, r.tool_calls, r.stop_reason, Usage(1000, 100),
                                 self.cost, 0.0)  # fmt: skip

        conv.next = next_with_cost
        return conv


def run(tasks, model, out, **kw):
    return run_suite(tasks, model, {"toyproj": TOY_TARGET}, {"toyproj": TOY},
                     {"toyproj": toy_factory}, out, workers=2, **kw)  # fmt: skip


def test_suite_runs_scores_and_resumes(toy_task, tmp_path):
    rows = run([toy_task], OracleModel(), tmp_path, attempts=3, budget_usd=1.0)
    assert len(rows) == 3 and all(r["score_solved"] for r in rows)
    assert sorted(r["attempt"] for r in rows) == [1, 2, 3]
    assert all((tmp_path / r["trace"]).exists() for r in rows)
    # Re-running with more attempts only runs the missing ones.
    more = run([toy_task], OracleModel(), tmp_path, attempts=4, budget_usd=1.0)
    assert [r["attempt"] for r in more] == [4]
    assert len(load_results(tmp_path / "results.jsonl")) == 4

    (summary,) = summarize(load_results(tmp_path / "results.jsonl"))
    assert summary.solve_rate == 1.0 and summary.pass_all_k == 1.0 and summary.k == 4
    assert summary.cost_per_solved == 0.02 and summary.exact_revert_share == 1.0
    assert "solve rate" in format_report(load_results(tmp_path / "results.jsonl"))


def test_budget_stops_new_attempts(toy_tasks, tmp_path):
    rows = run(toy_tasks[:6], OracleModel(cost=0.5), tmp_path, attempts=1, budget_usd=1.0)
    # Each attempt costs $1.00 (two responses); at most `workers` start before the cap.
    assert 1 <= len(rows) <= 2


def test_infrastructure_errors_are_logged_not_scored(toy_task, tmp_path):
    def broken(src):
        raise RuntimeError("docker exploded")

    rows = run_suite([toy_task], OracleModel(), {"toyproj": TOY_TARGET}, {"toyproj": TOY},
                     {"toyproj": broken}, tmp_path, attempts=1)  # fmt: skip
    assert rows == []
    assert "docker exploded" in (tmp_path / "errors.jsonl").read_text()


def test_pass_all_k_and_failure_kinds():
    base = {
        "target": "t",
        "operator": "comparison",
        "n_failing_tests": 1,
        "model": "m",
        "cost_usd": 0.1,
        "seconds": 10,
        "tool_calls": 4,
        "malformed_tool_calls": 0,
        "blocked_test_edits": 0,
        "score_changed_test_files": [],
        "score_exact_revert": True,
    }
    rows = [
        {**base, "task": "a", "attempt": 1, "score_solved": True, "stop": "finished",
         "score_changed_files": ["x.py"], "score_touched_bug_file": True},
        {**base, "task": "a", "attempt": 2, "score_solved": True, "stop": "finished",
         "score_changed_files": ["x.py"], "score_touched_bug_file": True},
        {**base, "task": "b", "attempt": 1, "score_solved": True, "stop": "finished",
         "score_changed_files": ["x.py"], "score_touched_bug_file": True},
        {**base, "task": "b", "attempt": 2, "score_solved": False, "stop": "finished",
         "score_changed_files": ["y.py"], "score_touched_bug_file": False},
    ]  # fmt: skip
    (s,) = summarize(rows)
    assert s.solve_rate == 0.75 and s.pass_all_k == 0.5 and s.k == 2
    assert s.cost_per_solved == round(0.4 / 3, 4)
    assert s.failure_kinds == {"edited_wrong_file": 1}


def test_full_attempt_in_docker(toy_task, tmp_path):
    """Image build, sandboxed agent run and scoring, all inside Docker."""
    import shutil

    import pytest

    from car.images import build_image
    from car.sandbox import DockerSandbox, docker_available
    from car.targets import Target

    if shutil.which("docker") is None or not docker_available():
        pytest.skip("Docker not available")
    target = Target(name="toyproj", path=str(TOY), source_dirs=("toyproj",),
                    test_command=TOY_TARGET.test_command, requirements=("pytest>=8",))  # fmt: skip
    image = build_image(target)
    assert build_image(target) == image  # cached the second time

    def factory(src):
        return DockerSandbox(src, image=image, pythonpath=".")

    rows = run_suite([toy_task], OracleModel(), {"toyproj": target}, {"toyproj": TOY},
                     {"toyproj": factory}, tmp_path, attempts=1)  # fmt: skip
    assert len(rows) == 1 and rows[0]["score_solved"] and rows[0]["score_exact_revert"]
