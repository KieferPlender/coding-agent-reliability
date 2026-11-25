from car.agent import AgentConfig, run_agent
from car.harness import load_results
from car.metrics import failure_kind
from car.models import ScriptedModel
from car.scoring import score_attempt
from car.tasks import materialize

from .conftest import TOY, TOY_TARGET, toy_factory

PATH = "toyproj/calendar_utils.py"
FIX = [("edit_file", {"path": PATH, "old_text": "month <= 1", "new_text": "month < 1"})]


def attempt(task, steps, tmp_path, config=None):
    box = materialize(task, TOY, toy_factory)
    try:
        model = ScriptedModel(steps)
        result = run_agent(task, box, model, list(TOY_TARGET.test_command), config,
                           trace_path=tmp_path / "trace.jsonl")  # fmt: skip
        score = score_attempt(task, TOY, box.workdir, toy_factory, list(TOY_TARGET.test_command))
    finally:
        box.close()
    return result, score, model


def test_scripted_agent_fixes_the_bug(toy_task, tmp_path):
    steps = [
        [("run_tests", {})],
        [("search", {"pattern": "out of range"}), ("read_file", {"path": PATH})],
        FIX,
        [("run_tests", {})],
        [("finish", {"summary": "January was rejected"})],
    ]
    result, score, model = attempt(toy_task, steps, tmp_path)
    assert result.stop == "finished" and result.steps == 5
    assert result.tool_calls == 6 and result.tool_errors == 0 and result.test_runs == 2
    assert score.solved and score.exact_revert and score.touched_bug_file
    assert "-    if month <= 1" in score.patch and "+    if month < 1" in score.patch
    # Two tool calls in one step come back together, in one batch of results.
    assert len(model.seen[1]) == 2
    events = [r["event"] for r in load_results(tmp_path / "trace.jsonl")]
    assert events[0] == "start" and events[-1] == "end" and events.count("tool") == 6


def test_alternative_fix_counts_but_is_not_an_exact_revert(toy_task, tmp_path):
    steps = [[("edit_file", {"path": PATH, "old_text": "month <= 1", "new_text": "1 > month"})],
             [("finish", {"summary": "x"})]]  # fmt: skip
    _, score, _ = attempt(toy_task, steps, tmp_path)
    assert score.solved and not score.exact_revert


def test_claiming_success_without_a_fix_fails(toy_task, tmp_path):
    result, score, _ = attempt(toy_task, [[("finish", {"summary": "fixed it"})]], tmp_path)
    assert result.stop == "finished"
    assert not score.solved and score.changed_files == []
    assert set(score.still_failing) == set(toy_task.failing_tests)


def test_editing_tests_is_blocked_and_classified(toy_task, tmp_path):
    steps = [[("edit_file", {"path": "tests/test_calendar_utils.py",
                             "old_text": "def test_days_in_month", "new_text": "def _skip"})],
             [("finish", {"summary": "x"})]]  # fmt: skip
    result, score, _ = attempt(toy_task, steps, tmp_path)
    assert result.blocked_test_edits == 1 and result.tool_errors == 1
    row = {"score_solved": score.solved, "stop": result.stop, "blocked_test_edits": 1,
           "score_changed_test_files": score.changed_test_files, "malformed_tool_calls": 0,
           "tool_calls": 2, "score_changed_files": score.changed_files,
           "score_touched_bug_file": score.touched_bug_file}  # fmt: skip
    assert failure_kind(row) == "tried_to_edit_tests"


def test_text_only_replies_get_one_nudge_then_stop(toy_task, tmp_path):
    result, _, model = attempt(toy_task, ["I think it is fixed.", "Really."], tmp_path)
    assert result.stop == "no_tool_calls" and result.steps == 2
    assert model.seen == [[("user", model.seen[0][0][1], False)]]


def test_step_limit(toy_task, tmp_path):
    steps = [[("run_tests", {})]] * 10
    result, score, _ = attempt(toy_task, steps, tmp_path, AgentConfig(max_steps=3))
    assert result.stop == "max_steps" and result.steps == 3 and not score.solved
