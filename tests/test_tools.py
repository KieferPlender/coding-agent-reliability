import pytest

from car.tasks import materialize
from car.tools import ToolBox, ToolError, is_test_file

from .conftest import TOY, TOY_TARGET, toy_factory


@pytest.fixture
def tools(toy_task):
    box = materialize(toy_task, TOY, toy_factory)
    yield ToolBox(box, list(TOY_TARGET.test_command))
    box.close()


def test_run_tests_reports_failure_with_tracebacks(tools):
    out = tools.execute("run_tests", {})
    assert out.startswith("TESTS FAILED")
    assert "test_days_in_month" in out
    assert tools.test_runs == 1 and tools.last_tests_passed is False


def test_read_list_and_search(tools):
    assert "toyproj/calendar_utils.py" in tools.execute("list_files", {"path": "."})
    text = tools.execute("read_file", {"path": "toyproj/calendar_utils.py"})
    assert "   11      if month <= 1 or month > 12:" in text
    hits = tools.execute("search", {"pattern": r"month <=? 1"})
    assert hits.startswith("toyproj/calendar_utils.py:11:")


def test_read_file_pages_and_accepts_quoted_numbers(tools):
    out = tools.execute("read_file", {"path": "toyproj/calendar_utils.py", "start_line": "20"})
    assert out.lstrip().startswith("20")


def test_edit_fixes_the_bug(tools):
    tools.execute(
        "edit_file",
        {"path": "toyproj/calendar_utils.py", "old_text": "month <= 1", "new_text": "month < 1"},
    )
    assert tools.execute("run_tests", {}).startswith("ALL TESTS PASSED")
    assert tools.edits == ["toyproj/calendar_utils.py"]


@pytest.mark.parametrize(
    "args, message",
    [
        ({"path": "toyproj/calendar_utils.py", "old_text": "nope", "new_text": "x"}, "not found"),
        ({"path": "toyproj/calendar_utils.py", "old_text": "return", "new_text": "x"}, "occurs"),
        ({"path": "../outside.py", "old_text": "a", "new_text": "b"}, "outside the repository"),
        ({"path": "tests/test_calendar_utils.py", "old_text": "a", "new_text": "b"}, "test files"),
    ],
)
def test_edit_errors_are_reported_to_the_model(tools, args, message):
    with pytest.raises(ToolError, match=message):
        tools.execute("edit_file", args)


def test_test_file_edits_are_counted(tools):
    with pytest.raises(ToolError):
        tools.execute("edit_file", {"path": "tests/test_calendar_utils.py", "old_text": "a",
                                    "new_text": "b"})  # fmt: skip
    assert tools.blocked_test_edits == 1


@pytest.mark.parametrize(
    "name, args, message",
    [
        ("read_file", {}, "missing"),
        ("read_file", {"path": "x", "colour": 1}, "unknown argument"),
        ("read_file", {"path": 3}, "must be string"),
        ("delete_everything", {}, "unknown tool"),
        ("search", {"pattern": "("}, "invalid regular expression"),
    ],
)
def test_bad_calls_are_rejected(tools, name, args, message):
    with pytest.raises(ToolError, match=message):
        tools.execute(name, args)


def test_is_test_file():
    assert is_test_file("tests/test_a.py")
    assert is_test_file("pkg/tests/helpers.py")
    assert is_test_file("pkg/conftest.py")
    assert not is_test_file("pkg/testing_utils.py")
    assert not is_test_file("pkg/core.py")
