import sys
from pathlib import Path

import pytest

from car.mutate import apply_mutation
from car.sandbox import LocalSandbox
from car.targets import Target
from car.tasks import (
    Task,
    failing_tests,
    generate_tasks,
    materialize,
    pytest_args,
    read_tasks,
    write_tasks,
)

TOY = Path(__file__).parent / "fixtures" / "toyproj"
TOY_TARGET = Target(
    name="toyproj",
    path=str(TOY),
    source_dirs=("toyproj",),
    test_command=("python", "-m", "pytest", "tests"),
)


def factory(src: Path) -> LocalSandbox:
    # The test runner's own interpreter already has pytest installed.
    return LocalSandbox(src, python=sys.executable, pythonpath=".")


@pytest.fixture(scope="module")
def generated() -> tuple[list[Task], object]:
    return generate_tasks(TOY_TARGET, TOY, make_sandbox=factory, max_candidates=None, workers=4)


def test_generates_tasks_and_rejects_untested_code(generated):
    tasks, report = generated
    assert report.accepted == len(tasks) > 5
    assert report.tested == report.candidates
    assert report.accepted + report.survived + report.timed_out == report.tested
    assert report.unclear == 0
    # Mutations in the untested helper can never fail a test.
    assert all(t.mutation.line < 27 for t in tasks)
    assert report.survived >= 1


def test_every_task_is_a_real_failing_bug(generated):
    tasks, _ = generated
    for task in tasks[:5]:
        assert task.failing_tests
        assert task.diff.startswith("--- a/toyproj/calendar_utils.py")
        with materialize(task, TOY, factory) as box:
            result = box.run(pytest_args(TOY_TARGET))
            assert not result.ok
            assert failing_tests(result) == set(task.failing_tests)


def test_reverting_the_bug_makes_tests_pass(generated):
    tasks, _ = generated
    task = tasks[0]
    with materialize(task, TOY, factory) as box:
        path = box.workdir / task.mutation.path
        path.write_text((TOY / task.mutation.path).read_text())
        assert box.run(pytest_args(TOY_TARGET)).ok


def test_generation_is_deterministic():
    a, _ = generate_tasks(TOY_TARGET, TOY, make_sandbox=factory, max_candidates=8, seed=7)
    b, _ = generate_tasks(TOY_TARGET, TOY, make_sandbox=factory, max_candidates=8, seed=7)
    assert [t.id for t in a] == [t.id for t in b]


def test_round_trip(tmp_path, generated):
    tasks, _ = generated
    path = tmp_path / "tasks.jsonl"
    write_tasks(tasks, path)
    loaded = read_tasks(path)
    assert [t.to_json() for t in loaded] == [t.to_json() for t in tasks]
    source = (TOY / loaded[0].mutation.path).read_text()
    assert apply_mutation(source, loaded[0].mutation) != source


def test_failing_tests_parses_pytest_summary():
    from car.sandbox import CommandResult

    out = (
        "..F.\n=== short test summary info ===\n"
        "FAILED tests/test_a.py::test_x - assert 1 == 2\n"
        "ERROR tests/test_b.py::test_y\n"
    )
    result = CommandResult(1, out, "", 0.1, False)
    assert failing_tests(result) == {"tests/test_a.py::test_x", "tests/test_b.py::test_y"}


def test_target_mutability_rules():
    t = Target(
        name="x", path=".", source_dirs=("pkg",), test_command=("pytest",), exclude=("pkg/vendor",)
    )
    assert t.is_mutable("pkg/core.py")
    assert not t.is_mutable("pkg/test_core.py")
    assert not t.is_mutable("pkg/conftest.py")
    assert not t.is_mutable("pkg/vendor/lib.py")
    assert not t.is_mutable("other/core.py")
    assert not t.is_mutable("pkg/data.json")


def test_sampling_balances_bug_types():
    from car.mutate import Mutation
    from car.tasks import sample_candidates

    many = [Mutation("a.py", i, 0, "1", "2", "off_by_one") for i in range(1, 100)]
    few = [Mutation("b.py", i, 0, "<", "<=", "comparison") for i in range(1, 5)]
    sample = sample_candidates(many + few, limit=8, seed=0)
    assert sum(m.operator == "comparison" for m in sample) == 4
    assert sample_candidates(many + few, 8, 0) == sample  # deterministic


def test_failing_tests_ignores_terminal_colours():
    # Projects with `addopts = "--color=yes"` colour the summary lines.
    from car.sandbox import CommandResult

    out = "\x1b[31mFAILED\x1b[0m tests/test_a.py::test_x - assert 1 == 2\n"
    assert failing_tests(CommandResult(1, out, "", 0.1, False)) == {"tests/test_a.py::test_x"}


def test_coloured_target_still_produces_tasks(tmp_path):
    # Regression: python-slugify forces colour output, which hid every failure.
    import shutil

    proj = tmp_path / "toyproj"
    shutil.copytree(TOY, proj)
    (proj / "pyproject.toml").write_text('[tool.pytest.ini_options]\naddopts = "--color=yes"\n')
    target = Target(
        name="colour",
        path=str(proj),
        source_dirs=("toyproj",),
        test_command=("python", "-m", "pytest", "tests"),
    )
    tasks, report = generate_tasks(target, proj, make_sandbox=factory, max_candidates=10)
    assert report.accepted > 0 and report.unclear == 0


def test_failures_without_parseable_output_are_unclear_not_survived(tmp_path):
    import shutil

    proj = tmp_path / "toyproj"
    shutil.copytree(TOY, proj)
    # A test command that fails without pytest's summary when the bug is present.
    target = Target(
        name="opaque",
        path=str(proj),
        source_dirs=("toyproj",),
        test_command=(
            "python",
            "-c",
            "import sys; from toyproj.calendar_utils import days_in_month as d;"
            "sys.exit(d(2023, 1) != 31)",
            "--",
        ),
    )
    _, report = generate_tasks(target, proj, make_sandbox=factory, max_candidates=None)
    assert report.unclear > 0
    assert report.accepted == 0
    assert len(report.unclear_details) == report.unclear
    assert all(d["exit_code"] != 0 and d["mutation"] for d in report.unclear_details)


def test_failing_tests_reads_subtest_failures():
    # unittest subTest / pytest subtests report failures as SUBFAILED(<params>) <id>.
    from car.sandbox import CommandResult

    out = (
        "SUBFAILED(text=\"'abc\\\\u200d'\") tests/test_m.py::Marks::test_joiners - Ass...\n"
        "SUBFAILED(separator='::') tests/test_w.py::Words::test_order - AssertionError\n"
        "SUBFAILED(text='a b') tests/test_w.py::Words::test_spaces\n"
        "SUBFAILED tests/test_w.py::Words::test_plain\n"
        "4 failed, 140 passed, 5 skipped, 148 subtests passed in 0.52s\n"
    )
    assert failing_tests(CommandResult(1, out, "", 0.1, False)) == {
        "tests/test_m.py::Marks::test_joiners",
        "tests/test_w.py::Words::test_order",
        "tests/test_w.py::Words::test_spaces",
        "tests/test_w.py::Words::test_plain",
    }


def test_real_pytest_subtest_failure_is_recognised(tmp_path):
    # End to end against pytest's actual output, not a hand-written sample.
    from car.sandbox import LocalSandbox

    (tmp_path / "test_sub.py").write_text(
        "import unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_values(self):\n"
        "        for v in (1, 2, 3):\n"
        "            with self.subTest(value=v, text='a b'):\n"
        "                self.assertNotEqual(v, 2)\n"
    )
    target = Target(
        name="sub",
        path=str(tmp_path),
        source_dirs=(".",),
        test_command=("python", "-m", "pytest", "test_sub.py"),
    )
    with LocalSandbox(tmp_path, python=sys.executable) as box:
        result = box.run(pytest_args(target))
    assert result.exit_code == 1
    assert failing_tests(result) == {"test_sub.py::T::test_values"}
