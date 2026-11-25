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
