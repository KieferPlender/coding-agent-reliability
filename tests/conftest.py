import sys
from pathlib import Path

import pytest

from car.sandbox import LocalSandbox
from car.targets import Target
from car.tasks import generate_tasks

TOY = Path(__file__).parent / "fixtures" / "toyproj"
TOY_TARGET = Target(
    name="toyproj",
    path=str(TOY),
    source_dirs=("toyproj",),
    test_command=("python", "-m", "pytest", "tests"),
)


def toy_factory(src: Path) -> LocalSandbox:
    # The test runner's own interpreter already has pytest installed.
    return LocalSandbox(src, python=sys.executable, pythonpath=".")


@pytest.fixture(scope="session")
def toy_tasks():
    tasks, _ = generate_tasks(TOY_TARGET, TOY, make_sandbox=toy_factory, max_candidates=None)
    return tasks


@pytest.fixture
def toy_task(toy_tasks):
    """The `month < 1` -> `month <= 1` bug: January is rejected."""
    return next(t for t in toy_tasks if t.mutation.line == 11 and t.mutation.original == "<")
