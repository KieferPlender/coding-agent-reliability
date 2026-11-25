from pathlib import Path

from car.cli import main
from car.tasks import read_tasks

TOY = Path(__file__).parent / "fixtures" / "toyproj"


def test_generate_end_to_end_with_relative_cache(tmp_path, monkeypatch):
    """`car generate` exactly as a user runs it: relative paths, a fresh venv."""
    (tmp_path / "targets.toml").write_text(
        f"""
[[target]]
name = "toy"
path = "{TOY.as_posix()}"
source_dirs = ["toyproj"]
test_command = ["python", "-m", "pytest", "tests"]
requirements = ["pytest>=8"]
"""
    )
    monkeypatch.chdir(tmp_path)
    code = main(
        ["generate", "toy", "--targets", "targets.toml", "--cache", ".car",
         "--max-candidates", "10"]
    )  # fmt: skip
    assert code == 0
    tasks = read_tasks(Path("tasks/toy.jsonl"))
    assert tasks and all(t.failing_tests for t in tasks)
    assert main(["show", "tasks/toy.jsonl"]) == 0


def test_show_rejects_non_task_files_cleanly(tmp_path, capsys):
    bad = tmp_path / "x.unclear.jsonl"
    bad.write_text('{"mutation": "a.py:1:0:<-><=", "exit_code": 3}\n')
    assert main(["show", str(bad)]) == 2
    assert "not a task file" in capsys.readouterr().err


def test_load_dotenv_sets_missing_variables_only(tmp_path, monkeypatch):
    from car.cli import load_dotenv

    env = tmp_path / ".env"
    env.write_text("# comment\nexport CAR_TEST_A='one'\nCAR_TEST_B=\"two\"\nCAR_TEST_C=keep\nbad\n")
    monkeypatch.delenv("CAR_TEST_A", raising=False)
    monkeypatch.delenv("CAR_TEST_B", raising=False)
    monkeypatch.setenv("CAR_TEST_C", "already-set")
    assert load_dotenv(env) == ["CAR_TEST_A", "CAR_TEST_B"]
    import os

    assert os.environ["CAR_TEST_A"] == "one" and os.environ["CAR_TEST_B"] == "two"
    assert os.environ["CAR_TEST_C"] == "already-set"
    assert load_dotenv(tmp_path / "missing") == []
