import shutil
import sys
from pathlib import Path

import pytest

from car.sandbox import DockerSandbox, LocalSandbox, docker_available

TOY = Path(__file__).parent / "fixtures" / "toyproj"


def test_local_sandbox_is_an_isolated_copy():
    with LocalSandbox(TOY, python=sys.executable) as box:
        (box.workdir / "toyproj" / "calendar_utils.py").write_text("broken = True\n")
        result = box.run(["python", "-c", "print('hi')"])
        assert result.ok and result.stdout.strip() == "hi"
        workdir = box.workdir
    assert not workdir.exists()
    assert "def is_leap" in (TOY / "toyproj" / "calendar_utils.py").read_text()


def test_local_sandbox_timeout():
    with LocalSandbox(TOY, python=sys.executable) as box:
        result = box.run(["python", "-c", "import time; time.sleep(10)"], timeout_s=1)
    assert result.timed_out and not result.ok


def test_local_sandbox_pythonpath_points_at_the_copy():
    with LocalSandbox(TOY, python=sys.executable, pythonpath=".") as box:
        (box.workdir / "toyproj" / "marker.py").write_text("VALUE = 42\n")
        result = box.run(["python", "-c", "from toyproj.marker import VALUE; print(VALUE)"])
    assert result.stdout.strip() == "42"


docker = pytest.mark.skipif(
    shutil.which("docker") is None or not docker_available(), reason="Docker not available"
)


@docker
@pytest.mark.docker
def test_docker_sandbox_has_no_network():
    with DockerSandbox(TOY, image="python:3.12-slim", pythonpath=".") as box:
        ok = box.run(["python", "-c", "import toyproj.calendar_utils as c; print(c.is_leap(2024))"])
        assert ok.ok and ok.stdout.strip() == "True"
        net = box.run(
            ["python", "-c", "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)"]
        )
        assert not net.ok


@docker
@pytest.mark.docker
def test_docker_sandbox_timeout():
    with DockerSandbox(TOY, image="python:3.12-slim") as box:
        result = box.run(["sleep", "30"], timeout_s=2)
    assert result.timed_out


def test_local_sandbox_accepts_relative_interpreter_path(tmp_path, monkeypatch):
    # `car generate` passes paths like `.car/envs/x/bin/python`, relative to where
    # it was started; commands run inside the sandbox, so it must be made absolute.
    link = tmp_path / "envs" / "python"
    link.parent.mkdir()
    link.symlink_to(sys.executable)
    monkeypatch.chdir(tmp_path)
    with LocalSandbox(TOY, python="envs/python") as box:
        result = box.run(["python", "-c", "print('ok')"])
    assert result.ok and result.stdout.strip() == "ok"
