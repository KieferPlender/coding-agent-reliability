"""Isolated workspaces where tests (and later, the agent's tools) run.

Two backends share one interface:

- ``LocalSandbox`` copies the repository into a temporary directory and runs
  commands with a timeout. Fast and dependency-free, but it does not block the
  network or limit memory. Use it for development and for generating tasks.
- ``DockerSandbox`` runs every command in a container with no network and a
  memory limit. Use it for agent runs, where the model executes code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Never copied into a workspace: caches and VCS data the task does not need.
IGNORED = shutil.ignore_patterns(
    ".git", "__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox"
)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Sandbox(Protocol):
    """A workspace containing one copy of a repository."""

    workdir: Path

    def run(self, command: list[str], timeout_s: float = 300) -> CommandResult: ...

    def close(self) -> None: ...


def _decode(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data if isinstance(data, str) else data.decode("utf-8", errors="replace")


class LocalSandbox:
    """Temporary copy of a repository; commands run as local subprocesses."""

    def __init__(
        self, source: Path, python: str | None = None, pythonpath: str | None = None
    ) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="car-"))
        self.workdir = self._tmp / "repo"
        shutil.copytree(source, self.workdir, ignore=IGNORED, symlinks=True)
        # Interpreter for "python" commands, e.g. a target's prepared venv. Made
        # absolute because commands run with the sandbox as working directory.
        self.python = os.path.abspath(python) if python and os.sep in python else python
        self.pythonpath = str(self.workdir / pythonpath) if pythonpath else None

    def run(self, command: list[str], timeout_s: float = 300) -> CommandResult:
        if self.python and command and command[0] == "python":
            command = [self.python, *command[1:]]
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": str(self._tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "LANG": "C.UTF-8",
        }
        if self.pythonpath:
            env["PYTHONPATH"] = self.pythonpath
        start = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                cwd=self.workdir,
                env=env,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                exit_code=-1,
                stdout=_decode(exc.stdout),
                stderr=_decode(exc.stderr),
                duration_s=time.monotonic() - start,
                timed_out=True,
            )
        return CommandResult(
            exit_code=proc.returncode,
            stdout=_decode(proc.stdout),
            stderr=_decode(proc.stderr),
            duration_s=time.monotonic() - start,
            timed_out=False,
        )

    def close(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> LocalSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerSandbox:
    """Copy of a repository mounted into a container with no network access.

    One long-lived container per sandbox; each command is a ``docker exec``.
    The image must already contain the target's dependencies.
    """

    def __init__(
        self,
        source: Path,
        image: str,
        memory: str = "2g",
        cpus: str = "2",
        docker: str = "docker",
        pythonpath: str | None = None,
    ) -> None:
        self._docker = docker
        self._tmp = Path(tempfile.mkdtemp(prefix="car-"))
        self.workdir = self._tmp / "repo"
        shutil.copytree(source, self.workdir, ignore=IGNORED, symlinks=True)
        self.container = f"car-{uuid.uuid4().hex[:12]}"
        subprocess.run(
            [
                docker, "run", "-d", "--rm",
                "--name", self.container,
                "--network", "none",
                "--memory", memory,
                "--cpus", cpus,
                "--pids-limit", "512",
                "-v", f"{self.workdir}:/work",
                "-w", "/work",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-e", "PYTHONHASHSEED=0",
                *(["-e", f"PYTHONPATH=/work/{pythonpath}"] if pythonpath else []),
                image, "sleep", "infinity",
            ],
            check=True,
            capture_output=True,
        )  # fmt: skip

    def run(self, command: list[str], timeout_s: float = 300) -> CommandResult:
        # `timeout` inside the container kills the process tree there, not just
        # the local docker client.
        wrapped = [
            self._docker, "exec", self.container,
            "timeout", "--kill-after=5", str(int(timeout_s)), *command,
        ]  # fmt: skip
        start = time.monotonic()
        try:
            proc = subprocess.run(wrapped, capture_output=True, timeout=timeout_s + 30, check=False)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(-1, _decode(exc.stdout), _decode(exc.stderr),
                                 time.monotonic() - start, True)  # fmt: skip
        timed_out = proc.returncode in (124, 137)
        return CommandResult(
            exit_code=-1 if timed_out else proc.returncode,
            stdout=_decode(proc.stdout),
            stderr=_decode(proc.stderr),
            duration_s=time.monotonic() - start,
            timed_out=timed_out,
        )

    def close(self) -> None:
        subprocess.run([self._docker, "rm", "-f", self.container],
                       capture_output=True, check=False)  # fmt: skip
        # Files written by the container may be owned by root; best effort.
        shutil.rmtree(self._tmp, ignore_errors=True)

    def __enter__(self) -> DockerSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def docker_available(docker: str = "docker") -> bool:
    try:
        proc = subprocess.run([docker, "info"], capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0
