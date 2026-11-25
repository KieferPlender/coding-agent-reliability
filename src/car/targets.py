"""Target repositories: real projects with good test suites that bugs are injected into."""

from __future__ import annotations

import subprocess
import sys
import tomllib
import venv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Target:
    name: str
    source_dirs: tuple[str, ...]  # where bugs may be injected, relative to the repo root
    test_command: tuple[str, ...]
    repo: str | None = None  # git URL; None for a local path (e.g. test fixtures)
    commit: str | None = None
    path: str | None = None  # local checkout, used instead of `repo`
    exclude: tuple[str, ...] = ()  # path prefixes never mutated (tests, vendored code)
    pythonpath: str = "."  # added to PYTHONPATH so tests import the sandbox copy
    requirements: tuple[str, ...] = field(default_factory=tuple)  # pip specs for the env

    def is_mutable(self, rel_path: str) -> bool:
        if not rel_path.endswith(".py"):
            return False
        if not any(
            rel_path == d or rel_path.startswith(d.rstrip("/") + "/") for d in self.source_dirs
        ):
            return False
        name = rel_path.rsplit("/", 1)[-1]
        if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
            return False
        return not any(
            rel_path.startswith(e.rstrip("/") + "/") or rel_path == e for e in self.exclude
        )


def load_targets(path: Path) -> dict[str, Target]:
    data = tomllib.loads(path.read_text())
    targets: dict[str, Target] = {}
    for raw in data.get("target", []):
        local = raw.get("path")
        target = Target(
            name=raw["name"],
            source_dirs=tuple(raw["source_dirs"]),
            test_command=tuple(raw["test_command"]),
            repo=raw.get("repo"),
            commit=raw.get("commit"),
            path=str((path.parent / local).resolve()) if local else None,
            exclude=tuple(raw.get("exclude", ())),
            pythonpath=raw.get("pythonpath", "."),
            requirements=tuple(raw.get("requirements", ())),
        )
        if not target.repo and not target.path:
            raise ValueError(f"target {target.name!r} needs `repo` or `path`")
        if target.repo and not target.commit:
            raise ValueError(f"target {target.name!r} must pin a `commit` for reproducibility")
        targets[target.name] = target
    return targets


def checkout(target: Target, cache: Path) -> Path:
    """Return a clean checkout of `target` at its pinned commit."""
    if target.path:
        return Path(target.path)
    assert target.repo and target.commit
    dest = cache / "repos" / target.name
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin", target.repo], check=True)
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    if head != target.commit:
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "-q", "--depth", "1", "origin", target.commit],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "-q", "--force", "FETCH_HEAD"], check=True
        )
    return dest


def prepare_env(target: Target, cache: Path) -> str:
    """Create (once) a virtualenv with the target's test dependencies; return its python.

    The target itself is NOT installed: tests import it from the sandbox copy via
    PYTHONPATH, so injected bugs are what the tests actually exercise.
    """
    env_dir = cache / "envs" / target.name
    python = env_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    marker = env_dir / ".car-ready"
    wanted = "\n".join(sorted(target.requirements))
    if marker.exists() and marker.read_text() == wanted:
        return str(python)
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    if target.requirements:
        subprocess.run(
            [str(python), "-m", "pip", "install", "-q", *target.requirements], check=True
        )
    marker.write_text(wanted)
    return str(python)
