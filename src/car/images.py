"""Docker images with a target's test dependencies, for sandboxed agent runs."""

from __future__ import annotations

import hashlib
import subprocess

from car.targets import Target

BASE_IMAGE = "python:3.12-slim"


def image_tag(target: Target) -> str:
    spec = "\n".join([BASE_IMAGE, *sorted(target.requirements)])
    return f"car-{target.name}:{hashlib.sha256(spec.encode()).hexdigest()[:12]}"


def dockerfile(target: Target) -> str:
    install = " ".join(f'"{r}"' for r in target.requirements)
    lines = [f"FROM {BASE_IMAGE}"]
    if install:
        lines.append(f"RUN pip install --no-cache-dir {install}")
    lines += ["WORKDIR /work", 'CMD ["sleep", "infinity"]']
    return "\n".join(lines) + "\n"


def build_image(target: Target, docker: str = "docker") -> str:
    """Build (or reuse) the image for `target` and return its tag."""
    tag = image_tag(target)
    exists = subprocess.run([docker, "image", "inspect", tag], capture_output=True, check=False)
    if exists.returncode != 0:
        subprocess.run(
            [docker, "build", "-t", tag, "-"],
            input=dockerfile(target).encode(),
            check=True,
        )
    return tag
