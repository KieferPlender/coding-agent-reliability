"""Command-line interface: `car generate`, `car show`."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from car.sandbox import DockerSandbox, LocalSandbox
from car.targets import Target, checkout, load_targets, prepare_env
from car.tasks import SandboxFactory, generate_tasks, read_tasks, write_tasks

DEFAULT_TARGETS = Path("targets/targets.toml")
DEFAULT_CACHE = Path(".car")


def sandbox_factory(target: Target, args: argparse.Namespace) -> SandboxFactory:
    """Docker if an image is given, otherwise a local venv with the target's deps."""
    if args.docker_image:
        image = args.docker_image
        return lambda src: DockerSandbox(src, image=image, pythonpath=target.pythonpath)
    python = prepare_env(target, args.cache)
    return lambda src: LocalSandbox(src, python=python, pythonpath=target.pythonpath)


def cmd_generate(args: argparse.Namespace) -> int:
    targets = load_targets(args.targets)
    names = args.target or sorted(targets)
    unknown = [n for n in names if n not in targets]
    if unknown:
        print(f"unknown target(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    for name in names:
        target = targets[name]
        repo = checkout(target, args.cache)
        factory = sandbox_factory(target, args)
        print(f"[{name}] generating from {repo}", file=sys.stderr)
        tasks, report = generate_tasks(
            target,
            repo,
            make_sandbox=factory,
            max_candidates=args.max_candidates,
            seed=args.seed,
            workers=args.workers,
            on_task=lambda t: print(f"  + {t.id}  {t.mutation.id}", file=sys.stderr),
        )
        out = args.out / f"{name}.jsonl"
        write_tasks(tasks, out)
        print(
            f"[{name}] {report.accepted} tasks from {report.tested} tested candidates "
            f"({report.candidates} possible; {report.survived} survived the tests, "
            f"{report.timed_out} timed out) -> {out}",
            file=sys.stderr,
        )
        if report.unclear:
            print(
                f"[{name}] WARNING: {report.unclear} candidates failed the tests but no "
                "failing test could be read from pytest's output; check the test command",
                file=sys.stderr,
            )
        if report.baseline_failures:
            print(f"[{name}] ignored failing baseline tests: {report.baseline_failures}",
                  file=sys.stderr)  # fmt: skip
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    tasks = [t for path in args.files for t in read_tasks(path)]
    print(f"{len(tasks)} tasks")
    for label, counter in (
        ("by target", Counter(t.target for t in tasks)),
        ("by bug type", Counter(t.operator for t in tasks)),
        ("by failing tests", Counter(_bucket(len(t.failing_tests)) for t in tasks)),
    ):
        print(f"\n{label}:")
        for key, n in sorted(counter.items()):
            print(f"  {key:<14} {n}")
    return 0


def _bucket(n: int) -> str:
    return "1" if n == 1 else "2-5" if n <= 5 else "6+"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="car", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate", help="inject bugs into target repos and keep the ones tests catch"
    )
    gen.add_argument("target", nargs="*", help="target names (default: all)")
    gen.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    gen.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    gen.add_argument("--out", type=Path, default=Path("tasks"))
    gen.add_argument("--max-candidates", type=int, default=200)
    gen.add_argument("--seed", type=int, default=0)
    gen.add_argument("--workers", type=int, default=4)
    gen.add_argument("--docker-image", help="run tests in this image instead of a local venv")
    gen.set_defaults(func=cmd_generate)

    show = sub.add_parser("show", help="summarize task files")
    show.add_argument("files", nargs="+", type=Path)
    show.set_defaults(func=cmd_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
