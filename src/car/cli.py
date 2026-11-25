"""Command-line interface: generate tasks, run agents on them, report results."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from car.sandbox import DockerSandbox, LocalSandbox, docker_available
from car.targets import Target, checkout, load_targets, prepare_env
from car.tasks import SandboxFactory, generate_tasks, read_tasks, select_tasks, write_tasks

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
            # Kept out of the task directory so `tasks/*.jsonl` only matches task files.
            log = args.out / "diagnostics" / f"{name}.unclear.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w", encoding="utf-8") as f:
                for detail in report.unclear_details:
                    f.write(json.dumps(detail) + "\n")
            print(
                f"[{name}] WARNING: {report.unclear} candidates failed the tests but no "
                f"failing test could be read from pytest's output; details in {log}",
                file=sys.stderr,
            )
        if report.baseline_failures:
            print(f"[{name}] ignored failing baseline tests: {report.baseline_failures}",
                  file=sys.stderr)  # fmt: skip
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    tasks = []
    for path in args.files:
        try:
            tasks.extend(read_tasks(path))
        except (ValueError, TypeError, KeyError) as exc:
            print(f"{path}: not a task file ({exc})", file=sys.stderr)
            return 2
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


def load_dotenv(path: Path = Path(".env")) -> list[str]:
    """Set variables from a local `.env` file (KEY=value lines) if not already set.

    Lets the API key live in a git-ignored file instead of shell history. Values are
    never printed; only the names of variables that were loaded are returned.
    """
    import os

    if not path.is_file():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def make_model(args: argparse.Namespace):
    from car.models import ClaudeModel, OpenAICompatModel

    if args.provider == "claude":
        return ClaudeModel(args.model, effort=args.effort)
    return OpenAICompatModel(args.model, base_url=args.base_url)


def cmd_run(args: argparse.Namespace) -> int:
    from car.agent import AgentConfig
    from car.harness import run_suite
    from car.images import build_image

    files = args.tasks or sorted(Path("tasks").glob("*.jsonl"))
    tasks = select_tasks([t for f in files for t in read_tasks(f)], args.limit, args.seed)
    if not tasks:
        print("no tasks found; run `car generate` first", file=sys.stderr)
        return 2
    targets = load_targets(args.targets)
    if args.sandbox == "local" and not args.unsafe_local_sandbox:
        print(
            "refusing to run agents without Docker: the model's edits are executed when the "
            "tests run. Use --sandbox docker, or pass --unsafe-local-sandbox if you accept "
            "running model-written code directly on this machine.",
            file=sys.stderr,
        )
        return 2
    if args.sandbox == "docker" and not docker_available():
        print("Docker is not running; start Docker Desktop and retry.", file=sys.stderr)
        return 2

    repos, factories = {}, {}
    for name in sorted({t.target for t in tasks}):
        target = targets[name]
        repos[name] = checkout(target, args.cache)
        if args.sandbox == "docker":
            image = build_image(target)
            factories[name] = lambda src, _i=image, _t=target: DockerSandbox(
                src, image=_i, pythonpath=_t.pythonpath
            )
        else:
            python = prepare_env(target, args.cache)
            factories[name] = lambda src, _p=python, _t=target: LocalSandbox(
                src, python=_p, pythonpath=_t.pythonpath
            )

    loaded = load_dotenv()
    if loaded:
        print(f"loaded from .env: {', '.join(loaded)}", file=sys.stderr)
    model = make_model(args)
    config = AgentConfig(max_steps=args.max_steps, max_cost_usd=args.max_cost_per_attempt)
    print(f"{model.name}: {len(tasks)} tasks x {args.attempts} attempts, budget ${args.budget:.2f}",
          file=sys.stderr)  # fmt: skip

    def progress(row: dict) -> None:
        mark = "solved" if row["score_solved"] else "failed"
        print(f"  {row['task']} #{row['attempt']}: {mark} ({row['stop']}, {row['steps']} steps, "
              f"${row['cost_usd']:.4f}, {row['seconds']:.0f}s)", file=sys.stderr)  # fmt: skip

    run_suite(tasks, model, targets, repos, factories, args.out, attempts=args.attempts,
              budget_usd=args.budget, workers=args.workers, config=config,
              on_result=progress)  # fmt: skip
    return cmd_report(argparse.Namespace(results=args.out))


def cmd_report(args: argparse.Namespace) -> int:
    from car.harness import load_results
    from car.metrics import format_report

    path = args.results / "results.jsonl" if args.results.is_dir() else args.results
    rows = load_results(path)
    if not rows:
        print(f"no results in {path}", file=sys.stderr)
        return 2
    print(format_report(rows))
    return 0


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

    run = sub.add_parser("run", help="run an agent on tasks and score the attempts")
    run.add_argument("--provider", choices=["claude", "local"], required=True)
    run.add_argument("--model", required=True,
                     help="e.g. claude-haiku-5-5, or the local server's model name")  # fmt: skip
    run.add_argument("--effort", default="medium", help="Claude effort level")
    run.add_argument("--base-url", default="http://localhost:11434/v1",
                     help="OpenAI-compatible endpoint for --provider local")  # fmt: skip
    run.add_argument("--tasks", nargs="*", type=Path, help="task files (default: tasks/*.jsonl)")
    run.add_argument("--limit", type=int, help="run a subset spread over targets")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--attempts", type=int, default=3)
    run.add_argument("--budget", type=float, default=2.0, help="USD; stops starting attempts")
    run.add_argument("--max-cost-per-attempt", type=float, default=0.50)
    run.add_argument("--max-steps", type=int, default=30)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--sandbox", choices=["docker", "local"], default="docker")
    run.add_argument("--unsafe-local-sandbox", action="store_true")
    run.add_argument("--out", type=Path, required=True, help="results directory")
    run.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    run.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="summarize a results directory")
    report.add_argument("results", type=Path)
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
