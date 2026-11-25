# coding-agent-reliability

**How reliably do coding agents fix bugs, and does a local model on a laptop come close to an API model?**

Public coding benchmarks are fixed sets of tasks that models may have seen during training, and they usually report whether a model succeeded *once*. This project measures something closer to what teams need before trusting an agent:

- **Fresh tasks from any repository.** Bugs are injected automatically into real, well-tested Python projects, so no model has seen them before.
- **Reliability, not luck.** Every task is run several times; the headline metric is *pass^k*, the chance an agent solves a task on **every** attempt.
- **Local vs API.** The same agent, tools and tasks run against Claude models and quantized open-weights models on a 16 GB MacBook, compared on solve rate, cost and time per *solved* bug.
- **Why agents fail.** Every run is traced, and failures are classified (wrong file, edited the tests, lost in test output, malformed tool calls, ...).

> **Status:** step 1 of 5 — task generation and sandboxing. See the [roadmap](#roadmap).

## How a task is made

1. Take a real project at a pinned commit, for example [`toolz`](https://github.com/pytoolz/toolz).
2. Make one small, realistic change to its source code: `<` becomes `<=`, `and` becomes `or`, `1` becomes `0`, ...
3. Run the project's own test suite. Keep the change only if the clean code passes and the changed code **fails** at least one test.

A real task generated from the small example project in [`tests/fixtures`](tests/fixtures/toyproj):

```diff
--- a/toyproj/calendar_utils.py
+++ b/toyproj/calendar_utils.py
@@ -8,7 +8,7 @@


 def days_in_month(year: int, month: int) -> int:
-    if month < 1 or month > 12:
+    if month <= 1 or month > 12:
         raise ValueError("month out of range")
     if month == 2 and is_leap(year):
         return 29
```

It breaks `test_days_in_month` and `test_clamp_day`: January is now rejected as out of range. Changing `day > last` to `day >= last` in the same file, by contrast, is never turned into a task: it returns the same value, so no test can notice it.

The agent then gets the broken repository and one instruction: *the tests fail, fix the source code without changing the tests.* Success is checked automatically by re-running the full test suite.

Step 3 filters out changes that are invisible to the tests, so every task has a real, observable bug and a clear definition of "fixed". Candidates are sampled evenly across bug types and files, so the task set isn't dominated by one kind of change or one large module.

## Targets

| Project | What it is | Candidate bugs |
|---|---|---|
| [toolz](https://github.com/pytoolz/toolz) | functional utilities | 418 |
| [boltons](https://github.com/mahmoud/boltons) | general-purpose utilities | 3,097 |
| [python-slugify](https://github.com/un33k/python-slugify) | URL slugs | 147 |
| [python-tabulate](https://github.com/astanin/python-tabulate) | text tables | 695 |
| [inflect](https://github.com/jaraco/inflect) | English plurals, ordinals, number words | 1,418 |

Candidate counts are before checking them against the tests. Targets are defined in [`targets/targets.toml`](targets/targets.toml); adding your own project takes a few lines.

## Usage

Requires Python 3.11+. Install into a virtual environment (Homebrew and most Linux distributions block system-wide `pip install`):

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e ".[dev]"
```

Generate up to 200 validated tasks per target (clones each target at its pinned commit and creates a virtualenv with its test dependencies):

```bash
car generate toolz --max-candidates 200
```

Summarize a task file:

```bash
car show tasks/toolz.jsonl
```

Tests can run in a local virtualenv (fast, for generating tasks) or in Docker (`--docker-image`), where each sandbox has **no network**, a memory limit and a process limit. Agent runs will always use Docker, because the model executes code.

## Design notes

- **Bugs are single-token edits applied to the original text**, not regenerated from a syntax tree, so the task diff is exactly one token and the file's formatting is untouched. Docstrings, type annotations and `*args` are never mutated; every candidate must still parse.
- **The target is never installed into its environment.** Tests import the sandbox copy via `PYTHONPATH`, so the injected bug is what the tests actually exercise.
- **Flaky tests can't create tasks.** The clean suite runs twice first; any test failing there is ignored when deciding whether a change broke something.
- **Everything is reproducible**: targets are pinned to commits, sampling is seeded, and each task stores its bug as a diff.

## Roadmap

1. ✅ **Task generation and sandboxing** — bug injection, validation against real test suites, local and Docker sandboxes.
2. **Baseline agent and harness** — a minimal agent loop (read, search, edit, run tests), full tracing, parallel runs with a cost cap, Claude via the Anthropic API and local models via an OpenAI-compatible server.
3. **First full run and failure analysis** — solve rate, pass^k, cost and time per solved task; failures grouped by cause.
4. **Targeted improvements** — one change at a time (shorter test output, a search tool, verify-before-finishing, ...), each measured on its own.
5. **Local vs API results** — 8B and 14B models on a 16 GB MacBook (MLX and llama.cpp) against Claude Haiku, Sonnet and Opus; write-up.

## License

MIT
