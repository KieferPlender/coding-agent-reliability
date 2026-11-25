"""Generate single-token bug candidates ("mutations") in Python source files.

Mutations are applied at the token level so the rest of the file keeps its exact
formatting: the resulting bug is a one-token diff, like a real typo-sized bug.
Every candidate is checked to still parse, so the agent never gets a syntax error
as its task.
"""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass

# Operator swaps that keep code syntactically valid and usually change behaviour.
OPERATOR_SWAPS: dict[str, tuple[str, ...]] = {
    "<": ("<=",),
    "<=": ("<",),
    ">": (">=",),
    ">=": (">",),
    "==": ("!=",),
    "!=": ("==",),
    "+": ("-",),
    "-": ("+",),
    "*": ("//",),
    "//": ("*",),
    "+=": ("-=",),
    "-=": ("+=",),
}

KEYWORD_SWAPS: dict[str, tuple[str, ...]] = {
    "and": ("or",),
    "or": ("and",),
    "True": ("False",),
    "False": ("True",),
    "is": ("is not",),
    "in": ("not in",),
}


@dataclass(frozen=True)
class Mutation:
    """One single-token change to one file."""

    path: str  # relative to the repository root, POSIX separators
    line: int  # 1-based
    col: int  # 0-based, as reported by tokenize
    original: str
    replacement: str
    operator: str  # short name of the mutation kind, for difficulty analysis

    @property
    def id(self) -> str:
        return f"{self.path}:{self.line}:{self.col}:{self.original}->{self.replacement}"


def _operator_kind(original: str) -> str:
    if original in ("<", "<=", ">", ">=", "==", "!="):
        return "comparison"
    if original in ("+", "-", "*", "//", "+=", "-="):
        return "arithmetic"
    if original in ("and", "or"):
        return "boolean"
    if original in ("True", "False"):
        return "constant"
    if original in ("is", "in"):
        return "membership"
    if original.isdigit():
        return "off_by_one"
    return "other"


def _replace_span(source: str, line: int, col: int, length: int, replacement: str) -> str:
    lines = source.splitlines(keepends=True)
    target = lines[line - 1]
    lines[line - 1] = target[:col] + replacement + target[col + length :]
    return "".join(lines)


def apply_mutation(source: str, mutation: Mutation) -> str:
    """Return `source` with `mutation` applied. Raises ValueError on mismatch."""
    lines = source.splitlines(keepends=True)
    if mutation.line > len(lines):
        raise ValueError(f"line {mutation.line} out of range for {mutation.path}")
    found = lines[mutation.line - 1][mutation.col : mutation.col + len(mutation.original)]
    if found != mutation.original:
        raise ValueError(
            f"expected {mutation.original!r} at {mutation.id}, found {found!r}; "
            "the file has changed since the mutation was generated"
        )
    return _replace_span(
        source, mutation.line, mutation.col, len(mutation.original), mutation.replacement
    )


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Lines covered by docstrings, which we never mutate."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                first = body[0]
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _annotation_lines(tree: ast.AST) -> set[tuple[int, int]]:
    """(line, col) positions inside type annotations, which rarely affect behaviour."""
    spans: set[tuple[int, int]] = set()

    def mark(node: ast.AST | None) -> None:
        if node is None:
            return
        for sub in ast.walk(node):
            if hasattr(sub, "lineno"):
                spans.add((sub.lineno, sub.col_offset))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            mark(node.returns)
            for arg in [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]:
                mark(arg.annotation)
        elif isinstance(node, ast.AnnAssign):
            mark(node.annotation)
    return spans


def find_mutations(source: str, path: str) -> list[Mutation]:
    """All valid single-token mutations of `source`, in file order."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    skip_lines = _docstring_lines(tree)
    annotations = _annotation_lines(tree)

    candidates: list[Mutation] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    prev: tokenize.TokenInfo | None = None
    for tok in tokens:
        line, col = tok.start
        if line in skip_lines or (line, col) in annotations:
            prev = tok
            continue
        replacements: tuple[str, ...] = ()
        if tok.type == tokenize.OP and tok.string in OPERATOR_SWAPS:
            # `*args`, `**kwargs` and keyword-only markers are not arithmetic.
            if tok.string == "*" and prev is not None and prev.string in ("(", ",", "lambda"):
                prev = tok
                continue
            replacements = OPERATOR_SWAPS[tok.string]
        elif tok.type == tokenize.NAME and tok.string in KEYWORD_SWAPS:
            # `not in` / `is not` are already negated; skip their first token.
            if tok.string in ("in", "is") and prev is not None and prev.string == "not":
                prev = tok
                continue
            # `is not` is already negated; `is` -> `is not` would give `is not not`.
            rest = source.splitlines()[line - 1][col + len(tok.string) :]
            if tok.string == "is" and rest.lstrip().startswith("not"):
                prev = tok
                continue
            # `for x in ...` is iteration, not membership.
            if tok.string == "in" and _is_for_target(source, line, col):
                prev = tok
                continue
            replacements = KEYWORD_SWAPS[tok.string]
        elif tok.type == tokenize.NUMBER and tok.string.isdigit() and len(tok.string) <= 3:
            value = int(tok.string)
            replacements = (str(value + 1),) if value == 0 else (str(value + 1), str(value - 1))
        for replacement in replacements:
            mutation = Mutation(
                path=path,
                line=line,
                col=col,
                original=tok.string,
                replacement=replacement,
                operator=_operator_kind(tok.string),
            )
            mutated = apply_mutation(source, mutation)
            try:
                ast.parse(mutated)
            except SyntaxError:
                continue
            candidates.append(mutation)
        prev = tok
    return candidates


def _is_for_target(source: str, line: int, col: int) -> bool:
    """True when the `in` at (line, col) belongs to a `for` loop or comprehension."""
    text = source.splitlines()[line - 1][:col]
    # Cheap and conservative: a `for` earlier on the same line (loops and
    # comprehensions) with no `if` after it means this `in` is iteration.
    head, sep, tail = text.rpartition(" for ")
    if not sep and not text.lstrip().startswith("for "):
        return False
    return " if " not in (tail if sep else text)
