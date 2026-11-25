import ast

import pytest

from car.mutate import Mutation, apply_mutation, find_mutations

SOURCE = '''"""Module docstring with < and == that must not be mutated."""


def f(a: int, b: int, *args, **kwargs) -> bool:
    """Docstring: a < b."""
    if a < b and b != 0:
        return True
    for x in args:
        if x in kwargs or x is not None:
            a += 1
    return a == 2
'''


def mutations_by_original(source: str) -> dict[str, list[Mutation]]:
    found: dict[str, list[Mutation]] = {}
    for m in find_mutations(source, "m.py"):
        found.setdefault(m.original, []).append(m)
    return found


def test_finds_expected_operator_swaps():
    found = mutations_by_original(SOURCE)
    assert [m.replacement for m in found["<"]] == ["<="]
    assert [m.replacement for m in found["and"]] == ["or"]
    assert [m.replacement for m in found["!="]] == ["=="]
    assert [m.replacement for m in found["+="]] == ["-="]
    assert [m.replacement for m in found["True"]] == ["False"]


def test_skips_docstrings_varargs_and_iteration():
    found = mutations_by_original(SOURCE)
    # Only the code `<`, not the ones in the two docstrings.
    assert len(found["<"]) == 1
    # `*args` is not multiplication.
    assert "*" not in found
    # `for x in args` is iteration; only the membership test `x in kwargs` mutates.
    assert len(found["in"]) == 1
    assert found["in"][0].line == 9


def test_every_mutation_parses_and_changes_one_token():
    for m in find_mutations(SOURCE, "m.py"):
        mutated = apply_mutation(SOURCE, m)
        ast.parse(mutated)
        changed = [
            (a, b) for a, b in zip(SOURCE.splitlines(), mutated.splitlines(), strict=True) if a != b
        ]
        assert len(changed) == 1


def test_is_not_is_left_alone():
    found = mutations_by_original("def g(x):\n    return x is not None\n")
    assert "is" not in found


def test_numbers_are_shifted_by_one():
    found = mutations_by_original("def g(xs):\n    return xs[1:]\n")
    assert sorted(m.replacement for m in found["1"]) == ["0", "2"]
    assert all(m.operator == "off_by_one" for m in found["1"])


def test_apply_rejects_stale_mutation():
    m = find_mutations(SOURCE, "m.py")[0]
    with pytest.raises(ValueError, match="has changed"):
        apply_mutation(SOURCE.replace("a < b", "a > b"), m)


def test_unparseable_source_yields_nothing():
    assert find_mutations("def broken(:\n", "m.py") == []
