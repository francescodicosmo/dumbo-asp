from dumbo_asp.primitives import SymbolicProgram
from dumbo_asp.queries import compute_minimal_unsatisfiable_subsets
from dumbo_utils.primitives import PositiveIntegerOrUnbounded


def test_compute_minimal_unsatisfiable_subsets():
    program = SymbolicProgram.parse("""
a.
b.
c.
:- a, b.
:- a, c.
:- b, c.
    """)
    res = compute_minimal_unsatisfiable_subsets(program, PositiveIntegerOrUnbounded.of_unbounded())
    assert len(res) == 3


def test_compute_minimal_unsatisfiable_subsets_over_ground_program():
    program = SymbolicProgram.parse("""
a(1..3).
:- a(X), a(Y), X < Y.
    """)
    res = compute_minimal_unsatisfiable_subsets(program, PositiveIntegerOrUnbounded.of_unbounded())
    assert len(res) == 1
    res = compute_minimal_unsatisfiable_subsets(program, PositiveIntegerOrUnbounded.of_unbounded(),
                                                over_the_ground_program=True)
    assert len(res) == 3
