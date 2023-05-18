import pytest
from dumbo_utils.primitives import PositiveIntegerOrUnbounded

from dumbo_asp.primitives import SymbolicProgram, Module, Model, GroundAtom
from dumbo_asp.queries import compute_minimal_unsatisfiable_subsets, validate_in_all_models, \
    validate_cannot_be_true_in_any_stable_model


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


def test_validate_in_all_models_transitive_closure():
    program = Module.expand_program(SymbolicProgram.parse("""
__apply_module__("@dumbo/transitive closure", (relation, link), (closure, link)).

link(a,b).
link(b,c).
    """))

    validate_in_all_models(
        program=program,
        true_atoms=Model.of_atoms("link(a,b) link(b,c) link(a,c)".split()),
    )

    with pytest.raises(ValueError):
        validate_in_all_models(program=program, true_atoms=Model.of_atoms("link(a,a)".split()))

    with pytest.raises(ValueError):
        validate_in_all_models(
            program=program,
            false_atoms=Model.of_atoms("link(a,a)".split()),
        )


def test_validate_in_all_models_transitive_closure_guaranteed():
    program = Module.expand_program(SymbolicProgram.parse("""
__apply_module__("@dumbo/transitive closure guaranteed", (relation, link), (closure, link)).

link(a,b).
link(b,c).
    """))

    with pytest.raises(ValueError):
        validate_in_all_models(program=program, true_atoms=Model.of_atoms("link(a,a)".split()))

    validate_cannot_be_true_in_any_stable_model(program, GroundAtom.parse("link(a,a)"))

    with pytest.raises(ValueError):
        validate_cannot_be_true_in_any_stable_model(program, GroundAtom.parse("link(a,c)"))

