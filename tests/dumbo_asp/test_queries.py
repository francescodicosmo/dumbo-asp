import pytest
from dumbo_utils.primitives import PositiveIntegerOrUnbounded

from dumbo_asp.primitives import SymbolicProgram, Template, Model, GroundAtom
from dumbo_asp.queries import compute_minimal_unsatisfiable_subsets, validate_in_all_models, \
    validate_cannot_be_true_in_any_stable_model, validate_cannot_be_extended_to_stable_model, enumerate_models, \
    enumerate_counter_models, validate_in_all_models_of_the_reduct


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


def test_enumerate_models():
    program = SymbolicProgram.parse("""
{a; b; c; d}.
:- c, d.
:- not c, not d.
    """)
    models = enumerate_models(program, true_atoms=Model.of_atoms("a"), false_atoms=Model.of_atoms("b"))
    assert len(models) == 2


def test_enumerate_models_2():
    program = SymbolicProgram.parse("""
a :- b.
    """)
    models = enumerate_models(program, unknown_atoms=Model.of_atoms("a b".split()))
    assert len(models) == 3


def test_enumerate_counter_models():
    program = SymbolicProgram.parse("""
a :- b.
    """)
    models = enumerate_counter_models(program, Model.of_atoms("a b".split()))
    assert len(models) == 2


def test_validate_in_all_models_transitive_closure():
    program = Template.expand_program(SymbolicProgram.parse("""
__apply_template__("@dumbo/transitive closure", (relation, link), (closure, link)).

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


def test_validate_in_all_models_of_the_reduct_transitive_closure():
    program = Template.expand_program(SymbolicProgram.parse("""
__apply_template__("@dumbo/transitive closure", (relation, link), (closure, link)).
    """))

    validate_in_all_models_of_the_reduct(
        program=program,
        model=Model.of_atoms("link(a,b) link(b,c)".split()),
        true_atoms=Model.of_atoms("link(a,c)".split()),
    )

    validate_in_all_models_of_the_reduct(
        program=program,
        model=Model.of_atoms("link(a,b) link(b,c)".split()),
        true_atoms=Model.of_atoms("link(a,b) link(b,c) link(a,c)".split()),
    )

    validate_in_all_models_of_the_reduct(
        program=program,
        model=Model.of_atoms("link(a,b) link(b,c) link(c,d)".split()),
        true_atoms=Model.of_atoms("link(a,b) link(b,c) link(a,c) link(a,d)".split()),
    )

    validate_in_all_models_of_the_reduct(
        program=program,
        model=Model.of_atoms("link(a,b) link(b,c) link(c,d) link(d,a)".split()),
        true_atoms=Model.of_atoms("link(b,a) link(c,a) link(a,a)".split()),
    )

    with pytest.raises(ValueError):
        validate_in_all_models_of_the_reduct(
            program=program,
            model=Model.of_atoms("link(a,b) link(b,c)".split()),
            true_atoms=Model.of_atoms("link(a,a)".split()),
        )

    with pytest.raises(ValueError):
        validate_in_all_models_of_the_reduct(
            program=program,
            model=Model.of_atoms("link(a,b) link(b,c) link(c,d)".split()),
            true_atoms=Model.of_atoms("link(b,b)".split()),
        )


def test_validate_in_all_models_for_unseen_atoms():
    program = Template.expand_program(SymbolicProgram.parse("""
__apply_template__("@dumbo/transitive closure", (relation, link), (closure, link)).
link(a,b).
    """))

    with pytest.raises(ValueError):
        validate_in_all_models(program=program, false_atoms=Model.of_atoms("link(c,d)".split()))


def test_validate_cannot_be_true_in_any_stable_model():
    program = Template.expand_program(SymbolicProgram.parse("""
__fail :- a, not __fail.
    """))

    with pytest.raises(ValueError):
        validate_in_all_models(program=program, false_atoms=Model.of_atoms("a".split()))

    validate_cannot_be_true_in_any_stable_model(program, GroundAtom.parse("a"))


def test_validate_cannot_be_true_in_any_stable_model_2():
    program = Template.expand_program(SymbolicProgram.parse("""
__fail :- a, not __fail.
:- not a, b.

{a}.
    """))

    validate_cannot_be_true_in_any_stable_model(program, GroundAtom.parse("b"))


def test_validate_cannot_be_extended_to_stable_model():
    program = Template.expand_program(SymbolicProgram.parse("""
{a; b}.
__fail :- a, b, not __fail.
__fail :- not a, not b, not __fail.
    """))

    with pytest.raises(ValueError):
        validate_in_all_models(program=program, true_atoms=Model.of_atoms("a".split()))
        validate_in_all_models(program=program, true_atoms=Model.of_atoms("b".split()))
        validate_in_all_models(program=program, false_atoms=Model.of_atoms("a".split()))
        validate_in_all_models(program=program, false_atoms=Model.of_atoms("b".split()))

    validate_cannot_be_extended_to_stable_model(program=program, true_atoms=Model.of_atoms("a b".split()))
    validate_cannot_be_extended_to_stable_model(program=program, false_atoms=Model.of_atoms("a b".split()))
