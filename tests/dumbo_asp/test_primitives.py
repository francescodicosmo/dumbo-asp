import pytest

from dumbo_asp.primitives import Predicate, Parser, GroundAtom, Model, SymbolicRule


def test_parser_error():
    with pytest.raises(Parser.Error) as err:
        Parser.parse_ground_term("\"a\nb")
    assert err.value.line == 1


@pytest.mark.parametrize("name", [
    "foo//1",
    "foo/-1",
    "foo bar",
    "Foo",
    "-foo",
])
def test_invalid_predicate_name(name):
    with pytest.raises(ValueError):
        Predicate.parse(name)


@pytest.mark.parametrize("name", [
    "foo/1",
    "foo/0",
    "foo",
])
def test_valid_predicate_name(name):
    assert Predicate.parse(name).name == name.split('/')[0]


def test_predicate_order():
    assert Predicate.parse("a/0") < Predicate.parse("a/1")
    assert Predicate.parse("a/1") > Predicate.parse("a/0")
    assert Predicate.parse("a/1") < Predicate.parse("b/0")
    assert Predicate.parse("a") > Predicate.parse("a/0")


@pytest.mark.parametrize("atom", [
    "foo(",
    "foo(1,)",
    "foo(_)",
    "foo(X)",
    "foo(1) bar(2)",
])
def test_invalid_ground_atom(atom):
    with pytest.raises(ValueError):
        GroundAtom.parse(atom)


@pytest.mark.parametrize("atom", [
    "foo",
    "foo(1)",
    "foo(x)",
])
def test_valid_ground_atom(atom):
    assert GroundAtom.parse(atom).predicate.name == atom.split('(')[0]


def test_ground_atom_order():
    assert GroundAtom.parse("a(1)") < GroundAtom.parse("a(2)")
    assert GroundAtom.parse("a(b)") > GroundAtom.parse("a(2)")
    assert GroundAtom.parse("a(b)") > GroundAtom.parse("a(a)")
    assert GroundAtom.parse("a(b)") > GroundAtom.parse("a(\"a\")")
    assert GroundAtom.parse("c(\"b\")") < GroundAtom.parse("c(a)")
    assert GroundAtom.parse("a(-1)") < GroundAtom.parse("a(0)")
    assert GroundAtom.parse("a(-a)") < GroundAtom.parse("a(a)")


@pytest.mark.parametrize("rule", [
    "a :- b.",
    "a(X) :- b(X).",
    "a(X) :- b(Y).",
])
def test_parse_valid_rule(rule):
    assert str(SymbolicRule.parse(rule)) == rule


@pytest.mark.parametrize("rule", [
    "a :- b.\na(X) :- b(X).",
    "a(X) :- b(.",
])
def test_parse_invalid_rule(rule):
    with pytest.raises(ValueError):
        SymbolicRule.parse(rule)


@pytest.mark.parametrize("program", [
    "a",
    "a : -- b.",
    "a\n\n : --\n\n b.",
])
def test_parse_invalid_program(program):
    with pytest.raises(ValueError):
        Parser.parse_program(program)


@pytest.mark.parametrize("atoms", [
    ["a", "b", "c"],
    ["a", "-b", "c"],
    ["a(1)", "a(2)", "a(b)"],
])
def test_valid_model_of_atoms(atoms):
    assert len(Model.of_atoms(atoms)) == len(atoms)
    assert len(Model.of_atoms(*atoms)) == len(atoms)


@pytest.mark.parametrize("atoms", [
    ["\"a\"", "b", "c"],
    ["a", "not b", "c"],
    ["a(X)", "a(2)", "a(b)"],
])
def test_invalid_model_of_atoms(atoms):
    with pytest.raises(ValueError):
        Model.of_atoms(atoms)


def test_model_drop():
    assert len(Model.of_atoms("a(1)", "a(1,2)", "b(1)", "c(2)").drop(Predicate.parse("b"))) == 3
    assert len(Model.of_atoms("a(1)", "a(1,2)", "b(1)", "c(2)").drop(Predicate.parse("a"))) == 2
    assert len(Model.of_atoms("a(1)", "a(1,2)", "b(1)", "c(2)").drop(Predicate.parse("a/2"))) == 3
