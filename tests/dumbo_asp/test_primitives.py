import clingo
import clingo.ast
import pytest

from dumbo_asp.primitives import Predicate, Parser, GroundAtom, Model, SymbolicRule, SymbolicProgram, SymbolicAtom, \
    SymbolicTerm


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
    "a\n\t:- b.",
    "a :- b( 1 ).",
])
def test_parse_valid_symbolic_rule(rule):
    assert str(SymbolicRule.parse(rule)) == rule


@pytest.mark.parametrize("rule", [
    "a :- b.\na(X) :- b(X).",
    "a(X) :- b(.",
])
def test_parse_invalid_symbolic_rule(rule):
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
    assert len(Model.of_elements("a(1)", "1", '"a(1)"').drop(Predicate.parse("a/1"))) == 2
    assert len(Model.of_elements("a(1)", "1", 2, '"a(1)"').drop(numbers=True)) == 3
    assert len(Model.of_elements("a(1)", "1", 2, '"a(1)"').drop(numbers=True, strings=True)) == 1


def test_model_of_control():
    control = clingo.Control()
    control.add("base", [], "c. a. b.")
    control.ground([("base", [])])
    model = Model.of_control(control)
    assert len(model) == 3
    assert model[0].predicate == Predicate.parse("a/0")
    assert model[1].predicate == Predicate.parse("b/0")
    assert model[2].predicate == Predicate.parse("c/0")


def test_model_of_control_with_show_numbers():
    control = clingo.Control()
    control.add("base", [], "#show 1.")
    control.ground([("base", [])])
    model = Model.of_control(control)
    assert len(model) == 1
    assert model[0] == 1


def test_no_model():
    control = clingo.Control()
    control.add("base", [], "a :- not a.")
    control.ground([("base", [])])
    with pytest.raises(ValueError):
        Model.of_control(control)


def test_model_of_control_cannot_be_used_for_more_than_one_model():
    control = clingo.Control(["0"])
    control.add("base", [], "{a}.")
    control.ground([("base", [])])
    with pytest.raises(ValueError):
        Model.of_control(control)


def test_model_as_facts():
    assert Model.of_atoms("a", "b", "c").as_facts == "a.\nb.\nc."


def test_model_block_up():
    assert Model.of_atoms("a", "b").block_up == ":- a, b."


def test_model_project():
    assert Model.of_atoms("a(1,2,3)").project(Predicate.parse("a/3"), 1).as_facts == "a(2,3)."


def test_model_substitute():
    assert Model.of_atoms("a(1,2,3)").substitute(Predicate.parse("a/3"), 1, Parser.parse_ground_term("5")).as_facts == \
           "a(5,2,3)."


def test_model_of_elements():
    assert Model.of_elements(1, "2", "\"3\"").as_facts == """
__number(1).
__string(\"2\").
__string(\"3\").
    """.strip()


def test_parse_symbolic_program():
    string = """
foo(X) :-
    bar(X,Y);
    not buzz(Y).
b :-  a.
    """.strip()
    program = SymbolicProgram.parse(string)
    assert str(program) == string
    assert len(program) == 2
    assert str(program[-1]) == string.split('\n')[-1]


def test_symbolic_rule_head_variables():
    assert SymbolicRule.parse("a(X) :- b(X,Y).").head_variables == ("X",)
    assert SymbolicRule.parse("{a(X,Z) : c(Z)} = 1 :- b(X,Y).").head_variables == ("X", "Z")


def test_symbolic_rule_body_variables():
    assert SymbolicRule.parse("a(X) :- b(X,Y).").body_variables == ("X", "Y")
    assert SymbolicRule.parse("{a(X,Z) : c(Z)} = 1 :- b(X,Y), not c(W).").body_variables == ("W", "X", "Y")


def test_symbolic_rule_global_safe_variables():
    assert SymbolicRule.parse("a(X) :- b(X,Y).").global_safe_variables == ("X", "Y")
    assert SymbolicRule.parse("a(X,Y) :- b(X).").global_safe_variables == ("X",)
    assert SymbolicRule.parse("a(X) :- b(X), not c(Y).").global_safe_variables == ("X",)
    assert SymbolicRule.parse("a(X) :- X = #count{Y : b(Y)} = X.").global_safe_variables == ("X",)


def test_symbolic_rule_with_extended_body():
    assert str(SymbolicRule.parse("a.").with_extended_body(SymbolicAtom.parse("b"))) == "a :- b."
    assert str(SymbolicRule.parse("a :- b.").with_extended_body(SymbolicAtom.parse("c"), clingo.ast.Sign.Negation)) == \
           "a :- b; not c."
    assert str(SymbolicRule.parse(" a( X , Y ) . ").with_extended_body(SymbolicAtom.parse(" b( Z ) "))) == \
           "a( X , Y )  :- b( Z )."


def test_symbolic_rule_body_as_string():
    assert SymbolicRule.parse("a :- b, c.").body_as_string() == "b; c"


def test_symbolic_rule_apply_variable_substitution():
    assert str(SymbolicRule.parse("a(X) :- b(X,Y).").apply_variable_substitution(X=SymbolicTerm.of_int(1))) == \
           "a(1) :- b(1,Y)."


def test_symbolic_term_parse():
    assert str(SymbolicTerm.parse("1")) == "1"


def test_program_herbrand_universe():
    assert SymbolicProgram.parse("a(X) :- X = 1..3.").herbrand_universe == {SymbolicTerm.of_int(x) for x in range(1, 4)}
    assert SymbolicProgram.parse("a(X,Y) :- X = 1..3, Y = 4..5.").herbrand_universe == \
           {SymbolicTerm.of_int(x) for x in range(1, 6)}
    assert SymbolicProgram.parse("a(b(c)).").herbrand_universe == {SymbolicTerm.parse("c")}


def test_program_herbrand_base():
    assert SymbolicProgram.parse("a(X) :- X = 1..3.").herbrand_base == Model.of_program("a(1..3).")


def test_symbolic_rule_is_fact():
    assert SymbolicRule.parse("a.").is_fact
    assert SymbolicRule.parse("a(1).").is_fact
    assert SymbolicRule.parse("a(x).").is_fact
    assert SymbolicRule.parse("a(X).").is_fact
    assert not SymbolicRule.parse("a | b.").is_fact


def test_symbolic_program_process_constants():
    assert str(SymbolicProgram.parse("""
__const__(x, 10).
a(x).
    """.strip()).process_constants()) == """%* __const__(x, 10). *%\na(10)."""


def test_symbolic_program_process_with_statements():
    assert str(SymbolicProgram.parse("""
__with__(foo(X)).
    a(X).
    b(X,Y) :- c(Y).
__end_with__.
    """.strip()).process_with_statements()) == """
%* __with__(foo(X)). *%
a(X) :- foo(X).
b(X,Y) :- c(Y); foo(X).
%* __end_with__. *%
""".strip()


def test_():
    assert Model.of_program(SymbolicProgram.parse("""
a(b(0),1).
__show__(
    a(
        X,
        Y
    ), 
    a(X,Y)
).
    """).process_constants()) == Model.of_atoms("a(10)")
