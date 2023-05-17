from unittest.mock import patch

import clingo
import clingo.ast
import pytest

from dumbo_asp.primitives import Predicate, Parser, GroundAtom, Model, SymbolicRule, SymbolicProgram, SymbolicAtom, \
    SymbolicTerm, Module


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


def test_symbolic_rule_predicates():
    assert set(SymbolicRule.parse("a(X) :- b(X), not c(X).").predicates) == \
           set(Predicate.parse(p) for p in "a b c".split())


def test_symbolic_program_predicates():
    assert set(SymbolicProgram.parse("""
a(X) :- b(X), not c(X).
:- #sum{X,d(X) : e(X)} = Y.
    """.strip()).predicates) == set(Predicate.parse(p) for p in "a b c e".split())


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


def test_expand_one_global_variable():
    rule = SymbolicRule.parse("""
block((row, Row), (Row, Col)) :- Row = 1..9, Col = 1..9.
    """.strip())
    program = SymbolicProgram.of(rule)
    rules = rule.expand_global_safe_variables(variables=["Row"], herbrand_base=program.herbrand_base)
    assert len(rules) == 9


def test_expand_all_global_variables():
    rule = SymbolicRule.parse("""
block((row, Row), (Row, Col)) :- Row = 1..9, Col = 1..9.
    """.strip())
    program = SymbolicProgram.of(rule)
    rules = rule.expand_global_safe_variables(variables=rule.global_safe_variables, herbrand_base=program.herbrand_base)
    assert len(rules) == 9 * 9


def test_expand_non_global_variable():
    rule = SymbolicRule.parse("""
block((row, Row), (Row, Col)) :- Row = 1..9, Col = 1..9.
        """.strip())
    program = SymbolicProgram.of(rule)
    with pytest.raises(ValueError):
        rule.expand_global_safe_variables(variables=["X"], herbrand_base=program.herbrand_base)


def test_expand_global_variables_may_need_extra_variables():
    rule = SymbolicRule.parse("""
block((sub, Row', Col'), (Row, Col)) :- Row = 1..9; Col = 1..9; Row' = (Row-1) / 3; Col' = (Col-1) / 3.
    """.strip())
    program = SymbolicProgram.of(rule)
    rules = rule.expand_global_safe_variables(variables=["Row'", "Col'"], herbrand_base=program.herbrand_base)
    assert len(rules) == 9


def test_expand_global_variables_in_program():
    program = SymbolicProgram.parse("""
block((row, Row), (Row, Col)) :- Row = 1..9, Col = 1..9.
block((col, Col), (Row, Col)) :- Row = 1..9, Col = 1..9.
    """.strip())
    program = program.expand_global_safe_variables(rule=program[0], variables=["Row"])
    assert len(program) == 9 + 1
    program = program.expand_global_safe_variables(rule=program[-1], variables=["Col"])
    assert len(program) == 9 + 9


def test_symbolic_term_int():
    term = SymbolicTerm.parse("123")
    assert term.is_int()
    assert term.int_value() == 123


def test_symbolic_term_string():
    term = SymbolicTerm.parse('"foo"')
    assert term.is_string()
    assert str(term) == '"foo"'


def test_symbolic_term_function():
    term = SymbolicTerm.parse("foo")
    term = SymbolicTerm.parse("foo(bar)")
    assert term.is_function()
    assert term.function_name == "foo"
    assert term.function_arity == 1


def test_symbolic_atom_match():
    atom1 = SymbolicAtom.parse("foo(bar)")
    atom2 = SymbolicAtom.parse("foo(X)")
    assert atom1.match(atom2)


def test_symbolic_atom_match_nested():
    atom1 = SymbolicAtom.parse("foo(bar(buzz))")
    atom2 = SymbolicAtom.parse("foo(bar(X))")
    assert atom1.match(atom2)


def test_program_move_up():
    program = SymbolicProgram.parse("""
given((1, 1), 6).
given((1, 3), 9).    
given((2, 9), 1).
given((7, 3), 4).
given((7, 4), 7).
given((8, 9), 8).
given((9, 7), 7).
given((9, 8), 1).
    """)
    program = program.move_up(SymbolicAtom.parse("""
given((7, Col), Value)
    """))
    assert program[0] == SymbolicRule.parse("given((7, 3), 4).")


def test_query_herbrand_base():
    program = SymbolicProgram.parse("""
block((sub, Row', Col'), (Row, Col)) :- Row = 1..9; Col = 1..9; Row' = (Row-1) / 3; Col' = (Col-1) / 3.
    """)
    res = program.query_herbrand_base(
        "Row, Col",
        "block((sub, Row', Col'), (Row, Col)), block((sub, Row', Col'), (7, 9))"
    )
    assert len(res) == 9


def test_expand_conditional_literal():
    program = SymbolicProgram.parse("""
{a(1..3)}.
b :- a(X) : X = 1..3.
    """)
    program = program.expand_global_and_local_variables()
    assert str(program) == """
{ a(1); a(2); a(3) }.
b :- a(1); a(2); a(3).
    """.strip()


def test_expand_negative_conditional_literal():
    program = SymbolicProgram.parse("""
{a(1..3)}.
b :- not a(X) : X = 1..3.
    """)
    program = program.expand_global_and_local_variables()
    assert str(program) == """
{ a(1); a(2); a(3) }.
b :- not a(1); not a(2); not a(3).
    """.strip()


def test_expand_conditional_literal_in_aggregate():
    program = SymbolicProgram.parse("""
{a(1..3)}.
b :- #sum{X : a(X)} >= 3.
    """)
    program = program.expand_global_and_local_variables()
    assert str(program) == """
{ a(1); a(2); a(3) }.
b :- 3 <= #sum { X: a(X) }.
    """.strip()


def test_expand_skips_disabled_rules_by_default():
    program = SymbolicProgram.of(SymbolicRule.parse("""
{a(X) : X = 1..3}.
    """).disable())
    program = program.expand_global_and_local_variables()
    assert str(program) == """
%* {a(X) : X = 1..3}. *%
    """.strip()


def test_expand_disabled_rule():
    program = SymbolicProgram.of(SymbolicRule.parse("""
{a(X) : X = 1..3}.
    """).disable())
    program = program.expand_global_and_local_variables(expand_also_disabled_rules=True)
    assert str(program) == """
%* { a(1); a(2); a(3) }. *%
    """.strip()


def test_expand_disabled_rule_into_several_rules():
    program = SymbolicProgram.of(SymbolicRule.parse("""
a(X) :- X = 1..3.
    """).disable())
    program = program.expand_global_and_local_variables(expand_also_disabled_rules=True)
    assert str(program) == """
%* a(1) :- 1 = (1..3). *%
%* a(2) :- 2 = (1..3). *%
%* a(3) :- 3 = (1..3). *%
    """.strip()


def test_expand_global_variables_in_rule_with_negation():
    program = SymbolicProgram.parse("""
{b(1)}.
{c(1)}.
a(X) :- b(X), not c(X).
    """)
    program = program.expand_global_safe_variables(rule=program[-1], variables=["X"])
    assert len(program) == 3


def test_predicate_renaming_in_symbolic_rule():
    rule = SymbolicRule.parse("a(b) :- b(a).")
    rule = rule.apply_predicate_renaming(a=Predicate.parse("c"))
    assert str(rule) == "c(b) :- b(a)."


def test_predicate_renaming_in_symbolic_program():
    program = SymbolicProgram.parse("""
a(b) :- b(a).
:- foo, bar, a(a), not a(0).
    """.strip()).apply_predicate_renaming(a=Predicate.parse("c"))
    assert str(program) == """
c(b) :- b(a).
#false :- foo; bar; c(a); not c(0).
    """.strip()


def test_module_str():
    program = SymbolicProgram.parse("""
a :- not __a.
__a :- not a.
    """.strip())
    module = Module(name=Module.Name.parse("main"), program=program)
    assert str(module) == """
__module__(main).
a :- not __a.
__a :- not a.
__end__.
    """.strip()


def test_module_instantiation_can_rename_global_predicates_as_local_predicates():
    module = Module(name=Module.Name.parse("main"), program=SymbolicProgram.parse("foo."))
    assert str(module.instantiate(foo=Predicate.parse("__foo"))) == "__foo."


def test_module_instantiation_cannot_rename_local_predicates():
    module = Module(name=Module.Name.parse("main"), program=SymbolicProgram.of())
    with pytest.raises(ValueError):
        module.instantiate(__foo=Predicate.parse("bar"))


@patch("dumbo_asp.utils.uuid", side_effect=[
    "ebc40a28_de77_494a_a139_972343be51a8",
    "29f7b13e_a41f_4de4_944a_5fb8e61b513c",
    "ae2179b9_607d_44ed_b51c_08b590679ca1",
])
def test_module_instantiation(uuid_patch):
    program = SymbolicProgram.parse("""
pred :- not __false.
__false :- not pred.
__static_foo.
    """.strip())
    module = Module(name=Module.Name.parse("main"), program=program)
    assert str(module.instantiate(pred=Predicate.parse("a"))) == f"""
a :- not __false_29f7b13e_a41f_4de4_944a_5fb8e61b513c.
__false_29f7b13e_a41f_4de4_944a_5fb8e61b513c :- not a.
_static_foo_ebc40a28_de77_494a_a139_972343be51a8.
    """.strip()
    assert str(module.instantiate(pred=Predicate.parse("b"))) == f"""
b :- not __false_ae2179b9_607d_44ed_b51c_08b590679ca1.
__false_ae2179b9_607d_44ed_b51c_08b590679ca1 :- not b.
_static_foo_ebc40a28_de77_494a_a139_972343be51a8.
    """.strip()


@patch("dumbo_asp.utils.uuid", side_effect=[
    "ebc40a28_de77_494a_a139_972343be51a8",
    "29f7b13e_a41f_4de4_944a_5fb8e61b513c",
    "ae2179b9_607d_44ed_b51c_08b590679ca1",
])
def test_module_expand_program(uuid_patch):
    program = SymbolicProgram.parse("""
__module__(choice).
    predicate(X) :- condition(X), not __false(X).
    __false(X) :- condition(X), not predicate(X).
__end__.

edb(1..3).
__apply_module__(choice, (predicate, a), (condition, edb)).
    """)
    program = Module.expand_program(program)
    assert str(program) == """
edb(1..3).
%* __apply_module__(choice, (predicate, a), (condition, edb)). *%
a(X) :- edb(X); not __false_29f7b13e_a41f_4de4_944a_5fb8e61b513c(X).
__false_29f7b13e_a41f_4de4_944a_5fb8e61b513c(X) :- edb(X); not a(X).
%* __end__. *%
    """.strip()


def test_module_expand_program_requires_modules_to_be_declared_before_they_are_expanded():
    program = SymbolicProgram.parse("""
edb(1..3).
__apply_module__(choice, (predicate, a), (condition, edb)).

__module__(choice).
    predicate(X) :- condition(X), not __false(X).
    __false(X) :- condition(X), not predicate(X).
__end__.
    """)
    with pytest.raises(KeyError):
        Module.expand_program(program)


def test_module_expand_program_cannot_expand_a_module_inside_itself():
    program = SymbolicProgram.parse("""
__module__(foo).
    __apply_module__(foo).
__end__.
    """)
    with pytest.raises(KeyError):
        Module.expand_program(program)


@patch("dumbo_asp.utils.uuid", side_effect=[
    "ebc40a28_de77_494a_a139_972343be51a8",
    "29f7b13e_a41f_4de4_944a_5fb8e61b513c",
    "ae2179b9_607d_44ed_b51c_08b590679ca1",
    "c21258ac_b2ed_4d20_a7a1_ae72332b3f55",
    "3a412a2f_5d34_4ebc_ae4b_d4a83bfd6107",
    "88456e97_a0e1_4c79_b470_bd7eb3537561",
])
def test_module_expand_apply_modules_inside_modules(uuid_patch):
    program = SymbolicProgram.parse("""
__module__(transitive_closure).
    tc(X,Y) :- r(X,Y).
    tc(X,Z) :- tc(X,Y), r(Y,Z).
__end__.

__module__(transitive_closure_check).
    __apply_module__(transitive_closure, (tc, __tc)).
    :- __tc(X,X).
__end__.


link(a,b).
link(b,a).
__apply_module__(transitive_closure_check, (r, link)).
    """.strip())
    program = Module.expand_program(program)
    assert str(program) == """
link(a,b).
link(b,a).
%* __apply_module__(transitive_closure_check, (r, link)). *%
%* __apply_module__(transitive_closure,(tc,__tc)). *%
__tc_c21258ac_b2ed_4d20_a7a1_ae72332b3f55(X,Y) :- link(X,Y).
__tc_c21258ac_b2ed_4d20_a7a1_ae72332b3f55(X,Z) :- __tc_c21258ac_b2ed_4d20_a7a1_ae72332b3f55(X,Y); link(Y,Z).
%* __end__. *%
#false :- __tc_c21258ac_b2ed_4d20_a7a1_ae72332b3f55(X,X).
%* __end__. *%
    """.strip()
    with pytest.raises(Model.NoModelError):
        Model.of_program(program)
