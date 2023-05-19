import subprocess
from pathlib import Path
from typing import Final, Iterable

import clingo
import typeguard
from clingox.reify import reify_program
from dumbo_utils.primitives import PositiveIntegerOrUnbounded
from dumbo_utils.validation import validate

from dumbo_asp.primitives import SymbolicProgram, Model, SymbolicAtom, SymbolicRule, SymbolicTerm, GroundAtom


@typeguard.typechecked
def compute_minimal_unsatisfiable_subsets(
        program: SymbolicProgram,
        up_to: PositiveIntegerOrUnbounded = PositiveIntegerOrUnbounded.of(1),
        *,
        over_the_ground_program: bool = False,
        clingo: Path = Path("clingo"),
        wasp: Path = Path("wasp"),
) -> list[SymbolicProgram]:
    predicate: Final = f"__mus__"
    if over_the_ground_program:
        rules = [
            SymbolicRule.parse(f"__constant{predicate}({';'.join(str(term) for term in program.herbrand_universe)}).")
        ]
        for index, rule in enumerate(program, start=1):
            terms = ','.join([str(index), *rule.global_safe_variables])
            rules.append(rule.with_extended_body(SymbolicAtom.parse(f"{predicate}({terms})")))
            variables = '; '.join(f"__constant{predicate}({variable})" for variable in rule.global_safe_variables)
            rules.append(SymbolicRule.parse(f"{{{predicate}({terms})}} :- {variables}."))
        mus_program = SymbolicProgram.of(rules)
    else:
        mus_program = SymbolicProgram.of(
            *(rule.with_extended_body(SymbolicAtom.parse(f"{predicate}({index})"))
              for index, rule in enumerate(program, start=1)),
            SymbolicRule.parse(
                f"{{{predicate}(1..{len(program)})}}."
            ),
        )
    # print(mus_program)
    res = subprocess.run(
        ["bash", "-c", f"{clingo} --output=smodels | {wasp} --silent --mus={predicate} -n {up_to if up_to.is_int else 0}"],
        input=str(mus_program).encode(),
        capture_output=True,
    )
    validate("exit code", res.returncode, equals=0, help_msg="Computation failed")
    lines = res.stdout.decode().split('\n')
    muses = [Model.of_atoms(line.split()[2:]) for line in lines if line]
    if not over_the_ground_program:
        return [SymbolicProgram.of(program[atom.arguments[0].number - 1] for atom in mus) for mus in muses]
    res = []
    for mus in muses:
        rules = []
        for atom in mus:
            rule = program[atom.arguments[0].number - 1]
            rules.append(rule.apply_variable_substitution(**{
                variable: SymbolicTerm.parse(str(atom.arguments[index]))
                for index, variable in enumerate(rule.global_safe_variables, start=1)
            }))
        res.append(SymbolicProgram.of(rules))
    return res


@typeguard.typechecked
def validate_in_all_models(
        program: SymbolicProgram, *,
        true_atoms: Iterable[GroundAtom] = (),
        false_atoms: Iterable[GroundAtom] = ()
) -> None:
    the_program = Model.of_atoms(
        reify_program(
            Model.of_atoms(true_atoms, false_atoms).as_choice_rules +
            str(program)
        )
    ).as_facts + META_MODELS

    def check(mode: bool, atoms):
        control = clingo.Control(["--enum-mode=cautious" if mode else "--enum-mode=brave"])
        control.add(the_program)
        control.ground([("base", [])])

        def collect(model):
            collect.atoms = set(at for at in Model.of_atoms(model.symbols(shown=True)))
        collect.atoms = None

        control.solve(on_model=collect)
        for atom in atoms:
            validate(f"{mode} atom", atom in collect.atoms, equals=mode,
                     help_msg=f"Atom {atom} was expected to be {str(mode).lower()} in all models")

    check(True, true_atoms)
    check(False, false_atoms)


@typeguard.typechecked
def validate_cannot_be_true_in_any_stable_model(
        program: SymbolicProgram,
        atom: GroundAtom,
        *,
        local_prefix: str = "__",
) -> None:
    global_predicates = [f"{predicate.name}({','.join(f'X' + str(i) for i in range(predicate.arity))})"
                         for predicate in program.predicates if not predicate.name.startswith(local_prefix)]
    the_program = Model.of_atoms(
        reify_program(
            f"""
{program}

% enforce truth of the atom (in "there" world)
{{{atom}}}.
:- not {atom}.
            """
        )
    ).as_facts + META_HT_MODELS + """
% not an equilibrium model
%:- hold(L,h) : hold(L,t).
    \n""" + '\n'.join(f"""
%:- output({predicate},B), conjunction(B,t), not conjunction(B,h).  % instability is not due to global predicates
    """.strip() for predicate in global_predicates) + """
#show T : output(T,B), conjunction(B,t), not conjunction(B,h).
    """

    control = clingo.Control()
    control.add(the_program)
    control.ground([("base", [])])

    def collect(model):
        collect.atoms = Model.of_atoms(model.symbols(shown=True))\
            .filter(when=lambda at: at.predicate_name.startswith(local_prefix))
    collect.atoms = None

    control.solve(on_model=collect)
    validate("some witness", collect.atoms is None, equals=False,
             help_msg=f"Instability not guaranteed by local predicates")


@typeguard.typechecked
def validate_cannot_be_extended_to_stable_model(
        program: SymbolicProgram,
        *,
        true_atoms: Iterable[GroundAtom] = (),
        false_atoms: Iterable[GroundAtom] = (),
        local_prefix: str = "__",
) -> None:
    # it may work by adding a rule like   __fail :- true_atoms, not false_atoms, not __fail.  (for __fail being fresh)
    raise ValueError


META_MODELS = """
atom( A ) :- atom_tuple(_,A).
atom(|L|) :-          literal_tuple(_,L).
atom(|L|) :- weighted_literal_tuple(_,L).

{ hold(A) : atom(A) }.

conjunction(B) :- literal_tuple(B),
        hold(L) : literal_tuple(B, L), L > 0;
    not hold(L) : literal_tuple(B,-L), L > 0.

body(normal(B)) :- rule(_,normal(B)), conjunction(B).
body(sum(B,G))  :- rule(_,sum(B,G)),
    #sum { W,L :     hold(L), weighted_literal_tuple(B, L,W), L > 0 ;
           W,L : not hold(L), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

  hold(A) : atom_tuple(H,A)   :- rule(disjunction(H),B), body(B).
{ hold(A) : atom_tuple(H,A) } :- rule(     choice(H),B), body(B).

#show.
#show T : output(T,B), conjunction(B).

% avoid warnings
atom_tuple(0,0) :- #false.
conjunction(0) :- #false.
literal_tuple(0) :- #false.
literal_tuple(0,0) :- #false.
weighted_literal_tuple(0,0) :- #false.
weighted_literal_tuple(0,0,0) :- #false.
rule(0,0) :- #false.
"""

META_HT_MODELS = """
#const option=1.

atom( A ) :- atom_tuple(_,A).
atom(|L|) :-          literal_tuple(_,L).
atom(|L|) :- weighted_literal_tuple(_,L).

model(h). model(t).

{ hold(A,h) } :- atom(A),    option = 1.
{ hold(A,t) } :- atom(A).
:- hold(L,h), not hold(L,t).

conjunction(B,M) :- model(M), literal_tuple(B),
        hold(L,M) : literal_tuple(B, L), L > 0;
    not hold(L,t) : literal_tuple(B,-L), L > 0.

body(normal(B),M) :- rule(_,normal(B)), conjunction(B,M).
body(sum(B,G),M)  :- model(M), rule(_,sum(B,G)),
    #sum { W,L :     hold(L,M), weighted_literal_tuple(B, L,W), L > 0 ;
           W,L : not hold(L,t), weighted_literal_tuple(B,-L,W), L > 0 } >= G.

               hold(A,M) :  atom_tuple(H,A)   :- rule(disjunction(H),B), body(B,M).
hold(A,M); not hold(A,t) :- atom_tuple(H,A),     rule(     choice(H),B), body(B,M).

#show.
#show (T,M) : output(T,B), conjunction(B,M).

% avoid warnings
atom_tuple(0,0) :- #false.
conjunction(0) :- #false.
literal_tuple(0) :- #false.
literal_tuple(0,0) :- #false.
weighted_literal_tuple(0,0) :- #false.
weighted_literal_tuple(0,0,0) :- #false.
rule(0,0) :- #false.
"""