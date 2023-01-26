import subprocess
import uuid
from pathlib import Path
from typing import Any, Optional, Final

import typeguard
from dumbo_asp.primitives import SymbolicProgram, GroundAtom, Model, SymbolicAtom, SymbolicRule, SymbolicTerm
from dumbo_utils.primitives import PositiveIntegerOrUnbounded
from dumbo_utils.validation import validate


@typeguard.typechecked
def compute_minimal_unsatisfiable_subsets(
        program: SymbolicProgram,
        up_to: PositiveIntegerOrUnbounded = PositiveIntegerOrUnbounded.of(1),
        *,
        over_the_ground_program: bool = False,
        clingo: Path = Path("clingo"),
        wasp: Path = Path("wasp"),
) -> list[SymbolicProgram]:
    predicate: Final = f"__mus_{str(uuid.uuid4()).replace('-', '_')}__"
    if over_the_ground_program:
        rules = []
        for index, rule in enumerate(program, start=1):
            terms = ','.join([str(index), *rule.global_safe_variables])
            rules.append(rule.with_extended_body(SymbolicAtom.parse(f"{predicate}({terms})")))
            rules.append(SymbolicRule.parse(f"{{{predicate}({terms})}} :- {rule.body_as_string()}."))
        mus_program = SymbolicProgram.of(rules)
    else:
        mus_program = SymbolicProgram.of(
            *(rule.with_extended_body(SymbolicAtom.parse(f"{predicate}({index})"))
              for index, rule in enumerate(program, start=1)),
            SymbolicRule.parse(
                f"{{{predicate}(1..{len(program)})}}."
            ),
        )
    res = subprocess.run(
        ["bash", "-c", f"clingo --output=smodels | wasp --silent --mus={predicate} -n {up_to if up_to.is_int else 0}"],
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
def explain_by_minimal_unsatisfiable_subsets(
        program: SymbolicProgram,
        answer_set: Model,
        atom: GroundAtom,
        up_to: PositiveIntegerOrUnbounded = PositiveIntegerOrUnbounded.of(1),
        *,
        over_the_ground_program: bool = False,
        clingo: Path = Path("clingo"),
        wasp: Path = Path("wasp"),
) -> list[SymbolicProgram]:
    rules = [rule for rule in program]
    true_atoms = {atom for atom in answer_set}
    for base_atom in program.herbrand_base:
        if base_atom == atom:
            if base_atom in true_atoms:
                rule = f":- %* truth of {atom} is implied by the above rules and... *%  {atom}."
            else:
                rule = f":- %* falsity of {atom} is implied by the above rules and... *%  not {atom}."
        else:
            if base_atom in true_atoms:
                rule = f":- %* by {base_atom} in the answer set *% not  {base_atom}."
            else:
                rule = f":- %* by not {base_atom} in the answer set *%  {base_atom}."
        rules.append(SymbolicRule.parse(rule))
    extended_program = SymbolicProgram.of(rules)
    return compute_minimal_unsatisfiable_subsets(
        extended_program,
        up_to,
        over_the_ground_program=over_the_ground_program,
        clingo=clingo,
        wasp=wasp,
    )
