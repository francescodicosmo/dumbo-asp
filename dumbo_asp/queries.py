import subprocess
from pathlib import Path
from typing import Final

import typeguard
from dumbo_utils.primitives import PositiveIntegerOrUnbounded
from dumbo_utils.validation import validate

from dumbo_asp.primitives import SymbolicProgram, Model, SymbolicAtom, SymbolicRule, SymbolicTerm


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
