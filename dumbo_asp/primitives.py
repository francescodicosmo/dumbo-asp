import copy
import dataclasses
import functools
import math
from dataclasses import InitVar
from functools import cached_property, cache
from typing import Callable, Optional, Iterable, Union, Any, Final

import clingo
import clingo.ast
import typeguard
from dumbo_utils.primitives import PrivateKey
from dumbo_utils.validation import validate, ValidationError

from dumbo_asp import utils
from dumbo_asp.utils import uuid


@typeguard.typechecked
class Parser:
    @dataclasses.dataclass(frozen=True)
    class Error(ValueError):
        parsed_string: str
        line: int
        begin: int
        end: int
        message: str

        key: InitVar[PrivateKey]
        __key = PrivateKey()

        def __post_init__(self, key: PrivateKey):
            self.__key.validate(key)

        @staticmethod
        def parse(error: str, parsed_string: str) -> "Parser.Error":
            parts = error.split(':', maxsplit=3)
            validate("prefix", parts[0], equals="<string>", help_msg="Unexpected source")
            validate("error", parts[3].startswith(" error: "), equals=True, help_msg="Unexpected error")
            begin, end = Parser.parse_range(parts[2])
            return Parser.Error(
                parsed_string=parsed_string,
                line=int(parts[1]),
                begin=begin,
                end=end,
                message=parts[3][len(" error: "):],
                key=Parser.Error.__key,
            )

        def drop(self, *, first: int = 0, last: int = 0) -> "Parser.Error":
            validate("one line", self.line, equals=1, help_msg="Can drop only from one line parsing")
            return Parser.Error(
                parsed_string=self.parsed_string[first:len(self.parsed_string) - last],
                line=self.line,
                begin=self.begin - first,
                end=self.end - first,
                message=self.message,
                key=Parser.Error.__key,
            )

        def __str__(self):
            lines = self.parsed_string.split('\n')
            width = math.floor(math.log10(len(lines))) + 1
            res = [f"Parsing error in line {self.line}, columns {self.begin}-{self.end}"]
            for line_index, the_line in enumerate(lines, start=1):
                res.append(f"{str(line_index).zfill(width)}| {the_line}")
                if line_index == self.line:
                    res.append('>' * width + '| ' + ' ' * (self.begin - 1) + '^' * (self.end - self.begin + 1))
            res.append(f"error: {self.message}")
            return '\n'.join(res)

    @staticmethod
    def parse_range(string: str) -> tuple[int, int]:
        parts = string.split('-', maxsplit=1)
        if len(parts) == 1:
            return int(parts[0]), int(parts[0])
        return int(parts[0]), int(parts[1])

    @staticmethod
    def parse_ground_term(string: str) -> clingo.Symbol:
        try:
            return clingo.parse_term(string)
        except RuntimeError as err:
            raise Parser.Error.parse(str(err), string)

    @staticmethod
    def parse_program(string: str) -> list[clingo.ast.AST]:
        def callback(ast):
            callback.res.append(ast)
        callback.res = []

        messages = []
        try:
            clingo.ast.parse_string(string, callback, logger=lambda code, message: messages.append((code, message)))
            validate("nonempty res", callback.res, min_len=1)
            validate("base program", callback.res[0].ast_type == clingo.ast.ASTType.Program and
                     callback.res[0].name == "base" and len(callback.res[0].parameters) == 0, equals=True)
            validate("only rules", [x for x in callback.res[1:] if x.ast_type != clingo.ast.ASTType.Rule], empty=True)
            return callback.res[1:]
        except RuntimeError:
            errors = [message[1] for message in messages if message[0] == clingo.MessageCode.RuntimeError]
            validate("errors", messages, length=1)
            raise Parser.Error.parse(errors[0], string)


@functools.total_ordering
@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Predicate:
    name: str
    arity: Optional[int]

    key: InitVar[PrivateKey]
    __key = PrivateKey()

    MAX_ARITY = 999

    def __post_init__(self, key: PrivateKey):
        self.__key.validate(key)

    @staticmethod
    def parse(name: str, arity: Optional[int] = None) -> "Predicate":
        split = name.split('/', maxsplit=1)
        if len(split) == 2:
            validate("arity not given", arity is None, equals=True, help_msg="The arity is given already in the name")
            name, arity = split[0], int(split[1])

        term = Parser.parse_ground_term(name)
        validate("name", term.type, equals=clingo.SymbolType.Function)
        validate("name", term.arguments, length=0)
        validate("name", term.negative, equals=False)
        if arity is not None:
            validate("arity", arity, min_value=0, max_value=Predicate.MAX_ARITY)
        return Predicate(
            name=term.name,
            arity=arity,
            key=Predicate.__key,
        )

    @staticmethod
    def of(term: clingo.Symbol) -> "Predicate":
        return Predicate(
            name=term.name,
            arity=len(term.arguments),
            key=Predicate.__key,
        )

    def match(self, other: "Predicate") -> bool:
        if self.name != other.name:
            return False
        if self.arity is None or other.arity is None:
            return True
        return self.arity == other.arity

    def __lt__(self, other: "Predicate"):
        if self.name < other.name:
            return True
        if self.name > other.name:
            return False

        if self.arity is None:
            return False
        if other.arity is None:
            return True

        return self.arity < other.arity


@functools.total_ordering
@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class GroundAtom:
    value: clingo.Symbol

    def __post_init__(self):
        validate("atom format", self.value.type, equals=clingo.SymbolType.Function,
                 help_msg="An atom must have a predicate name")

    @staticmethod
    def parse(string: str) -> "GroundAtom":
        return GroundAtom(Parser.parse_ground_term(string))

    @cached_property
    def predicate(self) -> Predicate:
        return Predicate.of(self.value)

    @property
    def predicate_name(self) -> str:
        return self.predicate.name

    @property
    def predicate_arity(self) -> int:
        return self.predicate.arity

    @cached_property
    def arguments(self) -> tuple[clingo.Symbol, ...]:
        return tuple(self.value.arguments)

    @property
    def strongly_negated(self) -> bool:
        return self.value.negative

    def __str__(self):
        return str(self.value)

    def __lt__(self, other: "GroundAtom"):
        if self.predicate < other.predicate:
            return True
        if self.predicate == other.predicate:
            for index, argument in enumerate(self.arguments):
                other_argument = other.arguments[index]
                if argument.type < other_argument.type:
                    return True
                if argument.type > other_argument.type:
                    return False
                if argument.type == clingo.SymbolType.Number:
                    if argument < other_argument:
                        return True
                    if argument > other_argument:
                        return False
                else:
                    s1, s2 = str(argument), str(other_argument)
                    if s1 < s2:
                        return True
                    if s1 > s2:
                        return False
        return False


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class SymbolicTerm:
    __value: clingo.ast.AST
    __parsed_string: Optional[str]

    key: InitVar[PrivateKey]
    __key = PrivateKey()

    def __post_init__(self, key: PrivateKey):
        self.__key.validate(key)
        validate("type", self.__value.ast_type, is_in=[clingo.ast.ASTType.SymbolicTerm, clingo.ast.ASTType.Function,
                                                       clingo.ast.ASTType.Variable])

    @staticmethod
    def parse(string: str) -> "SymbolicTerm":
        rule: Final = f":- a({string})."
        try:
            program = Parser.parse_program(rule)
        except Parser.Error as error:
            raise error.drop(first=3, last=1)

        validate("one rule", program, length=1,
                 help_msg=f"Unexpected sequence of {len(program)} rules in {utils.one_line(string)}")
        validate("one atom", program[0].body, length=1,
                 help_msg=f"Unexpected conjunction of {len(program[0].body)} atoms in {utils.one_line(string)}")
        atom = program[0].body[0].atom.symbol
        validate("arity", atom.arguments, length=1,
                 help_msg=f"Unexpected sequence of {len(atom.arguments)} terms in {utils.one_line(string)}")
        return SymbolicTerm(atom.arguments[0], utils.extract_parsed_string(rule, atom.arguments[0].location),
                            key=SymbolicTerm.__key)

    @staticmethod
    def of_int(value: int) -> "SymbolicTerm":
        return SymbolicTerm.parse(str(value))

    @staticmethod
    def of_string(value: str) -> "SymbolicTerm":
        return SymbolicTerm.parse(f'"{value}"')

    def __str__(self):
        return self.__parsed_string or str(self.__value)

    def is_int(self) -> bool:
        return self.__value.ast_type == clingo.ast.ASTType.SymbolicTerm and \
            self.__value.symbol.type == clingo.SymbolType.Number

    def is_string(self) -> bool:
        return self.__value.ast_type == clingo.ast.ASTType.SymbolicTerm and \
            self.__value.symbol.type == clingo.SymbolType.String

    def is_function(self) -> bool:
        return self.__value.ast_type == clingo.ast.ASTType.Function or \
            self.__value.ast_type == clingo.ast.ASTType.SymbolicTerm and \
            self.__value.symbol.type == clingo.SymbolType.Function

    def is_variable(self) -> bool:
        return self.__value.ast_type == clingo.ast.ASTType.Variable

    def int_value(self) -> int:
        return self.__value.symbol.number

    def string_value(self) -> str:
        return self.__value.symbol.string

    @property
    def function_name(self) -> str:
        return self.__value.name if self.__value.ast_type == clingo.ast.ASTType.Function else self.__value.symbol.name

    @property
    def function_arity(self) -> int:
        return len(self.__value.arguments if self.__value.ast_type == clingo.ast.ASTType.Function else
                   self.__value.symbol.arguments)

    @cached_property
    def arguments(self) -> tuple["SymbolicTerm", ...]:
        return tuple(SymbolicTerm.parse(str(argument)) for argument in self.__value.arguments)

    def make_copy_of_value(self) -> clingo.ast.AST:
        return copy.deepcopy(self.__value)

    def match(self, pattern: "SymbolicTerm") -> bool:
        if pattern.is_variable() or self.is_variable():
            return True
        if pattern.is_function():
            return self.is_function() and pattern.function_name == self.function_name and \
                pattern.function_arity == self.function_arity and \
                all(argument.match(pattern.arguments[index]) for index, argument in enumerate(self.arguments))
        return pattern == self


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class SymbolicAtom:
    __value: clingo.ast.AST
    __parsed_string: Optional[str]

    key: InitVar[PrivateKey]
    __key = PrivateKey()

    def __post_init__(self, key: PrivateKey):
        self.__key.validate(key)
        validate("type", self.__value.ast_type,
                 is_in=[clingo.ast.ASTType.SymbolicAtom, clingo.ast.ASTType.Function,
                        clingo.ast.ASTType.BooleanConstant])

    @staticmethod
    def of_false() -> "SymbolicAtom":
        return SymbolicAtom.parse("#false")

    @staticmethod
    def of_ground_atom(atom: GroundAtom) -> "SymbolicAtom":
        return SymbolicAtom.parse(str(atom))

    @staticmethod
    def parse(string: str) -> "SymbolicAtom":
        rule: Final = f":- {string}."
        try:
            program = Parser.parse_program(rule)
        except Parser.Error as error:
            raise error.drop(first=3, last=1)

        validate("one rule", program, length=1,
                 help_msg=f"Unexpected sequence of {len(program)} rules in {utils.one_line(string)}")
        validate("one atom", program[0].body, length=1,
                 help_msg=f"Unexpected conjunction of {len(program[0].body)} atoms in {utils.one_line(string)}")
        literal = program[0].body[0]
        validate("positive", literal.sign, equals=clingo.ast.Sign.NoSign,
                 help_msg=f"Unexpected default negation in {utils.one_line(string)}")
        if "value" in literal.atom.keys():
            validate("#false", literal.atom.value, equals=0)
            atom = SymbolicAtom.parse("foo").__value.update(name="#false")
        else:
            atom = literal.atom.symbol
        return SymbolicAtom(atom, utils.extract_parsed_string(rule, literal.location), key=SymbolicAtom.__key)

    @staticmethod
    def of(value: clingo.ast.AST) -> "SymbolicAtom":
        validate("value", value.ast_type, is_in=[
            clingo.ast.ASTType.Function
        ])
        return SymbolicAtom(value, None, key=SymbolicAtom.__key)

    def __str__(self):
        return self.__parsed_string or str(self.__value)

    def make_copy_of_value(self) -> clingo.ast.AST:
        return copy.deepcopy(self.__value)

    @cached_property
    def predicate(self) -> Predicate:
        return Predicate.parse(self.__value.name, len(self.__value.arguments))

    @property
    def predicate_name(self) -> str:
        if self.__value.name == '#false':
            return self.__value.name
        return self.predicate.name

    @property
    def predicate_arity(self) -> int:
        return self.predicate.arity

    @cached_property
    def arguments(self) -> tuple[SymbolicTerm, ...]:
        return tuple(SymbolicTerm.parse(str(argument)) for argument in self.__value.arguments)

    @property
    def strongly_negated(self) -> bool:
        return self.value.negative

    def match(self, *pattern: "SymbolicAtom") -> bool:
        for a_pattern in pattern:
            if self.predicate == a_pattern.predicate and \
                    all(argument.match(a_pattern.arguments[index]) for index, argument in enumerate(self.arguments)):
                return True
        return False


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class SymbolicRule:
    __value: clingo.ast.AST
    __parsed_string: Optional[str]
    disabled: bool

    key: InitVar[PrivateKey]
    __key = PrivateKey()

    def __post_init__(self, key: PrivateKey):
        self.__key.validate(key)
        validate("type", self.__value.ast_type, equals=clingo.ast.ASTType.Rule)

    @staticmethod
    def parse(string: str, disabled: bool = False) -> "SymbolicRule":
        program = Parser.parse_program(string)
        validate("one rule", program, length=1,
                 help_msg=f"Unexpected sequence of {len(program)} rules in {utils.one_line(string)}")
        return SymbolicRule(program[0], utils.extract_parsed_string(string, program[0].location), disabled=disabled,
                            key=SymbolicRule.__key)

    @staticmethod
    def of(value: clingo.ast.AST, disabled: bool = False) -> "SymbolicRule":
        validate("value", value.ast_type == clingo.ast.ASTType.Rule, equals=True)
        return SymbolicRule(value, None, disabled=disabled, key=SymbolicRule.__key)

    def __str__(self):
        res = self.__parsed_string or str(self.__value)
        return f"%* {res} *%" if self.disabled else res

    def transform(self, transformer: clingo.ast.Transformer) -> Any:
        transformer(self.__value)

    @property
    def is_fact(self) -> bool:
        return len(self.__value.body) == 0 and self.is_normal_rule

    @property
    def is_normal_rule(self) -> bool:
        return self.__value.head.ast_type == clingo.ast.ASTType.Literal and \
            self.__value.head.sign == clingo.ast.Sign.NoSign

    @property
    def head_atom(self) -> SymbolicAtom:
        if ("atom" in self.__value.head.keys()) and ("value" in self.__value.head.atom.keys()):
            validate("#false", self.__value.head.atom.value, equals=0)
            return SymbolicAtom.of_false()
        return SymbolicAtom.of(self.__value.head.atom.symbol)

    @cached_property
    def head_variables(self) -> tuple[str, ...]:
        res = set()

        class Transformer(clingo.ast.Transformer):
            def visit_Variable(self, node):
                res.add(str(node))
                return node

        Transformer().visit(self.__value.head)
        return tuple(sorted(res))

    @cached_property
    def body_variables(self) -> tuple[str, ...]:
        res = set()

        class Transformer(clingo.ast.Transformer):
            def visit_Variable(self, node):
                res.add(str(node))
                return node

        Transformer().visit_sequence(self.__value.body)
        return tuple(sorted(res))

    @cached_property
    def global_safe_variables(self) -> tuple[str, ...]:
        res = set()

        class Transformer(clingo.ast.Transformer):
            def visit_Literal(self, node):
                if node.sign == clingo.ast.Sign.NoSign:
                    self.visit_children(node)

            def visit_ConditionalLiteral(self, node):
                # a conditional literal cannot bound new variables
                # (this is not the case for "existential" variables, which are not covered at the moment)
                pass

            def visit_BodyAggregate(self, node):
                for guard in [node.left_guard, node.right_guard]:
                    if guard is not None and guard.comparison == clingo.ast.ComparisonOperator.Equal:
                        self.visit(guard.term)

            def visit_Variable(self, node):
                if node.name != '_':
                    res.add(node.name)
                return node

        Transformer().visit_sequence(self.__value.body)
        return tuple(sorted(res))

    @cached_property
    def predicates(self) -> tuple[Predicate, ...]:
        res = set()

        class Transformer(clingo.ast.Transformer):
            def visit_Function(self, node):
                res.add((node.name, len(node.arguments)))
                return node

            def visit_Literal(self, node):
                if "symbol" in node.atom.keys():
                    res.add((node.atom.symbol.name, len(node.atom.symbol.arguments)))
                elif "elements" in node.atom.keys():
                    for element in node.atom.elements:
                        self.visit(element.update(terms=[]))
                return node

        Transformer().visit(self.__value)
        return tuple(Predicate.parse(*pred) for pred in res)

    def disable(self) -> "SymbolicRule":
        return SymbolicRule(self.__value, self.__parsed_string, True, key=self.__key)

    def with_extended_body(self, atom: SymbolicAtom, sign: clingo.ast.Sign = clingo.ast.Sign.NoSign) -> "SymbolicRule":
        string = self.__parsed_string[:-1] if self.__parsed_string is not None else str(self)[:-1]
        literal = f"{atom}" if sign == clingo.ast.Sign.NoSign else \
            f"not {atom}" if sign == clingo.ast.Sign.Negation else \
                f"not not {atom}"
        return self.parse(f"{string}; {literal}." if len(self.__value.body) > 0 else f"{string} :- {literal}.",
                          self.disabled)

    def body_as_string(self, separator: str = "; ") -> str:
        return separator.join(str(x) for x in self.__value.body)

    def apply_variable_substitution(self, **kwargs: SymbolicTerm) -> "SymbolicRule":
        class Transformer(clingo.ast.Transformer):
            def visit_Variable(self, node):
                if str(node) not in kwargs.keys():
                    return node
                return kwargs[str(node)].make_copy_of_value()

        return self.of(Transformer().visit(self.__value), self.disabled)

    def apply_term_substitution(self, **kwargs: SymbolicTerm) -> "SymbolicRule":
        class Transformer(clingo.ast.Transformer):
            def visit_SymbolicTerm(self, node):
                if str(node) not in kwargs.keys():
                    return node
                return kwargs[str(node)].make_copy_of_value()

        return self.of(Transformer().visit(self.__value), self.disabled)

    def apply_predicate_renaming(self, **kwargs: Predicate) -> "SymbolicRule":
        class Transformer(clingo.ast.Transformer):
            def visit_Function(self, node):
                return node.update(name=kwargs[node.name].name) if node.name in kwargs.keys() else node

        return self.of(Transformer().visit(self.__value), self.disabled)

    def expand_global_safe_variables(self, *, variables: Iterable[str],
                                     herbrand_base: "Model") -> tuple["SymbolicRule", ...]:
        if not variables:
            return (self,)
        the_variables: Final = set(var for var in variables if var in self.global_safe_variables)
        validate("variables", set(variables), equals=the_variables)
        substitutions = herbrand_base.compute_substitutions(
            arguments=','.join(the_variables),
            number_of_arguments=len(the_variables),
            conjunctive_query=self.body_as_string(),
        )
        return tuple(
            self.apply_variable_substitution(
                **{var: SymbolicTerm.parse(str(substitution[index])) for index, var in enumerate(the_variables)}
            ) for substitution in substitutions
        )

    def expand_global_and_local_variables(self, *, herbrand_base: "Model") -> tuple["SymbolicRule", ...]:
        class Transformer(clingo.ast.Transformer):
            def __init__(self):
                super().__init__()
                self.substitutions = []

            def visit_ConditionalLiteral(self, node):
                substitutions = herbrand_base.compute_substitutions(
                    arguments=','.join(str(arg) for arg in node.literal.atom.symbol.arguments),
                    number_of_arguments=len(node.literal.atom.symbol.arguments),
                    conjunctive_query=f"{', '.join(str(condition) for condition in node.condition)}",
                )
                the_uuid: Final = f"__uuid_{uuid()}__"
                self.substitutions.append(
                    (
                        the_uuid,
                        [
                            (
                                "not " if node.literal.sign == clingo.ast.Sign.Negation else
                                "not not " if node.literal.sign == clingo.ast.Sign.DoubleNegation
                                else ""
                            ) +
                            f"{node.literal.atom.symbol.name}({','.join(str(arg) for arg in arguments)})"
                            for arguments in substitutions
                        ]
                    )
                )
                return clingo.ast.Literal(node.location, clingo.ast.Sign.NoSign,
                                          clingo.ast.Function(node.location, the_uuid, [], False))

        result = []
        for partial_ground_rule in self.expand_global_safe_variables(variables=self.global_safe_variables,
                                                                     herbrand_base=herbrand_base):
            transformer = Transformer()
            rule = str(self.of(transformer.visit(partial_ground_rule.__value), False))
            for predicate, atoms in transformer.substitutions:
                rule = rule.replace(predicate, '; '.join(atoms))
            result.append(SymbolicRule.parse(rule, disabled=self.disabled))
        return tuple(result)

    def match(self, *pattern: SymbolicAtom) -> bool:
        class Transformer(clingo.ast.Transformer):
            def visit_SymbolicAtom(self, node):
                atom = SymbolicAtom.of(node.symbol)
                if atom.match(*pattern):
                    Transformer.matched = True
                return node
        Transformer.matched = False

        Transformer().visit(self.__value)
        return Transformer.matched


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class SymbolicProgram:
    __rules: tuple[SymbolicRule, ...]
    __parsed_string: Optional[str]

    key: InitVar[PrivateKey]
    __key = PrivateKey()

    def __post_init__(self, key: PrivateKey):
        self.__key.validate(key)

    @staticmethod
    def of(*args: SymbolicRule | Iterable[SymbolicRule]) -> "SymbolicProgram":
        rules = []
        for arg in args:
            if type(arg) == SymbolicRule:
                rules.append(arg)
            else:
                rules.extend(arg)
        return SymbolicProgram(tuple(rules), None, key=SymbolicProgram.__key)

    @staticmethod
    def parse(string: str) -> "SymbolicProgram":
        rules = tuple(SymbolicRule.parse(utils.extract_parsed_string(string, rule.location))
                      for rule in Parser.parse_program(string))
        return SymbolicProgram(rules, string, key=SymbolicProgram.__key)

    def __str__(self):
        return '\n'.join(str(rule) for rule in self.__rules) if self.__parsed_string is None else self.__parsed_string

    def __len__(self):
        return len(self.__rules)

    def __getitem__(self, item: int):
        return self.__rules[item]

    @cached_property
    def herbrand_universe(self) -> set[SymbolicTerm]:
        res = set()

        def get_arguments(term):
            for argument in term.arguments:
                if argument.type == clingo.SymbolType.Function and argument.arguments:
                    get_arguments(argument)
                else:
                    res.add(SymbolicTerm.parse(str(argument)))

        for atom in self.herbrand_base:
            get_arguments(atom)
        return res

    @cached_property
    def herbrand_base(self) -> "Model":
        control = clingo.Control()
        control.add(str(self))
        control.ground([("base", [])])
        return Model.of_atoms(atom.symbol for atom in control.symbolic_atoms)

    @cached_property
    def predicates(self) -> tuple[Predicate, ...]:
        res = set()
        for rule in self:
            res.update(rule.predicates)
        return tuple(res)

    @cache
    def process_constants(self) -> "SymbolicProgram":
        rules = []
        constants = {}
        for rule in self:
            if rule.is_fact:
                head_atom = rule.head_atom
                if head_atom.predicate_name == "__const__":
                    validate("arity", head_atom.predicate_arity, equals=2, help_msg="Error in defining constant")
                    name, value = head_atom.arguments
                    constants[str(name)] = value
                    rules.append(rule.disable())
                    continue
            rules.append(rule.apply_term_substitution(**constants))

        return SymbolicProgram.of(rules)

    @cache
    def process_with_statements(self) -> "SymbolicProgram":
        rules = []
        statements_queue = []
        for rule in self:
            if rule.is_fact:
                head_atom = rule.head_atom
                if head_atom.predicate_name == "__with__":
                    statements_queue.append(tuple(SymbolicAtom.parse(str(argument))
                                                  for argument in head_atom.arguments))
                    rules.append(rule.disable())
                    continue
                if head_atom.predicate_name == "__end_with__":
                    validate("no arguments", head_atom.arguments, length=0)
                    statements_queue.pop()
                    rules.append(rule.disable())
                    continue
            for statement in statements_queue:
                for literal in statement:
                    rule = rule.with_extended_body(literal)
            rules.append(rule)
        validate("all __with__ are terminated", statements_queue, length=0,
                 help_msg=f"{len(statements_queue)} unterminated __with__ statements")

        return SymbolicProgram.of(rules)

    def apply_predicate_renaming(self, **kwargs: Predicate) -> "SymbolicProgram":
        return SymbolicProgram.of(rule.apply_predicate_renaming(**kwargs) for rule in self)

    def expand_global_safe_variables(self, *, rule: SymbolicRule, variables: Iterable[str]) -> "SymbolicProgram":
        rules = []
        for __rule in self.__rules:
            if rule != __rule:
                rules.append(__rule)
            else:
                rules.extend(__rule.expand_global_safe_variables(variables=variables, herbrand_base=self.herbrand_base))
        return SymbolicProgram.of(rules)

    def expand_global_and_local_variables(self, *, expand_also_disabled_rules: bool = False) -> "SymbolicProgram":
        rules = []
        for rule in self.__rules:
            if not rule.disabled or expand_also_disabled_rules:
                rules.extend(rule.expand_global_and_local_variables(herbrand_base=self.herbrand_base))
            else:
                rules.append(rule)
        return SymbolicProgram.of(rules)

    def move_up(self, *pattern: SymbolicAtom) -> "SymbolicProgram":
        def key(rule: SymbolicRule):
            return 0 if rule.match(*pattern) else 1
        return SymbolicProgram.of(sorted([rule for rule in self.__rules], key=key))

    def query_herbrand_base(self, query_head_arguments: str, query_body: str,
                            aux_program: Optional["SymbolicProgram"] = None) -> tuple[SymbolicAtom, ...]:
        predicate = f"__query_{uuid()}__"
        program = SymbolicProgram.parse(f"{self.herbrand_base.as_facts}\n"
                                        f"{predicate}({query_head_arguments}) :- {query_body}.")
        if aux_program is not None:
            program = SymbolicProgram.of(*program, *aux_program)
        model = Model.of_program(program).filter(lambda atom: atom.predicate.name == predicate)
        return tuple(SymbolicAtom.of_ground_atom(atom) for atom in model)


@typeguard.typechecked
@dataclasses.dataclass(frozen=True, order=True)
class Model:
    value: tuple[GroundAtom | int | str, ...]

    key: InitVar[PrivateKey]
    __key = PrivateKey()

    class NoModelError(ValueError):
        def __init__(self, *args):
            super().__init__("no stable model", *args)

    class MultipleModelsError(ValueError):
        def __init__(self, *args):
            super().__init__("more than one stable model", *args)

    @staticmethod
    def empty():
        return Model(key=Model.__key, value=())

    @staticmethod
    def of_control(control: clingo.Control) -> "Model":
        def on_model(model):
            if on_model.cost is not None and on_model.cost <= model.cost:
                on_model.exception = True
            on_model.cost = model.cost
            on_model.res = Model.of_elements(model.symbols(shown=True))
        on_model.cost = None
        on_model.res = None
        on_model.exception = False

        control.solve(on_model=on_model)
        if not on_model.res:
            raise Model.NoModelError
        if on_model.exception:
            raise Model.MultipleModelsError
        return on_model.res

    @staticmethod
    def of_program(*args: str | SymbolicProgram | Iterable[str | SymbolicProgram]) -> "Model":
        program = []

        for arg in args:
            if type(arg) is str:
                program.append(arg)
            elif type(arg) is SymbolicProgram:
                program.append(str(arg))
            else:
                program.extend(str(elem) for elem in arg)
        control = clingo.Control()
        control.add('\n'.join(program))
        control.ground([("base", [])])
        return Model.of_control(control)

    @staticmethod
    def of_atoms(*args: Union[str, clingo.Symbol, GroundAtom, Iterable[Union[str, clingo.Symbol, GroundAtom]]]) -> "Model":
        res = Model.of_elements(*args)
        validate("only atoms", res.contains_only_ground_atoms, equals=True,
                 help_msg="Use Model.of_elements() to create a model with numbers and strings")
        return res

    @staticmethod
    def of_elements(*args: Union[int, str, clingo.Symbol, GroundAtom, Iterable[Union[int, str, clingo.Symbol, GroundAtom]]]) -> "Model":
        def build(atom):
            if type(atom) in [GroundAtom, int]:
                return atom
            if type(atom) is clingo.Symbol:
                if atom.type == clingo.SymbolType.Number:
                    return atom.number
                if atom.type == clingo.SymbolType.String:
                    return atom.string
                return GroundAtom(atom)
            if type(atom) is str:
                try:
                    return GroundAtom.parse(atom)
                except ValidationError:
                    if atom[0] == '"' == atom[-1]:
                        return Parser.parse_ground_term(atom).string
                    return Parser.parse_ground_term(f'"{atom}"').string
            return None

        flattened = []
        for element in args:
            built_element = build(element)
            if built_element is not None:
                flattened.append(built_element)
            else:
                flattened.extend(build(atom) for atom in element)
        return Model(
            key=Model.__key,
            value=
            tuple(sorted(x for x in flattened if type(x) is int)) +
            tuple(sorted(x for x in flattened if type(x) is str)) +
            tuple(sorted(x for x in flattened if type(x) is GroundAtom))
        )

    def __post_init__(self, key: PrivateKey):
        self.__key.validate(key)

    def __str__(self):
        return ' '.join(str(x) for x in self.value)

    def __len__(self):
        return len(self.value)

    def __getitem__(self, item):
        return self.value[item]

    def __iter__(self):
        return self.value.__iter__()

    @cached_property
    def contains_only_ground_atoms(self) -> bool:
        return all(type(element) == GroundAtom for element in self)

    @property
    def as_facts(self) -> str:
        def build(element):
            if type(element) is int:
                return f"__number({element})."
            if type(element) is str:
                return f"__string(\"{element}\")."
            return f"{element}."

        return '\n'.join(build(element) for element in self)

    @property
    def as_choice_rules(self) -> str:
        def build(element):
            if type(element) is int:
                return f"{{__number({element})}}."
            if type(element) is str:
                return f"{{__string(\"{element}\")}}."
            return f"{{{element}}}."

        return '\n'.join(build(element) for element in self)

    def drop(self, predicate: Optional[Predicate] = None, numbers: bool = False, strings: bool = False) -> "Model":
        def when(element):
            if type(element) is GroundAtom:
                return predicate is None or not predicate.match(element.predicate)
            if type(element) is int:
                return not numbers
            assert type(element) is str
            return not strings

        return self.filter(when)

    def filter(self, when: Callable[[GroundAtom], bool]) -> "Model":
        return Model(key=self.__key, value=tuple(atom for atom in self if when(atom)))

    def map(self, fun: Callable[[GroundAtom], GroundAtom]) -> 'Model':
        return Model(key=self.__key, value=tuple(sorted(fun(atom) for atom in self)))

    def rename(self, predicate: Predicate, new_name: Predicate) -> "Model":
        validate("same arity", predicate.arity == new_name.arity, equals=True,
                 help_msg="Predicates must have the same arity")
        return self.map(lambda atom: atom if not predicate.match(atom.predicate) else GroundAtom(
            clingo.Function(new_name.name, atom.arguments)
        ))

    def substitute(self, predicate: Predicate, argument: int, term: clingo.Symbol) -> "Model":
        validate("argument", argument, min_value=1, max_value=predicate.arity, help_msg="Arguments are indexed from 1")

        def mapping(atom: GroundAtom) -> GroundAtom:
            if not predicate.match(atom.predicate):
                return atom
            return GroundAtom(clingo.Function(
                atom.predicate_name,
                [arg if index != argument else term for index, arg in enumerate(atom.arguments, start=1)]
            ))

        return self.map(mapping)

    def project(self, predicate: Predicate, argument: int) -> "Model":
        validate("argument", argument, min_value=1, max_value=predicate.arity, help_msg="Arguments are indexed from 1")

        def mapping(atom: GroundAtom) -> GroundAtom:
            if not predicate.match(atom.predicate):
                return atom
            return GroundAtom(clingo.Function(
                atom.predicate_name,
                [arg for index, arg in enumerate(atom.arguments, start=1) if index != argument]
            ))

        return self.map(mapping)

    @property
    def block_up(self) -> str:
        return ":- " + ", ".join([f"{atom}" for atom in self]) + '.'

    @cached_property
    def __compute_substitutions_control(self):
        program = self.as_choice_rules
        control = clingo.Control()
        control.add(program)
        control.ground([("base", [])])
        return control

    def compute_substitutions(self, *, arguments: str, number_of_arguments: int,
                              conjunctive_query: str) -> tuple[list[clingo.Symbol], ...]:
        predicate: Final = f"__query_{uuid()}__"
        self.__compute_substitutions_control.add(predicate, [], f"{predicate}({arguments}) :- {conjunctive_query}.")
        self.__compute_substitutions_control.ground([(predicate, [])])
        return tuple(
            atom.symbol.arguments
            for atom in self.__compute_substitutions_control.symbolic_atoms.by_signature(predicate, number_of_arguments)
        )


@typeguard.typechecked
@dataclasses.dataclass(frozen=True)
class Template:
    @dataclasses.dataclass(frozen=True)
    class Name:
        value: str
        key: InitVar[PrivateKey]
        __key = PrivateKey()

        def __post_init__(self, key: PrivateKey):
            self.__key.validate(key)

        @staticmethod
        def parse(name: str) -> "Template.Name":
            term = clingo.String(name)
            return Template.Name(
                value=term.string,
                key=Template.Name.__key,
            )

        def __str__(self):
            return self.value

    name: "Template.Name"
    program: SymbolicProgram
    __static_uuid: str = dataclasses.field(default_factory=lambda: utils.uuid(), init=False)

    __core_templates = {}

    @staticmethod
    def __init_core_templates__():
        validate("called once", Template.__core_templates, max_len=0, help_msg="Cannot be called twice")

        def register(template: str):
            name, program = template.strip().split('\n', maxsplit=1)
            name = f"@dumbo/{name}"
            assert name not in Template.__core_templates
            Template.__core_templates[name] = Template(
                Template.Name.parse(name),
                Template.expand_program(SymbolicProgram.parse(program.strip())),
            )

        def register_all(templates: str, *, sep="----"):
            for template in templates.strip().split(sep):
                if template:
                    lines = [line[4:] if index > 0 and len(line) >= 4 else line
                             for index, line in enumerate(template.strip().split('\n'))]
                    register('\n'.join(lines))

        for arity in range(10):
            terms = ','.join('X' + str(i) for i in range(arity))
            register(f"""
exact copy (arity {arity})
output({terms}) :- input({terms}).
:- output({terms}), not input({terms}).
            """)
            if arity > 0:
                register(f"collect arguments (arity {arity})\n" +
                         '\n'.join(f"output(X{index}) :- input({terms})." for index in range(arity)))

        register_all("""
symmetric closure
    closure(X,Y) :- relation(X,Y).
    closure(X,Y) :- relation(Y,X).
----

symmetric closure guaranteed
    __apply_template__("@dumbo/symmetric closure", (closure, __closure)).
    __apply_template__("@dumbo/exact copy (arity 2)", (input, __closure), (output, closure)).
----

reachable nodes
    reach(X) :- start(X).
    reach(Y) :- reach(X), link(X,Y).
----

connected graph
    __start(X) :- X = #min{Y : node(Y)}.
    __apply_template__("@dumbo/reachable nodes", (start, __start), (reach, __reach)).
    :- node(X), not __reach(X).
----

transitive closure
    closure(X,Y) :- relation(X,Y).
    closure(X,Z) :- closure(X,Y), relation(Y,Z).
----

transitive closure guaranteed
    __apply_template__("@dumbo/transitive closure", (closure, __closure)).
    __apply_template__("@dumbo/exact copy (arity 2)", (input, __closure), (output, closure)).
----

spanning tree of undirected graph
    {tree(X,Y) : link(X,Y), X < Y} = C - 1 :- C = #count{X : node(X)}.
    __apply_template__("@dumbo/symmetric closure", (relation, tree), (closure, __tree)).
    __apply_template__("@dumbo/connected graph", (link, __tree)).
----
        """)

    @staticmethod
    def core_template(name: str) -> "Template":
        return Template.__core_templates[name]

    @staticmethod
    def is_core_template(name: str) -> bool:
        return name in Template.__core_templates

    @staticmethod
    def core_templates() -> int:
        return len(Template.__core_templates)

    @staticmethod
    def core_templates_as_parsable_string() -> str:
        res = []
        for key, value in Template.__core_templates.items():
            res.append(str(value))
        return '\n'.join(res)

    @staticmethod
    def expand_program(program: SymbolicProgram, *, limit: int = 100_000, trace: bool = False) -> SymbolicProgram:
        templates = {}
        template_under_read = None
        res = []
        for rule in program:
            validate("avoid blow up", len(res) + (len(template_under_read[1]) if template_under_read else 0),
                     max_value=limit,
                     help_msg=f"The expansion takes more than {limit} rules. "
                              f"If you trust the code, try again by increasing the limit.")
            if rule.disabled or not rule.is_normal_rule:
                if template_under_read is not None:
                    template_under_read[1].append(rule)
                else:
                    res.append(rule)
            elif rule.head_atom.predicate_name == "__template__":
                validate("empty body", rule.is_fact, equals=True)
                validate("arity 1", rule.head_atom.predicate_arity, equals=1)
                validate("arg#0", rule.head_atom.arguments[0].is_string(), equals=True)
                validate("no nesting", template_under_read is None, equals=True)
                validate("not a core template", Template.is_core_template(rule.head_atom.predicate.name), equals=False)
                validate("not seen", rule.head_atom.predicate.name not in templates, equals=True)
                template_under_read = (rule.head_atom.arguments[0].string_value(), [])
            elif rule.head_atom.predicate_name == "__end__":
                validate("empty body", rule.is_fact, equals=True)
                validate("arity 0", rule.head_atom.predicate_arity, equals=0)
                validate("not in a template", template_under_read)
                if trace:
                    template_under_read[1].append(rule.disable())
                templates[template_under_read[0]] = Template(name=Template.Name.parse(template_under_read[0]),
                                                             program=SymbolicProgram.of(template_under_read[1]))
                template_under_read = None
            elif rule.head_atom.predicate_name == "__apply_template__":
                validate("empty body", rule.is_fact, equals=True)
                validate("arity >= 1", rule.head_atom.predicate_arity, min_value=1)
                validate("arg#0", rule.head_atom.arguments[0].is_string(), equals=True)
                template_name = rule.head_atom.arguments[0].string_value()
                template = Template.core_template(template_name) if Template.is_core_template(template_name) \
                    else templates[template_name]
                mapping = {}
                for argument in rule.head_atom.arguments[1:]:
                    validate("mapping args", argument.is_function(), equals=True)
                    validate("mapping args", argument.function_name, equals='')
                    validate("mapping args", argument.function_arity, equals=2)
                    validate("mapping args", argument.arguments[0].is_function(), equals=True)
                    validate("mapping args", argument.arguments[0].function_arity, equals=0)
                    validate("mapping args", argument.arguments[1].is_function(), equals=True)
                    validate("mapping args", argument.arguments[1].function_arity, equals=0)
                    mapping[argument.arguments[0].function_name] = Predicate.parse(argument.arguments[1].function_name)
                if template_under_read is None:
                    if trace:
                        res.append(rule.disable())
                    res.extend(r for r in template.instantiate(**mapping))
                else:
                    if trace:
                        template_under_read[1].append(rule.disable())
                    template_under_read[1].extend(r for r in template.instantiate(**mapping))
            elif template_under_read is not None:
                template_under_read[1].append(rule)
            else:
                res.append(rule)

        return SymbolicProgram.of(res)

    def __str__(self):
        return f"""__template__("{self.name}").\n{self.program}\n__end__."""

    def __repr__(self):
        return f"""Template(name="{self.name}", program={self.program})"""

    def instantiate(self, **kwargs: Predicate) -> SymbolicProgram:
        for arg in kwargs:
            validate("kwargs", arg.startswith('__'), equals=False,
                     help_msg="Local predicates cannot be renamed externally.")
        static_uuid = self.__static_uuid
        local_uuid = utils.uuid()
        mapping = {**kwargs}
        for predicate in self.program.predicates:
            if not predicate.name.endswith('__'):
                if predicate.name.startswith('__static_'):
                    mapping[predicate.name] = Predicate.parse(f"{predicate.name[1:]}_{static_uuid}")
                elif predicate.name.startswith('__'):
                    mapping[predicate.name] = Predicate.parse(f"{predicate.name}_{local_uuid}")
        return self.program.apply_predicate_renaming(**mapping)


Template.__init_core_templates__()
