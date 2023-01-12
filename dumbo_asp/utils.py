import logging
import re
from pathlib import Path
from typing import Callable, Final

import clingo.ast
import typeguard
import valid8
from rich.console import Console
from rich.logging import RichHandler
from rich.prompt import Prompt, Confirm

PROJECT_ROOT: Final = Path(__file__).parent.parent
NEW_LINE_SYMBOL: Final = '⏎'

console = Console()
prompt = Prompt(console=console)
confirm = Confirm(console=console)

validate = valid8.validate
ValidationError = valid8.ValidationError

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, markup=True, rich_tracebacks=True)],
)
log = logging.getLogger("rich")


@typeguard.typechecked
def pattern(regex: str) -> Callable[[str], bool]:
    r = re.compile(regex)

    def res(value):
        return bool(r.fullmatch(value))

    res.__name__ = f'pattern({regex})'
    return res


@typeguard.typechecked
def extract_parsed_string(string: str, location: clingo.ast.Location) -> str:
    lines = string.split('\n')
    res = []
    if location.begin.line == location.end.line:
        res.append(lines[location.begin.line - 1][location.begin.column - 1:location.end.column - 1])
    else:
        res.append(lines[location.begin.line - 1][location.begin.column - 1:])
        res.extend(lines[location.begin.line:location.end.line - 1])
        res.append(lines[location.end.line - 1][:location.end.column - 1])
    return '\n'.join(line.rstrip() for line in res if line.strip())


@typeguard.typechecked
def one_line(string: str) -> str:
    return NEW_LINE_SYMBOL.join(string.split('\n'))
