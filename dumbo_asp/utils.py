from uuid import uuid4
from pathlib import Path
from typing import Final

import clingo.ast
import typeguard

PROJECT_ROOT: Final = Path(__file__).parent.parent
NEW_LINE_SYMBOL: Final = '⏎'


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


def uuid() -> str:
    return str(uuid4()).replace('-', '_')
