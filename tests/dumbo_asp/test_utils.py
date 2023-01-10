import pytest

from dumbo_asp.utils import one_line, NEW_LINE_SYMBOL


@pytest.mark.parametrize("lines", [
    "a\nb\nc",
    "a\nb\n\n\nc",
    "a\n\nb\n\tc",
])
def test_one_line(lines):
    assert one_line(lines).split(NEW_LINE_SYMBOL) == lines.split('\n')
