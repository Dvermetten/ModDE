"""Run the Python examples of the documentation pages and the README.

Every ```python block that imports modde is executed as a standalone script
(blocks without the import are fragments that continue an earlier example).
Examples of the deprecated 0.0.4 API (``modde.legacy``) are skipped. The
notebooks are executed by the docs build instead.
"""
import re
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
BLOCK = re.compile(r"^(?P<indent>[ \t]*)```python\n(?P<code>.*?)^(?P=indent)```", re.M | re.S)


def _snippets():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        for i, match in enumerate(BLOCK.finditer(text)):
            indent = match.group("indent")
            code = "\n".join(line[len(indent):] if line.startswith(indent) else line
                             for line in match.group("code").splitlines())
            if "import modde" not in code or "modde.legacy" in code:
                continue
            name = f"{path.relative_to(ROOT).as_posix()}#{i}"
            yield pytest.param(code, id=name)


@pytest.mark.parametrize("code", list(_snippets()))
def test_snippet_runs(code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(code, "<snippet>", "exec"), {"__name__": "__snippet__"})
