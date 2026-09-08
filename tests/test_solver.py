"""Tests for solver response parsing and code execution."""

from docsem.normalize import extract_code, extract_evidence_id, extract_final
from docsem.solver import _run_code, last_number

RESPONSE = """EVIDENCE: b10
```python
total = 2 * 6
print(total)
```
FINAL: 12
"""


def test_extract_final():
    assert extract_final(RESPONSE) == "12"


def test_extract_evidence_id():
    assert extract_evidence_id(RESPONSE) == "b10"


def test_extract_code():
    assert "total = 2 * 6" in extract_code(RESPONSE)


def test_run_code():
    assert _run_code("print(2 * 6)") == "12"


def test_last_number():
    assert last_number("the answer is 42.5 percent") == "42.5"
