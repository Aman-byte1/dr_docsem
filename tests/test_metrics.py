"""Tests for answer normalization and scoring."""

from docsem.metrics import score_example
from docsem.normalize import normalize_answer, parse_number


def test_normalize_decimal():
    assert normalize_answer("10") == normalize_answer("10.0")
    assert normalize_answer(" 10 ") == normalize_answer("10")


def test_parse_number():
    assert parse_number("12,345.5") == 12345.5
    assert parse_number("no digits") is None


def test_score_example_joint():
    pred = {"answer": "10.0", "evidence": ["b10"]}
    gold = {"answer": "10", "evidence": ["b10"]}
    assert score_example(pred, gold)["joint"] is True


def test_score_example_evidence_mismatch():
    pred = {"answer": "10", "evidence": ["b09"]}
    gold = {"answer": "10", "evidence": ["b10"]}
    s = score_example(pred, gold)
    assert s["joint"] is False
    assert s["answer_exact"] is True
