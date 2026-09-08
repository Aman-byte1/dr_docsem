"""Tests for block parsing."""

from docsem.blocks import parse_blocks, valid_block_ids

TEXT = """b01: Title line
some continuation text
b02: Second block
b10: Tail block
"""


def test_parse_basic():
    blocks = parse_blocks(TEXT)
    assert set(blocks) == {"b01", "b02", "b10"}
    assert blocks["b01"] == "Title line some continuation text"


def test_ocr_noise_markers():
    noisy = "bO1: zero and letter-o confusion\nb1: single digit\n"
    blocks = parse_blocks(noisy)
    assert "b01" in blocks
    assert "b01" in valid_block_ids(blocks)
