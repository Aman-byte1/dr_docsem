"""Tests for evidence selection."""

from docsem.evidence import bm25_pick, candidate_ids, extract_keywords, keyword_pick

BLOCKS = {
    "b01": "Regional Operations Brief",
    "b06": "Inventory: 40 units of stock A in the west warehouse.",
    "b10": "A shark is 10 feet long and two remoras are 6 inches each; "
    "what percentage of the shark length is the pair?",
    "b13": "This block is narrative and is not a calculation request.",
}

QUERY = (
    "Use the relevant quantitative passage in this document to determine "
    "the outcome concerning shark and percentage."
)


def test_candidate_ids_excludes_front_matter():
    ids = candidate_ids(BLOCKS)
    assert "b01" not in ids
    assert "b10" in ids


def test_extract_keywords():
    assert extract_keywords(QUERY) == ["shark", "percentage"]


def test_keyword_pick_hits_gold():
    assert keyword_pick(QUERY, BLOCKS) == "b10"


def test_bm25_pick_hits_gold():
    assert bm25_pick(QUERY, BLOCKS) == "b10"
