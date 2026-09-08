"""Repair OCR text where inter-word spaces were dropped.

RapidOCR sometimes returns concatenated words ("RegionalOperationsBrief").
Numbers and block markers survive, but concatenated words destroy keyword
matching, BM25 scoring, and LLM readability. This module re-inserts spaces
using probabilistic word segmentation (wordsegment) on long alpha runs.

Safe heuristics only:
- alpha runs >= 14 chars get segmented ("preparedforthe" -> "prepared for the")
- shorter runs are left alone (real words like "proportionally" are 13 chars;
  segmenting them would corrupt text)
- numbers, punctuation, and single spaces are never touched
"""

import re

RUN = re.compile(r"[A-Za-z]{14,}")

_SEG = None


def _segmenter():
    global _SEG
    if _SEG is None:
        import wordsegment

        wordsegment.load()
        _SEG = wordsegment
    return _SEG


def _fix_run(m: re.Match) -> str:
    words = _segmenter().segment(m.group(0))
    return " ".join(words)


def fix_spacing(text: str) -> str:
    """Re-insert spaces in long concatenated alpha runs. Idempotent."""
    return RUN.sub(_fix_run, text)
