"""Answer normalization mirroring the official normalizer, plus number parsing."""

import re

FINAL_MARKER = re.compile(r"^(final\s*answer|answer|final)\s*[:\-]?\s*", re.IGNORECASE)


def normalize_answer(ans: str) -> str:
    s = str(ans).strip().lower()
    s = FINAL_MARKER.sub("", s).strip()
    s = s.strip().strip(".")
    # numeric equivalence: 10 == 10.0 == 010
    try:
        return str(float(s))
    except ValueError:
        return s


def parse_number(text: str):
    m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def extract_final(text: str):
    """Pull the FINAL: line out of a solver response."""
    m = re.search(r"FINAL\s*[:\-]\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else None


def extract_evidence_id(text: str, valid_ids=None):
    """Pull the EVIDENCE: line out of a solver response."""
    m = re.search(r"EVIDENCE\s*[:\-]\s*(b\s?\d{1,3})", text, re.IGNORECASE)
    if not m:
        return None
    bid = "b" + re.sub(r"\D", "", m.group(1))
    bid = f"b{int(bid[1:]):02d}"
    if valid_ids is not None and bid not in valid_ids:
        return None
    return bid


def extract_code(text: str):
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else None
