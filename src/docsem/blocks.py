"""Parse 'b01: ...' document blocks from raw page text (text layer or OCR lines)."""

import re

MARKER = re.compile(r"^\s*b\s?([0-9Ool]{1,3})\s*[:.\-]\s*(.*)$", re.IGNORECASE)
VALID_ID = re.compile(r"^b\d{2}$")


def _canon(num: str) -> str:
    num = num.lower().replace("o", "0").replace("l", "1")
    return f"b{int(num):02d}"


def parse_blocks(text: str) -> dict:
    """Split raw text into {block_id: block_text} using the visible bXX markers."""
    blocks = {}
    cur, buf = None, []

    def flush():
        if cur is not None:
            joined = " ".join(part.strip() for part in buf if part.strip())
            if cur in blocks and joined:
                blocks[cur] = (blocks[cur] + " " + joined).strip()
            elif joined:
                blocks[cur] = joined

    for line in text.splitlines():
        m = MARKER.match(line)
        if m:
            flush()
            cur = _canon(m.group(1))
            buf = [m.group(2)]
        elif cur is not None:
            buf.append(line)
    flush()
    return {k: v for k, v in blocks.items() if v}


def valid_block_ids(blocks: dict):
    return [b for b in blocks if VALID_ID.match(b)]


def sort_ids(ids):
    return sorted(ids, key=lambda b: int(b[1:]))
