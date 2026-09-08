"""13: Repair space-concatenation in already-extracted blocks (no re-OCR).

Applies docsem.textfix.fix_spacing to every block in data/derived/blocks_*.jsonl.
Run AFTER extraction finishes and BEFORE 03/04/05. Idempotent (already-fixed
text re-passes through unchanged).
"""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import blocks_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.textfix import fix_spacing  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    args = ap.parse_args()

    for split in args.splits.split(","):
        path = blocks_file(split)
        try:
            rows = read_jsonl(path)
        except FileNotFoundError:
            print(f"[{split}] no {path}, skipping")
            continue
        n_changed = 0
        for r in rows:
            for bid, text in list(r["blocks"].items()):
                fixed = fix_spacing(text)
                if fixed != text:
                    r["blocks"][bid] = fixed
                    n_changed += 1
        write_jsonl(path, rows)
        print(f"[{split}] fixed {n_changed} blocks -> {path}")


if __name__ == "__main__":
    main()
