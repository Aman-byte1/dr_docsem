"""02: Extract document blocks for all splits (text layer if available, else OCR)."""

import argparse
import sys
import time

sys.path.insert(0, "src")
from docsem.io_utils import RAW, blocks_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.blocks import parse_blocks, valid_block_ids  # noqa: E402
from docsem.pdf_images import page_texts  # noqa: E402


def extract_one(pdf_path, use_ocr: bool, dpi: int):
    if use_ocr:
        from docsem.ocr import ocr_document_blocks

        return ocr_document_blocks(pdf_path, dpi=dpi)
    return parse_blocks("\n".join(page_texts(pdf_path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--force-ocr", action="store_true")
    args = ap.parse_args()

    for split in args.splits.split(","):
        tasks = read_jsonl(RAW / split / "tasks.jsonl")
        rows = []
        t0 = time.time()
        for i, t in enumerate(tasks):
            pdf = RAW / t["document_pdf"]
            blocks = {}
            if not args.force_ocr:
                blocks = extract_one(pdf, use_ocr=False, dpi=args.dpi)
            if not valid_block_ids(blocks):
                blocks = extract_one(pdf, use_ocr=True, dpi=args.dpi)
            rows.append(
                {
                    "instance_id": t["instance_id"],
                    "user_query": t["user_query"],
                    "document_pdf": t["document_pdf"],
                    "n_blocks": len(valid_block_ids(blocks)),
                    "blocks": blocks,
                }
            )
            if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                rate = (i + 1) / (time.time() - t0)
                print(f"[{split}] {i + 1}/{len(tasks)} ({rate:.1f} docs/s)", flush=True)
        out = blocks_file(split)
        write_jsonl(out, rows)
        n_empty = sum(1 for r in rows if r["n_blocks"] == 0)
        print(f"[{split}] wrote {out} | docs with 0 blocks: {n_empty}")


if __name__ == "__main__":
    main()
