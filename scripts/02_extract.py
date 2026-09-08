"""02: Extract document blocks for all splits (text layer if available, else OCR).

Parallelized with a process pool (OCR is CPU-bound). Progress is checkpointed,
so an interrupted run resumes where it left off instead of starting over.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "src")
from docsem.io_utils import RAW, blocks_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.blocks import parse_blocks, valid_block_ids  # noqa: E402
from docsem.pdf_images import page_texts  # noqa: E402


def _init_worker():
    """Per-process setup: cap BLAS threads, preload the OCR engine once."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from docsem.ocr import get_ocr

    get_ocr()


def _worker(task, dpi: int, force_ocr: bool):
    pdf_path = RAW / task["document_pdf"]
    blocks = {}
    if not force_ocr:
        blocks = parse_blocks("\n".join(page_texts(pdf_path)))
    if not valid_block_ids(blocks):
        from docsem.ocr import ocr_document_blocks

        blocks = ocr_document_blocks(pdf_path, dpi=dpi)
    return {
        "instance_id": task["instance_id"],
        "user_query": task["user_query"],
        "document_pdf": task["document_pdf"],
        "n_blocks": len(valid_block_ids(blocks)),
        "blocks": blocks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--force-ocr", action="store_true")
    ap.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("OCR_WORKERS", "0")),
        help="process pool size; default min(8, cpu_count), override with OCR_WORKERS",
    )
    args = ap.parse_args()

    for split in args.splits.split(","):
        tasks = read_jsonl(RAW / split / "tasks.jsonl")
        out = blocks_file(split)

        # resume: keep rows already extracted in a previous (interrupted) run
        done = {}
        if out.exists() and not args.force_ocr:
            for r in read_jsonl(out):
                done[r["instance_id"]] = r
        todo = [t for t in tasks if t["instance_id"] not in done]
        print(f"[{split}] {len(done)} already extracted, {len(todo)} to go", flush=True)
        if not todo:
            n_empty = sum(1 for r in done.values() if r["n_blocks"] == 0)
            print(f"[{split}] wrote {out} | docs with 0 blocks: {n_empty}")
            continue

        workers = args.workers or max(4, min(24, (os.cpu_count() or 8) // 4))
        print(f"[{split}] using {workers} OCR workers", flush=True)
        t0 = time.time()
        rows = list(done.values())
        n_done, n_fail = 0, 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futures = [ex.submit(_worker, t, args.dpi, args.force_ocr) for t in todo]
            for task, fut in zip(todo, futures):  # keep document order
                try:
                    rows.append(fut.result())
                except Exception as e:  # one bad PDF must not kill the run
                    n_fail += 1
                    print(f"  !! failed {task['instance_id']}: {e!r}", flush=True)
                    rows.append(
                        {
                            "instance_id": task["instance_id"],
                            "user_query": task["user_query"],
                            "document_pdf": task["document_pdf"],
                            "n_blocks": 0,
                            "blocks": {},
                        }
                    )
                n_done += 1
                if n_done == 1:
                    print(f"[{split}] first doc done in {time.time() - t0:.1f}s", flush=True)
                if n_done % 25 == 0 or n_done == len(todo):
                    rate = n_done / (time.time() - t0)
                    eta = (len(todo) - n_done) / rate if rate else 0.0
                    print(
                        f"[{split}] {n_done}/{len(todo)} "
                        f"({rate:.1f} docs/s, ETA {eta / 60:.0f} min, fails={n_fail})",
                        flush=True,
                    )
                    write_jsonl(out, rows)  # checkpoint for resume
        write_jsonl(out, rows)
        n_empty = sum(1 for r in rows if r["n_blocks"] == 0)
        print(f"[{split}] wrote {out} | docs with 0 blocks: {n_empty} | failures: {n_fail}")


if __name__ == "__main__":
    main()
