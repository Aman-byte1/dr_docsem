"""02: Extract document blocks for all splits (text layer if available, else OCR).

Parallelized with a process pool (OCR is CPU-bound). Prints a newline-based
progress bar (log-friendly), and checkpointed so an interrupted run resumes
where it left off instead of starting over.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait


def _cpu_quota() -> int:
    """Effective CPU count in containers (cgroup quota wins over nproc)."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            quota, period = f.read().split()
            if quota != "max":
                return max(1, int(int(quota) / int(period)))
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read().strip())
            with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
                period = int(f.read().strip())
            if quota > 0:
                return max(1, quota // period)
    except OSError:
        pass
    return os.cpu_count() or 8


sys.path.insert(0, "src")
from docsem.io_utils import RAW, blocks_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.blocks import parse_blocks, valid_block_ids  # noqa: E402
from docsem.pdf_images import page_texts  # noqa: E402


def _bar(frac: float, width: int = 30) -> str:
    filled = int(width * max(0.0, min(1.0, frac)))
    return "|" + "#" * filled + "." * (width - filled) + "|"


def _fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


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
        help="process pool size; default max(4, cores//4), override with OCR_WORKERS",
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
        print(
            f"[{split}] {_bar(1 - len(todo) / max(1, len(tasks)))} "
            f"{len(done)}/{len(tasks)} already extracted, {len(todo)} to go",
            flush=True,
        )
        if not todo:
            n_empty = sum(1 for r in done.values() if r["n_blocks"] == 0)
            print(f"[{split}] wrote {out} | docs with 0 blocks: {n_empty}")
            continue

        # cgroup quota ~= real core budget; OCR workers use 1 intra-op thread
        # each (2 with CUDA), so give the pool the full quota.
        workers = args.workers or max(2, min(16, _cpu_quota()))
        print(f"[{split}] using {workers} OCR workers", flush=True)
        t0 = time.time()
        rows = list(done.values())
        n_todo, n_done, n_fail = len(todo), 0, 0
        last_done_at = time.time()

        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            fut2task = {ex.submit(_worker, t, args.dpi, args.force_ocr): t for t in todo}
            pending = set(fut2task)
            while pending:
                finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not finished:
                    idle = time.time() - last_done_at
                    print(
                        f"[{split}] ... no completions in the last {idle:.0f}s "
                        f"(workers busy or slow init; first-use model download "
                        f"can take a minute)",
                        flush=True,
                    )
                    continue
                for fut in finished:
                    task = fut2task[fut]
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
                    last_done_at = time.time()

                rate = n_done / (time.time() - t0)
                eta = (n_todo - n_done) / rate if rate else 0.0
                print(
                    f"[{split}] {_bar(n_done / n_todo)} "
                    f"{n_done}/{n_todo} ({100 * n_done / n_todo:.0f}%) "
                    f"{rate:.1f} docs/s ETA {_fmt_eta(eta)} fails={n_fail}",
                    flush=True,
                )
                write_jsonl(out, rows)  # checkpoint for resume

        write_jsonl(out, rows)
        n_empty = sum(1 for r in rows if r["n_blocks"] == 0)
        print(
            f"[{split}] DONE {_bar(1.0)} wrote {out} | "
            f"docs with 0 blocks: {n_empty} | failures: {n_fail}",
            flush=True,
        )


if __name__ == "__main__":
    main()
