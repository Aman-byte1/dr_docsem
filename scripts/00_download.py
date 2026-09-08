"""00: Download the DocSem dataset (tasks, labels, PDFs) into data/raw."""

import os
import sys

sys.path.insert(0, "src")
from docsem.io_utils import RAW, REPO_ID, ensure_dirs  # noqa: E402


def main():
    ensure_dirs()
    import time

    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    # Low worker count avoids a known huggingface_hub metadata race
    # (FileExistsError on .metadata) when downloads run slowly.
    workers = int(os.environ.get("HF_DOWNLOAD_WORKERS", "4"))
    print(f"downloading {REPO_ID} -> {RAW} (workers={workers})")
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=RAW,
                token=token,
                max_workers=workers,
            )
            break
        except Exception as e:
            print(f"attempt {attempt} failed: {e!r}; resuming download", flush=True)
            if attempt == 3:
                raise
            time.sleep(5)
    print("done. data/raw now contains train/, val/, test/ with tasks.jsonl, documents/, labels.jsonl")


if __name__ == "__main__":
    main()
