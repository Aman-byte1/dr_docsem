"""00: Download the DocSem dataset (tasks, labels, PDFs) into data/raw."""

import os
import sys

sys.path.insert(0, "src")
from docsem.io_utils import RAW, REPO_ID, ensure_dirs  # noqa: E402


def main():
    ensure_dirs()
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    print(f"downloading {REPO_ID} -> {RAW}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=RAW,
        token=token,
        max_workers=16,
    )
    print("done. data/raw now contains train/, val/, test/ with tasks.jsonl, documents/, labels.jsonl")


if __name__ == "__main__":
    main()
