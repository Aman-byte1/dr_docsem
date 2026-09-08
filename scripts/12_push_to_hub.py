"""12: Push/pull artifacts to the Hugging Face Hub (private repos under your account).

Backs up: extracted blocks (the expensive OCR output), trained models,
reports, and submission files. Run it again after each training stage.

Usage (token/user come from env vars - never hardcode them in the repo):
    export HF_TOKEN=hf_xxx
    export HF_USER=amanuelbyte
    python scripts/12_push_to_hub.py                 # upload everything found

Pull on a fresh pod:
    python scripts/12_push_to_hub.py --pull          # restore blocks+models
or selectively:
    huggingface-cli download amanuelbyte/docsem-blocks --repo-type dataset --local-dir data/derived
    huggingface-cli download amanuelbyte/docsem-deberta-evidence --local-dir models/deberta-evidence
    huggingface-cli download amanuelbyte/docsem-solver-sft --local-dir models/solver-sft-merged
"""

import argparse
import os
import sys

sys.path.insert(0, "src")
from docsem.io_utils import DERIVED, MODELS, OUTPUTS, REPORTS  # noqa: E402


def _jobs(user: str):
    return [
        ("model", f"{user}/docsem-deberta-evidence", MODELS / "deberta-evidence"),
        ("model", f"{user}/docsem-solver-sft", MODELS / "solver-sft-merged"),
        ("model", f"{user}/docsem-solver-grpo", MODELS / "solver-grpo-merged"),
        ("dataset", f"{user}/docsem-blocks", DERIVED),
        ("dataset", f"{user}/docsem-reports", REPORTS),
        ("dataset", f"{user}/docsem-outputs", OUTPUTS),
    ]


def push(user: str):
    from huggingface_hub import HfApi

    api = HfApi()
    for repo_type, repo_id, path in _jobs(user):
        if not path.exists() or not any(path.rglob("*")):
            print(f"skip {repo_id} ({path} missing or empty)")
            continue
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        print(f"uploading {path} -> {repo_id} ...", flush=True)
        api.upload_folder(folder_path=str(path), repo_id=repo_id, repo_type=repo_type)
        print(f"  done -> https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}")


def pull(user: str):
    from huggingface_hub import snapshot_download

    targets = [
        ("dataset", f"{user}/docsem-blocks", DERIVED),
        ("dataset", f"{user}/docsem-reports", REPORTS),
        ("dataset", f"{user}/docsem-outputs", OUTPUTS),
        ("model", f"{user}/docsem-deberta-evidence", MODELS / "deberta-evidence"),
        ("model", f"{user}/docsem-solver-sft", MODELS / "solver-sft-merged"),
        ("model", f"{user}/docsem-solver-grpo", MODELS / "solver-grpo-merged"),
    ]
    for repo_type, repo_id, path in targets:
        try:
            snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(path))
            print(f"restored {repo_id} -> {path}")
        except Exception as e:
            print(f"skip {repo_id}: {e.__class__.__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", action="store_true", help="restore from the Hub instead of uploading")
    ap.add_argument("--user", default=os.environ.get("HF_USER", "amanuelbyte"))
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("set HF_TOKEN first: export HF_TOKEN=hf_xxx")
    if args.pull:
        pull(args.user)
    else:
        push(args.user)


if __name__ == "__main__":
    main()
