"""Shared paths and small IO helpers."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
DERIVED = DATA / "derived"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
REPORTS = ROOT / "reports"

REPO_ID = "amitbcp/docinsights-2026-shared-task-data"

SPLITS = ("train", "val", "test")


def ensure_dirs():
    for p in (DATA, RAW, DERIVED, MODELS, OUTPUTS, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def split_manifest(split: str) -> Path:
    return RAW / split / "tasks.jsonl"


def labels_file() -> Path:
    return RAW / "train" / "labels.jsonl"


def blocks_file(split: str) -> Path:
    return DERIVED / f"blocks_{split}.jsonl"
