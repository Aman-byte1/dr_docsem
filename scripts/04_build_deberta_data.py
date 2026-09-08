"""04: Build DeBERTa training pairs from extracted train blocks."""

import random
import sys

sys.path.insert(0, "src")
from docsem.io_utils import REPORTS, blocks_file, labels_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.evidence import candidate_ids  # noqa: E402


def main():
    tasks = read_jsonl(blocks_file("train"))
    labels = {l["instance_id"]: l for l in read_jsonl(labels_file())}

    rows = []
    for t in tasks:
        gold = labels[t["instance_id"]]["evidence"][0]
        pool = candidate_ids(t["blocks"])
        if not pool:
            continue
        for bid in pool:
            rows.append(
                {
                    "query": t["user_query"],
                    "block_id": bid,
                    "block_text": t["blocks"][bid],
                    "label": 1 if bid == gold else 0,
                }
            )

    pos = sum(r["label"] for r in rows)
    rng = random.Random(0)
    rng.shuffle(rows)
    out = REPORTS / "deberta_pairs.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {out}: {len(rows)} pairs, {pos} positive")


if __name__ == "__main__":
    main()
