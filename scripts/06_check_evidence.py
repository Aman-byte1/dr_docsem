"""06: Evaluate evidence pickers (keyword, bm25, deberta) on train holdout."""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import REPORTS, blocks_file, labels_file, read_jsonl, write_json  # noqa: E402
from docsem.evidence import keyword_pick, bm25_pick, deberta_pick  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deberta", default=None, help="path to trained deberta dir")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tasks = read_jsonl(blocks_file("train"))
    labels = {l["instance_id"]: l for l in read_jsonl(labels_file())}
    if args.limit:
        tasks = tasks[-args.limit:]

    stats = {"n": len(tasks), "keyword": 0, "bm25": 0, "deberta": 0}
    for t in tasks:
        gold = labels[t["instance_id"]]["evidence"][0]
        blocks = t["blocks"]
        if keyword_pick(t["user_query"], blocks) == gold:
            stats["keyword"] += 1
        if bm25_pick(t["user_query"], blocks) == gold:
            stats["bm25"] += 1
        if args.deberta and deberta_pick(t["user_query"], blocks, args.deberta) == gold:
            stats["deberta"] += 1

    print(stats)
    write_json(REPORTS / "evidence_check.json", stats)


if __name__ == "__main__":
    main()
