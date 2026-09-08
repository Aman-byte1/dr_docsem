"""03: Check extraction quality against train labels."""

import collections
import re
import sys

sys.path.insert(0, "src")
from docsem.io_utils import RAW, REPORTS, blocks_file, labels_file, read_jsonl, write_json  # noqa: E402
from docsem.evidence import keyword_pick, bm25_pick  # noqa: E402


def main():
    tasks = read_jsonl(blocks_file("train"))
    labels = {l["instance_id"]: l for l in read_jsonl(labels_file())}

    stats = collections.Counter()
    for t in tasks:
        gold = labels[t["instance_id"]]["evidence"][0]
        blocks = t["blocks"]
        if not blocks:
            stats["no_blocks"] += 1
            continue
        if gold in blocks:
            stats["gold_block_extracted"] += 1
        else:
            stats["gold_block_missing"] += 1
        pick = keyword_pick(t["user_query"], blocks)
        if pick == gold:
            stats["keyword_correct"] += 1
        elif pick is not None:
            stats["keyword_wrong"] += 1
        else:
            stats["keyword_no_match"] += 1
        bpick = bm25_pick(t["user_query"], blocks)
        if bpick == gold:
            stats["bm25_correct"] += 1

    n = len(tasks)
    agg = {k: v for k, v in stats.items()}
    agg["n_docs"] = n
    print(f"n={n}")
    for k, v in sorted(agg.items()):
        pct = 100.0 * v / n if n else 0
        print(f"  {k}: {v} ({pct:.2f}%)")
    write_json(REPORTS / "extraction_check.json", agg)


if __name__ == "__main__":
    main()
