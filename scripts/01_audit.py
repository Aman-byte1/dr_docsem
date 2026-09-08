"""01: Audit the dataset. No models, fast."""

import collections
import json
import random
import sys

sys.path.insert(0, "src")
from docsem.io_utils import RAW, REPORTS, read_jsonl, write_json  # noqa: E402
from docsem.pdf_images import has_text_layer  # noqa: E402
from docsem.blocks import parse_blocks, valid_block_ids  # noqa: E402


def main(n_sample: int = 60):
    train_tasks = read_jsonl(RAW / "train" / "tasks.jsonl")
    labels = read_jsonl(RAW / "train" / "labels.jsonl")
    val_tasks = read_jsonl(RAW / "val" / "tasks.jsonl")
    test_tasks = read_jsonl(RAW / "test" / "tasks.jsonl")
    lab = {l["instance_id"]: l for l in labels}

    report = {
        "counts": {
            "train": len(train_tasks),
            "labels": len(labels),
            "val": len(val_tasks),
            "test": len(test_tasks),
        },
        "evidence_size_dist": dict(collections.Counter(len(l["evidence"]) for l in labels)),
        "evidence_id_dist": dict(
            sorted(collections.Counter(e for l in labels for e in l["evidence"]).items())
        ),
        "answer_kinds": {},
        "n_unique_answers": len({l["answer"] for l in labels}),
    }

    def kind(a):
        import re

        if re.fullmatch(r"-?\d+", a):
            return "int"
        if re.fullmatch(r"-?\d*\.\d+", a):
            return "decimal"
        return "other"

    report["answer_kinds"] = dict(collections.Counter(kind(l["answer"]) for l in labels))

    random.seed(0)
    sample = random.sample(train_tasks, min(n_sample, len(train_tasks)))
    text_layer_docs = 0
    block_counts = collections.Counter()
    kw_stats = collections.Counter()
    for t in sample:
        pdf = RAW / t["document_pdf"]
        if has_text_layer(pdf):
            text_layer_docs += 1
        from docsem.pdf_images import page_texts

        blocks = parse_blocks("\n".join(page_texts(pdf)))
        if blocks:
            block_counts[len(valid_block_ids(blocks))] += 1
        gold = lab[t["instance_id"]]["evidence"][0]
        gtext = blocks.get(gold, "").lower()
        import re

        m = re.search(r"concerning (\w+) and (\w+)", t["user_query"])
        if m:
            k1, k2 = m.group(1), m.group(2)
            both = k1 in gtext and k2 in gtext
            one = k1 in gtext or k2 in gtext
            kw_stats["both" if both else ("one" if one else "none")] += 1

    report["pdf_text_layer_docs_in_sample"] = f"{text_layer_docs}/{len(sample)}"
    report["block_counts_text_layer"] = dict(block_counts)
    report["query_keyword_coverage_in_gold"] = dict(kw_stats)

    out = REPORTS / "audit.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
