"""07: Build SFT data for the solver from train gold evidence + answers."""

import argparse
import json
import sys

sys.path.insert(0, "src")
from docsem.io_utils import REPORTS, blocks_file, labels_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--out", default=str(REPORTS / "sft_traces.jsonl"))
    args = ap.parse_args()

    tasks = read_jsonl(blocks_file("train"))
    labels = {l["instance_id"]: l for l in read_jsonl(labels_file())}

    rows = []
    for t in tasks:
        gold = labels[t["instance_id"]]
        blocks = t["blocks"]
        ev = gold["evidence"][0]
        if ev not in blocks:
            continue
        user_prompt = build_user_prompt(t["user_query"], {ev: blocks[ev]})
        rows.append(
            {
                "instance_id": t["instance_id"],
                "system": SYSTEM_PROMPT,
                "user": user_prompt,
                "answer": gold["answer"],
                "solver_model": args.solver_model,
            }
        )

    write_jsonl(args.out, rows)
    print(f"wrote {args.out}: {len(rows)} examples")
    print("next: run 08_gen_sft_traces.py to generate the assistant completions with a teacher model")


if __name__ == "__main__":
    main()
