"""11: Evaluate predictions and write submission files."""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import OUTPUTS, labels_file, read_jsonl, write_jsonl  # noqa: E402
from docsem.metrics import score_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gold", default=None, help="labels jsonl; default = train labels")
    ap.add_argument("--split", default=None, help="if val/test, also write submission file")
    ap.add_argument("--name", default="submission")
    args = ap.parse_args()

    preds = read_jsonl(args.preds)
    if args.gold:
        golds = read_jsonl(args.gold)
        agg = score_all(preds, golds)
        print("metrics:", agg)
    else:
        print("no gold provided - skipping metrics")

    if args.split:
        out = OUTPUTS / f"{args.name}_{args.split}.jsonl"
        sub = []
        for p in preds:
            ans = p.get("answer")
            ev = p.get("evidence") or []
            sub.append(
                {
                    "instance_id": p["instance_id"],
                    "answer": str(ans) if ans is not None else None,
                    "evidence": list(ev),
                }
            )
        write_jsonl(out, sub)
        n_null = sum(1 for s in sub if s["answer"] is None)
        print(f"wrote {out} ({len(sub)} rows, {n_null} null answers)")


if __name__ == "__main__":
    main()
