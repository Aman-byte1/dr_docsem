"""Shared pipeline runner: load blocks, pick evidence, solve, emit predictions."""

import argparse

from .io_utils import read_jsonl, blocks_file
from .evidence import pick_evidence, candidate_ids
from .solver import Solver, solve_one


def run_pipeline(
    split: str,
    model_dir: str,
    deberta_dir=None,
    n_samples: int = 8,
    temperature: float = 0.7,
    use_vllm: bool = True,
    limit: int = 0,
    out_path=None,
):
    tasks = read_jsonl(blocks_file(split))
    if limit:
        tasks = tasks[:limit]

    solver = Solver(model_dir, use_vllm=use_vllm)

    preds = []
    for i, t in enumerate(tasks):
        blocks = t["blocks"]
        ev, method = pick_evidence(query=t["user_query"], blocks=blocks, deberta_model_dir=deberta_dir)
        if ev is None:
            pool = candidate_ids(blocks)
            ev = pool[0] if pool else (sorted(blocks)[0] if blocks else "b01")
            method = method + "+fallback"
        ans, code = solve_one(
            solver,
            t["user_query"],
            blocks,
            ev,
            n_samples=n_samples,
            temperature=temperature,
        )
        preds.append(
            {
                "instance_id": t["instance_id"],
                "answer": str(ans) if ans is not None else None,
                "evidence": [ev],
                "_method": method,
                "_code": code,
            }
        )
        if (i + 1) % 25 == 0 or (i + 1) == len(tasks):
            print(f"[{i + 1}/{len(tasks)}] ev={ev} ({method}) ans={ans}", flush=True)
    if out_path:
        from .io_utils import write_jsonl

        write_jsonl(out_path, preds)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--deberta", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or f"outputs/preds_{args.split}.jsonl"
    run_pipeline(
        args.split,
        args.model,
        deberta_dir=args.deberta,
        n_samples=args.n,
        temperature=args.temperature,
        use_vllm=not args.no_vllm,
        limit=args.limit,
        out_path=out,
    )
    print("wrote", out)


if __name__ == "__main__":
    main()
