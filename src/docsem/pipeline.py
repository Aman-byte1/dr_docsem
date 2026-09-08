"""Shared pipeline runner: solver sees ALL blocks, picks evidence + answer."""

import argparse
import time

from .io_utils import read_jsonl, blocks_file
from .evidence import pick_evidence, candidate_ids
from .progress import line
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
    t0 = time.time()
    for i, t in enumerate(tasks):
        blocks = t["blocks"]
        # Solver sees ALL blocks — picks evidence + answer together
        ans, code, solver_ev = solve_one(
            solver,
            t["user_query"],
            blocks,
            n_samples=n_samples,
            temperature=temperature,
        )
        # Evidence: prefer solver's pick, fall back to DeBERTa/BM25
        if solver_ev:
            ev, method = solver_ev, "solver"
        else:
            ev, method = pick_evidence(
                query=t["user_query"], blocks=blocks,
                deberta_model_dir=deberta_dir,
            )
        if ev is None:
            pool = candidate_ids(blocks)
            ev = pool[0] if pool else (sorted(blocks)[0] if blocks else "b01")
            method = method + "+fallback"
        # Never submit null answers (competition counts them wrong)
        if ans is None:
            ans = "0"
            method = method + "+null_default"
        preds.append(
            {
                "instance_id": t["instance_id"],
                "answer": str(ans),
                "evidence": [ev],
                "_method": method,
                "_code": code,
            }
        )
        if (i + 1) % 5 == 0 or (i + 1) == len(tasks):
            print(
                line(f"[{split}]", i + 1, len(tasks), t0, extra=f"ev={ev}({method}) ans={ans}"),
                flush=True,
            )
            from .io_utils import write_jsonl

            write_jsonl(out_path, preds)  # checkpoint
    if out_path:
        from .io_utils import write_jsonl

        write_jsonl(out_path, preds)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--deberta", default=None, help="optional fallback evidence model")
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

