"""08: Generate SFT assistant traces with a teacher model (rejection sampling).

For each train example, sample a few completions from the teacher, execute the code,
keep the traces whose final answer matches gold. Output is chat-format JSONL for TRL SFTTrainer.
"""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import REPORTS, read_jsonl, write_jsonl  # noqa: E402
from docsem.solver import Solver, _run_code, parse_out  # noqa: E402
from docsem.normalize import extract_final, extract_code  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS / "sft_dataset.jsonl"))
    args = ap.parse_args()

    rows = read_jsonl(REPORTS / "sft_traces.jsonl")
    if args.limit:
        rows = rows[: args.limit]

    solver = Solver(args.solver_model, use_vllm=True)

    out_rows = []
    for i, r in enumerate(rows):
        outs = solver.chat(
            r["user"],
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            n=args.n_samples,
        )
        kept = 0
        for text in outs:
            final = extract_final(text)
            code = extract_code(text)
            if final is None or code is None:
                continue
            value = _run_code(code)
            num = parse_out(value) if value is not None else None
            if num is None:
                continue
            gold = parse_out(r["answer"])
            if num != gold:
                continue
            completion = text
            if not text.rstrip().endswith(final):
                completion = text.rstrip() + f"\nFINAL: {r['answer']}"
            out_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": r["system"]},
                        {"role": "user", "content": r["user"]},
                        {"role": "assistant", "content": completion},
                    ]
                }
            )
            kept += 1
            break  # one correct trace per example is enough
        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            print(f"[{i + 1}/{len(rows)}] kept so far: {len(out_rows)}", flush=True)

    write_jsonl(args.out, out_rows)
    print(f"wrote {args.out}: {len(out_rows)} traces (coverage {len(out_rows)}/{len(rows)})")


if __name__ == "__main__":
    main()
