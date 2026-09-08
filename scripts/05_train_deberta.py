"""05: Train DeBERTa-v3-base cross-encoder for evidence selection (minutes on A40)."""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import MODELS, REPORTS, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="microsoft/deberta-v3-base")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-6)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--out", default=str(MODELS / "deberta-evidence"))
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_cosine_schedule_with_warmup,
    )

    rows = read_jsonl(REPORTS / "deberta_pairs.jsonl")

    class PairDS(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            return r["query"], r["block_text"], r["label"]

    tok = AutoTokenizer.from_pretrained(args.model)
    # SDPA attention NaNs with DeBERTa's disentangled attention (known
    # transformers bug) - force the eager implementation.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, attn_implementation="eager"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def collate(batch):
        qs, bs, ls = zip(*batch)
        enc = tok(list(qs), list(bs), truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(ls)
        return enc

    loader = DataLoader(PairDS(), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    steps = len(loader) * int(args.epochs)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)

    model.train()
    step = 0
    n_nan = 0
    # bf16 autocast for speed; fp32 master weights avoid the fp16 overflow
    # that drives DeBERTa-v3 loss to NaN on this task.
    for epoch in range(int(args.epochs)):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                out = model(**batch)
            if not torch.isfinite(out.loss):
                n_nan += 1
                opt.zero_grad()
                continue
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(
                    f"epoch {epoch} step {step}/{steps} loss {out.loss.item():.4f} nan_batches={n_nan}",
                    flush=True,
                )
    if n_nan:
        print(f"WARNING: skipped {n_nan} non-finite-loss batches")
    attempted = step + n_nan
    if attempted and n_nan > 0.25 * attempted:
        raise RuntimeError(
            f"{n_nan}/{attempted} batches had non-finite loss - training is "
            "unstable; NOT saving a garbage model. Report this run."
        )

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
