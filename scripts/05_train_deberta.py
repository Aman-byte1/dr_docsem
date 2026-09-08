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
    ap.add_argument("--lr", type=float, default=2e-5)
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
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
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
    for epoch in range(int(args.epochs)):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {out.loss.item():.4f}", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
