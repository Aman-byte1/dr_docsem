"""09: LoRA SFT of the solver on generated traces (TRL SFTTrainer)."""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import MODELS, REPORTS, read_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver-model", default="Qwen/Qwen3-4B-Instruct")
    ap.add_argument("--data", default=str(REPORTS / "sft_dataset.jsonl"))
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--out", default=str(MODELS / "solver-sft"))
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    rows = read_jsonl(args.data)
    ds = Dataset.from_list([{"messages": r["messages"]} for r in rows])

    tok = AutoTokenizer.from_pretrained(args.solver_model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        max_length=args.max_len,
        packing=False,
        dataset_num_proc=4,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=args.solver_model,
        args=cfg,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    # save the LoRA adapter
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print("adapter saved ->", args.out)
    # merge into a full model so inference / GRPO can load it directly
    merged_out = args.out.rstrip("/") + "-merged"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_out)
    tok.save_pretrained(merged_out)
    print("merged model saved ->", merged_out)


if __name__ == "__main__":
    main()
