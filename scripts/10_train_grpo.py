"""10: Optional GRPO/RLVR polish on the SFT solver (verifiable reward, no external judge).

Reward per completion:
  +1.0 normalized answer exact match
  +0.5 evidence block id match (when the model echoes EVIDENCE:)
  -0.1 otherwise (wrong or unparseable)
Runs with TRL GRPOTrainer + vLLM rollouts. Train this AFTER 09_train_sft.py.
"""

import argparse
import sys

sys.path.insert(0, "src")
from docsem.io_utils import MODELS, REPORTS, blocks_file, labels_file, read_jsonl  # noqa: E402
from docsem.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from docsem.normalize import normalize_answer, parse_number  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=str(MODELS / "solver-sft-merged"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-gen", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=768)
    ap.add_argument("--out", default=str(MODELS / "solver-grpo-merged"))
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tasks = read_jsonl(blocks_file("train"))
    labels = {l["instance_id"]: l for l in read_jsonl(labels_file())}

    rows = []
    for t in tasks:
        gold = labels[t["instance_id"]]
        ev = gold["evidence"][0]
        if ev not in t["blocks"]:
            continue
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(t["user_query"], {ev: t["blocks"][ev]})},
                ],
                "gold_answer": normalize_answer(gold["answer"]),
                "gold_evidence": ev,
            }
        )
    ds = Dataset.from_list(rows)

    def reward_fn(completions, gold_answer, gold_evidence, **kwargs):
        rewards = []
        for comp, ga, ge in zip(completions, gold_answer, gold_evidence):
            text = comp[0]["content"] if isinstance(comp, list) else comp
            m_final = None
            import re as _re

            fm = _re.search(r"FINAL\s*[:\-]\s*(.+)", text, _re.IGNORECASE)
            m_final = fm.group(1).strip() if fm else None
            em = _re.search(r"EVIDENCE\s*[:\-]\s*(b\s?\d{1,3})", text, _re.IGNORECASE)
            ev_pred = "b" + _re.sub(r"\D", "", em.group(1)) if em else None
            r = -0.1
            if m_final is not None:
                pn = parse_number(m_final)
                if pn is not None and str(pn) == ga:
                    r = 1.0
                    if ev_pred == ge:
                        r += 0.5
            rewards.append(r)
        return rewards

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        max_prompt_length=4096,
        max_completion_length=args.max_completion,
        num_generations=args.num_gen,
        temperature=0.9,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.25,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model=args.base_model,
        args=cfg,
        train_dataset=ds,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )
    trainer.train()
    # GRPOTrainer saves the LoRA adapter only; merge it for direct inference
    trainer.save_model(args.out)
    print("adapter saved ->", args.out)
    merged_out = args.out.rstrip("/") + "-merged"
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(merged_out)
    print("merged model saved ->", merged_out)


if __name__ == "__main__":
    main()
