# RunPod quickstart (A40)

## 0. Pod setup

Create a RunPod pod with the **PyTorch 2.x / CUDA 12.x** template and an
A40. Then:

```bash
git clone https://github.com/Aman-byte1/dr_docsem.git
cd dr_docsem

# (optional but recommended) export your HF token for faster downloads
export HF_TOKEN=hf_xxx

pip install -r requirements.txt
pip install -r requirements-vllm.txt   # optional, much faster generation
```

## 1. Data

```bash
python scripts/00_download.py       # dataset -> data/raw (~1 GB)
python scripts/01_audit.py          # facts + audit report
python scripts/02_extract.py        # blocks for train/val/test (OCR, ~1-2 h)
python scripts/03_check_extraction.py
```

`03` prints how often the keyword rule already picks the gold block - this
should be very high given the query template.

## 2. Evidence cross-encoder

```bash
python scripts/04_build_deberta_data.py
python scripts/05_train_deberta.py             # ~5-10 min on A40
python scripts/06_check_evidence.py --deberta models/deberta-evidence --limit 200
```

## 3. Solver

```bash
# generate traces with the teacher (rejection sampling, keep correct only)
python scripts/07_build_sft_data.py
python scripts/08_gen_sft_traces.py --solver-model Qwen/Qwen3-4B-Instruct

# LoRA SFT (~30-60 min on A40)
python scripts/09_train_sft.py --solver-model Qwen/Qwen3-4B-Instruct

# optional RLVR polish (GRPO) on the merged SFT model
python scripts/10_train_grpo.py --base-model models/solver-sft-merged
```

## 4. Inference

Validation split (217 tasks):

```bash
python -m docsem.pipeline --split val --model models/solver-sft-merged \
    --deberta models/deberta-evidence --n 8 \
    --out outputs/preds_val.jsonl
```

Sanity-check on train first if you want a local metric:

```bash
python -m docsem.pipeline --split train --model models/solver-sft-merged \
    --deberta models/deberta-evidence --n 8 --limit 120 \
    --out outputs/preds_train_sample.jsonl
python scripts/11_eval_and_submit.py --preds outputs/preds_train_sample.jsonl \
    --gold data/raw/train/labels.jsonl
```

## 5. Submission

```bash
python scripts/11_eval_and_submit.py --preds outputs/preds_val.jsonl \
    --split val --name baseline-v1
# -> outputs/baseline-v1_val.jsonl  (upload this to the portal)
```

For the held-out test split the same flow works with `--split test`;
submission files must cover all 1,730 test ids (missing ids count wrong,
and `null` answers count wrong - never abstain).

## Test-attempt strategy

You get 3 scored test attempts with a 6 h cooldown; final rank uses the
best attempt. Plan attempt 1 (SFT model) early, then use attempts 2-3 for
the GRPO-polished model or a fixed normalizer.

## Troubleshooting

- vLLM install fails: everything falls back to transformers automatically
  (slower but functional). Use `--no-vllm` to force it.
- OCR misreads a block id: `src/docsem/blocks.py` normalizes common
  confusions (`O`->`0`, `l`->`1`) and accepts single-digit ids.
- DeBERTa training OOM: lower `--batch-size` to 16.
