#!/usr/bin/env bash
# run_all.sh — one-command pipeline:
#   clean -> 300dpi GPU extract (train,val) -> spacing fix -> evidence -> SFT
#   -> backup to Hub -> val inference -> val score
# Safe to re-run: extraction resumes from checkpoints; later stages rebuild.
set -uo pipefail
cd "$(dirname "$0")"

LOG=outputs/run_all.log
mkdir -p outputs
exec > >(tee -a "$LOG") 2>&1

banner() { echo; echo "==================== $* ===================="; }

banner "STAGE 0: kill stale jobs, clean stale artifacts"
for p in $(ps -eo pid,cmd | grep -E '02_extract|08_gen|docsem.pipeline|09_train' | grep -v grep | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
sleep 2
rm -f data/derived/blocks_train.jsonl data/derived/blocks_val.jsonl
rm -f reports/deberta_pairs.jsonl reports/sft_traces.jsonl reports/sft_dataset.jsonl
rm -rf models/deberta-evidence models/solver-sft models/solver-sft-merged
pip install -q wordsegment
echo "clean."

banner "STAGE 1: 300dpi GPU extraction (train,val) - ~25 min"
OCR_CUDA=1 OCR_WORKERS=4 python scripts/02_extract.py --splits train,val
n_done=$(grep -c DONE outputs/run_all.log || true)
if [ "$(grep -c '\[train\] DONE' "$LOG")" -lt 1 ] || [ "$(grep -c '\[val\] DONE' "$LOG")" -lt 1 ]; then
  echo "EXTRACTION INCOMPLETE - rerun this script (it resumes)"; exit 1
fi

banner "STAGE 2: spacing fix + evidence"
python scripts/13_fix_spacing.py --splits train,val
python scripts/03_check_extraction.py
python scripts/04_build_deberta_data.py
python scripts/05_train_deberta.py
python scripts/06_check_evidence.py --deberta models/deberta-evidence --limit 200

banner "STAGE 3: solver SFT"
python scripts/07_build_sft_data.py
python scripts/08_gen_sft_traces.py
python scripts/09_train_sft.py

banner "STAGE 4: backup to Hub"
if [ -n "${HF_TOKEN:-}" ]; then python scripts/12_push_to_hub.py; else echo "HF_TOKEN not set - skipping backup"; fi

banner "STAGE 5: validation inference + score"
python -m docsem.pipeline --split val --model models/solver-sft-merged \
  --deberta models/deberta-evidence --n 8 --out outputs/preds_val.jsonl
python scripts/11_eval_and_submit.py --preds outputs/preds_val.jsonl --split val --name sft-v1

banner "ALL DONE - upload outputs/sft-v1_val.jsonl to the portal"
[ -n "${HF_TOKEN:-}" ] && python scripts/12_push_to_hub.py
