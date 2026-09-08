# dr_docsem

DocSem (DocInsights 2026, EMNLP workshop) solution pipeline:
document-grounded quantitative reasoning with evidence attribution.

Given a PDF and a paraphrased query, the system locates the quantitative
passage (evidence block), computes the answer with an LLM that writes and
executes Python (program-of-thought), and emits a submission JSONL.

## Pipeline

1. **Download** the dataset (tasks/labels/PDFs) from Hugging Face.
2. **Extract** blocks from each PDF: try the embedded text layer first,
   fall back to OCR (RapidOCR) when the pages are images.
3. **Evidence**: pick the supporting block via a keyword rule
   (`concerning X and Y`), a DeBERTa-v3 cross-encoder, or BM25 fallback.
4. **Solver**: Qwen3-4B-Instruct fine-tuned (LoRA SFT) to emit
   `EVIDENCE:` + Python code + `FINAL:`; answers come from executing the
   code, with self-consistency voting over n samples.
5. **Optional GRPO/RLVR** polish with verifiable rewards.
6. **Eval + submission**: local metrics mirroring the official
   normalizer, then submission JSONL for the portal.

## Verified dataset facts (from the 908 train labels)

- Evidence is always exactly **one** block, always in `b06..b13`.
- Answers are always **plain integers** (no units, no `%`).
- `user_query` embeds two topic keywords (`concerning X and Y`) that
  appear in the gold block text.
- PDFs are image-only (no extractable text layer) -> OCR path required.

## Layout

```
src/docsem/       library (extraction, evidence, solver, metrics)
scripts/00..11    numbered pipeline steps
tests/            unit tests for parsing, evidence, metrics, solver IO
data/             downloaded dataset (gitignored)
models/           trained weights (gitignored)
outputs/          predictions + submissions (gitignored)
reports/          audit results + generated training files (gitignored)
```

## RunPod quickstart

See `POD_RUN.md` for the exact command sequence from a fresh pod.
