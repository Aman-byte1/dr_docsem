"""Program-of-Thought solver: LLM writes Python, we execute it in a sandbox."""

import io
import json
import re
import signal
import threading
import contextlib

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .normalize import extract_final, extract_evidence_id, extract_code

CODE_TIMEOUT = 6


def _run_code(code: str):
    """Execute solver-generated Python, capture the printed output. Timeout-guarded."""
    result = {"value": None}

    def target():
        buf = io.StringIO()
        safe_globals = {
            "__builtins__": __builtins__,
            "math": __import__("math"),
            "re": re,
            "json": json,
        }
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, safe_globals)
            result["value"] = buf.getvalue().strip()
        except Exception:
            result["value"] = None

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(CODE_TIMEOUT)
    return result["value"]


def last_number(text):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    if not m:
        return None
    return m[-1]


class Solver:
    """Generation wrapper: uses vLLM when available, else HF transformers."""

    def __init__(self, model_dir: str, use_vllm: bool = True, max_model_len: int = 8192):
        self.model_dir = model_dir
        self.backend = None
        try:
            if use_vllm:
                from vllm import LLM, SamplingParams

                self.llm = LLM(
                    model=model_dir,
                    dtype="bfloat16",
                    gpu_memory_utilization=0.85,
                    max_model_len=max_model_len,
                    enforce_eager=True,
                )
                self.sp = SamplingParams
                self.backend = "vllm"
                return
        except Exception:
            pass
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.backend = "hf"

    def chat(self, user_prompt: str, temperature: float, max_new_tokens: int, n: int = 1):
        if self.backend == "vllm":
            sp = self.sp(
                temperature=temperature if n > 1 else 0.0,
                top_p=0.95 if n > 1 else 1.0,
                n=n,
                max_tokens=max_new_tokens,
            )
            outs = self.llm.chat(
                [
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ]
                ],
                sp,
            )
            return [o.text for o in outs[0].outputs]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        text = self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        import torch

        enc = self.tok(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **enc,
            do_sample=n > 1,
            temperature=temperature if n > 1 else None,
            top_p=0.95 if n > 1 else None,
            num_return_sequences=n,
            max_new_tokens=max_new_tokens,
        )
        trimmed = out[:, enc["input_ids"].shape[1]:]
        return [self.tok.decode(t, skip_special_tokens=True) for t in trimmed]


def solve_one(
    solver: Solver,
    query: str,
    blocks: dict,
    n_samples: int = 1,
    temperature: float = 0.7,
    max_new_tokens: int = 768,
):
    """Send ALL blocks to the solver, let it pick evidence + compute answer.

    Returns (answer, code, evidence_id) — evidence_id is majority-voted from
    the solver's EVIDENCE: lines across n_samples completions.
    """
    user_prompt = build_user_prompt(query, blocks)
    outs = solver.chat(user_prompt, temperature, max_new_tokens, n=n_samples)

    votes, codes, ev_votes = {}, {}, {}
    valid_ids = set(blocks.keys())
    for text in outs:
        final = extract_final(text)
        if final is None:
            continue
        code = extract_code(text)
        value = _run_code(code) if code else None
        num = parse_out(value) if value is not None else parse_out(final)
        if num is None:
            num = parse_out(final)
        if num is None:
            continue
        votes[num] = votes.get(num, 0) + 1
        codes[num] = code
        # Extract evidence from solver output (majority vote)
        ev = extract_evidence_id(text, valid_ids=valid_ids)
        if ev:
            ev_votes[ev] = ev_votes.get(ev, 0) + 1
    if not votes:
        # fallback: last number seen in any output
        for text in outs:
            num = last_number(text)
            if num is not None:
                votes[num] = votes.get(num, 0) + 1
            ev = extract_evidence_id(text, valid_ids=valid_ids)
            if ev:
                ev_votes[ev] = ev_votes.get(ev, 0) + 1
    best_ev = max(ev_votes.items(), key=lambda kv: kv[1])[0] if ev_votes else None
    if not votes:
        return None, None, best_ev
    best = max(votes.items(), key=lambda kv: kv[1])[0]
    return best, codes.get(best), best_ev


def parse_out(value):
    from .normalize import parse_number

    n = parse_number(value)
    return n

