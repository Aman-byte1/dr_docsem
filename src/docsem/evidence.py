"""Evidence block selection: keyword rule -> DeBERTa cross-encoder -> BM25 fallback."""

import re

WORD = re.compile(r"[a-z0-9]+")

KW_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"concerning\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)",
        r"regarding\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)",
        r"involving\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)",
        r"about\s+([a-zA-Z]+)\s+and\s+([a-zA-Z]+)",
    )
]

# Verified on the 908 train labels: gold evidence is always a single block in b06..b13.
EVID_LO, EVID_HI = 6, 13


def _tokens(text: str):
    return WORD.findall(text.lower())


def extract_keywords(query: str):
    for pat in KW_PATTERNS:
        m = pat.search(query)
        if m:
            return [m.group(1).lower(), m.group(2).lower()]
    return []


def _has_kw(text_lower: str, kw: str) -> bool:
    return (
        kw in text_lower
        or (kw.endswith("s") and kw[:-1] in text_lower)
        or (kw + "s") in text_lower
    )


def candidate_ids(blocks: dict):
    ids = [b for b in blocks if b.startswith("b") and b[1:].isdigit()]
    pool = [b for b in ids if EVID_LO <= int(b[1:]) <= EVID_HI]
    return pool or ids


def keyword_pick(query: str, blocks: dict):
    """Return the block id when exactly one candidate contains both query keywords."""
    kws = extract_keywords(query)
    if not kws:
        return None
    pool = candidate_ids(blocks)
    hits = [b for b in pool if all(_has_kw(blocks[b].lower(), k) for k in kws)]
    return hits[0] if len(hits) == 1 else None


def bm25_scores(query: str, blocks: dict):
    from rank_bm25 import BM25Okapi

    pool = candidate_ids(blocks)
    if not pool:
        return {}
    bm25 = BM25Okapi([_tokens(blocks[b]) for b in pool])
    scores = bm25.get_scores(_tokens(query))
    kws = extract_keywords(query)
    out = {}
    for b, s in zip(pool, scores):
        low = blocks[b].lower()
        boost = 2.0 * sum(1 for k in kws if _has_kw(low, k))
        out[b] = float(s) + boost
    return out


def bm25_pick(query: str, blocks: dict):
    scores = bm25_scores(query, blocks)
    if not scores:
        return None
    return max(scores.items(), key=lambda kv: kv[1])[0]


_DEBERTA_CACHE = {}


def _load_deberta(model_dir: str):
    """Load once, reuse across tasks (reloading per call would be very slow)."""
    if model_dir not in _DEBERTA_CACHE:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        _DEBERTA_CACHE[model_dir] = (tok, model, device)
    return _DEBERTA_CACHE[model_dir]


def deberta_pick(query: str, blocks: dict, model_dir: str, batch_size: int = 64):
    import torch

    tok, model, device = _load_deberta(model_dir)

    pool = candidate_ids(blocks)
    pairs = [(query, f"{b}: {blocks[b]}") for b in pool]
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            enc = tok(
                [q for q, _ in chunk],
                [b for _, b in chunk],
                truncation=True,
                max_length=384,
                padding=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.tolist())
    return max(zip(pool, scores), key=lambda kv: kv[1])[0]


def pick_evidence(query: str, blocks: dict, deberta_model_dir=None):
    """Returns (block_id, method)."""
    kw = keyword_pick(query, blocks)
    if kw:
        return kw, "keyword"
    if deberta_model_dir:
        try:
            return deberta_pick(query, blocks, deberta_model_dir), "deberta"
        except Exception:
            pass
    return bm25_pick(query, blocks), "bm25"
