"""Joint metric: normalized answer exact match AND exact evidence block-set match."""

from .normalize import normalize_answer


def score_example(pred: dict, gold: dict) -> dict:
    pa = normalize_answer(pred.get("answer")) if pred.get("answer") is not None else None
    ga = normalize_answer(gold.get("answer")) if gold.get("answer") is not None else None
    pe = set(pred.get("evidence") or [])
    ge = set(gold.get("evidence") or [])

    answer_ok = pa is not None and pa == ga
    ev_exact = pe == ge and len(ge) > 0
    ev_f1 = 0.0
    if pe or ge:
        tp = len(pe & ge)
        prec = tp / len(pe) if pe else 0.0
        rec = tp / len(ge) if ge else 0.0
        ev_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "answer_exact": answer_ok,
        "evidence_exact": ev_exact,
        "evidence_f1": ev_f1,
        "joint": answer_ok and ev_exact,
    }


def score_all(preds: list, golds: list) -> dict:
    gold_by_id = {g["instance_id"]: g for g in golds}
    rows = []
    missing = 0
    for g in golds:
        p = next((x for x in preds if x["instance_id"] == g["instance_id"]), None)
        if p is None:
            missing += 1
            p = {"instance_id": g["instance_id"], "answer": None, "evidence": []}
        rows.append(score_example(p, g))
    n = len(golds)
    agg = {
        "n": n,
        "missing": missing,
        "joint": sum(r["joint"] for r in rows) / n if n else 0.0,
        "answer_exact": sum(r["answer_exact"] for r in rows) / n if n else 0.0,
        "evidence_exact": sum(r["evidence_exact"] for r in rows) / n if n else 0.0,
        "evidence_f1": sum(r["evidence_f1"] for r in rows) / n if n else 0.0,
    }
    return agg
