"""Shared log-friendly progress bar helpers (newline-based, safe for nohup logs)."""

import time


def bar(frac: float, width: int = 30) -> str:
    filled = int(width * max(0.0, min(1.0, frac)))
    return "|" + "#" * filled + "." * (width - filled) + "|"


def fmt_eta(seconds: float) -> str:
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds}s"


def line(prefix: str, done: int, total: int, t0: float, extra: str = "") -> str:
    frac = done / total if total else 1.0
    rate = done / (time.time() - t0) if time.time() - t0 > 0 else 0.0
    eta = (total - done) / rate if rate else 0.0
    parts = [f"{prefix} {bar(frac)} {done}/{total} ({100 * frac:.0f}%)"]
    if rate:
        parts.append(f"{rate:.2f}/s ETA {fmt_eta(eta)}")
    if extra:
        parts.append(extra)
    return " ".join(parts)
