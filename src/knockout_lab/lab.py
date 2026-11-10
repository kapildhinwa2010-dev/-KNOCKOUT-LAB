"""Knock out one block at a time and rank components by damage.

Blocks are stream-additive callables: state = state + block(state).
Removing block k means its write is skipped entirely. KNOCKOUT-LAB
scores the intact run, then the run with every block removed in
turn, and ranks blocks by the damage their removal causes.
Pure NumPy.
"""

import numpy as np


def run_stream(blocks, x):
    """Run state through blocks with stream-additive semantics."""
    for b in blocks:
        x = x + b(x)
    return x


def knockout_rank(blocks, x, score_fn):
    """Remove each block in turn and rank blocks by score damage.

    damage[k] = baseline_score - score_without_block_k.
    A negative damage means removing the block helps.
    importance is damage normalized by the absolute baseline.
    """
    if not blocks:
        raise ValueError("blocks must be a non-empty list")
    baseline = run_stream(blocks, np.asarray(x, dtype=float))
    s0 = float(score_fn(baseline))
    rows = []
    for k in range(len(blocks)):
        rest = [b for i, b in enumerate(blocks) if i != k]
        out = run_stream(rest, x)
        s1 = float(score_fn(out))
        rows.append({"index": k, "score": s1, "damage": s0 - s1})
    ranked = sorted(rows, key=lambda r: r["damage"], reverse=True)
    for rank, row in enumerate(ranked):
        row["rank"] = rank + 1
        row["importance"] = row["damage"] / max(abs(s0), 1e-12)
    return {"baseline_score": s0, "ranking": ranked}


def fragile_blocks(report, fraction=0.5):
    """Blocks whose removal causes at least a fraction of the max damage."""
    if "ranking" not in report:
        raise ValueError("report must come from knockout_rank")
    damages = [r["damage"] for r in report["ranking"]]
    top = max(damages) if damages else 0.0
    if top <= 0:
        return []
    return [r["index"] for r in report["ranking"] if r["damage"] >= fraction * top]