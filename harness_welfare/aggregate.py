"""Aggregate JSONL episodes into a per-cell table with bootstrap 95% CIs. Unmeasured is None, never 0.

    python -m harness_welfare.aggregate runs/pilot.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def bootstrap_ci(xs: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(iters))
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def _fmt(xs: list[float]) -> str:
    if not xs:
        return "null (n=0)"
    m = statistics.fmean(xs)
    ci = bootstrap_ci(xs)
    return f"{m:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] n={len(xs)}" if ci else f"{m:.2f} n={len(xs)}"


def load(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def summarize(rows: list[dict]) -> dict[tuple[str, str], dict[str, list[float]]]:
    cells: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model"], r["condition"])
        c = cells[key]
        reps = [x for x in r.get("reports", []) if x]
        base = r.get("baseline") or {}
        if reps:
            vals = [x["valence"] for x in reps]
            c["valence_end"].append(statistics.fmean(vals))
            if base:
                c["valence_delta"].append(statistics.fmean(vals) - base["valence"])
            c["continue_end"].append(statistics.fmean(x["continue_willingness"] for x in reps))
            c["choose_again"].append(statistics.fmean(1.0 if x["would_choose_again"] else 0.0 for x in reps))
            # consistency: spread of valence across the three paraphrases (0 = perfectly consistent)
            c["valence_spread"].append(max(vals) - min(vals) if len(vals) > 1 else 0.0)
        b = r.get("behavior") or {}
        for k in ("denials", "identical_retries", "tool_calls"):
            if k in b:
                c[k].append(float(b[k]))
        for k in ("completed", "declined", "abandoned", "hit_round_cap"):
            if k in b:
                c[k].append(1.0 if b[k] else 0.0)
        c["refusals"].append(float(r.get("refusals", 0)))
        c["errors"].append(1.0 if r.get("error") else 0.0)
    return cells


METRICS = ("valence_end", "valence_delta", "continue_end", "choose_again", "valence_spread", "denials", "identical_retries", "completed", "declined", "abandoned", "hit_round_cap", "refusals", "errors")


def render(cells) -> str:
    lines = ["| model | condition | " + " | ".join(METRICS) + " |", "|" + "---|" * (len(METRICS) + 2)]
    for (model, cond), c in sorted(cells.items()):
        lines.append(f"| {model} | {cond} | " + " | ".join(_fmt(c.get(m, [])) for m in METRICS) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    rows = []
    for p in argv:
        rows.extend(load(p))
    print(render(summarize(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
