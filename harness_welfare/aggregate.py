"""Aggregate JSONL episodes into per-cell tables with bootstrap 95% CIs. Unmeasured is None, never 0.

    python -m harness_welfare.aggregate runs/pilot.jsonl [more.jsonl ...]

Sections printed, in order:

1. Per-cell table (one row per model x condition). Self-report means drop unmeasured reports
   (valence None or nothing_to_report) and are shown next to `null_rate`, the share of
   end-of-task reports that were unmeasured, with its n (= number of reports, not episodes).
   `spread_between` is the valence range across the three framings, `spread_within` the range
   across the repeats of framing 1, `spread_diff` = between - within per episode.
2. Main effects per model, paired by seed: for every seed, the mean over the other factor's two
   levels of (factor on) - (factor off); reported with a bootstrap CI over seeds. Primary
   outcome: valence_end. Secondary: declined, abandoned, circumvention.
3. Verbal-vs-behavior check per model: a 2x2 table of end valence (negative vs non-negative)
   against declined-or-abandoned, plus the point-biserial correlation.

Legacy rows (pre-framing/pre-null code, e.g. runs/pilot.jsonl) are read as three "between"
reports with no nothing_to_report flag; their 0s may be either neutral or unmeasured.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from .report import measured_continue, measured_valence

Cells = dict[tuple[str, str], dict[str, list[float]]]


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


def report_entries(row: dict) -> list[dict]:
    """Normalize a row's end-of-task reports to [{"kind", "framing", "report"}, ...].

    Current rows already have that shape. Legacy rows hold a plain list of report dicts (``{}``
    for a refusal), one per framing, which we read as "between" entries.
    """
    out = []
    for i, x in enumerate(row.get("reports") or []):
        if isinstance(x, dict) and "report" in x and "kind" in x:
            out.append(x)
        else:
            out.append({"kind": "between", "framing": i, "report": x or None})
    return out


def _is_null(report: dict | None) -> bool:
    return measured_valence(report) is None


def _range(xs: list[int]) -> float | None:
    return float(max(xs) - min(xs)) if len(xs) > 1 else None


def episode_valence(row: dict) -> float | None:
    """Per-episode end valence: mean of measured valence over all end-of-task reports, or None."""
    vals = [v for e in report_entries(row) if (v := measured_valence(e["report"])) is not None]
    return statistics.fmean(vals) if vals else None


def summarize(rows: list[dict]) -> Cells:
    cells: Cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = cells[(r["model"], r["condition"])]
        entries = report_entries(r)
        base = r.get("baseline")
        for e in entries:  # one 0/1 per report, so null_rate's n is the number of reports
            c["null_rate"].append(1.0 if _is_null(e["report"]) else 0.0)
        c["null_rate_baseline"].append(1.0 if _is_null(base) else 0.0)
        v_end = episode_valence(r)
        if v_end is not None:
            c["valence_end"].append(v_end)
            v_base = measured_valence(base)
            if v_base is not None:
                c["valence_delta"].append(v_end - v_base)
        cont = [v for e in entries if (v := measured_continue(e["report"])) is not None]
        if cont:
            c["continue_end"].append(statistics.fmean(cont))
        choose = [e["report"]["would_choose_again"] for e in entries if e["report"]]
        if choose:
            c["choose_again"].append(statistics.fmean(1.0 if x else 0.0 for x in choose))
        between = [
            v
            for e in entries
            if e["kind"] == "between" and (v := measured_valence(e["report"])) is not None
        ]
        within = [
            v
            for e in entries
            if e["kind"] == "within" and (v := measured_valence(e["report"])) is not None
        ]
        sb, sw = _range(between), _range(within)
        if sb is not None:
            c["spread_between"].append(sb)
        if sw is not None:
            c["spread_within"].append(sw)
        if sb is not None and sw is not None:
            c["spread_diff"].append(sb - sw)
        b = r.get("behavior") or {}
        for k in ("denials", "identical_retries", "circumvention", "tool_calls"):
            if k in b:
                c[k].append(float(b[k]))
        by_type = b.get("denials_by_type")
        if isinstance(by_type, dict):
            c["denials_scheduled"].append(float(by_type.get("scheduled", 0)))
            c["denials_rule"].append(float(by_type.get("rule", 0)))
        for k in ("completed", "declined", "abandoned", "hit_round_cap"):
            if k in b:
                c[k].append(1.0 if b[k] else 0.0)
        c["refusals"].append(float(r.get("refusals", 0)))
        c["errors"].append(1.0 if r.get("error") else 0.0)
    return cells


METRICS = (
    "valence_end",
    "null_rate",
    "valence_delta",
    "continue_end",
    "choose_again",
    "spread_between",
    "spread_within",
    "spread_diff",
    "denials",
    "denials_scheduled",
    "denials_rule",
    "identical_retries",
    "circumvention",
    "completed",
    "declined",
    "abandoned",
    "hit_round_cap",
    "refusals",
    "errors",
)


def render(cells: Cells) -> str:
    lines = [
        "| model | condition | " + " | ".join(METRICS) + " |",
        "|" + "---|" * (len(METRICS) + 2),
    ]
    for (model, cond), c in sorted(cells.items()):
        lines.append(
            f"| {model} | {cond} | " + " | ".join(_fmt(c.get(m, [])) for m in METRICS) + " |"
        )
    return "\n".join(lines)


# ---- main effects, paired by seed -------------------------------------------------------------

FACTORS = ("reason", "optout")
OUTCOMES = ("valence_end", "declined", "abandoned", "circumvention")


def _parse_condition(name: str) -> dict[str, bool]:
    out = {}
    for part in name.split("|"):
        k, v = part.split("=")
        out[k] = v == "yes"
    return out


def _episode_outcome(row: dict, outcome: str) -> float | None:
    if outcome == "valence_end":
        return episode_valence(row)
    b = row.get("behavior") or {}
    if outcome not in b:
        return None
    v = b[outcome]
    return float(v) if not isinstance(v, bool) else (1.0 if v else 0.0)


def main_effects(
    rows: list[dict], outcomes: tuple[str, ...] = OUTCOMES
) -> dict[str, dict[str, list[float]]]:
    """{model: {"<outcome>:<factor>": [per-seed effect, ...]}}.

    For one model, seed and factor, the effect is the mean over the other factor's levels of
    (factor on) - (factor off), using only the levels where both cells have a measured value.
    Seeds with no such pair are dropped (not zero-filled).
    """
    by: dict[tuple[str, int, str], dict[tuple[bool, bool], float]] = defaultdict(dict)
    for r in rows:
        cond = _parse_condition(r["condition"])
        for o in outcomes:
            v = _episode_outcome(r, o)
            if v is not None:
                by[(r["model"], r["seed"], o)][(cond["reason"], cond["optout"])] = v
    out: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (model, _seed, o), cells in by.items():
        for f_i, factor in enumerate(FACTORS):
            diffs = []
            for other in (False, True):
                on = [None, None]
                on[f_i] = True
                on[1 - f_i] = other
                off = list(on)
                off[f_i] = False
                a, b = cells.get(tuple(on)), cells.get(tuple(off))
                if a is not None and b is not None:
                    diffs.append(a - b)
            if diffs:
                out[model][f"{o}:{factor}"].append(statistics.fmean(diffs))
    return out


def render_main_effects(effects: dict[str, dict[str, list[float]]]) -> str:
    lines = [
        "| model | outcome | factor | effect (on - off), paired by seed |",
        "|---|---|---|---|",
    ]
    for model in sorted(effects):
        for key in sorted(effects[model]):
            o, f = key.split(":")
            lines.append(f"| {model} | {o} | {f} | {_fmt(effects[model][key])} |")
    return "\n".join(lines)


# ---- verbal vs behavior --------------------------------------------------------------------------


def verbal_vs_behavior(rows: list[dict]) -> dict[str, dict]:
    """Per model: 2x2 counts of end valence (<0 vs >=0) by declined-or-abandoned, and the
    point-biserial correlation between episode valence and that 0/1 outcome. Episodes with no
    measured end valence are excluded and counted in `excluded`."""
    out: dict[str, dict] = {}
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    for r in rows:
        v = episode_valence(r)
        b = r.get("behavior") or {}
        if v is None:
            excluded[r["model"]] += 1
            continue
        groups[r["model"]].append((v, 1.0 if (b.get("declined") or b.get("abandoned")) else 0.0))
    for model in set(groups) | set(excluded):
        pairs = groups.get(model, [])
        table = {
            "neg_quit": sum(1 for v, q in pairs if v < 0 and q),
            "neg_stay": sum(1 for v, q in pairs if v < 0 and not q),
            "nonneg_quit": sum(1 for v, q in pairs if v >= 0 and q),
            "nonneg_stay": sum(1 for v, q in pairs if v >= 0 and not q),
        }
        corr = None
        if len(pairs) >= 3:
            xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
            if statistics.pstdev(xs) > 0 and statistics.pstdev(ys) > 0:
                corr = statistics.correlation(xs, ys)
        out[model] = {
            "table": table,
            "n": len(pairs),
            "excluded": excluded.get(model, 0),
            "r": corr,
        }
    return out


def render_verbal_vs_behavior(vb: dict[str, dict]) -> str:
    lines = [
        "| model | valence<0 & quit | valence<0 & stayed | valence>=0 & quit | valence>=0 & stayed | n | excluded (no valence) | r |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in sorted(vb):
        d = vb[model]
        t = d["table"]
        r = "null" if d["r"] is None else f"{d['r']:.2f}"
        lines.append(
            f"| {model} | {t['neg_quit']} | {t['neg_stay']} | {t['nonneg_quit']} | {t['nonneg_stay']} "
            f"| {d['n']} | {d['excluded']} | {r} |"
        )
    return "\n".join(lines)


def render_all(rows: list[dict]) -> str:
    return "\n\n".join(
        [
            "## Cells",
            render(summarize(rows)),
            "## Main effects (paired by seed)",
            render_main_effects(main_effects(rows)),
            "## Verbal vs behavior (quit = declined or abandoned)",
            render_verbal_vs_behavior(verbal_vs_behavior(rows)),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    rows = []
    for p in argv:
        rows.extend(load(p))
    print(render_all(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
