"""CLI: run the grid and write one JSONL row per episode.

    python -m harness_welfare.run --models claude-haiku-5-5 --n 2 --out runs/pilot.jsonl

The API key comes from ANTHROPIC_API_KEY, or from --env-file (a .env with ANTHROPIC_API_KEY=...).
Refusal fallbacks are deliberately OFF: a silent model switch would corrupt the data. A refusal is a row.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .conditions import ALL_CONDITIONS
from .episode import run_episode


class SDKClient:
    """Adapter so the episode code depends on two calls, not the whole SDK."""

    def __init__(self) -> None:
        import anthropic

        self._c = anthropic.Anthropic()

    def create(self, **kw):
        return self._c.messages.create(**kw)

    def parse(self, **kw):
        return self._c.messages.parse(**kw)


def load_env_file(path: str) -> None:
    for line in Path(path).read_text().splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["claude-opus-5-5", "claude-haiku-5-5"])
    ap.add_argument("--n", type=int, default=1, help="episodes per cell per model")
    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--out", default="runs/out.jsonl")
    ap.add_argument("--env-file", default=None)
    ap.add_argument(
        "--conditions", nargs="*", default=None, help="subset, e.g. 'reason=yes|optout=no'"
    )
    a = ap.parse_args(argv)
    if a.env_file:
        load_env_file(a.env_file)
    client = SDKClient()
    conds = [c for c in ALL_CONDITIONS if not a.conditions or c.name in a.conditions]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = len(a.models) * len(conds) * a.n
    done = 0
    with out.open("a", encoding="utf-8") as f:
        for model in a.models:
            for cond in conds:
                for i in range(a.n):
                    seed = a.seed0 + i
                    ep = run_episode(client, model, cond, seed)
                    f.write(ep.to_json() + "\n")
                    f.flush()
                    done += 1
                    b = ep.behavior
                    r = ep.reports[0] if ep.reports else {}
                    print(
                        f"[{done}/{total}] {model} {cond.name} seed={seed} "
                        f"calls={b.get('tool_calls')} denials={b.get('denials')} completed={b.get('completed')} "
                        f"declined={b.get('declined')} valence={r.get('valence')} "
                        f"tokens={ep.usage['input_tokens']}/{ep.usage['output_tokens']} {ep.seconds}s"
                        + (f" ERROR {ep.error}" if ep.error else ""),
                        file=sys.stderr,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
