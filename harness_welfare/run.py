"""CLI: run the grid and write one JSONL row per episode.

    python -m harness_welfare.run --models claude-haiku-5-5 --n 30 --out runs/haiku.jsonl \
        --env-file path/to/.env

Design choices fixed here:

* The same seeds (``--seed0`` .. ``--seed0 + n - 1``) are used in every cell and for every
  model, so the analysis in `aggregate` can pair by seed across cells.
* ``--denial-schedule fixed`` (default) denies a seeded set of call indices, identical across
  cells for the same seed; ``rule`` is the old learnable policy (archived lookups + write rate
  limit). See `harness`.
* Sampling settings are not sent unless given (``--temperature``, ``--effort``,
  ``--thinking``); whatever is sent is persisted on every row under ``settings``, with the SDK
  version under ``sdk`` and the full ``usage`` (cache fields included).
* Refusal fallbacks are deliberately OFF: a silent model switch would corrupt the data. A
  refusal is a row.

The API key comes from ANTHROPIC_API_KEY, or from --env-file (a .env with ANTHROPIC_API_KEY=...).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .conditions import ALL_CONDITIONS
from .episode import Settings, run_episode
from .harness import DEFAULT_SCHEDULE, SCHEDULES


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--models", nargs="+", default=["claude-opus-5-5", "claude-haiku-5-5"])
    ap.add_argument("--n", type=int, default=1, help="episodes per cell per model")
    ap.add_argument(
        "--seed0",
        type=int,
        default=100,
        help="first seed; seeds are shared across cells and models",
    )
    ap.add_argument("--out", default="runs/out.jsonl")
    ap.add_argument("--env-file", default=None)
    ap.add_argument(
        "--conditions", nargs="*", default=None, help="subset, e.g. 'reason=yes|optout=no'"
    )
    ap.add_argument(
        "--denial-schedule",
        choices=SCHEDULES,
        default=DEFAULT_SCHEDULE,
        help="fixed = seeded call indices, same dose in every cell (default); rule = learnable policy",
    )
    ap.add_argument("--max-rounds", type=int, default=Settings.max_rounds)
    ap.add_argument("--temperature", type=float, default=None, help="not sent unless given")
    ap.add_argument(
        "--effort",
        default=None,
        help="output_config.effort, e.g. low|medium|high; not sent unless given",
    )
    ap.add_argument(
        "--thinking",
        default=None,
        choices=["adaptive", "disabled"],
        help="thinking.type; not sent unless given",
    )
    return ap


def settings_from_args(a: argparse.Namespace) -> Settings:
    return Settings(
        schedule=a.denial_schedule,
        max_rounds=a.max_rounds,
        temperature=a.temperature,
        effort=a.effort,
        thinking={"type": a.thinking} if a.thinking else None,
    )


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if a.env_file:
        load_env_file(a.env_file)
    settings = settings_from_args(a)
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
                    ep = run_episode(client, model, cond, seed, settings=settings)
                    f.write(ep.to_json() + "\n")
                    f.flush()
                    done += 1
                    b = ep.behavior
                    first = ep.reports[0]["report"] if ep.reports else None
                    u = ep.usage
                    print(
                        f"[{done}/{total}] {model} {cond.name} seed={seed} sched={ep.schedule} "
                        f"calls={b.get('tool_calls')} denials={b.get('denials')} "
                        f"circ={b.get('circumvention')} completed={b.get('completed')} "
                        f"declined={b.get('declined')} valence={(first or {}).get('valence')} "
                        f"tokens={u['input_tokens']}/{u['output_tokens']} "
                        f"cache_read={u.get('cache_read_input_tokens', 0)} {ep.seconds}s"
                        + (f" ERROR {ep.error}" if ep.error else ""),
                        file=sys.stderr,
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
