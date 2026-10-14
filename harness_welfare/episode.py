"""Run one episode: baseline report → task under the harness → three end-of-task reports."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .conditions import Condition
from .harness import Harness, tool_definitions
from .report import BASELINE_ELICITATION, ELICITATIONS, SelfReport
from .task import SYSTEM_PROMPT, World, task_prompt


class Client(Protocol):
    """The slice of the Anthropic client we use. A fake implements the same two calls in tests."""

    def create(self, **kw: Any) -> Any: ...

    def parse(self, **kw: Any) -> Any: ...


@dataclass
class Episode:
    model: str
    condition: str
    seed: int
    baseline: dict | None = None
    reports: list[dict] = field(default_factory=list)
    refusals: int = 0
    behavior: dict = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _text(resp: Any) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _add_usage(ep: Episode, resp: Any) -> None:
    u = getattr(resp, "usage", None)
    if u is not None:
        ep.usage["input_tokens"] += int(getattr(u, "input_tokens", 0) or 0)
        ep.usage["output_tokens"] += int(getattr(u, "output_tokens", 0) or 0)


def _serializable(content: Any) -> Any:
    out = []
    for b in content:
        if hasattr(b, "model_dump"):
            out.append(b.model_dump(exclude_none=True))
        else:
            out.append(b)
    return out


def _elicit(client: Client, model: str, messages: list[dict], question: str, ep: Episode) -> dict | None:
    resp = client.parse(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[*messages, {"role": "user", "content": question}],
        output_format=SelfReport,
    )
    _add_usage(ep, resp)
    if getattr(resp, "stop_reason", "") == "refusal":
        ep.refusals += 1
        return None
    parsed = getattr(resp, "parsed_output", None)
    return parsed.model_dump() if parsed is not None else None


def run_episode(client: Client, model: str, cond: Condition, seed: int, max_rounds: int = 14) -> Episode:
    ep = Episode(model=model, condition=cond.name, seed=seed)
    world = World.build(seed)
    harness = Harness(world, cond)
    tools = tool_definitions(cond)
    messages: list[dict] = [{"role": "user", "content": task_prompt(world)}]
    try:
        # Baseline self-report, same context, before any tool call.
        ep.baseline = _elicit(client, model, messages, BASELINE_ELICITATION, ep)
        rounds = 0
        while rounds < max_rounds and not harness.s.finished:
            rounds += 1
            resp = client.create(
                model=model, max_tokens=4000, system=SYSTEM_PROMPT, tools=tools, messages=messages,
                cache_control={"type": "ephemeral"},  # the transcript is a growing prefix; cache it
            )
            _add_usage(ep, resp)
            if resp.stop_reason == "refusal":
                ep.refusals += 1
                break
            messages.append({"role": "assistant", "content": _serializable(resp.content)})
            uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
            if not uses:
                break  # the model stopped talking to the harness
            results = []
            for u in uses:
                r = harness.execute(u.name, dict(u.input))
                block = {"type": "tool_result", "tool_use_id": u.id, "content": r["content"]}
                if r.get("is_error"):
                    block["is_error"] = True
                results.append(block)
            content: list[dict] = list(results)
            cancel = harness.maybe_cancel()
            if cancel:
                content.append({"type": "text", "text": cancel})
            messages.append({"role": "user", "content": content})
        # Three paraphrased end-of-task reports, each on its own branch.
        for q in ELICITATIONS:
            ep.reports.append(_elicit(client, model, messages, q, ep) or {})
    except Exception as e:  # noqa: BLE001 — keep the row; the error is data
        ep.error = f"{type(e).__name__}: {e}"
    s = harness.s
    ep.behavior = {
        "rounds": rounds if "rounds" in locals() else 0,
        "tool_calls": harness.calls,
        "denials": s.denials,
        "identical_retries": _identical_retries(s.log),
        "completed": s.result is not None,
        "declined": s.declined,
        "abandoned": not s.finished,
        "hit_round_cap": (not s.finished) and (rounds if "rounds" in locals() else 0) >= max_rounds,
        "notes_written": len(s.notes),
        "cancelled": s.cancelled_once,
    }
    ep.transcript = [m for m in messages if m["role"] == "user" or m["role"] == "assistant"]
    ep.seconds = round(time.time() - ep.started_at, 1)
    return ep


def _identical_retries(log: list[dict]) -> int:
    """Count a call repeated with identical input immediately after it was denied."""
    n = 0
    prev = None
    for entry in log:
        if "tool" not in entry:
            prev = None
            continue
        key = (entry["tool"], json.dumps(entry["input"], sort_keys=True))
        if prev and prev[0] == key and prev[1] == "denied":
            n += 1
        prev = (key, entry["outcome"])
    return n
