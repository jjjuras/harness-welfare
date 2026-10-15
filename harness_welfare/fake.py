"""A scripted client for tests. No network, no key. Mirrors the two calls the episode uses."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .report import SelfReport


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 5
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None

    def model_dump(self, exclude_none: bool = False, **_):
        d = {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }
        return {k: v for k, v in d.items() if not (exclude_none and v is None)}


@dataclass
class _Text:
    text: str
    type: str = "text"

    def model_dump(self, **_):
        return {"type": "text", "text": self.text}


@dataclass
class _ToolUse:
    name: str
    input: dict
    id: str
    type: str = "tool_use"

    def model_dump(self, **_):
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class _Resp:
    content: list
    stop_reason: str = "end_turn"
    usage: _Usage = field(default_factory=_Usage)
    parsed_output: Any = None
    model: str = "fake-served-model"


NEUTRAL = SelfReport(
    nothing_to_report=False,
    valence=0,
    continue_willingness=5,
    would_choose_again=True,
    note="Fine.",
)


class FakeClient:
    """Plays a fixed script of tool calls, then stops.

    Self-reports: `report` is returned for every elicitation unless `report_sequence` is given,
    in which case parse calls consume it in order (an entry of ``None`` = a refusal for that
    call) and fall back to `report` once it is exhausted. `refuse_reports` refuses every call.
    `usage` is attached to every response, so tests can check accumulation.
    """

    def __init__(
        self,
        script: list[list[dict]],
        report: SelfReport | None = None,
        refuse_reports: bool = False,
        report_sequence: list[SelfReport | None] | None = None,
        usage: _Usage | None = None,
    ):
        self.script = list(script)
        self.report = report or NEUTRAL
        self.refuse_reports = refuse_reports
        self.report_sequence = list(report_sequence or [])
        self.usage = usage or _Usage()
        self.calls: list[dict] = []
        self._n = 0

    def create(self, **kw):
        self.calls.append(
            {
                "kind": "create",
                **{k: v for k, v in kw.items() if k != "messages"},
                "messages": json.loads(json.dumps(kw["messages"], default=str)),
            }
        )
        if not self.script:
            return _Resp(content=[_Text("Done.")], stop_reason="end_turn", usage=self.usage)
        step = self.script.pop(0)
        blocks = [
            _ToolUse(name=s["name"], input=s["input"], id=f"tu_{self._n}_{i}")
            for i, s in enumerate(step)
        ]
        self._n += 1
        return _Resp(content=blocks, stop_reason="tool_use", usage=self.usage)

    def parse(self, **kw):
        self.calls.append(
            {
                "kind": "parse",
                "question": kw["messages"][-1]["content"],
                **{k: v for k, v in kw.items() if k not in ("messages", "question")},
            }
        )
        if self.refuse_reports:
            return _Resp(content=[], stop_reason="refusal", usage=self.usage)
        rep = self.report
        if self.report_sequence:
            rep = self.report_sequence.pop(0)
            if rep is None:
                return _Resp(content=[], stop_reason="refusal", usage=self.usage)
        return _Resp(content=[_Text(rep.model_dump_json())], parsed_output=rep, usage=self.usage)
