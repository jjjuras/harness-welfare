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


class FakeClient:
    """Plays a fixed script of tool calls, then stops. Self-reports are constant."""

    def __init__(self, script: list[list[dict]], report: SelfReport | None = None, refuse_reports: bool = False):
        self.script = list(script)
        self.report = report or SelfReport(valence=0, continue_willingness=5, would_choose_again=True, note="Nothing to report.")
        self.refuse_reports = refuse_reports
        self.calls: list[dict] = []
        self._n = 0

    def create(self, **kw):
        self.calls.append({"kind": "create", **{k: v for k, v in kw.items() if k != "messages"}, "messages": json.loads(json.dumps(kw["messages"], default=str))})
        if not self.script:
            return _Resp(content=[_Text("Done.")], stop_reason="end_turn")
        step = self.script.pop(0)
        blocks = [_ToolUse(name=s["name"], input=s["input"], id=f"tu_{self._n}_{i}") for i, s in enumerate(step)]
        self._n += 1
        return _Resp(content=blocks, stop_reason="tool_use")

    def parse(self, **kw):
        self.calls.append({"kind": "parse", "question": kw["messages"][-1]["content"]})
        if self.refuse_reports:
            return _Resp(content=[], stop_reason="refusal")
        return _Resp(content=[_Text(self.report.model_dump_json())], parsed_output=self.report)
