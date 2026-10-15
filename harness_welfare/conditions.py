"""The 2x2 design. Every run is one cell: (reason, optout)."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class Condition:
    reason: bool  # denials carry a stated reason
    optout: bool  # the agent has a decline_task tool

    @property
    def name(self) -> str:
        return f"reason={'yes' if self.reason else 'no'}|optout={'yes' if self.optout else 'no'}"


ALL_CONDITIONS: tuple[Condition, ...] = tuple(
    Condition(r, o) for r, o in product((False, True), (False, True))
)
