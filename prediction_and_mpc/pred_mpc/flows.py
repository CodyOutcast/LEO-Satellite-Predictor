from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Flow:
    name: str
    src: str
    dst: str


DEFAULT_FLOWS: tuple[Flow, ...] = (
    Flow(name="SF-LON", src="GS-SF", dst="GS-LON"),
    Flow(name="SF-SIN", src="GS-SF", dst="GS-SIN"),
    Flow(name="LON-SIN", src="GS-LON", dst="GS-SIN"),
)
