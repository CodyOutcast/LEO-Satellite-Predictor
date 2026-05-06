from __future__ import annotations

from typing import Protocol

import networkx as nx


class Predictor(Protocol):
    def predict(self, t_idx: int, H: int) -> list[nx.Graph]:
        ...


class Controller(Protocol):
    def step(self, t_idx: int) -> dict[str, list[str]]:
        ...
