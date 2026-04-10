"""Time-varying LEO-ground network simulator primitives.

This package contains:
- Geometry utilities (ECEF, elevation, LOS/occlusion)
- A simple circular-orbit constellation provider (MVP)
- A link model (delay + reliability penalty with optional fading)
- A GraphBuilder that emits snapshot graphs G(t)

The public entrypoint for most users is `GraphBuilder`.
"""

from .config import (
    ConstellationConfig,
    EarthConstants,
    GraphBuilderConfig,
    GroundStation,
    LinkConstraints,
    LinkModelConfig,
    NodeFailureWindow,
    SimConfig,
)
from .graph_builder import GraphBuilder

__all__ = [
    "ConstellationConfig",
    "EarthConstants",
    "GraphBuilder",
    "GraphBuilderConfig",
    "GroundStation",
    "LinkConstraints",
    "LinkModelConfig",
    "NodeFailureWindow",
    "SimConfig",
]
