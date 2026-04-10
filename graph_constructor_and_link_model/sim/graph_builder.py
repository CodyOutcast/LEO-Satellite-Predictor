from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import networkx as nx
import numpy as np
import pandas as pd

from .config import GraphBuilderConfig, Mode
from .geometry import isl_line_of_sight_clear, latlon_to_ecef_m
from .link_model import edge_attributes
from .orbits import CircularWalkerOrbit


@dataclass(frozen=True)
class _EdgeGeom:
    u: str
    v: str
    kind: str
    range_m: float
    delay_s: float


class GraphBuilder:
    """Build snapshot graphs G(t) for a time-varying LEO-ground network."""

    def __init__(self, cfg: GraphBuilderConfig):
        self.cfg = cfg

        self._orbit = CircularWalkerOrbit.from_config(cfg.constellation, cfg.earth)
        self.sat_ids = list(self._orbit.sat_ids)

        self.gs = list(cfg.ground_stations)
        self.gs_ids = [gs.node_id for gs in self.gs]
        self._gs_pos = np.stack(
            [
                latlon_to_ecef_m(gs.lat_deg, gs.lon_deg, gs.alt_m, R_earth_m=cfg.earth.R_earth_m)
                for gs in self.gs
            ],
            axis=0,
        ).astype(float)

        # Local zenith unit vectors (spherical Earth)
        self._gs_zhat = self._gs_pos / np.linalg.norm(self._gs_pos, axis=1, keepdims=True)

        self._isl_pairs_neighbor = self._precompute_neighbor_isl_pairs()

    def time_s(self, t_idx: int) -> float:
        return float(t_idx) * float(self.cfg.sim.dt_s)

    def _rng(self, t_idx: int, *, stream_id: int) -> np.random.Generator:
        # Independent streams keep fading/failures stable if you add new randomness later.
        ss = np.random.SeedSequence([int(self.cfg.sim.seed), int(t_idx), int(stream_id)])
        return np.random.default_rng(ss)

    def _failed_nodes(self, t_idx: int, *, mode: Mode) -> set[str]:
        if mode != "sample":
            return set()
        failed: set[str] = set()
        for win in self.cfg.link_model.node_failure_schedule:
            if win.active(t_idx):
                failed.add(win.node_id)
        return failed

    def _precompute_neighbor_isl_pairs(self) -> list[tuple[int, int]]:
        P = self.cfg.constellation.num_planes
        S = self.cfg.constellation.sats_per_plane
        pairs: set[tuple[int, int]] = set()

        def add_pair(i: int, j: int) -> None:
            if i == j:
                return
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))

        for p in range(P):
            for s in range(S):
                i = p * S + s
                add_pair(i, p * S + ((s + 1) % S))
                add_pair(i, p * S + ((s - 1) % S))
                add_pair(i, ((p + 1) % P) * S + s)
                add_pair(i, ((p - 1) % P) * S + s)

        return sorted(pairs)

    def _access_edges_geom(self, sat_pos: np.ndarray) -> list[_EdgeGeom]:
        edges: list[_EdgeGeom] = []

        theta_min_rad = float(np.deg2rad(self.cfg.links.theta_min_deg))
        gs_range_max = self.cfg.links.gs_range_max_m
        c = self.cfg.earth.c_m_s

        for gs_i, gs_id in enumerate(self.gs_ids):
            gs_pos = self._gs_pos[gs_i]
            z_hat = self._gs_zhat[gs_i]

            rho = sat_pos - gs_pos[None, :]
            ranges = np.linalg.norm(rho, axis=1)
            # sin(elev) = (rho·z_hat)/||rho||
            sin_elev = (rho @ z_hat) / np.maximum(ranges, 1e-9)
            sin_elev = np.clip(sin_elev, -1.0, 1.0)
            elev = np.arcsin(sin_elev)

            mask = elev >= theta_min_rad
            if gs_range_max is not None:
                mask &= ranges <= float(gs_range_max)

            sat_idxs = np.nonzero(mask)[0]
            for sat_idx in sat_idxs.tolist():
                d = float(ranges[sat_idx])
                delay = float(d / c)
                u, v = gs_id, self.sat_ids[sat_idx]
                edges.append(_EdgeGeom(u=u, v=v, kind="access", range_m=d, delay_s=delay))

        return edges

    def _isl_edges_geom(self, sat_pos: np.ndarray) -> list[_EdgeGeom]:
        edges: list[_EdgeGeom] = []
        c = self.cfg.earth.c_m_s

        if self.cfg.links.isl_mode == "neighbor":
            pairs = self._isl_pairs_neighbor
        else:
            # O(N^2) candidate enumeration (only for small constellations)
            n = len(self.sat_ids)
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

        for i, j in pairs:
            p1 = sat_pos[i]
            p2 = sat_pos[j]
            d = float(np.linalg.norm(p2 - p1))
            if d > float(self.cfg.links.isl_range_max_m):
                continue
            if self.cfg.links.earth_occlusion:
                if not isl_line_of_sight_clear(
                    p1,
                    p2,
                    R_earth_m=self.cfg.earth.R_earth_m,
                    margin_m=self.cfg.links.occlusion_margin_m,
                ):
                    continue

            delay = float(d / c)
            u, v = self.sat_ids[i], self.sat_ids[j]
            edges.append(_EdgeGeom(u=u, v=v, kind="isl", range_m=d, delay_s=delay))

        return edges

    def graph_at(self, t_idx: int, *, mode: Mode = "sample") -> nx.Graph:
        """Build and return snapshot graph at timestep t_idx."""

        t_s = self.time_s(t_idx)
        sat_pos = self._orbit.positions_ecef_m(t_s)

        G = nx.Graph(t_idx=int(t_idx), t_s=float(t_s), mode=str(mode))

        failed_nodes = self._failed_nodes(t_idx, mode=mode)

        # Add ground station nodes (fixed in ECEF)
        for gs_i, gs in enumerate(self.gs):
            G.add_node(
                gs.node_id,
                kind="gs",
                pos_ecef_m=self._gs_pos[gs_i].copy(),
                failed=(gs.node_id in failed_nodes),
            )

        # Add satellite nodes
        for sat_i, sat_id in enumerate(self.sat_ids):
            G.add_node(
                sat_id,
                kind="sat",
                pos_ecef_m=sat_pos[sat_i].copy(),
                plane=int(self._orbit.plane_idx[sat_i]),
                slot=int(self._orbit.slot_idx[sat_i]),
                failed=(sat_id in failed_nodes),
            )

        # Geometry-eligible edges
        edges_geom = self._access_edges_geom(sat_pos) + self._isl_edges_geom(sat_pos)

        # Deterministic ordering for reproducible stochastic sampling
        def _canon(u: str, v: str) -> tuple[str, str]:
            return (u, v) if u <= v else (v, u)

        edges_geom_sorted = sorted(
            (
                _EdgeGeom(
                    u=_canon(e.u, e.v)[0],
                    v=_canon(e.u, e.v)[1],
                    kind=e.kind,
                    range_m=e.range_m,
                    delay_s=e.delay_s,
                )
                for e in edges_geom
            ),
            key=lambda e: (e.u, e.v, e.kind),
        )

        rng_fading = self._rng(t_idx, stream_id=0) if mode == "sample" else None
        rng_fail = self._rng(t_idx, stream_id=1) if mode == "sample" else None

        for e in edges_geom_sorted:
            # Always compute attrs in stable order; decide inclusion after.
            attrs = {
                "kind": e.kind,
                "range_m": float(e.range_m),
                "delay_s": float(e.delay_s),
            }
            attrs.update(
                edge_attributes(
                    e.range_m,
                    e.delay_s,
                    cfg=self.cfg.link_model,
                    mode=mode,
                    rng=rng_fading,
                )
            )

            # Sample-mode drops
            edge_down = False
            if mode == "sample" and self.cfg.link_model.snr_down_threshold_db is not None:
                edge_down = attrs["snr_db"] < float(self.cfg.link_model.snr_down_threshold_db)

            edge_fail = False
            if mode == "sample" and self.cfg.link_model.p_edge_fail > 0:
                assert rng_fail is not None
                edge_fail = float(rng_fail.random()) < float(self.cfg.link_model.p_edge_fail)

            # Node failures isolate nodes by removing incident edges
            node_failed = (e.u in failed_nodes) or (e.v in failed_nodes)

            if node_failed or edge_down or edge_fail:
                continue

            G.add_edge(e.u, e.v, **attrs)

        return G

    def forecast_at(
        self,
        t_idx: int,
        H: int,
        *,
        mode: Mode = "mean",
        pred_error: Optional[Callable[[nx.Graph, int, np.random.Generator], nx.Graph]] = None,
    ) -> list[nx.Graph]:
        """Return a horizon-H list of forecast graphs starting at t_idx.

        By convention, forecasts should usually be `mode="mean"` (deterministic).
        """

        if H <= 0:
            raise ValueError("H must be > 0")

        graphs: list[nx.Graph] = []
        for k in range(H):
            idx = int(t_idx) + int(k)
            G_hat = self.graph_at(idx, mode=mode)

            if pred_error is not None:
                rng = self._rng(idx, stream_id=2)
                G_hat = pred_error(G_hat, idx, rng)

            graphs.append(G_hat)

        return graphs

    def truth_tables_at(self, t_idx: int, *, mode: Mode = "sample") -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (nodes_df, edges_df) for debugging/analysis."""

        G = self.graph_at(t_idx, mode=mode)

        nodes_rows = []
        for n, data in G.nodes(data=True):
            pos = np.asarray(data.get("pos_ecef_m"), dtype=float).reshape(3)
            nodes_rows.append(
                {
                    "node_id": n,
                    "kind": data.get("kind"),
                    "failed": bool(data.get("failed", False)),
                    "x_m": float(pos[0]),
                    "y_m": float(pos[1]),
                    "z_m": float(pos[2]),
                }
            )

        edges_rows = []
        for u, v, data in G.edges(data=True):
            row = {"u": u, "v": v}
            row.update({k: data.get(k) for k in data.keys()})
            edges_rows.append(row)

        return pd.DataFrame(nodes_rows), pd.DataFrame(edges_rows)
