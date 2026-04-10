from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import LinkModelConfig, Mode


def _sigmoid(x: float) -> float:
    # Stable-ish sigmoid for moderate x
    return float(1.0 / (1.0 + np.exp(-x)))


def snr_nominal_db(distance_m: float, *, snr_ref_db: float, d_ref_m: float) -> float:
    d = float(max(distance_m, 1e-9))
    return float(snr_ref_db - 20.0 * np.log10(d / float(d_ref_m)))


@dataclass(frozen=True)
class LinkMetrics:
    snr_db_nominal: float
    snr_db: float
    p_succ: float
    penalty: float


def link_metrics(
    distance_m: float,
    *,
    cfg: LinkModelConfig,
    mode: Mode,
    rng: Optional[np.random.Generator] = None,
    eps: float = 1e-12,
) -> LinkMetrics:
    """Compute SNR proxy, success probability, and penalty for one link."""

    snr0 = snr_nominal_db(distance_m, snr_ref_db=cfg.snr_ref_db, d_ref_m=cfg.d_ref_m)

    snr = snr0
    if mode == "sample" and cfg.sigma_db > 0:
        if rng is None:
            raise ValueError("rng is required in sample mode when sigma_db > 0")
        snr = float(snr0 + rng.normal(0.0, cfg.sigma_db))

    x = (snr - cfg.snr_threshold_db) / cfg.snr_softness_db
    p_succ = _sigmoid(x)
    penalty = float(-np.log(max(p_succ, eps)))

    return LinkMetrics(
        snr_db_nominal=float(snr0),
        snr_db=float(snr),
        p_succ=float(p_succ),
        penalty=float(penalty),
    )


def edge_weight_s(delay_s: float, *, metrics: LinkMetrics, cfg: LinkModelConfig) -> float:
    return float(delay_s + cfg.w_rel_s * metrics.penalty)


def edge_attributes(
    distance_m: float,
    delay_s: float,
    *,
    cfg: LinkModelConfig,
    mode: Mode,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Return a dict of edge attributes including weight components."""

    m = link_metrics(distance_m, cfg=cfg, mode=mode, rng=rng)
    weight_rel_s = float(cfg.w_rel_s * m.penalty)
    weight = float(delay_s + weight_rel_s)

    return {
        "snr_db_nominal": m.snr_db_nominal,
        "snr_db": m.snr_db,
        "p_succ": m.p_succ,
        "penalty": m.penalty,
        "weight_delay_s": float(delay_s),
        "weight_rel_s": weight_rel_s,
        "weight": weight,
    }
