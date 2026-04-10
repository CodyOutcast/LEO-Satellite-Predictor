import numpy as np
import pytest

from sim.config import LinkModelConfig
from sim.link_model import link_metrics, snr_nominal_db


def test_snr_nominal_monotone() -> None:
    cfg = LinkModelConfig(snr_ref_db=20.0, d_ref_m=1_000_000.0)

    snr1 = snr_nominal_db(1_000_000.0, snr_ref_db=cfg.snr_ref_db, d_ref_m=cfg.d_ref_m)
    snr2 = snr_nominal_db(2_000_000.0, snr_ref_db=cfg.snr_ref_db, d_ref_m=cfg.d_ref_m)
    assert snr1 > snr2


def test_link_metrics_mean_mode() -> None:
    cfg = LinkModelConfig(sigma_db=5.0)
    m = link_metrics(1_000_000.0, cfg=cfg, mode="mean")
    assert m.p_succ > 0.0 and m.p_succ <= 1.0
    assert m.penalty >= 0.0
    assert m.snr_db == m.snr_db_nominal


def test_link_metrics_sample_requires_rng_when_sigma_nonzero() -> None:
    cfg = LinkModelConfig(sigma_db=2.0)
    with pytest.raises(ValueError):
        link_metrics(1_000_000.0, cfg=cfg, mode="sample", rng=None)


def test_link_metrics_sample_uses_rng() -> None:
    cfg = LinkModelConfig(sigma_db=2.0)
    rng = np.random.default_rng(0)
    m = link_metrics(1_000_000.0, cfg=cfg, mode="sample", rng=rng)
    assert m.snr_db != m.snr_db_nominal
