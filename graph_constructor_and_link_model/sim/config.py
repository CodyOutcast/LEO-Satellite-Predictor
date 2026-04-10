from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

Mode = Literal["mean", "sample"]


@dataclass(frozen=True)
class EarthConstants:
    """Physical constants used by the simulator (MVP spherical Earth)."""

    R_earth_m: float = 6_371_000.0
    mu_m3_s2: float = 3.986_004_418e14
    omega_earth_rad_s: float = 7.292_115_9e-5
    c_m_s: float = 299_792_458.0


@dataclass(frozen=True)
class SimConfig:
    dt_s: float
    t_end_s: float
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dt_s <= 0:
            raise ValueError("dt_s must be > 0")
        if self.t_end_s < 0:
            raise ValueError("t_end_s must be >= 0")

    @property
    def num_steps(self) -> int:
        # Inclusive of t=0
        return int(round(self.t_end_s / self.dt_s)) + 1


@dataclass(frozen=True)
class ConstellationConfig:
    """Simple circular-orbit constellation configuration.

    Notes:
    - All satellites share altitude and inclination.
    - Planes are spaced in RAAN.
    - Slots are spaced in mean anomaly, with an optional per-plane phase offset.
    """

    num_planes: int
    sats_per_plane: int
    altitude_m: float
    inclination_deg: float

    raan_offset_deg: float = 0.0
    raan_spacing_deg: Optional[float] = None

    # Adds p*phase_offset_deg to all sats in plane p (Walker-like phasing)
    phase_offset_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.num_planes <= 0:
            raise ValueError("num_planes must be > 0")
        if self.sats_per_plane <= 0:
            raise ValueError("sats_per_plane must be > 0")
        if self.altitude_m <= 0:
            raise ValueError("altitude_m must be > 0")


@dataclass(frozen=True)
class GroundStation:
    name: str
    lat_deg: float
    lon_deg: float
    alt_m: float = 0.0

    @property
    def node_id(self) -> str:
        return f"GS-{self.name}"


@dataclass(frozen=True)
class LinkConstraints:
    theta_min_deg: float = 10.0
    gs_range_max_m: Optional[float] = None

    isl_range_max_m: float = 2_000_000.0
    earth_occlusion: bool = True
    occlusion_margin_m: float = 0.0

    isl_mode: Literal["neighbor", "all_within_range"] = "neighbor"

    def __post_init__(self) -> None:
        if self.theta_min_deg < 0 or self.theta_min_deg > 90:
            raise ValueError("theta_min_deg must be in [0, 90]")
        if self.gs_range_max_m is not None and self.gs_range_max_m <= 0:
            raise ValueError("gs_range_max_m must be > 0 if provided")
        if self.isl_range_max_m <= 0:
            raise ValueError("isl_range_max_m must be > 0")


@dataclass(frozen=True)
class NodeFailureWindow:
    """Schedule a node to be treated as failed (isolated) in sample mode."""

    t_idx_start: int
    t_idx_end: int
    node_id: str

    def __post_init__(self) -> None:
        if self.t_idx_start < 0 or self.t_idx_end < 0:
            raise ValueError("t_idx_start/t_idx_end must be >= 0")
        if self.t_idx_end < self.t_idx_start:
            raise ValueError("t_idx_end must be >= t_idx_start")

    def active(self, t_idx: int) -> bool:
        return self.t_idx_start <= t_idx <= self.t_idx_end


@dataclass(frozen=True)
class LinkModelConfig:
    """Delay + reliability link model configuration."""

    snr_ref_db: float = 20.0
    d_ref_m: float = 1_000_000.0

    # Fading/interference noise std (dB). 0 disables fading.
    sigma_db: float = 0.0

    # Logistic mapping parameters: p_succ = sigmoid((snr_db - gamma)/k)
    snr_threshold_db: float = 10.0
    snr_softness_db: float = 2.0

    # Weight scaling in seconds: weight = delay_s + w_rel_s * penalty
    w_rel_s: float = 0.01

    # Optional binary down model in sample mode
    snr_down_threshold_db: Optional[float] = None

    # Optional independent random edge drop in sample mode
    p_edge_fail: float = 0.0

    # Optional node failure schedule in sample mode
    node_failure_schedule: Sequence[NodeFailureWindow] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.d_ref_m <= 0:
            raise ValueError("d_ref_m must be > 0")
        if self.sigma_db < 0:
            raise ValueError("sigma_db must be >= 0")
        if self.snr_softness_db <= 0:
            raise ValueError("snr_softness_db must be > 0")
        if self.w_rel_s < 0:
            raise ValueError("w_rel_s must be >= 0")
        if not (0.0 <= self.p_edge_fail <= 1.0):
            raise ValueError("p_edge_fail must be in [0, 1]")


@dataclass(frozen=True)
class GraphBuilderConfig:
    sim: SimConfig
    constellation: ConstellationConfig
    ground_stations: Sequence[GroundStation]

    links: LinkConstraints = field(default_factory=LinkConstraints)
    link_model: LinkModelConfig = field(default_factory=LinkModelConfig)
    earth: EarthConstants = field(default_factory=EarthConstants)
