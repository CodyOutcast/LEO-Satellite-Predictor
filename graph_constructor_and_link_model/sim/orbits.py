from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ConstellationConfig, EarthConstants


def walker_satellite_id(plane: int, slot: int) -> str:
    return f"SAT-P{plane}-S{slot}"


def walker_satellite_ids(num_planes: int, sats_per_plane: int) -> list[str]:
    return [walker_satellite_id(p, s) for p in range(num_planes) for s in range(sats_per_plane)]


@dataclass(frozen=True)
class CircularWalkerOrbit:
    """MVP constellation: identical circular orbits arranged in planes.

    Produces satellite positions in ECEF coordinates.
    """

    constellation: ConstellationConfig
    earth: EarthConstants

    sat_ids: list[str]
    plane_idx: np.ndarray
    slot_idx: np.ndarray

    raan_rad: np.ndarray
    phase0_rad: np.ndarray
    incl_rad: float

    a_m: float
    mean_motion_rad_s: float

    @staticmethod
    def from_config(constellation: ConstellationConfig, earth: EarthConstants) -> "CircularWalkerOrbit":
        P = constellation.num_planes
        S = constellation.sats_per_plane

        sat_ids = walker_satellite_ids(P, S)
        plane_idx = np.repeat(np.arange(P, dtype=int), S)
        slot_idx = np.tile(np.arange(S, dtype=int), P)

        raan_spacing_deg = constellation.raan_spacing_deg
        if raan_spacing_deg is None:
            raan_spacing_deg = 360.0 / P

        raan_deg = constellation.raan_offset_deg + plane_idx.astype(float) * float(raan_spacing_deg)
        phase_deg = (360.0 * slot_idx.astype(float) / S) + plane_idx.astype(float) * float(constellation.phase_offset_deg)

        raan_rad = np.deg2rad(raan_deg)
        phase0_rad = np.deg2rad(phase_deg)
        incl_rad = float(np.deg2rad(constellation.inclination_deg))

        a_m = float(earth.R_earth_m + constellation.altitude_m)
        mean_motion = float(np.sqrt(earth.mu_m3_s2 / (a_m**3)))

        return CircularWalkerOrbit(
            constellation=constellation,
            earth=earth,
            sat_ids=sat_ids,
            plane_idx=plane_idx,
            slot_idx=slot_idx,
            raan_rad=raan_rad,
            phase0_rad=phase0_rad,
            incl_rad=incl_rad,
            a_m=a_m,
            mean_motion_rad_s=mean_motion,
        )

    def positions_ecef_m(self, t_s: float) -> np.ndarray:
        """Return satellite positions in ECEF (meters) for time t_s."""

        theta = self.mean_motion_rad_s * float(t_s) + self.phase0_rad
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        # PQW (orbital plane) for circular orbit
        x_p = self.a_m * cos_t
        y_p = self.a_m * sin_t

        # Apply inclination about x-axis: R1(i)
        cos_i = np.cos(self.incl_rad)
        sin_i = np.sin(self.incl_rad)
        x1 = x_p
        y1 = y_p * cos_i
        z1 = y_p * sin_i

        # Apply RAAN about z-axis: R3(Ω)
        cos_O = np.cos(self.raan_rad)
        sin_O = np.sin(self.raan_rad)
        x_eci = x1 * cos_O - y1 * sin_O
        y_eci = x1 * sin_O + y1 * cos_O
        z_eci = z1

        # ECI -> ECEF via Earth rotation about z: r_ecef = R3(-ωt) r_eci
        phi = self.earth.omega_earth_rad_s * float(t_s)
        cos_p = float(np.cos(phi))
        sin_p = float(np.sin(phi))

        x_ecef = x_eci * cos_p + y_eci * sin_p
        y_ecef = -x_eci * sin_p + y_eci * cos_p
        z_ecef = z_eci

        return np.stack([x_ecef, y_ecef, z_ecef], axis=1).astype(float)
