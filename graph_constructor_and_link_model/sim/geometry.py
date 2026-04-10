from __future__ import annotations

import numpy as np


def latlon_to_ecef_m(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    *,
    R_earth_m: float,
) -> np.ndarray:
    """Convert geodetic-like (lat, lon, alt) to ECEF (spherical Earth MVP).

    This intentionally ignores Earth flattening (WGS84) to keep the MVP simple.
    """

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    r = R_earth_m + alt_m

    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    x = r * cos_lat * cos_lon
    y = r * cos_lat * sin_lon
    z = r * sin_lat

    return np.array([x, y, z], dtype=float)


def range_m(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.linalg.norm(p - q))


def elevation_rad(gs_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> float:
    """Elevation angle of satellite as seen from a ground station.

    Uses local zenith = radial direction of the ground station (spherical Earth).
    """

    rho = sat_ecef_m - gs_ecef_m
    rho_norm = np.linalg.norm(rho)
    if rho_norm == 0:
        return float(np.pi / 2)

    z_hat = gs_ecef_m / np.linalg.norm(gs_ecef_m)
    sin_elev = float(np.dot(rho, z_hat) / rho_norm)
    sin_elev = float(np.clip(sin_elev, -1.0, 1.0))
    return float(np.arcsin(sin_elev))


def elevation_deg(gs_ecef_m: np.ndarray, sat_ecef_m: np.ndarray) -> float:
    return float(np.rad2deg(elevation_rad(gs_ecef_m, sat_ecef_m)))


def isl_line_of_sight_clear(
    r1_ecef_m: np.ndarray,
    r2_ecef_m: np.ndarray,
    *,
    R_earth_m: float,
    margin_m: float = 0.0,
) -> bool:
    """Return True if the ISL segment does NOT intersect (or graze) Earth.

    Uses the minimum distance from the Earth center to the segment connecting
    r1 and r2 (distance-to-segment test).
    """

    r1 = np.asarray(r1_ecef_m, dtype=float)
    r2 = np.asarray(r2_ecef_m, dtype=float)
    d = r2 - r1

    denom = float(np.dot(d, d))
    if denom == 0:
        # Same point; treat as LOS clear (degenerate), but caller will likely skip.
        return True

    t_star = -float(np.dot(r1, d)) / denom
    t_star = float(np.clip(t_star, 0.0, 1.0))
    p = r1 + t_star * d

    min_dist = float(np.linalg.norm(p))
    return min_dist > (R_earth_m + margin_m)
