import numpy as np

from sim.config import EarthConstants
from sim.geometry import elevation_deg, isl_line_of_sight_clear, latlon_to_ecef_m


def test_latlon_to_ecef_axes() -> None:
    earth = EarthConstants()

    p = latlon_to_ecef_m(0.0, 0.0, 0.0, R_earth_m=earth.R_earth_m)
    assert np.allclose(p, [earth.R_earth_m, 0.0, 0.0], atol=1e-6)

    p = latlon_to_ecef_m(0.0, 90.0, 0.0, R_earth_m=earth.R_earth_m)
    assert abs(p[0]) < 1e-6
    assert np.allclose(p[1], earth.R_earth_m, atol=1e-6)
    assert abs(p[2]) < 1e-6

    p = latlon_to_ecef_m(90.0, 0.0, 0.0, R_earth_m=earth.R_earth_m)
    assert abs(p[0]) < 1e-6
    assert abs(p[1]) < 1e-6
    assert np.allclose(p[2], earth.R_earth_m, atol=1e-6)


def test_elevation_overhead_and_below_horizon() -> None:
    earth = EarthConstants()
    gs = latlon_to_ecef_m(0.0, 0.0, 0.0, R_earth_m=earth.R_earth_m)

    sat_overhead = np.array([earth.R_earth_m + 550_000.0, 0.0, 0.0])
    elev = elevation_deg(gs, sat_overhead)
    assert elev > 89.999

    sat_far = np.array([0.0, earth.R_earth_m + 550_000.0, 0.0])
    elev2 = elevation_deg(gs, sat_far)
    assert elev2 < 0.0


def test_isl_occlusion() -> None:
    earth = EarthConstants()
    h = 550_000.0

    r1 = np.array([earth.R_earth_m + h, 0.0, 0.0])
    r2 = np.array([-(earth.R_earth_m + h), 0.0, 0.0])
    assert isl_line_of_sight_clear(r1, r2, R_earth_m=earth.R_earth_m) is False

    r3 = np.array([earth.R_earth_m + h, 1000.0, 0.0])
    assert isl_line_of_sight_clear(r1, r3, R_earth_m=earth.R_earth_m) is True
