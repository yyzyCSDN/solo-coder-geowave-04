import numpy as np
import pytest
from geowave.microseismic.types import LocateStatus, Phase, Pick
from geowave.microseismic.velocity import HomogeneousModel, LayeredModel1D
from geowave.microseismic.detect import DetectorConfig, detect_picks
from geowave.microseismic.locate import LocatorConfig, locate_event
from geowave.microseismic.pipeline import MonitoringPipeline
from geowave.microseismic.associate import AssociatorConfig
from geowave.microseismic.synthetic import (SyntheticEvent, add_monitor_well,
                                          render_streams, surface_network)
from geowave.model.wavelet import ricker

BOUNDS = (-1500.0, 1500.0, -1500.0, 1500.0, 200.0, 2500.0)

@pytest.fixture
def stations():
    return add_monitor_well(surface_network(8, 1200.0), 300.0, 200.0)

@pytest.fixture
def model():
    return LayeredModel1D(np.array([0.0, 800.0, 1800.0]),
                          np.array([3600.0, 4300.0, 5000.0]),
                          np.array([2100.0, 2500.0, 2900.0]))

def perfect_picks(stations, model, ev, noise_s=0.003, seed=7):
    rng = np.random.default_rng(seed)
    return [Pick(n, ev.t0_s + model.travel_time(ph, (ev.x, ev.y, ev.z), s.xyz)
                 + rng.normal(0, noise_s), ph, 10.0)
            for n, s in stations.items() for ph in (Phase.P, Phase.S)]

def test_layered_matches_homogeneous_when_uniform():
    homo = HomogeneousModel(4000.0, 2300.0)
    lay = LayeredModel1D(np.array([0.0, 500.0, 1500.0]),
                         np.array([4000.0] * 3), np.array([2300.0] * 3))
    for src in ((100, 200, 800), (500, -300, 2000), (0, 0, 1000), (50, 50, 499)):
        for sta in ((0, 0, 0), (1000, 500, 0), (-200, 100, 700)):
            for ph in (Phase.P, Phase.S):
                assert abs(homo.travel_time(ph, src, sta)
                           - lay.travel_time(ph, src, sta)) < 1e-6

def test_layered_monotonic_in_offset(model):
    sta = (0.0, 0.0, 0.0)
    t = [model.travel_time(Phase.P, (x, 0.0, 1500.0), sta) for x in (100, 500, 1000, 2000)]
    assert all(b > a for a, b in zip(t, t[1:]))

def test_detector_finds_p_and_s():
    dt = 0.002
    n = 4000
    rng = np.random.default_rng(0)
    full = ricker(30.0, dt).normalized().samples
    wave = full[len(full) // 2:]
    x = rng.normal(0, 0.01, (3, n))
    for i0, comp, amp in ((1000, 2, 1.0), (1900, 0, 1.5)):
        x[comp, i0:i0 + len(wave)] += amp * wave
    picks = detect_picks(x, dt, 'S00')
    assert len(picks) == 2
    assert abs(picks[0].time_s - 1000 * dt) < 0.02
    assert abs(picks[1].time_s - 1900 * dt) < 0.02
    assert all(p.snr > 10 for p in picks)

def test_detector_quiet_on_noise():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, (3, 5000))
    assert detect_picks(x, 0.002, 'S00') == []

def test_locate_perfect_picks(stations, model):
    ev = SyntheticEvent(3.0, 200.0, -150.0, 1400.0)
    cfg = LocatorConfig(BOUNDS, grid_n=(13, 13, 13))
    r = locate_event(perfect_picks(stations, model, ev), stations, model, cfg)
    assert r.status is LocateStatus.LOCATED
    h = r.hypocenter
    assert abs(h.x - ev.x) < 10 and abs(h.y - ev.y) < 10 and abs(h.z - ev.z) < 15
    assert abs(h.t0_s - ev.t0_s) < 0.01
    u = r.uncertainty
    assert abs(h.x - ev.x) < 4 * u.sigma_x_m
    assert abs(h.y - ev.y) < 4 * u.sigma_y_m
    assert abs(h.z - ev.z) < 4 * u.sigma_z_m
    assert u.n_stations == len(stations) and u.n_phases == 2 * len(stations)
    assert 0 < u.azimuthal_gap_deg < 180

def test_insufficient_stations(stations, model):
    ev = SyntheticEvent(3.0, 200.0, -150.0, 1400.0)
    picks = perfect_picks(stations, model, ev)
    few = [p for p in picks if p.station in ('S00', 'S01', 'S02')]
    r = locate_event(few, stations, model, LocatorConfig(BOUNDS, grid_n=(13, 13, 13)))
    assert r.status is LocateStatus.INSUFFICIENT_DATA
    assert r.hypocenter is None
    assert any(d.code == 'insufficient_data' for d in r.diagnostics)
    assert r.diagnostics[0].context['n_stations'] == 3

def test_phase_conflict_single_bad_pick(stations, model):
    ev = SyntheticEvent(3.0, 200.0, -150.0, 1400.0)
    picks = perfect_picks(stations, model, ev)
    bad = next(p for p in picks if p.station == 'S03' and p.phase is Phase.P)
    picks[picks.index(bad)] = Pick(bad.station, bad.time_s + 0.3, Phase.P, bad.snr)
    r = locate_event(picks, stations, model, LocatorConfig(BOUNDS, grid_n=(13, 13, 13)))
    assert r.status in (LocateStatus.LOCATED, LocateStatus.LOCATED_DEGRADED)
    assert len(r.rejected) == 1 and r.rejected[0].station == 'S03'
    assert any(d.code == 'residual_outlier' for d in r.diagnostics)
    assert abs(r.hypocenter.x - ev.x) < 30 and abs(r.hypocenter.z - ev.z) < 40

def test_unresolvable_conflict_reports_failure(stations, model):
    ev = SyntheticEvent(3.0, 200.0, -150.0, 1400.0)
    picks = perfect_picks(stations, model, ev)
    rng = np.random.default_rng(11)
    for k, p in enumerate(picks):
        if p.station in ('S00', 'S01', 'S02', 'S03', 'S04', 'S05'):
            picks[k] = Pick(p.station, p.time_s + rng.uniform(0.1, 0.4), p.phase, p.snr)
    r = locate_event(picks, stations, model,
                     LocatorConfig(BOUNDS, grid_n=(13, 13, 13), max_reject=4))
    assert r.status is LocateStatus.ASSOCIATION_FAILED
    assert r.hypocenter is None
    assert any(d.code == 'unresolved_residuals' for d in r.diagnostics)

def test_out_of_model_event_not_forced(stations, model):
    ev = SyntheticEvent(3.0, 3200.0, -150.0, 1400.0)  # outside x bounds
    picks = perfect_picks(stations, model, ev)
    r = locate_event(picks, stations, model, LocatorConfig(BOUNDS, grid_n=(13, 13, 13)))
    assert r.status is LocateStatus.OUT_OF_MODEL
    assert r.hypocenter is None
    d = next(d for d in r.diagnostics if d.code == 'out_of_model')
    assert 'clamped_estimate' in d.context

def test_end_to_end(stations, model):
    events = [SyntheticEvent(3.0, 200.0, -150.0, 1400.0),
              SyntheticEvent(7.0, -400.0, 300.0, 1600.0),
              SyntheticEvent(11.0, 100.0, 500.0, 1200.0)]
    streams = render_streams(stations, model, events, 0.002, 15.0,
                             noise_std=0.01, seed=1, n_glitches=2)
    pipe = MonitoringPipeline(stations, model,
                              LocatorConfig(BOUNDS, grid_n=(13, 13, 13)),
                              DetectorConfig(),
                              AssociatorConfig(grid_n=(13, 13, 13)))
    results = pipe.run(streams, 0.002)
    located = [r for r in results if r.located]
    assert len(located) == 3
    for r in located:
        h = r.hypocenter
        err = min(((h.x - e.x) ** 2 + (h.y - e.y) ** 2 + (h.z - e.z) ** 2) ** 0.5
                  for e in events)
        dt = min(abs(h.t0_s - e.t0_s) for e in events)
        assert err < 50, f'location error {err:.0f} m'
        assert dt < 0.05, f'origin time error {dt:.3f} s'
        assert r.uncertainty.rms_s < 0.02
    # every failed result must carry diagnostics instead of a forced coordinate
    for r in results:
        if not r.located:
            assert r.hypocenter is None and r.diagnostics
