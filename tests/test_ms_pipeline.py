import numpy as np
from geowave.ms import diag
from geowave.ms.detect import DetectConfig, Pick, detect_station
from geowave.ms.geometry import Bounds3, Station3
from geowave.ms.locate import locate
from geowave.ms.pipeline import Pipeline, PipelineConfig
from geowave.ms.synth import make_events, make_stations, ricker, simulate
from geowave.ms.tt import HomogeneousTT

TT = HomogeneousTT(4500.0, 2600.0)
BOUNDS = Bounds3(-2000, 2000, -2000, 2000, 200, 3000)

def synth_picks(stations, ev, noise_s=0.005, seed=1):
    rng = np.random.default_rng(seed)
    picks = []
    for s in stations:
        r = float(np.sqrt((s.x_m - ev['x_m']) ** 2 + (s.y_m - ev['y_m']) ** 2
                          + (s.z_m - ev['z_m']) ** 2))
        for ph, v in (('P', TT.vp_mps), ('S', TT.vs_mps)):
            picks.append(Pick(s.name, ev['t0_s'] + r / v + float(rng.normal(0, noise_s)),
                              ph, 20.0, 0.005))
    return picks

def test_detect_p_and_s():
    fs = 500.0
    n = int(10 * fs)
    rng = np.random.default_rng(3)
    z, nn, e = (rng.normal(0, 1, n) for _ in range(3))
    tp, ts = 2.0, 2.5
    for arr, t0, amp in ((z, tp, 30.0), (nn, ts, 40.0), (e, ts, 30.0)):
        i0 = int(t0 * fs)
        w = np.arange(-75, 76)
        arr[i0 - 75:i0 + 76] += amp * ricker(30.0, w / fs)
    picks = detect_station(z, nn, e, fs, 'ST', DetectConfig())
    assert [p.phase for p in picks] == ['P', 'S']
    assert abs(picks[0].time_s - tp) < 0.05
    assert abs(picks[1].time_s - ts) < 0.05

def test_locate_recovers_source():
    stations = make_stations()
    ev = dict(t0_s=3.0, x_m=500.0, y_m=300.0, z_m=2000.0)
    out = locate(synth_picks(stations, ev), stations, TT, BOUNDS, (0.0, 0.0, 1500.0), 2.8)
    assert out.ok
    err = np.sqrt((out.x_m - 500) ** 2 + (out.y_m - 300) ** 2 + (out.z_m - 2000) ** 2)
    assert err < 50
    assert abs(out.t0_s - 3.0) < 0.02
    assert out.sig_z_m < 200

def test_out_of_model_kept_as_diagnostic():
    stations = make_stations()
    ev = dict(t0_s=3.0, x_m=300.0, y_m=-400.0, z_m=3800.0)
    out = locate(synth_picks(stations, ev), stations, TT, BOUNDS, (300.0, -400.0, 2900.0), 3.2)
    assert not out.ok
    assert diag.OUT_OF_MODEL in out.reasons
    assert out.estimate()['z_m'] > BOUNDS.z_max

def test_too_few_stations():
    stations = make_stations()[:3]
    ev = dict(t0_s=3.0, x_m=500.0, y_m=300.0, z_m=2000.0)
    out = locate(synth_picks(stations, ev), stations, TT, BOUNDS, (0.0, 0.0, 1500.0), 2.8)
    assert not out.ok
    assert diag.TOO_FEW_STATIONS in out.reasons

def test_phase_conflict():
    stations = make_stations()
    ev = dict(t0_s=3.0, x_m=500.0, y_m=300.0, z_m=2000.0)
    picks = synth_picks(stations, ev)
    rng = np.random.default_rng(11)
    for i in range(14):
        p = picks[i % len(picks)]
        picks.append(Pick(p.station, ev['t0_s'] + float(rng.uniform(4.5, 6.0)), p.phase, 20.0, 0.005))
    out = locate(picks, stations, TT, BOUNDS, (0.0, 0.0, 1500.0), 2.8)
    assert not out.ok
    assert diag.PHASE_CONFLICT in out.reasons

def test_unstable_when_array_degenerate():
    stations = [Station3(f'L{k}', x, 0.0, 0.0) for k, x in enumerate((-1200, -600, 0, 600, 1200))]
    ev = dict(t0_s=3.0, x_m=100.0, y_m=10.0, z_m=2000.0)
    out = locate(synth_picks(stations, ev), stations, TT, BOUNDS, (0.0, 0.0, 1500.0), 2.8)
    assert not out.ok
    assert diag.UNSTABLE in out.reasons

def test_end_to_end():
    stations = make_stations()
    truth = make_events()
    traces = simulate(stations, truth)
    rep = Pipeline(stations, TT, BOUNDS, PipelineConfig()).run(traces, fs_hz=500.0)
    assert len(rep.events) >= 2
    e1 = min(rep.events, key=lambda e: abs(e.t0_s - truth[0]['t0_s']))
    err = np.sqrt((e1.x_m - 600) ** 2 + (e1.y_m - 400) ** 2 + (e1.z_m - 1800) ** 2)
    assert err < 150
    assert abs(e1.t0_s - 5.0) < 0.1
    reasons = {r for f in rep.failures for r in f.reasons}
    assert diag.OUT_OF_MODEL in reasons
    assert diag.TOO_FEW_STATIONS in reasons
