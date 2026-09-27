from __future__ import annotations
import numpy as np
from geowave.ms.geometry import Station3

def ricker(f_hz: float, t_s) -> np.ndarray:
    a = np.pi * f_hz * np.asarray(t_s, float)
    return (1.0 - 2.0 * a * a) * np.exp(-a * a)

def make_stations() -> list[Station3]:
    sts = [Station3('S00', 0.0, 0.0, 0.0)]
    for k, ang in enumerate((30, 150, 270)):
        a = np.radians(ang)
        sts.append(Station3(f'S1{k}', 600 * np.cos(a), 600 * np.sin(a), 0.0))
    for k in range(6):
        a = np.radians(60 * k)
        sts.append(Station3(f'S2{k}', 1200 * np.cos(a), 1200 * np.sin(a), 0.0))
    for d in (400, 700, 1000):
        sts.append(Station3(f'BH{d}', 150.0, -250.0, float(d)))
    return sts

def make_events() -> list[dict]:
    return [
        dict(name='E1', t0_s=5.0, x_m=600.0, y_m=400.0, z_m=1800.0, amp=1.0),
        dict(name='E2', t0_s=15.0, x_m=-700.0, y_m=900.0, z_m=2100.0, amp=0.9),
        dict(name='E3', t0_s=25.0, x_m=300.0, y_m=-400.0, z_m=3800.0, amp=1.0),   # 模型底界之外
        dict(name='E4', t0_s=35.0, x_m=200.0, y_m=-300.0, z_m=2200.0, amp=0.12, s_ratio=1.0),  # 弱事件
    ]

def simulate(stations, events, fs_hz: float = 500.0, dur_s: float = 45.0, seed: int = 7,
             noise: float = 1.0, f0_hz: float = 30.0, vp_mps: float = 4500.0, vs_mps: float = 2600.0):
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * fs_hz))
    traces = {s.name: {c: rng.normal(0.0, noise, n) for c in 'ZNE'} for s in stations}

    def add_wave(name, t_arr, amp, vec, f_hz):
        i0 = int(round(t_arr * fs_hz))
        w = int(0.25 * fs_hz)
        a, b = max(0, i0 - w), min(n, i0 + w)
        if b <= a:
            return
        t = (np.arange(a, b) - i0) / fs_hz
        wav = amp * ricker(f_hz, t)
        for c, u in zip('ZNE', (vec[2], vec[1], vec[0])):
            traces[name][c][a:b] += u * wav

    for ev in events:
        sr = ev.get('s_ratio', 2.0)
        for s in stations:
            d = np.array([s.x_m - ev['x_m'], s.y_m - ev['y_m'], s.z_m - ev['z_m']])
            r = float(np.linalg.norm(d))
            ray = d / r
            amp_p = ev['amp'] * 40.0 * (1000.0 / r)
            sh = np.array([ray[1], -ray[0], 0.0])
            nrm = float(np.linalg.norm(sh))
            sh = sh / nrm if nrm > 1e-9 else np.array([1.0, 0.0, 0.0])
            sv = np.cross(sh, ray)
            pol_s = 0.8 * sh + 0.35 * sv
            pol_s /= np.linalg.norm(pol_s)
            add_wave(s.name, ev['t0_s'] + r / vp_mps, amp_p, ray, f0_hz)
            add_wave(s.name, ev['t0_s'] + r / vs_mps, amp_p * sr, pol_s, 0.75 * f0_hz)

    for t_g, name, amp_g in ((8.2, 'S20', 6.0), (19.5, 'S12', 5.0), (30.1, 'BH700', 7.0)):
        vec = rng.normal(size=3)
        add_wave(name, t_g, amp_g, vec / np.linalg.norm(vec), 25.0)

    e1 = events[0]
    dist = {s.name: float(np.sqrt((s.x_m - e1['x_m']) ** 2 + (s.y_m - e1['y_m']) ** 2
                                  + (s.z_m - e1['z_m']) ** 2)) for s in stations if s.z_m == 0.0}
    near = min(dist, key=dist.get)
    t_p = e1['t0_s'] + dist[near] / vp_mps
    add_wave(near, t_p + 0.20, 10.0, np.array([0.8, 0.6, 0.0]), 25.0)  # 水平占优干扰, 制造同台 S 冲突
    return traces
