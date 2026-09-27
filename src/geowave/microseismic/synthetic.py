from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.microseismic.types import Phase, Station
from geowave.microseismic.velocity import VelocityModel
from geowave.model.wavelet import ricker

@dataclass(frozen=True)
class SyntheticEvent:
    t0_s: float
    x: float
    y: float
    z: float
    magnitude: float = 1.0

def surface_network(n: int = 8, radius_m: float = 1200.0,
                    center: tuple[float, float] = (0.0, 0.0)) -> dict[str, Station]:
    out = {}
    for i in range(n):
        a = 2 * np.pi * i / n
        out[f'S{i:02d}'] = Station(f'S{i:02d}', center[0] + radius_m * np.cos(a),
                                   center[1] + radius_m * np.sin(a), 0.0)
    return out

def add_monitor_well(stations: dict[str, Station], x: float, y: float,
                     depths=(300.0, 600.0, 900.0, 1200.0)) -> dict[str, Station]:
    for k, z in enumerate(depths):
        stations[f'W{k:02d}'] = Station(f'W{k:02d}', x, y, float(z))
    return stations

def render_streams(stations: dict[str, Station], model: VelocityModel,
                   events: list[SyntheticEvent], dt_s: float, duration_s: float,
                   noise_std: float = 0.01, seed: int = 0,
                   n_glitches: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(round(duration_s / dt_s))
    full = ricker(30.0, dt_s).normalized().samples
    wave = full[len(full) // 2:]  # causal, sharp onset at the arrival time
    streams: dict[str, np.ndarray] = {}
    for name, sta in stations.items():
        data = rng.normal(0.0, noise_std, (3, n))
        for ev in events:
            src = (ev.x, ev.y, ev.z)
            d = float(np.linalg.norm(np.subtract(src, sta.xyz)))
            amp = ev.magnitude * 1500.0 / max(d, 100.0)
            for phase, comp, scale in ((Phase.P, 2, 1.0), (Phase.S, 0, 1.6), (Phase.S, 1, 1.4)):
                t = ev.t0_s + model.travel_time(phase, src, sta.xyz)
                i = int(round(t / dt_s))
                lo, hi = i, min(n, i + len(wave))
                if hi <= lo:
                    continue
                data[comp, lo:hi] += amp * scale * wave[:hi - lo]
        for _ in range(n_glitches):
            g = int(rng.integers(n // 4, 3 * n // 4))
            c = int(rng.integers(0, 3))
            data[c, g:g + 3] += 0.4 * rng.choice([-1.0, 1.0])
        streams[name] = data
    return streams
