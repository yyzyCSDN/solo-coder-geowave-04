from __future__ import annotations
import time
import numpy as np
from geowave.ms.geometry import Bounds3
from geowave.ms.tt import HomogeneousTT
from geowave.ms.synth import make_stations, make_events, simulate
from geowave.ms.pipeline import Pipeline, PipelineConfig

def main():
    stations = make_stations()
    truth = make_events()
    tt = HomogeneousTT(4500.0, 2600.0)
    bounds = Bounds3(-2000, 2000, -2000, 2000, 200, 3000)
    t_start = time.time()
    traces = simulate(stations, truth)
    rep = Pipeline(stations, tt, bounds, PipelineConfig()).run(traces, fs_hz=500.0)
    print(rep.summary())
    print(f'\n流程耗时 {time.time() - t_start:.1f} s')
    print('\n--- 与真值对比(仅演示验证用) ---')
    for ev in rep.events:
        c = min(truth, key=lambda e: abs(e['t0_s'] - ev.t0_s))
        err = float(np.sqrt((ev.x_m - c['x_m']) ** 2 + (ev.y_m - c['y_m']) ** 2
                            + (ev.z_m - c['z_m']) ** 2))
        print(f"t0={ev.t0_s:6.3f}s ↔ {c['name']}: 位置误差 {err:.1f} m, "
              f"发震时刻误差 {ev.t0_s - c['t0_s']:+.3f} s")

if __name__ == '__main__':
    main()
