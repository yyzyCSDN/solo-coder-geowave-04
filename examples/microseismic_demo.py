"""End-to-end microseismic monitoring demo on synthetic frac-stage data.

Scenario: 8 surface stations + 4 geophones in a monitor well, 1D layered
velocity model, three frac events plus random noise glitches. The pipeline
detects picks on the continuous streams, associates phases across the
network and locates each event, reporting origin time, hypocenter,
uncertainty and - for clusters that cannot be reconciled - failure
diagnostics instead of a forced coordinate.

Run:  python3 examples/microseismic_demo.py
"""
from __future__ import annotations
import numpy as np
from geowave.microseismic.locate import LocatorConfig
from geowave.microseismic.pipeline import MonitoringPipeline
from geowave.microseismic.synthetic import (SyntheticEvent, add_monitor_well,
                                          render_streams, surface_network)
from geowave.microseismic.velocity import LayeredModel1D

def main():
    stations = add_monitor_well(surface_network(n=8, radius_m=1200.0), 300.0, 200.0)
    model = LayeredModel1D(np.array([0.0, 800.0, 1800.0]),
                           np.array([3600.0, 4300.0, 5000.0]),
                           np.array([2100.0, 2500.0, 2900.0]))
    bounds = (-1500.0, 1500.0, -1500.0, 1500.0, 200.0, 2500.0)
    truth = [SyntheticEvent(3.0, 200.0, -150.0, 1400.0),
             SyntheticEvent(7.0, -400.0, 300.0, 1600.0),
             SyntheticEvent(11.0, 100.0, 500.0, 1200.0)]
    streams = render_streams(stations, model, truth, dt_s=0.002, duration_s=15.0,
                             noise_std=0.01, seed=1, n_glitches=2)
    pipe = MonitoringPipeline(stations, model, LocatorConfig(bounds))
    results = pipe.run(streams, dt_s=0.002)

    print(f'{len(results)} event clusters from {len(stations)} stations\n')
    hdr = f'{"status":<18} {"t0 [s]":>8} {"x [m]":>8} {"y [m]":>8} {"z [m]":>8}' \
          f' {"sig_h":>6} {"sig_z":>6} {"rms":>6} {"ph":>3} {"gap":>4}'
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        if r.hypocenter:
            h, u = r.hypocenter, r.uncertainty
            sig_h = max(u.sigma_x_m, u.sigma_y_m)
            print(f'{r.status.value:<18} {h.t0_s:8.3f} {h.x:8.1f} {h.y:8.1f} {h.z:8.1f}'
                  f' {sig_h:6.1f} {u.sigma_z_m:6.1f} {u.rms_s * 1e3:5.1f}m'
                  f' {u.n_phases:3d} {u.azimuthal_gap_deg:4.0f}')
        else:
            print(f'{r.status.value:<18} {"--":>8} {"--":>8} {"--":>8} {"--":>8}')
        for d in r.diagnostics:
            print(f'    [{d.code}] {d.message}')
    print('\ntruth:')
    for e in truth:
        print(f'    t0={e.t0_s:6.2f}  x={e.x:7.1f}  y={e.y:7.1f}  z={e.z:7.1f}')

if __name__ == '__main__':
    main()
