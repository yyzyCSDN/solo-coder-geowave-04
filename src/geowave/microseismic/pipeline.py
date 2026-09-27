from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from geowave.microseismic.associate import AssociatorConfig, associate
from geowave.microseismic.detect import DetectorConfig, detect_picks
from geowave.microseismic.locate import LocatorConfig, locate_event
from geowave.microseismic.types import EventResult, LocateStatus, Pick, Station
from geowave.microseismic.velocity import VelocityModel

@dataclass
class MonitoringPipeline:
    stations: dict[str, Station]
    model: VelocityModel
    locator: LocatorConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    associator: AssociatorConfig = field(default_factory=AssociatorConfig)

    def detect(self, streams: dict[str, np.ndarray], dt_s: float,
               t0_s: float = 0.0) -> list[Pick]:
        picks: list[Pick] = []
        for name, trace in streams.items():
            if name not in self.stations:
                continue
            picks.extend(detect_picks(trace, dt_s, name, t0_s, self.detector))
        return sorted(picks, key=lambda p: p.time_s)

    def run_picks(self, picks: list[Pick]) -> list[EventResult]:
        results: list[EventResult] = []
        for ev in associate(picks, self.stations, self.model, self.associator, self.locator):
            if ev.fail_status is not None:
                results.append(EventResult(ev.fail_status, None, None, ev.picks, [],
                                           ev.diagnostics))
                continue
            results.append(locate_event(ev.picks, self.stations, self.model, self.locator,
                                        ev.diagnostics))
        results.sort(key=lambda r: r.hypocenter.t0_s if r.hypocenter else
                     (r.picks[0].time_s if r.picks else 0.0))
        return results

    def run(self, streams: dict[str, np.ndarray], dt_s: float,
            t0_s: float = 0.0) -> list[EventResult]:
        return self.run_picks(self.detect(streams, dt_s, t0_s))
