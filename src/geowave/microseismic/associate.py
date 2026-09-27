from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from geowave.microseismic.locate import LocatorConfig
from geowave.microseismic.types import (Diagnostic, LocateStatus, Phase, Pick, Station)
from geowave.microseismic.velocity import VelocityModel

@dataclass(frozen=True)
class AssociatorConfig:
    min_stations: int = 4
    t0_window_s: float = 0.15            # full width of the origin-time agreement window
    s_window_slack_s: float = 0.20
    grid_n: tuple[int, int, int] = (21, 21, 21)  # voting grid

@dataclass
class AssociatedEvent:
    picks: list[Pick]
    diagnostics: list[Diagnostic] = field(default_factory=list)
    fail_status: LocateStatus | None = None

def _nodes(bounds: tuple, grid_n: tuple[int, int, int]) -> np.ndarray:
    axes = [np.linspace(bounds[2 * k], bounds[2 * k + 1], grid_n[k]) for k in range(3)]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

def _best_window(t0_imp: np.ndarray, width_s: float) -> tuple[int, float, int]:
    """Node whose implied origin times have the densest window of width_s."""
    s = np.sort(t0_imp, axis=1)
    best_row, best_lo, best_cnt = -1, 0.0, 0
    step = 256
    for a in range(0, len(s), step):
        chunk = s[a:a + step]
        lo = chunk[:, :, None]
        hi = chunk[:, None, :]
        cnt = ((hi >= lo) & (hi < lo + width_s)).sum(axis=2)  # (rows, starts)
        r, c = np.unravel_index(int(np.argmax(cnt)), cnt.shape)
        if int(cnt[r, c]) > best_cnt:
            best_cnt = int(cnt[r, c])
            best_row = a + r
            best_lo = float(chunk[r, c])
    return best_row, best_lo, best_cnt

def associate(picks: list[Pick], stations: dict[str, Station], model: VelocityModel,
              cfg: AssociatorConfig, loc_cfg: LocatorConfig) -> list[AssociatedEvent]:
    events: list[AssociatedEvent] = []
    remaining = sorted(picks, key=lambda p: p.time_s)
    nodes = _nodes(loc_cfg.bounds, cfg.grid_n)
    while len(remaining) >= cfg.min_stations:
        sta = np.array([stations[p.station].xyz for p in remaining])
        t = np.array([p.time_s for p in remaining])
        tt_p = model.travel_time_table(Phase.P, nodes, sta)
        t0_imp = t[None, :] - tt_p  # (n_nodes, n_picks)
        row, lo, cnt = _best_window(t0_imp, cfg.t0_window_s)
        if cnt < cfg.min_stations:
            break
        in_win = np.flatnonzero((t0_imp[row] >= lo) & (t0_imp[row] < lo + cfg.t0_window_s))
        center = float(np.median(t0_imp[row, in_win]))  # robust consensus origin time
        half = 0.5 * cfg.t0_window_s
        by_sta: dict[str, tuple[float, int]] = {}
        for k, p in enumerate(remaining):
            d = abs(t0_imp[row, k] - center)
            if d <= half and (p.station not in by_sta or d < by_sta[p.station][0]):
                by_sta[p.station] = (d, k)
        if len(by_sta) < cfg.min_stations:
            break
        members = sorted(k for _, k in by_sta.values())
        node = nodes[row]
        t0 = float(np.mean([t0_imp[row, k] for k in members]))
        member_set = set(members)
        p_picks = [remaining[k].with_phase(Phase.P) for k in members]
        s_picks, s_used, diag = _attach_s(remaining, member_set, p_picks, stations,
                                          model, node, t0, cfg)
        events.append(AssociatedEvent(p_picks + s_picks, diag))
        drop = member_set | s_used
        remaining = [p for k, p in enumerate(remaining) if k not in drop]
    return events

def _attach_s(all_picks: list[Pick], p_idx: set[int], p_picks: list[Pick],
              stations: dict[str, Station], model: VelocityModel, node: np.ndarray,
              t0: float, cfg: AssociatorConfig):
    diag: list[Diagnostic] = []
    p_by_station = {p.station: p for p in p_picks}
    sta_all = np.array([s.xyz for s in stations.values()])
    ts_pred = model.travel_time_table(Phase.S, node[None, :], sta_all)[0]
    ts_of = {name: float(ts_pred[k]) for k, name in enumerate(stations)}
    s_picks: list[Pick] = []
    s_used: set[int] = set()
    for k, q in enumerate(all_picks):
        if k in p_idx:
            continue
        t_pred = t0 + ts_of[q.station]
        window = cfg.s_window_slack_s + 0.1 * ts_of[q.station]
        if abs(q.time_s - t_pred) > window:
            continue
        if q.station in p_by_station and q.time_s <= p_by_station[q.station].time_s:
            continue
        s_used.add(k)  # candidate; final choice below per station
    chosen: dict[str, tuple[float, int]] = {}
    for k in s_used:
        q = all_picks[k]
        d = abs(q.time_s - (t0 + ts_of[q.station]))
        if q.station not in chosen or d < chosen[q.station][0]:
            chosen[q.station] = (d, k)
    for name, (d, k) in chosen.items():
        q = all_picks[k]
        s_picks.append(q.with_phase(Phase.S))
        alts = [all_picks[j].time_s for j in s_used
                if all_picks[j].station == name and j != k]
        if alts:
            diag.append(Diagnostic(
                'ambiguous_s_pick',
                f'{len(alts) + 1} candidate S picks at {name}, took {q.time_s:.4f}s '
                '(nearest to prediction)',
                {'station': name, 'chosen_s': q.time_s, 'alternates_s': alts}))
    return s_picks, set(k for _, k in chosen.values()), diag
