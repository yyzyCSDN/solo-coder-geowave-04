from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from geowave.ms import diag
from geowave.ms.geometry import Bounds3

@dataclass(frozen=True)
class AssocConfig:
    dx_m: float = 100.0
    dy_m: float = 100.0
    dz_m: float = 100.0
    tol_s: float = 0.08       # 发震时刻聚类半窗宽
    min_stations: int = 4
    min_phases: int = 6
    max_events: int = 20

@dataclass
class Candidate:
    pick_idx: list[int]
    x_m: float
    y_m: float
    z_m: float
    t0_s: float
    spread_s: float
    n_stations: int
    conflicts: list[str] = field(default_factory=list)
    reason: str = ''

    @property
    def n_phases(self) -> int:
        return len(self.pick_idx)

def _grid(bounds: Bounds3, cfg: AssocConfig):
    xs = np.arange(bounds.x_min, bounds.x_max + 0.5 * cfg.dx_m, cfg.dx_m)
    ys = np.arange(bounds.y_min, bounds.y_max + 0.5 * cfg.dy_m, cfg.dy_m)
    zs = np.arange(bounds.z_min, bounds.z_max + 0.5 * cfg.dz_m, cfg.dz_m)
    x, y, z = np.meshgrid(xs, ys, zs, indexing='ij')
    return x.ravel(), y.ravel(), z.ravel()

def _window_best(s_sorted: np.ndarray, tol_s: float, chunk: int = 4096):
    """每行(网格点)内, 以各点为中心的 ±tol 窗覆盖的最大点数及其位置。"""
    n = s_sorted.shape[0]
    best_cnt = np.zeros(n, int)
    best_pos = np.zeros(n, int)
    for a in range(0, n, chunk):
        sc = s_sorted[a:a + chunk]
        d = (np.abs(sc[:, :, None] - sc[:, None, :]) <= tol_s).sum(axis=2)
        best_cnt[a:a + chunk] = d.max(axis=1)
        best_pos[a:a + chunk] = d.argmax(axis=1)
    return best_cnt, best_pos

def associate(picks, stations, tt, bounds: Bounds3, cfg: AssocConfig = AssocConfig()):
    """网格投票关联: 每个网格点上 t0 = t_pick - T(grid, sta, phase) 最聚集的一簇即候选事件;
    票数并列时取簇内离散度最小者。返回 (accepted, weak, leftover_idx)。"""
    pos = {s.name: i for i, s in enumerate(stations)}
    gx, gy, gz = _grid(bounds, cfg)
    cube = np.empty((gx.size, len(stations), 2))
    for si, s in enumerate(stations):
        cube[:, si, 0] = tt.travel_time(gx, gy, gz, s, 'P')
        cube[:, si, 1] = tt.travel_time(gx, gy, gz, s, 'S')
    p_sta = np.array([pos[p.station] for p in picks])
    p_ph = np.array([0 if p.phase == 'P' else 1 for p in picks])
    p_t = np.array([p.time_s for p in picks])
    remaining = np.arange(len(picks))
    accepted: list[Candidate] = []
    weak: Candidate | None = None
    for _ in range(cfg.max_events):
        if len(remaining) < cfg.min_phases:
            break
        t0 = p_t[remaining][None, :] - cube[:, p_sta[remaining], p_ph[remaining]]
        srt = np.sort(t0, axis=1)
        counts, bpos = _window_best(srt, cfg.tol_s)
        maxc = int(counts.max())
        tie = np.where(counts == maxc)[0]
        g, best_std = -1, np.inf
        for gt in tie[:512]:
            row_t = t0[gt]
            mem_t = np.abs(row_t - srt[gt, bpos[gt]]) <= cfg.tol_s
            std_t = float(np.std(row_t[mem_t])) if mem_t.sum() > 1 else 0.0
            if std_t < best_std:
                best_std, g = std_t, int(gt)
        row = t0[g]
        center = float(srt[g, bpos[g]])
        for _ in range(3):
            mem = np.abs(row - center) <= cfg.tol_s
            if not mem.any():
                break
            new = float(np.median(row[mem]))
            if new == center:
                break
            center = new
        best: dict[tuple[str, str], tuple[float, int]] = {}
        conflicts: list[str] = []
        for k in np.where(mem)[0]:
            p = picks[remaining[k]]
            err = abs(float(row[k]) - center)
            key = (p.station, p.phase)
            if key in best:
                conflicts.append(f'{p.station} {p.phase} 重复到时,已保留最接近聚类中心者')
                if err < best[key][0]:
                    best[key] = (err, int(k))
            else:
                best[key] = (err, int(k))
        kept = [v[1] for v in best.values()]
        n_sta = len({picks[remaining[k]].station for k in kept})
        cand = Candidate(
            pick_idx=sorted(int(remaining[k]) for k in kept),
            x_m=float(gx[g]), y_m=float(gy[g]), z_m=float(gz[g]),
            t0_s=center,
            spread_s=float(np.std(row[kept])) if len(kept) > 1 else 0.0,
            n_stations=n_sta, conflicts=conflicts)
        if len(kept) < cfg.min_phases or n_sta < cfg.min_stations:
            cand.reason = diag.TOO_FEW_STATIONS if n_sta < cfg.min_stations else diag.TOO_FEW_PHASES
            weak = cand
            break
        accepted.append(cand)
        drop = set(cand.pick_idx)
        remaining = np.array([i for i in remaining if i not in drop])
    return accepted, weak, remaining.tolist()
