from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares
from geowave.ms import diag

@dataclass(frozen=True)
class LocateConfig:
    min_stations: int = 4
    min_phases: int = 6
    require_s: bool = True
    outlier_nsigma: float = 3.0
    min_abs_tol_s: float = 0.03   # 离群阈值下限, 防止 MAD 过小时误剔
    max_outlier_frac: float = 0.34
    max_outlier_iters: int = 5
    max_sigma_m: float = 800.0
    max_gap_deg: float = 240.0
    bounds_margin_m: float = 1.0
    consistency_m: float = 150.0   # 多起点重解允许的最大偏差
    consistency_s: float = 0.1

@dataclass
class LocateOutcome:
    ok: bool
    x_m: float
    y_m: float
    z_m: float
    t0_s: float
    sig_x_m: float = float('nan')
    sig_y_m: float = float('nan')
    sig_z_m: float = float('nan')
    sig_t0_s: float = float('nan')
    rms_s: float = float('nan')
    gap_deg: float = float('nan')
    n_phases: int = 0
    n_stations: int = 0
    n_outliers: int = 0
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    picks: list = field(default_factory=list)

    def estimate(self) -> dict[str, float] | None:
        if not np.isfinite(self.x_m):
            return None
        return {'x_m': self.x_m, 'y_m': self.y_m, 'z_m': self.z_m, 't0_s': self.t0_s}

def _azimuth_gap_deg(x_m: float, y_m: float, sta_xy: np.ndarray) -> float:
    az = np.mod(np.arctan2(sta_xy[:, 0] - x_m, sta_xy[:, 1] - y_m), 2 * np.pi)
    az.sort()
    d = np.diff(np.concatenate([az, az[:1] + 2 * np.pi]))
    return float(np.degrees(d.max()))

def locate(picks, stations, tt, bounds, xyz0, t0_0: float, cfg: LocateConfig = LocateConfig()) -> LocateOutcome:
    reasons: list[str] = []
    warnings: list[str] = []
    sta_map = {s.name: s for s in stations}
    n_sta = len({p.station for p in picks})
    if n_sta < cfg.min_stations:
        reasons.append(diag.TOO_FEW_STATIONS)
    if len(picks) < cfg.min_phases:
        reasons.append(diag.TOO_FEW_PHASES)
    if cfg.require_s and all(p.phase != 'S' for p in picks):
        reasons.append(diag.NO_S_PHASE)
    if reasons:
        return LocateOutcome(False, float('nan'), float('nan'), float('nan'), float('nan'),
                             reasons=reasons, n_phases=len(picks), n_stations=n_sta)
    sx = np.array([sta_map[p.station].x_m for p in picks])
    sy = np.array([sta_map[p.station].y_m for p in picks])
    sz = np.array([sta_map[p.station].z_m for p in picks])
    vel = np.array([tt.velocity(p.phase) for p in picks])
    tobs = np.array([p.time_s for p in picks])
    sig = np.array([max(p.sigma_s, 1e-3) for p in picks])

    def res(theta, idx):
        x, y, z, t0 = theta
        r = np.sqrt((x - sx[idx]) ** 2 + (y - sy[idx]) ** 2 + (z - sz[idx]) ** 2)
        return (t0 + np.maximum(r, 1.0) / vel[idx] - tobs[idx]) / sig[idx]

    def jac(theta, idx):
        x, y, z, t0 = theta
        r = np.maximum(np.sqrt((x - sx[idx]) ** 2 + (y - sy[idx]) ** 2 + (z - sz[idx]) ** 2), 1.0)
        j = np.empty((len(idx), 4))
        j[:, 0] = (x - sx[idx]) / (vel[idx] * r * sig[idx])
        j[:, 1] = (y - sy[idx]) / (vel[idx] * r * sig[idx])
        j[:, 2] = (z - sz[idx]) / (vel[idx] * r * sig[idx])
        j[:, 3] = 1.0 / sig[idx]
        return j

    kept = np.arange(len(picks))
    sol = least_squares(res, [xyz0[0], xyz0[1], xyz0[2], t0_0], jac=jac,
                        args=(kept,), loss='soft_l1', f_scale=1.0)
    dropped: list[int] = []
    for _ in range(cfg.max_outlier_iters):
        r_raw = res(sol.x, kept) * sig[kept]
        med = float(np.median(r_raw))
        mad = float(np.median(np.abs(r_raw - med)))
        thr = max(cfg.outlier_nsigma * 1.4826 * mad, cfg.min_abs_tol_s)
        bad = np.where(np.abs(r_raw - med) > thr)[0]
        if bad.size == 0:
            break
        order = bad[np.argsort(-np.abs(r_raw[bad] - med))]
        n_drop = min(len(order), max(1, len(kept) // 3))
        drop = kept[order[:n_drop]]
        dropped.extend(int(i) for i in drop)
        kept = np.setdiff1d(kept, drop)
        if len(kept) < 4:
            break
        sol = least_squares(res, sol.x, jac=jac, args=(kept,), loss='soft_l1', f_scale=1.0)
    if len(dropped) > cfg.max_outlier_frac * len(picks):
        reasons.append(diag.PHASE_CONFLICT)
    if len(kept) < 4:
        reasons.append(diag.TOO_FEW_PHASES)
    x, y, z, t0 = (float(v) for v in sol.x)
    # 多起点一致性: 围绕解扰动重解, 收敛到不同解说明存在多解/局部极小
    for px, py, pz, pt in ((300.0, -300.0, 500.0, 0.15), (-300.0, 300.0, -400.0, -0.15),
                           (-300.0, -300.0, 300.0, 0.10), (300.0, 300.0, -300.0, -0.10)):
        try:
            alt = least_squares(res, [x + px, y + py, z + pz, t0 + pt], jac=jac,
                                args=(kept,), loss='soft_l1', f_scale=1.0)
            d = float(np.linalg.norm(alt.x[:3] - sol.x[:3]))
            if not np.isfinite(d) or d > cfg.consistency_m or abs(float(alt.x[3]) - t0) > cfg.consistency_s:
                reasons.append(diag.UNSTABLE)
                break
        except Exception:
            reasons.append(diag.UNSTABLE)
            break
    j = jac(sol.x, kept)
    dof = max(len(kept) - 4, 1)
    rw = res(sol.x, kept)
    s2 = float(rw @ rw) / dof
    colscale = np.linalg.norm(j, axis=0)
    colscale[colscale == 0] = 1.0
    g = j / colscale   # 列归一化, 避免 t0 列与空间列量纲悬殊导致法方程病态
    try:
        # J = G·diag(colscale) => (JᵀJ)⁻¹ = diag(1/colscale)·(GᵀG)⁻¹·diag(1/colscale)
        cov = s2 * (np.linalg.inv(g.T @ g) / np.outer(colscale, colscale))
        psig = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        psig = np.full(4, np.inf)
    sig_x, sig_y, sig_z, sig_t0 = (float(v) for v in psig)
    if not bounds.contains(x, y, z, cfg.bounds_margin_m):
        reasons.append(diag.OUT_OF_MODEL)
    if not np.all(np.isfinite(psig)) or max(sig_x, sig_y, sig_z) > cfg.max_sigma_m:
        reasons.append(diag.UNSTABLE)
    kept_picks = [picks[i] for i in kept]
    gap = _azimuth_gap_deg(x, y, np.column_stack([sx[kept], sy[kept]]))
    if gap > cfg.max_gap_deg:
        warnings.append(diag.LARGE_GAP)
    r_raw = res(sol.x, kept) * sig[kept]
    rms = float(np.sqrt(np.mean(r_raw ** 2)))
    return LocateOutcome(not reasons, x, y, z, t0, sig_x, sig_y, sig_z, sig_t0, rms, gap,
                         len(kept), len({p.station for p in kept_picks}), len(dropped),
                         reasons, warnings, kept_picks)
