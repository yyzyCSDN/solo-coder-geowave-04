from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from geowave.microseismic.types import (Diagnostic, EventResult, Hypocenter, LocateStatus,
                                      Phase, Pick, Station, Uncertainty)
from geowave.microseismic.velocity import VelocityModel

@dataclass(frozen=True)
class LocatorConfig:
    # model volume: xmin, xmax, ymin, ymax, zmin, zmax  (m, z positive down)
    bounds: tuple[float, float, float, float, float, float]
    grid_n: tuple[int, int, int] = (21, 21, 21)
    min_stations: int = 4
    min_phases: int = 6
    residual_reject_s: float = 0.06
    max_reject: int = 4
    max_rms_s: float = 0.05  # above this after rejection, the pick set is irreconcilable
    boundary_tol_m: float = 75.0
    max_nfev: int = 200

def _arrays(picks: list[Pick], stations: dict[str, Station]):
    sta = np.array([stations[p.station].xyz for p in picks])
    t = np.array([p.time_s for p in picks])
    is_p = np.array([p.phase is Phase.P for p in picks])  # str-Enum breaks in numpy arrays
    w = np.array([max(p.weight, 1e-3) for p in picks])
    return sta, t, is_p, w

def _tt_table(model: VelocityModel, nodes: np.ndarray, sta: np.ndarray,
              is_p: np.ndarray) -> np.ndarray:
    tp = model.travel_time_table(Phase.P, nodes, sta)
    if is_p.all():
        return tp
    ts = model.travel_time_table(Phase.S, nodes, sta)
    return np.where(is_p[None, :], tp, ts)

def grid_misfit(model: VelocityModel, sta: np.ndarray, t: np.ndarray, is_p: np.ndarray,
                w: np.ndarray, cfg: LocatorConfig):
    xmin, xmax, ymin, ymax, zmin, zmax = cfg.bounds
    gx, gy, gz = (np.linspace(a, b, n) for (a, b), n in
                  zip(((xmin, xmax), (ymin, ymax), (zmin, zmax)), cfg.grid_n))
    x, y, z = np.meshgrid(gx, gy, gz, indexing='ij')
    nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    tt = _tt_table(model, nodes, sta, is_p)
    arr = t[None, :] - tt
    t0 = (arr * w).sum(1) / w.sum()
    r = arr - t0[:, None]
    rms = np.sqrt((r * r * w).sum(1) / w.sum())
    i = int(np.argmin(rms))
    return nodes[i], float(t0[i]), float(rms[i]), nodes, rms

def _on_boundary(xyz: np.ndarray, cfg: LocatorConfig) -> list[str]:
    names = ('xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax')
    return [names[k] for k in range(6)
            if abs(xyz[k // 2] - cfg.bounds[k]) <= cfg.boundary_tol_m]

def locate_event(picks: list[Pick], stations: dict[str, Station], model: VelocityModel,
                 cfg: LocatorConfig, diagnostics: list[Diagnostic] | None = None) -> EventResult:
    diag = list(diagnostics or [])
    rejected: list[Pick] = []
    used = [p for p in picks if p.phase is not None]
    n_sta = len({p.station for p in used})
    if n_sta < cfg.min_stations or len(used) < cfg.min_phases:
        diag.append(Diagnostic(
            'insufficient_data',
            f'need >= {cfg.min_stations} stations and >= {cfg.min_phases} phases, '
            f'got {n_sta} stations / {len(used)} phases',
            {'n_stations': n_sta, 'n_phases': len(used)}))
        return EventResult(LocateStatus.INSUFFICIENT_DATA, None, None, used, rejected, diag)

    xmin, xmax, ymin, ymax, zmin, zmax = cfg.bounds
    dropped = 0
    while True:
        sta, t, is_p, w = _arrays(used, stations)
        node, t0, rms0, _, _ = grid_misfit(model, sta, t, is_p, w, cfg)
        x0 = np.array([node[0], node[1], node[2], t0])
        sqrt_w = np.sqrt(w)

        def resid(m):
            tt = _tt_table(model, np.array([[m[0], m[1], m[2]]]), sta, is_p)[0]
            return (t - (m[3] + tt)) * sqrt_w

        try:
            fit = least_squares(resid, x0, bounds=([xmin, ymin, zmin, t.min() - 120.0],
                                                   [xmax, ymax, zmax, t.max() + 1.0]),
                                max_nfev=cfg.max_nfev)
        except Exception as exc:  # solver failure: keep diagnostics, no forced point
            diag.append(Diagnostic('no_convergence', f'least_squares raised: {exc}'))
            return EventResult(LocateStatus.NO_CONVERGENCE, None, None, used, rejected, diag)
        if fit.status <= 0:
            diag.append(Diagnostic('no_convergence', f'solver stopped: {fit.message}'))
            return EventResult(LocateStatus.NO_CONVERGENCE, None, None, used, rejected, diag)
        r_unw = resid(fit.x) / sqrt_w
        worst = int(np.argmax(np.abs(r_unw)))
        if (abs(r_unw[worst]) > cfg.residual_reject_s and dropped < cfg.max_reject
                and len(used) - 1 >= cfg.min_phases
                and (sum(1 for p in used if p.station == used[worst].station) > 1
                     or n_sta - 1 >= cfg.min_stations)):
            bad = used.pop(worst)
            rejected.append(bad)
            dropped += 1
            diag.append(Diagnostic(
                'residual_outlier',
                f'dropped {bad.station} {bad.phase.value} pick at {bad.time_s:.4f}s, '
                f'residual {r_unw[worst] * 1e3:.0f} ms',
                {'station': bad.station, 'phase': bad.phase.value,
                 'time_s': bad.time_s, 'residual_s': float(r_unw[worst])}))
            n_sta = len({p.station for p in used})
            continue
        break

    xyz = fit.x[:3]
    faces = _on_boundary(xyz, cfg)
    if faces:
        diag.append(Diagnostic(
            'out_of_model',
            f'solution pinned at model boundary ({", ".join(faces)}); '
            'true source likely outside the model volume, no coordinate reported',
            {'clamped_estimate': {'x': float(xyz[0]), 'y': float(xyz[1]), 'z': float(xyz[2]),
                                  't0_s': float(fit.x[3])},
             'boundary_faces': faces}))
        return EventResult(LocateStatus.OUT_OF_MODEL, None, None, used, rejected, diag)

    sta, t, is_p, w = _arrays(used, stations)
    r_unw = resid(fit.x) / np.sqrt(w)
    rms = float(np.sqrt(np.mean(r_unw ** 2)))
    if rms > cfg.max_rms_s:
        worst = int(np.argmax(np.abs(r_unw)))
        diag.append(Diagnostic(
            'unresolved_residuals',
            f'phase conflict: rms {rms * 1e3:.0f} ms > {cfg.max_rms_s * 1e3:.0f} ms after '
            f'{dropped} rejections; worst pick {used[worst].station} '
            f'{used[worst].phase.value} off by {r_unw[worst] * 1e3:.0f} ms; '
            'no coordinate reported',
            {'rms_s': rms, 'n_rejected': dropped,
             'worst': {'station': used[worst].station, 'phase': used[worst].phase.value,
                       'residual_s': float(r_unw[worst])}}))
        return EventResult(LocateStatus.ASSOCIATION_FAILED, None, None, used, rejected, diag)
    n = len(used)
    dof = max(n - 4, 1)
    s2 = float((r_unw ** 2).sum() / dof)
    jtj = fit.jac.T @ fit.jac
    cond = float(np.linalg.cond(jtj))
    cov = s2 * np.linalg.pinv(jtj)
    degraded = dropped > 0 or n <= cfg.min_phases
    if not np.isfinite(cond) or cond > 1e12:
        degraded = True
        diag.append(Diagnostic('ill_conditioned',
                               f'normal-matrix condition number {cond:.2e}; uncertainty inflated',
                               {'condition': cond}))
    pos_cov = cov[:3, :3]
    eig = np.linalg.eigvalsh(pos_cov)
    axes = tuple(float(np.sqrt(max(e, 0.0))) for e in eig)
    epi = xyz[:2]
    az = np.arctan2(sta[:, 1] - epi[1], sta[:, 0] - epi[0])
    az = np.sort(np.mod(az, 2 * np.pi))
    gap = float(np.max(np.diff(np.concatenate([az, az[:1] + 2 * np.pi]))))
    unc = Uncertainty(cov, float(np.sqrt(max(cov[0, 0], 0))), float(np.sqrt(max(cov[1, 1], 0))),
                      float(np.sqrt(max(cov[2, 2], 0))), float(np.sqrt(max(cov[3, 3], 0))),
                      axes, float(np.degrees(gap)), float(np.sqrt(np.mean(r_unw ** 2))),
                      n, len({p.station for p in used}))
    status = LocateStatus.LOCATED_DEGRADED if degraded else LocateStatus.LOCATED
    hypo = Hypocenter(float(xyz[0]), float(xyz[1]), float(xyz[2]), float(fit.x[3]))
    return EventResult(status, hypo, unc, used, rejected, diag)
