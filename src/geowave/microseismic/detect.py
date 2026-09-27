from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.microseismic.types import Pick

@dataclass(frozen=True)
class DetectorConfig:
    sta_s: float = 0.03
    lta_s: float = 0.6
    on: float = 4.0
    off: float = 1.5
    min_separation_s: float = 0.1
    aic_back_s: float = 0.05   # short asymmetric AIC window around the trigger
    aic_fwd_s: float = 0.025
    snr_window_s: float = 0.5
    min_snr: float = 3.0

def _causal_mean(x: np.ndarray, n: int) -> np.ndarray:
    c = np.concatenate(([0.0], np.cumsum(x)))
    out = np.zeros_like(x, dtype=float)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    out[:n - 1] = out[n - 1] if x.size >= n else 0.0
    return out

def frozen_lta_ratio(energy: np.ndarray, n_sta: int, n_lta: int,
                     on: float, off: float) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """STA/LTA with the LTA frozen while a trigger is active, so a P arrival
    does not poison the noise estimate for a following S arrival."""
    sta = _causal_mean(energy, n_sta)
    lta = _causal_mean(energy, n_lta)
    n = energy.size
    ratio = np.zeros(n)
    lta_frozen = np.zeros(n)
    triggers: list[tuple[int, int]] = []
    frozen = lta[n_lta] if n > n_lta else 1.0
    in_trig = False
    start = 0
    freeze_until = 0  # delayed unfreeze: let the coda flush out of the LTA window
    for i in range(n_lta, n):
        if not in_trig and i >= freeze_until:
            frozen = lta[i]
        lta_frozen[i] = frozen
        r = sta[i] / frozen if frozen > 0 else 0.0
        ratio[i] = r
        if not in_trig and r >= on:
            in_trig, start = True, i
        elif in_trig and r < off:
            triggers.append((start, i))
            freeze_until = i + n_lta
            in_trig = False
    if in_trig:
        triggers.append((start, n - 1))
    return ratio, lta_frozen, triggers

def aic_pick(x: np.ndarray) -> int:
    x = np.asarray(x, float)
    n = x.size
    if n < 8:
        return n // 2
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))

    def var(lo, hi):
        m = hi - lo
        s1 = c1[hi] - c1[lo]
        s2 = c2[hi] - c2[lo]
        return np.maximum(s2 / m - (s1 / m) ** 2, 1e-20)

    k = np.arange(2, n - 2)
    aic = k * np.log(var(0, k)) + (n - k - 1) * np.log(var(k, n))
    return int(k[int(np.argmin(aic))])

def detect_picks(trace: np.ndarray, dt_s: float, station: str, t0_s: float = 0.0,
                 cfg: DetectorConfig = DetectorConfig()) -> list[Pick]:
    a = np.asarray(trace, float)
    comps = a[None, :] if a.ndim == 1 else a
    energy = (comps ** 2).sum(axis=0)
    n_sta = max(2, int(round(cfg.sta_s / dt_s)))
    n_lta = max(n_sta + 2, int(round(cfg.lta_s / dt_s)))
    n_back = max(4, int(round(cfg.aic_back_s / dt_s)))
    n_fwd = max(4, int(round(cfg.aic_fwd_s / dt_s)))
    n_snr = max(2, int(round(cfg.snr_window_s / dt_s)))
    _, lta_frozen, triggers = frozen_lta_ratio(energy, n_sta, n_lta, cfg.on, cfg.off)
    picks: list[Pick] = []
    for start, end in triggers:
        lo = max(0, start - n_back)
        hi = min(energy.size, start + n_fwd)
        i_ref = lo + aic_pick(energy[lo:hi]) if hi - lo >= 8 else start
        bg = float(lta_frozen[start])
        peak = float(np.max(energy[start:start + n_snr])) if start < energy.size else 0.0
        snr = float(np.sqrt(peak / bg)) if bg > 0 else 0.0
        if snr >= cfg.min_snr:
            picks.append(Pick(station, t0_s + i_ref * dt_s, None, snr))
    picks.sort(key=lambda p: p.time_s)
    merged: list[Pick] = []
    for p in picks:
        if merged and p.time_s - merged[-1].time_s < cfg.min_separation_s:
            if p.snr > merged[-1].snr:
                merged[-1] = p
        else:
            merged.append(p)
    return merged
