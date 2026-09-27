from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Pick:
    station: str
    time_s: float      # 相对记录起点
    phase: str         # 'P' / 'S'
    snr: float
    sigma_s: float     # 到时不确定度
    vshare: float = 0.0  # 初动短窗垂向能量占比(诊断用)

@dataclass(frozen=True)
class DetectConfig:
    sta_s: float = 0.02
    lta_s: float = 1.0
    thr_on: float = 2.5
    thr_off: float = 1.5
    min_gap_s: float = 0.08
    max_dur_s: float = 2.0
    pick_back_s: float = 0.12   # 触发点向前搜索包络峰值的窗长
    env_smooth_s: float = 0.008
    pick_frac: float = 0.25     # 到时 = 包络升至峰值该比例处
    label_win_s: float = 0.12
    p_vshare: float = 0.30      # 垂向能量占比超过该值判为 P
    f0_hz: float = 30.0
    min_snr: float = 2.0

def _causal_mean(x: np.ndarray, w: int) -> np.ndarray:
    n = len(x)
    out = np.zeros(n)
    if w <= 0 or n < w:
        return out
    s = np.concatenate(([0.0], np.cumsum(x)))
    idx = np.arange(w - 1, n)
    out[w - 1:] = (s[idx + 1] - s[idx + 1 - w]) / w
    return out

def _triggers(cf: np.ndarray, fs_hz: float, cfg: DetectConfig) -> list[tuple[int, float]]:
    nsta = max(1, int(round(cfg.sta_s * fs_hz)))
    nlta = max(nsta + 1, int(round(cfg.lta_s * fs_hz)))
    sta = _causal_mean(cf, nsta)
    lta = _causal_mean(cf, nlta)
    min_gap = int(cfg.min_gap_s * fs_hz)
    max_dur = int(cfg.max_dur_s * fs_hz)
    out: list[tuple[int, float]] = []
    on = False
    frozen = 1.0
    i_on = 0
    last_on = -10 ** 9
    peak = 0.0
    for i in range(nlta - 1, len(cf)):
        if not on:
            lv = lta[i]
            r = sta[i] / lv if lv > 0 else 0.0
            if r >= cfg.thr_on and i - last_on >= min_gap:
                on, i_on, last_on, frozen, peak = True, i, i, lv, r
        else:
            r = sta[i] / frozen if frozen > 0 else 0.0
            peak = max(peak, r)
            if r < cfg.thr_off or i - i_on >= max_dur:
                out.append((i_on, peak))
                on = False
    if on:
        out.append((i_on, peak))
    return out

def _onset_pick(cf: np.ndarray, i_trig: int, fs_hz: float, cfg: DetectConfig) -> int:
    """触发点前窗内找包络峰值, 回退到峰值固定比例处; 各台偏差一致, 残差由 t0 吸收。"""
    i0 = max(0, i_trig - int(cfg.pick_back_s * fs_hz))
    i1 = min(len(cf), i_trig + int(0.1 * fs_hz))
    env = _causal_mean(cf[i0:i1], max(1, int(cfg.env_smooth_s * fs_hz)))
    if env.size < 4:
        return i_trig
    i_pk = int(np.argmax(env))
    thr = cfg.pick_frac * env[i_pk]
    i = i_pk
    while i > 0 and env[i] > thr:
        i -= 1
    return i0 + i

def detect_station(z, n, e, fs_hz: float, station: str, cfg: DetectConfig = DetectConfig()) -> list[Pick]:
    """单台三分量检测: 总能量 STA/LTA(冻结 LTA) -> 包络比例到时 -> 垂向占比判别 P/S。"""
    z = np.asarray(z, float)
    n = np.asarray(n, float)
    e = np.asarray(e, float)
    cf = z * z + n * n + e * e
    lw = max(1, int(cfg.label_win_s * fs_hz))
    nlta = int(cfg.lta_s * fs_hz)
    picks: list[Pick] = []
    for i_on, _peak in _triggers(cf, fs_hz, cfg):
        ip = _onset_pick(cf, i_on, fs_hz, cfg)
        ez = float(np.sum(z[ip:ip + lw] ** 2))
        eh = float(np.sum(n[ip:ip + lw] ** 2) + np.sum(e[ip:ip + lw] ** 2))
        vshare = ez / (ez + eh + 1e-30)
        noise = float(np.median(cf[max(0, ip - nlta):ip])) + 1e-30
        sig = float(np.max(cf[ip:ip + max(1, int(0.15 * fs_hz))])) if ip < len(cf) else 0.0
        snr = float(np.sqrt(max(sig / noise, 0.0)))
        if snr < cfg.min_snr:
            continue
        sigma = float(np.clip(1.0 / (2 * np.pi * cfg.f0_hz * snr), 0.002, 0.05))
        picks.append(Pick(station, ip / fs_hz, 'P' if vshare >= cfg.p_vshare else 'S', snr, sigma, vshare))
    return picks
