from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from geowave.microseismic.types import Phase

class VelocityModel:
    def travel_time(self, phase: Phase, src: tuple, sta: tuple) -> float:
        return float(self.travel_time_table(phase, [src], [sta])[0, 0])

    def travel_time_table(self, phase: Phase, srcs, stas) -> np.ndarray:
        raise NotImplementedError

    def velocity_at(self, phase: Phase, z: float) -> float:
        raise NotImplementedError

@dataclass(frozen=True)
class HomogeneousModel(VelocityModel):
    vp_mps: float
    vs_mps: float

    def travel_time_table(self, phase: Phase, srcs, stas) -> np.ndarray:
        srcs = np.asarray(srcs, float).reshape(-1, 3)
        stas = np.asarray(stas, float).reshape(-1, 3)
        v = self.vp_mps if phase is Phase.P else self.vs_mps
        d = np.linalg.norm(srcs[:, None, :] - stas[None, :, :], axis=-1)
        return d / v

    def velocity_at(self, phase: Phase, z: float) -> float:
        return self.vp_mps if phase is Phase.P else self.vs_mps

@dataclass(frozen=True)
class LayeredModel1D(VelocityModel):
    interfaces_m: np.ndarray  # layer top depths, increasing
    vp_mps: np.ndarray
    vs_mps: np.ndarray

    def __post_init__(self):
        z = np.asarray(self.interfaces_m, float)
        vp = np.asarray(self.vp_mps, float)
        vs = np.asarray(self.vs_mps, float)
        if z.ndim != 1 or z.shape != vp.shape or z.shape != vs.shape:
            raise ValueError('interfaces/vp/vs shape mismatch')
        if np.any(np.diff(z) <= 0):
            raise ValueError('interfaces must strictly increase')
        if np.any(vp <= 0) or np.any(vs <= 0):
            raise ValueError('nonpositive velocity')
        object.__setattr__(self, 'interfaces_m', z)
        object.__setattr__(self, 'vp_mps', vp)
        object.__setattr__(self, 'vs_mps', vs)

    def velocity_at(self, phase: Phase, z: float) -> float:
        v = self.vp_mps if phase is Phase.P else self.vs_mps
        i = min(int(np.searchsorted(self.interfaces_m, z, 'right')) - 1, len(v) - 1)
        return float(v[max(i, 0)])

    def _overlap(self, z1, z2, v):
        lo = np.minimum(z1, z2)[..., None]
        hi = np.maximum(z1, z2)[..., None]
        tops = self.interfaces_m
        bots = np.concatenate([tops[1:], [np.inf]])
        h = np.clip(np.minimum(hi, bots) - np.maximum(lo, tops), 0.0, None)
        return h  # (..., n_layers) thickness of each layer between the two depths

    def travel_time_table(self, phase: Phase, srcs, stas) -> np.ndarray:
        srcs = np.asarray(srcs, float).reshape(-1, 3)
        stas = np.asarray(stas, float).reshape(-1, 3)
        v = self.vp_mps if phase is Phase.P else self.vs_mps
        z1 = np.broadcast_to(srcs[:, None, 2], (len(srcs), len(stas)))
        z2 = np.broadcast_to(stas[None, :, 2], (len(srcs), len(stas)))
        r = np.hypot(srcs[:, None, 0] - stas[None, :, 0], srcs[:, None, 1] - stas[None, :, 1])
        h = self._overlap(z1, z2, v)  # (n, m, L)
        thick = h.sum(-1)
        # degenerate: both endpoints at (nearly) the same depth -> horizontal ray at that depth
        flat = thick < 1e-6
        vmax = np.where(h > 0, v, 0.0).max(-1)
        vmax = np.maximum(vmax, 1e-9)
        p_lo = np.zeros_like(r)
        p_hi = 0.9999 / vmax
        for _ in range(70):
            p = 0.5 * (p_lo + p_hi)
            s = np.clip(p[..., None] * v, 0.0, 0.9999)
            x = (h * s / np.sqrt(1.0 - s * s)).sum(-1)
            need = x < r
            p_lo = np.where(need, p, p_lo)
            p_hi = np.where(need, p_hi, p)
        p = 0.5 * (p_lo + p_hi)
        s = np.clip(p[..., None] * v, 0.0, 0.9999)
        t = (h / (v * np.sqrt(1.0 - s * s))).sum(-1)
        if np.any(flat):
            zmid = 0.5 * (z1 + z2)
            vi = np.array([self.velocity_at(phase, float(z)) for z in np.ravel(zmid)]).reshape(zmid.shape)
            t = np.where(flat, r / vi, t)
        return t
