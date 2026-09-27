from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class Phase(str, Enum):
    P = 'P'
    S = 'S'

@dataclass(frozen=True)
class Station:
    name: str
    x: float
    y: float
    z: float  # depth, positive down; surface = 0

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

@dataclass(frozen=True)
class Pick:
    station: str
    time_s: float
    phase: Phase | None = None  # None = unlabeled detector output
    snr: float = 0.0
    weight: float = 1.0

    def with_phase(self, phase: Phase) -> 'Pick':
        return Pick(self.station, self.time_s, phase, self.snr, self.weight)

class LocateStatus(str, Enum):
    LOCATED = 'located'                      # clean solution, full redundancy
    LOCATED_DEGRADED = 'located_degraded'    # located, but picks rejected / low redundancy / ill-conditioned
    INSUFFICIENT_DATA = 'insufficient_data'  # too few stations or phases
    ASSOCIATION_FAILED = 'association_failed'# picks not moveout-consistent, no viable hypocenter
    OUT_OF_MODEL = 'out_of_model'            # solution driven against model boundary
    NO_CONVERGENCE = 'no_convergence'

@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    context: dict = field(default_factory=dict)

@dataclass(frozen=True)
class Hypocenter:
    x: float
    y: float
    z: float
    t0_s: float

@dataclass(frozen=True)
class Uncertainty:
    covariance: np.ndarray            # 4x4 over (x, y, z, t0)
    sigma_x_m: float
    sigma_y_m: float
    sigma_z_m: float
    sigma_t_s: float
    ellipsoid_axes_m: tuple[float, float, float]  # 1-sigma semi-axis lengths
    azimuthal_gap_deg: float
    rms_s: float
    n_phases: int
    n_stations: int

@dataclass
class EventResult:
    status: LocateStatus
    hypocenter: Hypocenter | None
    uncertainty: Uncertainty | None
    picks: list[Pick]
    rejected: list[Pick]
    diagnostics: list[Diagnostic]

    @property
    def located(self) -> bool:
        return self.status in (LocateStatus.LOCATED, LocateStatus.LOCATED_DEGRADED)
