from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Station3:
    name: str
    x_m: float
    y_m: float
    z_m: float = 0.0  # 深度, 向下为正, 地表为 0

@dataclass(frozen=True)
class Bounds3:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def contains(self, x: float, y: float, z: float, margin_m: float = 0.0) -> bool:
        return (self.x_min - margin_m <= x <= self.x_max + margin_m
                and self.y_min - margin_m <= y <= self.y_max + margin_m
                and self.z_min - margin_m <= z <= self.z_max + margin_m)
