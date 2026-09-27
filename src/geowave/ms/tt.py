from __future__ import annotations
import numpy as np
from geowave.ms.geometry import Station3

class HomogeneousTT:
    """均匀半空间直射线走时; 任何实现 velocity(phase)/travel_time(x,y,z,station,phase)
    的对象(层状模型走时表、Eikonal 预计算网格等)都可替换注入流程。"""

    def __init__(self, vp_mps: float, vs_mps: float):
        if vp_mps <= 0 or vs_mps <= 0:
            raise ValueError('velocity must be positive')
        self.vp_mps = float(vp_mps)
        self.vs_mps = float(vs_mps)

    def velocity(self, phase: str) -> float:
        if phase == 'P':
            return self.vp_mps
        if phase == 'S':
            return self.vs_mps
        raise ValueError(f'unknown phase {phase}')

    def travel_time(self, x_m, y_m, z_m, st: Station3, phase: str):
        r = np.sqrt((np.asarray(x_m, float) - st.x_m) ** 2
                    + (np.asarray(y_m, float) - st.y_m) ** 2
                    + (np.asarray(z_m, float) - st.z_m) ** 2)
        return r / self.velocity(phase)
