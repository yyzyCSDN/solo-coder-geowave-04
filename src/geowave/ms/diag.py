from __future__ import annotations

TOO_FEW_STATIONS = 'too_few_stations'
TOO_FEW_PHASES = 'too_few_phases'
NO_S_PHASE = 'no_s_phase'
PHASE_CONFLICT = 'phase_conflict'
OUT_OF_MODEL = 'out_of_model'
UNSTABLE = 'unstable_solution'
WEAK_CANDIDATE = 'weak_candidate'
LARGE_GAP = 'large_azimuth_gap'

TEXT_CN = {
    TOO_FEW_STATIONS: '台站数不足',
    TOO_FEW_PHASES: '相位数不足',
    NO_S_PHASE: '缺少S波约束',
    PHASE_CONFLICT: '相位冲突(离群到时比例过高)',
    OUT_OF_MODEL: '解落在模型范围外',
    UNSTABLE: '解不稳定(不确定度超限或法方程病态)',
    WEAK_CANDIDATE: '关联票数不足,疑似弱事件或噪声',
    LARGE_GAP: '方位角空隙过大',
}
