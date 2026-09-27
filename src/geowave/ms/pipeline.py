from __future__ import annotations
from dataclasses import dataclass, field
from geowave.ms import diag
from geowave.ms.detect import DetectConfig, detect_station
from geowave.ms.associate import AssocConfig, associate
from geowave.ms.locate import LocateConfig, LocateOutcome, locate

@dataclass(frozen=True)
class PipelineConfig:
    detect: DetectConfig = DetectConfig()
    assoc: AssocConfig = AssocConfig()
    locate: LocateConfig = LocateConfig()
    orphan_min_phases: int = 6   # 未关联到时超过该数则提示疑似漏报

@dataclass
class Failure:
    stage: str                 # 'associate' / 'locate'
    reasons: list[str]
    detail: str
    estimate: dict | None = None   # 尽力估计(无效, 仅供诊断)

@dataclass
class Report:
    events: list[LocateOutcome]
    failures: list[Failure]
    picks: list
    n_unassociated: int

    def summary(self) -> str:
        stage_cn = {'detect': '检测', 'associate': '关联', 'locate': '定位'}
        lines = ['=' * 64, '微震监测流程报告',
                 f'到时拾取 {len(self.picks)} 条 | 定位成功 {len(self.events)} 个 | '
                 f'失败/诊断 {len(self.failures)} 项 | 未关联 {self.n_unassociated} 条']
        for i, ev in enumerate(self.events, 1):
            lines.append(f'\n[事件 {i}] 定位成功')
            lines.append(f'  发震时刻 t0 = {ev.t0_s:.3f} ± {ev.sig_t0_s * 1e3:.0f} ms (相对记录起点)')
            lines.append(f'  位置 x = {ev.x_m:.1f} ± {ev.sig_x_m:.1f} m, y = {ev.y_m:.1f} ± {ev.sig_y_m:.1f} m')
            lines.append(f'  深度 z = {ev.z_m:.1f} ± {ev.sig_z_m:.1f} m')
            lines.append(f'  RMS 残差 {ev.rms_s * 1e3:.1f} ms | 相位 {ev.n_phases} 条/{ev.n_stations} 台 | '
                         f'方位空隙 {ev.gap_deg:.0f}° | 离群剔除 {ev.n_outliers} 条')
            for w in ev.warnings:
                lines.append(f'  警告: {diag.TEXT_CN.get(w, w)}')
        for i, f in enumerate(self.failures, 1):
            reason = ', '.join(diag.TEXT_CN.get(r, r) for r in f.reasons)
            lines.append(f'\n[诊断 {i}] 阶段: {stage_cn.get(f.stage, f.stage)} | 原因: {reason}')
            lines.append(f'  {f.detail}')
            if f.estimate:
                e = f.estimate
                lines.append(f"  尽力估计(无效,仅供诊断): x={e['x_m']:.0f} m, y={e['y_m']:.0f} m, "
                             f"z={e['z_m']:.0f} m, t0={e['t0_s']:.2f} s")
        return '\n'.join(lines)

class Pipeline:
    """检测 -> 跨台关联 -> 三维定位; 失败场景保留诊断, 不输出唯一坐标。"""

    def __init__(self, stations, tt, bounds, cfg: PipelineConfig = PipelineConfig()):
        self.stations = list(stations)
        self.tt = tt
        self.bounds = bounds
        self.cfg = cfg

    def run(self, traces: dict[str, dict[str, object]], fs_hz: float) -> Report:
        picks = []
        for st in self.stations:
            tr = traces[st.name]
            picks.extend(detect_station(tr['Z'], tr['N'], tr['E'], fs_hz, st.name, self.cfg.detect))
        picks.sort(key=lambda p: p.time_s)
        accepted, weak, leftover = associate(picks, self.stations, self.tt, self.bounds, self.cfg.assoc)
        events: list[LocateOutcome] = []
        failures: list[Failure] = []
        for cand in accepted:
            out = locate([picks[i] for i in cand.pick_idx], self.stations, self.tt,
                         self.bounds, (cand.x_m, cand.y_m, cand.z_m), cand.t0_s, self.cfg.locate)
            out.warnings.extend(cand.conflicts)
            if out.ok:
                events.append(out)
            else:
                failures.append(Failure(
                    'locate', out.reasons,
                    f"候选 t0={cand.t0_s:.2f}s 未通过质量控制: "
                    + '; '.join(diag.TEXT_CN.get(r, r) for r in out.reasons),
                    out.estimate()))
        if weak is not None:
            failures.append(Failure(
                'associate', [weak.reason],
                f'最强候选 {weak.n_phases} 相位/{weak.n_stations} 台, 低于阈值'
                f'({self.cfg.assoc.min_phases} 相位/{self.cfg.assoc.min_stations} 台), 未予定位',
                {'x_m': weak.x_m, 'y_m': weak.y_m, 'z_m': weak.z_m, 't0_s': weak.t0_s}))
        elif len(leftover) >= self.cfg.orphan_min_phases:
            failures.append(Failure('associate', [diag.WEAK_CANDIDATE],
                                    f'{len(leftover)} 条到时未能关联成事件, 疑似漏报或噪声', None))
        return Report(events, failures, picks, len(leftover))
