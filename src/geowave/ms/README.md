# geowave.ms — 压裂微震监测流程

从连续记录到三维震源的完整流程:**检测 → 跨台相位关联 → 三维定位**,输出发震时刻、
位置及不确定度;台站不足、相位冲突、解越界等场景保留失败诊断,不硬给唯一坐标。

## 流程

```
连续三分量记录 (dict[台站]['Z'/'N'/'E'])
  └─ detect:    总能量 STA/LTA(冻结 LTA)触发 → 包络固定比例到时 → 垂向能量占比判 P/S
  └─ associate: 三维走时网格上对每个候选点聚类 t0 = t_pick − T(grid, 台, 相),
                票数最多且簇最紧的一簇即候选事件; 同台同相重复到时保留最近者并记冲突
  └─ locate:    soft_l1 稳健最小二乘(x,y,z,t0) → MAD 离群迭代剔除 →
                多起点重解一致性检验 → 列归一化法方程求协方差(1σ) → 质量控制
  └─ Report:    成功事件(含不确定度) + 失败诊断(含尽力估计, 标注无效)
```

## 用法

```python
from geowave.ms.geometry import Bounds3
from geowave.ms.tt import HomogeneousTT
from geowave.ms.pipeline import Pipeline, PipelineConfig

pipe = Pipeline(stations, HomogeneousTT(vp_mps=4500, vs_mps=2600),
                Bounds3(-2000, 2000, -2000, 2000, 200, 3000), PipelineConfig())
report = pipe.run(traces, fs_hz=500.0)   # traces[台名]['Z'/'N'/'E'] 为等采样序列
print(report.summary())
```

演示(合成台阵: 10 地表 + 3 井中; 4 个事件含越界与弱事件):

```bash
python3 -m geowave.ms.demo
```

## 失败诊断(均保留在 Report.failures, 不输出有效坐标)

| 原因 | 触发条件 |
|---|---|
| 台站数不足 / 相位数不足 | 关联或定位时低于阈值(默认 4 台 / 6 相位) |
| 缺少S波约束 | 仅 P 波时深度-发震时刻耦合, 默认拒绝(可关) |
| 相位冲突 | 离群剔除比例超过阈值(默认 34%), 疑似相位错配 |
| 解落在模型范围外 | 解超出 Bounds3(含小裕量), 尽力估计仅供诊断 |
| 解不稳定 | 1σ 超阈、法方程病态, 或多起点重解不收敛到同一解 |
| 关联票数不足 | 剩余到时无法成簇, 疑似弱事件或噪声(提示漏报) |

不确定度: 由加权残差与列归一化法方程给出 x/y/z/t0 的 1σ, 另报 RMS 残差、
方位角空隙、相位/台站数、离群剔除数, 供质控判断。

## 接口与替换点

- **走时模型**: 任何实现 `velocity(phase)` / `travel_time(x, y, z, station, phase)`
  的对象均可注入(层状模型、Eikonal 走时表, 如 `geowave.ray.travel_table`)。
- **P/S 判别**: 现为垂向能量占比启发式(近垂直入射适用); 生产可换偏振分析或
  PhaseNet 等学习型拾取器, `Pick` 结构不变。
- **阈值**: `DetectConfig` / `AssocConfig` / `LocateConfig` 全部可配。

## 已知局限

- 到时拾取为包络比例法, 偏差随波形主频变化(由 t0 吸收), 极低信噪比下离群由
  定位阶段剔除; 大水量噪声记录建议先带通。
- 关联网格步长(默认 100 m)决定候选位置精度, 精定位在 locate 阶段完成。
- 均匀模型下演示误差 ~12 m; 层状模型请替换走时表, 否则系统差会体现在残差中。
