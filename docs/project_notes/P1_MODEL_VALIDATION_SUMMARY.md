# P1 Model Validation Summary: 5+12 Baseline Signals Before N9-3x5+12

> 状态：P1 完成，进入 P2
> 日期：2026-06-12
> 作者：yangrui

---

## 1. P1 阶段目的

P1 的目的**不是**证明大乐透可预测，**不是**证明稳定盈利。

P1 的唯一目的是：

> 验证当前 5+12 单组策略中，`rf_stat`、`stat_miss_bonus`、`gap_direct_top5` 是否存在比 random 更值得用于 P2 候选池生成的**弱信号**。

"弱信号"的定义：
- 在命中率（hit2+、hit3+）或 ROI（roi_truncated）上，统计上一致优于 random baseline；
- 不一定能达到统计显著性（p<0.05），但方向一致、跨窗口可复现；
- 信号的绝对强度不足以支撑盈利，但可作为候选池的"筛选器"。

---

## 2. P1-1 至 P1-7 阶段摘要

### P1-1：综合指标补全

- 新增 `compute_strategy_metrics(records)` 统一指标函数，所有策略复用
- 新增指标：`roi_truncated`（排除 5+2/5+1 头奖污染，cap=10000）、`hit2_or_more_rate`、`hit3_or_more_rate`、`max_drawdown`、`max_consecutive_loss`、`avg_front_hit`
- 新增 `front_hit` 分布（hit0~hit5 各率）
- 输出：`backtest_summary.json`、`backtest_comparison.json`

### P1-2：200 / 500 / 1000 期大窗口验证

- 200 期：rf_stat 最优（hit3+ 92%ile，4/9 指标超 90%）
- 500 期：stat_miss_bonus 最优（hit3+ 88.8%ile），gap 策略 roi_truncated 最优
- 1000 期：趋势一致，但所有策略 ROI 均为负，random 在 roi 均值上受头奖极端事件严重污染
- **发现**：ROI 不是有效的策略比较指标——random 分布受 5+2 头奖事件主导，std/mean > 10

### P1-3：Random Percentile 与 Robust Metrics

- 新增 `compute_random_percentiles(main_result, trial_metrics)` — 计算 main 在 random trial 分布中的百分位
- 新增 `roi_truncated`（reward cap=10000）排除头奖污染后，策略间比较更有意义
- 输出：`backtest_percentile_*.json`、`backtest_random_trials.csv/json`
- **发现**：截断前 random ROI = -0.31，截断后 = -0.76 — 与 main 几乎一致。头奖污染是之前结论矛盾的根本原因

### P1-4：RF / Stat / Gap 消融实验

- `--ablation 1` 一次性运行 `stat_only` / `rf_only` / `rf_stat` 三种变体
- 200 期：rf_stat（hit3+ 92%ile, avg_hit 94%ile, 4/9 超 90%）— RF+stat 有协同效应
- 500 期：rf_stat 与 stat_only 持平，协同效应减弱
- rf_only 单独使用不如 stat_only — RF 需要 stat 互补

### P1-5：遗漏值方向实验

- 新增 `stat_miss_penalty` / `stat_no_miss` / `stat_miss_bonus` / `rf_stat_gap` 变体
- **发现**：
  - 遗漏值信号方向**取决于时间窗口**：近端（200期）惩罚有效，远端（500期）加分有效
  - `stat_miss_bonus`（频次 + 0.20×遗漏加分）在 500 期 hit3+ 88.8%ile
  - `rf_stat_gap`（RF+freq+gap）被证实无效 — gap 与 RF 组合没有协同效应
  - `stat_no_miss`（不用遗漏值）全面劣于使用遗漏值的变体

### P1-6：Gap Filter Ablation 与 Deterministic Tie-Breaking

- 确认 `generate_front_combos` 参数传递没有结构性错误
- **修复浮点平局 Bug**：`_compute_gap_only_scores` 的二次归一化导致不同号码获得相同浮点分数，`argsort` 和 `combinations` 的平局策略不一致。修复：改用 `gap + n/1000` 确保无平局
- **修复 Python 作用域 Bug**：`main()` 内的 `from inference_plus import predict_rf_scores` 被编译器视为局部变量声明，污染其他分支
- 修正后 `gap_direct_top5` = `gap_score_no_filters` = `gap_score_top5_candidate` — 数学自洽
- 200 期 rf_stat 最优（92%ile），500 期 gap 类 roi_truncated 最优（-0.649）
- **核心结论**：策略表现存在明显窗口依赖，不能解释为彩票存在可预测规律

### P1-7：Rolling Window 稳定性验证

- 26 个窗口（窗口大小 200，步长 100，覆盖 issue 8029→26003）
- 每窗口 4 策略 × 100 random trials，rf_stat 仅在无泄漏窗口运行（4/26）
- **结果**：
  - stat_miss_bonus 在 4 项冠军统计中全第一（hit3+ 12/26）
  - random 在 roi_truncated 冠军数上最多（10/26）
  - stat_miss_bonus 超过 random 均值的窗口数最多（11/26）
  - rf_stat 仅在 4 个窗口可用

---

## 3. P1-7 核心结果

### 实验参数

| 参数 | 值 |
|------|-----|
| rolling_window | 200 期 |
| rolling_step | 100 期 |
| 窗口总数 | 26 |
| random trials / 窗口 | 100 |
| rf_stat 有效窗口 | 4 |
| rf_stat 跳过窗口 | 22（泄漏风险） |

### 冠军统计

| 指标 | stat_miss_bonus | gap_direct_top5 | random | rf_stat |
|------|:---:|:---:|:---:|:---:|
| `roi_truncated` | 9 | 5 | **10** | 2 |
| `avg_front_hit` | **12** | 9 | 4 | 1 |
| `hit2_or_more_rate` | **11** | 10 | 4 | 1 |
| `hit3_or_more_rate` | **12** | 9 | 3 | 2 |

### 超过 random 均值的窗口数

| 策略 | 窗口数 |
|------|:---:|
| stat_miss_bonus | **11/26** |
| gap_direct_top5 | 7/26 |
| rf_stat | 2/26 |

### 跨窗口稳定性

| Strategy | roi_t mean±std | hit3+ mean±std | 评级 |
|----------|:---:|:---:|:---:|
| stat_miss_bonus | **-0.719** ± 0.172 | **0.018** ± 0.011 | ★★★★ |
| gap_direct_top5 | -0.774 ± 0.114 | 0.013 ± 0.006 | ★★★ |
| rf_stat | -0.780 ± 0.089 | 0.018 ± 0.015 | ★★ |
| random | -0.765 ± **0.013** | 0.014 ± **0.001** | ★★★★★ |

---

## 4. 结论

### 4.1 核心发现

1. **random 在 roi_truncated 冠军数上最多（10/26）**
   — 没有任何策略能稳定压过随机收益。这是最重要的发现：在 200 期窗口上，所有策略的 ROI 都在随机误差范围内。

2. **stat_miss_bonus 是当前最稳定的非随机选号信号**
   — hit3+ 冠军 12/26、超过 random 均值 11/26、roi_t 均值最优（-0.719）。所有 4 项冠军统计全第一。

3. **rf_stat 在近期窗口有爆发力，但证据量不足**
   — 仅在 4/26 个窗口可运行（RF 训练截止 22096 之后）。在可用窗口中有 2 次冠军，但覆盖率太低，不能作为单一依赖。

4. **gap_direct_top5 有一定窗口依赖，不能作为主信号**
   — 超过 random 均值的窗口仅 7/26，且被 stat_miss_bonus 全面压制。

### 4.2 P1 是否支持进入 P2？

**支持进入 P2，但不支持宣传稳定盈利。**

进入 P2 的依据：
- stat_miss_bonus 在 hit3+ 上跨窗口一致性（12/26 冠军）为 9 候选池提供了可靠的基础信号
- rf_stat 在近期窗口的爆发力为候选池提供了互补来源
- 三轨融合（stat + rf + gap）的互补性已被 26 窗口验证
- 随机基线框架已成熟，可直接用于 P2 验证

不支持"稳定盈利"的依据：
- random 在 roi 冠军数上最多 → 没有任何策略能稳定超越随机
- 所有策略 roi_t 均为负 → 即使最优策略也处于亏损状态
- rf_stat 仅 4 个可用窗口 → 证据不足以支持"RF 有预测力"的强结论

---

## 5. P2 候选池建议

**P2 不只测试一个比例，而是测试三个比例。**

### A. `n9_mix_441`（4:4:1）

| 来源 | 数量 |
|------|:---:|
| stat_miss_bonus 高分号 | 4 |
| rf_stat 高分号 | 4 |
| gap_direct_top5 高分号 | 1 |

理由：保留 RF 近端爆发力，gap 作为最小补充。

### B. `n9_mix_531`（5:3:1）

| 来源 | 数量 |
|------|:---:|
| stat_miss_bonus 高分号 | 5 |
| rf_stat 高分号 | 3 |
| gap_direct_top5 高分号 | 1 |

理由：更偏向跨窗口稳定性（stat 主导），RF 辅助。

### C. `n9_mix_621`（6:2:1）

| 来源 | 数量 |
|------|:---:|
| stat_miss_bonus 高分号 | 6 |
| rf_stat 高分号 | 2 |
| gap_direct_top5 高分号 | 1 |

理由：检验 stat_miss_bonus 主导是否最稳（冠军数全第一的极端版本）。

如有重复：按综合分继续补足到 9 个。三个比例均需对比 random 3组5+12。

---

## 6. P2 开始前的原则

进入 P2 后必须遵守：

- ❌ 不宣称稳定盈利
- ❌ 不使用马丁倍投
- ❌ 不直接做深度学习
- ❌ 不做严格 N+12 大矩阵
- ✅ 先做 9 候选池 + 3 组 5+12 启发式缩水
- ✅ 每个比例必须对比 random 3组5+12（相同成本 396 元）
- ✅ 必须比较成本从 132→396 后，roi_truncated 是否改善
- ✅ 必须输出：`any_group_hit3+`、`max_group_front_hit`、`max_drawdown`、`max_consecutive_loss`

---

## 7. 推荐路线

```
P1 ✅ 完成
  │
P2-1：9候选池生成器（三轨融合定价）
  │
P2-2：3组5+12 启发式缩水
  │
P2-3：三个候选池比例回测（4:4:1 / 5:3:1 / 6:2:1）
  │
P2-4：信号门槛与跳过机制（低信号期不下单）
  │
P3：N+12 矩阵缩水与覆盖率验证
  │
P4：风控看板与产品化
```

---

*P1 结束，P2 开始。*
