# P1 to P2 Route Decision: From 5+12 Single Strategy to 9-Candidate 3x5+12 Heuristic System

> 状态：已决策，待 P1-7 完成后进入 P2  
> 日期：2026-06-11  
> 作者：yangrui

---

## 1. 当前项目定位

本项目**不是**"保证预测大乐透开奖号码"的系统，**不承诺**稳定中奖或稳定盈利。

当前定位：

> 大乐透历史数据分析、策略回测、前区候选池构建、5+12 / N+12 方案模拟与风控评估平台。

核心目标（保留用户原始思路）：

- **不以一等、二等大奖为主目标** — 极低概率，不作为策略优化方向；
- **后区 12 个全包** — 消除后区不确定性；
- **主要优化前区** — 聚焦前区命中率（3+ 为主要目标）；
- **目标是提高前区中 3 个的触发概率** — 前区 3+ 是"回血"的基准线；
- **通过小奖覆盖成本，追求长期小额现金流** — 盈利期望仍为负，但追求降低亏损率；
- **所有策略必须通过历史回测和 random baseline 验证** — 不做未经回测的策略上线。

---

## 2. 当前已完成阶段（P1 进展）

### P1-1：综合指标补全
- `compute_strategy_metrics(records)` 统一指标计算函数
- 新增：`roi_truncated`（排除头奖污染）、`hit2_or_more_rate`、`hit3_or_more_rate`、`max_drawdown`、`max_consecutive_loss`
- 输出：`backtest_summary.json`、`backtest_comparison.json`

### P1-2：200 / 500 / 1000 期大窗口验证
- 500 期 × 500 random trials 验证
- `stat_miss_bonus` 在 hit3+ 上一致优于 random（z≈0.7-0.9）
- ROI 受 5+2 头奖极端方差污染，无法作为直接比较指标

### P1-3：Random Percentile 与 Robust Metrics
- 输出每个策略在 random 分布中的百分位
- `roi_truncated`（cap=10000）排除头奖污染后，策略间比较更有意义
- `backtest_percentile_*.json`、`backtest_random_trials.csv/json`

### P1-4：RF/Stat/Gap 消融实验
- `stat_only` / `rf_only` / `rf_stat` 三种模式对比
- 200 期：`rf_stat` 最优（hit3+ 92%ile）
- 500 期：`rf_stat` 与 `stat_only` 持平，优势减弱

### P1-5：遗漏值方向实验
- 新增 `stat_miss_penalty` / `stat_no_miss` / `stat_miss_bonus` / `rf_stat_gap` 变体
- 发现：`stat_miss_bonus`（遗漏加分）在 500 期 hit3+ 最强（88.8%ile）
- `rf_stat_gap`（RF+freq+gap）不理想，被证实方向错误

### P1-6：Gap Filter Ablation 与 Deterministic Tie-Breaking
- 确认 `generate_front_combos` 参数传递没有结构性错误
- 修复浮点平局问题（`_compute_gap_only_scores` 改用 `gap + n/1000`）
- `gap_direct_top5`、`gap_score_no_filters`、`gap_score_top5_candidate` 数学自洽
- 200 期 `rf_stat` 最优，500 期 gap 类 roi_truncated 最优
- 策略表现存在明显窗口依赖，但不能解释为彩票存在可预测规律

---

## 3. 新上传思路的吸收与取舍

### 吸收（纳入主线）

| 思路 | 状态 | 说明 |
|------|------|------|
| 数据工程与因子工程 | ✅ 纳入 | 频次/遗漏/结构特征是基础 |
| 多模型评分 | ✅ 纳入 | RF + stat + gap 三轨并行 |
| 严格回测 | ✅ 纳入 | 已实现 walk_forward + random baseline |
| 风控看板 | 后续 P4 | Streamlit 已有基础 |
| N+12 / 矩阵缩水 | 后续 P3 | 作为产品化模块 |
| 后区全包 + 前区优化 | ✅ 核心结构 | 保持不变 |

### 暂缓或不作为主线

| 思路 | 状态 | 原因 |
|------|------|------|
| 马丁倍投 | ❌ 暂缓 | 不能改变负期望，只会放大回撤 |
| 深度学习直接预测号码 | ❌ 暂缓 | 几千期数据极易过拟合 |
| GAN / DRL | ❌ 暂缓 | 同上，且无回测验证手段 |
| 奖池溢价防撞号 | ❌ 暂缓 | 偏向一二等奖，不符合"小奖回血"主线 |
| 严格 N+12 保 3 中 3 矩阵 | ❌ 暂缓 | 成本可能倒挂，先做启发式缩水 |

---

## 4. 最终路线选择

### 路线 A（P1 收尾）→ 路线 B（P2：9候选池 + 3x5+12 启发式缩水）

**P1-7：Rolling Window 稳定性验证**（当前正在执行）

目标：验证 rf_stat / stat_miss_bonus / gap_direct_top5 在不同 200 期窗口中的稳定性，回答"哪个策略最可靠"。

### P2 主策略设计（暂定）

```
strategy_name = n9_3x_5plus12
```

核心结构：

- 每期生成 **9 个前区候选号**（来自 rf_stat、stat_miss_bonus、gap_direct_top5 三轨）；
- 从 9 个号中生成 **3 组 5 码**（启发式缩水，非严格矩阵）；
- 每组 5 码搭配 **12 个后区全包**（66 注）；
- **单期成本**：3 × 132 = **396 元**；
- **触发条件**：至少一组前区命中 ≥ 3；
- **预期回收**（触发时）：约 725 元（3+2: 200, 3+1: 15×20=300, 3+0: 5×45=225）；
- **净收益**（触发时）：725 - 396 = **329 元**；
- **非触发时**：净亏损 396 元。

> ⚠️ 该方案**不是**数学保证盈利。必须通过严格回测验证触发频率 × 单次回收是否覆盖总成本。

---

## 5. P2 候选池初步设计

暂定 9 个前区候选号来源比例：

| 来源 | 数量 | 依据 |
|------|:---:|------|
| `rf_stat` 高分号 | **4** | 近期窗口表现突出（hit3+ 92%ile@200期） |
| `stat_miss_bonus` 高分号 | **3** | 跨窗口命中率最稳定（hit3+ 88.8%ile@500期） |
| `gap_direct_top5` 高分号 | **2** | 长窗口 roi_truncated 最优（保留互补信号） |

如有重复：按综合分继续补足到 9 个。

**该比例不是最终定论。** P1-7 Rolling Window 的目的之一就是判断：

- 如果 rf_stat 跨窗口更稳定 → 提高 rf_stat 权重；
- 如果 stat_miss_bonus 在 hit3+ 上更稳定 → 提高 miss_bonus 权重；
- 如果 gap 只在部分长窗口有效 → 保留但降低权重；
- 如果某策略只在单窗口有效 → 不能作为主来源。

---

## 6. P1-7 滚动窗口验证任务

在正式进入 P2 前，必须先完成：

### 实验设计

| 参数 | 值 |
|------|-----|
| `rolling_window` | 200 期 |
| `rolling_step` | 100 期 |
| `rolling_random_trials` | 100 |
| 窗口总数 | ~26 个（覆盖 issue 8000→26000） |

### 策略

| 策略 | 说明 |
|------|------|
| `rf_stat_default` | RF:stat=0.65:0.35 融合，仅在无泄漏窗口运行 |
| `stat_miss_bonus_default` | 频次加权 + 0.20×遗漏加分 |
| `gap_direct_top5` | Top-5 遗漏值最大，no_filters，单人下单 |
| `random` | 每窗口 100 trials 随机基线 |

### 输出

| 文件 | 内容 |
|------|------|
| `outputs/rolling_window_summary.csv` | 每窗口每策略一行 |
| `outputs/rolling_window_summary.json` | 同上 JSON |
| `outputs/rolling_window_rankings.json` | 冠军统计 + 超 random 窗口数 |

### 待回答

1. rf_stat 是否只在近期窗口有效？
2. gap_direct_top5 是否只在长窗口整体表现好？
3. stat_miss_bonus 是否比 gap 更稳定？
4. 哪个策略在最多窗口中 roi_truncated 最优？
5. 哪个策略在最多窗口中 hit3+ 最优？
6. 是否有足够依据进入 P2？
7. P2 的 9 候选池比例 4:3:2 是否需要调整？

---

## 7. 当前明确不要做的事

在 P1-7 完成前，**禁止**：

- 9 候选池工程实现
- 3 组 5+12 缩水工程
- N+12 严格矩阵覆盖
- 马丁倍投
- 深度学习模型
- Streamlit UI 更新
- 权重网格搜索
- 任何"稳定盈利"宣传文案

---

## 8. 推荐路线

```
P1-7：Rolling Window 稳定性验证          ← 当前
    ↓
P1-8：P1 总结与信号稳定性结论             ← 收尾
    ↓
P2-1：9候选池生成器（三轨融合定价）       ← 进入 P2
    ↓
P2-2：3组5+12 启发式缩水                 
    ↓
P2-3：n9_3x_5plus12 回测 + random baseline
    ↓
P2-4：加入信号门槛与跳过机制（低信号期不下单）
    ↓
P3：N+12 矩阵缩水与覆盖率验证
    ↓
P4：风控看板与产品化（Streamlit 增强）
```

---

*文档结束*
