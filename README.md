# 大乐透 5+12 策略预测项目（LSTM + RF + 统计融合）

> 说明：本项目用于算法研究与回测验证，不保证实际中奖与盈利。彩票具有强随机性，请理性参与。

## 1. 项目简介

本项目在原始大乐透预测流程基础上，增加了完整的 **5+12 实战策略链路**：

- 前区不再只给 5 个号码，而是先生成 **Top-N 候选池**（默认 10）
- 从候选池组合 `C(N,5)`，并应用规则过滤（奇偶、大小、和值、重号）
- 后区固定执行 **1~12 全包（66 注）**
- 支持 **LSTM + RF + 统计特征** 融合评分
- 支持 **历史滚动回测 + 网格调参 + 最优参数自动回写 config.py**

---

## 2. 当前目录结构（核心文件）

- `get_data.py`：抓取大乐透/双色球历史数据
- `run_train_model.py`：原始 LSTM(+CRF) 训练
- `run_predict.py`：原始单组预测
- `feature_engineering.py`：大乐透特征工程（频次、遗漏、结构特征）
- `run_train_rf_model.py`：前区 RF 训练脚本
- `inference_plus.py`：5+12 推理脚本（融合评分 + 组合过滤 + 注单导出）
- `backtest_plus.py`：5+12 回测与网格搜索脚本（支持自动回写最优参数）
- `config.py`：统一参数配置

---

## 3. 环境安装

建议 Python 3.9~3.11（Windows / Linux 都可）。

```bash
pip install -r requirements.txt
```

如果你只想跑新策略，至少需要这些关键依赖：

- `pandas`
- `numpy`
- `scikit-learn`
- `loguru`

> 注：原始 LSTM 相关功能依赖 TensorFlow 及 `tensorflow-addons`，请按 `requirements.txt` 安装。

---

## 4. 快速开始（推荐顺序）

### 第一步：抓取大乐透数据

```bash
python get_data.py --name dlt
```

输出数据文件：`data/dlt/data.csv`

---

### 第二步：训练原始 LSTM 模型（可选但推荐）

```bash
python run_train_model.py --name dlt --train_test_split 0.8
```

模型输出目录：

- `model/dlt/red_ball_model/`
- `model/dlt/blue_ball_model/`

---

### 第三步：训练 RF 前区模型

```bash
python run_train_rf_model.py --name dlt --train_test_split 0.8 --min_history 120
```

模型输出：

- `model/dlt/rf_front_model.pkl`
- `model/dlt/rf_front_meta.json`

---

### 第四步：执行 5+12 推理

```bash
python inference_plus.py --name dlt --use_lstm 1 --use_rf 1 --save 1
```

说明：

- `--use_lstm 1`：启用 LSTM 代理分
- `--use_rf 1`：启用 RF 分
- `--save 1`：保存结果文件

输出文件（`outputs/`）：

- `inference_plus_<期号>.json`
- `front_combos_<期号>.csv`
- `tickets_<期号>.csv`

其中 `tickets_<期号>.csv` 为最佳前区组合对应的后区全包注单（66 注，132 元）。

---

### 第五步：执行回测 + 网格调参

```bash
python backtest_plus.py --name dlt --start_offset 200 --run_grid 1 --use_rf 1 --save_grid 1
```

参数说明：

- `--start_offset`：从最近多少期开始回测
- `--run_grid 1`：开启网格搜索
- `--use_rf 1`：回测时启用 RF 融合
- `--save_grid 1`：保存网格结果到 `outputs/backtest_grid_results.json`

---

### 第六步：自动回写最优参数到 `config.py`

```bash
python backtest_plus.py --name dlt --start_offset 200 --run_grid 1 --use_rf 1 --auto_apply_best 1
```

会自动把网格最优参数写回：

- `plus_strategy["dlt"]["top_n_front"]`
- `plus_strategy["dlt"]["play_front_combos"]`
- `plus_strategy["dlt"]["ensemble_weights"]`

---

## 5. 配置说明（`config.py`）

主要关注：`plus_strategy["dlt"]`

- `top_n_front`：前区候选池大小（建议 8~10）
- `max_front_combos`：候选前区组合排名保留上限
- `play_front_combos`：每期实际下注前区组合数（每组配 66 注后区）
- `rule_filters`：组合过滤规则
- `score_windows`：统计特征窗口（短/中/长）
- `ensemble_weights`：融合权重（`lstm/rf/stat`）
- `payouts`：3+2 / 3+1 / 3+0 奖金近似（用于策略收益估算）

RF 参数位于：`rf_args["dlt"]`

---

## 6. 融合策略逻辑

在 `inference_plus.py` 中，每个前区号码（1~35）得到三类分数：

1. `lstm`：LSTM 5 码预测映射成 35 维代理分
2. `rf`：RF 对每个号码是否出现的概率分
3. `stat`：基于频次、遗漏、窗口统计的分数

最终融合：

\[
score = w_{lstm} \cdot s_{lstm} + w_{rf} \cdot s_{rf} + w_{stat} \cdot s_{stat}
\]

然后取 Top-N，组合成前区 5 码并过滤。

---

## 7. 常见问题

### Q1：为什么回测收益可能为负？
彩票本质是高随机过程，历史拟合不代表未来收益，负收益是常见情况。

### Q2：没有 RF 模型能否运行？
可以。`inference_plus.py` 在找不到 RF 模型时会自动退化为 LSTM+统计分。

### Q3：双色球是否支持新策略脚本？
目前 `inference_plus.py` 与 `backtest_plus.py` 仅支持 `dlt`。

---

## 8. 免责声明

本仓库仅用于技术研究与学习交流，不构成任何投资/投注建议。请遵守当地法律法规，理性参与。