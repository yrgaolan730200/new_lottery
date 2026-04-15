# 大乐透 5+12 策略预测项目（LSTM + RF + 统计融合）

> 说明：本项目用于算法研究与回测验证，不保证实际中奖与盈利。彩票具有强随机性，请理性参与。

## 1. 项目简介

本项目在原始大乐透预测流程基础上，增加了完整的 **5+12 实战策略链路**：

- 前区不再只给 5 个号码，而是先生成 **Top-N 候选池**（默认 10）
- 从候选池组合 `C(N,5)`，并应用规则过滤（奇偶、大小、和值、重号）
- 后区固定执行 **1~12 全包（66 注）**
- 支持 **LSTM + RF + 统计特征** 融合评分
- 支持 **历史滚动回测 + 网格调参 + 最优参数自动回写 config.py**
- 新增 **Streamlit Web 页面**，可视化操作全流程

---

## 2. 当前目录结构（核心文件）

- `get_data.py`：抓取大乐透/双色球历史数据
- `run_train_model.py`：原始 LSTM(+CRF) 训练
- `run_predict.py`：原始单组预测
- `feature_engineering.py`：大乐透特征工程（频次、遗漏、结构特征）
- `run_train_rf_model.py`：前区 RF 训练脚本
- `inference_plus.py`：5+12 推理脚本（融合评分 + 组合过滤 + 注单导出）
- `backtest_plus.py`：5+12 回测与网格搜索脚本（支持自动回写最优参数）
- `app_streamlit.py`：Streamlit Web 应用入口
- `config.py`：统一参数配置

---

## 3. 环境安装

建议 Python 3.8（你当前 `lottery` 环境就是 3.8.20）。

```bash
pip install -r requirements.txt
```

如果 `scikit-learn==1.5.2` 在你的环境安装失败，可改用：

```bash
conda install -c conda-forge scikit-learn=1.3.2 -y
```

---

## 4. 命令行运行流程（CLI）

### 第一步：抓取大乐透数据

```bash
python get_data.py --name dlt
```

输出数据文件：`data/dlt/data.csv`

### 第二步：训练原始 LSTM 模型（可选但推荐）

```bash
python run_train_model.py --name dlt --train_test_split 0.8
```

### 第三步：训练 RF 前区模型

```bash
python run_train_rf_model.py --name dlt --train_test_split 0.8 --min_history 120
```

输出：

- `model/dlt/rf_front_model.pkl`
- `model/dlt/rf_front_meta.json`

### 第四步：执行 5+12 推理

```bash
python inference_plus.py --name dlt --use_lstm 1 --use_rf 1 --save 1
```

输出（`outputs/`）：

- `inference_plus_<期号>.json`
- `front_combos_<期号>.csv`
- `tickets_<期号>.csv`

### 第五步：回测 + 网格

```bash
python backtest_plus.py --name dlt --start_offset 200 --run_grid 1 --use_rf 1 --save_grid 1
```

### 第六步：自动回写最优参数（带保护开关）

```bash
python backtest_plus.py --name dlt --start_offset 200 --run_grid 1 --use_rf 1 --auto_apply_best 1 --apply_only_if_better 1
```

含义：仅当网格最优 `profit` 优于当前配置时才回写 `config.py`。

---

## 5. Web 版运行流程（Streamlit）

### 5.1 启动

```bash
streamlit run app_streamlit.py
```

浏览器会自动打开（默认 `http://localhost:8501`）。

### 5.2 页面功能

- **数据更新**：一键执行 `get_data.py --name dlt`
- **模型训练**：
  - 训练 LSTM
  - 训练 RF
- **5+12预测**：
  - 勾选是否启用 LSTM/RF
  - 展示目标期号、Top候选、最佳前区5码、总注数和成本
  - 展示注单明细
- **回测调参**：
  - 设置回测窗口
  - 启用/关闭 RF
  - 网格搜索
  - 自动回写最优参数
  - 仅优于当前参数才回写（保护开关）

---

## 6. 配置说明（`config.py`）

主要看 `plus_strategy["dlt"]`：

- `top_n_front`：前区候选池大小（建议 8~10）
- `max_front_combos`：候选组合保留上限
- `play_front_combos`：每期实际下注的前区组合数
- `rule_filters`：组合过滤规则
- `score_windows`：统计特征窗口
- `ensemble_weights`：融合权重（`lstm/rf/stat`）
- `payouts`：3+2 / 3+1 / 3+0 奖金近似

---

## 7. 免责声明

本仓库仅用于技术研究与学习交流，不构成任何投资/投注建议。请遵守当地法律法规，理性参与。