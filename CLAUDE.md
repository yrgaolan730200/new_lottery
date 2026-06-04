# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

大乐透预测研究项目，使用 **LSTM+CRF**、**Random Forest** 和**统计特征**三者融合，通过"5+12策略"（前区5码 + 后区1~12全包共66注）生成投注推荐。包含 **Streamlit Web 界面**和完整流水线的 CLI 脚本。

## 运行环境

- Python 3.8（conda 环境 `lottery`）
- TensorFlow 1.x 兼容模式（graph/session API，全局禁用 eager execution）

```bash
pip install -r requirements.txt
# 如果 scikit-learn==1.5.2 安装失败，改用：
conda install -c conda-forge scikit-learn=1.3.2 -y
```

## 常用命令

### 数据抓取

```bash
python get_data.py --name dlt          # 抓取大乐透历史数据 → data/dlt/data.csv
```

### 模型训练

```bash
python run_train_model.py --name dlt --train_test_split 0.8          # 训练 LSTM+CRF
python run_train_rf_model.py --name dlt --train_test_split 0.8 --min_history 120  # 训练 RF
```

### 预测推理（5+12）

```bash
python inference_plus.py --name dlt --use_lstm 1 --use_rf 1 --save 1
# 输出：outputs/inference_plus_<期号>.json、front_combos_<期号>.csv、tickets_<期号>.csv
```

### 回测与网格搜索

```bash
# 回测 + 网格搜索
python backtest_plus.py --name dlt --start_offset 200 --run_grid 1 --use_rf 1 --save_grid 1
# 回测 + 网格搜索 + 自动回写最优参数（仅当优于当前配置时）
python backtest_plus.py --name dlt --start_offset 200 --run_grid 1 --use_rf 1 --auto_apply_best 1 --apply_only_if_better 1
```

### Web 界面

```bash
streamlit run app_streamlit.py          # 启动后访问 http://localhost:8501
```

## 架构

### 流水线顺序（严格按此执行）

1. **`get_data.py`** — 从 500.com 抓取彩票历史数据，写入 `data/dlt/data.csv`（列：期数、红球_1…5、蓝球_1…2）
2. **`run_train_model.py`** — 分别训练红球和蓝球的 LSTM+CRF 模型，保存 TF checkpoint 到 `model/dlt/`
3. **`run_train_rf_model.py`** — 基于前区特征训练多输出 Random Forest，保存 `model/dlt/rf_front_model.pkl` 和元信息 JSON
4. **`inference_plus.py`** — 融合推理：将 LSTM 代理分 + RF 概率 + 统计分融合为 35 维排序向量，生成 Top-N 候选池，按奇偶/大小/和值/重号规则过滤组合，最终输出注单
5. **`backtest_plus.py`** — 滚动窗口回测 + 对 `(top_n_front, play_front_combos, ensemble_weights)` 做网格搜索，可自动将最优参数回写到 `config.py`

### 核心模块

| 文件 | 职责 |
|---|---|
| `config.py` | 所有参数：LSTM 模型参数、RF 参数、5+12 策略（候选池大小、过滤规则、融合权重、奖金） |
| `modeling.py` | `LstmWithCRFModel`（LSTM + CRF 解码，用于序列预测）和 `SignalLstmModel`（简易 LSTM，用于双色球蓝球） |
| `feature_engineering.py` | RF 特征构建：频次、遗漏、上期结构特征、重号比率。同时提供融合评分用的 `build_lstm_proxy_scores()` 和 `calc_stat_scores()` |
| `run_predict.py` | 加载已训练的 LSTM checkpoint 并执行原始预测。由 `inference_plus.py` 动态导入（通过保护 `sys.argv` 避免参数冲突） |
| `app_streamlit.py` | Streamlit 界面，通过 `subprocess.run` 调用所有脚本。共 5 个标签页：规则介绍、数据更新、模型训练、5+12预测、回测调参 |

### 融合评分公式

```
score(n) = w_lstm × lstm_score(n) + w_rf × rf_score(n) + w_stat × stat_score(n)
```

- **LSTM 分**：将 LSTM 预测的 5 个前区号码按排名权重映射为 35 维向量
- **RF 分**：每个号码的二分类正类概率（多输出分类器的 `predict_proba`）
- **统计分**：短/中/长期窗口频次的加权组合减去遗漏惩罚

### 5+12 策略机制

- **前区（1~35）**：按融合分选出 Top-N 候选号码，生成 `C(N,5)` 个组合，经奇偶/大小/和值/重号规则过滤，取评分最高的组合
- **后区（1~12）**：始终全包 — 共 `C(12,2) = 66` 种配对
- 每张票 = 1 组前区 + 66 组后区 = 66 注 × 2元 = 132元

### 重要约束

- **TF1 graph/session**：所有涉及 TF 的脚本必须在任何其他 TF 操作之前调用 `tf.compat.v1.disable_eager_execution()`。`run_predict.py` 在模块级别执行此操作 — 因此 `inference_plus.py` 通过保护 `sys.argv` 来动态导入它
- **数据排序方向不一致**：`feature_engineering.py` 按升序排列（`sort_values("期数")`），而 `inference_plus.py` 按降序排列（`sort_values("期数", ascending=False)`）— 注意不要混淆排序方向
- **仅 DLT 完全支持** 5+12 流水线；SSQ（双色球）仅有基础的 LSTM 训练/预测功能，没有 `inference_plus` 或 `backtest_plus` 支持
