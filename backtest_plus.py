# -*- coding:utf-8 -*-
"""
大乐透 5+12 策略回测与调参（重构版）

改进：
- 统一使用 DLT_PRIZE_TABLE 逐注计奖（不再仅处理 front_hit==3）
- RF 支持 none / static / walk_forward 三种模式，默认 walk_forward
- static 模式打印泄漏风险警告
- 移除 LSTM proxy，回测仅使用 RF + stat 融合
- 新增 3 种基线策略：随机、热号、冷号遗漏
- 统一输出 backtest_report.csv
"""
import os
import re
import json
import argparse
import itertools
import random
import warnings

import numpy as np
import pandas as pd
from loguru import logger

from config import (
    name_path, data_file_name, plus_strategy, rf_args, get_dlt_prize_table,
)
from feature_engineering import (
    load_dlt_history,
    build_front_training_dataset,
    build_feature_for_next_issue,
    calc_stat_scores,
    calc_front_frequency,
    calc_front_missing,
)
from inference_plus import (
    build_ensemble_scores,
    generate_front_combos,
    maybe_load_rf_model,
    predict_rf_scores,
)


FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK_COLS = ["蓝球_1", "蓝球_2"]

# ============================================================
#  奖金计算 —— 使用 DLT_PRIZE_TABLE 逐注计奖
# ============================================================

def calc_5_plus_12_reward(front_combo, target_front_set, target_back_set, issue):
    """计算一组前区组合 + 后区全包（66注）的总奖金。

    根据 issue 自动选择对应的奖金表（区分 26014 前后旧/新规则）。
    遍历全部 66 个后区组合逐注计奖，不遗漏任何 (front_hit, back_hit) 组合。

    Args:
        front_combo: 前区5码组合 (iterable)
        target_front_set: 实际开奖前区 (set)
        target_back_set: 实际开奖后区 (set)
        issue: 目标期号（int），用于确定适用奖金表

    Returns:
        (total_reward, front_hit): 总奖金（元），前区命中数
    """
    prize_table = get_dlt_prize_table(issue)

    front_hit = len(set(front_combo).intersection(target_front_set))
    total_reward = 0

    # 遍历后区全包 66 组配对（1~12 选 2）
    for b1 in range(1, 13):
        for b2 in range(b1 + 1, 13):
            back_hit = len({b1, b2}.intersection(target_back_set))
            prize = prize_table.get((front_hit, back_hit), 0)
            total_reward += prize

    return total_reward, front_hit


# ============================================================
#  RF 训练辅助
# ============================================================

def _train_rf_on_history(history_df_asc, windows, min_history):
    """在给定历史上训练 RF 模型（用于 walk_forward 模式）。"""
    if len(history_df_asc) <= min_history:
        return None
    x_train, y_train, _ = build_front_training_dataset(
        history_df_asc, windows=windows, min_history=min_history
    )
    if len(x_train) == 0:
        return None

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier

    cfg = rf_args["dlt"]
    base = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        random_state=cfg["random_state"],
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base)
    model.fit(x_train, y_train)
    return model


# ============================================================
#  可变评分模式 —— 支持 gap 方向实验
# ============================================================

def _compute_stat_miss_penalty(hist, windows):
    """当前逻辑：短中长期频次加权 - 0.20 * miss（遗漏惩罚）。"""
    return calc_stat_scores(hist, windows=windows)


def _compute_stat_no_miss(hist, windows):
    """只用频次，不使用遗漏值。"""
    short, mid, long = windows
    s_freq = calc_front_frequency(hist, short)
    m_freq = calc_front_frequency(hist, mid)
    l_freq = calc_front_frequency(hist, long)
    raw = 0.45 * s_freq + 0.35 * m_freq + 0.20 * l_freq
    return _safe_normalize(raw)


def _compute_stat_miss_bonus(hist, windows):
    """频次加权 + 0.20 * miss（遗漏加分：越久未出越高分）。"""
    short, mid, long = windows
    s_freq = calc_front_frequency(hist, short)
    m_freq = calc_front_frequency(hist, mid)
    l_freq = calc_front_frequency(hist, long)
    miss = calc_front_missing(hist)
    raw = 0.45 * s_freq + 0.35 * m_freq + 0.20 * l_freq + 0.20 * miss
    return _safe_normalize(raw)


def _compute_gap_only_scores(hist, windows=None):
    """纯遗漏值评分。

    使用原始遗漏值 + 微小号码偏移作为次级排序键。
    score[n] = gap[n] + n/1000.0

    这样当两个号码遗漏值相同时，号码大的优先级更高，
    确保 argsort 和 combinations 的 tie-breaking 一致。
    偏移量 (1/1000) 远小于最小遗漏差 (1)，不会改变主排序。
    """
    n_front = 35
    miss = np.zeros(n_front, dtype=float)
    if len(hist) == 0:
        return miss
    max_back = len(hist) + 1
    for n in range(1, n_front + 1):
        gap = max_back
        for back_idx, (_, row) in enumerate(hist.iloc[::-1].iterrows(), start=1):
            if n in {int(row[c]) for c in FRONT_COLS}:
                gap = back_idx
                break
        miss[n - 1] = float(gap) + n / 1000.0
    return miss


def _compute_freq_only_scores(hist, windows):
    """纯频次评分（用于 rf_stat_gap 的 freq 组件）。"""
    short, mid, long = windows
    s_freq = calc_front_frequency(hist, short)
    m_freq = calc_front_frequency(hist, mid)
    l_freq = calc_front_frequency(hist, long)
    raw = 0.45 * s_freq + 0.35 * m_freq + 0.20 * l_freq
    return _safe_normalize(raw)


def _safe_normalize(v):
    """安全归一化到 [0,1]，处理全零/全等异常。"""
    arr = np.array(v, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# ============================================================
#  核心回测（RF + stat 融合，无 LSTM）
# ============================================================

def run_backtest_core(
    data_asc,
    start_idx,
    end_idx,
    top_n_front,
    max_front_combos,
    play_front_combos,
    ensemble_weights,
    rule_filters,
    rf_mode="walk_forward",
    rf_model_static=None,
    rf_meta=None,
    strategy_name="main",
    score_mode="stat_miss_penalty",
    rng_seed=None,
):
    """执行 5+12 策略滚动回测。

    每期使用 data_asc.iloc[:idx] 作为已知历史，预测第 idx 期。
    选择评分最高的前区组合 + 后区全包（66注），逐注计奖。
    奖金计算根据每期期号自动选择适用的奖级表（26014前后）。

    Args:
        data_asc: 按期号升序的历史数据
        start_idx, end_idx: 回测起止索引
        top_n_front, max_front_combos, play_front_combos: 策略参数
        ensemble_weights: 融合权重
        rule_filters: 组合过滤规则
        rf_mode: 'none' | 'static' | 'walk_forward'
        rf_model_static: static 模式的预训练 RF 模型
        rf_meta: RF 元信息
        strategy_name: 策略标识（输出到报表）
        rng_seed: 随机种子（用于可复现）

    Returns:
        dict: 汇总统计 + records 列表
    """
    strategy_cfg = plus_strategy["dlt"]
    score_windows = (
        int(strategy_cfg["score_windows"]["short"]),
        int(strategy_cfg["score_windows"]["mid"]),
        int(strategy_cfg["score_windows"]["long"]),
    )
    min_history = int(rf_meta["min_history"]) if rf_meta else 120

    # 临时覆盖策略参数
    old_top_n = strategy_cfg["top_n_front"]
    old_max = strategy_cfg["max_front_combos"]
    old_weights = strategy_cfg["ensemble_weights"].copy()
    old_rules = strategy_cfg["rule_filters"].copy()

    strategy_cfg["top_n_front"] = int(top_n_front)
    strategy_cfg["max_front_combos"] = int(max_front_combos)
    strategy_cfg["ensemble_weights"] = ensemble_weights
    strategy_cfg["rule_filters"] = rule_filters

    records = []
    total_cost = 0
    total_reward = 0

    try:
        for idx in range(start_idx, end_idx + 1):
            hist = data_asc.iloc[:idx]         # 已知历史（不含目标期）
            target = data_asc.iloc[idx]        # 目标期

            last_front_set = set(int(hist.iloc[-1][c]) for c in FRONT_COLS)
            target_front_set = set(int(target[c]) for c in FRONT_COLS)
            target_back_set = set(int(target[c]) for c in BACK_COLS)

            # --- RF 模型 ---
            rf_model = None
            if rf_mode == "static":
                rf_model = rf_model_static
            elif rf_mode == "walk_forward":
                rf_model = _train_rf_on_history(hist, score_windows, min_history)

            # --- 融合评分（无 LSTM，按 score_mode 切换）---
            if score_mode == "stat_miss_penalty":
                stat_scores = _compute_stat_miss_penalty(hist, score_windows)
            elif score_mode == "stat_no_miss":
                stat_scores = _compute_stat_no_miss(hist, score_windows)
            elif score_mode == "stat_miss_bonus":
                stat_scores = _compute_stat_miss_bonus(hist, score_windows)
            elif score_mode == "gap_only":
                stat_scores = _compute_gap_only_scores(hist)
            else:
                stat_scores = _compute_stat_miss_penalty(hist, score_windows)

            if rf_model is not None and len(hist) >= min_history:
                rf_scores = predict_rf_scores(rf_model, hist,
                                              windows=score_windows,
                                              min_history=min_history)
            else:
                rf_scores = np.zeros(35, dtype=float)

            # rf_stat_gap: RF + freq + gap 三组件融合
            gap_scores = np.zeros(35, dtype=float)
            freq_only = np.zeros(35, dtype=float)
            if score_mode == "rf_stat_gap":
                gap_scores = _compute_gap_only_scores(hist)
                freq_only = _compute_freq_only_scores(hist, score_windows)

            # LSTM 分置零（回测中不运行 LSTM 推理）
            lstm_scores = np.zeros(35, dtype=float)

            w = _normalize_weights_no_lstm(ensemble_weights)

            if score_mode == "rf_stat_gap":
                # 三组件融合：rf + freq + gap
                # 使用 ensemble_weights 传递特殊权重:
                # lstm槽 → gap权重, rf槽 → rf权重, stat槽 → freq权重
                ensemble = (w.get("lstm", 0.0) * gap_scores +
                            w.get("rf", 0.50) * rf_scores +
                            w.get("stat", 0.30) * freq_only)
                # 确保 rf_stat_gap 权重正确（不被 _normalize_weights_no_lstm 改动）
                _w_gap = ensemble_weights.get("lstm", 0.20)
                _w_rf = ensemble_weights.get("rf", 0.50)
                _w_freq = ensemble_weights.get("stat", 0.30)
                _total = _w_gap + _w_rf + _w_freq
                if _total > 0:
                    ensemble = ((_w_gap / _total) * gap_scores +
                                (_w_rf / _total) * rf_scores +
                                (_w_freq / _total) * freq_only)
            else:
                ensemble = (w["lstm"] * lstm_scores +
                            w["rf"] * rf_scores +
                            w["stat"] * stat_scores)

            # --- 生成前区组合 ---
            _, combos = generate_front_combos(ensemble, last_front_set)

            if len(combos) == 0:
                continue

            played = combos[:max(1, int(play_front_combos))]
            period_cost = 132 * len(played)  # 每组前区配 66 注后区 = 132 元
            period_reward = 0
            best_front_hit = 0
            selected_fronts = []

            for combo, _ in played:
                reward, fh = calc_5_plus_12_reward(
                    combo, target_front_set, target_back_set, int(target["期数"])
                )
                period_reward += reward
                best_front_hit = max(best_front_hit, fh)
                selected_fronts.append(
                    " ".join(["{:02d}".format(x) for x in sorted(combo)])
                )

            period_profit = period_reward - period_cost
            total_cost += period_cost
            total_reward += period_reward

            actual_front = " ".join(
                ["{:02d}".format(int(target[c])) for c in FRONT_COLS]
            )

            records.append({
                "issue": int(target["期数"]),
                "strategy_name": strategy_name,
                "selected_front": " | ".join(selected_fronts),
                "actual_front": actual_front,
                "front_hit": int(best_front_hit),
                "cost": int(period_cost),
                "reward": int(period_reward),
                "profit": int(period_profit),
                "cumulative_profit": int(total_reward - total_cost),
            })
    finally:
        strategy_cfg["top_n_front"] = old_top_n
        strategy_cfg["max_front_combos"] = old_max
        strategy_cfg["ensemble_weights"] = old_weights
        strategy_cfg["rule_filters"] = old_rules

    metrics = compute_strategy_metrics(records)
    metrics["records"] = records
    return metrics


def _normalize_weights_no_lstm(weights):
    """归一化权重，并将 LSTM 权重按比例重新分配给 RF 和 stat。

    注意：必须在更新 rf_w/stat_w 之前计算 base_total，
    否则更新后的 rf_w 会污染 stat_w 的分母，导致分配比例错误。
    """
    lstm_w = weights.get("lstm", 0.0)
    rf_w = weights.get("rf", 0.0)
    stat_w = weights.get("stat", 0.0)

    base_total = rf_w + stat_w  # 必须在更新前保存！

    # LSTM 不可用时，将其权重按比例分配给 RF 和 stat
    if lstm_w > 0 and base_total > 0:
        rf_w += lstm_w * (rf_w / base_total)      # 使用原始 rf_w 占比
        stat_w += lstm_w * (stat_w / base_total)   # 使用原始 stat_w 占比

    lstm_w = 0.0  # LSTM 在回测中不可用
    total = rf_w + stat_w
    if total <= 0:
        return {"lstm": 0.0, "rf": 0.5, "stat": 0.5}
    return {"lstm": 0.0, "rf": rf_w / total, "stat": stat_w / total}


# ============================================================
#  通用指标计算 —— 所有策略统一调用
# ============================================================

def compute_strategy_metrics(records):
    """从逐期 records 计算全套回测评估指标。

    所有策略（main / random / hot_number / gap_only）复用此函数，
    确保指标定义和计算方式完全一致。

    Args:
        records: list of dict，每期一条，至少包含:
            strategy_name, cost, reward, profit, cumulative_profit, front_hit

    Returns:
        dict: 包含 periods, total_cost, total_reward, total_profit, roi,
              avg_front_hit, hit2_or_more_rate, hit3_or_more_rate,
              max_drawdown, max_consecutive_loss, avg_period_profit, profit_std
    """
    # 截断奖励阈值：单期 reward > cap 时截断，排除极端头奖污染
    _REWARD_CAP = 10000

    if not records:
        return {
            "strategy_name": "unknown",
            "periods": 0,
            "total_cost": 0, "total_reward": 0, "total_profit": 0, "roi": 0.0,
            "total_reward_truncated": 0, "total_profit_truncated": 0, "roi_truncated": 0.0,
            "avg_front_hit": 0.0,
            "hit0_rate": 0.0, "hit1_rate": 0.0, "hit2_rate": 0.0,
            "hit3_rate": 0.0, "hit4_rate": 0.0, "hit5_rate": 0.0,
            "hit2_or_more_rate": 0.0, "hit3_or_more_rate": 0.0,
            "max_drawdown": 0.0, "max_consecutive_loss": 0,
            "avg_period_profit": 0.0, "avg_period_profit_truncated": 0.0,
            "profit_std": 0.0,
        }

    df = pd.DataFrame(records)
    periods = len(df)
    total_cost = int(df["cost"].sum())
    total_reward = int(df["reward"].sum())
    total_profit = int(df["profit"].sum())
    roi = float(total_profit / total_cost) if total_cost > 0 else 0.0

    # --- 截断收益（排除极端头奖污染）---
    reward_truncated = np.minimum(df["reward"].values.astype(float), _REWARD_CAP)
    total_reward_truncated = int(reward_truncated.sum())
    total_profit_truncated = int(total_reward_truncated - total_cost)
    roi_truncated = float(total_profit_truncated / total_cost) if total_cost > 0 else 0.0
    avg_period_profit_truncated = float(total_profit_truncated / periods) if periods > 0 else 0.0

    # --- 前区命中分布（0..5 精确计数）---
    hit_counts = df["front_hit"].value_counts().to_dict()
    hit0_rate = float(hit_counts.get(0, 0) / periods)
    hit1_rate = float(hit_counts.get(1, 0) / periods)
    hit2_rate = float(hit_counts.get(2, 0) / periods)
    hit3_rate = float(hit_counts.get(3, 0) / periods)
    hit4_rate = float(hit_counts.get(4, 0) / periods)
    hit5_rate = float(hit_counts.get(5, 0) / periods)
    avg_front_hit = float(df["front_hit"].mean())
    hit2_or_more = hit2_rate + hit3_rate + hit4_rate + hit5_rate
    hit3_or_more = hit3_rate + hit4_rate + hit5_rate

    # 最大回撤：cumulative_profit 曲线从历史最高点的最大跌幅
    cum = df["cumulative_profit"].values.astype(float)
    peak = np.maximum.accumulate(cum)
    max_drawdown = float((peak - cum).max()) if len(cum) > 0 else 0.0

    # 最长连续亏损期数：profit < 0 为亏损，profit >= 0 中断
    profit_arr = df["profit"].values
    max_loss_streak = 0
    current_streak = 0
    for p in profit_arr:
        if p < 0:
            current_streak += 1
            max_loss_streak = max(max_loss_streak, current_streak)
        else:
            current_streak = 0

    avg_period_profit = float(total_profit / periods) if periods > 0 else 0.0
    profit_std = float(df["profit"].std(ddof=0)) if periods > 1 else 0.0

    name = records[0].get("strategy_name", "unknown") if records else "unknown"

    return {
        "strategy_name": name,
        "periods": periods,
        "total_cost": total_cost,
        "total_reward": total_reward,
        "total_profit": total_profit,
        "roi": roi,
        # 截断收益（reward_cap=10000，排除5+2/5+1等极端事件）
        "total_reward_truncated": total_reward_truncated,
        "total_profit_truncated": total_profit_truncated,
        "roi_truncated": roi_truncated,
        "avg_period_profit_truncated": avg_period_profit_truncated,
        # 前区命中分布
        "avg_front_hit": avg_front_hit,
        "hit0_rate": hit0_rate, "hit1_rate": hit1_rate,
        "hit2_rate": hit2_rate, "hit3_rate": hit3_rate,
        "hit4_rate": hit4_rate, "hit5_rate": hit5_rate,
        "hit2_or_more_rate": hit2_or_more,
        "hit3_or_more_rate": hit3_or_more,
        # 风险指标
        "max_drawdown": max_drawdown,
        "max_consecutive_loss": max_loss_streak,
        "avg_period_profit": avg_period_profit,
        "profit_std": profit_std,
    }


# ============================================================
#  基线策略
# ============================================================

def run_random_baseline(data_asc, start_idx, end_idx,
                        n_trials=1, rng_seed=42):
    """随机基线：每期随机选 5 个前区号码 + 后区全包。

    运行 n_trials 次独立试验：
    1. 每个 trial 独立生成随机号码，计算完整的逐期 records
    2. 每个 trial 调用 compute_strategy_metrics 得到完整指标
    3. 所有 trial 的指标取均值作为最终结果
    4. n_trials > 1 时额外计算各指标的标准差

    注意：hit3_or_more_rate 等比例指标的正确计算方式是
    "先在每个 trial 内计算比例，再对 trials 求均值/std"，
    而非"把所有 trial 的前区命中平均后再判断 >=3"。

    奖金计算自动根据每期期号选择适用的奖级表（26014前后）。

    Returns:
        dict: 包含所有 metrics 的均值，以及 records（多 trial 均值）。
              当 n_trials>1 时额外包含 _std 后缀的标准差字段。
    """
    rng = random.Random(rng_seed)
    all_trial_metrics = []   # 每个 trial 的完整 metrics
    all_trial_records = []   # 所有 trial 的原始 records（用于合并输出）

    for trial in range(n_trials):
        records = []
        total_cost = 0
        total_reward = 0
        trial_seed = rng_seed + trial * 10000 if rng_seed is not None else None
        trial_rng = random.Random(trial_seed)

        for idx in range(start_idx, end_idx + 1):
            target = data_asc.iloc[idx]
            target_front_set = set(int(target[c]) for c in FRONT_COLS)
            target_back_set = set(int(target[c]) for c in BACK_COLS)

            # 随机选 5 个前区号码
            combo = tuple(sorted(trial_rng.sample(range(1, 36), 5)))
            reward, fh = calc_5_plus_12_reward(
                combo, target_front_set, target_back_set, int(target["期数"])
            )

            period_cost = 132  # 1 组前区 × 66 注后区
            period_profit = reward - period_cost
            total_cost += period_cost
            total_reward += reward

            actual_front = " ".join(
                ["{:02d}".format(int(target[c])) for c in FRONT_COLS]
            )

            records.append({
                "issue": int(target["期数"]),
                "strategy_name": f"random_trial_{trial+1}",
                "selected_front": " ".join(["{:02d}".format(x) for x in combo]),
                "actual_front": actual_front,
                "front_hit": int(fh),
                "cost": int(period_cost),
                "reward": int(reward),
                "profit": int(period_profit),
                "cumulative_profit": int(total_reward - total_cost),
            })

        if len(records) == 0:
            continue

        # 每个 trial 独立计算完整指标（包括 hit2/hit3/drawdown 等）
        trial_metrics = compute_strategy_metrics(records)
        trial_metrics["strategy_name"] = "random"
        all_trial_metrics.append(trial_metrics)
        all_trial_records.append(records)

    if len(all_trial_metrics) == 0:
        empty = compute_strategy_metrics([])
        empty["strategy_name"] = "random"
        empty["records"] = []
        return empty

    # --- 合并逐期 records（按 issue 对多 trial 取平均，用于 CSV 明细输出）---
    merged_df = pd.DataFrame([r for recs in all_trial_records for r in recs])
    summary = merged_df.groupby("issue").agg({
        "cost": "mean", "reward": "mean", "profit": "mean",
        "front_hit": "mean",
    }).reset_index()
    summary["strategy_name"] = "random"
    summary["cumulative_profit"] = summary["profit"].cumsum()
    summary["selected_front"] = merged_df.groupby("issue")["selected_front"].first().values
    summary["actual_front"] = merged_df.groupby("issue")["actual_front"].first().values
    summary["front_hit"] = summary["front_hit"].round(2)
    records_out = summary.to_dict("records")

    # --- 跨 trial 聚合指标：各 trial 独立计算，再取均值 ---
    metric_keys = [
        "periods", "total_cost", "total_reward", "total_profit", "roi",
        "total_reward_truncated", "total_profit_truncated", "roi_truncated",
        "avg_period_profit_truncated",
        "avg_front_hit",
        "hit0_rate", "hit1_rate", "hit2_rate", "hit3_rate",
        "hit4_rate", "hit5_rate",
        "hit2_or_more_rate", "hit3_or_more_rate",
        "max_drawdown", "max_consecutive_loss",
        "avg_period_profit", "profit_std",
    ]
    result = {"strategy_name": "random", "records": records_out}

    for key in metric_keys:
        values = [m[key] for m in all_trial_metrics]
        result[key] = float(np.mean(values))

    # 整数类型保持为 int
    for int_key in ["periods", "total_cost", "total_reward", "total_profit",
                     "total_reward_truncated", "total_profit_truncated",
                     "max_consecutive_loss"]:
        if int_key in result:
            result[int_key] = int(round(result[int_key]))

    # n_trials > 1 时附加标准差
    if n_trials > 1:
        std_keys = [
            "total_profit", "total_profit_truncated",
            "roi", "roi_truncated",
            "avg_front_hit",
            "hit2_or_more_rate", "hit3_or_more_rate",
            "max_drawdown", "max_consecutive_loss",
        ]
        for key in std_keys:
            values = [m.get(key, 0.0) for m in all_trial_metrics]
            if len(values) > 1:
                result[key + "_std"] = float(np.std(values, ddof=1))
            else:
                result[key + "_std"] = 0.0

    # 附加所有 trial 级指标（用于 percentile 计算和输出）
    result["_trial_metrics"] = all_trial_metrics

    return result


def run_hot_number_baseline(data_asc, start_idx, end_idx, lookback=50):
    """热号基线：每期选最近 lookback 期内出现频次最高的 5 个号码 + 后区全包。
    奖金计算自动根据每期期号选择适用的奖级表（26014前后）。
    """
    records = []
    total_cost = 0
    total_reward = 0

    for idx in range(start_idx, end_idx + 1):
        hist = data_asc.iloc[:idx]
        target = data_asc.iloc[idx]
        target_front_set = set(int(target[c]) for c in FRONT_COLS)
        target_back_set = set(int(target[c]) for c in BACK_COLS)

        # 热号：最近 lookback 期频次最高的 5 个号码
        freq = calc_front_frequency(hist.tail(lookback), window_size=lookback)
        top_indices = np.argsort(freq)[::-1][:5]
        combo = tuple(sorted((top_indices + 1).tolist()))

        reward, fh = calc_5_plus_12_reward(
            combo, target_front_set, target_back_set, int(target["期数"])
        )

        period_cost = 132
        period_profit = reward - period_cost
        total_cost += period_cost
        total_reward += reward

        actual_front = " ".join(
            ["{:02d}".format(int(target[c])) for c in FRONT_COLS]
        )

        records.append({
            "issue": int(target["期数"]),
            "strategy_name": "hot_number",
            "selected_front": " ".join(["{:02d}".format(x) for x in combo]),
            "actual_front": actual_front,
            "front_hit": int(fh),
            "cost": int(period_cost),
            "reward": int(reward),
            "profit": int(period_profit),
            "cumulative_profit": int(total_reward - total_cost),
        })

    metrics = compute_strategy_metrics(records)
    metrics["records"] = records
    return metrics


def run_gap_only_baseline(data_asc, start_idx, end_idx):
    """冷号遗漏基线：每期选遗漏值最大（最久未出）的 5 个号码 + 后区全包。
    奖金计算自动根据每期期号选择适用的奖级表（26014前后）。
    """
    records = []
    total_cost = 0
    total_reward = 0

    for idx in range(start_idx, end_idx + 1):
        hist = data_asc.iloc[:idx]
        target = data_asc.iloc[idx]
        target_front_set = set(int(target[c]) for c in FRONT_COLS)
        target_back_set = set(int(target[c]) for c in BACK_COLS)

        # 冷号：遗漏值最大的 5 个号码
        miss = calc_front_missing(hist)
        top_indices = np.argsort(miss)[::-1][:5]
        combo = tuple(sorted((top_indices + 1).tolist()))

        reward, fh = calc_5_plus_12_reward(
            combo, target_front_set, target_back_set, int(target["期数"])
        )

        period_cost = 132
        period_profit = reward - period_cost
        total_cost += period_cost
        total_reward += reward

        actual_front = " ".join(
            ["{:02d}".format(int(target[c])) for c in FRONT_COLS]
        )

        records.append({
            "issue": int(target["期数"]),
            "strategy_name": "gap_only",
            "selected_front": " ".join(["{:02d}".format(x) for x in combo]),
            "actual_front": actual_front,
            "front_hit": int(fh),
            "cost": int(period_cost),
            "reward": int(reward),
            "profit": int(period_profit),
            "cumulative_profit": int(total_reward - total_cost),
        })

    metrics = compute_strategy_metrics(records)
    metrics["records"] = records
    return metrics


# ============================================================
#  网格搜索
# ============================================================

def run_grid_search(data_asc, start_idx, end_idx, base_strategy,
                    rf_mode="walk_forward"):
    """网格搜索最优参数组合。奖金计算自动根据每期期号选择奖级表。"""
    top_n_list = [8, 9, 10]
    play_front_list = [1, 2]
    weight_list = [
        {"lstm": 0.0, "rf": 0.70, "stat": 0.30},
        {"lstm": 0.0, "rf": 0.60, "stat": 0.40},
        {"lstm": 0.0, "rf": 0.50, "stat": 0.50},
        {"lstm": 0.0, "rf": 0.40, "stat": 0.60},
        {"lstm": 0.0, "rf": 0.30, "stat": 0.70},
    ]

    results = []
    for top_n, play_n, weights in itertools.product(top_n_list, play_front_list, weight_list):
        r = run_backtest_core(
            data_asc=data_asc,
            start_idx=start_idx,
            end_idx=end_idx,
            top_n_front=top_n,
            max_front_combos=base_strategy["max_front_combos"],
            play_front_combos=play_n,
            ensemble_weights=weights,
            rule_filters=base_strategy["rule_filters"],
            rf_mode=rf_mode,
            strategy_name=f"grid_{top_n}_{play_n}",
        )
        results.append({
            "top_n_front": top_n,
            "play_front_combos": play_n,
            "weights": weights,
            "profit": r["total_profit"],
            "roi": r["roi"],
            "hit3_or_more_rate": r["hit3_or_more_rate"],
        })

    return sorted(results, key=lambda x: (x["profit"], x["roi"]), reverse=True)


# ============================================================
#  输出 & 参数回写
# ============================================================

def save_backtest_report(all_results, out_path):
    """将多个策略的回测记录合并输出为 CSV，同时保存估算说明元数据。"""
    all_records = []
    for result in all_results:
        all_records.extend(result.get("records", []))

    if not all_records:
        logger.warning("没有回测记录可输出")
        return

    df = pd.DataFrame(all_records)
    # 计算每行的 ROI（累积收益 / 累积成本）
    df["cumulative_cost"] = df.groupby("strategy_name")["cost"].cumsum()
    df["roi"] = np.where(
        df["cumulative_cost"] > 0,
        df["cumulative_profit"] / df["cumulative_cost"],
        0.0
    )
    df["roi"] = df["roi"].round(6)

    # 列顺序
    cols = [
        "issue", "strategy_name", "selected_front", "actual_front",
        "front_hit", "cost", "reward", "profit",
        "cumulative_profit", "cumulative_cost", "roi",
    ]
    df = df[[c for c in cols if c in df.columns]]

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("回测报告已保存: {} ({} 行)".format(out_path, len(df)))

    # 同时保存估算说明元数据
    meta = {
        "disclaimer": (
            "本报告中的 ROI / profit 均为估算值。"
            "一等奖/二等奖使用固定近似值（10,000,000 / 200,000 元），"
            "实际为浮动奖金。"
            "未计入奖池超 8 亿时的额外派奖、节假日加奖等活动。"
            "26014 期起使用 2026 新规则奖金表，此前使用旧规则。"
        ),
        "prize_table_info": {
            "reform_issue": 26014,
            "floating_prizes_approximate": True,
            "bonus_events_excluded": True,
            "differences_pre_post_2026": {
                "(3,1)": "10 → 15 元",
                "(2,2)": "10 → 15 元",
            },
        },
        "strategies": [r["strategy_name"] for r in all_results],
        "total_periods_per_strategy": len(all_results[0].get("records", [])),
    }
    meta_path = out_path.replace(".csv", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info("估算说明已保存: {}".format(meta_path))


def save_backtest_summary(all_results, out_path):
    """将各策略的汇总指标输出为 JSON（每个策略一行）。

    对 random baseline，如果存在 _std 后缀字段，一并输出。
    """
    summary_rows = []
    metric_fields = [
        "strategy_name", "periods",
        "total_cost", "total_reward", "total_profit", "roi",
        "total_reward_truncated", "total_profit_truncated", "roi_truncated",
        "avg_front_hit",
        "hit0_rate", "hit1_rate", "hit2_rate", "hit3_rate",
        "hit4_rate", "hit5_rate",
        "hit2_or_more_rate", "hit3_or_more_rate",
        "max_drawdown", "max_consecutive_loss",
        "avg_period_profit", "avg_period_profit_truncated",
        "profit_std",
    ]
    std_suffixes = [
        "total_profit_std", "total_profit_truncated_std",
        "roi_std", "roi_truncated_std",
        "avg_front_hit_std",
        "hit2_or_more_rate_std", "hit3_or_more_rate_std",
        "max_drawdown_std", "max_consecutive_loss_std",
    ]

    for r in all_results:
        row = {}
        for f in metric_fields:
            if f in r:
                row[f] = r[f]
        # 附加标准差字段（仅 random multi-trial 有）
        for s in std_suffixes:
            if s in r:
                row[s] = r[s]
        summary_rows.append(row)

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    logger.info("策略汇总已保存: {} ({} 个策略)".format(out_path, len(summary_rows)))


def save_backtest_comparison(main_result, random_result, out_path):
    """保存 main vs random 的相对指标对比。"""
    if main_result is None or random_result is None:
        return

    def _get(r, key):
        return r.get(key, 0.0)

    comparison = {
        "main_vs_random_roi_delta":
            round(_get(main_result, "roi") - _get(random_result, "roi"), 6),
        "main_vs_random_profit_delta":
            int(_get(main_result, "total_profit") - _get(random_result, "total_profit")),
        "main_vs_random_avg_front_hit_delta":
            round(_get(main_result, "avg_front_hit") - _get(random_result, "avg_front_hit"), 4),
        "main_vs_random_hit2_delta":
            round(_get(main_result, "hit2_or_more_rate") - _get(random_result, "hit2_or_more_rate"), 6),
        "main_vs_random_hit3_delta":
            round(_get(main_result, "hit3_or_more_rate") - _get(random_result, "hit3_or_more_rate"), 6),
        "main_vs_random_max_drawdown_delta":
            round(_get(main_result, "max_drawdown") - _get(random_result, "max_drawdown"), 2),
        "main_vs_random_max_loss_streak_delta":
            int(_get(main_result, "max_consecutive_loss") - _get(random_result, "max_consecutive_loss")),
    }

    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info("策略对比已保存: {}".format(out_path))

    # 打印对比结论
    logger.info("=" * 60)
    logger.info("main vs random 对比:")
    for k, v in comparison.items():
        direction = "↑ main better" if v > 0 else ("↓ main worse" if v < 0 else "  equal")
        logger.info("  {}: {}{}  {}".format(k, '+' if v > 0 else '', v, direction))


# ============================================================
#  Random percentile & trial 级输出
# ============================================================

def compute_random_percentiles(main_result, trial_metrics_list):
    """计算 main 策略在 random trial 分布中的百分位。

    - 越大越好的指标：percentile = P(random <= main)
    - 越小越好的指标：percentile = P(random >= main)

    percentile 越高表示 main 相对 random 越好。
    """
    if not trial_metrics_list:
        return {}

    higher_better = [
        "roi", "total_profit", "avg_front_hit",
        "hit2_or_more_rate", "hit3_or_more_rate",
        "roi_truncated", "total_profit_truncated",
    ]
    lower_better = [
        "max_drawdown", "max_consecutive_loss",
    ]

    percentiles = {}
    for key in higher_better:
        main_val = main_result.get(key, 0.0)
        random_vals = [m.get(key, 0.0) for m in trial_metrics_list]
        count_le = sum(1 for v in random_vals if v <= main_val)
        percentiles["main_" + key + "_percentile"] = (
            float(count_le / len(random_vals)) if random_vals else 0.0
        )

    for key in lower_better:
        main_val = main_result.get(key, 0.0)
        random_vals = [m.get(key, 0.0) for m in trial_metrics_list]
        count_ge = sum(1 for v in random_vals if v >= main_val)
        percentiles["main_" + key + "_percentile"] = (
            float(count_ge / len(random_vals)) if random_vals else 0.0
        )

    return percentiles


def _build_interpretation(percentiles):
    """生成百分位的人类可读解释。"""
    interp = {}
    labels = {
        "main_roi_percentile": "roi",
        "main_total_profit_percentile": "total_profit",
        "main_avg_front_hit_percentile": "avg_front_hit",
        "main_hit2_or_more_rate_percentile": "hit2_or_more_rate",
        "main_hit3_or_more_rate_percentile": "hit3_or_more_rate",
        "main_max_drawdown_percentile": "max_drawdown",
        "main_max_consecutive_loss_percentile": "max_consecutive_loss",
        "main_roi_truncated_percentile": "roi_truncated",
        "main_total_profit_truncated_percentile": "total_profit_truncated",
    }
    for key, label in labels.items():
        pct = percentiles.get(key)
        if pct is not None:
            pct_pct = round(pct * 100, 1)
            interp[label] = f"main 超过了 {pct_pct}% 的 random trials"
    return interp


def save_random_trials(trial_metrics_list, csv_path, json_path):
    """保存 random trial 级汇总数据为 CSV 和 JSON。"""
    if not trial_metrics_list:
        return

    # CSV
    trial_fields = [
        "periods", "total_cost", "total_reward", "total_profit", "roi",
        "total_reward_truncated", "total_profit_truncated", "roi_truncated",
        "avg_front_hit",
        "hit0_rate", "hit1_rate", "hit2_rate", "hit3_rate",
        "hit4_rate", "hit5_rate",
        "hit2_or_more_rate", "hit3_or_more_rate",
        "max_drawdown", "max_consecutive_loss",
        "avg_period_profit", "avg_period_profit_truncated",
        "profit_std",
    ]
    rows = []
    for i, m in enumerate(trial_metrics_list):
        row = {"trial_id": i + 1}
        for f in trial_fields:
            row[f] = m.get(f, None)
        rows.append(row)
    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(csv_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("随机 trial 明细已保存: {} ({} 行)".format(csv_path, len(df)))

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    logger.info("随机 trial JSON 已保存: {}".format(json_path))


def save_backtest_percentile(percentiles, interpretation, n_trials, out_path):
    """保存 main vs random percentile 文件。"""
    payload = {
        "random_trials": n_trials,
        "main_vs_random_percentiles": percentiles,
        "interpretation": interpretation,
    }
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Percentile 文件已保存: {}".format(out_path))


def apply_best_params_to_config(config_path, best):
    """将最优参数回写到 config.py。"""
    if not os.path.exists(config_path):
        raise Exception("config.py 不存在: {}".format(config_path))

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    content, n1 = re.subn(
        r'("top_n_front"\s*:\s*)\d+',
        r'\g<1>{}'.format(int(best["top_n_front"])), content, count=1
    )
    content, n2 = re.subn(
        r'("play_front_combos"\s*:\s*)\d+',
        r'\g<1>{}'.format(int(best["play_front_combos"])), content, count=1
    )

    w = best["weights"]
    content, n3 = re.subn(
        r'("lstm"\s*:\s*)[0-9.]+',
        r'\g<1>{:.2f}'.format(float(w["lstm"])), content, count=1
    )
    content, n4 = re.subn(
        r'("rf"\s*:\s*)[0-9.]+',
        r'\g<1>{:.2f}'.format(float(w["rf"])), content, count=1
    )
    content, n5 = re.subn(
        r'("stat"\s*:\s*)[0-9.]+',
        r'\g<1>{:.2f}'.format(float(w["stat"])), content, count=1
    )

    if min(n1, n2, n3, n4, n5) == 0:
        raise Exception("回写失败：未匹配到全部参数键，请检查 config.py 结构")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_current_params_for_guard(config_path):
    """提取当前 config.py 中的策略参数用于保护开关比较。"""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _search(pattern, cast=float, default=None):
        m = re.search(pattern, content)
        if not m:
            return default
        return cast(m.group(1))

    return {
        "top_n_front": _search(r'"top_n_front"\s*:\s*(\d+)', int),
        "play_front_combos": _search(r'"play_front_combos"\s*:\s*(\d+)', int),
        "weights": {
            "lstm": _search(r'"lstm"\s*:\s*([0-9.]+)', float),
            "rf": _search(r'"rf"\s*:\s*([0-9.]+)', float),
            "stat": _search(r'"stat"\s*:\s*([0-9.]+)', float),
        }
    }


def pick_result_by_params(grid_results, params):
    """在网格结果中查找与给定参数匹配的项。"""
    for item in grid_results:
        if int(item["top_n_front"]) != int(params["top_n_front"]):
            continue
        if int(item["play_front_combos"]) != int(params["play_front_combos"]):
            continue
        w = item["weights"]
        if (
            abs(float(w["lstm"]) - float(params["weights"]["lstm"])) < 1e-9
            and abs(float(w["rf"]) - float(params["weights"]["rf"])) < 1e-9
            and abs(float(w["stat"]) - float(params["weights"]["stat"])) < 1e-9
        ):
            return item
    return None


def save_grid_results(results, out_path):
    """保存网格搜索结果。"""
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# ============================================================
#  日志辅助
# ============================================================

# ============================================================
#  9候选池生成器 (P2-1)
# ============================================================

N9_PROFILES = {
    "n9_mix_441": {"stat_miss_bonus": 4, "rf_stat": 4, "gap_direct_top5": 1},
    "n9_mix_531": {"stat_miss_bonus": 5, "rf_stat": 3, "gap_direct_top5": 1},
    "n9_mix_621": {"stat_miss_bonus": 6, "rf_stat": 2, "gap_direct_top5": 1},
}


def _get_ranked_by_source(hist, rf_model_static=None, rf_meta=None,
                          score_windows=(10, 30, 100), min_history=120):
    """计算三个来源各自的前区排序（1~35，分数从高到低）。

    Returns:
        dict: {"stat_miss_bonus": [n1, n2, ...],
               "rf_stat": [n1, n2, ...] or None,
               "gap_direct_top5": [n1, n2, ...]}
    """
    result = {}

    # A) stat_miss_bonus
    _scores_smb = _compute_stat_miss_bonus(hist, score_windows)
    result["stat_miss_bonus"] = (np.argsort(_scores_smb)[::-1] + 1).tolist()

    # B) gap_direct_top5
    _scores_gap = _compute_gap_only_scores(hist)
    result["gap_direct_top5"] = (np.argsort(_scores_gap)[::-1] + 1).tolist()

    # C) rf_stat
    _rf_ranked = None
    if rf_model_static is not None and len(hist) >= min_history:
        try:
            _rf_scores = predict_rf_scores(rf_model_static, hist,
                                           windows=score_windows,
                                           min_history=min_history)
            _stat_for_rf = _compute_stat_miss_penalty(hist, score_windows)
            # RF:stat = 0.65:0.35 融合
            _ensemble = 0.65 * _rf_scores + 0.35 * _stat_for_rf
            _rf_ranked = (np.argsort(_ensemble)[::-1] + 1).tolist()
        except Exception:
            _rf_ranked = None
    result["rf_stat"] = _rf_ranked

    return result


def generate_n9_candidate_pool(hist, profile_name,
                               rf_model_static=None, rf_meta=None,
                               score_windows=(10, 30, 100), min_history=120):
    """按指定 profile 配额生成 9 个前区候选号。

    Args:
        hist: 截至目标期之前的历史数据 (升序 DataFrame)
        profile_name: "n9_mix_441" / "n9_mix_531" / "n9_mix_621"
        rf_model_static: 预训练 RF 模型（可为 None）
        rf_meta: RF 元信息
        score_windows: 评分窗口
        min_history: 最小历史期数

    Returns:
        dict: {
            "profile_name": ...,
            "candidates": [9个号码],
            "candidate_sources": {num: [source_names]},
            "ranked_by_source": {...},
            "fill_reason": "...",
            "rf_available": bool,
        }
    """
    profile = N9_PROFILES.get(profile_name)
    if profile is None:
        return {"profile_name": profile_name, "candidates": [],
                "error": "unknown profile"}

    ranked = _get_ranked_by_source(hist, rf_model_static, rf_meta,
                                   score_windows, min_history)
    rf_available = ranked["rf_stat"] is not None

    # 按配额依次取号，去重
    candidates = []
    sources = {}
    used = set()

    source_order = ["stat_miss_bonus", "rf_stat", "gap_direct_top5"]
    for src in source_order:
        quota = profile.get(src, 0)
        if quota <= 0:
            continue
        src_ranked = ranked.get(src)
        if src_ranked is None:
            continue
        count = 0
        # 第一遍：收集唯一号码以满足配额
        for n in src_ranked:
            _key = str(n).zfill(2)
            if n not in used:
                candidates.append(n)
                used.add(n)
                sources.setdefault(_key, []).append(src)
                count += 1
            else:
                # 号码已被前面来源选中，追加当前来源到 sources
                if src not in sources.get(_key, []):
                    sources[_key].append(src)
            if count >= quota:
                break
        if count < quota:
            # 该来源不足，继续从下一轮补齐
            for n in src_ranked:
                _key = str(n).zfill(2)
                if n not in used:
                    candidates.append(n)
                    used.add(n)
                    sources.setdefault(_key, []).append(src + "_overflow")
                    count += 1
                else:
                    if src not in sources.get(_key, []):
                        sources[_key].append(src)
                if count >= quota:
                    break

    # 如果不足 9 个，用综合分补齐
    fill_reason = "quota_ok"
    if len(candidates) < 9:
        fill_reason = "filled_from_composite"
        # 综合分权重 = 当前 profile 配额（动态生成）
        composite = np.zeros(35, dtype=float)
        _weights = {
            "stat_miss_bonus": float(profile.get("stat_miss_bonus", 0)),
            "rf_stat": float(profile.get("rf_stat", 0)) if rf_available else 0.0,
            "gap_direct_top5": float(profile.get("gap_direct_top5", 0)),
        }
        _total_w = sum(_weights.values())
        if _total_w > 0:
            for src in source_order:
                _src_ranked = ranked.get(src)
                if _src_ranked is None:
                    continue
                _w = _weights[src] / _total_w
                # 排名分：排名越前分越高
                for _pos, _n in enumerate(_src_ranked):
                    composite[_n - 1] += _w * (35.0 - _pos) / 35.0
        _comp_ranked = (np.argsort(composite)[::-1] + 1).tolist()
        for n in _comp_ranked:
            if n not in used:
                candidates.append(n)
                used.add(n)
                sources.setdefault(str(n).zfill(2), []).append("composite")
                if len(candidates) >= 9:
                    break

    # 截断到 9 个
    candidates = candidates[:9]

    # 构建 ranked_by_source（前 10 每个来源）
    ranked_by_source = {}
    for src in source_order:
        _r = ranked.get(src)
        ranked_by_source[src] = _r[:10] if _r else []

    return {
        "profile_name": profile_name,
        "candidates": candidates,
        "candidate_sources": sources,
        "ranked_by_source": ranked_by_source,
        "fill_reason": fill_reason,
        "rf_available": rf_available,
    }


def _extract_window_row(result, window_id, start_issue, end_issue):
    """从策略结果中提取滚动窗口所需的关键指标。"""
    return {
        "window_id": window_id,
        "window_start_issue": start_issue,
        "window_end_issue": end_issue,
        "strategy_name": result.get("strategy_name", "unknown"),
        "roi_truncated": result.get("roi_truncated", 0.0),
        "avg_front_hit": result.get("avg_front_hit", 0.0),
        "hit2_or_more_rate": result.get("hit2_or_more_rate", 0.0),
        "hit3_or_more_rate": result.get("hit3_or_more_rate", 0.0),
        "max_consecutive_loss": result.get("max_consecutive_loss", 0),
    }


def _log_strategy_metrics(r, prefix="  "):
    """统一格式打印策略指标。"""
    logger.info("{}profit={:>8d}  roi={:>8.4f}  roi_t={:>8.4f}  "
                "avg_hit={:.2f}  hit2+={:.4f}  hit3+={:.4f}  "
                "maxDD={:>8.0f}  maxLoss={:>3d}期".format(
                    prefix,
                    r.get("total_profit", 0),
                    r.get("roi", 0.0),
                    r.get("roi_truncated", 0.0),
                    r.get("avg_front_hit", 0.0),
                    r.get("hit2_or_more_rate", 0.0),
                    r.get("hit3_or_more_rate", 0.0),
                    r.get("max_drawdown", 0.0),
                    int(r.get("max_consecutive_loss", 0)),
                ))


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="大乐透 5+12 策略回测（支持 walk_forward RF + 多种基线）"
    )
    parser.add_argument("--name", default="dlt", type=str)
    parser.add_argument("--start_offset", default=200, type=int,
                        help="回测最近多少期")
    parser.add_argument("--run_grid", default=0, type=int,
                        help="是否执行网格搜索")
    parser.add_argument("--rf_mode", default="walk_forward", type=str,
                        choices=["none", "static", "walk_forward"],
                        help="RF 模式: none=纯统计分, static=静态预训练模型, "
                             "walk_forward=逐期滚动重训（默认）")
    parser.add_argument("--auto_apply_best", default=0, type=int,
                        help="是否自动回写最优参数到 config.py")
    parser.add_argument("--apply_only_if_better", default=1, type=int,
                        help="仅当优于当前参数才回写")
    parser.add_argument("--save_grid", default=1, type=int)
    parser.add_argument("--baselines", default=1, type=int,
                        help="是否运行基线策略对比（random/hot_number/gap_only）")
    parser.add_argument("--random_trials", default=1, type=int,
                        help="随机基线重复试验次数")
    parser.add_argument("--output_report", default=1, type=int,
                        help="是否输出 backtest_report.csv")
    parser.add_argument("--ablation", default=0, type=int,
                        help="消融实验模式：1=运行 stat_only/rf_only/rf_stat "
                             "+ random/hot_number/gap_only 全部变体")
    parser.add_argument("--gap_experiment", default=0, type=int,
                        help="gap方向实验：1=运行 stat_miss_penalty/"
                             "stat_no_miss/stat_miss_bonus/rf_stat/rf_stat_gap/"
                             "gap_only/random/hot_number 全部变体")
    parser.add_argument("--filter_experiment", default=0, type=int,
                        help="组合过滤消融实验：1=运行 gap_direct_top5/"
                             "gap_score_default/gap_score_no_filters/"
                             "gap_score_relaxed_filters/gap_score_top5_candidate/"
                             "stat_miss_bonus/rf_stat/random 全部变体")
    parser.add_argument("--rolling_stability", default=0, type=int,
                        help="滚动窗口稳定性验证：1=启用")
    parser.add_argument("--rolling_window", default=200, type=int,
                        help="滚动窗口大小（默认200期）")
    parser.add_argument("--rolling_step", default=100, type=int,
                        help="滚动窗口步长（默认100期）")
    parser.add_argument("--rolling_random_trials", default=100, type=int,
                        help="每个滚动窗口中random基线的trial数")
    parser.add_argument("--n9_candidates", default=0, type=int,
                        help="9候选池预览模式：1=启用")
    parser.add_argument("--n9_profile", default="all", type=str,
                        help="候选池profile: n9_mix_441 / n9_mix_531 / n9_mix_621 / all")
    parser.add_argument("--n9_preview_last", default=5, type=int,
                        help="预览最近N期的候选池")
    args = parser.parse_args()

    if args.name != "dlt":
        raise Exception("backtest_plus.py 当前仅支持 dlt")

    # --- 加载数据 ---
    data_path = "{}{}".format(name_path["dlt"]["path"], data_file_name)
    data_asc = load_dlt_history(data_path)
    logger.info("数据加载完成，期号范围: {} → {}（升序: 旧→新），共 {} 期".format(
        int(data_asc["期数"].iloc[0]),
        int(data_asc["期数"].iloc[-1]),
        len(data_asc),
    ))

    start_idx = max(121, len(data_asc) - int(args.start_offset))
    end_idx = len(data_asc) - 1
    strategy = plus_strategy["dlt"]

    logger.info("回测窗口: 期号 {} → {} (索引 {} → {})".format(
        int(data_asc.iloc[start_idx]["期数"]),
        int(data_asc.iloc[end_idx]["期数"]),
        start_idx, end_idx,
    ))

    # --- RF 模式处理 ---
    rf_model_static = None
    rf_meta = None

    if args.rf_mode == "static":
        rf_model_static, rf_meta = maybe_load_rf_model()
        if rf_model_static is None:
            logger.error("static 模式需要预训练 RF 模型，但未找到。"
                         "请先执行 run_train_rf_model.py")
            logger.warning("降级为 rf_mode=none")
            args.rf_mode = "none"
        else:
            # 检查泄漏风险
            train_end = (rf_meta.get("train_issue_end")
                         if rf_meta else None)
            backtest_start = int(data_asc.iloc[start_idx]["期数"])
            if train_end is not None and train_end >= backtest_start:
                logger.warning(
                    "⚠️  泄漏风险：RF 训练截止期号 {} >= 回测起始期号 {}。"
                    "静态模型可能已经见过回测期数据，回测结果将偏高（不可信）。"
                    "建议使用 --rf_mode walk_forward 消除此风险。".format(
                        train_end, backtest_start
                    )
                )
            else:
                logger.info(
                    "RF static 模式：训练截止期号 {} < 回测起始期号 {}，无泄漏。".format(
                        train_end, backtest_start
                    )
                )

    if args.rf_mode == "none":
        rf_meta = {"min_history": 120}

    # 消融实验模式：如果 --ablation 1 且需要 RF，尝试加载
    if int(args.ablation) == 1:
        _rf_ab, _rf_meta_ab = maybe_load_rf_model()
        if _rf_ab is not None:
            # 预检：用少量历史数据测试 RF 是否能正常预测
            try:
                _test_hist = data_asc.iloc[:200]
                _ = predict_rf_scores(_rf_ab, _test_hist, (10, 30, 100), 120)
                logger.info("消融实验：RF 模型已加载并通过预检，将运行 rf_only 和 rf_stat")
            except Exception as _e:
                logger.error(
                    "RF 模型预检失败（可能是 sklearn 版本不兼容）: {}。".format(_e)
                )
                logger.error(
                    "请用当前 sklearn 版本重新训练 RF 模型："
                    "python run_train_rf_model.py --name dlt "
                    "--train_test_split 0.8 --min_history 120"
                )
                _rf_ab = None
        else:
            logger.warning(
                "消融实验：RF 模型不可用，将跳过 rf_only 和 rf_stat。"
                "请先执行: python run_train_rf_model.py --name dlt "
                "--train_test_split 0.8 --min_history 120"
            )
        # 检查泄漏
        rf_model_static, rf_meta = _rf_ab, _rf_meta_ab
        if rf_model_static is not None:
            train_end = (rf_meta.get("train_issue_end") if rf_meta else None)
            backtest_start = int(data_asc.iloc[start_idx]["期数"])
            if train_end is not None and train_end >= backtest_start:
                logger.warning(
                    "⚠️  RF 训练截止期号 {} >= 回测起始期号 {}。"
                    "消融结果可能有泄漏偏差。".format(train_end, backtest_start)
                )

    # --- 运行策略回测 ---
    all_results = []
    main_result = None  # will point to stat_miss_penalty or stat_only for comparison

    if int(args.n9_candidates) == 1:
        # ================================================================
        #  P2-1: 9候选池预览模式
        # ================================================================
        _n9_last = int(args.n9_preview_last)
        _n9_profiles = (["n9_mix_441", "n9_mix_531", "n9_mix_621"]
                        if args.n9_profile == "all"
                        else [args.n9_profile])

        # 加载 RF
        _rf_n9, _rf_meta_n9 = maybe_load_rf_model()
        _rf_ok_n9 = False
        if _rf_n9 is not None:
            try:
                _ = predict_rf_scores(_rf_n9, data_asc.iloc[:200], (10, 30, 100), 120)
                _rf_ok_n9 = True
                logger.info("N9 preview: RF loaded, train_end={}".format(
                    _rf_meta_n9.get("train_issue_end") if _rf_meta_n9 else "?"))
            except Exception as _e:
                logger.warning("N9 preview: RF pre-check fail: {}".format(_e))

        # 预览最近 _n9_last 期
        _n9_rows = []
        _n9_start = max(121, len(data_asc) - _n9_last)
        _score_windows = (int(strategy["score_windows"]["short"]),
                          int(strategy["score_windows"]["mid"]),
                          int(strategy["score_windows"]["long"]))

        logger.info("=" * 60)
        logger.info("9候选池预览: 最近 {} 期, profiles={}".format(
            _n9_last, _n9_profiles))

        for _idx in range(_n9_start, len(data_asc)):
            _hist = data_asc.iloc[:_idx]
            _target = data_asc.iloc[_idx]
            _issue = int(_target["期数"])
            _actual = sorted([int(_target[c]) for c in FRONT_COLS])

            for _prof in _n9_profiles:
                _pool = generate_n9_candidate_pool(
                    _hist, _prof,
                    rf_model_static=_rf_n9 if _rf_ok_n9 else None,
                    rf_meta=_rf_meta_n9,
                    score_windows=_score_windows,
                )
                _cands = _pool["candidates"]
                _hit_count = len(set(_cands).intersection(set(_actual)))
                _n9_rows.append({
                    "issue": _issue,
                    "profile_name": _prof,
                    "candidates": _cands,
                    "candidate_count": len(_cands),
                    "actual_front": _actual,
                    "hit_in_pool": _hit_count,
                    "stat_top": _pool["ranked_by_source"].get("stat_miss_bonus", [])[:5],
                    "rf_top": _pool["ranked_by_source"].get("rf_stat", [])[:5],
                    "gap_top": _pool["ranked_by_source"].get("gap_direct_top5", [])[:5],
                    "candidate_sources": _pool.get("candidate_sources", {}),
                    "fill_reason": _pool.get("fill_reason", ""),
                    "rf_available": _pool.get("rf_available", False),
                })
                logger.info("  Issue {} | {}: candidates={} hit={}/9 actual={}".format(
                    _issue, _prof, _cands, _hit_count, _actual))

        # 保存
        _n9_df = pd.DataFrame(_n9_rows)
        _n9_csv = os.path.join("outputs", "n9_candidate_pool_preview.csv")
        _n9_json = os.path.join("outputs", "n9_candidate_pool_preview.json")
        out_dir = os.path.dirname(_n9_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        _n9_df.to_csv(_n9_csv, index=False, encoding="utf-8-sig")
        with open(_n9_json, "w", encoding="utf-8") as f:
            json.dump(_n9_rows, f, ensure_ascii=False, indent=2)
        logger.info("候选池预览已保存: {} / {}".format(_n9_csv, _n9_json))

        # 摘要
        logger.info("--- 候选池预览摘要 ---")
        for _prof in _n9_profiles:
            _sub = _n9_df[_n9_df["profile_name"] == _prof]
            _all9 = (_sub["candidate_count"] == 9).all()
            _avg_hit = _sub["hit_in_pool"].mean()
            logger.info("  {}: all_9={}, avg_hit_in_pool={:.2f}".format(
                _prof, _all9, _avg_hit))
        logger.info("RF available: {}".format(_rf_ok_n9))

        # 自检
        _errors = 0
        for _i, _row in enumerate(_n9_rows):
            _cands = _row["candidates"]
            _issue = _row["issue"]
            _prof = _row["profile_name"]
            if len(_cands) != 9:
                logger.warning("SELF-CHECK FAIL: Issue {} {} candidate_count={}".format(
                    _issue, _prof, len(_cands)))
                _errors += 1
            if len(set(_cands)) != len(_cands):
                logger.warning("SELF-CHECK FAIL: Issue {} {} has duplicates: {}".format(
                    _issue, _prof, _cands))
                _errors += 1
            for _n in _cands:
                if _n < 1 or _n > 35:
                    logger.warning("SELF-CHECK FAIL: Issue {} {} num {} out of range".format(
                        _issue, _prof, _n))
                    _errors += 1
        if _errors == 0:
            logger.info("自检通过: {}行全部合法".format(len(_n9_rows)))
        else:
            logger.warning("自检发现 {} 个问题".format(_errors))

        logger.info("9候选池预览完成")
        return

    if int(args.rolling_stability) == 1:
        # ================================================================
        #  滚动窗口稳定性验证
        # ================================================================
        _wsize = int(args.rolling_window)
        _wstep = int(args.rolling_step)
        _rw_trials = int(args.rolling_random_trials)
        logger.info("=" * 60)
        logger.info("滚动窗口稳定性验证: window={}, step={}".format(_wsize, _wstep))

        _rf_rw, _rf_meta_rw = maybe_load_rf_model()
        _rf_ok_rw = False
        _rf_train_end = None
        if _rf_rw is not None:
            try:
                _ = predict_rf_scores(_rf_rw, data_asc.iloc[:200], (10, 30, 100), 120)
                _rf_ok_rw = True
                _rf_train_end = (_rf_meta_rw.get("train_issue_end") if _rf_meta_rw else None)
                logger.info("Rolling: RF loaded, train_end={}".format(_rf_train_end))
            except Exception as _e:
                logger.warning("Rolling: RF pre-check fail, skip rf_stat: {}".format(_e))

        _rw_results = []
        _min_start = 121
        _max_start = len(data_asc) - _wsize

        for _wstart in range(_min_start, _max_start + 1, _wstep):
            _wend = _wstart + _wsize - 1
            _wstart_issue = int(data_asc.iloc[_wstart]["期数"])
            _wend_issue = int(data_asc.iloc[_wend]["期数"])
            _wid = (_wstart - _min_start) // _wstep + 1

            logger.info("--- Window {}: {} → {} (idx {}→{}) ---".format(
                _wid, _wstart_issue, _wend_issue, _wstart, _wend))

            _common = dict(
                data_asc=data_asc, start_idx=_wstart, end_idx=_wend,
                max_front_combos=strategy["max_front_combos"],
                play_front_combos=int(strategy.get("play_front_combos", 1)),
            )
            _no_filters = {"odd_min": 0, "odd_max": 5, "big_min": 0, "big_max": 5,
                           "sum_min": 15, "sum_max": 170, "max_overlap_with_last": 5}
            _def_rules = dict(strategy["rule_filters"])

            # A) gap_direct_top5
            _r = run_backtest_core(
                **_common, top_n_front=5,
                ensemble_weights={"lstm": 0, "rf": 0, "stat": 1},
                rule_filters=_no_filters,
                rf_mode="none", score_mode="gap_only",
                strategy_name="gap_direct_top5",
            )
            _rw_results.append(_extract_window_row(_r, _wid, _wstart_issue, _wend_issue))

            # B) stat_miss_bonus
            _r = run_backtest_core(
                **_common, top_n_front=10,
                ensemble_weights={"lstm": 0, "rf": 0, "stat": 1},
                rule_filters=_def_rules,
                rf_mode="none", score_mode="stat_miss_bonus",
                strategy_name="stat_miss_bonus",
            )
            _rw_results.append(_extract_window_row(_r, _wid, _wstart_issue, _wend_issue))

            # C) rf_stat (仅当 RF 可用且无泄漏)
            _rf_safe = (_rf_ok_rw and _rf_train_end is not None
                        and _rf_train_end < _wstart_issue)
            if _rf_safe:
                _r = run_backtest_core(
                    **_common, top_n_front=10,
                    ensemble_weights={"lstm": 0, "rf": 0.65, "stat": 0.35},
                    rule_filters=_def_rules,
                    rf_mode="static", rf_model_static=_rf_rw,
                    rf_meta=_rf_meta_rw,
                    score_mode="stat_miss_penalty",
                    strategy_name="rf_stat",
                )
                _rw_results.append(_extract_window_row(_r, _wid, _wstart_issue, _wend_issue))
            else:
                logger.info("  rf_stat: SKIP (RF unavailable or leakage risk)")
                _rw_results.append({
                    "window_id": _wid,
                    "window_start_issue": _wstart_issue,
                    "window_end_issue": _wend_issue,
                    "strategy_name": "rf_stat",
                    "roi_truncated": None, "avg_front_hit": None,
                    "hit2_or_more_rate": None, "hit3_or_more_rate": None,
                    "max_consecutive_loss": None,
                    "skipped_reason": "leakage_risk_rf_train_end_ge_window_start",
                })

            # D) random
            _r = run_random_baseline(
                data_asc, _wstart, _wend,
                n_trials=_rw_trials,
                rng_seed=42 + _wid,
            )
            _rw_results.append(_extract_window_row(_r, _wid, _wstart_issue, _wend_issue))

        # 保存
        _rw_df = pd.DataFrame(_rw_results)
        _rw_csv = os.path.join("outputs", "rolling_window_summary.csv")
        _rw_json = os.path.join("outputs", "rolling_window_summary.json")
        out_dir = os.path.dirname(_rw_csv)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        _rw_df.to_csv(_rw_csv, index=False, encoding="utf-8-sig")
        with open(_rw_json, "w", encoding="utf-8") as f:
            json.dump(_rw_results, f, ensure_ascii=False, indent=2)
        logger.info("滚动窗口已保存: {} / {}".format(_rw_csv, _rw_json))

        # 排名统计
        _rankings = {}
        for _col in ["roi_truncated", "avg_front_hit",
                     "hit2_or_more_rate", "hit3_or_more_rate"]:
            _best = _rw_df.loc[_rw_df.groupby("window_id")[_col].idxmax()]
            _counts = _best["strategy_name"].value_counts().to_dict()
            _rankings[_col + "_champion"] = _counts

        # 超过 random 均值的窗口数
        _rand_mean = _rw_df[_rw_df["strategy_name"] == "random"].groupby(
            "window_id")["roi_truncated"].mean()
        for _sn in ["gap_direct_top5", "stat_miss_bonus", "rf_stat"]:
            _sub = _rw_df[_rw_df["strategy_name"] == _sn]
            _above = sum(1 for _, _row in _sub.iterrows()
                         if _row["roi_truncated"] > _rand_mean.get(_row["window_id"], -99))
            _rankings[_sn + "_above_random_mean"] = int(_above)

        # rf_stat 有效/跳过（排除 None 值的跳过行）
        _rf_valid = len(_rw_df[(_rw_df["strategy_name"] == "rf_stat") &
                               (_rw_df["roi_truncated"].notna())])
        _rf_total = _wid
        _rankings["rf_stat_valid_windows"] = int(_rf_valid)
        _rankings["rf_stat_skipped_windows"] = _rf_total - int(_rf_valid)

        _rankings_path = os.path.join("outputs", "rolling_window_rankings.json")
        with open(_rankings_path, "w", encoding="utf-8") as f:
            json.dump(_rankings, f, ensure_ascii=False, indent=2)
        logger.info("排名统计已保存: {}".format(_rankings_path))

        # 冠军统计
        logger.info("=" * 60)
        logger.info("滚动窗口冠军统计 ({} 窗口)".format(_wid))
        for _col in ["roi_truncated", "avg_front_hit",
                     "hit2_or_more_rate", "hit3_or_more_rate"]:
            _counts = _rankings.get(_col + "_champion", {})
            logger.info("  {} 冠军:".format(_col))
            for _sn, _cnt in sorted(_counts.items(), key=lambda x: -x[1]):
                logger.info("    {}: {}/{}".format(_sn, _cnt, _wid))

        # 跨窗口稳定性
        logger.info("--- 跨窗口均值±std ---")
        for _sn in _rw_df["strategy_name"].unique():
            _sub = _rw_df[_rw_df["strategy_name"] == _sn]
            logger.info("  {:>24s}: roi_t={:.4f}±{:.4f}  hit3+={:.4f}±{:.4f}".format(
                _sn,
                _sub["roi_truncated"].mean(), _sub["roi_truncated"].std(),
                _sub["hit3_or_more_rate"].mean(), _sub["hit3_or_more_rate"].std(),
            ))
        logger.info("  超过 random roi_t 均值:")
        for _sn in ["gap_direct_top5", "stat_miss_bonus", "rf_stat"]:
            _key = _sn + "_above_random_mean"
            logger.info("    {}: {}/{}".format(_sn, _rankings.get(_key, 0), _wid))
        logger.info("  rf_stat 有效/跳过: {}/{}".format(
            _rankings.get("rf_stat_valid_windows", 0),
            _rankings.get("rf_stat_skipped_windows", 0)))

        logger.info("滚动窗口验证完成")
        return

    if int(args.filter_experiment) == 1:
        # ================================================================
        #  组合过滤消融实验：判断哪一环节削弱了 gap 信号
        # ================================================================
        logger.info("=" * 60)
        logger.info("组合过滤消融实验：测试 top_n/rule_filters/combos 的影响...")

        # 加载 RF（如需要）
        _rf_fe, _rf_meta_fe = maybe_load_rf_model()
        _rf_ok_fe = False
        if _rf_fe is not None:
            try:
                _ = predict_rf_scores(_rf_fe, data_asc.iloc[:200], (10, 30, 100), 120)
                _rf_ok_fe = True
                logger.info("Filter 实验：RF 模型已加载并通过预检")
            except Exception as _e:
                logger.error("RF pre-check fail: {}".format(_e))
        else:
            logger.warning("RF 模型不可用，将跳过 rf_stat")

        # 规则定义
        _default_rules = dict(strategy["rule_filters"])
        _no_filters = {"odd_min": 0, "odd_max": 5, "big_min": 0, "big_max": 5,
                       "sum_min": 15, "sum_max": 170, "max_overlap_with_last": 5}
        _relaxed = {"odd_min": 0, "odd_max": 5, "big_min": 0, "big_max": 5,
                    "sum_min": 30, "sum_max": 145, "max_overlap_with_last": 3}

        _filter_variants = [
            ("gap_direct_top5", None, None, None, None, None),
            ("gap_score_default", "gap_only", 10, _default_rules, "none",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}),
            ("gap_score_no_filters", "gap_only", 10, _no_filters, "none",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}),
            ("gap_score_relaxed_filters", "gap_only", 10, _relaxed, "none",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}),
            ("gap_score_top5_candidate", "gap_only", 5, _default_rules, "none",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}),
            ("stat_miss_bonus_default", "stat_miss_bonus", 10, _default_rules, "none",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}),
        ]
        if _rf_ok_fe:
            _filter_variants.append(
                ("rf_stat_default", "stat_miss_penalty", 10, _default_rules, "static",
                 {"lstm": 0.0, "rf": 0.65, "stat": 0.35}),
            )

        for _name, _smode, _tn, _rules, _rfm, _w in _filter_variants:
            logger.info("--- {} ---".format(_name))
            if _name == "gap_direct_top5":
                # 使用与 gap_score 一致的评分函数而非 run_gap_only_baseline
                # (run_gap_only_baseline 使用 calc_front_missing 的 argsort 平局，
                #  与 _compute_gap_only_scores 的 +n/1000 平局策略不同)
                result = run_backtest_core(
                    data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
                    top_n_front=5,
                    max_front_combos=1,
                    play_front_combos=1,
                    ensemble_weights={"lstm": 0.0, "rf": 0.0, "stat": 1.0},
                    rule_filters={"odd_min": 0, "odd_max": 5, "big_min": 0,
                                  "big_max": 5, "sum_min": 15, "sum_max": 170,
                                  "max_overlap_with_last": 5},
                    rf_mode="none",
                    score_mode="gap_only",
                    strategy_name="gap_direct_top5",
                )
            else:
                _rf_use = _rf_fe if _rfm == "static" else None
                _rf_meta_use = _rf_meta_fe if _rfm == "static" else {"min_history": 120}
                result = run_backtest_core(
                    data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
                    top_n_front=_tn,
                    max_front_combos=strategy["max_front_combos"],
                    play_front_combos=int(strategy.get("play_front_combos", 1)),
                    ensemble_weights=_w,
                    rule_filters=_rules,
                    rf_mode=_rfm, rf_model_static=_rf_use,
                    rf_meta=_rf_meta_use,
                    score_mode=_smode,
                    strategy_name=_name,
                )
            _log_strategy_metrics(result)
            all_results.append(result)
            if _name == "gap_direct_top5":
                main_result = result

    elif int(args.gap_experiment) == 1:
        # ================================================================
        #  Gap 方向实验：测试遗漏值的不同处理方式
        # ================================================================
        logger.info("=" * 60)
        logger.info("Gap 方向实验模式：依次运行遗漏值惩罚/忽略/加分变体...")

        # 加载 RF（如需要）
        _rf_gap, _rf_meta_gap = maybe_load_rf_model()
        _rf_ok = False
        if _rf_gap is not None:
            try:
                _test_hist = data_asc.iloc[:200]
                _ = predict_rf_scores(_rf_gap, _test_hist, (10, 30, 100), 120)
                _rf_ok = True
                logger.info("Gap 实验：RF 模型已加载并通过预检")
            except Exception as _e:
                logger.error("RF pre-check fail, skip rf_stat/rf_stat_gap: {}".format(_e))
        else:
            logger.warning("RF 模型不可用，将跳过 rf_stat 和 rf_stat_gap")

        _gap_variants = [
            ("stat_miss_penalty", "stat_miss_penalty",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}, "none", None),
            ("stat_no_miss", "stat_no_miss",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}, "none", None),
            ("stat_miss_bonus", "stat_miss_bonus",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}, "none", None),
            ("gap_score", "gap_only",
             {"lstm": 0.0, "rf": 0.0, "stat": 1.0}, "none", None),
        ]
        if _rf_ok:
            _gap_variants += [
                ("rf_stat", "stat_miss_penalty",
                 {"lstm": 0.0, "rf": 0.65, "stat": 0.35}, "static", _rf_gap),
                ("rf_stat_gap", "rf_stat_gap",
                 {"lstm": 0.20, "rf": 0.50, "stat": 0.30}, "static", _rf_gap),
            ]

        for _name, _smode, _w, _rfm, _rfm_static in _gap_variants:
            logger.info("--- {} ---".format(_name))
            _rf_meta_use = _rf_meta_gap if _rfm_static is not None else {"min_history": 120}
            result = run_backtest_core(
                data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
                top_n_front=strategy["top_n_front"],
                max_front_combos=strategy["max_front_combos"],
                play_front_combos=int(strategy.get("play_front_combos", 1)),
                ensemble_weights=_w,
                rule_filters=strategy["rule_filters"],
                rf_mode=_rfm, rf_model_static=_rfm_static,
                rf_meta=_rf_meta_use,
                score_mode=_smode,
                strategy_name=_name,
            )
            _log_strategy_metrics(result)
            all_results.append(result)
            if _name == "stat_miss_penalty":
                main_result = result

    elif int(args.ablation) == 1:
        # ================================================================
        #  消融实验：依次运行所有变体
        # ================================================================
        logger.info("=" * 60)
        logger.info("消融实验模式：依次运行各策略变体...")

        # A) stat_only：纯统计分
        logger.info("--- stat_only (统计分) ---")
        stat_result = run_backtest_core(
            data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
            top_n_front=strategy["top_n_front"],
            max_front_combos=strategy["max_front_combos"],
            play_front_combos=int(strategy.get("play_front_combos", 1)),
            ensemble_weights=strategy["ensemble_weights"],
            rule_filters=strategy["rule_filters"],
            rf_mode="none",
            strategy_name="stat_only",
        )
        _log_strategy_metrics(stat_result)
        all_results.append(stat_result)
        main_result = stat_result

        # B) rf_only：只用 RF 分
        if rf_model_static is not None:
            logger.info("--- rf_only (RF分) ---")
            rf_only_result = run_backtest_core(
                data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
                top_n_front=strategy["top_n_front"],
                max_front_combos=strategy["max_front_combos"],
                play_front_combos=int(strategy.get("play_front_combos", 1)),
                ensemble_weights={"lstm": 0.0, "rf": 1.0, "stat": 0.0},
                rule_filters=strategy["rule_filters"],
                rf_mode="static",
                rf_model_static=rf_model_static, rf_meta=rf_meta,
                strategy_name="rf_only",
            )
            _log_strategy_metrics(rf_only_result)
            all_results.append(rf_only_result)

            # C) rf_stat：RF + stat 融合
            logger.info("--- rf_stat (RF+stat融合) ---")
            rf_stat_result = run_backtest_core(
                data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
                top_n_front=strategy["top_n_front"],
                max_front_combos=strategy["max_front_combos"],
                play_front_combos=int(strategy.get("play_front_combos", 1)),
                ensemble_weights={"lstm": 0.0, "rf": 0.65, "stat": 0.35},
                rule_filters=strategy["rule_filters"],
                rf_mode="static",
                rf_model_static=rf_model_static, rf_meta=rf_meta,
                strategy_name="rf_stat",
            )
            _log_strategy_metrics(rf_stat_result)
            all_results.append(rf_stat_result)
        else:
            logger.warning("跳过 rf_only / rf_stat（RF 模型不可用）")

    else:
        # --- 标准模式：单一 main 策略 ---
        logger.info("=" * 60)
        logger.info("开始主策略回测（rf_mode={}）...".format(args.rf_mode))

        main_result = run_backtest_core(
            data_asc=data_asc, start_idx=start_idx, end_idx=end_idx,
            top_n_front=strategy["top_n_front"],
            max_front_combos=strategy["max_front_combos"],
            play_front_combos=int(strategy.get("play_front_combos", 1)),
            ensemble_weights=strategy["ensemble_weights"],
            rule_filters=strategy["rule_filters"],
            rf_mode=args.rf_mode,
            rf_model_static=rf_model_static, rf_meta=rf_meta,
            strategy_name="main",
        )
        _log_strategy_metrics(main_result)
        all_results.append(main_result)

    # --- 运行基线策略（消融和标准模式共用）---
    if int(args.baselines) == 1:
        logger.info("=" * 60)
        logger.info("运行基线策略对比...")

        # 1) 随机基线
        random_result = run_random_baseline(
            data_asc, start_idx, end_idx,
            n_trials=int(args.random_trials),
            rng_seed=42,
        )
        logger.info("随机基线:")
        _log_strategy_metrics(random_result)
        if int(args.random_trials) > 1:
            logger.info("    mean±std ({} trials):".format(args.random_trials))
            logger.info("      profit={:.0f}±{:.0f}  roi={:.4f}±{:.4f}  "
                        "avg_hit={:.2f}±{:.2f}".format(
                            random_result["total_profit"],
                            random_result.get("total_profit_std", 0),
                            random_result["roi"],
                            random_result.get("roi_std", 0),
                            random_result["avg_front_hit"],
                            random_result.get("avg_front_hit_std", 0),
                        ))
            logger.info("      hit2+={:.4f}±{:.4f}  hit3+={:.4f}±{:.4f}  "
                        "maxDD={:.0f}±{:.0f}".format(
                            random_result["hit2_or_more_rate"],
                            random_result.get("hit2_or_more_rate_std", 0),
                            random_result["hit3_or_more_rate"],
                            random_result.get("hit3_or_more_rate_std", 0),
                            random_result["max_drawdown"],
                            random_result.get("max_drawdown_std", 0),
                        ))
        all_results.append(random_result)

        # 2) 热号基线
        hot_result = run_hot_number_baseline(
            data_asc, start_idx, end_idx, lookback=50,
        )
        logger.info("热号基线:")
        _log_strategy_metrics(hot_result)
        all_results.append(hot_result)

        # 3) 冷号遗漏基线
        gap_result = run_gap_only_baseline(
            data_asc, start_idx, end_idx,
        )
        logger.info("冷号基线:")
        _log_strategy_metrics(gap_result)
        all_results.append(gap_result)

        # 对比摘要
        logger.info("=" * 60)
        logger.info("策略对比摘要 (roi_t = roi_truncated@cap=10000):")
        for r in all_results:
            logger.info("  {:>16s}: profit={:>8d}  roi={:>8.4f}  roi_t={:>8.4f}  "
                        "avg_hit={:.2f}  hit2+={:.4f}  hit3+={:.4f}".format(
                            r["strategy_name"], r["total_profit"], r["roi"],
                            r.get("roi_truncated", 0.0),
                            r.get("avg_front_hit", 0.0),
                            r.get("hit2_or_more_rate", 0.0),
                            r.get("hit3_or_more_rate", 0.0),
                        ))

    # --- 输出回测报告 ---
    if int(args.output_report) == 1:
        report_path = os.path.join("outputs", "backtest_report.csv")
        save_backtest_report(all_results, report_path)

        # 策略汇总 JSON
        summary_path = os.path.join("outputs", "backtest_summary.json")
        save_backtest_summary(all_results, summary_path)

        # main (或 stat_only) vs random 对比
        if int(args.baselines) == 1 and main_result is not None:
            comparison_path = os.path.join("outputs", "backtest_comparison.json")
            save_backtest_comparison(main_result, random_result, comparison_path)

            # --- P1-3: random trial 明细 & percentile ---
            trial_metrics = random_result.get("_trial_metrics", [])
            if trial_metrics:
                save_random_trials(
                    trial_metrics,
                    os.path.join("outputs", "backtest_random_trials.csv"),
                    os.path.join("outputs", "backtest_random_trials.json"),
                )

                # 实验模式：为每个非 baseline 策略变体计算 percentile
                _pct_targets = [main_result] if main_result is not None else []
                _extra_names = set()
                if int(args.filter_experiment) == 1:
                    _extra_names = {"gap_score_default",
                                    "gap_score_no_filters",
                                    "gap_score_relaxed_filters",
                                    "gap_score_top5_candidate",
                                    "stat_miss_bonus_default",
                                    "rf_stat_default", "gap_only", "hot_number"}
                elif int(args.gap_experiment) == 1:
                    _extra_names = {"stat_no_miss", "stat_miss_bonus",
                                    "rf_stat", "rf_stat_gap",
                                    "gap_score", "gap_only", "hot_number"}
                elif int(args.ablation) == 1:
                    _extra_names = {"rf_only", "rf_stat"}
                for r in all_results:
                    if r["strategy_name"] in _extra_names:
                        _pct_targets.append(r)

                for _target in _pct_targets:
                    _sname = _target["strategy_name"]
                    percentiles = compute_random_percentiles(_target, trial_metrics)
                    interpretation = _build_interpretation(percentiles)
                    pct_path = os.path.join(
                        "outputs",
                        "backtest_percentile_{}.json".format(_sname)
                    )
                    save_backtest_percentile(
                        percentiles, interpretation,
                        len(trial_metrics), pct_path,
                    )

                    # --- P1-3 日志摘要 ---
                    logger.info("=" * 60)
                    logger.info("P1-3: {} vs random percentile 摘要".format(_sname))
                    logger.info("  {}_hit3_or_more_rate={:.4f}  "
                                "random(mean={:.4f}±{:.4f})  "
                                "percentile={:.1f}%".format(
                                    _sname,
                                    _target.get("hit3_or_more_rate", 0),
                                    random_result.get("hit3_or_more_rate", 0),
                                    random_result.get("hit3_or_more_rate_std", 0),
                                    percentiles.get(
                                        "main_hit3_or_more_rate_percentile", 0
                                    ) * 100,
                                ))
                    logger.info("  {}_avg_front_hit={:.4f}  "
                                "percentile={:.1f}%".format(
                                    _sname,
                                    _target.get("avg_front_hit", 0),
                                    percentiles.get(
                                        "main_avg_front_hit_percentile", 0
                                    ) * 100,
                                ))
                    logger.info("  {}_roi_truncated={:.4f}  "
                                "percentile={:.1f}%".format(
                                    _sname,
                                    _target.get("roi_truncated", 0),
                                    percentiles.get(
                                        "main_roi_truncated_percentile", 0
                                    ) * 100,
                                ))
                    _over_90 = sum(
                        1 for v in percentiles.values() if v >= 0.90
                    )
                    logger.info(
                        "  {} 超过 90% random trials 的指标数: {}/{}".format(
                            _sname, _over_90, len(percentiles)
                        ))

    # --- 网格搜索 ---
    if int(args.run_grid) == 1:
        logger.info("=" * 60)
        logger.info("开始网格搜索...")
        gs = run_grid_search(
            data_asc, start_idx, end_idx, strategy,
            rf_mode=args.rf_mode,
        )
        best = gs[0]
        logger.info(
            "网格最佳: top_n={}, play_front_combos={}, weights={}, "
            "profit={}, roi={:.4f}, hit3+={:.4f}".format(
                best["top_n_front"], best["play_front_combos"],
                best["weights"], best["profit"], best["roi"],
                best["hit3_or_more_rate"]
            ))

        if int(args.save_grid) == 1:
            save_grid_results(gs, os.path.join("outputs", "backtest_grid_results.json"))
            logger.info("网格结果已保存: outputs/backtest_grid_results.json")

        if int(args.auto_apply_best) == 1:
            config_path = os.path.join(os.getcwd(), "config.py")
            do_apply = True
            if int(args.apply_only_if_better) == 1:
                current_params = extract_current_params_for_guard(config_path)
                current_item = pick_result_by_params(gs, current_params)
                if current_item is None:
                    logger.warning("未在网格结果中找到当前参数组合，默认允许回写")
                else:
                    if best["profit"] <= current_item["profit"]:
                        do_apply = False
                        logger.info("保护开关生效：网格最优未优于当前参数，跳过回写")
                    else:
                        logger.info("保护开关生效：网格最优优于当前参数，允许回写")

            if do_apply:
                apply_best_params_to_config(config_path, best)
                logger.info("最优参数已回写到 config.py")


if __name__ == "__main__":
    main()
