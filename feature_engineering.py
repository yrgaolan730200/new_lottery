# -*- coding:utf-8 -*-
"""
大乐透特征工程工具
"""
import numpy as np
import pandas as pd


FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK_COLS = ["蓝球_1", "蓝球_2"]


def load_dlt_history(csv_path):
    """读取大乐透历史数据，并按期号升序排列（旧 -> 新）"""
    df = pd.read_csv(csv_path)
    if "期数" not in df.columns:
        raise Exception("历史数据缺少期数列")
    for c in FRONT_COLS + BACK_COLS:
        df[c] = df[c].astype(int)
    df["期数"] = df["期数"].astype(int)
    df = df.sort_values("期数").reset_index(drop=True)
    return df


def _front_set(draw_row):
    return {int(draw_row[c]) for c in FRONT_COLS}


def _normalize(v):
    arr = np.array(v, dtype=float)
    min_v = arr.min()
    max_v = arr.max()
    if max_v - min_v < 1e-12:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def calc_front_frequency(history_df, window_size, n_front=35):
    """近window_size期前区频次（归一化到[0,1]）"""
    sub = history_df.tail(window_size)
    freq = np.zeros(n_front, dtype=float)
    if len(sub) == 0:
        return freq
    for _, row in sub.iterrows():
        for c in FRONT_COLS:
            freq[int(row[c]) - 1] += 1.0
    return freq / len(sub)


def calc_front_missing(history_df, n_front=35):
    """前区遗漏值（越久未出现值越大），归一化到[0,1]"""
    miss = np.zeros(n_front, dtype=float)
    if len(history_df) == 0:
        return miss
    max_back = len(history_df) + 1
    # 从最近一期向前搜索最近出现位置
    for n in range(1, n_front + 1):
        gap = max_back
        for back_idx, (_, row) in enumerate(history_df.iloc[::-1].iterrows(), start=1):
            if n in _front_set(row):
                gap = back_idx
                break
        miss[n - 1] = gap
    return miss / miss.max()


def calc_last_draw_features(history_df):
    """上一期结构特征"""
    if len(history_df) == 0:
        return np.zeros(6, dtype=float)
    nums = sorted([int(history_df.iloc[-1][c]) for c in FRONT_COLS])
    odd = sum(n % 2 for n in nums)
    big = sum(n > 17 for n in nums)
    total = sum(nums)
    consecutive = 0
    for i in range(1, len(nums)):
        if nums[i] - nums[i - 1] == 1:
            consecutive += 1
    span = nums[-1] - nums[0]
    mean_val = float(np.mean(nums))
    return np.array([
        odd / 5.0,
        big / 5.0,
        total / (35.0 * 5),
        consecutive / 4.0,
        span / 34.0,
        mean_val / 35.0
    ], dtype=float)


def calc_overlap_feature(history_df):
    """最近两期前区重号比率"""
    if len(history_df) < 2:
        return np.array([0.0], dtype=float)
    curr = _front_set(history_df.iloc[-1])
    prev = _front_set(history_df.iloc[-2])
    overlap = len(curr.intersection(prev))
    return np.array([overlap / 5.0], dtype=float)


def build_front_feature_row(history_df, windows=(10, 30, 100)):
    """基于历史（不含目标期）构建单条特征"""
    features = []
    for w in windows:
        features.append(calc_front_frequency(history_df, w))
    for w in windows:
        features.append(calc_front_missing(history_df.tail(w)))

    # 最近一期 one-hot
    one_hot_last = np.zeros(35, dtype=float)
    if len(history_df) > 0:
        for c in FRONT_COLS:
            one_hot_last[int(history_df.iloc[-1][c]) - 1] = 1.0
    features.append(one_hot_last)

    features.append(calc_last_draw_features(history_df))
    features.append(calc_overlap_feature(history_df))

    return np.concatenate(features, axis=0)


def build_front_training_dataset(history_df, windows=(10, 30, 100), min_history=120):
    """构建RF训练集: X.shape=(samples, feats), y.shape=(samples, 35)"""
    if len(history_df) <= min_history:
        raise Exception("历史数据不足，无法构建训练集")

    x_data = []
    y_data = []
    issues = []

    for idx in range(min_history, len(history_df)):
        hist = history_df.iloc[:idx]
        target = history_df.iloc[idx]
        x = build_front_feature_row(hist, windows=windows)
        y = np.zeros(35, dtype=int)
        for c in FRONT_COLS:
            y[int(target[c]) - 1] = 1

        x_data.append(x)
        y_data.append(y)
        issues.append(int(target["期数"]))

    return np.array(x_data), np.array(y_data), np.array(issues)


def build_feature_for_next_issue(history_df, windows=(10, 30, 100), min_history=120):
    """构建下一期预测所需特征"""
    if len(history_df) < min_history:
        raise Exception("历史数据不足，无法构建下一期特征")
    return build_front_feature_row(history_df, windows=windows).reshape(1, -1)


def calc_stat_scores(history_df, windows=(10, 30, 100)):
    """统计打分（频次+遗漏融合）"""
    short, mid, long = windows
    s_freq = calc_front_frequency(history_df, short)
    m_freq = calc_front_frequency(history_df, mid)
    l_freq = calc_front_frequency(history_df, long)
    miss = calc_front_missing(history_df)

    # 出现频次越高越加分，遗漏过高适度扣分（偏向热号+温号）
    raw = 0.45 * s_freq + 0.35 * m_freq + 0.20 * l_freq - 0.20 * miss
    raw = _normalize(raw)
    return raw


def build_lstm_proxy_scores(pred_front_nums):
    """将LSTM给出的5个前区号映射为35维分数"""
    scores = np.zeros(35, dtype=float)
    if not pred_front_nums:
        return scores
    rank_weights = [1.0, 0.9, 0.8, 0.7, 0.6]
    for i, n in enumerate(pred_front_nums):
        if 1 <= int(n) <= 35:
            w = rank_weights[i] if i < len(rank_weights) else 0.5
            scores[int(n) - 1] = w
    return _normalize(scores)
