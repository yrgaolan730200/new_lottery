# -*- coding: utf-8 -*-
"""
Author: BigCat
"""
import os

ball_name = [
    ("红球", "red"),
    ("蓝球", "blue")
]

data_file_name = "data.csv"

name_path = {
    "ssq": {
        "name": "双色球",
        "path": "data/ssq/"
    },
    "dlt": {
        "name": "大乐透",
        "path": "data/dlt/"
    }
}

model_path = os.getcwd() + "/model/"

model_args = {
    "ssq": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "sequence_len": 6,
            "red_n_class": 33,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_n_class": 16,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/ssq/red_ball_model/",
            "blue": model_path + "/ssq/blue_ball_model/"
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red_sequence_len": 5,
            "red_n_class": 35,
            "red_epochs": 1,
            "red_embedding_size": 32,
            "red_hidden_size": 32,
            "red_layer_size": 1,
            "blue_sequence_len": 2,
            "blue_n_class": 12,
            "blue_epochs": 1,
            "blue_embedding_size": 32,
            "blue_hidden_size": 32,
            "blue_layer_size": 1
        },
        "train_args": {
            "red_learning_rate": 0.001,
            "red_beta1": 0.9,
            "red_beta2": 0.999,
            "red_epsilon": 1e-08,
            "blue_learning_rate": 0.001,
            "blue_beta1": 0.9,
            "blue_beta2": 0.999,
            "blue_epsilon": 1e-08
        },
        "path": {
            "red": model_path + "/dlt/red_ball_model/",
            "blue": model_path + "/dlt/blue_ball_model/"
        }
    }
}

# 模型名
pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"

# --- 大乐透 5+12 策略参数 ---
plus_strategy = {
    "dlt": {
        # 前区 Top-N 候选池
        "top_n_front": 10,
        # 候选池排序后保留的前区组合上限
        "max_front_combos": 20,
        # 实际每期下注的前区组合数（每组固定配后区66注）
        "play_front_combos": 1,
        # 前区过滤规则
        "rule_filters": {
            "odd_min": 1,
            "odd_max": 4,
            "big_min": 1,
            "big_max": 4,
            "sum_min": 55,
            "sum_max": 125,
            "max_overlap_with_last": 2
        },
        # 候选号码评分窗口
        "score_windows": {
            "short": 10,
            "mid": 30,
            "long": 100
        },
        # LSTM + RF + 统计分 融合权重
        "ensemble_weights": {
            "lstm": 0.35,
            "rf": 0.45,
            "stat": 0.20
        },
        # 回测参数
        "backtest": {
            "start_issue": None,
            "end_issue": None,
            "rolling_train_size": 500
        },
        # 奖金近似（用于策略收益评估，单位：元）
        "payouts": {
            "3+2": 200,
            "3+1": 10,
            "3+0": 5
        }
    }
}

# --- 大乐透奖金表（5+12策略逐注计奖用）---
# 大乐透奖级在 2026 年 2 月进行了改革（约第 26014 期起生效）。
# 主要变化：部分固定奖金额度上调（如 (3,1)/(2,2) 从 10 元调至 15 元），
#          奖池分配比例调整、浮动奖计算方式变更。
#
# 以下奖金表中：
#  - 一等奖/二等奖为浮动奖金，取历史近似值用于回测估算；
#  - 未计入浮动加奖/派奖活动（如奖池超 8 亿时的额外派奖）；
#  - backtest_report.csv 中 ROI 均为估算值，不代表真实收益。
# 键 = (front_hit, back_hit)，值 = 单注奖金（元）
# 未列出的 (front_hit, back_hit) 组合均为未中奖（0元）

# 2026年改革前（issue < 26014）
_DLT_PRIZE_TABLE_PRE_2026 = {
    (5, 2): 10_000_000,  # 一等奖（浮动，近似值）
    (5, 1): 200_000,     # 二等奖（浮动，近似值）
    (5, 0): 10_000,      # 三等奖
    (4, 2): 3_000,       # 四等奖
    (4, 1): 300,         # 五等奖
    (4, 0): 100,         # 六等奖
    (3, 2): 200,         # 六等奖
    (3, 1): 10,          # 七等奖（旧规则）
    (2, 2): 10,          # 七等奖（旧规则）
    (3, 0): 5,           # 八等奖
    (2, 1): 5,           # 八等奖
    (1, 2): 5,           # 八等奖
    (0, 2): 5,           # 八等奖
}

# 2026年改革后（issue >= 26014）
_DLT_PRIZE_TABLE_2026 = {
    (5, 2): 10_000_000,  # 一等奖（浮动，近似值）
    (5, 1): 200_000,     # 二等奖（浮动，近似值）
    (5, 0): 10_000,      # 三等奖
    (4, 2): 3_000,       # 四等奖
    (4, 1): 300,         # 五等奖
    (4, 0): 100,         # 七等奖
    (3, 2): 200,         # 六等奖
    (3, 1): 15,          # 八等奖（2026新规则上调）
    (2, 2): 15,          # 八等奖（2026新规则上调）
    (3, 0): 5,           # 九等奖
    (2, 1): 5,           # 九等奖
    (1, 2): 5,           # 九等奖
    (0, 2): 5,           # 九等奖
}

# 改革分界线期号（含）：从此期起使用 2026 新规则
_DLT_REFORM_ISSUE = 26014


def get_dlt_prize_table(issue):
    """根据期号返回对应的奖金表。

    26014 之前使用旧规则，26014 及之后使用 2026 新规则。
    如果 issue 无法判断（如 None 或未来期），默认使用 2026 新规则。

    Args:
        issue: 大乐透期号（int），如 26014

    Returns:
        dict: (front_hit, back_hit) → 单注奖金（元）
    """
    if issue is not None and int(issue) < _DLT_REFORM_ISSUE:
        return _DLT_PRIZE_TABLE_PRE_2026
    return _DLT_PRIZE_TABLE_2026


# RF 训练参数与存储路径
rf_args = {
    "dlt": {
        "front_model_path": model_path + "/dlt/rf_front_model.pkl",
        "front_meta_path": model_path + "/dlt/rf_front_meta.json",
        "n_estimators": 300,
        "max_depth": 8,
        "min_samples_split": 8,
        "min_samples_leaf": 3,
        "random_state": 42
    }
}
