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
