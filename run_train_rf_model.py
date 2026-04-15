# -*- coding:utf-8 -*-
"""
训练大乐透前区 RF 模型
"""
import os
import json
import pickle
import argparse

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from config import name_path, data_file_name, rf_args, plus_strategy
from feature_engineering import load_dlt_history, build_front_training_dataset


def train_rf_model(train_x, train_y, args_cfg):
    base = RandomForestClassifier(
        n_estimators=args_cfg["n_estimators"],
        max_depth=args_cfg["max_depth"],
        min_samples_split=args_cfg["min_samples_split"],
        min_samples_leaf=args_cfg["min_samples_leaf"],
        random_state=args_cfg["random_state"],
        n_jobs=-1,
    )
    model = MultiOutputClassifier(base)
    model.fit(train_x, train_y)
    return model


def evaluate_topk_hit(model, test_x, test_y, top_k=10):
    if len(test_x) == 0:
        return 0.0

    proba_list = model.predict_proba(test_x)
    scores = np.array([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in proba_list]).T

    hit_ratios = []
    for i in range(scores.shape[0]):
        topk = set((np.argsort(scores[i])[::-1][:top_k] + 1).tolist())
        true_set = set((np.where(test_y[i] == 1)[0] + 1).tolist())
        hit_ratios.append(len(topk.intersection(true_set)) / 5.0)
    return float(np.mean(hit_ratios))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dlt", type=str)
    parser.add_argument("--train_test_split", default=0.8, type=float)
    parser.add_argument("--min_history", default=120, type=int)
    args = parser.parse_args()

    if args.name != "dlt":
        raise Exception("run_train_rf_model.py 当前仅支持 dlt")

    data_path = "{}{}".format(name_path["dlt"]["path"], data_file_name)
    if not os.path.exists(data_path):
        raise Exception("未找到大乐透数据，请先执行 get_data.py --name dlt")

    history = load_dlt_history(data_path)
    strategy = plus_strategy["dlt"]
    windows = (
        int(strategy["score_windows"]["short"]),
        int(strategy["score_windows"]["mid"]),
        int(strategy["score_windows"]["long"]),
    )

    x_data, y_data, issues = build_front_training_dataset(history, windows=windows, min_history=args.min_history)
    split_idx = int(len(x_data) * args.train_test_split)
    train_x, test_x = x_data[:split_idx], x_data[split_idx:]
    train_y, test_y = y_data[:split_idx], y_data[split_idx:]
    train_issue, test_issue = issues[:split_idx], issues[split_idx:]

    cfg = rf_args["dlt"]
    model = train_rf_model(train_x, train_y, cfg)

    topk = int(strategy["top_n_front"])
    hit = evaluate_topk_hit(model, test_x, test_y, top_k=topk)
    logger.info("RF 测试集 Top-{} 平均命中比例: {:.4f}".format(topk, hit))
    if len(test_issue) > 0:
        logger.info("测试期号区间: {} -> {}".format(int(test_issue[0]), int(test_issue[-1])))

    model_dir = os.path.dirname(cfg["front_model_path"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(cfg["front_model_path"], "wb") as f:
        pickle.dump(model, f)

    meta = {
        "name": "dlt_front_rf",
        "features_windows": list(windows),
        "min_history": args.min_history,
        "train_size": int(len(train_x)),
        "test_size": int(len(test_x)),
        "train_issue_start": int(train_issue[0]) if len(train_issue) else None,
        "train_issue_end": int(train_issue[-1]) if len(train_issue) else None,
        "test_issue_start": int(test_issue[0]) if len(test_issue) else None,
        "test_issue_end": int(test_issue[-1]) if len(test_issue) else None,
        "topk_hit_ratio": hit,
    }
    with open(cfg["front_meta_path"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("RF模型已保存: {}".format(cfg["front_model_path"]))
    logger.info("RF元信息已保存: {}".format(cfg["front_meta_path"]))


if __name__ == "__main__":
    main()
