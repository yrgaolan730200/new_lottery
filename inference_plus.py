# -*- coding:utf-8 -*-
"""
大乐透 5+12 推理脚本
"""
import os
import sys
import json
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from loguru import logger

from config import name_path, data_file_name, plus_strategy, rf_args
from feature_engineering import (
    calc_stat_scores,
    build_lstm_proxy_scores,
    build_feature_for_next_issue,
)


# 延迟导入 run_predict，避免其在导入时提前消费本脚本命令行参数

def _load_lstm_predict_funcs():
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        import run_predict as rp
    finally:
        sys.argv = old_argv
    return rp.load_model, rp.get_final_result


def load_local_dlt_data_desc():
    path = "{}{}".format(name_path["dlt"]["path"], data_file_name)
    if not os.path.exists(path):
        raise Exception("未找到大乐透历史数据，请先执行 get_data.py --name dlt")
    data = pd.read_csv(path)
    for c in ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5", "蓝球_1", "蓝球_2"]:
        data[c] = data[c].astype(int)
    data["期数"] = data["期数"].astype(int)
    # 保持与 run_predict.py 一致：降序（最近一期在前）
    data = data.sort_values("期数", ascending=False).reset_index(drop=True)
    return data


def extract_front_nums_from_pred(pred_d):
    keys = [k for k in pred_d.keys() if ("红球_" in k or "red_" in k)]
    keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
    nums = [int(pred_d[k]) for k in keys]
    return nums


def maybe_load_rf_model():
    p = rf_args["dlt"]["front_model_path"]
    if not os.path.exists(p):
        logger.warning("RF模型不存在，将仅使用LSTM代理分数+统计分数")
        return None, None
    import pickle
    with open(p, "rb") as f:
        model = pickle.load(f)

    meta_p = rf_args["dlt"]["front_meta_path"]
    meta = None
    if os.path.exists(meta_p):
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, meta


def predict_rf_scores(rf_model, history_df_asc, windows=(10, 30, 100), min_history=120):
    if rf_model is None:
        return np.zeros(35, dtype=float)
    if len(history_df_asc) < min_history:
        return np.zeros(35, dtype=float)

    x = build_feature_for_next_issue(history_df_asc, windows=windows, min_history=min_history)

    # multi-output RF: predict_proba 返回 list(len=35)
    proba_list = rf_model.predict_proba(x)
    rf_scores = []
    for p in proba_list:
        # 二分类正类概率
        if p.shape[1] == 1:
            rf_scores.append(float(p[0, 0]))
        else:
            rf_scores.append(float(p[0, 1]))
    return np.array(rf_scores, dtype=float)


def normalize_weights(weights):
    total = weights["lstm"] + weights["rf"] + weights["stat"]
    if total <= 0:
        return {"lstm": 1.0, "rf": 0.0, "stat": 0.0}
    return {
        "lstm": weights["lstm"] / total,
        "rf": weights["rf"] / total,
        "stat": weights["stat"] / total
    }


def build_ensemble_scores(history_df_asc, lstm_front_nums, rf_model=None, rf_meta=None):
    strategy = plus_strategy["dlt"]
    score_windows_d = strategy["score_windows"]
    windows = (
        int(score_windows_d["short"]),
        int(score_windows_d["mid"]),
        int(score_windows_d["long"]),
    )
    min_history = int(rf_meta["min_history"]) if rf_meta else 120

    lstm_scores = build_lstm_proxy_scores(lstm_front_nums)
    stat_scores = calc_stat_scores(history_df_asc, windows=windows)
    rf_scores = predict_rf_scores(rf_model, history_df_asc, windows=windows, min_history=min_history)

    w = normalize_weights(strategy["ensemble_weights"])
    ensemble = w["lstm"] * lstm_scores + w["rf"] * rf_scores + w["stat"] * stat_scores
    return ensemble, {"lstm": lstm_scores, "rf": rf_scores, "stat": stat_scores}


def passes_filters(combo, last_front_set, rule_filters):
    odd = sum(n % 2 for n in combo)
    big = sum(n > 17 for n in combo)
    total = sum(combo)
    overlap = len(set(combo).intersection(last_front_set))

    if odd < rule_filters["odd_min"] or odd > rule_filters["odd_max"]:
        return False
    if big < rule_filters["big_min"] or big > rule_filters["big_max"]:
        return False
    if total < rule_filters["sum_min"] or total > rule_filters["sum_max"]:
        return False
    if overlap > rule_filters["max_overlap_with_last"]:
        return False
    return True


def generate_front_combos(ensemble_scores, last_front_set):
    strategy = plus_strategy["dlt"]
    top_n = int(strategy["top_n_front"])
    max_front_combos = int(strategy["max_front_combos"])
    rule_filters = strategy["rule_filters"]

    ranked = np.argsort(ensemble_scores)[::-1] + 1
    top_candidates = sorted(ranked[:top_n].tolist())

    valid = []
    for combo in combinations(top_candidates, 5):
        if passes_filters(combo, last_front_set, rule_filters):
            score = float(np.sum([ensemble_scores[n - 1] for n in combo]))
            valid.append((combo, score))

    # 如果过滤过严导致无候选，放宽为不过滤
    if len(valid) == 0:
        for combo in combinations(top_candidates, 5):
            score = float(np.sum([ensemble_scores[n - 1] for n in combo]))
            valid.append((combo, score))

    valid = sorted(valid, key=lambda x: x[1], reverse=True)
    return top_candidates, valid[:max_front_combos]


def generate_5_plus_12_tickets(front_combo):
    back_pairs = list(combinations(range(1, 13), 2))
    tickets = []
    for b1, b2 in back_pairs:
        tickets.append({
            "front": list(front_combo),
            "back": [b1, b2]
        })
    return tickets


def infer_next_issue(use_lstm=True, use_rf=True):
    # 历史数据
    data_desc = load_local_dlt_data_desc()
    data_asc = data_desc.sort_values("期数").reset_index(drop=True)

    latest_issue = int(data_desc.iloc[0]["期数"])
    next_issue = latest_issue + 1

    # 上一期前区用于重号过滤
    last_front_set = set(int(data_desc.iloc[0][c]) for c in ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"])

    # LSTM 5码预测（用于代理打分）
    lstm_front_nums = []
    if use_lstm:
        load_model, get_final_result = _load_lstm_predict_funcs()
        red_graph, red_sess, blue_graph, blue_sess, pred_key_d, _ = load_model("dlt")
        windows_size = 3
        predict_features = data_desc.iloc[:windows_size]
        pred_d = get_final_result(red_graph, red_sess, blue_graph, blue_sess, pred_key_d, "dlt", predict_features)
        lstm_front_nums = extract_front_nums_from_pred(pred_d)
        logger.info("LSTM前区预测5码: {}".format(lstm_front_nums))

    # RF
    rf_model, rf_meta = (None, None)
    if use_rf:
        rf_model, rf_meta = maybe_load_rf_model()

    ensemble_scores, comp_scores = build_ensemble_scores(data_asc, lstm_front_nums, rf_model, rf_meta)
    top_candidates, front_combo_rank = generate_front_combos(ensemble_scores, last_front_set)

    best_front = front_combo_rank[0][0]
    tickets = generate_5_plus_12_tickets(best_front)

    result = {
        "latest_issue": latest_issue,
        "next_issue": next_issue,
        "top_candidates": top_candidates,
        "best_front_combo": list(best_front),
        "best_front_score": float(front_combo_rank[0][1]),
        "front_combo_count": len(front_combo_rank),
        "back_pairs_count": len(tickets),
        "total_bets": len(tickets),
        "total_cost": len(tickets) * 2,
        "lstm_front_nums": lstm_front_nums,
        "component_top10": {
            "ensemble": (np.argsort(ensemble_scores)[::-1][:10] + 1).tolist(),
            "lstm": (np.argsort(comp_scores["lstm"])[::-1][:10] + 1).tolist(),
            "rf": (np.argsort(comp_scores["rf"])[::-1][:10] + 1).tolist(),
            "stat": (np.argsort(comp_scores["stat"])[::-1][:10] + 1).tolist()
        }
    }
    return result, tickets, front_combo_rank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dlt", type=str)
    parser.add_argument("--use_lstm", default=1, type=int)
    parser.add_argument("--use_rf", default=1, type=int)
    parser.add_argument("--save", default=1, type=int)
    args = parser.parse_args()

    if args.name != "dlt":
        raise Exception("inference_plus.py 当前仅支持 dlt")

    result, tickets, combo_rank = infer_next_issue(use_lstm=bool(args.use_lstm), use_rf=bool(args.use_rf))

    logger.info("预测目标期号: {}".format(result["next_issue"]))
    logger.info("前区Top候选: {}".format(result["top_candidates"]))
    logger.info("最佳前区5码: {}".format(result["best_front_combo"]))
    logger.info("后区全包数: {}（固定1~12选2）".format(result["back_pairs_count"]))
    logger.info("总注数: {}, 总成本: {}元".format(result["total_bets"], result["total_cost"]))

    if args.save:
        out_dir = "outputs"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        issue = result["next_issue"]
        result_path = os.path.join(out_dir, "inference_plus_{}.json".format(issue))
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 组合明细
        combo_df = pd.DataFrame([
            {
                "rank": i + 1,
                "front_combo": " ".join(["{:02d}".format(x) for x in combo]),
                "score": score
            }
            for i, (combo, score) in enumerate(combo_rank)
        ])
        combo_df.to_csv(os.path.join(out_dir, "front_combos_{}.csv".format(issue)), index=False, encoding="utf-8-sig")

        # 注单明细（最佳前区）
        ticket_df = pd.DataFrame([
            {
                "issue": issue,
                "front": " ".join(["{:02d}".format(x) for x in t["front"]]),
                "back": "{:02d} {:02d}".format(t["back"][0], t["back"][1])
            }
            for t in tickets
        ])
        ticket_df.to_csv(os.path.join(out_dir, "tickets_{}.csv".format(issue)), index=False, encoding="utf-8-sig")

        logger.info("结果已保存: {}".format(result_path))


if __name__ == "__main__":
    main()
