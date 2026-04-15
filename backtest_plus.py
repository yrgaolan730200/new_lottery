# -*- coding:utf-8 -*-
"""
大乐透 5+12 策略回测与调参
支持：
1) RF 融合回测
2) 网格搜索后自动回写 config.py 最优参数
3) 仅当优于当前参数时才回写（保护开关）
"""
import os
import re
import json
import argparse
import itertools

from loguru import logger

from config import name_path, data_file_name, plus_strategy
from feature_engineering import load_dlt_history
from inference_plus import build_ensemble_scores, generate_front_combos, maybe_load_rf_model


FRONT_COLS = ["红球_1", "红球_2", "红球_3", "红球_4", "红球_5"]
BACK_COLS = ["蓝球_1", "蓝球_2"]


def calc_period_reward(front_hit_count, target_back_set, payouts):
    """单个前区组合的66注收益（后区1~12全包）"""
    if front_hit_count != 3:
        return 0

    reward = 0
    for b1 in range(1, 13):
        for b2 in range(b1 + 1, 13):
            bh = len({b1, b2}.intersection(target_back_set))
            if bh == 2:
                reward += int(payouts["3+2"])
            elif bh == 1:
                reward += int(payouts["3+1"])
            else:
                reward += int(payouts["3+0"])
    return reward


def default_lstm_proxy_nums(history_df_asc):
    """回测场景下，用上一期号码作为LSTM代理输入"""
    if len(history_df_asc) == 0:
        return []
    row = history_df_asc.iloc[-1]
    return [int(row[c]) for c in FRONT_COLS]


def run_backtest_core(
    data_asc,
    start_idx,
    end_idx,
    top_n_front,
    max_front_combos,
    play_front_combos,
    ensemble_weights,
    rule_filters,
    payouts,
    rf_model=None,
    rf_meta=None,
):
    strategy = plus_strategy["dlt"]
    old_top_n = strategy["top_n_front"]
    old_max_front_combos = strategy["max_front_combos"]
    old_weights = strategy["ensemble_weights"].copy()
    old_rule_filters = strategy["rule_filters"].copy()

    strategy["top_n_front"] = int(top_n_front)
    strategy["max_front_combos"] = int(max_front_combos)
    strategy["ensemble_weights"] = ensemble_weights
    strategy["rule_filters"] = rule_filters

    records = []
    total_cost = 0
    total_reward = 0

    try:
        for idx in range(start_idx, end_idx + 1):
            hist = data_asc.iloc[:idx]
            target = data_asc.iloc[idx]

            last_front_set = set(int(hist.iloc[-1][c]) for c in FRONT_COLS)
            target_front_set = set(int(target[c]) for c in FRONT_COLS)
            target_back_set = set(int(target[c]) for c in BACK_COLS)

            lstm_proxy = default_lstm_proxy_nums(hist)
            ensemble_scores, _ = build_ensemble_scores(
                history_df_asc=hist,
                lstm_front_nums=lstm_proxy,
                rf_model=rf_model,
                rf_meta=rf_meta,
            )
            _, combos = generate_front_combos(ensemble_scores, last_front_set)

            if len(combos) == 0:
                continue

            played = combos[:max(1, int(play_front_combos))]
            period_cost = 132 * len(played)
            period_reward = 0
            best_front_hit = 0

            for combo, _ in played:
                front_hit = len(set(combo).intersection(target_front_set))
                best_front_hit = max(best_front_hit, front_hit)
                period_reward += calc_period_reward(front_hit, target_back_set, payouts)

            period_profit = period_reward - period_cost
            total_cost += period_cost
            total_reward += period_reward

            records.append({
                "issue": int(target["期数"]),
                "played_front_count": len(played),
                "best_front_hit": int(best_front_hit),
                "period_cost": int(period_cost),
                "period_reward": int(period_reward),
                "period_profit": int(period_profit),
                "cum_profit": int(total_reward - total_cost),
            })
    finally:
        strategy["top_n_front"] = old_top_n
        strategy["max_front_combos"] = old_max_front_combos
        strategy["ensemble_weights"] = old_weights
        strategy["rule_filters"] = old_rule_filters

    hit3_count = sum(1 for r in records if r["best_front_hit"] >= 3)
    return {
        "periods": len(records),
        "total_cost": int(total_cost),
        "total_reward": int(total_reward),
        "total_profit": int(total_reward - total_cost),
        "roi": float((total_reward - total_cost) / total_cost) if total_cost > 0 else 0.0,
        "hit3_or_more_rate": float(hit3_count / len(records)) if records else 0.0,
        "records": records,
    }


def run_grid_search(data_asc, start_idx, end_idx, base_strategy, rf_model=None, rf_meta=None):
    top_n_list = [8, 9, 10]
    play_front_list = [1, 2]
    weight_list = [
        {"lstm": 0.30, "rf": 0.50, "stat": 0.20},
        {"lstm": 0.35, "rf": 0.45, "stat": 0.20},
        {"lstm": 0.25, "rf": 0.55, "stat": 0.20},
        {"lstm": 0.40, "rf": 0.00, "stat": 0.60},
        {"lstm": 0.30, "rf": 0.00, "stat": 0.70},
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
            payouts=base_strategy["payouts"],
            rf_model=rf_model,
            rf_meta=rf_meta,
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


def apply_best_params_to_config(config_path, best):
    if not os.path.exists(config_path):
        raise Exception("config.py 不存在: {}".format(config_path))

    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    content, n1 = re.subn(r'("top_n_front"\s*:\s*)\d+', r'\g<1>{}'.format(int(best["top_n_front"])), content, count=1)
    content, n2 = re.subn(r'("play_front_combos"\s*:\s*)\d+', r'\g<1>{}'.format(int(best["play_front_combos"])), content, count=1)

    w = best["weights"]
    content, n3 = re.subn(r'("lstm"\s*:\s*)[0-9.]+', r'\g<1>{:.2f}'.format(float(w["lstm"])), content, count=1)
    content, n4 = re.subn(r'("rf"\s*:\s*)[0-9.]+', r'\g<1>{:.2f}'.format(float(w["rf"])), content, count=1)
    content, n5 = re.subn(r'("stat"\s*:\s*)[0-9.]+', r'\g<1>{:.2f}'.format(float(w["stat"])), content, count=1)

    if min(n1, n2, n3, n4, n5) == 0:
        raise Exception("回写失败：未匹配到全部参数键，请检查 config.py 结构")

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(content)


def extract_current_params_for_guard(config_path):
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
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="dlt", type=str)
    parser.add_argument("--start_offset", default=200, type=int)
    parser.add_argument("--run_grid", default=1, type=int)
    parser.add_argument("--use_rf", default=1, type=int)
    parser.add_argument("--auto_apply_best", default=0, type=int)
    parser.add_argument("--apply_only_if_better", default=1, type=int, help="仅当网格最优优于当前参数时才回写")
    parser.add_argument("--save_grid", default=1, type=int)
    args = parser.parse_args()

    if args.name != "dlt":
        raise Exception("backtest_plus.py 当前仅支持 dlt")

    data_path = "{}{}".format(name_path["dlt"]["path"], data_file_name)
    data_asc = load_dlt_history(data_path)

    start_idx = max(121, len(data_asc) - int(args.start_offset))
    end_idx = len(data_asc) - 1
    strategy = plus_strategy["dlt"]

    rf_model, rf_meta = (None, None)
    if int(args.use_rf) == 1:
        rf_model, rf_meta = maybe_load_rf_model()
        if rf_model is None:
            logger.warning("未加载到RF模型，回测将退化为非RF模式")

    logger.info("开始回测，期数范围: {} -> {}".format(int(data_asc.iloc[start_idx]["期数"]), int(data_asc.iloc[end_idx]["期数"])))

    result = run_backtest_core(
        data_asc=data_asc,
        start_idx=start_idx,
        end_idx=end_idx,
        top_n_front=strategy["top_n_front"],
        max_front_combos=strategy["max_front_combos"],
        play_front_combos=int(strategy.get("play_front_combos", 1)),
        ensemble_weights=strategy["ensemble_weights"],
        rule_filters=strategy["rule_filters"],
        payouts=strategy["payouts"],
        rf_model=rf_model,
        rf_meta=rf_meta,
    )

    logger.info("默认参数回测结束: periods={}, cost={}, reward={}, profit={}, roi={:.4f}, hit3+={:.4f}".format(
        result["periods"], result["total_cost"], result["total_reward"], result["total_profit"], result["roi"], result["hit3_or_more_rate"]
    ))

    if int(args.run_grid) == 1:
        gs = run_grid_search(data_asc, start_idx, end_idx, strategy, rf_model=rf_model, rf_meta=rf_meta)
        best = gs[0]
        logger.info("网格搜索最佳参数: top_n={}, play_front_combos={}, weights={}, profit={}, roi={:.4f}, hit3+={:.4f}".format(
            best["top_n_front"], best["play_front_combos"], best["weights"], best["profit"], best["roi"], best["hit3_or_more_rate"]
        ))

        if int(args.save_grid) == 1:
            save_grid_results(gs, os.path.join("outputs", "backtest_grid_results.json"))
            logger.info("网格搜索结果已保存: outputs/backtest_grid_results.json")

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
