# -*- coding:utf-8 -*-
"""
Streamlit Web 应用：大乐透 5+12 策略工作台
"""
import os
import sys
import glob
import json
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
DATA_DLT_PATH = os.path.join(PROJECT_DIR, "data", "dlt", "data.csv")


st.set_page_config(
    page_title="大乐透 5+12 策略平台",
    page_icon="🎯",
    layout="wide",
)


st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top right, #111827 0%, #0b1020 45%, #05070f 100%);
        color: #f5f7ff;
    }
    .stApp h1, .stApp h2, .stApp h3 {
        color: #e7ebff;
        letter-spacing: 0.3px;
    }
    .metric-card {
        border-radius: 14px;
        padding: 14px 18px;
        background: linear-gradient(135deg, rgba(23,32,64,0.85), rgba(14,20,40,0.85));
        border: 1px solid rgba(109,130,255,0.28);
    }
    .small-note {
        color: #9fb0ff;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🎯 大乐透 5+12 智能策略平台")
st.caption("LSTM + RF + 统计融合 · 组合过滤 · 回测调参 · 一站式操作")


def run_python_script(script_name, args=None):
    if args is None:
        args = []
    script_path = os.path.join(PROJECT_DIR, script_name)
    cmd = [sys.executable, script_path] + args
    p = subprocess.run(
        cmd,
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    merged_output = (p.stdout or "") + "\n" + (p.stderr or "")
    return p.returncode, merged_output.strip()


def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def render_command_result(title, returncode, output):
    if returncode == 0:
        st.success(f"✅ {title} 执行成功")
    else:
        st.error(f"❌ {title} 执行失败（exit code={returncode}）")
    st.code(output if output else "(无输出)", language="bash")


def load_latest_inference_json():
    p = latest_file(os.path.join(OUTPUT_DIR, "inference_plus_*.json"))
    if not p:
        return None, None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return p, data


def load_latest_tickets_csv(issue):
    p = os.path.join(OUTPUT_DIR, f"tickets_{issue}.csv")
    if os.path.exists(p):
        return p, pd.read_csv(p)
    return None, None


with st.sidebar:
    st.subheader("⚙️ 快速导航")
    st.markdown("- 数据更新\n- 模型训练\n- 5+12预测\n- 回测调参")

    st.markdown("---")
    st.subheader("🚀 启动方式")
    st.code("streamlit run app_streamlit.py", language="bash")

    st.markdown("---")
    st.markdown(
        "<div class='small-note'>"
        "提示：若首次运行较慢，通常是 TensorFlow/模型加载耗时。"
        "</div>",
        unsafe_allow_html=True,
    )


tab_data, tab_train, tab_infer, tab_backtest = st.tabs([
    "📥 数据更新",
    "🧠 模型训练",
    "🔮 5+12预测",
    "📈 回测调参",
])


with tab_data:
    st.subheader("获取最新大乐透历史数据")
    st.write("点击按钮后将调用 `get_data.py --name dlt`。")

    if st.button("更新大乐透数据", use_container_width=True):
        rc, out = run_python_script("get_data.py", ["--name", "dlt"])
        render_command_result("更新大乐透数据", rc, out)

    st.markdown("### 当前数据预览")
    if os.path.exists(DATA_DLT_PATH):
        df = pd.read_csv(DATA_DLT_PATH)
        st.write(f"总行数：{len(df)}")
        st.dataframe(df.head(15), use_container_width=True)
    else:
        st.warning("尚未找到 `data/dlt/data.csv`，请先更新数据。")


with tab_train:
    st.subheader("模型训练")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 训练原始 LSTM 模型（dlt）")
        split_lstm = st.slider("LSTM 训练集比例", 0.50, 0.95, 0.80, 0.01)
        if st.button("开始训练 LSTM", use_container_width=True):
            rc, out = run_python_script(
                "run_train_model.py",
                ["--name", "dlt", "--train_test_split", str(split_lstm)],
            )
            render_command_result("训练 LSTM", rc, out)

    with c2:
        st.markdown("#### 训练 RF 前区模型")
        split_rf = st.slider("RF 训练集比例", 0.50, 0.95, 0.80, 0.01)
        min_history = st.number_input("RF最小历史期数", min_value=60, max_value=400, value=120, step=10)
        if st.button("开始训练 RF", use_container_width=True):
            rc, out = run_python_script(
                "run_train_rf_model.py",
                [
                    "--name", "dlt",
                    "--train_test_split", str(split_rf),
                    "--min_history", str(int(min_history)),
                ],
            )
            render_command_result("训练 RF", rc, out)


with tab_infer:
    st.subheader("5+12 智能预测")

    col1, col2, col3 = st.columns(3)
    use_lstm = col1.checkbox("使用LSTM打分", value=True)
    use_rf = col2.checkbox("使用RF打分", value=True)
    save_result = col3.checkbox("保存结果文件", value=True)

    if st.button("执行预测", type="primary", use_container_width=True):
        rc, out = run_python_script(
            "inference_plus.py",
            [
                "--name", "dlt",
                "--use_lstm", "1" if use_lstm else "0",
                "--use_rf", "1" if use_rf else "0",
                "--save", "1" if save_result else "0",
            ],
        )
        render_command_result("5+12预测", rc, out)

    result_path, infer_data = load_latest_inference_json()
    if infer_data:
        st.markdown("### 最近一次预测结果")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("目标期号", infer_data.get("next_issue"))
        m2.metric("总注数", infer_data.get("total_bets"))
        m3.metric("总成本", f"{infer_data.get('total_cost')} 元")
        m4.metric("后区组合", infer_data.get("back_pairs_count"))

        st.write("前区 Top 候选：", infer_data.get("top_candidates"))
        st.write("最佳前区 5 码：", infer_data.get("best_front_combo"))

        with st.expander("查看融合分Top10明细", expanded=False):
            top10 = infer_data.get("component_top10", {})
            st.json(top10)

        issue = infer_data.get("next_issue")
        ticket_path, ticket_df = load_latest_tickets_csv(issue)
        if ticket_df is not None:
            st.markdown("### 注单明细（最佳前区 + 后区全包）")
            st.dataframe(ticket_df, use_container_width=True, height=320)

        st.caption(f"结果文件：{result_path}")
    else:
        st.info("还没有预测结果文件。执行一次预测后会在这里展示。")


with tab_backtest:
    st.subheader("历史回测 & 网格调参")

    c1, c2, c3 = st.columns(3)
    start_offset = c1.number_input("回测最近多少期", min_value=80, max_value=1000, value=200, step=20)
    use_rf_bt = c2.checkbox("回测启用RF融合", value=True)
    run_grid = c3.checkbox("执行网格搜索", value=True)

    c4, c5, c6 = st.columns(3)
    save_grid = c4.checkbox("保存网格结果", value=True)
    auto_apply_best = c5.checkbox("自动回写最优参数到config.py", value=False)
    apply_only_if_better = c6.checkbox("仅当优于当前参数才回写", value=True)

    if st.button("开始回测", use_container_width=True):
        args = [
            "--name", "dlt",
            "--start_offset", str(int(start_offset)),
            "--run_grid", "1" if run_grid else "0",
            "--use_rf", "1" if use_rf_bt else "0",
            "--save_grid", "1" if save_grid else "0",
            "--auto_apply_best", "1" if auto_apply_best else "0",
            "--apply_only_if_better", "1" if apply_only_if_better else "0",
        ]
        rc, out = run_python_script("backtest_plus.py", args)
        render_command_result("回测调参", rc, out)

    grid_path = os.path.join(OUTPUT_DIR, "backtest_grid_results.json")
    if os.path.exists(grid_path):
        with open(grid_path, "r", encoding="utf-8") as f:
            grid = json.load(f)
        if grid:
            st.markdown("### 最近一次网格搜索 Top 10")
            grid_df = pd.DataFrame(grid[:10])
            st.dataframe(grid_df, use_container_width=True)
            st.caption(f"网格结果文件：{grid_path}")
    else:
        st.info("尚未检测到网格结果文件。")


st.markdown("---")
st.caption(f"页面生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} · 当前解释器：{sys.executable}")
