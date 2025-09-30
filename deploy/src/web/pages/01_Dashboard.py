"""
Dashboard - 产线级管理看板（OEE/WIP/良率/SPC）
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Dashboard - MoldingFlow AI",
    page_icon="📈",
    layout="wide"
)

st.title("产线管理看板")
st.markdown("---")

# 模拟数据（贴近工厂产线）
@st.cache_data
def load_dashboard_data():
    now = datetime.now()
    dates = pd.date_range(start=now - timedelta(days=30), end=now, freq='D')

    # 关键 KPI
    kpis = {
        'oee': 89.7,
        'yield_rate': 96.2,
        'throughput_dph': 520,  # 每小时产出 Die per hour
        'wip_hours': 5.4,
        'downtime_rate': 12.4,
        'alerts_open': 5
    }

    # 良率与产出趋势
    trend = pd.DataFrame({
        'date': dates,
        'yield_rate': np.clip(np.random.normal(96, 1.0, len(dates)), 92, 99),
        'throughput_dph': np.clip(np.random.normal(520, 40, len(dates)), 400, 650)
    })

    # 设备稼动（示例 4 台模压机）
    equipment = pd.DataFrame({
        'equipment_id': ['MOLD-001', 'MOLD-002', 'MOLD-003', 'MOLD-004'],
        'oee': [92.3, 87.1, 88.5, 91.0],
        'availability': [95, 90, 91, 96],
        'performance': [93, 89, 90, 92],
        'quality': [99, 98, 98.5, 99.2],
        'downtime_hrs': [3.2, 6.5, 5.1, 2.7]
    })

    # SPC 示例数据（Xbar-R）
    spc_points = 25
    xbar = np.random.normal(0.08, 0.006, spc_points)
    rbar = np.abs(np.random.normal(0.012, 0.003, spc_points))
    spc = pd.DataFrame({
        'sample': list(range(1, spc_points + 1)),
        'xbar': xbar,
        'rbar': rbar
    })

    # 告警列表示例
    alerts = [
        {"time": (now - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M'), "level": "高", "message": "MOLD-002 停机 > 30min", "status": "处理中"},
        {"time": (now - timedelta(hours=5)).strftime('%Y-%m-%d %H:%M'), "level": "中", "message": "批次 LOT-2024015 空洞率偏高", "status": "分析中"},
        {"time": (now - timedelta(hours=9)).strftime('%Y-%m-%d %H:%M'), "level": "低", "message": "MOLD-003 计划保养提醒", "status": "待安排"}
    ]

    return kpis, trend, equipment, spc, alerts


kpis, trend, equipment, spc, alerts = load_dashboard_data()

# 顶部 KPI 区（OEE / 良率 / 产出 / WIP / 停机 / 未结告警）
st.subheader("📈 关键指标")
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    st.metric("OEE", f"{kpis['oee']:.1f}%", "+1.8%")
with c2:
    st.metric("良率", f"{kpis['yield_rate']:.1f}%", "+0.6%")
with c3:
    st.metric("产出 (DPH)", f"{kpis['throughput_dph']}", "+28")
with c4:
    st.metric("WIP 周转", f"{kpis['wip_hours']:.1f}h", "-0.3h")
with c5:
    st.metric("停机比例", f"{kpis['downtime_rate']:.1f}%", "-1.2%")
with c6:
    st.metric("未结告警", f"{kpis['alerts_open']}", "+1")

st.markdown("---")

# 左：良率趋势；右：产出趋势
st.subheader("📈 趋势分析")
lc, rc = st.columns(2)
with lc:
    fig_yield = px.line(trend, x='date', y='yield_rate', title='良率趋势', labels={'yield_rate': '良率 (%)'})
    st.plotly_chart(fig_yield, use_container_width=True)
with rc:
    fig_tp = px.line(trend, x='date', y='throughput_dph', title='产出 (DPH) 趋势', labels={'throughput_dph': 'DPH'})
    st.plotly_chart(fig_tp, use_container_width=True)

# 设备稼动与停机
st.subheader("🏭 设备稼动与停机")
ec1, ec2 = st.columns(2)
with ec1:
    # 重构数据为长格式以支持分组柱状图
    equipment_melted = equipment.melt(id_vars=['equipment_id'], 
                                     value_vars=['availability', 'performance', 'quality'],
                                     var_name='metric', value_name='value')
    fig_oee = px.bar(equipment_melted, x='equipment_id', y='value', color='metric',
                     barmode='group', title='设备 A/P/Q 构成 (%)')
    st.plotly_chart(fig_oee, use_container_width=True)
with ec2:
    fig_dt = px.bar(equipment, x='equipment_id', y='downtime_hrs', title='近 7 日停机小时')
    st.plotly_chart(fig_dt, use_container_width=True)

# SPC 控制图（Xbar、R）
st.subheader("📏 SPC 控制图")
sc1, sc2 = st.columns(2)
ucl_x, lcl_x = 0.095, 0.065
with sc1:
    fig_x = go.Figure()
    fig_x.add_trace(go.Scatter(x=spc['sample'], y=spc['xbar'], mode='lines+markers', name='Xbar'))
    fig_x.add_hline(y=ucl_x, line_dash='dash', line_color='red', annotation_text='UCL')
    fig_x.add_hline(y=lcl_x, line_dash='dash', line_color='red', annotation_text='LCL')
    fig_x.update_layout(title='Xbar 控制图', xaxis_title='样本', yaxis_title='均值 (mm)')
    st.plotly_chart(fig_x, use_container_width=True)

ucl_r, lcl_r = 0.02, 0.005
with sc2:
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(x=spc['sample'], y=spc['rbar'], mode='lines+markers', name='R'))
    fig_r.add_hline(y=ucl_r, line_dash='dash', line_color='red', annotation_text='UCL')
    fig_r.add_hline(y=lcl_r, line_dash='dash', line_color='red', annotation_text='LCL')
    fig_r.update_layout(title='R 控制图', xaxis_title='样本', yaxis_title='极差 (mm)')
    st.plotly_chart(fig_r, use_container_width=True)

# 告警列表
st.subheader("最新告警")
for alert in alerts:
    level_color = {"高": "🔴", "中": "🟡", "低": "🟢"}
    st.write(f"{level_color[alert['level']]} **{alert['level']}** - {alert['message']} ({alert['status']}) - {alert['time']}")

# 快速操作
st.subheader("快速操作")
oc1, oc2, oc3 = st.columns(3)
with oc1:
    if st.button("生成日报", type="primary"):
        st.success("日报已生成并发送")
with oc2:
    if st.button("🔍 深度分析"):
        st.info("启动深度分析...")
with oc3:
    if st.button("📧 发送告警"):
        st.warning("告警邮件已发送")
