"""
SPC - 统计过程控制
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="SPC - MoldingFlow AI",
    page_icon="📈",
    layout="wide"
)

st.title("统计过程控制")
st.markdown("---")

# 模拟SPC数据
@st.cache_data
def load_spc_data():
    """加载SPC数据"""
    # 生成时间序列数据
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
    
    # 翘曲数据
    warpage_data = pd.DataFrame({
        'timestamp': dates,
        'warpage': np.random.normal(0.08, 0.02, len(dates)),
        'product': np.random.choice(['BGA-256', 'QFP-144', 'LGA-169'], len(dates)),
        'equipment': np.random.choice(['MOLD-001', 'MOLD-002', 'MOLD-003'], len(dates)),
        'lot_id': [f"LOT-{np.random.randint(2024001, 2024020)}" for _ in range(len(dates))]
    })
    
    # 空洞率数据
    void_data = pd.DataFrame({
        'timestamp': dates,
        'void_rate': np.random.normal(2.0, 0.5, len(dates)),
        'product': np.random.choice(['BGA-256', 'QFP-144', 'LGA-169'], len(dates)),
        'equipment': np.random.choice(['MOLD-001', 'MOLD-002', 'MOLD-003'], len(dates)),
        'lot_id': [f"LOT-{np.random.randint(2024001, 2024020)}" for _ in range(len(dates))]
    })
    
    return warpage_data, void_data

# 加载数据
warpage_df, void_df = load_spc_data()

# 筛选器
st.subheader("🔍 数据筛选")
col1, col2, col3, col4 = st.columns(4)

with col1:
    product_filter = st.selectbox("产品", ["全部"] + list(warpage_df['product'].unique()))

with col2:
    equipment_filter = st.selectbox("设备", ["全部"] + list(warpage_df['equipment'].unique()))

with col3:
    time_range = st.selectbox("时间范围", ["最近24小时", "最近7天", "最近30天", "自定义"])

with col4:
    chart_type = st.selectbox("图表类型", ["Xbar-R", "EWMA", "CUSUM", "I-MR"])

# 应用筛选
filtered_warpage = warpage_df.copy()
filtered_void = void_df.copy()

if product_filter != "全部":
    filtered_warpage = filtered_warpage[filtered_warpage['product'] == product_filter]
    filtered_void = filtered_void[filtered_void['product'] == product_filter]

if equipment_filter != "全部":
    filtered_warpage = filtered_warpage[filtered_warpage['equipment'] == equipment_filter]
    filtered_void = filtered_void[filtered_void['equipment'] == equipment_filter]

if time_range != "自定义":
    hours = {"最近24小时": 24, "最近7天": 168, "最近30天": 720}[time_range]
    cutoff_time = datetime.now() - timedelta(hours=hours)
    filtered_warpage = filtered_warpage[filtered_warpage['timestamp'] >= cutoff_time]
    filtered_void = filtered_void[filtered_void['timestamp'] >= cutoff_time]

st.markdown("---")

# SPC图表
st.subheader("📈 SPC控制图")

# 翘曲控制图
col1, col2 = st.columns(2)

with col1:
    st.write("**翘曲控制图**")
    
    # 计算控制限
    warpage_mean = filtered_warpage['warpage'].mean()
    warpage_std = filtered_warpage['warpage'].std()
    ucl = warpage_mean + 3 * warpage_std
    lcl = warpage_mean - 3 * warpage_std
    
    # 创建控制图
    fig_warpage = go.Figure()
    
    # 数据点
    fig_warpage.add_trace(go.Scatter(
        x=filtered_warpage['timestamp'],
        y=filtered_warpage['warpage'],
        mode='lines+markers',
        name='翘曲值',
        line=dict(color='blue')
    ))
    
    # 中心线
    fig_warpage.add_hline(y=warpage_mean, line_dash="dash", line_color="green", 
                         annotation_text=f"中心线: {warpage_mean:.3f}")
    
    # 控制限
    fig_warpage.add_hline(y=ucl, line_dash="dash", line_color="red", 
                         annotation_text=f"UCL: {ucl:.3f}")
    fig_warpage.add_hline(y=lcl, line_dash="dash", line_color="red", 
                         annotation_text=f"LCL: {lcl:.3f}")
    
    fig_warpage.update_layout(
        title="翘曲控制图",
        xaxis_title="时间",
        yaxis_title="翘曲 (mm)",
        showlegend=True
    )
    
    st.plotly_chart(fig_warpage, use_container_width=True)

with col2:
    st.write("**空洞率控制图**")
    
    # 计算控制限
    void_mean = filtered_void['void_rate'].mean()
    void_std = filtered_void['void_rate'].std()
    ucl_void = void_mean + 3 * void_std
    lcl_void = void_mean - 3 * void_std
    
    # 创建控制图
    fig_void = go.Figure()
    
    # 数据点
    fig_void.add_trace(go.Scatter(
        x=filtered_void['timestamp'],
        y=filtered_void['void_rate'],
        mode='lines+markers',
        name='空洞率',
        line=dict(color='orange')
    ))
    
    # 中心线
    fig_void.add_hline(y=void_mean, line_dash="dash", line_color="green", 
                      annotation_text=f"中心线: {void_mean:.1f}%")
    
    # 控制限
    fig_void.add_hline(y=ucl_void, line_dash="dash", line_color="red", 
                      annotation_text=f"UCL: {ucl_void:.1f}%")
    fig_void.add_hline(y=lcl_void, line_dash="dash", line_color="red", 
                      annotation_text=f"LCL: {lcl_void:.1f}%")
    
    fig_void.update_layout(
        title="空洞率控制图",
        xaxis_title="时间",
        yaxis_title="空洞率 (%)",
        showlegend=True
    )
    
    st.plotly_chart(fig_void, use_container_width=True)

# 西格玛规则检查
st.subheader("🔍 西格玛规则检查")

# 模拟规则触发
rules_triggered = [
    {"rule": "规则1: 1点超出3σ", "count": 2, "severity": "高"},
    {"rule": "规则2: 连续9点在同一侧", "count": 1, "severity": "中"},
    {"rule": "规则3: 连续6点递增或递减", "count": 0, "severity": "低"},
    {"rule": "规则4: 连续14点交替上下", "count": 0, "severity": "低"},
    {"rule": "规则5: 连续3点中有2点超出2σ", "count": 1, "severity": "中"}
]

col1, col2 = st.columns(2)

with col1:
    st.write("**翘曲规则触发**")
    for rule in rules_triggered:
        if rule["count"] > 0:
            color = "🔴" if rule["severity"] == "高" else "🟡" if rule["severity"] == "中" else "🟢"
            st.write(f"{color} {rule['rule']}: {rule['count']}次")

with col2:
    st.write("**空洞率规则触发**")
    for rule in rules_triggered:
        if rule["count"] > 0:
            color = "🔴" if rule["severity"] == "高" else "🟡" if rule["severity"] == "中" else "🟢"
            st.write(f"{color} {rule['rule']}: {rule['count']}次")

# 异常分析
st.subheader("异常分析")

# 异常点检测
outliers_warpage = filtered_warpage[
    (filtered_warpage['warpage'] > ucl) | (filtered_warpage['warpage'] < lcl)
]
outliers_void = filtered_void[
    (filtered_void['void_rate'] > ucl_void) | (filtered_void['void_rate'] < lcl_void)
]

col1, col2 = st.columns(2)

with col1:
    st.write("**翘曲异常点**")
    if not outliers_warpage.empty:
        st.dataframe(outliers_warpage[['timestamp', 'warpage', 'product', 'equipment', 'lot_id']], 
                    use_container_width=True)
    else:
        st.success("无异常点")

with col2:
    st.write("**空洞率异常点**")
    if not outliers_void.empty:
        st.dataframe(outliers_void[['timestamp', 'void_rate', 'product', 'equipment', 'lot_id']], 
                    use_container_width=True)
    else:
        st.success("无异常点")

# 趋势分析
st.subheader("📈 趋势分析")

# 移动平均
window_size = st.slider("移动平均窗口", 5, 50, 20)

filtered_warpage['ma'] = filtered_warpage['warpage'].rolling(window=window_size).mean()
filtered_void['ma'] = filtered_void['void_rate'].rolling(window=window_size).mean()

col1, col2 = st.columns(2)

with col1:
    fig_ma_warpage = go.Figure()
    fig_ma_warpage.add_trace(go.Scatter(
        x=filtered_warpage['timestamp'],
        y=filtered_warpage['warpage'],
        mode='lines',
        name='原始数据',
        line=dict(color='lightblue')
    ))
    fig_ma_warpage.add_trace(go.Scatter(
        x=filtered_warpage['timestamp'],
        y=filtered_warpage['ma'],
        mode='lines',
        name=f'{window_size}点移动平均',
        line=dict(color='blue', width=2)
    ))
    fig_ma_warpage.update_layout(title="翘曲移动平均", xaxis_title="时间", yaxis_title="翘曲 (mm)")
    st.plotly_chart(fig_ma_warpage, use_container_width=True)

with col2:
    fig_ma_void = go.Figure()
    fig_ma_void.add_trace(go.Scatter(
        x=filtered_void['timestamp'],
        y=filtered_void['void_rate'],
        mode='lines',
        name='原始数据',
        line=dict(color='lightcoral')
    ))
    fig_ma_void.add_trace(go.Scatter(
        x=filtered_void['timestamp'],
        y=filtered_void['ma'],
        mode='lines',
        name=f'{window_size}点移动平均',
        line=dict(color='red', width=2)
    ))
    fig_ma_void.update_layout(title="空洞率移动平均", xaxis_title="时间", yaxis_title="空洞率 (%)")
    st.plotly_chart(fig_ma_void, use_container_width=True)

# 统计信息
st.subheader("统计信息")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("翘曲均值", f"{warpage_mean:.3f}mm")

with col2:
    st.metric("翘曲标准差", f"{warpage_std:.3f}mm")

with col3:
    st.metric("空洞率均值", f"{void_mean:.1f}%")

with col4:
    st.metric("空洞率标准差", f"{void_std:.1f}%")

# 操作按钮
st.subheader("操作")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("生成SPC报告", type="primary"):
        st.success("SPC报告已生成")

with col2:
    if st.button("设置告警"):
        st.info("告警设置已更新")

with col3:
    if st.button("导出数据"):
        st.info("数据已导出")

with col4:
    if st.button("刷新数据"):
        st.rerun()
