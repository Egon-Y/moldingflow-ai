"""
Optimize - 智能优化
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Optimize - MoldingFlow AI",
    page_icon="🔧",
    layout="wide"
)

st.title("智能优化")
st.markdown("---")

# 优化目标设置
st.subheader("优化目标设置")
col1, col2 = st.columns(2)

with col1:
    st.write("**目标权重**")
    warpage_weight = st.slider("翘曲权重", 0.0, 1.0, 0.4, 0.1)
    void_weight = st.slider("空洞权重", 0.0, 1.0, 0.3, 0.1)
    cycle_weight = st.slider("周期权重", 0.0, 1.0, 0.2, 0.1)
    energy_weight = st.slider("能耗权重", 0.0, 1.0, 0.1, 0.1)

with col2:
    st.write("**约束条件**")
    max_warpage = st.number_input("最大翘曲 (mm)", 0.01, 0.5, 0.1, 0.01)
    max_void = st.number_input("最大空洞率 (%)", 0.1, 10.0, 2.0, 0.1)
    max_cycle = st.number_input("最大周期 (min)", 1.0, 60.0, 30.0, 1.0)
    max_energy = st.number_input("最大能耗 (kWh)", 1.0, 100.0, 50.0, 1.0)

# 当前参数
st.subheader("⚙️ 当前工艺参数")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**温度参数**")
    temp_ramp = st.slider("升温速率 (°C/min)", 0.5, 5.0, 2.0, 0.1)
    temp_hold = st.slider("保温温度 (°C)", 150, 200, 175, 1)
    temp_time = st.slider("保温时间 (min)", 5, 60, 30, 1)

with col2:
    st.write("**压力参数**")
    pressure_ramp = st.slider("升压速率 (MPa/min)", 1.0, 20.0, 10.0, 0.5)
    pressure_hold = st.slider("保压压力 (MPa)", 50, 150, 100, 1)
    pressure_time = st.slider("保压时间 (min)", 5, 60, 20, 1)

with col3:
    st.write("**冷却参数**")
    cool_rate = st.slider("冷却速率 (°C/min)", 0.1, 2.0, 1.0, 0.1)
    cool_temp = st.slider("冷却温度 (°C)", 20, 100, 50, 1)
    cool_time = st.slider("冷却时间 (min)", 10, 120, 60, 1)

# 材料参数
st.subheader("材料参数")
col1, col2, col3 = st.columns(3)

with col1:
    viscosity = st.number_input("粘度 (Pa·s)", 1000, 10000, 3000, 100)
    cte = st.number_input("CTE (ppm/°C)", 10, 50, 30, 1)

with col2:
    density = st.number_input("密度 (g/cm³)", 1.0, 3.0, 1.2, 0.1)
    thermal_cond = st.number_input("热导率 (W/m·K)", 0.1, 2.0, 0.5, 0.1)

with col3:
    batch_id = st.text_input("材料批次ID", "EMC-2024-001")
    supplier = st.selectbox("供应商", ["供应商A", "供应商B", "供应商C"])

# 优化算法选择
st.subheader("优化算法")
algorithm = st.selectbox("选择算法", ["贝叶斯优化", "遗传算法", "粒子群优化", "模拟退火"])

# 开始优化
if st.button("开始优化", type="primary"):
    with st.spinner("正在优化参数..."):
        # 模拟优化过程
        import time
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"优化进度: {i+1}%")
            time.sleep(0.05)
        
        st.success("优化完成！")

# 优化结果
st.subheader("优化结果")

# 模拟优化结果
@st.cache_data
def generate_optimization_results():
    """生成优化结果"""
    # 生成多个候选方案
    candidates = []
    for i in range(10):
        candidates.append({
            '方案': f"方案 {i+1}",
            '翘曲 (mm)': np.random.normal(0.08, 0.02),
            '空洞率 (%)': np.random.normal(2.0, 0.5),
            '周期 (min)': np.random.normal(25, 5),
            '能耗 (kWh)': np.random.normal(40, 10),
            '综合评分': np.random.normal(85, 10)
        })
    
    return pd.DataFrame(candidates)

results_df = generate_optimization_results()

# 显示结果表格
st.dataframe(results_df, use_container_width=True)

# 帕累托前沿
st.subheader("📈 帕累托前沿")
col1, col2 = st.columns(2)

with col1:
    # 翘曲 vs 空洞率
    fig1 = px.scatter(results_df, x='翘曲 (mm)', y='空洞率 (%)',
                     title='翘曲 vs 空洞率',
                     hover_data=['方案', '综合评分'])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # 周期 vs 能耗
    fig2 = px.scatter(results_df, x='周期 (min)', y='能耗 (kWh)',
                     title='周期 vs 能耗',
                     hover_data=['方案', '综合评分'])
    st.plotly_chart(fig2, use_container_width=True)

# 推荐方案
st.subheader("⭐ 推荐方案")
best_solution = results_df.loc[results_df['综合评分'].idxmax()]

col1, col2 = st.columns(2)

with col1:
    st.write("**最优参数组合**")
    st.write(f"翘曲: {best_solution['翘曲 (mm)']:.3f}mm")
    st.write(f"空洞率: {best_solution['空洞率 (%)']:.1f}%")
    st.write(f"周期: {best_solution['周期 (min)']:.1f}min")
    st.write(f"能耗: {best_solution['能耗 (kWh)']:.1f}kWh")
    st.write(f"综合评分: {best_solution['综合评分']:.1f}")

with col2:
    # 参数对比雷达图
    categories = ['翘曲', '空洞率', '周期', '能耗']
    values = [
        best_solution['翘曲 (mm)'] * 10,  # 归一化
        best_solution['空洞率 (%)'] * 10,
        best_solution['周期 (min)'] / 2,
        best_solution['能耗 (kWh)'] / 2
    ]
    
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='推荐方案'
    ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True,
        title="参数对比雷达图"
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

# 操作按钮
st.subheader("⚡ 操作")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("✅ 采用推荐方案", type="primary"):
        st.success("方案已采用")

with col2:
    if st.button("📋 生成e-Recipe"):
        st.info("e-Recipe已生成")

with col3:
    if st.button("详细分析"):
        st.info("详细分析已生成")

with col4:
    if st.button("💾 保存方案"):
        st.success("方案已保存")

# 历史优化记录
st.subheader("📚 历史优化记录")
# 生成基于当前时间的历史记录
now = datetime.now()
history_dates = pd.date_range(start=now - timedelta(days=9), end=now, freq='D')
history_data = pd.DataFrame({
    '时间': history_dates,
    '产品': ['BGA-256'] * 5 + ['QFP-144'] * 3 + ['LGA-169'] * 2,
    '优化前评分': np.random.normal(70, 10, 10),
    '优化后评分': np.random.normal(85, 8, 10),
    '改善幅度': np.random.normal(15, 5, 10)
})

st.dataframe(history_data, use_container_width=True)
