"""
Lots - 批次与派工（在制品、设备占用、切换损失）
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Lots - MoldingFlow AI",
    page_icon="📄",
    layout="wide"
)

st.title("批次与派工")
st.markdown("---")

# 模拟批次数据
@st.cache_data
def load_lots_data():
    """加载批次/派工/设备占用数据（模拟）"""
    lots = []
    for i in range(24):
        lot_id = f"LOT-{2024001 + i}"
        status = np.random.choice(['排队', '派工中', '生产中', '完成', '暂停'], p=[0.15, 0.15, 0.45, 0.2, 0.05])
        product = np.random.choice(['BGA-256', 'QFP-144', 'LGA-169', 'CSP-100'])
        equipment = f"MOLD-{np.random.randint(1, 5):03d}"
        setup_loss = np.random.choice([10, 20, 30, 45], p=[0.4, 0.3, 0.2, 0.1])  # 切换损失（分钟）
        
        lots.append({
            'lot_id': lot_id,
            'product': product,
            'status': status,
            'equipment': equipment,
            'start_time': datetime.now() - timedelta(hours=np.random.randint(1, 72)),
            'progress': np.random.randint(0, 101),
            'predicted_warpage': np.random.normal(0.08, 0.02),
            'predicted_void': np.random.normal(0.02, 0.005),
            'risk_level': np.random.choice(['低', '中', '高'], p=[0.6, 0.3, 0.1]),
            'priority': np.random.choice(['低', '中', '高'], p=[0.3, 0.5, 0.2]),
            'setup_loss_min': setup_loss
        })
    
    equipment_usage = pd.DataFrame({
        'equipment_id': [f"MOLD-{i:03d}" for i in range(1, 5)],
        'running': np.random.randint(1, 6, 4),
        'queued': np.random.randint(0, 4, 4),
        'changeover_min': np.random.choice([10, 20, 30, 45], 4)
    })
    
    return pd.DataFrame(lots), equipment_usage

# 加载数据
lots_df, equipment_usage = load_lots_data()

# 筛选器
st.subheader("🔍 批次筛选")
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_filter = st.selectbox("状态", ["全部"] + list(lots_df['status'].unique()))

with col2:
    product_filter = st.selectbox("产品", ["全部"] + list(lots_df['product'].unique()))

with col3:
    equipment_filter = st.selectbox("设备", ["全部"] + list(lots_df['equipment'].unique()))

with col4:
    risk_filter = st.selectbox("风险等级", ["全部"] + list(lots_df['risk_level'].unique()))

# 应用筛选
filtered_lots = lots_df.copy()
if status_filter != "全部":
    filtered_lots = filtered_lots[filtered_lots['status'] == status_filter]
if product_filter != "全部":
    filtered_lots = filtered_lots[filtered_lots['product'] == product_filter]
if equipment_filter != "全部":
    filtered_lots = filtered_lots[filtered_lots['equipment'] == equipment_filter]
if risk_filter != "全部":
    filtered_lots = filtered_lots[filtered_lots['risk_level'] == risk_filter]

st.markdown("---")

# 批次列表
st.subheader("📋 批次/派工列表")

# 显示批次表格
display_df = filtered_lots.copy()
display_df['start_time'] = display_df['start_time'].dt.strftime('%Y-%m-%d %H:%M')
display_df['predicted_warpage'] = display_df['predicted_warpage'].apply(lambda x: f"{x:.3f}mm")
display_df['predicted_void'] = display_df['predicted_void'].apply(lambda x: f"{x:.1%}")

# 添加颜色编码 - 适配深色主题
def color_risk(val):
    if val == '高':
        return 'background-color: #4a1a1a; color: #ff6b6b'
    elif val == '中':
        return 'background-color: #4a3a1a; color: #ffd93d'
    else:
        return 'background-color: #1a4a1a; color: #6bcf7f'

def color_priority(val):
    if val == '高':
        return 'background-color: #4a1a1a; color: #ff6b6b'
    elif val == '中':
        return 'background-color: #4a3a1a; color: #ffd93d'
    else:
        return 'background-color: #1a4a1a; color: #6bcf7f'

styled_df = display_df.style.applymap(color_risk, subset=['risk_level']).applymap(color_priority, subset=['priority'])
st.dataframe(styled_df, use_container_width=True)

# 批次详情
if not filtered_lots.empty:
    st.subheader("🔍 批次详情与派工")
    
    # 选择批次
    selected_lot = st.selectbox("选择批次", filtered_lots['lot_id'].tolist())
    lot_info = filtered_lots[filtered_lots['lot_id'] == selected_lot].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**批次ID**: {lot_info['lot_id']}")
        st.write(f"**产品**: {lot_info['product']}")
        st.write(f"**状态**: {lot_info['status']}")
        st.write(f"**设备**: {lot_info['equipment']}")
        st.write(f"**进度**: {lot_info['progress']}%")
        
        # 进度条
        st.progress(lot_info['progress'] / 100)
    
    with col2:
        st.write(f"**预测翘曲**: {lot_info['predicted_warpage']:.3f}mm")
        st.write(f"**预测空洞率**: {lot_info['predicted_void']:.1%}")
        st.write(f"**风险等级**: {lot_info['risk_level']}")
        st.write(f"**开始时间**: {lot_info['start_time'].strftime('%Y-%m-%d %H:%M')}")
        
        # 风险指示器
        risk_color = {"高": "🔴", "中": "🟡", "低": "🟢"}
        st.write(f"**风险状态**: {risk_color[lot_info['risk_level']]} {lot_info['risk_level']}")
    
    # 3D可视化（模拟）
    st.subheader("3D可视化")
    
    if st.button("生成预测可视化"):
        # 模拟3D数据
        n_points = 100
        x = np.random.randn(n_points)
        y = np.random.randn(n_points)
        z = np.random.randn(n_points) * 0.1
        
        # 翘曲热力图
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="翘曲 (mm)")
            )
        ))
        
        fig.update_layout(
            title=f"批次 {selected_lot} 翘曲预测",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="翘曲 (mm)"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 操作按钮
    st.subheader("⚡ 批次操作")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ 开始生产", type="primary"):
            st.success("批次已开始生产")
    
    with col2:
        if st.button("⏸️ 暂停"):
            st.warning("批次已暂停")
    
    with col3:
        if st.button("⏹️ 停止"):
            st.error("批次已停止")
    
    with col4:
        if st.button("生成报告"):
            st.info("报告已生成")

    # 派工建议（基于切换损失和优先级的简单规则）
    st.subheader("🧭 派工建议")
    rule_tip = "优先派高优先级，若切换损失>30min则建议批内继续，以降低换型损失"
    st.caption(rule_tip)
    st.table(
        lots_df.sort_values(["priority", "setup_loss_min"], ascending=[False, True])[
            ["lot_id", "equipment", "priority", "setup_loss_min", "status"]
        ].head(5)
    )

st.subheader("🏭 设备占用概览")
ec1, ec2 = st.columns(2)
with ec1:
    # 重构数据为长格式以支持分组柱状图
    usage_melted = equipment_usage.melt(id_vars=['equipment_id'], 
                                       value_vars=['running', 'queued'],
                                       var_name='status', value_name='count')
    fig_q = px.bar(usage_melted, x='equipment_id', y='count', color='status',
                   barmode='group', title='设备运行与排队批次')
    st.plotly_chart(fig_q, use_container_width=True)
with ec2:
    fig_c = px.bar(equipment_usage, x='equipment_id', y='changeover_min', title='平均换型损失 (min)')
    st.plotly_chart(fig_c, use_container_width=True)

# 统计信息
st.subheader("统计信息")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("总批次数", len(filtered_lots))
with col2:
    st.metric("生产中", len(filtered_lots[filtered_lots['status'].isin(['派工中','生产中'])]))
with col3:
    st.metric("高风险", len(filtered_lots[filtered_lots['risk_level'] == '高']))
with col4:
    avg_warpage = filtered_lots['predicted_warpage'].mean()
    st.metric("平均翘曲", f"{avg_warpage:.3f}mm")
