"""
Metrology - 量测管理
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Metrology - MoldingFlow AI",
    page_icon="📏",
    layout="wide"
)

st.title("📏 量测管理")
st.markdown("---")

# 模拟量测数据
@st.cache_data
def load_metrology_data():
    """加载量测数据"""
    # 生成量测数据
    measurements = []
    for i in range(50):
        lot_id = f"LOT-{2024001 + i}"
        product = np.random.choice(['BGA-256', 'QFP-144', 'LGA-169', 'CSP-100'])
        equipment = np.random.choice(['MOLD-001', 'MOLD-002', 'MOLD-003'])
        
        # 翘曲量测
        warpage_x = np.random.normal(0, 0.05)
        warpage_y = np.random.normal(0, 0.05)
        warpage_z = np.random.normal(0, 0.02)
        warpage_magnitude = np.sqrt(warpage_x**2 + warpage_y**2 + warpage_z**2)
        
        # 空洞量测
        void_rate = np.random.normal(2.0, 0.5)
        void_count = np.random.poisson(5)
        
        # 尺寸量测
        length = np.random.normal(10.0, 0.1)
        width = np.random.normal(10.0, 0.1)
        height = np.random.normal(2.0, 0.05)
        
        measurements.append({
            'measurement_id': f"MEAS-{2024001 + i}",
            'lot_id': lot_id,
            'product': product,
            'equipment': equipment,
            'measurement_time': datetime.now() - timedelta(hours=np.random.randint(1, 168)),
            'warpage_x': warpage_x,
            'warpage_y': warpage_y,
            'warpage_z': warpage_z,
            'warpage_magnitude': warpage_magnitude,
            'void_rate': void_rate,
            'void_count': void_count,
            'length': length,
            'width': width,
            'height': height,
            'operator': np.random.choice(['OP-001', 'OP-002', 'OP-003']),
            'status': np.random.choice(['合格', '不合格', '待复测'], p=[0.8, 0.15, 0.05])
        })
    
    return pd.DataFrame(measurements)

# 加载数据
measurements_df = load_metrology_data()

# 筛选器
st.subheader("🔍 量测筛选")
col1, col2, col3, col4 = st.columns(4)

with col1:
    product_filter = st.selectbox("产品", ["全部"] + list(measurements_df['product'].unique()))

with col2:
    equipment_filter = st.selectbox("设备", ["全部"] + list(measurements_df['equipment'].unique()))

with col3:
    status_filter = st.selectbox("状态", ["全部"] + list(measurements_df['status'].unique()))

with col4:
    time_range = st.selectbox("时间范围", ["最近24小时", "最近7天", "最近30天", "自定义"])

# 应用筛选
filtered_measurements = measurements_df.copy()

if product_filter != "全部":
    filtered_measurements = filtered_measurements[filtered_measurements['product'] == product_filter]
if equipment_filter != "全部":
    filtered_measurements = filtered_measurements[filtered_measurements['equipment'] == equipment_filter]
if status_filter != "全部":
    filtered_measurements = filtered_measurements[filtered_measurements['status'] == status_filter]

if time_range != "自定义":
    hours = {"最近24小时": 24, "最近7天": 168, "最近30天": 720}[time_range]
    cutoff_time = datetime.now() - timedelta(hours=hours)
    filtered_measurements = filtered_measurements[filtered_measurements['measurement_time'] >= cutoff_time]

st.markdown("---")

# 量测列表
st.subheader("📋 量测列表")

# 显示量测表格
display_df = filtered_measurements.copy()
display_df['measurement_time'] = display_df['measurement_time'].dt.strftime('%Y-%m-%d %H:%M')
display_df['warpage_magnitude'] = display_df['warpage_magnitude'].apply(lambda x: f"{x:.3f}mm")
display_df['void_rate'] = display_df['void_rate'].apply(lambda x: f"{x:.1f}%")

# 添加状态颜色 - 适配深色主题
def color_status(val):
    colors = {
        '合格': 'background-color: #1a4a1a; color: #6bcf7f',
        '不合格': 'background-color: #4a1a1a; color: #ff6b6b',
        '待复测': 'background-color: #4a3a1a; color: #ffd93d'
    }
    return colors.get(val, '')

styled_df = display_df.style.applymap(color_status, subset=['status'])
st.dataframe(styled_df, use_container_width=True)

# 量测详情
if not filtered_measurements.empty:
    st.subheader("🔍 量测详情")
    
    # 选择量测
    selected_measurement = st.selectbox("选择量测", filtered_measurements['measurement_id'].tolist())
    measurement_info = filtered_measurements[filtered_measurements['measurement_id'] == selected_measurement].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**量测ID**: {measurement_info['measurement_id']}")
        st.write(f"**批次ID**: {measurement_info['lot_id']}")
        st.write(f"**产品**: {measurement_info['product']}")
        st.write(f"**设备**: {measurement_info['equipment']}")
        st.write(f"**量测时间**: {measurement_info['measurement_time'].strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**操作员**: {measurement_info['operator']}")
    
    with col2:
        st.write(f"**翘曲X**: {measurement_info['warpage_x']:.3f}mm")
        st.write(f"**翘曲Y**: {measurement_info['warpage_y']:.3f}mm")
        st.write(f"**翘曲Z**: {measurement_info['warpage_z']:.3f}mm")
        st.write(f"**翘曲幅度**: {measurement_info['warpage_magnitude']:.3f}mm")
        st.write(f"**空洞率**: {measurement_info['void_rate']:.1f}%")
        st.write(f"**空洞数量**: {measurement_info['void_count']}")
        
        # 状态指示器
        status_colors = {
            '合格': '🟢', '不合格': '🔴', '待复测': '🟡'
        }
        st.write(f"**状态**: {status_colors[measurement_info['status']]} {measurement_info['status']}")
    
    # 3D可视化
    st.subheader("3D可视化")
    
    if st.button("生成3D可视化"):
        # 模拟3D数据
        n_points = 100
        x = np.random.randn(n_points) * 5
        y = np.random.randn(n_points) * 5
        z = np.random.randn(n_points) * 0.1 + measurement_info['warpage_magnitude']
        
        # 翘曲3D图
        fig_3d = go.Figure(data=go.Scatter3d(
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
        
        fig_3d.update_layout(
            title=f"量测 {selected_measurement} 3D翘曲图",
            scene=dict(
                xaxis_title="X (mm)",
                yaxis_title="Y (mm)",
                zaxis_title="翘曲 (mm)"
            )
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # 尺寸分布
    st.subheader("📐 尺寸分布")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 翘曲分布
        fig_warpage = px.histogram(filtered_measurements, x='warpage_magnitude', 
                                 title='翘曲幅度分布', nbins=20)
        st.plotly_chart(fig_warpage, use_container_width=True)
    
    with col2:
        # 空洞率分布
        fig_void = px.histogram(filtered_measurements, x='void_rate', 
                               title='空洞率分布', nbins=20)
        st.plotly_chart(fig_void, use_container_width=True)
    
    # 操作按钮
    st.subheader("⚡ 量测操作")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("生成报告", type="primary"):
            st.success("量测报告已生成")
    
    with col2:
        if st.button("重新量测"):
            st.info("重新量测已安排")
    
    with col3:
        if st.button("导出数据"):
            st.info("数据已导出")
    
    with col4:
        if st.button("📧 发送通知"):
            st.info("通知已发送")

# 统计分析
st.subheader("统计分析")

# 按产品统计
product_stats = filtered_measurements.groupby('product').agg({
    'warpage_magnitude': ['mean', 'std', 'min', 'max'],
    'void_rate': ['mean', 'std', 'min', 'max'],
    'status': lambda x: (x == '合格').sum() / len(x) * 100
}).round(3)

st.write("**按产品统计**")
st.dataframe(product_stats, use_container_width=True)

# 按设备统计
equipment_stats = filtered_measurements.groupby('equipment').agg({
    'warpage_magnitude': ['mean', 'std'],
    'void_rate': ['mean', 'std'],
    'status': lambda x: (x == '合格').sum() / len(x) * 100
}).round(3)

st.write("**按设备统计**")
st.dataframe(equipment_stats, use_container_width=True)

# 趋势分析
st.subheader("📈 趋势分析")

# 时间序列图
col1, col2 = st.columns(2)

with col1:
    fig_trend_warpage = px.line(filtered_measurements, x='measurement_time', y='warpage_magnitude',
                               title='翘曲趋势', color='product')
    st.plotly_chart(fig_trend_warpage, use_container_width=True)

with col2:
    fig_trend_void = px.line(filtered_measurements, x='measurement_time', y='void_rate',
                            title='空洞率趋势', color='product')
    st.plotly_chart(fig_trend_void, use_container_width=True)

# 散点图分析
st.subheader("🔍 散点图分析")

fig_scatter = px.scatter(filtered_measurements, x='warpage_magnitude', y='void_rate',
                        color='product', size='void_count',
                        title='翘曲 vs 空洞率',
                        hover_data=['lot_id', 'equipment', 'status'])
st.plotly_chart(fig_scatter, use_container_width=True)

# 统计信息
st.subheader("统计信息")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("总量测数", len(filtered_measurements))

with col2:
    st.metric("合格率", f"{(filtered_measurements['status'] == '合格').sum() / len(filtered_measurements) * 100:.1f}%")

with col3:
    avg_warpage = filtered_measurements['warpage_magnitude'].mean()
    st.metric("平均翘曲", f"{avg_warpage:.3f}mm")

with col4:
    avg_void = filtered_measurements['void_rate'].mean()
    st.metric("平均空洞率", f"{avg_void:.1f}%")
