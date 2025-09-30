"""
Recipes - 配方管理
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Recipes - MoldingFlow AI",
    page_icon="📋",
    layout="wide"
)

st.title("📋 配方管理")
st.markdown("---")

# 模拟配方数据
@st.cache_data
def load_recipes_data():
    """加载配方数据"""
    recipes = []
    for i in range(15):
        recipe_id = f"RECIPE-{2024001 + i}"
        status = np.random.choice(['草稿', '待审批', '已批准', '已发布', '已废弃'], p=[0.2, 0.3, 0.3, 0.15, 0.05])
        product = np.random.choice(['BGA-256', 'QFP-144', 'LGA-169', 'CSP-100'])
        version = f"v{np.random.randint(1, 5)}.{np.random.randint(0, 10)}"
        
        recipes.append({
            'recipe_id': recipe_id,
            'product': product,
            'version': version,
            'status': status,
            'created_by': np.random.choice(['PE-001', 'PE-002', 'PE-003']),
            'created_time': datetime.now() - timedelta(days=np.random.randint(1, 30)),
            'approved_by': np.random.choice(['Manager-001', 'Manager-002', '']) if status in ['已批准', '已发布'] else '',
            'approved_time': datetime.now() - timedelta(days=np.random.randint(1, 15)) if status in ['已批准', '已发布'] else None,
            'usage_count': np.random.randint(0, 100),
            'success_rate': np.random.normal(95, 3)
        })
    
    return pd.DataFrame(recipes)

# 加载数据
recipes_df = load_recipes_data()

# 筛选器
st.subheader("🔍 配方筛选")
col1, col2, col3, col4 = st.columns(4)

with col1:
    status_filter = st.selectbox("状态", ["全部"] + list(recipes_df['status'].unique()))

with col2:
    product_filter = st.selectbox("产品", ["全部"] + list(recipes_df['product'].unique()))

with col3:
    creator_filter = st.selectbox("创建者", ["全部"] + list(recipes_df['created_by'].unique()))

with col4:
    date_filter = st.selectbox("时间范围", ["全部", "最近7天", "最近30天", "最近90天"])

# 应用筛选
filtered_recipes = recipes_df.copy()
if status_filter != "全部":
    filtered_recipes = filtered_recipes[filtered_recipes['status'] == status_filter]
if product_filter != "全部":
    filtered_recipes = filtered_recipes[filtered_recipes['product'] == product_filter]
if creator_filter != "全部":
    filtered_recipes = filtered_recipes[filtered_recipes['created_by'] == creator_filter]

if date_filter != "全部":
    days = {"最近7天": 7, "最近30天": 30, "最近90天": 90}[date_filter]
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_recipes = filtered_recipes[filtered_recipes['created_time'] >= cutoff_date]

st.markdown("---")

# 配方列表
st.subheader("📋 配方列表")

# 显示配方表格
display_df = filtered_recipes.copy()
display_df['created_time'] = display_df['created_time'].dt.strftime('%Y-%m-%d %H:%M')
if 'approved_time' in display_df.columns:
    display_df['approved_time'] = display_df['approved_time'].apply(
        lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notna(x) else '未审批'
    )
else:
    display_df['approved_time'] = '未审批'
display_df['success_rate'] = display_df['success_rate'].apply(lambda x: f"{x:.1f}%")

# 添加状态颜色 - 适配深色主题
def color_status(val):
    colors = {
        '草稿': 'background-color: #3a3a3a; color: #e0e0e0',
        '待审批': 'background-color: #4a3a1a; color: #ffd93d',
        '已批准': 'background-color: #1a4a1a; color: #6bcf7f',
        '已发布': 'background-color: #1a3a4a; color: #6bb6ff',
        '已废弃': 'background-color: #4a1a1a; color: #ff6b6b'
    }
    return colors.get(val, '')

styled_df = display_df.style.applymap(color_status, subset=['status'])
st.dataframe(styled_df, use_container_width=True)

# 配方详情
if not filtered_recipes.empty:
    st.subheader("🔍 配方详情")
    
    # 选择配方
    selected_recipe = st.selectbox("选择配方", filtered_recipes['recipe_id'].tolist())
    recipe_info = filtered_recipes[filtered_recipes['recipe_id'] == selected_recipe].iloc[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**配方ID**: {recipe_info['recipe_id']}")
        st.write(f"**产品**: {recipe_info['product']}")
        st.write(f"**版本**: {recipe_info['version']}")
        st.write(f"**状态**: {recipe_info['status']}")
        st.write(f"**创建者**: {recipe_info['created_by']}")
        st.write(f"**创建时间**: {recipe_info['created_time'].strftime('%Y-%m-%d %H:%M')}")
    
    with col2:
        st.write(f"**审批者**: {recipe_info['approved_by'] if recipe_info['approved_by'] else '未审批'}")
        if recipe_info['approved_time'] and pd.notna(recipe_info['approved_time']):
            st.write(f"**审批时间**: {recipe_info['approved_time'].strftime('%Y-%m-%d %H:%M')}")
        else:
            st.write("**审批时间**: 未审批")
        st.write(f"**使用次数**: {recipe_info['usage_count']}")
        st.write(f"**成功率**: {recipe_info['success_rate']:.1f}%")
        
        # 状态指示器
        status_colors = {
            '草稿': '🟡', '待审批': '🟠', '已批准': '🟢', 
            '已发布': '🔵', '已废弃': '🔴'
        }
        st.write(f"**状态**: {status_colors[recipe_info['status']]} {recipe_info['status']}")
    
    # 工艺参数详情
    st.subheader("工艺参数")
    
    # 模拟工艺参数
    if st.button("查看详细参数"):
        param_data = {
            '温度参数': {
                '升温速率': '2.0 °C/min',
                '保温温度': '175 °C',
                '保温时间': '30 min',
                '冷却速率': '1.0 °C/min'
            },
            '压力参数': {
                '升压速率': '10.0 MPa/min',
                '保压压力': '100 MPa',
                '保压时间': '20 min',
                '泄压速率': '5.0 MPa/min'
            },
            '材料参数': {
                '粘度': '3000 Pa·s',
                'CTE': '30 ppm/°C',
                '密度': '1.2 g/cm³',
                '热导率': '0.5 W/m·K'
            }
        }
        
        for category, params in param_data.items():
            st.write(f"**{category}**")
            for param, value in params.items():
                st.write(f"- {param}: {value}")
            st.write("")
    
    # 使用历史
    st.subheader("📈 使用历史")
    
    # 模拟使用历史数据
    usage_dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    usage_data = pd.DataFrame({
        'date': usage_dates,
        'usage_count': np.random.poisson(3, len(usage_dates)),
        'success_rate': np.random.normal(95, 2, len(usage_dates))
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_usage = px.bar(usage_data, x='date', y='usage_count', title='使用次数趋势')
        st.plotly_chart(fig_usage, use_container_width=True)
    
    with col2:
        fig_success = px.line(usage_data, x='date', y='success_rate', title='成功率趋势')
        st.plotly_chart(fig_success, use_container_width=True)
    
    # 操作按钮
    st.subheader("⚡ 配方操作")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("✏️ 编辑", type="primary"):
            st.success("进入编辑模式")
    
    with col2:
        if st.button("📋 复制"):
            st.info("配方已复制")
    
    with col3:
        if st.button("导出"):
            st.info("配方已导出")
    
    with col4:
        if st.button("版本管理"):
            st.info("版本管理已打开")
    
    with col5:
        if st.button("🗑️ 删除"):
            st.error("配方已删除")

# 创建新配方
st.subheader("➕ 创建新配方")
with st.expander("点击展开"):
    col1, col2 = st.columns(2)
    
    with col1:
        new_product = st.selectbox("产品", ["BGA-256", "QFP-144", "LGA-169", "CSP-100"])
        new_version = st.text_input("版本", "v1.0")
        new_creator = st.selectbox("创建者", ["PE-001", "PE-002", "PE-003"])
    
    with col2:
        new_description = st.text_area("描述", "新配方描述...")
        new_priority = st.selectbox("优先级", ["低", "中", "高"])
    
    if st.button("✅ 创建配方", type="primary"):
        st.success("新配方已创建")

# 统计信息
st.subheader("统计信息")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("总配方数", len(filtered_recipes))

with col2:
    st.metric("已发布", len(filtered_recipes[filtered_recipes['status'] == '已发布']))

with col3:
    st.metric("待审批", len(filtered_recipes[filtered_recipes['status'] == '待审批']))

with col4:
    avg_success = filtered_recipes['success_rate'].mean()
    st.metric("平均成功率", f"{avg_success:.1f}%")
