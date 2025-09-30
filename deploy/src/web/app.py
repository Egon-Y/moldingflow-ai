"""
MoldingFlow AI - 主应用入口
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta


# 设置页面配置
st.set_page_config(
    page_title="MoldingFlow AI - 模压翘曲预测与智能推荐系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 主页面

def main():
    """主页面"""
    st.title("MoldingFlow AI")
    st.subheader("半导体模压 · 产线级 AI 平台")
    
    # 业务概览
    st.markdown("""
    ### 业务概览
    面向量产产线，覆盖从投产计划、派工流转、设备稼动到 SPC/良率闭环：
    - **产线监控**：WIP/Throughput/OEE 实时可视化与告警联动
    - **工艺控制**：工艺窗、配方版本与批准、参数变更留痕
    - **质量分析**：SPC 控制图、西格玛规则、良率拆分与回溯
    - **AI 助手**：3D 翘曲/空洞风险预测，参数推荐与效果评估
    """)
    
    # 功能卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **产线看板**
        - 实时KPI监控
        - WIP/产出/良率趋势
        - 设备稼动与停机原因
        - 告警与异常工单
        """)
    
    with col2:
        st.markdown("""
        **批次与派工**
        - 批次队列/派工/在制品
        - 设备占用与切换损失
        - 参数与工艺窗合规
        - 预测结果与优先级
        """)
    
    with col3:
        st.markdown("""
        **AI 优化**
        - 多目标优化（翘曲/空洞/节拍）
        - 约束与配方合规检查
        - 参数推荐与回放
        - A/B 验证与归档
        """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **配方管理**
        - 电子化配方
        - 版本控制
        - 审批流程
        - 使用统计
        """)
    
    with col2:
        st.markdown("""
        **统计过程控制**
        - SPC控制图
        - 西格玛规则
        - 异常检测
        - 趋势分析
        """)
    
    with col3:
        st.markdown("""
        **量测管理**
        - 量测数据管理
        - 3D可视化
        - 统计分析
        - 质量报告
        """)
    
    # 系统效益
    st.markdown("---")
    st.subheader("系统效益")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("产线 OEE", "89.7%", "+1.8%")
    
    with col2:
        st.metric("月均良率", "96.2%", "+0.6%")
    
    with col3:
        st.metric("WIP 周转", "5.4h", "-0.3h")
    
    with col4:
        st.metric("停机缩短", "-12.4%", "本周")
    
    # 快速开始
    st.markdown("---")
    st.subheader("快速开始")
    
    # 添加自定义CSS样式
    st.markdown("""
    <style>
    .quick-start-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin: 20px 0;
    }
    
    .quick-start-button {
        flex: 1;
        padding: 20px;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-size: 16px;
        font-weight: 600;
        text-decoration: none;
        display: block;
        position: relative;
        overflow: hidden;
    }
    
    .quick-start-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .quick-start-button:hover::before {
        left: 100%;
    }
    
    .primary-button {
        background: linear-gradient(135deg, #00E5FF 0%, #0099CC 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.3);
    }
    
    .primary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.4);
    }
    
    .secondary-button {
        background: linear-gradient(135deg, #2C3E50 0%, #34495E 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
        border: 1px solid #4A5F7A;
    }
    
    .secondary-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(44, 62, 80, 0.4);
        background: linear-gradient(135deg, #34495E 0%, #2C3E50 100%);
    }
    
    .ai-button {
        background: linear-gradient(135deg, #8E44AD 0%, #9B59B6 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(142, 68, 173, 0.3);
    }
    
    .ai-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(142, 68, 173, 0.4);
    }
    
    .button-icon {
        font-size: 24px;
        margin-bottom: 8px;
        display: block;
    }
    
    .button-text {
        font-size: 16px;
        font-weight: 600;
    }
    
    .button-desc {
        font-size: 12px;
        opacity: 0.8;
        margin-top: 4px;
    }
    
    /* 统一按钮宽度 */
    div[data-testid="column"] button {
        width: 133% !important;
        min-width: 266px !important;
        margin-left: -16.5% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 查看产线看板", type="primary", key="dashboard_btn"):
            st.switch_page("pages/01_Dashboard.py")
    
    with col2:
        if st.button("📋 批次与派工", key="lots_btn"):
            st.switch_page("pages/02_Lots.py")
    
    with col3:
        if st.button("⚡ AI 优化", key="optimize_btn"):
            st.switch_page("pages/03_Optimize.py")
    
    # 添加按钮描述
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 8px; color: #B0BEC5; font-size: 14px; line-height: 1.4; text-align: left; padding-left: 30px;">
        <div>
        实时监控产线状态<br>KPI指标与告警
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 8px; color: #B0BEC5; font-size: 14px; line-height: 1.4; text-align: left; padding-left: 30px;">
        <div>
        批次队列管理<br>派工调度优化
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="display: flex; justify-content: flex-start; align-items: center; margin-top: 8px; color: #B0BEC5; font-size: 14px; line-height: 1.4; text-align: left; padding-left: 30px;">
        <div>
        智能参数优化<br>多目标决策
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 系统状态
    st.markdown("---")
    st.subheader("系统状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.success("系统运行正常")
    
    with col2:
        st.info("数据源连接正常")
    
    with col3:
        st.success("AI模型就绪")
    
    with col4:
        st.info("告警系统正常")
    
    # 最新动态
    st.markdown("---")
    st.subheader("最新动态")
    
    # 生成基于当前时间的新闻条目
    now = datetime.now()
    news_items = [
        f"{now.strftime('%Y-%m-%d')}: 系统v1.0.0正式发布",
        f"{(now - timedelta(days=1)).strftime('%Y-%m-%d')}: 新增SPC统计过程控制功能",
        f"{(now - timedelta(days=2)).strftime('%Y-%m-%d')}: 优化3D可视化性能",
        f"{(now - timedelta(days=3)).strftime('%Y-%m-%d')}: 新增多目标优化算法",
        f"{(now - timedelta(days=4)).strftime('%Y-%m-%d')}: 完善用户权限管理"
    ]
    
    for item in news_items:
        st.write(f"• {item}")

if __name__ == "__main__":
    main()