"""
侧边栏组件 - 科技化导航
"""
import streamlit as st
from streamlit_option_menu import option_menu
from typing import Dict, List
import json


class TechSidebar:
    """科技化侧边栏"""
    
    def __init__(self):
        self.menu_items = [
            {
                "label": "🏠 总览",
                "key": "dashboard",
                "icon": "house"
            },
            {
                "label": "📦 批次管理",
                "key": "lots",
                "icon": "box"
            },
            {
                "label": "🎯 智能优化",
                "key": "optimize",
                "icon": "target"
            },
            {
                "label": "📋 配方中心",
                "key": "recipes",
                "icon": "file-text"
            },
            {
                "label": "📊 SPC控制",
                "key": "spc",
                "icon": "graph-up"
            },
            {
                "label": "📏 量测管理",
                "key": "metrology",
                "icon": "ruler"
            },
            {
                "label": "⚙️ 系统设置",
                "key": "settings",
                "icon": "gear"
            }
        ]
    
    def render(self) -> str:
        """渲染侧边栏"""
        with st.sidebar:
            # 系统标题
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="color: #00d4ff; margin: 0;">🔬 MoldingFlow AI</h2>
                <p style="color: #666; margin: 0.5rem 0;">模压翘曲预测与智能推荐系统</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # 用户信息
            self._render_user_info()
            
            st.markdown("---")
            
            # 导航菜单
            selected = option_menu(
                menu_title=None,
                options=[item["label"] for item in self.menu_items],
                icons=[item["icon"] for item in self.menu_items],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#00d4ff", "font-size": "16px"},
                    "nav-link": {
                        "font-size": "14px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee"
                    },
                    "nav-link-selected": {"background-color": "#00d4ff"},
                }
            )
            
            st.markdown("---")
            
            # 系统状态
            self._render_system_status()
            
            st.markdown("---")
            
            # 快速操作
            self._render_quick_actions()
            
            # 返回选中的页面
            for item in self.menu_items:
                if item["label"] == selected:
                    return item["key"]
            
            return "dashboard"
    
    def _render_user_info(self):
        """渲染用户信息"""
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1rem; border-radius: 10px; margin: 1rem 0;">
            <div style="color: white; text-align: center;">
                <h4 style="margin: 0;">👤 张工程师</h4>
                <p style="margin: 0.5rem 0; font-size: 12px;">工艺工程师 | 工艺部</p>
                <div style="display: flex; justify-content: space-between; font-size: 11px;">
                    <span>🟢 在线</span>
                    <span>14:30</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_system_status(self):
        """渲染系统状态"""
        st.markdown("### 📊 系统状态")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("AI模型", "🟢 就绪", "v1.0")
            st.metric("数据源", "🟢 连接", "4/4")
        
        with col2:
            st.metric("设备", "🟡 运行", "3/4")
            st.metric("告警", "🔴 2", "活跃")
        
        # 实时数据流
        st.markdown("### 📡 实时数据")
        
        # 模拟实时数据
        import time
        current_time = time.strftime("%H:%M:%S")
        
        st.markdown(f"""
        <div style="background: #1e1e1e; color: #00ff00; padding: 0.5rem; 
                    border-radius: 5px; font-family: monospace; font-size: 12px;">
            <div>⏰ {current_time}</div>
            <div>📊 批次: 12 活跃</div>
            <div>🔧 设备: 87% 利用率</div>
            <div>📈 良率: 95.2%</div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_quick_actions(self):
        """渲染快速操作"""
        st.markdown("### ⚡ 快速操作")
        
        if st.button("🔮 智能预测", use_container_width=True):
            st.session_state.quick_action = "predict"
        
        if st.button("🎯 参数优化", use_container_width=True):
            st.session_state.quick_action = "optimize"
        
        if st.button("📊 生成报告", use_container_width=True):
            st.session_state.quick_action = "report"
        
        if st.button("🔔 查看告警", use_container_width=True):
            st.session_state.quick_action = "alerts"


class TechHeader:
    """科技化头部"""
    
    def __init__(self):
        pass
    
    def render(self, title: str, subtitle: str = ""):
        """渲染头部"""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <div style="color: white; text-align: center;">
                <h1 style="margin: 0; font-size: 2.5rem;">{title}</h1>
                {f'<p style="margin: 0.5rem 0; font-size: 1.2rem; opacity: 0.9;">{subtitle}</p>' if subtitle else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)


class TechCard:
    """科技化卡片"""
    
    def __init__(self):
        pass
    
    def render(self, title: str, content: str, icon: str = "📊", color: str = "#00d4ff"):
        """渲染卡片"""
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                    border: 1px solid {color}; border-radius: 10px; padding: 1.5rem; 
                    margin: 1rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <span style="font-size: 2rem; margin-right: 1rem;">{icon}</span>
                <h3 style="margin: 0; color: {color};">{title}</h3>
            </div>
            <div style="color: #333;">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)


class TechMetric:
    """科技化指标"""
    
    def __init__(self):
        pass
    
    def render(self, label: str, value: str, delta: str = "", color: str = "#00d4ff"):
        """渲染指标"""
        delta_html = f'<span style="color: #28a745; font-size: 0.8rem;">{delta}</span>' if delta else ''
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                    border: 1px solid {color}; border-radius: 8px; padding: 1rem; 
                    text-align: center; margin: 0.5rem 0;">
            <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">{label}</div>
            <div style="color: {color}; font-size: 1.5rem; font-weight: bold;">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)


class TechAlert:
    """科技化告警"""
    
    def __init__(self):
        pass
    
    def render(self, message: str, type: str = "info"):
        """渲染告警"""
        colors = {
            "info": "#00d4ff",
            "success": "#28a745",
            "warning": "#ffc107",
            "error": "#dc3545"
        }
        
        icons = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌"
        }
        
        color = colors.get(type, colors["info"])
        icon = icons.get(type, icons["info"])
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}20 0%, {color}10 100%); 
                    border-left: 4px solid {color}; border-radius: 5px; padding: 1rem; 
                    margin: 1rem 0; display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 1rem;">{icon}</span>
            <span style="color: #333;">{message}</span>
        </div>
        """, unsafe_allow_html=True)
