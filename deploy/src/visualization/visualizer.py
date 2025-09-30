"""
3D可视化模块 - 翘曲形貌和空洞风险可视化
"""
import numpy as np
import pyvista as pv
import trimesh
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
from typing import Dict, List, Tuple, Optional, Union
import torch
from pathlib import Path


class WarpageVisualizer:
    """翘曲可视化器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.setup_pyvista()
        
    def setup_pyvista(self):
        """设置PyVista"""
        # 设置PyVista主题
        pv.set_plot_theme("document")
        
        # 设置默认参数
        self.default_camera_position = [1, 1, 1]
        self.default_window_size = [800, 600]
        
    def visualize_warpage_3d(self, 
                            mesh: trimesh.Trimesh, 
                            warpage_vectors: np.ndarray,
                            void_risks: np.ndarray = None,
                            title: str = "翘曲形貌预测") -> pv.Plotter:
        """3D翘曲形貌可视化"""
        
        # 创建PyVista网格
        pv_mesh = pv.wrap(mesh.vertices)
        pv_mesh.faces = mesh.faces
        
        # 计算翘曲幅度
        warpage_magnitude = np.linalg.norm(warpage_vectors, axis=1)
        
        # 创建变形后的网格
        deformed_vertices = mesh.vertices + warpage_vectors
        deformed_mesh = pv.wrap(deformed_vertices)
        deformed_mesh.faces = mesh.faces
        
        # 创建绘图器
        plotter = pv.Plotter(window_size=self.default_window_size)
        
        # 添加原始网格（半透明）
        plotter.add_mesh(
            pv_mesh,
            color='lightblue',
            opacity=0.3,
            show_edges=True,
            edge_color='gray',
            label='原始形状'
        )
        
        # 添加变形后的网格
        plotter.add_mesh(
            deformed_mesh,
            scalars=warpage_magnitude,
            cmap='RdYlBu_r',
            show_edges=True,
            edge_color='black',
            label='翘曲形貌'
        )
        
        # 添加翘曲向量箭头
        if len(warpage_vectors) > 0:
            # 采样显示箭头（避免过于密集）
            sample_indices = np.linspace(0, len(warpage_vectors)-1, 
                                       min(100, len(warpage_vectors)), dtype=int)
            
            arrow_centers = mesh.vertices[sample_indices]
            arrow_vectors = warpage_vectors[sample_indices]
            
            plotter.add_arrows(
                arrow_centers,
                arrow_vectors,
                color='red',
                scale=10,  # 放大显示
                label='翘曲向量'
            )
        
        # 设置标题和标签
        plotter.add_title(title)
        plotter.add_legend()
        
        # 设置相机位置
        plotter.camera_position = self.default_camera_position
        
        return plotter
    
    def visualize_void_risk(self, 
                          mesh: trimesh.Trimesh, 
                          void_risks: np.ndarray,
                          title: str = "空洞风险分布") -> pv.Plotter:
        """空洞风险可视化"""
        
        # 创建PyVista网格
        pv_mesh = pv.wrap(mesh.vertices)
        pv_mesh.faces = mesh.faces
        
        # 创建绘图器
        plotter = pv.Plotter(window_size=self.default_window_size)
        
        # 添加风险热力图
        plotter.add_mesh(
            pv_mesh,
            scalars=void_risks,
            cmap='Reds',
            show_edges=True,
            edge_color='black',
            label='空洞风险'
        )
        
        # 添加风险阈值等高线
        if np.max(void_risks) > 0.5:  # 如果存在高风险区域
            plotter.add_mesh(
                pv_mesh.contour([0.5, 0.8]),
                color='red',
                line_width=3,
                label='高风险区域'
            )
        
        # 设置标题
        plotter.add_title(title)
        plotter.add_legend()
        
        # 设置相机位置
        plotter.camera_position = self.default_camera_position
        
        return plotter
    
    def create_comparison_plot(self, 
                             original_mesh: trimesh.Trimesh,
                             warpage_vectors: np.ndarray,
                             void_risks: np.ndarray,
                             title: str = "翘曲与空洞风险对比") -> go.Figure:
        """创建对比图"""
        
        # 创建子图
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['翘曲形貌', '空洞风险'],
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
        )
        
        # 计算翘曲幅度
        warpage_magnitude = np.linalg.norm(warpage_vectors, axis=1)
        
        # 翘曲形貌图
        fig.add_trace(
            go.Scatter3d(
                x=original_mesh.vertices[:, 0],
                y=original_mesh.vertices[:, 1],
                z=original_mesh.vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=warpage_magnitude,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="翘曲幅度")
                ),
                name='翘曲形貌'
            ),
            row=1, col=1
        )
        
        # 空洞风险图
        fig.add_trace(
            go.Scatter3d(
                x=original_mesh.vertices[:, 0],
                y=original_mesh.vertices[:, 1],
                z=original_mesh.vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=void_risks,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="空洞风险")
                ),
                name='空洞风险'
            ),
            row=1, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title=title,
            showlegend=False,
            height=600
        )
        
        return fig
    
    def create_parameter_sensitivity_plot(self, 
                                        parameter_values: Dict[str, List[float]],
                                        warpage_results: List[float],
                                        void_results: List[float]) -> go.Figure:
        """创建参数敏感性分析图"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['温度敏感性', '压力敏感性', '时间敏感性', '综合敏感性'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # 温度敏感性
        if 'temperature' in parameter_values:
            fig.add_trace(
                go.Scatter(
                    x=parameter_values['temperature'],
                    y=warpage_results,
                    mode='lines+markers',
                    name='翘曲',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # 压力敏感性
        if 'pressure' in parameter_values:
            fig.add_trace(
                go.Scatter(
                    x=parameter_values['pressure'],
                    y=warpage_results,
                    mode='lines+markers',
                    name='翘曲',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
        
        # 时间敏感性
        if 'time' in parameter_values:
            fig.add_trace(
                go.Scatter(
                    x=parameter_values['time'],
                    y=void_results,
                    mode='lines+markers',
                    name='空洞风险',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # 综合敏感性
        combined_score = np.array(warpage_results) + np.array(void_results)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(combined_score))),
                y=combined_score,
                mode='lines+markers',
                name='综合评分',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title="参数敏感性分析",
            showlegend=True,
            height=800
        )
        
        return fig


class StreamlitVisualizer:
    """Streamlit可视化组件"""
    
    def __init__(self):
        self.setup_streamlit_config()
    
    def setup_streamlit_config(self):
        """设置Streamlit配置"""
        st.set_page_config(
            page_title="MoldingFlow AI",
            page_icon="🔬",
            layout="wide"
        )
    
    def render_3d_visualization(self, 
                              mesh: trimesh.Trimesh,
                              warpage_vectors: np.ndarray,
                              void_risks: np.ndarray = None):
        """渲染3D可视化"""
        
        # 创建可视化器
        visualizer = WarpageVisualizer()
        
        # 创建对比图
        fig = visualizer.create_comparison_plot(
            mesh, warpage_vectors, void_risks or np.zeros(len(mesh.vertices))
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
    
    def render_parameter_controls(self, 
                                default_params: Dict[str, float]) -> Dict[str, float]:
        """渲染参数控制面板"""
        
        st.sidebar.header("制程参数调整")
        
        # 温度控制
        temperature = st.sidebar.slider(
            "温度 (°C)",
            min_value=150.0,
            max_value=200.0,
            value=default_params.get('temperature', 175.0),
            step=1.0
        )
        
        # 压力控制
        pressure = st.sidebar.slider(
            "压力 (MPa)",
            min_value=50.0,
            max_value=150.0,
            value=default_params.get('pressure', 100.0),
            step=1.0
        )
        
        # 时间控制
        time = st.sidebar.slider(
            "时间 (s)",
            min_value=30.0,
            max_value=120.0,
            value=default_params.get('time', 75.0),
            step=1.0
        )
        
        # 粘度控制
        viscosity = st.sidebar.slider(
            "粘度 (Pa·s)",
            min_value=1000.0,
            max_value=5000.0,
            value=default_params.get('viscosity', 3000.0),
            step=100.0
        )
        
        # CTE控制
        cte = st.sidebar.slider(
            "CTE (ppm/°C)",
            min_value=10.0,
            max_value=50.0,
            value=default_params.get('cte', 30.0),
            step=1.0
        )
        
        return {
            'temperature': temperature,
            'pressure': pressure,
            'time': time,
            'viscosity': viscosity,
            'cte': cte
        }
    
    def render_results_summary(self, 
                             warpage_magnitude: float,
                             void_risk: float,
                             parameters: Dict[str, float]):
        """渲染结果摘要"""
        
        # 创建指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="翘曲幅度",
                value=f"{warpage_magnitude:.3f} mm",
                delta=None
            )
        
        with col2:
            st.metric(
                label="空洞风险",
                value=f"{void_risk:.1%}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="温度",
                value=f"{parameters['temperature']:.1f} °C",
                delta=None
            )
        
        with col4:
            st.metric(
                label="压力",
                value=f"{parameters['pressure']:.1f} MPa",
                delta=None
            )
        
        # 风险评估
        st.subheader("风险评估")
        
        if warpage_magnitude > 0.1:
            st.error("⚠️ 翘曲风险较高，建议调整参数")
        elif warpage_magnitude > 0.05:
            st.warning("⚠️ 翘曲风险中等，建议监控")
        else:
            st.success("✅ 翘曲风险较低")
        
        if void_risk > 0.8:
            st.error("⚠️ 空洞风险较高，建议调整参数")
        elif void_risk > 0.5:
            st.warning("⚠️ 空洞风险中等，建议监控")
        else:
            st.success("✅ 空洞风险较低")


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.template_path = Path("templates")
        self.template_path.mkdir(exist_ok=True)
    
    def generate_analysis_report(self, 
                               mesh: trimesh.Trimesh,
                               warpage_vectors: np.ndarray,
                               void_risks: np.ndarray,
                               parameters: Dict[str, float],
                               recommendations: Dict[str, float]) -> str:
        """生成分析报告"""
        
        # 计算统计信息
        warpage_magnitude = np.linalg.norm(warpage_vectors, axis=1)
        max_warpage = np.max(warpage_magnitude)
        avg_warpage = np.mean(warpage_magnitude)
        max_void_risk = np.max(void_risks)
        avg_void_risk = np.mean(void_risks)
        
        # 生成报告内容
        report = f"""
# MoldingFlow AI 分析报告

## 执行摘要
- 最大翘曲幅度: {max_warpage:.3f} mm
- 平均翘曲幅度: {avg_warpage:.3f} mm
- 最大空洞风险: {max_void_risk:.1%}
- 平均空洞风险: {avg_void_risk:.1%}

## 当前参数
- 温度: {parameters['temperature']:.1f} °C
- 压力: {parameters['pressure']:.1f} MPa
- 时间: {parameters['time']:.1f} s
- 粘度: {parameters['viscosity']:.1f} Pa·s
- CTE: {parameters['cte']:.1f} ppm/°C

## 优化建议
- 推荐温度: {recommendations['temperature']:.1f} °C
- 推荐压力: {recommendations['pressure']:.1f} MPa
- 推荐时间: {recommendations['time']:.1f} s
- 推荐粘度: {recommendations['viscosity']:.1f} Pa·s
- 推荐CTE: {recommendations['cte']:.1f} ppm/°C

## 风险评估
"""
        
        # 添加风险评估
        if max_warpage > 0.1:
            report += "- ⚠️ 翘曲风险较高，建议调整参数\n"
        elif max_warpage > 0.05:
            report += "- ⚠️ 翘曲风险中等，建议监控\n"
        else:
            report += "- ✅ 翘曲风险较低\n"
        
        if max_void_risk > 0.8:
            report += "- ⚠️ 空洞风险较高，建议调整参数\n"
        elif max_void_risk > 0.5:
            report += "- ⚠️ 空洞风险中等，建议监控\n"
        else:
            report += "- ✅ 空洞风险较低\n"
        
        return report
