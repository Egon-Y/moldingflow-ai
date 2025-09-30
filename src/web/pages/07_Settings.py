"""
Settings - 系统设置
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

st.set_page_config(
    page_title="Settings - MoldingFlow AI",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 系统设置")
st.markdown("---")

# 设置标签页
tab1, tab2, tab3, tab4, tab5 = st.tabs(["用户管理", "数据源配置", "告警设置", "系统参数", "关于"])

with tab1:
    st.subheader("👥 用户管理")
    
    # 用户列表
    users_data = pd.DataFrame({
        '用户ID': ['PE-001', 'PE-002', 'ME-001', 'QA-001', 'Manager-001'],
        '姓名': ['张工程师', '李工程师', '王制造', '赵质量', '陈经理'],
        '角色': ['工艺工程师', '工艺工程师', '制造工程师', '质量工程师', '经理'],
        '部门': ['工艺部', '工艺部', '制造部', '质量部', '管理部'],
        '状态': ['活跃', '活跃', '活跃', '活跃', '活跃'],
        '最后登录': [
            (datetime.now() - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(minutes=45)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M')
        ]
    })
    
    st.dataframe(users_data, use_container_width=True)
    
    # 添加用户
    st.subheader("➕ 添加用户")
    col1, col2 = st.columns(2)
    
    with col1:
        new_user_id = st.text_input("用户ID", key="new_user_id")
        new_name = st.text_input("姓名", key="new_name")
        new_role = st.selectbox("角色", ["工艺工程师", "制造工程师", "质量工程师", "经理", "操作员"], key="new_role")
    
    with col2:
        new_department = st.selectbox("部门", ["工艺部", "制造部", "质量部", "管理部", "IT部"], key="new_department")
        new_status = st.selectbox("状态", ["活跃", "禁用"], key="new_status")
        new_password = st.text_input("密码", type="password", key="new_password")
    
    if st.button("✅ 添加用户", type="primary"):
        st.success("用户已添加")
    
    # 权限设置
    st.subheader("🔐 权限设置")
    
    role_permissions = {
        '工艺工程师': ['查看', '编辑', '创建配方', '优化参数'],
        '制造工程师': ['查看', '操作批次', '查看设备'],
        '质量工程师': ['查看', 'SPC分析', '量测管理'],
        '经理': ['全部权限'],
        '操作员': ['查看', '操作批次']
    }
    
    selected_role = st.selectbox("选择角色", list(role_permissions.keys()), key="selected_role")
    st.write(f"**{selected_role}权限**: {', '.join(role_permissions[selected_role])}")

with tab2:
    st.subheader("🔌 数据源配置")
    
    # 数据源列表
    data_sources = pd.DataFrame({
        '数据源': ['MES系统', '量测设备', 'X-Ray设备', '环境监控'],
        '类型': ['REST API', 'OPC UA', 'REST API', 'MQTT'],
        '状态': ['连接', '连接', '断开', '连接'],
        '最后同步': [
            (datetime.now() - timedelta(minutes=2)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(minutes=7)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(minutes=25)).strftime('%Y-%m-%d %H:%M'),
            (datetime.now() - timedelta(minutes=12)).strftime('%Y-%m-%d %H:%M')
        ],
        '数据量': ['1.2GB', '856MB', '0MB', '234MB']
    })
    
    st.dataframe(data_sources, use_container_width=True)
    
    # 添加数据源
    st.subheader("➕ 添加数据源")
    col1, col2 = st.columns(2)
    
    with col1:
        source_name = st.text_input("数据源名称", key="source_name")
        source_type = st.selectbox("类型", ["REST API", "OPC UA", "MQTT", "数据库", "文件"], key="source_type")
        source_url = st.text_input("连接地址", key="source_url")
    
    with col2:
        source_username = st.text_input("用户名", key="source_username")
        source_password = st.text_input("密码", type="password", key="source_password")
        source_interval = st.number_input("同步间隔(秒)", 1, 3600, 60, key="source_interval")
    
    if st.button("✅ 添加数据源", type="primary"):
        st.success("数据源已添加")
    
    # 数据同步
    st.subheader("数据同步")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("同步所有", type="primary"):
            st.success("所有数据源已同步")
    
    with col2:
        if st.button("检查状态"):
            st.info("数据源状态检查完成")
    
    with col3:
        if st.button("清理缓存"):
            st.info("缓存已清理")

with tab3:
    st.subheader("告警设置")
    
    # 告警规则
    st.write("**告警规则**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**翘曲告警**")
        warpage_threshold = st.number_input("翘曲阈值 (mm)", 0.01, 1.0, 0.1, 0.01, key="warpage_threshold")
        warpage_enabled = st.checkbox("启用翘曲告警", value=True, key="warpage_enabled")
        
        st.write("**空洞率告警**")
        void_threshold = st.number_input("空洞率阈值 (%)", 0.1, 10.0, 2.0, 0.1, key="void_threshold")
        void_enabled = st.checkbox("启用空洞率告警", value=True, key="void_enabled")
    
    with col2:
        st.write("**设备告警**")
        equipment_threshold = st.number_input("设备利用率阈值 (%)", 50, 100, 90, 1, key="equipment_threshold")
        equipment_enabled = st.checkbox("启用设备告警", value=True, key="equipment_enabled")
        
        st.write("**SPC告警**")
        spc_rules = st.multiselect("SPC规则", ["规则1", "规则2", "规则3", "规则4", "规则5"], default=["规则1", "规则2"], key="spc_rules")
        spc_enabled = st.checkbox("启用SPC告警", value=True, key="spc_enabled")
    
    # 告警通知
    st.subheader("📧 告警通知")
    
    notification_methods = st.multiselect("通知方式", ["邮件", "短信", "微信", "钉钉"], default=["邮件"], key="notification_methods")
    
    col1, col2 = st.columns(2)
    
    with col1:
        email_list = st.text_area("邮件列表", "engineer1@company.com, engineer2@company.com", key="email_list")
        sms_list = st.text_area("短信列表", "13800138000, 13800138001", key="sms_list")
    
    with col2:
        wechat_list = st.text_area("微信列表", "user1, user2", key="wechat_list")
        dingtalk_webhook = st.text_input("钉钉Webhook", "https://oapi.dingtalk.com/robot/send?access_token=xxx", key="dingtalk_webhook")
    
    if st.button("✅ 保存告警设置", type="primary"):
        st.success("告警设置已保存")

with tab4:
    st.subheader("⚙️ 系统参数")
    
    # 模型参数
    st.write("**模型参数**")
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox("模型类型", ["DNN", "GNN", "混合模型"], index=0, key="model_type")
        prediction_interval = st.number_input("预测间隔(秒)", 1, 3600, 60, key="prediction_interval")
        confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.8, 0.05, key="confidence_threshold")
    
    with col2:
        optimization_algorithm = st.selectbox("优化算法", ["贝叶斯优化", "遗传算法", "粒子群优化", "模拟退火"], index=0, key="optimization_algorithm")
        max_iterations = st.number_input("最大迭代次数", 10, 1000, 100, key="max_iterations")
        convergence_threshold = st.number_input("收敛阈值", 0.001, 0.1, 0.01, 0.001, key="convergence_threshold")
    
    # 数据参数
    st.write("**数据参数**")
    col1, col2 = st.columns(2)
    
    with col1:
        data_retention_days = st.number_input("数据保留天数", 30, 365, 90, key="data_retention_days")
        batch_size = st.number_input("批处理大小", 1, 1000, 32, key="batch_size")
        cache_size = st.number_input("缓存大小(MB)", 100, 10000, 1000, key="cache_size")
    
    with col2:
        data_compression = st.checkbox("启用数据压缩", value=True, key="data_compression")
        auto_backup = st.checkbox("自动备份", value=True, key="auto_backup")
        backup_interval = st.number_input("备份间隔(小时)", 1, 168, 24, key="backup_interval")
    
    # 界面参数
    st.write("**界面参数**")
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("主题", ["浅色", "深色", "自动"], key="theme")
        language = st.selectbox("语言", ["中文", "English", "日本語"], key="language")
        timezone = st.selectbox("时区", ["Asia/Shanghai", "UTC", "America/New_York"], key="timezone")
    
    with col2:
        refresh_interval = st.number_input("自动刷新间隔(秒)", 5, 300, 30, key="refresh_interval")
        max_chart_points = st.number_input("图表最大点数", 100, 10000, 1000, key="max_chart_points")
        enable_animations = st.checkbox("启用动画", value=True, key="enable_animations")
    
    if st.button("✅ 保存系统参数", type="primary"):
        st.success("系统参数已保存")

with tab5:
    st.subheader("ℹ️ 关于系统")
    
    # 系统信息
    system_info = {
        "系统名称": "MoldingFlow AI",
        "版本": "v1.0.0",
        "构建日期": datetime.now().strftime('%Y-%m-%d'),
        "Python版本": "3.9.0",
        "Streamlit版本": "1.25.0",
        "数据库": "SQLite",
        "部署环境": "Windows 10"
    }
    
    for key, value in system_info.items():
        st.write(f"**{key}**: {value}")
    
    # 系统状态
    st.subheader("系统状态")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("CPU使用率", "45%")
    
    with col2:
        st.metric("内存使用率", "67%")
    
    with col3:
        st.metric("磁盘使用率", "23%")
    
    with col4:
        st.metric("网络延迟", "12ms")
    
    # 日志信息
    st.subheader("📝 系统日志")
    
    # 生成基于当前时间的日志条目
    now = datetime.now()
    log_entries = [
        {"时间": (now - timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S'), "级别": "INFO", "消息": "系统启动成功"},
        {"时间": (now - timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S'), "级别": "WARN", "消息": "数据源连接超时"},
        {"时间": (now - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S'), "级别": "INFO", "消息": "模型预测完成"},
        {"时间": (now - timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S'), "级别": "ERROR", "消息": "数据库连接失败"},
        {"时间": (now - timedelta(minutes=25)).strftime('%Y-%m-%d %H:%M:%S'), "级别": "INFO", "消息": "用户登录成功"}
    ]
    
    log_df = pd.DataFrame(log_entries)
    st.dataframe(log_df, use_container_width=True)
    
    # 操作按钮
    st.subheader("⚡ 系统操作")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("重启系统", type="primary"):
            st.success("系统重启中...")
    
    with col2:
        if st.button("系统诊断"):
            st.info("系统诊断完成")
    
    with col3:
        if st.button("清理日志"):
            st.info("日志已清理")
    
    with col4:
        if st.button("导出配置"):
            st.info("配置已导出")
