import streamlit as st
from typing import Optional

def kpi_card(label: str, value: str, delta: Optional[str] = None, help_text: Optional[str] = None, icon: Optional[str] = None) -> None:
    """标准化 KPI 指标卡，统一风格与布局。"""
    container = st.container()
    with container:
        cols = st.columns([1, 1])
        with cols[0]:
            if icon:
                st.markdown(f"**{icon} {label}**")
            else:
                st.markdown(f"**{label}**")
            if help_text:
                st.caption(help_text)
        with cols[1]:
            st.metric(label="", value=value, delta=delta)


def tag(text: str, color: str = "#00E5FF") -> None:
    """小标签组件。"""
    st.markdown(
        f"<span style='display:inline-block;padding:2px 8px;border-radius:8px;background:{color}20;color:{color};font-size:12px;'>"
        f"{text}</span>",
        unsafe_allow_html=True,
    )
