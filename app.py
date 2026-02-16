"""
🌍 EarthDial v3 — Wildfire Prevention & Counterfactual Decision System
Built with NVIDIA Nemotron + GPU-Accelerated Graph Optimization

Cinematic demo with auto-loop and "Take Control" interactive mode.

#NVIDIAGTC 2026 | earthdial.ai
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarthDial v3 | NVIDIA Nemotron",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── NVIDIA-Grade CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ═══ GLOBAL ═══ */
    :root {
        --nvidia-green: #76B900;
        --nvidia-green-glow: rgba(118,185,0,0.4);
        --nvidia-green-dim: rgba(118,185,0,0.15);
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: rgba(18,18,26,0.85);
        --bg-glass: rgba(255,255,255,0.03);
        --border-subtle: rgba(255,255,255,0.06);
        --border-glow: rgba(118,185,0,0.3);
        --text-primary: #f0f0f5;
        --text-secondary: #8a8a9a;
        --text-dim: #555566;
        --danger: #ff3b3b;
        --warning: #ffaa00;
        --info: #00b4d8;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    div[data-testid="stStatusWidget"] {display: none;}

    .main .block-container {
        padding: 0.5rem 1.5rem 2rem 1.5rem;
        max-width: 100%;
    }

    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, sans-serif;
    }

    /* ═══ TOPBAR ═══ */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 24px;
        background: linear-gradient(180deg, rgba(10,10,15,0.98) 0%, rgba(10,10,15,0.85) 100%);
        border-bottom: 1px solid var(--border-subtle);
        backdrop-filter: blur(20px);
        position: relative;
        z-index: 100;
        margin: -0.5rem -1.5rem 0 -1.5rem;
    }
    .topbar::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--nvidia-green), transparent);
        opacity: 0.5;
    }
    .topbar-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .topbar-logo {
        width: 36px;
        height: 36px;
        background: var(--nvidia-green);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 0 20px var(--nvidia-green-glow);
    }
    .topbar-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.3px;
    }
    .topbar-subtitle {
        font-size: 0.7rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .topbar-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .topbar-badge {
        background: rgba(118,185,0,0.12);
        border: 1px solid rgba(118,185,0,0.3);
        color: var(--nvidia-green);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .topbar-status {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        color: var(--text-secondary);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: var(--nvidia-green);
        box-shadow: 0 0 8px var(--nvidia-green-glow);
        animation: statusPulse 2s ease-in-out infinite;
    }
    .status-dot.danger {
        background: var(--danger);
        box-shadow: 0 0 8px rgba(255,59,59,0.4);
    }
    @keyframes statusPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(0.85); }
    }

    /* ═══ THREAT BANNER ═══ */
    .threat-banner {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        padding: 8px 20px;
        background: linear-gradient(90deg, rgba(255,59,59,0.08), rgba(255,59,59,0.15), rgba(255,59,59,0.08));
        border: 1px solid rgba(255,59,59,0.2);
        border-radius: 8px;
        margin: 12px 0;
        animation: threatPulse 3s ease-in-out infinite;
    }
    @keyframes threatPulse {
        0%, 100% { border-color: rgba(255,59,59,0.2); }
        50% { border-color: rgba(255,59,59,0.5); }
    }
    .threat-banner-icon {
        font-size: 0.85rem;
    }
    .threat-banner-text {
        font-size: 0.72rem;
        font-weight: 600;
        color: var(--danger);
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    .threat-banner-time {
        font-size: 0.68rem;
        color: var(--text-secondary);
        font-family: 'JetBrains Mono', monospace;
    }

    /* ═══ GLASS CARDS ═══ */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: var(--border-glow);
        box-shadow: 0 0 30px rgba(118,185,0,0.05);
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    }

    /* ═══ STAT CARDS ═══ */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }
    .stat-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 3px;
        height: 100%;
    }
    .stat-card.green::before { background: var(--nvidia-green); }
    .stat-card.red::before { background: var(--danger); }
    .stat-card.orange::before { background: var(--warning); }
    .stat-card.blue::before { background: var(--info); }
    .stat-label {
        font-size: 0.65rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.1;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-value.green { color: var(--nvidia-green); }
    .stat-value.red { color: var(--danger); }
    .stat-value.orange { color: var(--warning); }
    .stat-value.blue { color: var(--info); }
    .stat-delta {
        font-size: 0.7rem;
        margin-top: 4px;
        font-weight: 500;
    }
    .stat-delta.up { color: var(--danger); }
    .stat-delta.down { color: var(--nvidia-green); }

    /* ═══ SECTION HEADERS ═══ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 20px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border-subtle);
    }
    .section-icon {
        width: 28px;
        height: 28px;
        background: var(--nvidia-green-dim);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
    }
    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: 0.3px;
    }
    .section-subtitle {
        font-size: 0.68rem;
        color: var(--text-secondary);
    }

    /* ═══ TAKE CONTROL BUTTON ═══ */
    .take-control-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
    }

    /* ═══ DEMO PHASE INDICATOR ═══ */
    .phase-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 10px 20px;
        background: rgba(118,185,0,0.08);
        border: 1px solid rgba(118,185,0,0.2);
        border-radius: 8px;
        margin: 12px 0;
    }
    .phase-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--text-dim);
        transition: all 0.3s ease;
    }
    .phase-dot.active {
        background: var(--nvidia-green);
        box-shadow: 0 0 10px var(--nvidia-green-glow);
        transform: scale(1.3);
    }
    .phase-dot.completed {
        background: var(--nvidia-green);
        opacity: 0.5;
    }
    .phase-label {
        font-size: 0.72rem;
        color: var(--nvidia-green);
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 8px;
    }

    /* ═══ DATA TABLE ═══ */
    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.75rem;
    }
    .data-table th {
        background: rgba(118,185,0,0.08);
        color: var(--text-secondary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        padding: 10px 14px;
        text-align: left;
        font-size: 0.65rem;
        border-bottom: 1px solid var(--border-subtle);
    }
    .data-table td {
        padding: 10px 14px;
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-primary);
    }
    .data-table tr:hover td {
        background: rgba(118,185,0,0.03);
    }

    /* ═══ RISK BADGE ═══ */
    .risk-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .risk-badge.extreme {
        background: rgba(255,59,59,0.15);
        color: var(--danger);
        border: 1px solid rgba(255,59,59,0.3);
    }
    .risk-badge.high {
        background: rgba(255,170,0,0.15);
        color: var(--warning);
        border: 1px solid rgba(255,170,0,0.3);
    }
    .risk-badge.moderate {
        background: rgba(0,180,216,0.15);
        color: var(--info);
        border: 1px solid rgba(0,180,216,0.3);
    }
    .risk-badge.low {
        background: rgba(118,185,0,0.15);
        color: var(--nvidia-green);
        border: 1px solid rgba(118,185,0,0.3);
    }

    /* ═══ PREVENTION BRIEF BOX ═══ */
    .brief-box {
        background: rgba(18,18,26,0.9);
        border: 1px solid rgba(118,185,0,0.2);
        border-radius: 12px;
        padding: 24px 28px;
        font-family: 'Inter', sans-serif;
        font-size: 0.82rem;
        line-height: 1.7;
        color: var(--text-primary);
        max-height: 600px;
        overflow-y: auto;
    }
    .brief-box h1, .brief-box h2, .brief-box h3 {
        color: var(--nvidia-green);
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 6px;
    }
    .brief-box strong {
        color: var(--nvidia-green);
    }

    /* ═══ FOOTER ═══ */
    .app-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 16px 0;
        border-top: 1px solid var(--border-subtle);
        margin-top: 30px;
        font-size: 0.68rem;
        color: var(--text-dim);
    }
    .footer-left {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .footer-link {
        color: var(--text-secondary);
        text-decoration: none;
        transition: color 0.2s;
    }
    .footer-link:hover {
        color: var(--nvidia-green);
    }
    .footer-nvidia {
        display: flex;
        align-items: center;
        gap: 6px;
        color: var(--nvidia-green);
        font-weight: 600;
    }

    /* ═══ OVERRIDE STREAMLIT DEFAULTS ═══ */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-secondary);
        border-radius: 10px;
        padding: 4px;
        gap: 4px;
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: var(--text-secondary);
        font-weight: 500;
        font-size: 0.78rem;
        padding: 8px 16px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: var(--nvidia-green-dim) !important;
        color: var(--nvidia-green) !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
    }

    /* Metric styling */
    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--nvidia-green), #5a9400) !important;
        color: #000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 10px var(--nvidia-green-glow) !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px var(--nvidia-green-glow) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border-glow) !important;
        color: var(--nvidia-green) !important;
        box-shadow: none !important;
    }

    /* Checkbox styling */
    .stCheckbox label {
        font-size: 0.78rem !important;
        color: var(--text-secondary) !important;
    }

    /* Slider styling */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: var(--nvidia-green) !important;
    }

    /* Selectbox */
    .stSelectbox [data-baseweb="select"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: var(--nvidia-green) !important;
    }

    /* ═══ SIDEBAR ═══ */
    section[data-testid="stSidebar"] {
        background: var(--bg-primary);
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        font-size: 0.82rem;
    }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--text-dim); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

    /* ═══ MAP CONTAINER ═══ */
    .map-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
        position: relative;
    }
    .map-overlay-label {
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(10,10,15,0.85);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.68rem;
        color: var(--text-secondary);
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        z-index: 10;
        backdrop-filter: blur(10px);
    }

    /* ═══ PLAN CARD ═══ */
    .plan-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        transition: all 0.2s ease;
    }
    .plan-card:hover {
        border-color: var(--border-glow);
    }
    .plan-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .plan-rank {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--nvidia-green);
        font-weight: 700;
    }
    .plan-efficiency {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        font-weight: 700;
    }
    .plan-detail {
        font-size: 0.72rem;
        color: var(--text-secondary);
        margin: 2px 0;
    }

    /* ═══ ANIMATIONS ═══ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeIn 0.4s ease-out forwards;
    }

    @keyframes glowPulse {
        0%, 100% { box-shadow: 0 0 15px var(--nvidia-green-glow); }
        50% { box-shadow: 0 0 30px var(--nvidia-green-glow), 0 0 60px rgba(118,185,0,0.1); }
    }
    .glow-pulse {
        animation: glowPulse 3s ease-in-out infinite;
    }

    @keyframes scanline {
        0% { transform: translateY(-100%); }
        100% { transform: translateY(100vh); }
    }

    /* ═══ VOICEOVER EXPERIENCE ═══ */
    .vo-start-overlay {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        text-align: center;
    }
    .vo-start-btn {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--nvidia-green), #5a9400);
        border: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 40px var(--nvidia-green-glow), 0 0 80px rgba(118,185,0,0.15);
        transition: all 0.3s ease;
        animation: glowPulse 2s ease-in-out infinite;
    }
    .vo-start-btn:hover {
        transform: scale(1.08);
        box-shadow: 0 0 60px var(--nvidia-green-glow), 0 0 100px rgba(118,185,0,0.25);
    }
    .vo-start-btn svg {
        width: 40px;
        height: 40px;
        fill: #000;
        margin-left: 6px;
    }
    .vo-start-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-top: 24px;
        letter-spacing: -0.5px;
    }
    .vo-start-sub {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 8px;
        max-width: 450px;
    }
    .vo-start-duration {
        font-size: 0.7rem;
        color: var(--nvidia-green);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        margin-top: 16px;
        letter-spacing: 1px;
    }

    .vo-progress-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 20px;
        background: rgba(118,185,0,0.06);
        border: 1px solid rgba(118,185,0,0.15);
        border-radius: 8px;
        margin: 8px 0;
    }
    .vo-progress-bar-bg {
        flex: 1;
        height: 4px;
        background: var(--border-subtle);
        border-radius: 2px;
        overflow: hidden;
    }
    .vo-progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--nvidia-green), #a0e000);
        border-radius: 2px;
        transition: width 0.5s linear;
    }
    .vo-time {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: var(--nvidia-green);
        font-weight: 600;
        min-width: 45px;
    }
    .vo-label {
        font-size: 0.65rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hide the auto-advance button */
    .auto-advance-hidden {
        position: absolute;
        left: -9999px;
        top: -9999px;
        opacity: 0;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State ──────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "terrain_df": None,
        "risk_df": None,
        "powerlines_df": None,
        "substations_df": None,
        "facilities_df": None,
        "wind_df": None,
        "disabled_lines": set(),
        "nemotron_connected": False,
        "nemotron_engine": None,
        "prevention_brief": None,
        "counterfactual_explanation": None,
        "shutoff_plans": None,
        "data_loaded": False,
        "selected_plan": None,
        "show_fire_spread": True,
        "show_wind": True,
        "show_risk_columns": True,
        "demo_mode": True,
        "demo_phase": 0,
        "interactive_mode": False,
        "demo_playing": False,
        "demo_audio_start": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Load Data ──────────────────────────────────────────────────────────────
@st.cache_data
def load_all_data():
    from data_generator import (
        generate_terrain_grid, generate_weather_timeline,
        get_substation_df, get_power_lines_df,
        get_critical_facilities_df, compute_powerline_proximity,
        generate_wind_field,
    )
    from risk_engine import compute_ignition_risk
    from config import WEATHER

    terrain = generate_terrain_grid()
    powerlines = get_power_lines_df()
    substations = get_substation_df()
    facilities = get_critical_facilities_df()
    proximity = compute_powerline_proximity(terrain, powerlines)
    risk_terrain = compute_ignition_risk(terrain, proximity, WEATHER)
    wind = generate_wind_field(risk_terrain)
    weather_timeline = generate_weather_timeline()

    return risk_terrain, powerlines, substations, facilities, wind, weather_timeline, proximity


if not st.session_state.data_loaded:
    risk_terrain, powerlines, substations, facilities, wind, weather_timeline, proximity = load_all_data()
    st.session_state.terrain_df = risk_terrain
    st.session_state.risk_df = risk_terrain.copy()
    st.session_state.powerlines_df = powerlines
    st.session_state.substations_df = substations
    st.session_state.facilities_df = facilities
    st.session_state.wind_df = wind
    st.session_state.weather_timeline = weather_timeline
    st.session_state.proximity = proximity
    st.session_state.data_loaded = True


# ─── Auto-connect Nemotron ──────────────────────────────────────────────────
def auto_connect_nemotron():
    if st.session_state.nemotron_connected:
        return
    try:
        api_key = st.secrets.get("NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
    except Exception:
        api_key = os.getenv("NVIDIA_API_KEY", "")
    if api_key:
        try:
            from nemotron_prevention import NemotronPreventionEngine
            engine = NemotronPreventionEngine(api_key=api_key)
            st.session_state.nemotron_engine = engine
            st.session_state.nemotron_connected = True
        except Exception:
            pass

auto_connect_nemotron()


# ─── Helper: Current timestamp ──────────────────────────────────────────────
from datetime import datetime, timezone
def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


# ═══════════════════════════════════════════════════════════════════════════
# TOPBAR
# ═══════════════════════════════════════════════════════════════════════════
nemotron_status = "ONLINE" if st.session_state.nemotron_connected else "OFFLINE"
nemotron_dot_class = "" if st.session_state.nemotron_connected else "danger"

st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-logo">🌍</div>
        <div>
            <div class="topbar-title">EarthDial v3</div>
            <div class="topbar-subtitle">Wildfire Prevention & Counterfactual Decision System</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="topbar-status">
            <div class="status-dot {nemotron_dot_class}"></div>
            Nemotron {nemotron_status}
        </div>
        <div class="topbar-badge">NVIDIA NEMOTRON</div>
        <div class="topbar-badge">GPU GRAPH OPT</div>
        <div class="topbar-status">
            <div class="status-dot danger"></div>
            RED FLAG ACTIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# THREAT BANNER
# ═══════════════════════════════════════════════════════════════════════════
from config import WEATHER
st.markdown(f"""
<div class="threat-banner">
    <span class="threat-banner-icon">⚠️</span>
    <span class="threat-banner-text">RED FLAG WARNING — Diablo Wind Event — Sonoma County, CA</span>
    <span class="threat-banner-time">Wind: {WEATHER['wind_speed_mph']}mph · Gusts: {WEATHER['wind_gust_mph']}mph · Humidity: {WEATHER['humidity_pct']}% · {WEATHER['temperature_f']}°F</span>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# VOICEOVER AUDIO CONTROLLER
# ═══════════════════════════════════════════════════════════════════════════
PHASE_TRANSITIONS = [0, 38, 56, 74, 92]  # start seconds for each phase
AUDIO_DURATION = 118.94

# Calculate elapsed time and current phase when demo is playing
demo_elapsed = 0.0
demo_auto_phase = 0

if st.session_state.demo_playing and st.session_state.demo_audio_start is not None:
    demo_elapsed = time.time() - st.session_state.demo_audio_start
    if demo_elapsed >= AUDIO_DURATION:
        # Demo is finished
        st.session_state.demo_playing = False
        st.session_state.demo_audio_start = None
        st.session_state.demo_phase = 4  # stay on last phase
    else:
        # Determine phase from elapsed time
        for i, t in enumerate(PHASE_TRANSITIONS):
            if demo_elapsed >= t:
                demo_auto_phase = i
        st.session_state.demo_phase = demo_auto_phase


# ═══════════════════════════════════════════════════════════════════════════
# TAKE CONTROL / DEMO MODE TOGGLE
# ═══════════════════════════════════════════════════════════════════════════
tc_col1, tc_col2, tc_col3 = st.columns([1, 6, 1])
with tc_col1:
    if st.session_state.interactive_mode:
        if st.button("◀ DEMO MODE", use_container_width=True):
            st.session_state.interactive_mode = False
            st.session_state.demo_phase = 0
            st.session_state.demo_playing = False
            st.session_state.demo_audio_start = None
            st.rerun()
    else:
        demo_phase_names = ["THREAT MAP", "GRID ANALYSIS", "AI OPTIMIZATION", "COUNTERFACTUAL", "PREVENTION BRIEF"]
        current_phase = st.session_state.demo_phase % len(demo_phase_names)
        st.markdown(f"""
        <div style="text-align:center; padding:4px;">
            <div style="font-size:0.6rem; color:var(--text-dim); text-transform:uppercase; letter-spacing:1px;">Demo Phase</div>
            <div style="font-size:0.85rem; color:var(--nvidia-green); font-weight:700;">{current_phase + 1}/5</div>
        </div>
        """, unsafe_allow_html=True)

with tc_col2:
    if not st.session_state.interactive_mode:
        demo_phase_names = ["THREAT MAP", "GRID ANALYSIS", "AI OPTIMIZATION", "COUNTERFACTUAL", "PREVENTION BRIEF"]
        current_phase = st.session_state.demo_phase % len(demo_phase_names)

        if st.session_state.demo_playing:
            # Show voiceover progress bar
            progress_pct = min(100, (demo_elapsed / AUDIO_DURATION) * 100)
            elapsed_min = int(demo_elapsed) // 60
            elapsed_sec = int(demo_elapsed) % 60
            total_min = int(AUDIO_DURATION) // 60
            total_sec = int(AUDIO_DURATION) % 60

            dots_html = ""
            for i, name in enumerate(demo_phase_names):
                if i < current_phase:
                    dots_html += f'<div class="phase-dot completed" title="{name}"></div>'
                elif i == current_phase:
                    dots_html += f'<div class="phase-dot active" title="{name}"></div>'
                else:
                    dots_html += f'<div class="phase-dot" title="{name}"></div>'

            st.markdown(f"""
            <div class="vo-progress-container">
                <span class="vo-label">🔊 VOICEOVER</span>
                {dots_html}
                <span class="phase-label">{demo_phase_names[current_phase]}</span>
                <div class="vo-progress-bar-bg">
                    <div class="vo-progress-bar-fill" style="width:{progress_pct:.1f}%"></div>
                </div>
                <span class="vo-time">{elapsed_min}:{elapsed_sec:02d} / {total_min}:{total_sec:02d}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Phase dots (no audio playing)
            dots_html = ""
            for i, name in enumerate(demo_phase_names):
                if i < current_phase:
                    dots_html += f'<div class="phase-dot completed" title="{name}"></div>'
                elif i == current_phase:
                    dots_html += f'<div class="phase-dot active" title="{name}"></div>'
                else:
                    dots_html += f'<div class="phase-dot" title="{name}"></div>'

            st.markdown(f"""
            <div class="phase-indicator">
                {dots_html}
                <span class="phase-label">{demo_phase_names[current_phase]}</span>
            </div>
            """, unsafe_allow_html=True)

with tc_col3:
    if not st.session_state.interactive_mode:
        if st.button("🎮 TAKE CONTROL", use_container_width=True):
            st.session_state.interactive_mode = True
            st.session_state.demo_playing = False
            st.session_state.demo_audio_start = None
            st.rerun()
    else:
        st.markdown(f"""
        <div style="text-align:center; padding:4px;">
            <div style="font-size:0.6rem; color:var(--nvidia-green); text-transform:uppercase; letter-spacing:1px; font-weight:600;">Interactive Mode</div>
            <div style="font-size:0.7rem; color:var(--text-secondary);">Full Control</div>
        </div>
        """, unsafe_allow_html=True)

# ── Stop audio JS when entering interactive mode ──
if st.session_state.interactive_mode:
    components.html("""
    <script>
    (function() {
        const audio = window.parent._earthdialAudio;
        if (audio) { audio.pause(); audio.currentTime = 0; }
    })();
    </script>
    """, height=0)


# ═══════════════════════════════════════════════════════════════════════════
# STAT OVERVIEW ROW
# ═══════════════════════════════════════════════════════════════════════════
risk_df = st.session_state.risk_df
extreme = int((risk_df["ignition_risk"] > 0.75).sum())
high = int(((risk_df["ignition_risk"] > 0.55) & (risk_df["ignition_risk"] <= 0.75)).sum())
moderate = int(((risk_df["ignition_risk"] > 0.3) & (risk_df["ignition_risk"] <= 0.55)).sum())
low = int((risk_df["ignition_risk"] <= 0.3).sum())
mean_risk = risk_df["ignition_risk"].mean()

st.markdown(f"""
<div class="stat-grid">
    <div class="stat-card red">
        <div class="stat-label">Extreme Risk Cells</div>
        <div class="stat-value red">{extreme}</div>
        <div class="stat-delta up">▲ Critical threshold exceeded</div>
    </div>
    <div class="stat-card orange">
        <div class="stat-label">High Risk Cells</div>
        <div class="stat-value orange">{high}</div>
        <div class="stat-delta up">▲ Above normal baseline</div>
    </div>
    <div class="stat-card blue">
        <div class="stat-label">Moderate Risk</div>
        <div class="stat-value blue">{moderate}</div>
    </div>
    <div class="stat-card green">
        <div class="stat-label">Mean Risk Index</div>
        <div class="stat-value green">{mean_risk:.3f}</div>
        <div class="stat-delta up">▲ 340% above seasonal avg</div>
    </div>
    <div class="stat-card green">
        <div class="stat-label">Monitored Area</div>
        <div class="stat-value">{len(risk_df):,}</div>
        <div class="stat-delta" style="color:var(--text-secondary);">Grid cells active</div>
    </div>
    <div class="stat-card blue">
        <div class="stat-label">Nemotron Model</div>
        <div class="stat-value" style="font-size:0.9rem;">49B-v1</div>
        <div class="stat-delta" style="color:var(--nvidia-green);">{'● Connected' if st.session_state.nemotron_connected else '○ Offline'}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TABS (INTERACTIVE MODE) or AUTO-DEMO PANELS
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.interactive_mode:
    # ─── FULL INTERACTIVE MODE ──────────────────────────────────────────────
    tab_map, tab_grid, tab_prevention, tab_counterfactual, tab_timeline = st.tabs([
        "🗺️ Threat Map",
        "⚡ Grid Control",
        "📋 Prevention Brief",
        "🔄 Counterfactual",
        "📈 Weather Timeline",
    ])

    # ── TAB 1: 3D THREAT MAP ──
    with tab_map:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🗺️</div>
            <div>
                <div class="section-title">Real-Time 3D Ignition Risk Surface</div>
                <div class="section-subtitle">GPU-rendered risk columns · Power grid arcs · Fire spread projections · Wind vectors</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Visualization controls
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.session_state.show_risk_columns = st.checkbox("3D Risk Columns", value=True, key="ic_risk")
        with vc2:
            st.session_state.show_wind = st.checkbox("Wind Field", value=True, key="ic_wind")
        with vc3:
            st.session_state.show_fire_spread = st.checkbox("Fire Spread", value=True, key="ic_fire")
        with vc4:
            show_heatmap = st.checkbox("Heatmap (2D)", value=False, key="ic_heat")

        # Build map
        max_risk_idx = risk_df["ignition_risk"].idxmax()
        max_risk_point = risk_df.loc[max_risk_idx]

        from risk_engine import compute_multiple_spread_scenarios
        fire_scenarios = []
        if st.session_state.show_fire_spread:
            fire_scenarios = compute_multiple_spread_scenarios(
                ignition_lat=max_risk_point["lat"],
                ignition_lon=max_risk_point["lon"],
                hours_list=[3, 6, 12, 24],
            )

        from visualization import build_full_3d_map
        deck = build_full_3d_map(
            terrain_df=st.session_state.risk_df,
            powerlines_df=st.session_state.powerlines_df,
            substations_df=st.session_state.substations_df,
            facilities_df=st.session_state.facilities_df,
            wind_df=st.session_state.wind_df if st.session_state.show_wind else None,
            fire_scenarios=fire_scenarios if st.session_state.show_fire_spread else None,
            disabled_lines=st.session_state.disabled_lines,
            affected_facility_ids=set(),
            ignition_point=(max_risk_point["lat"], max_risk_point["lon"]),
            show_risk_columns=st.session_state.show_risk_columns,
            show_heatmap=show_heatmap,
            show_wind=st.session_state.show_wind,
            show_fire_spread=st.session_state.show_fire_spread,
        )

        st.pydeck_chart(deck, height=650, use_container_width=True)

        # Legend
        st.markdown("""
        <div class="glass-card" style="margin-top:12px;">
            <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.72rem; color:var(--text-secondary);">
                <span>🔴 <strong style="color:var(--danger)">Red columns</strong> = Extreme ignition risk</span>
                <span>🟠 <strong style="color:var(--warning)">Orange columns</strong> = High risk</span>
                <span>🔵 <strong style="color:var(--info)">Blue arcs</strong> = Active power lines</span>
                <span>⚪ <strong>Dots</strong> = Critical facilities</span>
                <span>🟢 <strong style="color:var(--nvidia-green)">Green dots</strong> = Substations</span>
                <span>➡️ <strong>Lines</strong> = Wind field</span>
                <span>🔶 <strong style="color:#ff9800">Polygons</strong> = Fire spread (3/6/12/24h)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: GRID CONTROL ──
    with tab_grid:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">⚡</div>
            <div>
                <div class="section-title">Intelligent De-Energization Control</div>
                <div class="section-subtitle">GPU-accelerated graph optimization · Critical load preservation · Surgical shutoff planning</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("##### Power Line Control")

            powerlines_df = st.session_state.powerlines_df
            new_disabled = set()

            # Build table
            line_rows = ""
            for _, pl in powerlines_df.iterrows():
                veg = pl["vegetation_risk"]
                if veg > 0.75:
                    badge = '<span class="risk-badge extreme">EXTREME</span>'
                elif veg > 0.5:
                    badge = '<span class="risk-badge high">HIGH</span>'
                else:
                    badge = '<span class="risk-badge moderate">MODERATE</span>'

                is_disabled = st.checkbox(
                    f"{pl['name']} ({pl['voltage_kv']}kV) | Veg: {pl['vegetation_risk']:.0%}",
                    value=pl["id"] in st.session_state.disabled_lines,
                    key=f"iline_{pl['id']}",
                )
                if is_disabled:
                    new_disabled.add(pl["id"])

            if new_disabled != st.session_state.disabled_lines:
                st.session_state.disabled_lines = new_disabled
                from data_generator import compute_powerline_proximity, get_power_lines_df
                from risk_engine import compute_ignition_risk
                updated_pl = get_power_lines_df()
                updated_pl["active"] = ~updated_pl["id"].isin(new_disabled)
                new_proximity = compute_powerline_proximity(
                    st.session_state.terrain_df,
                    updated_pl[updated_pl["active"]],
                )
                new_risk = compute_ignition_risk(st.session_state.terrain_df, new_proximity)
                st.session_state.risk_df = new_risk

        with col_right:
            st.markdown("##### AI-Optimized Shutoff Plans")
            st.markdown('<div style="font-size:0.75rem; color:var(--text-secondary);">GPU graph optimization finds optimal combinations minimizing fire risk while preserving critical loads.</div>', unsafe_allow_html=True)

            from grid_optimizer import GridOptimizer
            optimizer = GridOptimizer()

            protect_critical = st.checkbox("🏥 Protect critical facilities", value=True, key="iprotect")
            max_shutoffs = st.slider("Max lines to disable", 1, 5, 3, key="imax")

            if st.button("🧠 COMPUTE OPTIMAL PLANS", use_container_width=True):
                with st.spinner("Running GPU-accelerated graph optimization..."):
                    plans = optimizer.optimize_shutoffs(
                        weather=WEATHER,
                        max_shutoffs=max_shutoffs,
                        protect_critical=protect_critical,
                    )
                    st.session_state.shutoff_plans = plans

            if st.session_state.shutoff_plans:
                for plan in st.session_state.shutoff_plans[:5]:
                    eff_color = "green" if plan['efficiency_ratio'] > 2 else "orange" if plan['efficiency_ratio'] > 1 else "red"
                    st.markdown(f"""
                    <div class="plan-card">
                        <div class="plan-header">
                            <span class="plan-rank">PLAN #{plan['rank']}</span>
                            <span class="plan-efficiency" style="color:var(--{'nvidia-green' if eff_color == 'green' else 'warning' if eff_color == 'orange' else 'danger'});">
                                EFF: {plan['efficiency_ratio']:.2f}
                            </span>
                        </div>
                        <div class="plan-detail"><strong>Lines:</strong> {', '.join(plan['line_names'])}</div>
                        <div class="plan-detail"><strong>Risk removed:</strong> {plan['total_risk_removed']:.4f} | <strong>Confidence:</strong> {plan['confidence']:.0%}</div>
                        <div class="plan-detail"><strong>Grid connected:</strong> {'✅' if plan['grid_connected'] else '❌'} | <strong>Facilities impacted:</strong> {plan['critical_facilities_impacted']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button(f"Apply Plan #{plan['rank']}", key=f"iapply_{plan['rank']}", use_container_width=True):
                        st.session_state.disabled_lines = set(plan['lines_disabled'])
                        st.session_state.selected_plan = plan
                        st.rerun()

        # Impact assessment
        if st.session_state.disabled_lines:
            st.markdown("---")
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">⚠️</div>
                <div>
                    <div class="section-title">Impact Assessment</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            affected = optimizer.get_affected_facilities(st.session_state.disabled_lines)
            connectivity = optimizer.check_grid_connectivity(st.session_state.disabled_lines)

            ia1, ia2, ia3, ia4 = st.columns(4)
            with ia1:
                st.metric("Grid Status", "✅ Connected" if connectivity["connected"] else "❌ Fragmented")
            with ia2:
                st.metric("Components", connectivity["num_components"])
            with ia3:
                st.metric("Facilities Impacted", len(affected))
            with ia4:
                st.metric("Lines Disabled", len(st.session_state.disabled_lines))

    # ── TAB 3: PREVENTION BRIEF ──
    with tab_prevention:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📋</div>
            <div>
                <div class="section-title">AI-Generated Prevention Brief</div>
                <div class="section-subtitle">NVIDIA Nemotron synthesizes risk data into operator-ready prevention orders</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.nemotron_connected:
            st.warning("⚠️ Nemotron not connected. Add NVIDIA_API_KEY to secrets.")
        else:
            if st.button("🧠 GENERATE PREVENTION BRIEF", use_container_width=True):
                with st.spinner("Nemotron analyzing risk data and generating prevention orders..."):
                    risk_stats = {
                        "mean_risk": round(float(risk_df["ignition_risk"].mean()), 4),
                        "extreme_cells": extreme,
                        "high_cells": high,
                        "total_cells": len(risk_df),
                    }
                    affected = optimizer.get_affected_facilities(st.session_state.disabled_lines)
                    brief = st.session_state.nemotron_engine.generate_prevention_brief(
                        weather=WEATHER,
                        risk_stats=risk_stats,
                        shutoff_plan=st.session_state.selected_plan,
                        affected_facilities=affected if affected else None,
                    )
                    st.session_state.prevention_brief = brief

            if st.session_state.prevention_brief:
                st.markdown(f'<div class="brief-box">{st.session_state.prevention_brief}</div>', unsafe_allow_html=True)

    # ── TAB 4: COUNTERFACTUAL ──
    with tab_counterfactual:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🔄</div>
            <div>
                <div class="section-title">Counterfactual Simulation Engine</div>
                <div class="section-subtitle">Toggle interventions and watch the risk surface respond · Nemotron explains the causal chain</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("⚡ De-energize highest-risk line", use_container_width=True, key="icf_highest"):
                line_risks = optimizer.compute_line_risk_scores(WEATHER)
                highest = max(line_risks, key=line_risks.get)
                st.session_state.disabled_lines.add(highest)
                st.rerun()
        with qc2:
            if st.button("⚡ Apply top AI plan", use_container_width=True, key="icf_top"):
                if st.session_state.shutoff_plans:
                    top = st.session_state.shutoff_plans[0]
                    st.session_state.disabled_lines = set(top['lines_disabled'])
                    st.session_state.selected_plan = top
                    st.rerun()
                else:
                    st.info("Run grid optimization first")
        with qc3:
            if st.button("🔄 Reset all lines", use_container_width=True, key="icf_reset"):
                st.session_state.disabled_lines = set()
                from data_generator import compute_powerline_proximity, get_power_lines_df
                from risk_engine import compute_ignition_risk
                powerlines = get_power_lines_df()
                proximity = compute_powerline_proximity(st.session_state.terrain_df, powerlines)
                st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, proximity)
                st.rerun()

        if st.session_state.disabled_lines:
            from data_generator import compute_powerline_proximity, get_power_lines_df
            from risk_engine import compute_ignition_risk, compute_risk_reduction

            all_pl = get_power_lines_df()
            orig_proximity = compute_powerline_proximity(st.session_state.terrain_df, all_pl)
            orig_risk = compute_ignition_risk(st.session_state.terrain_df, orig_proximity)
            current_risk = st.session_state.risk_df
            reduction = compute_risk_reduction(st.session_state.terrain_df, orig_risk, current_risk)

            st.markdown(f"""
            <div class="stat-grid">
                <div class="stat-card green">
                    <div class="stat-label">Risk Reduction</div>
                    <div class="stat-value green">-{reduction['reduction_pct']:.1f}%</div>
                </div>
                <div class="stat-card green">
                    <div class="stat-label">Extreme Cells Eliminated</div>
                    <div class="stat-value green">-{reduction['extreme_cells_eliminated']}</div>
                </div>
                <div class="stat-card blue">
                    <div class="stat-label">Risk Before</div>
                    <div class="stat-value">{reduction['mean_risk_before']:.4f}</div>
                </div>
                <div class="stat-card blue">
                    <div class="stat-label">Risk After</div>
                    <div class="stat-value">{reduction['mean_risk_after']:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.session_state.nemotron_connected:
                if st.button("🧠 EXPLAIN COUNTERFACTUAL", use_container_width=True, key="icf_explain"):
                    with st.spinner("Nemotron analyzing causal chain..."):
                        disabled_names = [
                            optimizer.power_lines[pl_id]["name"]
                            for pl_id in st.session_state.disabled_lines
                        ]
                        action = f"De-energize: {', '.join(disabled_names)}"
                        affected = optimizer.get_affected_facilities(st.session_state.disabled_lines)
                        explanation = st.session_state.nemotron_engine.generate_counterfactual_explanation(
                            action_taken=action,
                            risk_before=reduction,
                            risk_after=reduction,
                            affected_facilities=affected,
                        )
                        st.session_state.counterfactual_explanation = explanation

                if st.session_state.counterfactual_explanation:
                    st.markdown(f'<div class="brief-box">{st.session_state.counterfactual_explanation}</div>', unsafe_allow_html=True)
        else:
            st.info("Disable some power lines to see counterfactual analysis.")

    # ── TAB 5: WEATHER TIMELINE ──
    with tab_timeline:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📈</div>
            <div>
                <div class="section-title">72-Hour Weather Forecast & Risk Evolution</div>
                <div class="section-subtitle">Diablo wind event building over Sonoma County · Peak conditions at hour 18-24</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        wx_data = st.session_state.weather_timeline

        st.markdown("##### Wind Speed (mph)")
        wind_chart_data = wx_data[["hour", "wind_speed_mph", "wind_gust_mph"]].set_index("hour")
        st.area_chart(wind_chart_data, color=["#00b4d8", "#ff3b3b"], height=250)

        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("##### Temperature (°F)")
            st.line_chart(wx_data.set_index("hour")["temperature_f"], color="#ff3b3b", height=200)
        with wc2:
            st.markdown("##### Humidity (%)")
            st.line_chart(wx_data.set_index("hour")["humidity_pct"], color="#00b4d8", height=200)

        peak = wx_data.loc[wx_data["wind_speed_mph"].idxmax()]
        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card red">
                <div class="stat-label">Peak Wind</div>
                <div class="stat-value red">{peak['wind_speed_mph']:.0f}<span style="font-size:0.9rem;"> mph</span></div>
                <div class="stat-delta" style="color:var(--text-secondary);">Hour {int(peak['hour'])}</div>
            </div>
            <div class="stat-card red">
                <div class="stat-label">Peak Gust</div>
                <div class="stat-value red">{peak['wind_gust_mph']:.0f}<span style="font-size:0.9rem;"> mph</span></div>
            </div>
            <div class="stat-card blue">
                <div class="stat-label">Min Humidity</div>
                <div class="stat-value blue">{wx_data['humidity_pct'].min():.0f}<span style="font-size:0.9rem;">%</span></div>
            </div>
            <div class="stat-card orange">
                <div class="stat-label">Max Temperature</div>
                <div class="stat-value orange">{wx_data['temperature_f'].max():.0f}<span style="font-size:0.9rem;">°F</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


else:
    # ═══════════════════════════════════════════════════════════════════════
    # DEMO MODE — VOICEOVER-SYNCED CINEMATIC PRESENTATION
    # ═══════════════════════════════════════════════════════════════════════
    demo_phase_names = ["THREAT MAP", "GRID ANALYSIS", "AI OPTIMIZATION", "COUNTERFACTUAL", "PREVENTION BRIEF"]
    current_phase = st.session_state.demo_phase % len(demo_phase_names)

    # Hidden auto-advance button (clicked by JS to trigger rerun)
    advance_col = st.columns(1)[0]
    with advance_col:
        if st.button("_advance", key="_vo_advance", type="secondary"):
            pass  # Phase is calculated from elapsed time above, just rerun

    # Hide the auto-advance button with JS (runs in components iframe)
    components.html("""
    <script>
    (function() {
        var doc = window.parent.document;
        var buttons = doc.querySelectorAll('button');
        for (var i = 0; i < buttons.length; i++) {
            if (buttons[i].textContent.trim() === '_advance') {
                var el = buttons[i].closest('[data-testid]') || buttons[i].parentElement;
                el.style.height = '0';
                el.style.overflow = 'hidden';
                el.style.margin = '0';
                el.style.padding = '0';
                buttons[i].style.height = '0';
                buttons[i].style.overflow = 'hidden';
            }
        }
    })();
    </script>
    """, height=0)

    # ── START EXPERIENCE or ACTIVE VOICEOVER ──
    if not st.session_state.demo_playing:
        # Show START EXPERIENCE overlay
        st.markdown("""
        <div class="vo-start-overlay">
            <div class="vo-start-title">EarthDial v3</div>
            <div class="vo-start-sub">Experience the full wildfire prevention system with narrated AI voiceover — powered by NVIDIA Nemotron.</div>
            <div class="vo-start-duration">⏱ 1:58 · 5 PHASES · AI NARRATED</div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([2, 1, 2])
        with sc2:
            if st.button("▶  START EXPERIENCE", use_container_width=True, key="start_vo"):
                st.session_state.demo_playing = True
                st.session_state.demo_audio_start = time.time()
                st.session_state.demo_phase = 0
                st.rerun()

        # Manual navigation when voiceover is NOT playing
        st.markdown("<br>", unsafe_allow_html=True)
        nav1, nav2, nav3 = st.columns([1, 6, 1])
        with nav1:
            if st.button("◀ PREV", use_container_width=True, key="demo_prev"):
                st.session_state.demo_phase = max(0, st.session_state.demo_phase - 1)
                st.rerun()
        with nav2:
            st.markdown(f"""
            <div style="text-align:center; font-size:0.7rem; color:var(--text-dim);">
                Or browse phases manually below
            </div>
            """, unsafe_allow_html=True)
        with nav3:
            if st.button("NEXT ▶", use_container_width=True, key="demo_next"):
                st.session_state.demo_phase = min(4, st.session_state.demo_phase + 1)
                st.rerun()

    else:
        # ── VOICEOVER IS PLAYING — inject audio controller ──
        elapsed_safe = max(0, demo_elapsed)

        # Calculate time until next phase transition (for auto-rerun)
        next_phase_idx = demo_auto_phase + 1
        if next_phase_idx < len(PHASE_TRANSITIONS):
            ms_until_next = max(500, int((PHASE_TRANSITIONS[next_phase_idx] - elapsed_safe) * 1000))
        else:
            # Last phase — wait until audio ends
            ms_until_next = max(500, int((AUDIO_DURATION - elapsed_safe) * 1000))

        # Cap the rerun interval — at least every 2 seconds for progress bar updates
        ms_until_rerun = min(ms_until_next, 2000)

        components.html(f"""
        <script>
        (function() {{
            // Create or get persistent audio object in parent window
            let audio = window.parent._earthdialAudio;
            if (!audio) {{
                audio = new Audio();
                audio.preload = 'auto';
                audio.src = './app/static/voiceover.mp3';
                window.parent._earthdialAudio = audio;
            }}

            // Sync playback position (only correct if drifted > 3s)
            const targetTime = {elapsed_safe:.2f};
            if (audio.paused) {{
                audio.currentTime = targetTime;
                audio.play().catch(function(e) {{ console.log('Autoplay:', e); }});
            }} else if (Math.abs(audio.currentTime - targetTime) > 3) {{
                audio.currentTime = targetTime;
            }}

            // Schedule auto-rerun for next phase transition or progress update
            setTimeout(function() {{
                // Find and click the hidden advance button
                const doc = window.parent.document;
                const buttons = doc.querySelectorAll('button');
                for (let i = 0; i < buttons.length; i++) {{
                    if (buttons[i].textContent.trim() === '_advance') {{
                        buttons[i].click();
                        break;
                    }}
                }}
            }}, {ms_until_rerun});
        }})();
        </script>
        """, height=0)

    # ── PHASE 0: THREAT MAP ──
    if current_phase == 0:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🗺️</div>
            <div>
                <div class="section-title">Real-Time 3D Ignition Risk Surface</div>
                <div class="section-subtitle">1,600 terrain cells · Rothermel-inspired fire model · Live wind & fuel analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        max_risk_idx = risk_df["ignition_risk"].idxmax()
        max_risk_point = risk_df.loc[max_risk_idx]

        from risk_engine import compute_multiple_spread_scenarios
        fire_scenarios = compute_multiple_spread_scenarios(
            ignition_lat=max_risk_point["lat"],
            ignition_lon=max_risk_point["lon"],
            hours_list=[3, 6, 12, 24],
        )

        from visualization import build_full_3d_map
        deck = build_full_3d_map(
            terrain_df=st.session_state.risk_df,
            powerlines_df=st.session_state.powerlines_df,
            substations_df=st.session_state.substations_df,
            facilities_df=st.session_state.facilities_df,
            wind_df=st.session_state.wind_df,
            fire_scenarios=fire_scenarios,
            disabled_lines=st.session_state.disabled_lines,
            affected_facility_ids=set(),
            ignition_point=(max_risk_point["lat"], max_risk_point["lon"]),
            show_risk_columns=True,
            show_heatmap=False,
            show_wind=True,
            show_fire_spread=True,
        )

        st.pydeck_chart(deck, height=650, use_container_width=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:12px;">
            <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.72rem; color:var(--text-secondary);">
                <span>🔴 <strong style="color:#ff3b3b">Red columns</strong> = Extreme ignition risk</span>
                <span>🟠 <strong style="color:#ffaa00">Orange columns</strong> = High risk</span>
                <span>🔵 <strong style="color:#00b4d8">Blue arcs</strong> = Active power lines</span>
                <span>🟢 <strong style="color:#76B900">Green dots</strong> = Substations</span>
                <span>➡️ <strong>Lines</strong> = Wind direction & speed</span>
                <span>🔶 <strong style="color:#ff9800">Polygons</strong> = Projected fire spread (3/6/12/24h)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PHASE 1: GRID ANALYSIS ──
    elif current_phase == 1:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">⚡</div>
            <div>
                <div class="section-title">Power Grid Risk Analysis</div>
                <div class="section-subtitle">8 transmission lines · 5 substations · 8 critical facilities · Real-time vegetation risk scoring</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        from grid_optimizer import GridOptimizer
        optimizer = GridOptimizer()
        line_risks = optimizer.compute_line_risk_scores(WEATHER)

        # Power line risk table
        pl_df = st.session_state.powerlines_df
        table_rows = ""
        for _, pl in pl_df.iterrows():
            score = line_risks.get(pl["id"], 0)
            veg = pl["vegetation_risk"]
            if score > 0.7:
                badge = '<span class="risk-badge extreme">EXTREME</span>'
            elif score > 0.5:
                badge = '<span class="risk-badge high">HIGH</span>'
            elif score > 0.3:
                badge = '<span class="risk-badge moderate">MODERATE</span>'
            else:
                badge = '<span class="risk-badge low">LOW</span>'

            table_rows += f"""
            <tr>
                <td style="font-weight:600; color:var(--text-primary);">{pl['name']}</td>
                <td><span style="font-family:'JetBrains Mono'; color:var(--text-primary);">{pl['voltage_kv']}</span> kV</td>
                <td>{badge}</td>
                <td><span style="font-family:'JetBrains Mono'; color:var(--{'danger' if veg > 0.75 else 'warning' if veg > 0.5 else 'nvidia-green'});">{veg:.0%}</span></td>
                <td><span style="font-family:'JetBrains Mono';">{pl['age_years']}</span> yrs</td>
                <td><span style="font-family:'JetBrains Mono'; font-weight:700; color:var(--{'danger' if score > 0.7 else 'warning' if score > 0.5 else 'nvidia-green'});">{score:.3f}</span></td>
            </tr>"""

        st.markdown(f"""
        <div class="glass-card">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Line Name</th>
                        <th>Voltage</th>
                        <th>Risk Level</th>
                        <th>Vegetation</th>
                        <th>Age</th>
                        <th>Score</th>
                    </tr>
                </thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        # Critical facilities
        st.markdown("""
        <div class="section-header" style="margin-top:20px;">
            <div class="section-icon">🏥</div>
            <div>
                <div class="section-title">Critical Facility Dependencies</div>
                <div class="section-subtitle">Priority facilities on monitored power feeders</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        from config import CRITICAL_FACILITIES, FACILITY_ICONS
        fac_rows = ""
        for cf in CRITICAL_FACILITIES:
            icon = FACILITY_ICONS.get(cf["type"], "📍")
            pri_badge = f'<span class="risk-badge extreme">P{cf["priority"]}</span>' if cf["priority"] == 1 else f'<span class="risk-badge moderate">P{cf["priority"]}</span>'
            fac_rows += f"""
            <tr>
                <td>{icon} {cf['name']}</td>
                <td>{cf['type'].replace('_', ' ').title()}</td>
                <td>{cf['feeder']}</td>
                <td>{pri_badge}</td>
            </tr>"""

        st.markdown(f"""
        <div class="glass-card">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Facility</th>
                        <th>Type</th>
                        <th>Feeder Line</th>
                        <th>Priority</th>
                    </tr>
                </thead>
                <tbody>{fac_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── PHASE 2: AI OPTIMIZATION ──
    elif current_phase == 2:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🧠</div>
            <div>
                <div class="section-title">GPU-Accelerated Shutoff Optimization</div>
                <div class="section-subtitle">Brute-force combination search · Risk/disruption tradeoff analysis · Critical load preservation</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        from grid_optimizer import GridOptimizer
        optimizer = GridOptimizer()

        if not st.session_state.shutoff_plans:
            with st.spinner("Running GPU-accelerated graph optimization..."):
                plans = optimizer.optimize_shutoffs(
                    weather=WEATHER,
                    max_shutoffs=3,
                    protect_critical=True,
                )
                st.session_state.shutoff_plans = plans

        if st.session_state.shutoff_plans:
            for plan in st.session_state.shutoff_plans[:5]:
                eff = plan['efficiency_ratio']
                eff_label = "OPTIMAL" if eff > 2.5 else "GOOD" if eff > 1.5 else "MODERATE"
                eff_color = "nvidia-green" if eff > 2.5 else "warning" if eff > 1.5 else "text-secondary"

                affected_text = ""
                if plan['affected_facilities']:
                    affected_names = [f['name'] for f in plan['affected_facilities']]
                    affected_text = f'<div class="plan-detail" style="color:var(--danger);">⚠️ Impacts: {", ".join(affected_names)}</div>'

                st.markdown(f"""
                <div class="plan-card">
                    <div class="plan-header">
                        <span class="plan-rank">PLAN #{plan['rank']} — {eff_label}</span>
                        <span style="display:flex; gap:12px;">
                            <span class="plan-efficiency" style="color:var(--{eff_color});">EFF {plan['efficiency_ratio']:.2f}</span>
                            <span style="font-family:'JetBrains Mono'; font-size:0.75rem; color:var(--nvidia-green);">CONF {plan['confidence']:.0%}</span>
                        </span>
                    </div>
                    <div class="plan-detail"><strong style="color:var(--nvidia-green);">Shutoff:</strong> {', '.join(plan['line_names'])}</div>
                    <div class="plan-detail"><strong>Risk removed:</strong> <span style="color:var(--nvidia-green); font-family:'JetBrains Mono';">{plan['total_risk_removed']:.4f}</span> · <strong>Disruption:</strong> <span style="font-family:'JetBrains Mono';">{plan['disruption_score']:.4f}</span></div>
                    <div class="plan-detail"><strong>Grid:</strong> {'✅ Connected' if plan['grid_connected'] else '❌ Fragmented'} · <strong>Components:</strong> {plan['num_components']}</div>
                    {affected_text}
                </div>
                """, unsafe_allow_html=True)

            # Apply best plan button
            if st.button("⚡ APPLY OPTIMAL PLAN", use_container_width=True, key="demo_apply"):
                top = st.session_state.shutoff_plans[0]
                st.session_state.disabled_lines = set(top['lines_disabled'])
                st.session_state.selected_plan = top
                st.rerun()

    # ── PHASE 3: COUNTERFACTUAL ──
    elif current_phase == 3:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🔄</div>
            <div>
                <div class="section-title">Counterfactual Scenario Comparison</div>
                <div class="section-subtitle">"What if we de-energize?" · Before/after risk surface · Nemotron causal analysis</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Auto-apply top plan if none applied
        from grid_optimizer import GridOptimizer
        optimizer = GridOptimizer()

        if not st.session_state.disabled_lines and st.session_state.shutoff_plans:
            top = st.session_state.shutoff_plans[0]
            st.session_state.disabled_lines = set(top['lines_disabled'])
            st.session_state.selected_plan = top
            from data_generator import compute_powerline_proximity, get_power_lines_df
            from risk_engine import compute_ignition_risk
            updated_pl = get_power_lines_df()
            updated_pl["active"] = ~updated_pl["id"].isin(st.session_state.disabled_lines)
            new_proximity = compute_powerline_proximity(
                st.session_state.terrain_df,
                updated_pl[updated_pl["active"]],
            )
            st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, new_proximity)

        if st.session_state.disabled_lines:
            from data_generator import compute_powerline_proximity, get_power_lines_df
            from risk_engine import compute_ignition_risk, compute_risk_reduction

            all_pl = get_power_lines_df()
            orig_proximity = compute_powerline_proximity(st.session_state.terrain_df, all_pl)
            orig_risk = compute_ignition_risk(st.session_state.terrain_df, orig_proximity)
            current_risk = st.session_state.risk_df
            reduction = compute_risk_reduction(st.session_state.terrain_df, orig_risk, current_risk)

            # Before vs After
            col_b, col_a = st.columns(2)
            with col_b:
                st.markdown("""
                <div class="glass-card" style="border-color: rgba(255,59,59,0.3);">
                    <div style="text-align:center; font-size:0.7rem; color:var(--danger); text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:10px;">BEFORE INTERVENTION</div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="stat-grid">
                        <div class="stat-card red">
                            <div class="stat-label">Mean Risk</div>
                            <div class="stat-value red" style="font-size:1.5rem;">{reduction['mean_risk_before']:.4f}</div>
                        </div>
                        <div class="stat-card red">
                            <div class="stat-label">Extreme Cells</div>
                            <div class="stat-value red" style="font-size:1.5rem;">{reduction['extreme_cells_before']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_a:
                st.markdown("""
                <div class="glass-card" style="border-color: rgba(118,185,0,0.3);">
                    <div style="text-align:center; font-size:0.7rem; color:var(--nvidia-green); text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:10px;">AFTER INTERVENTION</div>
                """, unsafe_allow_html=True)
                st.markdown(f"""
                    <div class="stat-grid">
                        <div class="stat-card green">
                            <div class="stat-label">Mean Risk</div>
                            <div class="stat-value green" style="font-size:1.5rem;">{reduction['mean_risk_after']:.4f}</div>
                        </div>
                        <div class="stat-card green">
                            <div class="stat-label">Extreme Cells</div>
                            <div class="stat-value green" style="font-size:1.5rem;">{reduction['extreme_cells_after']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Big reduction stat
            st.markdown(f"""
            <div class="glass-card glow-pulse" style="text-align:center; margin:16px 0; border-color: rgba(118,185,0,0.3);">
                <div style="font-size:0.7rem; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1.5px;">Total Risk Reduction</div>
                <div style="font-size:3rem; font-weight:900; color:var(--nvidia-green); font-family:'JetBrains Mono', monospace;">-{reduction['reduction_pct']:.1f}%</div>
                <div style="font-size:0.75rem; color:var(--text-secondary);">{reduction['extreme_cells_eliminated']} extreme risk cells eliminated</div>
            </div>
            """, unsafe_allow_html=True)

            # Show updated map
            max_risk_idx = st.session_state.risk_df["ignition_risk"].idxmax()
            max_risk_point = st.session_state.risk_df.loc[max_risk_idx]

            from risk_engine import compute_multiple_spread_scenarios
            fire_scenarios = compute_multiple_spread_scenarios(
                ignition_lat=max_risk_point["lat"],
                ignition_lon=max_risk_point["lon"],
                hours_list=[3, 6, 12, 24],
            )

            from visualization import build_full_3d_map
            deck = build_full_3d_map(
                terrain_df=st.session_state.risk_df,
                powerlines_df=st.session_state.powerlines_df,
                substations_df=st.session_state.substations_df,
                facilities_df=st.session_state.facilities_df,
                wind_df=st.session_state.wind_df,
                fire_scenarios=fire_scenarios,
                disabled_lines=st.session_state.disabled_lines,
                affected_facility_ids=set(),
                ignition_point=(max_risk_point["lat"], max_risk_point["lon"]),
                show_risk_columns=True,
                show_heatmap=False,
                show_wind=True,
                show_fire_spread=True,
            )

            st.pydeck_chart(deck, height=500, use_container_width=True)

        else:
            st.info("Navigate to Phase 3 (AI Optimization) first to generate shutoff plans.")

    # ── PHASE 4: PREVENTION BRIEF ──
    elif current_phase == 4:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">📋</div>
            <div>
                <div class="section-title">Nemotron-Generated Prevention Brief</div>
                <div class="section-subtitle">NVIDIA Llama-3.3-Nemotron-Super-49B synthesizes all data into an operator-ready prevention order</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.nemotron_connected:
            if not st.session_state.prevention_brief:
                if st.button("🧠 GENERATE PREVENTION BRIEF", use_container_width=True, key="demo_brief"):
                    with st.spinner("Nemotron is analyzing risk data and generating prevention orders..."):
                        risk_stats = {
                            "mean_risk": round(float(risk_df["ignition_risk"].mean()), 4),
                            "extreme_cells": extreme,
                            "high_cells": high,
                            "total_cells": len(risk_df),
                        }
                        from grid_optimizer import GridOptimizer
                        optimizer = GridOptimizer()
                        affected = optimizer.get_affected_facilities(st.session_state.disabled_lines)
                        brief = st.session_state.nemotron_engine.generate_prevention_brief(
                            weather=WEATHER,
                            risk_stats=risk_stats,
                            shutoff_plan=st.session_state.selected_plan,
                            affected_facilities=affected if affected else None,
                        )
                        st.session_state.prevention_brief = brief
                        st.rerun()
            else:
                st.markdown(f'<div class="brief-box">{st.session_state.prevention_brief}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding:40px;">
                <div style="font-size:2rem;">🧠</div>
                <div style="font-size:0.85rem; color:var(--text-primary); font-weight:600; margin-top:12px;">Nemotron Prevention Brief</div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-top:8px;">
                    Connect NVIDIA API key to generate AI-powered prevention orders.<br>
                    The Nemotron model synthesizes all risk data, weather conditions, and optimization results<br>
                    into a formal, operator-ready document with actionable recommendations.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── DEMO COMPLETE OVERLAY (shown after voiceover ends) ──
    if not st.session_state.demo_playing and st.session_state.demo_phase == 4 and st.session_state.demo_audio_start is None:
        # Stop audio in parent
        components.html("""
        <script>
        (function() {
            const audio = window.parent._earthdialAudio;
            if (audio) { audio.pause(); audio.currentTime = 0; }
        })();
        </script>
        """, height=0)

        st.markdown("""
        <div style="text-align:center; padding:40px 20px; margin-top:20px;">
            <div style="font-size:2rem; margin-bottom:12px;">✅</div>
            <div style="font-size:1.2rem; font-weight:700; color:var(--nvidia-green);">Experience Complete</div>
            <div style="font-size:0.78rem; color:var(--text-secondary); margin-top:8px;">
                You've seen the full EarthDial v3 pipeline — from real-time risk analysis to AI-generated prevention orders.
            </div>
        </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3 = st.columns([2, 1, 2])
        with rc2:
            if st.button("🔄 REPLAY", use_container_width=True, key="replay_vo"):
                st.session_state.demo_playing = True
                st.session_state.demo_audio_start = time.time()
                st.session_state.demo_phase = 0
                st.session_state.disabled_lines = set()
                st.session_state.shutoff_plans = None
                st.session_state.prevention_brief = None
                st.session_state.counterfactual_explanation = None
                st.session_state.selected_plan = None
                # Reset risk data
                from data_generator import compute_powerline_proximity, get_power_lines_df
                from risk_engine import compute_ignition_risk
                plines = get_power_lines_df()
                prox = compute_powerline_proximity(st.session_state.terrain_df, plines)
                st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, prox)
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-footer">
    <div class="footer-left">
        <span>🌍 EarthDial v3.0</span>
        <span>·</span>
        <span class="footer-nvidia">NVIDIA Nemotron · GPU Graph Optimization</span>
        <span>·</span>
        <span>Earth-2 Ecosystem Compatible</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="color:var(--nvidia-green); font-weight:600;">#NVIDIAGTC 2026</span>
        <span>·</span>
        <span>earthdial.ai</span>
    </div>
</div>
""", unsafe_allow_html=True)
