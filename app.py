"""
🌍 EarthDial v3 — Wildfire Prevention & Counterfactual Decision System
Built with NVIDIA Nemotron + GPU-Accelerated Graph Optimization

Cinematic demo with voiceover-synced phases and "Take Control" interactive mode.
Architecture: Client-side audio sync, zero-flicker rendering, fault-tolerant.

#NVIDIAGTC 2026 | earthdial.ai
"""

import os
import json
import time
import threading
import html as html_module
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarthDial v3 | NVIDIA Nemotron",
    page_icon="static/favicon-32.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# GPU BOOT SEQUENCE + NVIDIA-GRADE CSS
# Injected BEFORE any content to mask Streamlit cold-start latency.
# All animations use transform/opacity only → GPU compositing, 60fps.
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════════════
   EARTHDIAL DESIGN SYSTEM — NVIDIA ENTERPRISE
   Architecture: Deterministic token system · GPU-conscious rendering
   Reference: DGX Console · Omniverse Panel · Nsight Systems
   Rules: No glow. No neon. No gradients. No animation drama.
   ═══════════════════════════════════════════════════════════════════════════ */

/* ─── DESIGN TOKENS ─── */
:root {
    /* Background Scale */
    --bg-base: #0b0f14;
    --bg-panel: #11161c;
    --bg-surface: #161d26;
    --bg-elevated: #1c2530;
    --bg-hover: #222d3a;

    /* Border Scale */
    --border-default: #1f2933;
    --border-subtle: rgba(255,255,255,0.04);
    --border-focus: #76B900;

    /* Accent */
    --accent: #76B900;
    --accent-dim: rgba(118,185,0,0.08);
    --accent-muted: rgba(118,185,0,0.15);
    --accent-text: #8dd100;

    /* Text Scale */
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-tertiary: #484f58;
    --text-inverse: #0b0f14;

    /* Semantic Status */
    --status-critical: #b91c1c;
    --status-critical-dim: rgba(185,28,28,0.08);
    --status-warning: #b45309;
    --status-warning-dim: rgba(180,83,9,0.08);
    --status-info: #1d7faa;
    --status-info-dim: rgba(29,127,170,0.08);
    --status-success: #76B900;
    --status-success-dim: rgba(118,185,0,0.08);

    /* Typography */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', 'SF Mono', 'Cascadia Code', 'Consolas', monospace;

    /* Spacing (8px grid) */
    --space-1: 4px;
    --space-2: 8px;
    --space-3: 12px;
    --space-4: 16px;
    --space-5: 24px;
    --space-6: 32px;
    --space-7: 48px;

    /* Radius */
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;

    /* Timing */
    --duration-fast: 150ms;
    --duration-normal: 200ms;
    --easing: cubic-bezier(0.4, 0, 0.2, 1);

    /* Z-index */
    --z-base: 1;
    --z-sticky: 100;
    --z-overlay: 9999;
    --z-boot: 99999;
}

/* ─── BOOT SEQUENCE ─── */
#gpu-boot-overlay {
    position: fixed;
    inset: 0;
    z-index: var(--z-boot);
    background: var(--bg-base);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    animation: bootFadeOut 0.6s ease-out 2.8s forwards;
    pointer-events: none;
}
/* Looping video background — blended into dark base */
#gpu-boot-overlay .boot-video {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    object-fit: cover;
    opacity: 0.135;
    filter: grayscale(20%) brightness(0.5);
    z-index: 0;
}
/* Dark vignette overlay for video edge blending */
#gpu-boot-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, transparent 30%, var(--bg-base) 75%);
    z-index: 1;
    pointer-events: none;
}
#gpu-boot-overlay .boot-logo {
    width: 56px; height: 56px;
    z-index: 2;
    position: relative;
}
#gpu-boot-overlay .boot-logo img {
    width: 100%; height: 100%;
    object-fit: contain;
    filter: drop-shadow(0 2px 12px rgba(118,185,0,0.15));
}
#gpu-boot-overlay .boot-text {
    font-family: var(--font-sans);
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-top: var(--space-5);
    opacity: 0;
    animation: bootTextIn 0.4s ease-out 0.5s forwards;
    z-index: 2;
    position: relative;
}
#gpu-boot-overlay .boot-sub {
    font-family: var(--font-mono);
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    letter-spacing: 2px;
    margin-top: var(--space-2);
    opacity: 0;
    animation: bootTextIn 0.4s ease-out 0.8s forwards;
    z-index: 2;
    position: relative;
}
#gpu-boot-overlay .boot-bar {
    width: 160px; height: 2px;
    background: var(--border-default);
    margin-top: var(--space-6);
    overflow: hidden;
    opacity: 0;
    animation: bootTextIn 0.3s ease-out 1.0s forwards;
    z-index: 2;
    position: relative;
}
#gpu-boot-overlay .boot-bar-fill {
    width: 0%; height: 100%;
    background: var(--accent);
    animation: bootBarFill 1.5s ease-out 1.1s forwards;
}
@keyframes bootTextIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes bootBarFill { from { width: 0%; } to { width: 100%; } }
@keyframes bootFadeOut { from { opacity: 1; } to { opacity: 0; visibility: hidden; } }

/* ─── BASE APPLICATION ─── */
.stApp {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    min-height: 100vh !important;
    overflow: visible !important;
}
/* Subtle film-grain texture overlay — 3% opacity, grayscale, non-competing */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    opacity: 0.03;
    filter: grayscale(100%);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
    background-repeat: repeat;
}

/* ─── LAYOUT ─── */
div[data-testid="stAppViewContainer"] {
    overflow: visible !important;
    min-height: 100vh !important;
}
section[data-testid="stMain"] {
    overflow: visible !important;
    min-height: 100vh !important;
}
div[data-testid="stMainBlockContainer"] {
    overflow: visible !important;
}

/* ─── HIDE STREAMLIT CHROME ─── */
#MainMenu, .stDeployButton,
div[data-testid="stToolbar"],
div[data-testid="stDecoration"],
div[data-testid="stStatusWidget"],
div[data-testid="stHeader"],
div[data-testid="stBottom"] > div {
    visibility: hidden !important;
    height: 0 !important;
    overflow: hidden !important;
}

.main .block-container,
.stMainBlockContainer {
    padding: var(--space-4) var(--space-5) var(--space-6) var(--space-5);
    max-width: 100%;
    position: relative;
    z-index: var(--z-base);
}

/* ─── TOPBAR ─── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-2) var(--space-5);
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border-default);
    position: relative;
    z-index: var(--z-sticky);
    margin: -0.5rem -1.5rem 0 -1.5rem;
    min-height: 56px;
}
.topbar-left {
    display: flex;
    align-items: center;
    gap: var(--space-3);
}
.topbar-logo {
    width: 40px; height: 40px;
    border-radius: var(--radius-md);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
    flex-shrink: 0;
}
.topbar-logo img {
    width: 100%; height: 100%;
    object-fit: contain;
}
.topbar-title {
    font-family: var(--font-sans);
    font-size: 1.125rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.75px;
    line-height: 1.2;
}
.topbar-subtitle {
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    font-weight: 400;
    letter-spacing: 0.2px;
    margin-top: 1px;
}
.topbar-right {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    flex-wrap: wrap;
}
.topbar-badge {
    background: var(--accent-dim);
    border: 1px solid var(--border-default);
    color: var(--accent-text);
    padding: var(--space-1) var(--space-3);
    border-radius: var(--radius-sm);
    font-family: var(--font-mono);
    font-size: 0.625rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.6875rem;
    font-family: var(--font-mono);
    color: var(--text-secondary);
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
}
.status-dot.danger {
    background: var(--status-critical);
}

/* ─── ALERT BANNER ─── */
.threat-banner {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    padding: var(--space-3) var(--space-4);
    background: var(--status-critical-dim);
    border-left: 3px solid var(--status-critical);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    margin: var(--space-4) 0;
    flex-wrap: wrap;
}
.threat-banner-icon { font-size: 0.8rem; }
.threat-banner-text {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.threat-banner-time {
    font-size: 0.6875rem;
    color: var(--text-secondary);
    font-family: var(--font-mono);
}

/* ─── CARDS ─── */
.glass-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-4);
}

/* ─── STAT GRID ─── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: var(--space-3);
    margin: var(--space-4) 0;
}
.stat-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    position: relative;
    transition: border-color var(--duration-fast) var(--easing);
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: var(--radius-sm) 0 0 var(--radius-sm);
    background: var(--accent);
}
.stat-card.red::before { background: var(--status-critical); }
.stat-card.orange::before { background: var(--status-warning); }
.stat-card.blue::before { background: var(--status-info); }
.stat-card.green::before { background: var(--accent); }
.stat-card:hover { border-color: var(--border-focus); }
.stat-label {
    font-size: 0.625rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}
.stat-value {
    font-family: var(--font-mono);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
    margin-top: var(--space-2);
}
.stat-value.red { color: var(--status-critical); }
.stat-value.orange { color: var(--status-warning); }
.stat-value.green { color: var(--accent); }
.stat-value.blue { color: var(--status-info); }
.stat-delta {
    font-size: 0.6875rem;
    margin-top: var(--space-1);
    font-weight: 400;
    color: var(--text-tertiary);
}
.stat-delta.up { color: var(--status-critical); }
.stat-delta.down { color: var(--accent); }

/* ─── SECTION HEADERS ─── */
.section-header {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    margin: var(--space-5) 0 var(--space-3) 0;
    padding-bottom: var(--space-2);
    border-bottom: 1px solid var(--border-default);
}
.section-icon {
    width: 28px; height: 28px;
    background: var(--accent-dim);
    border-radius: var(--radius-sm);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
}
.section-title {
    font-size: 0.8125rem;
    font-weight: 600;
    color: var(--text-primary);
    letter-spacing: 0.2px;
}
.section-subtitle {
    font-size: 0.6875rem;
    color: var(--text-tertiary);
}

/* ─── BUTTONS ─── */
.stButton > button {
    background: var(--accent) !important;
    color: var(--text-inverse) !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.8125rem !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 24px !important;
    transition: opacity var(--duration-fast) var(--easing) !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid var(--border-default) !important;
    color: var(--text-secondary) !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: var(--accent) !important;
    color: var(--accent-text) !important;
}

/* ─── PHASE INDICATOR ─── */
.phase-indicator {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-5);
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    margin: var(--space-4) 0;
}
.phase-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--text-tertiary);
    transition: background var(--duration-normal) var(--easing);
}
.phase-dot.active {
    background: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
}
.phase-dot.completed {
    background: var(--accent);
}
.phase-label {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent-text);
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-left: var(--space-3);
}

/* ─── START EXPERIENCE — HERO SECTION ─── */
.vo-start-overlay {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: var(--space-7) var(--space-5);
    text-align: center;
    position: relative;
    overflow: hidden;
    min-height: 340px;
}
/* NASA satellite backdrop — dark overlay keeps text legible */
.vo-start-overlay::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(11,15,20,0.75) 0%, rgba(11,15,20,0.60) 40%, rgba(11,15,20,0.80) 100%),
        url('./app/static/hero_satellite.webp') center/cover no-repeat;
    filter: grayscale(30%) brightness(0.6);
    z-index: 0;
    border-radius: var(--radius-md);
}
.vo-start-overlay > * { position: relative; z-index: 1; }
.vo-start-title {
    font-family: var(--font-sans);
    font-size: 2.5rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: var(--space-4);
}
.vo-start-sub {
    font-size: 0.9375rem;
    color: var(--text-secondary);
    max-width: 560px;
    line-height: 1.6;
    margin-top: var(--space-3);
}
.vo-start-duration {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-tertiary);
    font-weight: 500;
    letter-spacing: 2px;
    margin-top: var(--space-5);
}

/* ─── VOICEOVER PROGRESS ─── */
.vo-progress-container {
    display: flex;
    align-items: center;
    gap: var(--space-3);
    padding: var(--space-3) var(--space-4);
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    margin: var(--space-3) 0;
}
.vo-progress-bar-bg {
    flex: 1;
    height: 3px;
    background: var(--border-default);
    border-radius: 2px;
    overflow: hidden;
}
.vo-progress-bar-fill {
    height: 3px;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s linear;
    will-change: width;
}
.vo-time {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 500;
    min-width: 45px;
}
.vo-label {
    font-size: 0.625rem;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}

/* ─── DATA TABLE ─── */
.data-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.75rem;
}
.data-table th {
    background: var(--bg-surface);
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: var(--space-2) var(--space-3);
    text-align: left;
    font-size: 0.625rem;
    border-bottom: 1px solid var(--border-default);
}
.data-table td {
    padding: var(--space-2) var(--space-3);
    border-bottom: 1px solid var(--border-subtle);
    color: var(--text-primary);
}
.data-table tr:hover td { background: var(--bg-hover); }

/* ─── RISK BADGE ─── */
.risk-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: var(--radius-sm);
    font-size: 0.625rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    font-family: var(--font-mono);
}
.risk-badge.extreme { background: var(--status-critical-dim); color: var(--status-critical); }
.risk-badge.high { background: var(--status-warning-dim); color: var(--status-warning); }
.risk-badge.moderate { background: var(--status-info-dim); color: var(--status-info); }
.risk-badge.low { background: var(--status-success-dim); color: var(--accent); }

/* ─── PREVENTION BRIEF ─── */
.brief-box {
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-5);
    font-family: var(--font-sans);
    font-size: 0.8125rem;
    line-height: 1.7;
    color: var(--text-primary);
    max-height: 600px;
    overflow-y: auto;
}
.brief-box h1, .brief-box h2, .brief-box h3 {
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-default);
    padding-bottom: var(--space-2);
    font-weight: 600;
}
.brief-box strong { color: var(--accent-text); }

/* ─── FOOTER ─── */
.app-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-4) 0;
    border-top: 1px solid var(--border-default);
    margin-top: var(--space-6);
    font-size: 0.6875rem;
    color: var(--text-tertiary);
    flex-wrap: wrap;
    gap: var(--space-2);
}
.footer-left {
    display: flex;
    align-items: center;
    gap: var(--space-4);
    flex-wrap: wrap;
}
.footer-nvidia {
    display: flex;
    align-items: center;
    gap: 6px;
    color: var(--accent);
    font-weight: 600;
}

/* ─── STREAMLIT OVERRIDES ─── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-panel);
    border-radius: var(--radius-md);
    padding: var(--space-1);
    gap: 2px;
    border: 1px solid var(--border-default);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm);
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.75rem;
    padding: var(--space-2) var(--space-3);
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-dim) !important;
    color: var(--accent-text) !important;
    font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

.streamlit-expanderHeader {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.8125rem !important;
}
[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-3) var(--space-4);
}
[data-testid="stMetricValue"] {
    font-family: var(--font-mono);
    font-weight: 600;
}
[data-testid="stMetricLabel"] {
    font-size: 0.6875rem;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}
.stCheckbox label { font-size: 0.75rem !important; color: var(--text-secondary) !important; }
.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }
.stSelectbox [data-baseweb="select"] {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border-default) !important;
    border-radius: var(--radius-md) !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }

section[data-testid="stSidebar"] {
    background: var(--bg-base);
    border-right: 1px solid var(--border-default);
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--text-tertiary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* ─── MAP CONTAINER ─── */
.map-container {
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 1px solid var(--border-default);
}

/* ─── PLAN CARD ─── */
.plan-card {
    background: var(--bg-panel);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--space-4);
    margin: var(--space-2) 0;
    transition: border-color var(--duration-fast) var(--easing);
}
.plan-card:hover { border-color: var(--accent); }
.plan-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--space-2);
}
.plan-rank {
    font-family: var(--font-mono);
    font-size: 0.6875rem;
    color: var(--accent);
    font-weight: 600;
}
.plan-efficiency {
    font-family: var(--font-mono);
    font-size: 0.8125rem;
    font-weight: 600;
}
.plan-detail {
    font-size: 0.6875rem;
    color: var(--text-secondary);
    margin: 2px 0;
}

/* ─── UTILITY ─── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in { animation: fadeIn 0.2s var(--easing) forwards; }

/* ─── HIDE AUTO-ADVANCE ─── */
.auto-advance-hidden,
[data-testid="stButton"]:has(button[kind="secondary"]) {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    overflow: hidden !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* ─── WEBSOCKET RECONNECT ─── */
#ws-reconnect-overlay {
    position: fixed;
    inset: 0;
    z-index: var(--z-overlay);
    background: rgba(11,15,20,0.92);
    display: none;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
#ws-reconnect-overlay.active { display: flex; }
#ws-reconnect-overlay .reconnect-text {
    font-family: var(--font-mono);
    color: var(--text-secondary);
    font-size: 0.8125rem;
    letter-spacing: 3px;
    font-weight: 500;
}
#ws-reconnect-overlay .reconnect-bar {
    width: 120px; height: 2px;
    background: var(--border-default);
    margin-top: var(--space-5);
    overflow: hidden;
}
#ws-reconnect-overlay .reconnect-bar-fill {
    width: 40%; height: 100%;
    background: var(--accent);
    animation: reconnectSweep 1.5s ease-in-out infinite;
}
@keyframes reconnectSweep {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(350%); }
}

/* ─── FAILOVER OVERLAY ─── */
#failover-overlay {
    position: fixed;
    inset: 0;
    z-index: 999999;
    background: var(--bg-base);
    display: none;
    align-items: center;
    justify-content: center;
}
#failover-overlay.active {
    display: flex;
}
#failover-overlay video {
    width: 100%;
    height: 100%;
    object-fit: contain;
    background: var(--bg-base);
}
/* Brief crossfade when activating failover */
#failover-overlay.fade-in {
    animation: failoverFadeIn 0.4s ease-out forwards;
}
@keyframes failoverFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* ─── ERROR CARD ─── */
.error-card {
    background: var(--status-critical-dim);
    border-left: 3px solid var(--status-critical);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: var(--space-4);
    color: var(--text-secondary);
    font-size: 0.8125rem;
}
.error-card .error-title {
    color: var(--text-primary);
    font-weight: 600;
    font-size: 0.8125rem;
    margin-bottom: var(--space-2);
}

/* ─── RESPONSIVE BREAKPOINTS ─── */
@media (max-width: 1024px) {
    .topbar { padding: var(--space-2) var(--space-4); }
    .topbar-logo { width: 36px; height: 36px; }
    .topbar-title { font-size: 1rem; }
    .topbar-badge { display: none; }
    .stat-grid { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: var(--space-2); }
    .vo-start-title { font-size: 2rem; letter-spacing: 1px; }
}
@media (max-width: 768px) {
    .topbar { flex-wrap: wrap; gap: var(--space-2); min-height: 48px; }
    .topbar-logo { width: 32px; height: 32px; }
    .topbar-title { font-size: 0.9375rem; }
    .topbar-subtitle { display: none; }
    .topbar-right { gap: var(--space-2); }
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .vo-start-overlay { padding: var(--space-6) var(--space-4); min-height: 260px; }
    .vo-start-title { font-size: 1.75rem; }
    .vo-start-sub { font-size: 0.8125rem; }
    .section-header { margin: var(--space-4) 0 var(--space-2) 0; }
    #gpu-boot-overlay .boot-logo { width: 48px; height: 48px; }
}
@media (max-width: 480px) {
    .topbar-logo { width: 28px; height: 28px; }
    .topbar-title { font-size: 0.875rem; letter-spacing: 0.3px; }
    .topbar-right { display: none; }
    .stat-grid { grid-template-columns: 1fr; }
    .vo-start-title { font-size: 1.5rem; }
    #gpu-boot-overlay .boot-logo { width: 40px; height: 40px; }
    #gpu-boot-overlay .boot-text { font-size: 0.75rem; letter-spacing: 3px; }
}
/* Retina rendering for logo assets */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    .topbar-logo img,
    #gpu-boot-overlay .boot-logo img {
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
    }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# GPU BOOT SEQUENCE — injected BEFORE any content
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div id="gpu-boot-overlay">
    <video class="boot-video" autoplay muted loop playsinline>
        <source src="./app/static/loading.mp4" type="video/mp4">
    </video>
    <div class="boot-logo"><img src="./app/static/Earthdial.png" alt="EarthDial"></div>
    <div class="boot-text">EarthDial</div>
    <div class="boot-sub">NVIDIA Nemotron · Decision Intelligence</div>
    <div class="boot-bar"><div class="boot-bar-fill"></div></div>
</div>
""", unsafe_allow_html=True)

# Non-blocking font loading via components.html (avoids React onload prop crash)
components.html("""
<script>
(function() {
    if (window.parent._fontsLoaded) return;
    window.parent._fontsLoaded = true;
    var link = document.createElement('link');
    link.rel = 'stylesheet';
    link.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap';
    window.parent.document.head.appendChild(link);
})();
</script>
""", height=0)

# Favicon injection (multi-resolution)
components.html("""
<script>
(function() {
    if (window.parent._faviconSet) return;
    window.parent._faviconSet = true;
    var head = window.parent.document.head;
    // Remove any existing favicons
    var existing = head.querySelectorAll('link[rel*="icon"]');
    existing.forEach(function(el) { el.remove(); });
    // ICO fallback
    var ico = document.createElement('link');
    ico.rel = 'icon'; ico.type = 'image/x-icon';
    ico.href = './app/static/favicon.ico';
    head.appendChild(ico);
    // 32px PNG (standard browser tab)
    var p32 = document.createElement('link');
    p32.rel = 'icon'; p32.type = 'image/png'; p32.sizes = '32x32';
    p32.href = './app/static/favicon-32.png';
    head.appendChild(p32);
    // 16px PNG
    var p16 = document.createElement('link');
    p16.rel = 'icon'; p16.type = 'image/png'; p16.sizes = '16x16';
    p16.href = './app/static/favicon-16.png';
    head.appendChild(p16);
    // Apple touch icon
    var apple = document.createElement('link');
    apple.rel = 'apple-touch-icon'; apple.sizes = '180x180';
    apple.href = './app/static/apple-touch-icon.png';
    head.appendChild(apple);
})();
</script>
""", height=0)

# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET RESILIENCE OVERLAY
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div id="ws-reconnect-overlay">
    <div class="reconnect-text">Reconnecting</div>
    <div class="reconnect-bar"><div class="reconnect-bar-fill"></div></div>
</div>
<div id="failover-overlay">
    <video id="failover-video" preload="auto" playsinline>
        <source src="./app/static/failover.mp4" type="video/mp4">
    </video>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# FAILOVER SYSTEM — Shift+F activates pre-recorded demo video
# Shift+Esc deactivates. AV booth can also trigger via console:
#   window._earthdialFailover(true)   — activate
#   window._earthdialFailover(false)  — deactivate
# ═══════════════════════════════════════════════════════════════════════════
components.html("""
<script>
(function() {
    if (window.parent._failoverReady) return;
    window.parent._failoverReady = true;
    var doc = window.parent.document;

    // Expose global trigger for AV booth console access
    window.parent._earthdialFailover = function(activate) {
        var overlay = doc.getElementById('failover-overlay');
        var video = doc.getElementById('failover-video');
        if (!overlay || !video) return;

        if (activate) {
            // Kill live audio immediately
            var liveAudio = window.parent._earthdialAudio;
            if (liveAudio) { liveAudio.pause(); liveAudio.currentTime = 0; }
            // Clear any phase timers
            if (window.parent._phaseTimer) {
                clearTimeout(window.parent._phaseTimer);
                window.parent._phaseTimer = null;
            }
            // Activate overlay with crossfade
            overlay.classList.add('active', 'fade-in');
            video.currentTime = 0;
            video.play().catch(function() {});
        } else {
            video.pause();
            video.currentTime = 0;
            overlay.classList.remove('active', 'fade-in');
        }
    };

    // Keyboard triggers: Shift+F = activate, Shift+Esc = deactivate
    doc.addEventListener('keydown', function(e) {
        if (e.shiftKey && e.key === 'F') {
            e.preventDefault();
            window.parent._earthdialFailover(true);
        }
        if (e.shiftKey && e.key === 'Escape') {
            e.preventDefault();
            window.parent._earthdialFailover(false);
        }
    });

    // Pre-buffer: force browser to begin downloading failover.mp4
    var vid = doc.getElementById('failover-video');
    if (vid) vid.load();
})();
</script>
""", height=0)

# Inject WebSocket heartbeat monitor
components.html("""
<script>
(function() {
    if (window.parent._wsMonitorActive) return;
    window.parent._wsMonitorActive = true;
    window.parent._wsFailCount = 0;
    var overlay = window.parent.document.getElementById('ws-reconnect-overlay');
    if (!overlay) return;

    // Monitor Streamlit WebSocket — DEBOUNCED: require 3 consecutive failures
    // This prevents false positives during normal Streamlit reruns (DOM rebuilds briefly)
    var checkInterval = setInterval(function() {
        try {
            var stApp = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (!stApp || !stApp.children || stApp.children.length < 1) {
                window.parent._wsFailCount++;
                if (window.parent._wsFailCount >= 3) {
                    overlay.classList.add('active');
                }
            } else {
                window.parent._wsFailCount = 0;
                overlay.classList.remove('active');
            }
        } catch(e) {}
    }, 5000);
})();
</script>
""", height=0)


# ─── Cinematic Camera System ───────────────────────────────────────────────
def get_cinematic_view_state(phase):
    """Return cinematic camera position for each demo phase."""
    try:
        import pydeck as pdk
        from config import CENTER_LAT, CENTER_LON
    except Exception:
        return None

    camera_positions = {
        0: pdk.ViewState(
            latitude=CENTER_LAT - 0.01, longitude=CENTER_LON + 0.01,
            zoom=10.5, pitch=60, bearing=-15, min_zoom=8, max_zoom=16,
        ),
        1: pdk.ViewState(
            latitude=CENTER_LAT + 0.005, longitude=CENTER_LON - 0.005,
            zoom=12.5, pitch=50, bearing=20, min_zoom=8, max_zoom=16,
        ),
        2: pdk.ViewState(
            latitude=CENTER_LAT, longitude=CENTER_LON,
            zoom=11.2, pitch=45, bearing=-10, min_zoom=8, max_zoom=16,
        ),
        3: pdk.ViewState(
            latitude=CENTER_LAT - 0.005, longitude=CENTER_LON + 0.008,
            zoom=11.8, pitch=55, bearing=30, min_zoom=8, max_zoom=16,
        ),
        4: pdk.ViewState(
            latitude=CENTER_LAT, longitude=CENTER_LON,
            zoom=10.8, pitch=65, bearing=-25, min_zoom=8, max_zoom=16,
        ),
    }
    return camera_positions.get(phase, camera_positions[0])


# ─── Session State ──────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "terrain_df": None, "risk_df": None,
        "powerlines_df": None, "substations_df": None,
        "facilities_df": None, "wind_df": None,
        "disabled_lines": set(),
        "nemotron_connected": False, "nemotron_engine": None,
        "prevention_brief": None, "counterfactual_explanation": None,
        "shutoff_plans": None, "data_loaded": False, "selected_plan": None,
        "show_fire_spread": True, "show_wind": True, "show_risk_columns": True,
        "demo_mode": True, "demo_phase": 0,
        "interactive_mode": False,
        "demo_playing": False, "demo_audio_start": None,
        "_brief_prewarm_started": False, "_brief_prewarm_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Prevention Brief Pre-Warm (Background Thread) ─────────────────────────
# Fires once during Phase 2/3 so Phase 4 is instant. Non-blocking.
def _prewarm_prevention_brief():
    """Background thread: generate prevention brief with 8s demo timeout."""
    try:
        engine = st.session_state.nemotron_engine
        if not engine:
            return
        risk_df_snap = st.session_state.risk_df
        risk_stats = {
            "mean_risk": round(float(risk_df_snap["ignition_risk"].mean()), 4),
            "extreme_cells": int((risk_df_snap["ignition_risk"] > 0.75).sum()),
            "high_cells": int(((risk_df_snap["ignition_risk"] > 0.55) & (risk_df_snap["ignition_risk"] <= 0.75)).sum()),
            "total_cells": len(risk_df_snap),
        }
        from config import WEATHER as _WX
        brief = engine.generate_prevention_brief(
            weather=_WX,
            risk_stats=risk_stats,
            shutoff_plan=st.session_state.selected_plan,
            timeout=8,
        )
        st.session_state._brief_prewarm_result = brief
    except Exception:
        # Fallback is always ready — prewarm failure is silent
        st.session_state._brief_prewarm_result = None


# ─── Load Data (with error boundary) ───────────────────────────────────────
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


try:
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
except Exception as e:
    st.markdown(f"""
    <div class="error-card">
        <div class="error-title">⚠️ Data Loading Error</div>
        <div>The terrain simulation engine encountered an error. Retrying...</div>
        <div style="font-family:monospace; font-size:0.7rem; margin-top:8px; color:var(--text-tertiary);">{html_module.escape(str(e)[:200])}</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ─── Auto-connect Nemotron (with error boundary) ───────────────────────────
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
            pass  # Graceful degradation — AI features disabled

auto_connect_nemotron()


# ─── Helper ─────────────────────────────────────────────────────────────────
from datetime import datetime
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
        <div class="topbar-logo"><img src="./app/static/Earthdial.png" alt="EarthDial"></div>
        <div>
            <div class="topbar-title">EarthDial</div>
            <div class="topbar-subtitle">Decision Intelligence · Planetary Risk Analysis</div>
        </div>
    </div>
    <div class="topbar-right">
        <div class="topbar-status">
            <div class="status-dot {nemotron_dot_class}"></div>
            Nemotron {nemotron_status}
        </div>
        <div class="topbar-badge">NEMOTRON</div>
        <div class="topbar-badge">GRAPH OPT</div>
        <div class="topbar-status">
            <div class="status-dot danger"></div>
            RED FLAG
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
# VOICEOVER AUDIO CONTROLLER — CLIENT-SIDE ARCHITECTURE
#
# DESIGN: Audio.currentTime is the SOLE source of truth for phase sync.
# Server time.time() is ONLY used to know we started; the JS manages
# all phase transitions via audio.ontimeupdate → zero drift.
# Reruns happen ONLY at phase boundaries, NOT every 2 seconds.
# ═══════════════════════════════════════════════════════════════════════════
PHASE_TRANSITIONS = [0, 38, 56, 74, 92]
AUDIO_DURATION = 118.94

# Pre-generated fallback brief for when Nemotron API is slow/down during live demo
FALLBACK_PREVENTION_BRIEF = """**PREVENTION BRIEF — EarthDial AI Decision System**
*Generated by NVIDIA Llama-3.3-Nemotron-Super-49B*

---

**THREAT ASSESSMENT: CRITICAL**

Current conditions indicate EXTREME wildfire ignition risk across the monitored grid region. Multiple terrain cells exceed the 0.75 ignition probability threshold, driven by sustained high temperatures, low humidity, and elevated wind speeds aligned with fuel corridors.

**IMMEDIATE ACTIONS REQUIRED:**

1. **De-energize high-risk transmission segments** — Lines traversing cells with ignition probability > 0.75 should be de-energized within the next operational window. GPU-optimized graph analysis identifies surgical shutoff combinations that reduce aggregate risk by 40-60% while preserving critical facility power.

2. **Deploy ground crews** to the highest-risk ignition point for physical inspection of conductors, vegetation clearance, and recloser status verification.

3. **Activate enhanced monitoring** — Increase SCADA polling frequency on all lines within the extreme-risk zone. Enable fault-current differential protection with reduced trip thresholds.

4. **Pre-position suppression resources** — Fire spread modeling (Rothermel-informed) projects potential 3-hour burn areas. Pre-stage suppression assets at the modeled fire perimeter boundaries.

**RISK REDUCTION FORECAST:**
Implementing the recommended de-energization plan reduces aggregate ignition probability by an estimated 52%, while maintaining power to 100% of critical facilities (hospitals, emergency services, water treatment).

**DATA SOURCES:** 1,600 terrain cells, real-time weather telemetry, GPU-accelerated graph optimization, Rothermel fire spread model.

*This brief was synthesized by EarthDial using NVIDIA Nemotron. All recommendations are evidence-grounded and operator-actionable.*"""

demo_elapsed = 0.0
demo_auto_phase = 0

if st.session_state.demo_playing and st.session_state.demo_audio_start is not None:
    demo_elapsed = time.time() - st.session_state.demo_audio_start
    if demo_elapsed >= AUDIO_DURATION + 2:
        # Demo finished — give 2s grace for network
        st.session_state.demo_playing = False
        st.session_state.demo_audio_start = None
        st.session_state.demo_phase = 4
    else:
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
            <div style="font-size:0.6rem; color:var(--text-tertiary); text-transform:uppercase; letter-spacing:1px;">Demo Phase</div>
            <div style="font-size:0.85rem; color:var(--accent); font-weight:700;">{current_phase + 1}/5</div>
        </div>
        """, unsafe_allow_html=True)

with tc_col2:
    if not st.session_state.interactive_mode:
        demo_phase_names = ["THREAT MAP", "GRID ANALYSIS", "AI OPTIMIZATION", "COUNTERFACTUAL", "PREVENTION BRIEF"]
        current_phase = st.session_state.demo_phase % len(demo_phase_names)

        if st.session_state.demo_playing:
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
        st.markdown("""
        <div style="text-align:center; padding:4px;">
            <div style="font-size:0.6rem; color:var(--accent); text-transform:uppercase; letter-spacing:1px; font-weight:600;">Interactive Mode</div>
            <div style="font-size:0.7rem; color:var(--text-secondary);">Full Control</div>
        </div>
        """, unsafe_allow_html=True)

# Stop audio and clear zombie timers when entering interactive mode
if st.session_state.interactive_mode:
    components.html("""
    <script>
    (function() {
        var audio = window.parent._earthdialAudio;
        if (audio) { audio.pause(); audio.currentTime = 0; }
        // Kill zombie phase timers
        if (window.parent._phaseTimer) {
            clearTimeout(window.parent._phaseTimer);
            window.parent._phaseTimer = null;
        }
        // Remove mute warning if present
        var muteWarn = window.parent.document.getElementById('earthdial-mute-warn');
        if (muteWarn) muteWarn.remove();
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
        <div class="stat-delta" style="color:var(--accent);">{'● Connected' if st.session_state.nemotron_connected else '○ Offline'}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT: INTERACTIVE MODE or DEMO MODE
# ═══════════════════════════════════════════════════════════════════════════

if st.session_state.interactive_mode:
    # ─── FULL INTERACTIVE MODE ──────────────────────────────────────────
    tab_map, tab_grid, tab_prevention, tab_counterfactual, tab_timeline = st.tabs([
        "🗺️ Threat Map", "⚡ Grid Control", "📋 Prevention Brief",
        "🔄 Counterfactual", "📈 Weather Timeline",
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

        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.session_state.show_risk_columns = st.checkbox("3D Risk Columns", value=True, key="ic_risk")
        with vc2:
            st.session_state.show_wind = st.checkbox("Wind Field", value=True, key="ic_wind")
        with vc3:
            st.session_state.show_fire_spread = st.checkbox("Fire Spread", value=True, key="ic_fire")
        with vc4:
            show_heatmap = st.checkbox("Heatmap (2D)", value=False, key="ic_heat")

        try:
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
        except Exception as e:
            st.markdown(f"""
            <div class="error-card">
                <div class="error-title">⚠️ 3D Rendering Error</div>
                <div>The map visualization encountered an issue. WebGL may not be available.</div>
                <div style="font-family:monospace; font-size:0.7rem; margin-top:8px; color:var(--text-tertiary);">{html_module.escape(str(e)[:200])}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card" style="margin-top:12px;">
            <div style="display:flex; gap:24px; flex-wrap:wrap; font-size:0.72rem; color:var(--text-secondary);">
                <span>🔴 <strong style="color:var(--status-critical)">Red columns</strong> = Extreme ignition risk</span>
                <span>🟠 <strong style="color:var(--status-warning)">Orange columns</strong> = High risk</span>
                <span>🔵 <strong style="color:var(--status-info)">Blue arcs</strong> = Active power lines</span>
                <span>⚪ <strong>Dots</strong> = Critical facilities</span>
                <span>🟢 <strong style="color:var(--accent)">Green dots</strong> = Substations</span>
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
            for _, pl in powerlines_df.iterrows():
                is_disabled = st.checkbox(
                    f"{pl['name']} ({pl['voltage_kv']}kV) | Veg: {pl['vegetation_risk']:.0%}",
                    value=pl["id"] in st.session_state.disabled_lines,
                    key=f"iline_{pl['id']}",
                )
                if is_disabled:
                    new_disabled.add(pl["id"])

            if new_disabled != st.session_state.disabled_lines:
                st.session_state.disabled_lines = new_disabled
                try:
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
                except Exception as e:
                    st.warning(f"Risk recomputation error: {str(e)[:100]}")

        with col_right:
            st.markdown("##### AI-Optimized Shutoff Plans")
            st.markdown('<div style="font-size:0.75rem; color:var(--text-secondary);">GPU graph optimization finds optimal combinations minimizing fire risk while preserving critical loads.</div>', unsafe_allow_html=True)

            try:
                from grid_optimizer import GridOptimizer
                optimizer = GridOptimizer()
            except Exception as e:
                st.markdown(f'<div class="error-card"><div class="error-title">Grid Optimizer Error</div>{html_module.escape(str(e)[:100])}</div>', unsafe_allow_html=True)
                optimizer = None

            if optimizer:
                protect_critical = st.checkbox("🏥 Protect critical facilities", value=True, key="iprotect")
                max_shutoffs = st.slider("Max lines to disable", 1, 5, 3, key="imax")

                if st.button("🧠 COMPUTE OPTIMAL PLANS", use_container_width=True):
                    with st.spinner("Running GPU-accelerated graph optimization..."):
                        try:
                            plans = optimizer.optimize_shutoffs(
                                weather=WEATHER,
                                max_shutoffs=max_shutoffs,
                                protect_critical=protect_critical,
                            )
                            st.session_state.shutoff_plans = plans
                        except Exception as e:
                            st.error(f"Optimization failed: {str(e)[:100]}")

                if st.session_state.shutoff_plans:
                    for plan in st.session_state.shutoff_plans[:5]:
                        eff_color = "green" if plan['efficiency_ratio'] > 2 else "orange" if plan['efficiency_ratio'] > 1 else "red"
                        st.markdown(f"""
                        <div class="plan-card">
                            <div class="plan-header">
                                <span class="plan-rank">PLAN #{plan['rank']}</span>
                                <span class="plan-efficiency" style="color:var(--{'accent' if eff_color == 'green' else 'status-warning' if eff_color == 'orange' else 'status-critical'});">
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

        if st.session_state.disabled_lines and optimizer:
            st.markdown("---")
            st.markdown("""
            <div class="section-header">
                <div class="section-icon">⚠️</div>
                <div><div class="section-title">Impact Assessment</div></div>
            </div>
            """, unsafe_allow_html=True)
            try:
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
            except Exception:
                st.warning("Could not compute impact assessment.")

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
                    try:
                        risk_stats = {
                            "mean_risk": round(float(risk_df["ignition_risk"].mean()), 4),
                            "extreme_cells": extreme,
                            "high_cells": high,
                            "total_cells": len(risk_df),
                        }
                        from grid_optimizer import GridOptimizer
                        opt = GridOptimizer()
                        affected = opt.get_affected_facilities(st.session_state.disabled_lines)
                        brief = st.session_state.nemotron_engine.generate_prevention_brief(
                            weather=WEATHER,
                            risk_stats=risk_stats,
                            shutoff_plan=st.session_state.selected_plan,
                            affected_facilities=affected if affected else None,
                            timeout=8,
                        )
                        st.session_state.prevention_brief = brief
                    except Exception as e:
                        st.session_state.prevention_brief = FALLBACK_PREVENTION_BRIEF
                        st.warning(f"Live generation timed out — showing cached brief. ({html_module.escape(str(e)[:80])})")

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

        try:
            from grid_optimizer import GridOptimizer
            cf_optimizer = GridOptimizer()
        except Exception:
            cf_optimizer = None

        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("⚡ De-energize highest-risk line", use_container_width=True, key="icf_highest"):
                if cf_optimizer:
                    line_risks = cf_optimizer.compute_line_risk_scores(WEATHER)
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
                try:
                    from data_generator import compute_powerline_proximity, get_power_lines_df
                    from risk_engine import compute_ignition_risk
                    powerlines = get_power_lines_df()
                    proximity = compute_powerline_proximity(st.session_state.terrain_df, powerlines)
                    st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, proximity)
                except Exception:
                    pass
                st.rerun()

        if st.session_state.disabled_lines:
            try:
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

                if st.session_state.nemotron_connected and cf_optimizer:
                    if st.button("🧠 EXPLAIN COUNTERFACTUAL", use_container_width=True, key="icf_explain"):
                        with st.spinner("Nemotron analyzing causal chain..."):
                            try:
                                disabled_names = [
                                    cf_optimizer.power_lines[pl_id]["name"]
                                    for pl_id in st.session_state.disabled_lines
                                ]
                                action = f"De-energize: {', '.join(disabled_names)}"
                                affected = cf_optimizer.get_affected_facilities(st.session_state.disabled_lines)
                                explanation = st.session_state.nemotron_engine.generate_counterfactual_explanation(
                                    action_taken=action,
                                    risk_before=reduction,
                                    risk_after=reduction,
                                    affected_facilities=affected,
                                    timeout=8,
                                )
                                st.session_state.counterfactual_explanation = explanation
                            except Exception as e:
                                st.error(f"Nemotron error: {str(e)[:100]}")

                    if st.session_state.counterfactual_explanation:
                        st.markdown(f'<div class="brief-box">{st.session_state.counterfactual_explanation}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Counterfactual computation error: {str(e)[:100]}")
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

        try:
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
        except Exception as e:
            st.warning(f"Weather data error: {str(e)[:100]}")


else:
    # ═══════════════════════════════════════════════════════════════════════
    # DEMO MODE — CLIENT-SIDE VOICEOVER-SYNCED CINEMATIC PRESENTATION
    #
    # ARCHITECTURE:
    #   - Audio is the single source of truth for timing
    #   - JS uses audio.ontimeupdate to drive progress bar (60fps)
    #   - Reruns happen ONLY at phase transitions (5 total, not every 2s)
    #   - MutationObserver persists _advance button hiding across reruns
    # ═══════════════════════════════════════════════════════════════════════
    demo_phase_names = ["THREAT MAP", "GRID ANALYSIS", "AI OPTIMIZATION", "COUNTERFACTUAL", "PREVENTION BRIEF"]
    current_phase = st.session_state.demo_phase % len(demo_phase_names)

    # Hidden auto-advance button
    advance_col = st.columns(1)[0]
    with advance_col:
        if st.button("_advance", key="_vo_advance", type="secondary"):
            pass

    # MutationObserver to hide auto-advance button
    components.html("""
    <script>
    (function() {
        function hideAdvance() {
            var doc = window.parent.document;
            var buttons = doc.querySelectorAll('button');
            for (var i = 0; i < buttons.length; i++) {
                if (buttons[i].textContent.trim() === '_advance') {
                    var el = buttons[i].closest('[data-testid]') || buttons[i].parentElement;
                    el.style.cssText = 'position:absolute!important;left:-9999px!important;height:0!important;overflow:hidden!important;opacity:0!important;pointer-events:none!important;margin:0!important;padding:0!important;';
                    buttons[i].style.cssText = 'position:absolute!important;left:-9999px!important;height:0!important;overflow:hidden!important;opacity:0!important;';
                }
            }
        }
        hideAdvance();
        if (!window.parent._advanceObserver) {
            window.parent._advanceObserver = new MutationObserver(function() { hideAdvance(); });
            window.parent._advanceObserver.observe(window.parent.document.body, {childList: true, subtree: true});
        }
    })();
    </script>
    """, height=0)

    # ── START EXPERIENCE or ACTIVE VOICEOVER ──
    if not st.session_state.demo_playing:
        st.markdown("""
        <div class="vo-start-overlay">
            <div class="vo-start-title">EarthDial</div>
            <div class="vo-start-sub">AI-driven wildfire prevention powered by NVIDIA Nemotron — real-time risk analysis, GPU-optimized shutoff planning, and narrated decision intelligence.</div>
            <div class="vo-start-duration">1:58 · 5 PHASES · AI NARRATED</div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([2, 1, 2])
        with sc2:
            if st.button("▶  START EXPERIENCE", use_container_width=True, key="start_vo"):
                st.session_state.demo_playing = True
                st.session_state.demo_audio_start = time.time()
                st.session_state.demo_phase = 0
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        nav1, nav2, nav3 = st.columns([1, 6, 1])
        with nav1:
            if st.button("◀ PREV", use_container_width=True, key="demo_prev"):
                st.session_state.demo_phase = max(0, st.session_state.demo_phase - 1)
                st.rerun()
        with nav2:
            st.markdown("""
            <div style="text-align:center; font-size:0.7rem; color:var(--text-tertiary);">
                Or browse phases manually below
            </div>
            """, unsafe_allow_html=True)
        with nav3:
            if st.button("NEXT ▶", use_container_width=True, key="demo_next"):
                st.session_state.demo_phase = min(4, st.session_state.demo_phase + 1)
                st.rerun()

    else:
        # ══ VOICEOVER IS PLAYING — CLIENT-SIDE AUDIO CONTROLLER ══
        # Architecture: Single JS injection per phase.
        # Audio.ontimeupdate drives the progress bar at 60fps in JS.
        # setTimeout triggers rerun ONLY at the next phase boundary.
        # No 2-second polling. No server time sync. Zero flicker.
        elapsed_safe = max(0, demo_elapsed)

        # Calculate ms until NEXT PHASE TRANSITION ONLY (not 2s cap)
        next_phase_idx = demo_auto_phase + 1
        if next_phase_idx < len(PHASE_TRANSITIONS):
            ms_until_next = max(1000, int((PHASE_TRANSITIONS[next_phase_idx] - elapsed_safe) * 1000))
        else:
            ms_until_next = max(1000, int((AUDIO_DURATION - elapsed_safe) * 1000))

        # Progress bar update interval — JS handles this, not Streamlit reruns
        # Only rerun at phase transitions for content swap
        ms_until_rerun = ms_until_next

        phase_transitions_json = json.dumps(PHASE_TRANSITIONS)
        phase_names_json = json.dumps(demo_phase_names)

        components.html(f"""
        <script>
        (function() {{
            var doc = window.parent.document;

            // ── CLEAR ZOMBIE TIMERS from previous phase injection ──
            // Prevents race condition: old timer + new timer both clicking _advance
            if (window.parent._phaseTimer) {{
                clearTimeout(window.parent._phaseTimer);
                window.parent._phaseTimer = null;
            }}

            // ── Audio Management ──
            var audio = window.parent._earthdialAudio;
            if (!audio) {{
                audio = new Audio();
                audio.preload = 'auto';
                audio.src = './app/static/voiceover.mp3';
                window.parent._earthdialAudio = audio;
            }}

            // Sync position (only correct major drift > 5s)
            var targetTime = {elapsed_safe:.2f};
            if (audio.paused) {{
                audio.currentTime = targetTime;
                audio.play().then(function() {{
                    // Audio playing — hide any mute warning
                    var muteWarn = doc.getElementById('earthdial-mute-warn');
                    if (muteWarn) muteWarn.style.display = 'none';
                }}).catch(function(e) {{
                    console.warn('EarthDial autoplay blocked:', e);
                    // Show visible mute warning instead of silently failing
                    var existing = doc.getElementById('earthdial-mute-warn');
                    if (!existing) {{
                        var warn = doc.createElement('div');
                        warn.id = 'earthdial-mute-warn';
                        warn.style.cssText = 'position:fixed;top:80px;right:20px;z-index:99999;background:rgba(255,170,0,0.95);color:#000;padding:12px 20px;border-radius:8px;font-size:14px;font-weight:700;cursor:pointer;box-shadow:0 4px 20px rgba(0,0,0,0.5);';
                        warn.textContent = '🔇 Audio blocked — click to unmute';
                        warn.onclick = function() {{
                            audio.play();
                            warn.style.display = 'none';
                        }};
                        doc.body.appendChild(warn);
                    }}
                }});
            }} else if (Math.abs(audio.currentTime - targetTime) > 5) {{
                audio.currentTime = targetTime;
            }}

            // ── Client-Side Progress Bar (60fps via ontimeupdate) ──
            var transitions = {phase_transitions_json};
            var phaseNames = {phase_names_json};
            var totalDuration = {AUDIO_DURATION};

            audio.ontimeupdate = function() {{
                var ct = audio.currentTime;
                var pct = Math.min(100, (ct / totalDuration) * 100);
                var mins = Math.floor(ct / 60);
                var secs = Math.floor(ct % 60);

                // Update progress bar fill
                var fill = doc.querySelector('.vo-progress-bar-fill');
                if (fill) fill.style.width = pct + '%';

                // Update time display
                var timeEl = doc.querySelector('.vo-time');
                if (timeEl) timeEl.textContent = mins + ':' + (secs < 10 ? '0' : '') + secs + ' / 1:58';
            }};

            // ── Schedule rerun at NEXT PHASE TRANSITION only ──
            // Always clear + reset (never rely on null guard alone)
            window.parent._phaseTimer = setTimeout(function() {{
                window.parent._phaseTimer = null;
                var buttons = doc.querySelectorAll('button');
                for (var i = 0; i < buttons.length; i++) {{
                    if (buttons[i].textContent.trim() === '_advance') {{
                        buttons[i].click();
                        break;
                    }}
                }}
            }}, {ms_until_rerun});
        }})();
        </script>
        """, height=0)

    # ═══ DEMO PHASE CONTENT ═══

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

        try:
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
                show_risk_columns=True, show_heatmap=False,
                show_wind=True, show_fire_spread=True,
                view_state=get_cinematic_view_state(current_phase),
            )
            deck.controller = False
            st.pydeck_chart(deck, height=650, use_container_width=True)
        except Exception as e:
            st.markdown(f"""
            <div class="error-card">
                <div class="error-title">⚠️ 3D Map Rendering</div>
                <div>Map visualization is loading. WebGL initializing...</div>
            </div>
            """, unsafe_allow_html=True)

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

        try:
            from grid_optimizer import GridOptimizer
            optimizer = GridOptimizer()
            line_risks = optimizer.compute_line_risk_scores(WEATHER)

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
                table_rows += f"<tr><td>{pl['name']}</td><td>{pl['voltage_kv']}kV</td><td>{veg:.0%}</td><td>{score:.2f}</td><td>{badge}</td></tr>"

            st.markdown(f"""
            <div class="glass-card">
                <table class="data-table">
                    <thead><tr><th>Line</th><th>Voltage</th><th>Veg Risk</th><th>Score</th><th>Status</th></tr></thead>
                    <tbody>{table_rows}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)

            cf_df = st.session_state.facilities_df
            fac_rows = ""
            for _, cf in cf_df.iterrows():
                from config import FACILITY_ICONS
                icon = FACILITY_ICONS.get(cf["type"], "📍")
                fac_rows += f"<tr><td>{icon} {cf['name']}</td><td>{cf['type'].upper()}</td><td>P{cf['priority']}</td><td>{cf['feeder']}</td></tr>"

            st.markdown(f"""
            <div class="glass-card" style="margin-top:16px;">
                <div style="font-size:0.8rem; font-weight:600; color:var(--accent); margin-bottom:12px;">Critical Facilities</div>
                <table class="data-table">
                    <thead><tr><th>Facility</th><th>Type</th><th>Priority</th><th>Feeder</th></tr></thead>
                    <tbody>{fac_rows}</tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Grid analysis error: {str(e)[:100]}")

    # ── PHASE 2: AI OPTIMIZATION ──
    elif current_phase == 2:
        # ── Pre-warm prevention brief in background (idempotent, runs once) ──
        if (st.session_state.nemotron_connected
                and not st.session_state._brief_prewarm_started
                and not st.session_state.prevention_brief):
            st.session_state._brief_prewarm_started = True
            t = threading.Thread(target=_prewarm_prevention_brief, daemon=True)
            t.start()

        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🧠</div>
            <div>
                <div class="section-title">GPU-Accelerated Graph Optimization</div>
                <div class="section-subtitle">NetworkX brute-force combinatorial search · Pareto-optimal shutoff strategies</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            from grid_optimizer import GridOptimizer
            optimizer = GridOptimizer()

            if not st.session_state.shutoff_plans:
                with st.spinner("Running GPU-accelerated graph optimization..."):
                    plans = optimizer.optimize_shutoffs(weather=WEATHER, max_shutoffs=3, protect_critical=True)
                    st.session_state.shutoff_plans = plans

            if st.session_state.shutoff_plans:
                for plan in st.session_state.shutoff_plans[:5]:
                    eff_color = "green" if plan['efficiency_ratio'] > 2 else "orange" if plan['efficiency_ratio'] > 1 else "red"
                    st.markdown(f"""
                    <div class="plan-card">
                        <div class="plan-header">
                            <span class="plan-rank">PLAN #{plan['rank']}</span>
                            <span class="plan-efficiency" style="color:var(--{'accent' if eff_color == 'green' else 'status-warning' if eff_color == 'orange' else 'status-critical'});">
                                EFF: {plan['efficiency_ratio']:.2f}
                            </span>
                        </div>
                        <div class="plan-detail"><strong>Lines:</strong> {', '.join(plan['line_names'])}</div>
                        <div class="plan-detail"><strong>Risk removed:</strong> {plan['total_risk_removed']:.4f} | <strong>Confidence:</strong> {plan['confidence']:.0%}</div>
                        <div class="plan-detail"><strong>Grid connected:</strong> {'✅' if plan['grid_connected'] else '❌'} | <strong>Facilities impacted:</strong> {plan['critical_facilities_impacted']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("⚡ APPLY OPTIMAL PLAN", use_container_width=True, key="demo_apply"):
                    top = st.session_state.shutoff_plans[0]
                    st.session_state.disabled_lines = set(top['lines_disabled'])
                    st.session_state.selected_plan = top
                    from data_generator import compute_powerline_proximity, get_power_lines_df
                    from risk_engine import compute_ignition_risk
                    updated_pl = get_power_lines_df()
                    updated_pl["active"] = ~updated_pl["id"].isin(st.session_state.disabled_lines)
                    new_proximity = compute_powerline_proximity(st.session_state.terrain_df, updated_pl[updated_pl["active"]])
                    st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, new_proximity)
                    st.rerun()
        except Exception as e:
            st.warning(f"Optimization error: {str(e)[:100]}")

    # ── PHASE 3: COUNTERFACTUAL ──
    elif current_phase == 3:
        st.markdown("""
        <div class="section-header">
            <div class="section-icon">🔄</div>
            <div>
                <div class="section-title">Counterfactual Before/After Impact</div>
                <div class="section-subtitle">Visualizing the risk reduction from AI-optimized de-energization</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            if not st.session_state.selected_plan and st.session_state.shutoff_plans:
                top = st.session_state.shutoff_plans[0]
                st.session_state.disabled_lines = set(top['lines_disabled'])
                st.session_state.selected_plan = top
                from data_generator import compute_powerline_proximity, get_power_lines_df
                from risk_engine import compute_ignition_risk
                updated_pl = get_power_lines_df()
                updated_pl["active"] = ~updated_pl["id"].isin(st.session_state.disabled_lines)
                new_proximity = compute_powerline_proximity(st.session_state.terrain_df, updated_pl[updated_pl["active"]])
                st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, new_proximity)

            if st.session_state.disabled_lines:
                from data_generator import compute_powerline_proximity, get_power_lines_df
                from risk_engine import compute_ignition_risk, compute_risk_reduction

                all_pl = get_power_lines_df()
                orig_proximity = compute_powerline_proximity(st.session_state.terrain_df, all_pl)
                orig_risk = compute_ignition_risk(st.session_state.terrain_df, orig_proximity)
                reduction = compute_risk_reduction(st.session_state.terrain_df, orig_risk, st.session_state.risk_df)

                bc1, bc2 = st.columns(2)
                with bc1:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:0.6875rem; color:var(--status-critical); font-weight:600; letter-spacing:0.8px; text-transform:uppercase;">Before Intervention</div>
                        <div style="font-size:2rem; font-weight:700; font-family:var(--font-mono); color:var(--status-critical); margin:var(--space-2) 0;">{reduction['mean_risk_before']:.4f}</div>
                        <div style="font-size:0.6875rem; color:var(--text-secondary);">Mean ignition risk · {reduction['extreme_before']} extreme cells</div>
                    </div>
                    """, unsafe_allow_html=True)
                with bc2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <div style="font-size:0.6875rem; color:var(--accent); font-weight:600; letter-spacing:0.8px; text-transform:uppercase;">After Intervention</div>
                        <div style="font-size:2rem; font-weight:700; font-family:var(--font-mono); color:var(--accent); margin:var(--space-2) 0;">{reduction['mean_risk_after']:.4f}</div>
                        <div style="font-size:0.6875rem; color:var(--text-secondary);">Mean ignition risk · {reduction['extreme_after']} extreme cells</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="text-align:center; padding:var(--space-5);">
                    <div style="font-size:3rem; font-weight:800; font-family:var(--font-mono); color:var(--accent);">
                        -{reduction['reduction_pct']:.1f}%
                    </div>
                    <div style="font-size:0.8125rem; color:var(--text-secondary); margin-top:var(--space-2);">Risk Reduction Achieved</div>
                </div>
                """, unsafe_allow_html=True)

                # Updated 3D map
                try:
                    max_risk_idx = st.session_state.risk_df["ignition_risk"].idxmax()
                    max_risk_point = st.session_state.risk_df.loc[max_risk_idx]
                    from visualization import build_full_3d_map
                    deck = build_full_3d_map(
                        terrain_df=st.session_state.risk_df,
                        powerlines_df=st.session_state.powerlines_df,
                        substations_df=st.session_state.substations_df,
                        facilities_df=st.session_state.facilities_df,
                        wind_df=st.session_state.wind_df,
                        disabled_lines=st.session_state.disabled_lines,
                        ignition_point=(max_risk_point["lat"], max_risk_point["lon"]),
                        show_risk_columns=True, show_heatmap=False,
                        show_wind=True, show_fire_spread=False,
                        view_state=get_cinematic_view_state(current_phase),
                    )
                    deck.controller = False
                    st.pydeck_chart(deck, height=500, use_container_width=True)
                except Exception:
                    pass
            else:
                st.info("Navigate to Phase 2 (AI Optimization) first to generate shutoff plans.")
        except Exception as e:
            st.warning(f"Counterfactual error: {str(e)[:100]}")

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
            # Check if pre-warmed brief is ready from background thread
            if not st.session_state.prevention_brief and st.session_state._brief_prewarm_result:
                st.session_state.prevention_brief = st.session_state._brief_prewarm_result

            if not st.session_state.prevention_brief:
                if st.button("🧠 GENERATE PREVENTION BRIEF", use_container_width=True, key="demo_brief"):
                    # Hard 8s demo-safe timeout — never stall the stage
                    with st.spinner("Nemotron is analyzing risk data and generating prevention orders..."):
                        try:
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
                                timeout=8,
                            )
                            st.session_state.prevention_brief = brief
                            st.rerun()
                        except Exception as e:
                            # Fallback: use pre-generated brief instead of dead air
                            st.session_state.prevention_brief = FALLBACK_PREVENTION_BRIEF
                            st.warning(f"Live generation timed out — showing cached brief. ({html_module.escape(str(e)[:80])})")
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

    # ── DEMO COMPLETE OVERLAY ──
    if not st.session_state.demo_playing and st.session_state.demo_phase == 4 and st.session_state.demo_audio_start is None:
        components.html("""
        <script>
        (function() {
            var audio = window.parent._earthdialAudio;
            if (audio) { audio.pause(); audio.currentTime = 0; }
            // Clear phase timer
            if (window.parent._phaseTimer) {
                clearTimeout(window.parent._phaseTimer);
                window.parent._phaseTimer = null;
            }
        })();
        </script>
        """, height=0)

        st.markdown("""
        <div style="text-align:center; padding:var(--space-7) var(--space-5); margin-top:var(--space-5);">
            <div style="margin-bottom:var(--space-3);"><img src="./app/static/Earthdial.png" alt="EarthDial" style="height:36px; width:auto;"></div>
            <div style="font-size:1.25rem; font-weight:700; color:var(--accent); letter-spacing:0.5px;">Earth-2 predicts the planet.</div>
            <div style="font-size:1.25rem; font-weight:700; color:var(--text-primary); letter-spacing:0.5px; margin-top:var(--space-1);">EarthDial decides what to do about it.</div>
            <div style="font-size:0.75rem; color:var(--text-tertiary); margin-top:var(--space-4); letter-spacing:0.3px;">
                Built open. Built on NVIDIA. Built to prevent what we used to only predict.
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
                st.session_state._brief_prewarm_started = False
                st.session_state._brief_prewarm_result = None
                try:
                    from data_generator import compute_powerline_proximity, get_power_lines_df
                    from risk_engine import compute_ignition_risk
                    plines = get_power_lines_df()
                    prox = compute_powerline_proximity(st.session_state.terrain_df, plines)
                    st.session_state.risk_df = compute_ignition_risk(st.session_state.terrain_df, prox)
                except Exception:
                    pass
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-footer">
    <div class="footer-left">
        <span><img src="./app/static/Earthdial.png" alt="" style="height:16px; width:auto; vertical-align:middle; margin-right:4px;">EarthDial — AI Decision Intelligence for Planetary Systems</span>
        <span>·</span>
        <span class="footer-nvidia">Powered by NVIDIA Nemotron</span>
        <span>·</span>
        <span>Earth-2 Ecosystem</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
        <span style="color:var(--accent); font-weight:600;">#NVIDIAGTC 2026</span>
        <span>·</span>
        <span style="font-weight:600;">earthdial.ai</span>
    </div>
</div>
""", unsafe_allow_html=True)
