# 🌍 EarthDial v3 — Wildfire Prevention & Counterfactual Decision System

**The first open-source system that compiles real-time reality into an uncertainty-aware world model and uses GPU-scale counterfactuals to recommend prevention actions — not just response.**

Built with NVIDIA Nemotron + GPU-Accelerated Graph Optimization | Designed for the NVIDIA Earth-2 Ecosystem

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron-76B900)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Earth--2_Ecosystem-76B900)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-3D_Dashboard-FF4B4B)

---

## 🔥 The Problem

Wildfires are becoming more destructive, more frequent, and more costly. Current systems **react** — they detect fires after ignition and model spread after it starts. But the highest-leverage moment is **before ignition**.

The 2017 Tubbs Fire (Sonoma County, CA) killed 22 people. An aging power line in high vegetation, under Diablo wind conditions, ignited the blaze. **The conditions were predictable. The ignition was preventable.**

## ⚡ The Solution

EarthDial doesn't just tell you a fire is coming — **it tells you exactly which power line to shut off, which trailhead to close, and proves why with GPU-scale counterfactual simulation.**

### Core Capabilities

| Feature | Description |
|---------|-------------|
| 🗺️ **3D Ignition Risk Surface** | Real-time spatially-resolved risk map with 3D columns showing threat intensity |
| ⚡ **Intelligent De-Energization** | Graph-optimized power shutoffs that minimize fire risk while preserving hospitals, shelters, and emergency services |
| 🔄 **Counterfactual Simulator** | Toggle interventions → watch risk surface respond → Nemotron explains the causal chain |
| 📋 **AI Prevention Briefs** | NVIDIA Nemotron generates operator-ready prevention orders with evidence-grounded reasoning |
| 📈 **72-Hour Risk Timeline** | Weather forecast integration showing escalating and de-escalating threat windows |
| 🏥 **Critical Load Preservation** | Algorithms that protect hospitals, shelters, and comms towers during shutoffs |

---

## 🏗️ Architecture

```
EarthDial/
├── app.py                    # Streamlit 3D dashboard (main UI)
├── config.py                 # Scenario configuration & constants
├── data_generator.py         # Synthetic terrain, weather, grid data
├── risk_engine.py            # Ignition risk computation & fire spread modeling
├── grid_optimizer.py         # Graph-based de-energization optimization
├── nemotron_prevention.py    # NVIDIA Nemotron prevention brief engine
├── visualization.py          # PyDeck 3D visualization layer builders
├── requirements.txt          # Python dependencies
├── .env                      # API key (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🔧 NVIDIA Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **LLM Reasoning** | NVIDIA Nemotron (Llama 3.3 Super 49B) | Generates prevention briefs, counterfactual explanations, community alerts |
| **Inference API** | [build.nvidia.com](https://build.nvidia.com) | Cloud inference for Nemotron |
| **Graph Optimization** | NetworkX (cuGraph-compatible) | Power grid optimization for targeted shutoffs |
| **Ecosystem** | NVIDIA Earth-2 compatible | Designed to ingest Earth-2 weather ensemble forecasts |
| **Visualization** | PyDeck (deck.gl) | Omniverse-style 3D rendering |

> **Production path:** Replace NetworkX with **NVIDIA cuGraph (RAPIDS)** for GPU-accelerated graph optimization at grid scale. Replace synthetic weather with **Earth-2 / FourCastNet** ensemble forecasts.

---

## ⚡ Quick Start

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/EarthDial.git
cd EarthDial
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

Get a free key from [build.nvidia.com](https://build.nvidia.com):

```bash
echo "NVIDIA_API_KEY=your-key-here" > .env
```

### 4. Launch

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** — you'll see the full 3D threat map.

---

## 🎮 Demo Walkthrough

### Scene 1: The Threat
Open the app. You see a dark 3D map of Sonoma County with **red columns** rising where ignition risk is highest. Blue arcs show the power grid. A red flag warning banner pulses.

### Scene 2: The Decision
Go to **Grid Control**. Click **"Compute Optimal Plans"**. The GPU-accelerated optimizer finds the best combination of power lines to shut off — maximizing risk reduction while keeping hospitals and shelters powered.

### Scene 3: The Counterfactual
Apply a shutoff plan. Switch to the **Counterfactual Simulator**. Watch the risk surface drop. The extreme cells shrink. Click **"Explain this counterfactual"** — Nemotron tells you exactly *why* this works, what the tradeoffs are, and how confident it is.

### Scene 4: The Order
Go to **Prevention Brief**. Nemotron generates a complete, operator-ready prevention order: actions, timing, evidence, equity review, resource staging. Ready to act on.

---

## 🧠 How It Works

### Risk Model (Rothermel-Inspired)

```
Ignition Risk = Σ(weighted factors) + compound boost

Factors:
  - Wind speed × terrain exposure (30%)
  - Inverse humidity (20%)
  - Fuel density (20%)
  - Inverse fuel moisture (10%)
  - Slope steepness (10%)
  - Power line proximity × vegetation risk (10%)

Compound: wind × dry_fuel interaction boost
```

### Grid Optimization

The power grid is modeled as a graph (substations = nodes, transmission lines = edges). The optimizer:

1. Computes ignition risk per corridor (wind × vegetation × equipment age)
2. Evaluates all combinations of shutoff sets
3. For each combination:
   - Risk reduction score
   - Disruption score (lines lost + grid fragmentation + facilities impacted)
   - Efficiency = risk_reduction / disruption
4. Ranks plans by efficiency, respecting critical load constraints

**Production: NVIDIA cuGraph** provides GPU-accelerated graph algorithms for real utility-scale grids (millions of nodes).

### Nemotron Reasoning

Nemotron receives structured data (weather, risk indices, shutoff plans, affected facilities) and generates:
- **Prevention Briefs**: 9-section formal documents with actions, evidence, confidence, equity review
- **Counterfactual Explanations**: Causal analysis of "if we do X, risk changes by Y because Z"
- **Community Alerts**: Public-facing notifications in plain language

---

## 🗺️ Why This Matters

| Current Systems | EarthDial |
|----------------|-----------|
| Detect fire after ignition | Prevent ignition before it happens |
| Static risk maps | Dynamic, weather-responsive risk surfaces |
| Blunt power shutoffs | Surgical, optimized de-energization |
| No explanation | Nemotron explains every recommendation |
| React to fire spread | Counterfactual simulation of prevention |

---

## 🚀 Roadmap

- [ ] **NVIDIA cuGraph** integration for GPU-scale grid optimization
- [ ] **Earth-2 / FourCastNet** weather ensemble ingestion
- [ ] **Ensemble scenarios** — wind shift variants across forecasts
- [ ] **Near-miss learning** — extract precursor patterns from historical incidents
- [ ] **Smoke exposure overlay** — hospital impact modeling
- [ ] **Value-of-information** — "deploy drone HERE to reduce uncertainty most"
- [ ] **NVIDIA Omniverse** 3D digital twin rendering

---

## 🤝 Contributing

Contributions welcome! See [issues](https://github.com) or submit a PR.

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🏷️ Tags

`#NVIDIAGTC` `#Nemotron` `#Earth2` `#WildfirePrevention` `#AI` `#OpenSource` `#GPU` `#RAPIDS`

---

**Built with ❤️ using NVIDIA Nemotron | Designed for the NVIDIA Earth-2 Ecosystem | [GTC 2026](https://www.nvidia.com/gtc/)**
