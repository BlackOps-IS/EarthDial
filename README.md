# 🌍 EarthDial — AI Decision Intelligence for Planetary Systems

> **Earth-2 predicts the planet. EarthDial decides what to do about it.**

The open-source AI decision layer that turns planetary forecasts into prevention. GPU-accelerated graph optimization. NVIDIA Nemotron reasoning. Real-time counterfactual simulation. Operator-ready decision synthesis.

**🔗 Live: [earthdial.ai](https://earthdial.ai)** — Watch the cinematic demo, then take control.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron_49B-76B900)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Earth--2_Ecosystem-76B900)
![License](https://img.shields.io/badge/License-MIT-green)
![Streamlit](https://img.shields.io/badge/Streamlit-3D_Dashboard-FF4B4B)
![GPU](https://img.shields.io/badge/GPU-Accelerated-76B900)

---

## The Problem

Weather AI can now forecast disasters with extraordinary accuracy. But a forecast doesn't shut off a power line. A forecast doesn't reroute an evacuation. A forecast doesn't protect a hospital.

**The gap between prediction and prevention kills people.**

The 2017 Tubbs Fire killed 22 people in Sonoma County, CA. An aging power line in high vegetation, under Diablo wind conditions, ignited the blaze. The conditions were predictable. The ignition was preventable.

## The Solution

EarthDial closes the prediction-to-prevention gap.

| What It Does | How |
|-------------|-----|
| **Compute ignition risk surfaces** | Rothermel-inspired model across 1,600+ terrain cells in real time |
| **Optimize infrastructure interventions** | GPU-accelerated graph optimization finds surgical shutoffs |
| **Prove every decision** | Counterfactual simulation: "If we do X, risk drops by Y — here's why" |
| **Synthesize prevention orders** | NVIDIA Nemotron generates operator-ready documents |
| **Preserve critical services** | Algorithms protect hospitals, shelters, and comms during shutoffs |

---

## Live Demo

Visit **[earthdial.ai](https://earthdial.ai)** — the demo begins automatically:

1. **The Threat** — 3D risk surface materializes over Sonoma County under Red Flag conditions
2. **Grid Intelligence** — Power line risk scoring across 8 transmission lines and 8 critical facilities
3. **AI Optimization** — GPU graph optimizer finds the surgical intervention plan
4. **Counterfactual Proof** — Before/after comparison shows risk reduction with causal explanation
5. **Prevention Order** — Nemotron synthesizes a formal, operator-ready prevention brief

After the 2-minute demo, click **TAKE CONTROL** to explore the full system interactively.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   EARTHDIAL PLATFORM                     │
│                                                          │
│  Forecast     →  Risk         →  Graph        →  Decision│
│  Ingestion       Computation     Optimization    Synthesis│
│  (Earth-2        (Rothermel     (NetworkX →     (Nemotron │
│   compatible)     model)         cuGraph)        49B)     │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │        3D Visualization (PyDeck / deck.gl)          │  │
│  │  Risk Columns · Grid Arcs · Fire Spread · Wind      │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

```
EarthDial/
├── app.py                    # Streamlit 3D dashboard
├── config.py                 # Scenario configuration
├── data_generator.py         # Terrain, weather, grid data
├── risk_engine.py            # Ignition risk & fire spread
├── grid_optimizer.py         # Graph-based optimization
├── nemotron_prevention.py    # NVIDIA Nemotron integration
├── visualization.py          # PyDeck 3D layer builders
├── docker/
│   ├── Dockerfile            # Production container
│   └── docker-compose.yml    # Full stack orchestration
├── static/
│   └── voiceover.mp3         # Demo narration
└── requirements.txt
```

---

## NVIDIA Technology Stack

| Component | Technology | Role |
|-----------|-----------|------|
| **Reasoning** | NVIDIA Nemotron (Llama-3.3 Super 49B) | Prevention briefs, counterfactual explanations, community alerts |
| **Inference** | [build.nvidia.com](https://build.nvidia.com) API | Cloud inference → Triton (production path) |
| **Graph Optimization** | NetworkX → NVIDIA cuGraph | GPU-accelerated combinatorial infrastructure optimization |
| **Ecosystem** | Earth-2 compatible | Designed to ingest FourCastNet / CorrDiff ensemble forecasts |
| **Visualization** | PyDeck (deck.gl) | WebGL 3D rendering with cinematic camera system |

> **Production path:** cuGraph (RAPIDS) for GPU-parallel graph optimization at utility scale. Triton Inference Server for local Nemotron. Earth-2 ensemble ingestion for multi-forecast decision fusion.

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/BlackOps-IS/EarthDial.git
cd EarthDial
echo "NVIDIA_API_KEY=your-key-here" > .env
cd docker && docker compose up
```

Open **http://localhost:8501**

### Option 2: Local Install

```bash
git clone https://github.com/BlackOps-IS/EarthDial.git
cd EarthDial
pip install -r requirements.txt
echo "NVIDIA_API_KEY=your-key-here" > .env
python -m streamlit run app.py
```

Get a free NVIDIA API key from [build.nvidia.com](https://build.nvidia.com).

---

## How It Works

### Risk Model (Rothermel-Inspired)

```
Ignition Risk = Σ(weighted factors) + compound boost

Factors:
  Wind speed × terrain exposure     (30%)
  Inverse humidity                  (20%)
  Fuel density                      (20%)
  Inverse fuel moisture             (10%)
  Slope steepness                   (10%)
  Power line proximity × vegetation (10%)

Compound: wind × dry_fuel interaction boost
```

### Grid Optimization

The power grid is modeled as a graph. The optimizer evaluates all combinations of shutoff sets:

1. Compute ignition risk per corridor (wind × vegetation × age × voltage)
2. Evaluate each combination: risk reduction vs. disruption
3. Enforce constraints: grid connectivity, critical facility preservation
4. Rank by efficiency = risk_reduction / disruption_score

### Nemotron Reasoning

Nemotron receives structured data and produces:
- **Prevention Briefs:** 9-section formal documents with actions, evidence, confidence
- **Counterfactual Explanations:** Causal analysis of interventions
- **Community Alerts:** Public-facing notifications in plain language

This is reasoning, not chatbot generation. Structured input → structured output → actionable decision.

---

## Ecosystem Positioning

| | Earth-2 | EarthDial |
|--|---------|-----------|
| **Function** | Forecasting engine | Decision reasoning engine |
| **Input** | Atmospheric observations | Forecast outputs + infrastructure data |
| **Output** | Weather predictions | Prevention orders + intervention plans |
| **Relationship** | Upstream (predicts) | Downstream (decides) |

EarthDial doesn't compete with Earth-2. It completes the stack.

**Prediction without decision is just data. Decision without prediction is just guessing.**

---

## Benchmarks

| Operation | Latency | Scale |
|-----------|---------|-------|
| Risk surface computation | <50ms | 1,600 cells |
| Graph optimization (8 lines, 3 shutoffs) | <200ms | 56 combinations |
| Nemotron prevention brief | <3s | 49B parameters |
| Counterfactual recomputation | <100ms | Full grid |
| 3D map rendering | 60fps | 10 composited layers |

---

## Roadmap

- [ ] **NVIDIA cuGraph** — GPU-parallel graph optimization for utility-scale grids (100K+ nodes)
- [ ] **Earth-2 ingestion** — Direct FourCastNet/CorrDiff ensemble forecast consumption
- [ ] **Ensemble decision fusion** — Run N forecasts → N decision pipelines → consensus optimal plan
- [ ] **Triton Inference Server** — Local Nemotron deployment on DGX
- [ ] **Multi-hazard fusion** — Joint wildfire + flood + seismic optimization
- [ ] **Autonomous alerts** — Agentic pipeline with continuous monitoring and auto-generated orders
- [ ] **NVIDIA Omniverse** — Digital twin visualization with photorealistic terrain
- [ ] **NIM microservices** — Containerized reasoning components

---

## Contributing

Contributions welcome! Open an issue or submit a PR.

- Fork the repository
- Create a feature branch
- Make your changes
- Open a pull request

---

## License

MIT License — see [LICENSE](LICENSE).

---

## Links

- **Live Demo:** [earthdial.ai](https://earthdial.ai)
- **GitHub:** [github.com/BlackOps-IS/EarthDial](https://github.com/BlackOps-IS/EarthDial)
- **NVIDIA Nemotron:** [build.nvidia.com](https://build.nvidia.com)
- **NVIDIA Earth-2:** [nvidia.com/earth-2](https://www.nvidia.com/en-us/high-performance-computing/earth-2/)

---

**Built open. Built on NVIDIA. Built to prevent what we used to only predict.**

`#NVIDIAGTC` `#Nemotron` `#Earth2` `#GPU` `#OpenSource` `#DecisionIntelligence`
