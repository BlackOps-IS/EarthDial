# EarthDial v3 — Voiceover Sync Guide

## AUDIO: final.mp3 — Runtime: 1:58 (118.94 seconds)

### ElevenLabs Settings Used
- **Voice:** Guav (Indian-style)
- **Speed:** ~40% (slightly slower than center)
- **Stability:** ~55% (center)
- **Similarity:** ~75% (high)
- **Style Exaggeration:** ~15% (minimal)
- **Speaker boost:** ON

---

## EXACT SYNC — WHEN TO CLICK EACH PHASE

You have ONE job during recording: click NEXT at the right moments.
Play final.mp3 and follow this guide EXACTLY.

**Total: 248 words across 13 segments at ~2.31 words/sec**

### PHASE 1 — THREAT MAP (0:00 → 0:37)

| TIME    | ACTION                                             | WHAT THE VOICE IS SAYING                                                 |
|---------|-----------------------------------------------------|--------------------------------------------------------------------------|
| 0:00    | START RECORDING. Phase 1 visible (3D Threat Map).  | "What if we could prevent wildfires, before they start?"                 |
| 0:06    | Stay on Phase 1. Slowly pan the 3D map with mouse. | "This is EarthDial version three…"                                       |
| 0:13    | Still Phase 1. Let the risk columns be visible.     | "Sixteen hundred terrain cells, each analyzed in real time…"             |
| 0:22    | Still Phase 1. Don't click yet.                     | "…extreme ignition risk zones, driven by forty-five mile per hour…"      |
| 0:34    | Voice says "Red Flag Warning."                      | "…Red Flag Warning conditions."                                          |
| 0:37    | Phase 1 audio complete. 1.5s gap before next phase. | *(breathe — get ready to click)*                                         |

### PHASE 2 — GRID ANALYSIS (click NEXT at 0:38)

| TIME    | ACTION                                              | WHAT THE VOICE IS SAYING                                                 |
|---------|------------------------------------------------------|--------------------------------------------------------------------------|
| **0:38** | **👆 CLICK NEXT → Phase 2 (Grid Analysis)**        | *(click during the gap)*                                                 |
| 0:39    | Stay on Phase 2. Let the table load.                | "The blue arcs are live power transmission lines…"                       |
| 0:44    | Still Phase 2. Hover over high-risk lines in table. | "Our risk engine scores each line using vegetation encroachment…"        |
| 0:55    | Voice says "…computed simultaneously."               | "…computed simultaneously."                                              |

### PHASE 3 — AI OPTIMIZATION (click NEXT at 0:56)

| TIME    | ACTION                                               | WHAT THE VOICE IS SAYING                                                 |
|---------|-------------------------------------------------------|--------------------------------------------------------------------------|
| **0:56** | **👆 CLICK NEXT → Phase 3 (AI Optimization)**       | *(click during the gap)*                                                 |
| 0:57    | Stay on Phase 3. Plans computing/loading.            | "Now watch. Our GPU graph optimizer…"                                    |
| 1:03    | Still Phase 3. Scroll through plan cards.            | "…finds the surgical cut that maximizes risk reduction…"                 |
| 1:13    | Voice says "Intelligent de-energization."             | "Intelligent de-energization."                                           |

### PHASE 4 — COUNTERFACTUAL (click NEXT at 1:14)

| TIME    | ACTION                                               | WHAT THE VOICE IS SAYING                                                 |
|---------|-------------------------------------------------------|--------------------------------------------------------------------------|
| **1:14** | **👆 CLICK NEXT → Phase 4 (Counterfactual)**        | *(click during the gap)*                                                 |
| 1:15    | Stay on Phase 4. Before/after cards visible.         | "Here's the counterfactual. On the left, before intervention…"           |
| 1:21    | Still Phase 4. Big green reduction number visible.   | "…this is a decision engine. Every action is explainable…"               |
| 1:31    | Voice says "…every tradeoff is quantified."           | "…every tradeoff is quantified."                                         |

### PHASE 5 — PREVENTION BRIEF (click NEXT at 1:32)

| TIME    | ACTION                                               | WHAT THE VOICE IS SAYING                                                 |
|---------|-------------------------------------------------------|--------------------------------------------------------------------------|
| **1:32** | **👆 CLICK NEXT → Phase 5 (Prevention Brief)**      | *(click during the gap)*                                                 |
| 1:33    | Stay on Phase 5. Brief box visible or generating.    | "Finally, Nemotron. NVIDIA's Llama three point three…"                   |
| 1:40    | Still Phase 5.                                        | "Not a suggestion. A prevention order…"                                  |
| 1:50    | Voice building to finale.                             | "…equity aware, and NVIDIA accelerated."                                 |
| 1:52    | Click "🎮 TAKE CONTROL" button.                      | "This is EarthDial. Prevention, not response. Built with NVIDIA."        |
| 1:58    | **STOP RECORDING.**                                  | *(Audio ends at exactly 1:58.94)*                                        |

---

## SUMMARY — 4 CLICKS + 1 BONUS

| Click # | Time     | Button       | Phase You're Moving TO       |
|---------|----------|--------------|-------------------------------|
| 1       | **0:38** | NEXT ▶       | Phase 2: Grid Analysis        |
| 2       | **0:56** | NEXT ▶       | Phase 3: AI Optimization      |
| 3       | **1:14** | NEXT ▶       | Phase 4: Counterfactual       |
| 4       | **1:32** | NEXT ▶       | Phase 5: Prevention Brief     |
| BONUS   | **1:52** | TAKE CONTROL | Interactive mode (wow moment) |

---

## PHASE DURATIONS (for reference)

| Phase | Name              | Duration | Words |
|-------|-------------------|----------|-------|
| 1     | Threat Map        | 37.6s    | 78    |
| 2     | Grid Analysis     | 16.7s    | 39    |
| 3     | AI Optimization   | 16.3s    | 38    |
| 4     | Counterfactual    | 16.7s    | 39    |
| 5     | Prevention Brief  | 25.6s    | 54    |

---

## RECORDING SETUP (OBS Studio)

1. Open OBS Studio (free: obsproject.com)
2. Add **Window Capture** → select browser with earthdial.ai
3. Add **Audio Output** → your final.mp3 playing through media player
4. Resolution: **1920×1080** or **4K** if possible
5. FPS: **60**
6. Browser should be **fullscreen** (F11)
7. Have the 3D map visible and slowly rotating before you start

### Pre-Recording Checklist:
- [ ] App running at earthdial.ai (or localhost)
- [ ] Demo Mode active (Phase 1 visible)
- [ ] final.mp3 loaded in media player, paused at 0:00
- [ ] OBS recording started
- [ ] Press PLAY on final.mp3
- [ ] Follow the click guide above
- [ ] Stop OBS at 1:58

---

## LINKEDIN POST (FINAL)

```
I built a wildfire prevention system powered by NVIDIA Nemotron.

EarthDial v3 doesn't just detect fires — it prevents them.

Using GPU-accelerated graph optimization and NVIDIA's Llama-3.3-Nemotron-Super-49B, it:

→ Computes real-time ignition risk across 1,600 terrain cells
→ Finds surgical power shutoff plans that minimize fire risk while keeping hospitals online
→ Generates operator-ready prevention briefs — not just alerts
→ Runs counterfactual simulations: "What if we de-energize this line?"

🎮 Live demo: https://earthdial.ai
📂 Open source: https://github.com/BlackOps-IS/EarthDial
🎥 Watch the 2-minute demo ⬆️

This is prevention, not response. Built with NVIDIA.

@Carter Abdallah @Nader Khalil #NVIDIAGTC #EarthDial #WildfirePrevention #Nemotron
```
