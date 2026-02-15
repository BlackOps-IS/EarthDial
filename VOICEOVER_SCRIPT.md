# EarthDial v3 — Voiceover Sync Guide

## AUDIO: final.mp3 — Runtime: 1:42 (102 seconds)

### ElevenLabs Settings Used
- **Speed:** ~40% (slightly slower than center)
- **Stability:** ~55% (center)
- **Similarity:** ~75% (high)
- **Style Exaggeration:** ~15% (minimal)
- **Speaker boost:** ON

---

## EXACT SYNC — WHEN TO CLICK EACH PHASE

You have ONE job during recording: click NEXT at the right moments.
Play final.mp3 and follow this guide EXACTLY.

| TIME  | ACTION                        | WHAT THE VOICE IS SAYING                                    |
|-------|-------------------------------|-------------------------------------------------------------|
| 0:00  | START RECORDING. Phase 1 visible (3D Threat Map). | "What if we could prevent wildfires, before they start?" |
| 0:05  | Stay on Phase 1. Slowly pan the 3D map with mouse. | "This is EarthDial version three..." |
| 0:15  | Still Phase 1. Let the risk columns be visible. | "Sixteen hundred terrain cells, each analyzed in real time..." |
| 0:22  | Still Phase 1. Don't click yet. | "...extreme ignition risk zones, driven by forty five mile per hour..." |
| 0:33  | Voice says "Red Flag Warning." Pause 1 second. | (Phase 1 wraps up) |
| **0:35** | **👆 CLICK NEXT → Phase 2 (Grid Analysis)** | "The blue arcs are live power transmission lines..." |
| 0:40  | Stay on Phase 2. Let the table load. | "...every one of them is a potential ignition source." |
| 0:43  | Still Phase 2. Hover over high-risk lines in table. | "Our risk engine scores each line using vegetation encroachment..." |
| 0:50  | Voice says "...computed simultaneously." | (Phase 2 wraps up) |
| **0:52** | **👆 CLICK NEXT → Phase 3 (AI Optimization)** | "Now watch. Our GPU graph optimizer..." |
| 0:55  | Stay on Phase 3. Plans computing/loading. | "...evaluates every possible combination of power shutoffs..." |
| 1:00  | Still Phase 3. Scroll through plan cards. | "...finds the surgical cut that maximizes risk reduction..." |
| 1:05  | Voice says "Intelligent de-energization." | (Phase 3 wraps up) |
| **1:07** | **👆 CLICK NEXT → Phase 4 (Counterfactual)** | "Here's the counterfactual. On the left, before intervention..." |
| 1:10  | Stay on Phase 4. Before/after cards visible. | "...On the right, after. Risk drops. Extreme cells eliminated." |
| 1:15  | Still Phase 4. Big green reduction number visible. | "...this is a decision engine. Every action is explainable..." |
| 1:20  | Voice says "...every tradeoff is quantified." | (Phase 4 wraps up) |
| **1:22** | **👆 CLICK NEXT → Phase 5 (Prevention Brief)** | "Finally, Nemotron. NVIDIA's Llama three point three..." |
| 1:28  | Stay on Phase 5. Brief box visible or generating. | "...generates a formal, operator ready prevention brief." |
| 1:33  | Still Phase 5. | "Not a suggestion. A prevention order..." |
| 1:37  | Voice says "...equity aware." | (Closing line coming) |
| 1:38  | Click "🎮 TAKE CONTROL" button. | "This is EarthDial. Prevention, not response. Built with NVIDIA." |
| 1:42  | **STOP RECORDING.** | (Audio ends) |

---

## SUMMARY — 4 CLICKS TOTAL

| Click # | Time    | Button      | Phase You're Moving TO      |
|---------|---------|-------------|------------------------------|
| 1       | **0:35** | NEXT ▶      | Phase 2: Grid Analysis       |
| 2       | **0:52** | NEXT ▶      | Phase 3: AI Optimization     |
| 3       | **1:07** | NEXT ▶      | Phase 4: Counterfactual      |
| 4       | **1:22** | NEXT ▶      | Phase 5: Prevention Brief    |
| BONUS   | **1:38** | TAKE CONTROL| Interactive mode (wow moment) |

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
- [ ] Stop OBS at 1:42

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
📂 Open source: [YOUR_GITHUB_URL]
🎥 Watch the 90-second demo ⬆️

This is prevention, not response. Built with NVIDIA.

@Carter Abdallah @Nader Khalil #NVIDIAGTC #EarthDial #WildfirePrevention #Nemotron
```
