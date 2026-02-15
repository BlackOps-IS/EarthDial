"""
EarthDial v3 — Nemotron Prevention Engine
Uses NVIDIA Nemotron to generate operator-ready prevention briefs,
counterfactual explanations, and evidence-grounded action plans.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NEMOTRON_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1"


def _get_api_key():
    """Get API key from Streamlit secrets (cloud) or .env (local)."""
    try:
        import streamlit as st
        return st.secrets.get("NVIDIA_API_KEY", os.getenv("NVIDIA_API_KEY"))
    except Exception:
        return os.getenv("NVIDIA_API_KEY")


class NemotronPreventionEngine:
    """Generates AI-powered prevention briefs using NVIDIA Nemotron."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or _get_api_key()
        if not self.api_key:
            raise ValueError("NVIDIA API key required. Set NVIDIA_API_KEY in .env")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_nemotron(self, system_prompt: str, user_content: str, max_tokens: int = 3000) -> str:
        """Send a request to Nemotron."""
        payload = {
            "model": NEMOTRON_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.25,
            "max_tokens": max_tokens,
            "top_p": 0.9,
        }

        response = requests.post(
            f"{NVIDIA_API_BASE}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def generate_prevention_brief(
        self,
        weather: dict,
        risk_stats: dict,
        shutoff_plan: dict = None,
        affected_facilities: list = None,
        risk_reduction: dict = None,
    ) -> str:
        """
        Generate a comprehensive, operator-ready prevention brief.

        This is the core Nemotron output — a structured document that
        an operator could act on immediately.
        """
        system_prompt = """You are EarthDial, an AI-powered wildfire prevention system built on NVIDIA technology. 
You generate formal, operator-ready Prevention Briefs for utility operators and emergency managers.

Your briefs must be:
- Actionable: specific actions with clear owners
- Evidence-grounded: cite the sensor/weather data that drives each recommendation
- Risk-quantified: include probability estimates and confidence intervals
- Equity-aware: flag if actions disproportionately impact specific communities
- Time-bound: include recommended action windows

Format as a professional prevention order document."""

        user_content = f"""Generate a Prevention Brief for the following situation:

## WEATHER CONDITIONS
- Wind: {weather.get('wind_speed_mph', 'N/A')} mph sustained, gusts to {weather.get('wind_gust_mph', 'N/A')} mph
- Wind Direction: {weather.get('wind_direction_deg', 'N/A')}° (Diablo wind pattern)
- Temperature: {weather.get('temperature_f', 'N/A')}°F
- Humidity: {weather.get('humidity_pct', 'N/A')}%
- Lightning Probability: {weather.get('lightning_probability', 'N/A')}
- Red Flag Warning: {'ACTIVE' if weather.get('red_flag') else 'No'}

## RISK ASSESSMENT
- Mean ignition risk index: {risk_stats.get('mean_risk', 'N/A')}
- Extreme risk cells: {risk_stats.get('extreme_cells', 'N/A')}
- High risk cells: {risk_stats.get('high_cells', 'N/A')}
- Total area monitored: {risk_stats.get('total_cells', 'N/A')} grid cells

## RECOMMENDED SHUTOFF PLAN
{json.dumps(shutoff_plan, indent=2) if shutoff_plan else 'No plan selected yet.'}

## AFFECTED CRITICAL FACILITIES
{json.dumps(affected_facilities, indent=2) if affected_facilities else 'None affected.'}

## RISK REDUCTION (if shutoff applied)
{json.dumps(risk_reduction, indent=2) if risk_reduction else 'Not yet computed.'}

Generate a complete Prevention Brief with:
1. SITUATION SUMMARY (2-3 sentences)
2. THREAT ASSESSMENT (rated Critical/High/Moderate/Low with justification)
3. RECOMMENDED ACTIONS (numbered, prioritized, with time windows)
4. DE-ENERGIZATION ORDERS (specific lines, with justification and impact)
5. CRITICAL LOAD PROTECTION PLAN (how to maintain power to hospitals, shelters, etc.)
6. COMMUNITY NOTIFICATIONS (what to communicate to the public)
7. RESOURCE STAGING (where to pre-position suppression assets)
8. CONFIDENCE ASSESSMENT (what data gaps exist, what would change the plan)
9. EQUITY REVIEW (which communities are most impacted, mitigation steps)"""

        return self._call_nemotron(system_prompt, user_content, max_tokens=4000)

    def generate_counterfactual_explanation(
        self,
        action_taken: str,
        risk_before: dict,
        risk_after: dict,
        affected_facilities: list = None,
    ) -> str:
        """
        Generate an explanation of a counterfactual scenario.
        "If we do X, here's what changes and why."
        """
        system_prompt = """You are EarthDial, an AI wildfire prevention system. 
You explain counterfactual scenarios in clear, precise language.
Compare the before/after states and explain the causal chain.
Be concise but thorough. Use bullet points."""

        user_content = f"""Explain this counterfactual intervention:

ACTION: {action_taken}

RISK BEFORE:
- Mean ignition risk: {risk_before.get('mean_risk_before', 'N/A')}
- Extreme risk cells: {risk_before.get('extreme_cells_before', 'N/A')}
- High risk cells: {risk_before.get('high_risk_cells_before', 'N/A')}

RISK AFTER:
- Mean ignition risk: {risk_after.get('mean_risk_after', 'N/A')}
- Extreme risk cells: {risk_after.get('extreme_cells_after', 'N/A')}
- High risk cells: {risk_after.get('high_risk_cells_after', 'N/A')}
- Risk reduction: {risk_after.get('reduction_pct', 'N/A')}%

AFFECTED FACILITIES: {json.dumps(affected_facilities) if affected_facilities else 'None'}

Explain:
1. WHY this action reduces risk (causal mechanism)
2. WHAT the tradeoffs are (disruption vs. safety)
3. HOW CONFIDENT we are (and what uncertainty remains)
4. WHAT ELSE should be done alongside this action"""

        return self._call_nemotron(system_prompt, user_content, max_tokens=2000)

    def generate_community_alert(
        self,
        weather: dict,
        risk_level: str,
        actions_taken: list[str],
        zones_affected: list[str] = None,
    ) -> str:
        """Generate a public-facing community alert."""
        system_prompt = """You are EarthDial generating a community safety alert.
Write in clear, non-technical language accessible to all residents.
Include specific actions residents should take.
Be calm but urgent. Do not cause panic."""

        user_content = f"""Generate a community wildfire prevention alert:

CONDITIONS: Wind {weather['wind_speed_mph']}mph, Humidity {weather['humidity_pct']}%, Temp {weather['temperature_f']}°F
RISK LEVEL: {risk_level}
ACTIONS BEING TAKEN: {', '.join(actions_taken)}
ZONES AFFECTED: {', '.join(zones_affected) if zones_affected else 'County-wide'}

Include: what's happening, what we're doing, what residents should do, emergency contacts."""

        return self._call_nemotron(system_prompt, user_content, max_tokens=1500)

    def test_connection(self) -> bool:
        """Test Nemotron API connection."""
        try:
            result = self._call_nemotron(
                "You are a helpful assistant.",
                "Respond with exactly: EarthDial connected.",
                max_tokens=20,
            )
            return "connected" in result.lower() or "earthdial" in result.lower()
        except Exception:
            return False
