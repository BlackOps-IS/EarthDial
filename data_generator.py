"""
EarthDial v3 — Synthetic Data Generator
Generates realistic terrain, fuel, and weather data for Sonoma County scenario.
"""

import numpy as np
import pandas as pd
from config import (
    CENTER_LAT, CENTER_LON, GRID_ROWS, GRID_COLS,
    GRID_STEP_LAT, GRID_STEP_LON, WEATHER,
    SUBSTATIONS, POWER_LINES, CRITICAL_FACILITIES,
)


def generate_terrain_grid() -> pd.DataFrame:
    """
    Generate a grid of terrain points with elevation, slope, fuel properties.
    Models the hilly terrain of Sonoma County wine country.
    """
    np.random.seed(42)

    rows = []
    for i in range(GRID_ROWS):
        for j in range(GRID_COLS):
            lat = CENTER_LAT - (GRID_ROWS / 2 - i) * GRID_STEP_LAT
            lon = CENTER_LON - (GRID_COLS / 2 - j) * GRID_STEP_LON

            # Elevation: rolling hills with ridges (Sonoma Mountains)
            base_elevation = 150  # meters
            ridge_1 = 300 * np.exp(-((i - 10)**2 + (j - 30)**2) / 80)
            ridge_2 = 250 * np.exp(-((i - 30)**2 + (j - 15)**2) / 60)
            valley = -100 * np.exp(-((i - 20)**2 + (j - 20)**2) / 120)
            noise = np.random.normal(0, 20)
            elevation = max(30, base_elevation + ridge_1 + ridge_2 + valley + noise)

            # Slope derived from elevation gradient (simplified)
            slope = min(45, max(0, (ridge_1 + ridge_2) / 15 + np.random.normal(5, 3)))

            # Fuel density: higher on slopes, lower in valleys/developed areas
            developed = np.exp(-((i - 20)**2 + (j - 20)**2) / 200)
            fuel_density = np.clip(0.3 + 0.5 * (elevation / 500) - 0.4 * developed + np.random.normal(0, 0.1), 0.05, 1.0)

            # Fuel moisture: very low during red flag (Diablo winds)
            fuel_moisture = np.clip(0.06 + 0.04 * np.random.random() + 0.1 * (1 - fuel_density), 0.03, 0.25)

            # Wind exposure: higher on ridges
            wind_exposure = np.clip(0.3 + 0.7 * (elevation / 500) + np.random.normal(0, 0.05), 0.1, 1.0)

            rows.append({
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "elevation": round(elevation, 1),
                "slope": round(slope, 1),
                "fuel_density": round(fuel_density, 3),
                "fuel_moisture": round(fuel_moisture, 3),
                "wind_exposure": round(wind_exposure, 3),
                "grid_i": i,
                "grid_j": j,
            })

    return pd.DataFrame(rows)


def generate_weather_timeline(hours: int = 72) -> pd.DataFrame:
    """
    Generate hourly weather forecast showing escalating red flag conditions.
    Models Diablo wind event building over 72 hours.
    """
    np.random.seed(123)
    rows = []

    for h in range(hours):
        # Wind ramps up, peaks at hours 18-36, then slowly drops
        wind_phase = np.sin(np.pi * h / 36) if h < 36 else max(0.3, np.sin(np.pi * (72 - h) / 72))
        wind_speed = WEATHER["wind_speed_mph"] * (0.4 + 0.6 * wind_phase) + np.random.normal(0, 3)
        wind_gust = wind_speed * (1.3 + 0.2 * np.random.random())

        # Humidity drops as winds increase
        humidity = max(3, WEATHER["humidity_pct"] * (1.5 - 0.5 * wind_phase) + np.random.normal(0, 2))

        # Temperature peaks midday
        hour_of_day = h % 24
        temp_cycle = 10 * np.sin(np.pi * (hour_of_day - 6) / 12) if 6 <= hour_of_day <= 18 else -5
        temperature = WEATHER["temperature_f"] + temp_cycle + np.random.normal(0, 2)

        # Lightning probability increases with instability
        lightning = np.clip(WEATHER["lightning_probability"] * wind_phase + np.random.normal(0, 0.02), 0, 0.3)

        rows.append({
            "hour": h,
            "wind_speed_mph": round(max(5, wind_speed), 1),
            "wind_gust_mph": round(max(8, wind_gust), 1),
            "wind_direction_deg": round(WEATHER["wind_direction_deg"] + np.random.normal(0, 8), 1),
            "temperature_f": round(temperature, 1),
            "humidity_pct": round(max(2, humidity), 1),
            "lightning_probability": round(max(0, lightning), 3),
        })

    return pd.DataFrame(rows)


def get_substation_df() -> pd.DataFrame:
    """Get substations as a DataFrame."""
    return pd.DataFrame(SUBSTATIONS)


def get_power_lines_df() -> pd.DataFrame:
    """Get power lines with resolved coordinates from substations."""
    sub_lookup = {s["id"]: s for s in SUBSTATIONS}
    rows = []
    for pl in POWER_LINES:
        from_sub = sub_lookup[pl["from"]]
        to_sub = sub_lookup[pl["to"]]

        # Add intermediate points along the line with slight offsets for realism
        mid_lat = (from_sub["lat"] + to_sub["lat"]) / 2 + np.random.uniform(-0.005, 0.005)
        mid_lon = (from_sub["lon"] + to_sub["lon"]) / 2 + np.random.uniform(-0.005, 0.005)

        rows.append({
            **pl,
            "from_lat": from_sub["lat"],
            "from_lon": from_sub["lon"],
            "to_lat": to_sub["lat"],
            "to_lon": to_sub["lon"],
            "mid_lat": mid_lat,
            "mid_lon": mid_lon,
            "from_name": from_sub["name"],
            "to_name": to_sub["name"],
            "active": True,
        })
    return pd.DataFrame(rows)


def get_critical_facilities_df() -> pd.DataFrame:
    """Get critical facilities as a DataFrame."""
    return pd.DataFrame(CRITICAL_FACILITIES)


def compute_powerline_proximity(terrain_df: pd.DataFrame, powerlines_df: pd.DataFrame) -> np.ndarray:
    """
    Compute proximity of each terrain point to nearest power line.
    Returns array of proximity scores (0-1, higher = closer).
    """
    proximities = np.zeros(len(terrain_df))

    for _, pl in powerlines_df.iterrows():
        if not pl.get("active", True):
            continue

        # Simple point-to-segment distance proxy
        for idx, point in terrain_df.iterrows():
            dx = point["lon"] - (pl["from_lon"] + pl["to_lon"]) / 2
            dy = point["lat"] - (pl["from_lat"] + pl["to_lat"]) / 2
            dist = np.sqrt(dx**2 + dy**2)
            proximity = np.exp(-dist / 0.02) * pl["vegetation_risk"]
            proximities[idx] = max(proximities[idx], proximity)

    return np.clip(proximities, 0, 1)


def generate_wind_field(terrain_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate wind vectors at sampled points for visualization.
    Wind is channeled by terrain (accelerates over ridges, decelerates in valleys).
    """
    np.random.seed(99)
    base_dir = np.radians(WEATHER["wind_direction_deg"])
    base_speed = WEATHER["wind_speed_mph"]

    # Sample every 4th point for wind arrows
    sampled = terrain_df.iloc[::4].copy()

    wind_rows = []
    for _, pt in sampled.iterrows():
        # Wind accelerates with exposure
        local_speed = base_speed * pt["wind_exposure"] + np.random.normal(0, 3)
        # Small direction perturbation from terrain
        local_dir = base_dir + np.random.normal(0, 0.15)

        # Arrow endpoint (for visualization)
        arrow_len = 0.003 * (local_speed / base_speed)
        end_lat = pt["lat"] + arrow_len * np.cos(local_dir)
        end_lon = pt["lon"] + arrow_len * np.sin(local_dir)

        wind_rows.append({
            "lat": pt["lat"],
            "lon": pt["lon"],
            "end_lat": round(end_lat, 6),
            "end_lon": round(end_lon, 6),
            "speed": round(max(5, local_speed), 1),
            "direction": round(np.degrees(local_dir), 1),
            "elevation": pt["elevation"],
        })

    return pd.DataFrame(wind_rows)
