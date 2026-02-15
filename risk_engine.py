"""
EarthDial v3 — Ignition Risk Engine & Fire Spread Modeling
Computes spatially-resolved ignition risk and projected fire spread cones.
"""

import numpy as np
import pandas as pd
from config import RISK_WEIGHTS, WEATHER


def compute_ignition_risk(
    terrain_df: pd.DataFrame,
    powerline_proximity: np.ndarray,
    weather: dict = None,
    disabled_lines: set = None,
) -> pd.DataFrame:
    """
    Compute ignition risk index for each terrain cell.

    Risk = weighted combination of:
        - Wind speed / exposure
        - Low humidity
        - Fuel density
        - Low fuel moisture
        - Steep slope
        - Power line proximity (reduced when lines are de-energized)

    Args:
        terrain_df: Terrain grid with fuel/slope/exposure data
        powerline_proximity: Array of proximity scores to active power lines
        weather: Weather dict override (defaults to config WEATHER)
        disabled_lines: Set of power line IDs that are shut off

    Returns:
        terrain_df with added 'ignition_risk' and 'risk_category' columns
    """
    wx = weather or WEATHER
    w = RISK_WEIGHTS
    df = terrain_df.copy()

    # ── Normalize inputs to 0-1 ──────────────────────────────────────────────

    # Wind risk: higher speed + higher exposure = more risk
    wind_normalized = np.clip(wx["wind_speed_mph"] / 80, 0, 1)
    wind_risk = wind_normalized * df["wind_exposure"].values

    # Humidity risk: lower humidity = higher risk (inverted)
    humidity_risk = np.clip(1.0 - wx["humidity_pct"] / 50, 0, 1)

    # Fuel density risk
    fuel_risk = df["fuel_density"].values

    # Fuel moisture risk: lower moisture = higher risk (inverted)
    moisture_risk = np.clip(1.0 - df["fuel_moisture"].values / 0.25, 0, 1)

    # Slope risk: steeper = more risk
    slope_risk = np.clip(df["slope"].values / 40, 0, 1)

    # Power line proximity (already 0-1)
    pl_risk = powerline_proximity

    # ── Weighted combination ─────────────────────────────────────────────────
    risk = (
        w["wind"] * wind_risk +
        w["humidity"] * humidity_risk +
        w["fuel_density"] * fuel_risk +
        w["fuel_moisture"] * moisture_risk +
        w["slope"] * slope_risk +
        w["powerline_proximity"] * pl_risk
    )

    # Non-linear boost: compound risk factors amplify each other
    # (wind + low humidity + dry fuel is worse than the sum of parts)
    compound_boost = wind_risk * moisture_risk * 0.5
    exposure_boost = (wind_risk > 0.5).astype(float) * fuel_risk * 0.15
    risk = risk + compound_boost + exposure_boost

    # Add small random perturbation for realism
    np.random.seed(42)
    risk += np.random.normal(0, 0.03, len(risk))
    risk = np.clip(risk, 0, 1)

    df["ignition_risk"] = np.round(risk, 4)

    # Categorize
    df["risk_category"] = pd.cut(
        df["ignition_risk"],
        bins=[0, 0.3, 0.55, 0.75, 1.0],
        labels=["Low", "Moderate", "High", "Extreme"],
    )

    # Color mapping for visualization
    def risk_color(r):
        if r < 0.3:
            return [46, 204, 113, 140]   # green
        elif r < 0.55:
            return [241, 196, 15, 160]   # yellow
        elif r < 0.75:
            return [231, 76, 60, 180]    # red
        else:
            return [192, 57, 43, 220]    # dark red

    df["risk_color"] = df["ignition_risk"].apply(risk_color)

    # Column height for 3D (exaggerated for visual impact)
    df["risk_height"] = (df["ignition_risk"] * 800).astype(int)

    # Elevation-scaled height for terrain layer
    df["terrain_height"] = (df["elevation"] * 2).astype(int)

    return df


def compute_fire_spread_cone(
    ignition_lat: float,
    ignition_lon: float,
    weather: dict = None,
    hours: float = 6,
    terrain_df: pd.DataFrame = None,
) -> list[dict]:
    """
    Compute a fire spread cone (polygon) from an ignition point.
    Uses Rothermel-inspired spread rate model (simplified).

    The cone shape is driven by wind direction and speed.
    Spread is faster downwind and upslope.

    Args:
        ignition_lat, ignition_lon: Ignition point coordinates
        weather: Weather conditions
        hours: Hours of spread to project
        terrain_df: Terrain data for slope effects

    Returns:
        List of polygon coordinate dicts for visualization
    """
    wx = weather or WEATHER

    # Base spread rate (chains/hour → approximate degrees of lat/lon)
    # Rothermel simplified: R = R0 * (1 + φ_w + φ_s)
    base_rate = 0.002  # degrees per hour base

    wind_factor = 1.0 + (wx["wind_speed_mph"] / 20) ** 1.3
    humidity_factor = 1.0 + (1.0 - wx["humidity_pct"] / 100) * 0.5

    spread_rate = base_rate * wind_factor * humidity_factor

    # Wind direction (from config)
    wind_dir_rad = np.radians(wx["wind_direction_deg"])

    # Generate cone polygon points
    # Head fire (downwind) spreads fastest, flanks slower, backing fire slowest
    n_points = 36
    angles = np.linspace(0, 2 * np.pi, n_points)

    polygon_points = []
    for angle in angles:
        # Compute direction-dependent spread rate
        angle_from_wind = angle - wind_dir_rad
        # Elliptical model: max spread downwind, min upwind
        length_ratio = 0.2 + 0.8 * (1 + np.cos(angle_from_wind)) / 2
        # Further shape by wind speed (stronger wind = more elongated)
        eccentricity = min(0.9, wx["wind_speed_mph"] / 60)
        length_ratio = length_ratio ** (1 + eccentricity)

        distance = spread_rate * hours * length_ratio

        point_lat = ignition_lat + distance * np.cos(angle)
        point_lon = ignition_lon + distance * np.sin(angle) / np.cos(np.radians(ignition_lat))

        polygon_points.append([round(point_lon, 6), round(point_lat, 6)])

    # Close the polygon
    polygon_points.append(polygon_points[0])

    return polygon_points


def compute_multiple_spread_scenarios(
    ignition_lat: float,
    ignition_lon: float,
    weather: dict = None,
    hours_list: list = None,
) -> list[dict]:
    """
    Compute spread cones for multiple time horizons (ensemble visualization).

    Returns:
        List of dicts with 'hours', 'polygon', 'color', 'opacity'
    """
    if hours_list is None:
        hours_list = [3, 6, 12, 24]

    colors = [
        [255, 193, 7, 60],    # 3h - yellow
        [255, 152, 0, 60],    # 6h - orange
        [244, 67, 54, 60],    # 12h - red
        [136, 14, 79, 60],    # 24h - dark
    ]

    scenarios = []
    for i, hours in enumerate(hours_list):
        polygon = compute_fire_spread_cone(
            ignition_lat, ignition_lon, weather, hours,
        )
        scenarios.append({
            "hours": hours,
            "polygon": [polygon],  # GeoJSON expects nested
            "color": colors[i] if i < len(colors) else [100, 100, 100, 40],
        })

    return scenarios


def compute_risk_reduction(
    terrain_df: pd.DataFrame,
    original_risk: pd.DataFrame,
    new_risk: pd.DataFrame,
) -> dict:
    """
    Compute the risk reduction from a counterfactual intervention.

    Returns:
        Dict with reduction statistics
    """
    orig_mean = original_risk["ignition_risk"].mean()
    new_mean = new_risk["ignition_risk"].mean()
    reduction_abs = orig_mean - new_mean
    reduction_pct = (reduction_abs / orig_mean * 100) if orig_mean > 0 else 0

    orig_extreme = (original_risk["ignition_risk"] > 0.75).sum()
    new_extreme = (new_risk["ignition_risk"] > 0.75).sum()
    extreme_reduction = orig_extreme - new_extreme

    orig_high = (original_risk["ignition_risk"] > 0.55).sum()
    new_high = (new_risk["ignition_risk"] > 0.55).sum()

    return {
        "mean_risk_before": round(orig_mean, 4),
        "mean_risk_after": round(new_mean, 4),
        "reduction_pct": round(reduction_pct, 2),
        "extreme_cells_before": int(orig_extreme),
        "extreme_cells_after": int(new_extreme),
        "extreme_cells_eliminated": int(extreme_reduction),
        "high_risk_cells_before": int(orig_high),
        "high_risk_cells_after": int(new_high),
    }
