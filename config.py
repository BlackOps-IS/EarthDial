"""
EarthDial v3 — Configuration & Constants
Scenario: Sonoma County, California — Red Flag Warning
"""

# ─── Scenario Center (Sonoma County, CA) ────────────────────────────────────
CENTER_LAT = 38.52
CENTER_LON = -122.82
MAP_ZOOM = 11
MAP_PITCH = 55
MAP_BEARING = -20

# ─── Grid Resolution ────────────────────────────────────────────────────────
GRID_ROWS = 40
GRID_COLS = 40
GRID_STEP_LAT = 0.005   # ~550 m per step
GRID_STEP_LON = 0.006

# ─── Weather Scenario: Red Flag Warning ─────────────────────────────────────
WEATHER = {
    "wind_speed_mph": 45,
    "wind_gust_mph": 68,
    "wind_direction_deg": 30,        # NNE (Diablo winds)
    "temperature_f": 98,
    "humidity_pct": 8,
    "lightning_probability": 0.05,
    "forecast_hours": 72,
    "red_flag": True,
}

# ─── Risk Weights ───────────────────────────────────────────────────────────
RISK_WEIGHTS = {
    "wind": 0.30,
    "humidity": 0.20,
    "fuel_density": 0.20,
    "fuel_moisture": 0.10,
    "slope": 0.10,
    "powerline_proximity": 0.10,
}

# ─── Power Grid ─────────────────────────────────────────────────────────────
SUBSTATIONS = [
    {"id": "SUB-01", "name": "Sonoma Valley Substation",   "lat": 38.505, "lon": -122.84, "capacity_mw": 120},
    {"id": "SUB-02", "name": "Bennett Ridge Substation",   "lat": 38.545, "lon": -122.78, "capacity_mw": 85},
    {"id": "SUB-03", "name": "Mark West Substation",       "lat": 38.530, "lon": -122.73, "capacity_mw": 150},
    {"id": "SUB-04", "name": "Glen Ellen Substation",      "lat": 38.490, "lon": -122.86, "capacity_mw": 60},
    {"id": "SUB-05", "name": "Kenwood Substation",         "lat": 38.560, "lon": -122.70, "capacity_mw": 95},
]

POWER_LINES = [
    {"id": "PL-01", "name": "Bennett-Mark West 115kV",       "from": "SUB-02", "to": "SUB-03", "voltage_kv": 115, "vegetation_risk": 0.85, "age_years": 42},
    {"id": "PL-02", "name": "Sonoma-Glen Ellen 60kV",        "from": "SUB-01", "to": "SUB-04", "voltage_kv": 60,  "vegetation_risk": 0.45, "age_years": 28},
    {"id": "PL-03", "name": "Glen Ellen-Bennett 60kV",       "from": "SUB-04", "to": "SUB-02", "voltage_kv": 60,  "vegetation_risk": 0.72, "age_years": 35},
    {"id": "PL-04", "name": "Mark West-Kenwood 115kV",       "from": "SUB-03", "to": "SUB-05", "voltage_kv": 115, "vegetation_risk": 0.90, "age_years": 38},
    {"id": "PL-05", "name": "Sonoma-Bennett Ridge 115kV",    "from": "SUB-01", "to": "SUB-02", "voltage_kv": 115, "vegetation_risk": 0.60, "age_years": 25},
    {"id": "PL-06", "name": "Kenwood-Bennett 60kV",          "from": "SUB-05", "to": "SUB-02", "voltage_kv": 60,  "vegetation_risk": 0.78, "age_years": 51},
    {"id": "PL-07", "name": "Sonoma-Kenwood Trunk 230kV",    "from": "SUB-01", "to": "SUB-05", "voltage_kv": 230, "vegetation_risk": 0.55, "age_years": 18},
    {"id": "PL-08", "name": "Glen Ellen-Kenwood Feeder 60kV","from": "SUB-04", "to": "SUB-05", "voltage_kv": 60,  "vegetation_risk": 0.82, "age_years": 45},
]

# ─── Critical Facilities ───────────────────────────────────────────────────
CRITICAL_FACILITIES = [
    {"id": "CF-01", "name": "Sonoma Valley Hospital",           "type": "hospital",  "lat": 38.502, "lon": -122.835, "feeder": "PL-02", "priority": 1},
    {"id": "CF-02", "name": "Mark West Emergency Shelter",      "type": "shelter",   "lat": 38.528, "lon": -122.74,  "feeder": "PL-01", "priority": 1},
    {"id": "CF-03", "name": "Bennett Ridge Comm Tower",         "type": "comms",     "lat": 38.548, "lon": -122.79,  "feeder": "PL-01", "priority": 2},
    {"id": "CF-04", "name": "Glen Ellen Water Pump Station",    "type": "water",     "lat": 38.488, "lon": -122.855, "feeder": "PL-02", "priority": 1},
    {"id": "CF-05", "name": "Kenwood Fire Station",             "type": "fire_stn",  "lat": 38.558, "lon": -122.71,  "feeder": "PL-04", "priority": 1},
    {"id": "CF-06", "name": "Sonoma County EOC",                "type": "eoc",       "lat": 38.510, "lon": -122.82,  "feeder": "PL-05", "priority": 1},
    {"id": "CF-07", "name": "Highway 12 Traffic Control",       "type": "traffic",   "lat": 38.515, "lon": -122.78,  "feeder": "PL-05", "priority": 3},
    {"id": "CF-08", "name": "Oakmont Senior Living",            "type": "shelter",   "lat": 38.540, "lon": -122.75,  "feeder": "PL-01", "priority": 1},
]

# ─── Visualization ──────────────────────────────────────────────────────────
COLORS = {
    "risk_low":     [46, 204, 113],     # green
    "risk_medium":  [241, 196, 15],     # yellow
    "risk_high":    [231, 76, 60],      # red
    "risk_extreme": [192, 57, 43],      # dark red
    "grid_active":  [52, 152, 219],     # blue
    "grid_off":     [149, 165, 166],    # gray
    "grid_danger":  [231, 76, 60],      # red
    "hospital":     [255, 255, 255],    # white
    "shelter":      [155, 89, 182],     # purple
    "comms":        [52, 152, 219],     # blue
    "water":        [26, 188, 156],     # teal
    "fire_stn":     [231, 76, 60],      # red
    "eoc":          [241, 196, 15],     # yellow
    "traffic":      [243, 156, 18],     # orange
    "fire_cone":    [255, 87, 34, 80],  # orange translucent
    "nvidia_green": [118, 185, 0],      # NVIDIA green
}

FACILITY_ICONS = {
    "hospital": "🏥",
    "shelter":  "🏠",
    "comms":    "📡",
    "water":    "💧",
    "fire_stn": "🚒",
    "eoc":      "🏛️",
    "traffic":  "🚦",
}
