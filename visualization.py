"""
EarthDial v3 — 3D Visualization Layer
PyDeck-based Omniverse-style 3D rendering of terrain, risk surfaces,
power grid, fire spread cones, and wind fields.
"""

import pydeck as pdk
import pandas as pd
import numpy as np
from config import (
    CENTER_LAT, CENTER_LON, MAP_ZOOM, MAP_PITCH, MAP_BEARING,
    COLORS, FACILITY_ICONS,
)


def get_view_state(lat=CENTER_LAT, lon=CENTER_LON, zoom=MAP_ZOOM, pitch=MAP_PITCH, bearing=MAP_BEARING):
    """Create the 3D camera view state."""
    return pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=zoom,
        pitch=pitch,
        bearing=bearing,
        min_zoom=8,
        max_zoom=16,
    )


def create_risk_column_layer(terrain_df: pd.DataFrame) -> pdk.Layer:
    """
    3D columns showing ignition risk — the signature EarthDial visual.
    Height = risk level, Color = risk severity.
    """
    data = terrain_df[["lat", "lon", "ignition_risk", "risk_height", "risk_color"]].copy()
    data = data[data["ignition_risk"] > 0.2]  # Only show meaningful risk

    return pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=["lon", "lat"],
        get_elevation="risk_height",
        elevation_scale=1,
        radius=150,
        get_fill_color="risk_color",
        pickable=True,
        auto_highlight=True,
        coverage=0.85,
    )


def create_terrain_layer(terrain_df: pd.DataFrame) -> pdk.Layer:
    """
    Semi-transparent terrain elevation base layer.
    Shows the topography underneath the risk data.
    """
    data = terrain_df[["lat", "lon", "elevation", "terrain_height"]].copy()

    return pdk.Layer(
        "ColumnLayer",
        data=data,
        get_position=["lon", "lat"],
        get_elevation="terrain_height",
        elevation_scale=0.5,
        radius=140,
        get_fill_color=[34, 139, 34, 50],  # forest green, very transparent
        pickable=False,
        coverage=0.9,
    )


def create_risk_heatmap_layer(terrain_df: pd.DataFrame) -> pdk.Layer:
    """
    2D heatmap overlay of ignition risk (alternative to columns).
    Useful for a top-down view.
    """
    data = terrain_df[["lat", "lon", "ignition_risk"]].copy()

    return pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position=["lon", "lat"],
        get_weight="ignition_risk",
        radiusPixels=30,
        intensity=1.5,
        threshold=0.1,
        color_range=[
            [46, 204, 113],
            [241, 196, 15],
            [231, 76, 60],
            [192, 57, 43],
            [120, 0, 0],
        ],
        opacity=0.6,
    )


def create_power_grid_layer(powerlines_df: pd.DataFrame, disabled_lines: set = None) -> pdk.Layer:
    """
    3D arc layer showing power grid connections.
    Active lines = blue arcs, high-risk = red arcs, disabled = gray arcs.
    """
    disabled = disabled_lines or set()

    data = []
    for _, pl in powerlines_df.iterrows():
        is_disabled = pl["id"] in disabled
        veg_risk = pl["vegetation_risk"]

        if is_disabled:
            color = COLORS["grid_off"]
            height = 0.2
        elif veg_risk > 0.75:
            color = COLORS["grid_danger"]
            height = 0.8
        else:
            color = COLORS["grid_active"]
            height = 0.5

        data.append({
            "from_lat": pl["from_lat"],
            "from_lon": pl["from_lon"],
            "to_lat": pl["to_lat"],
            "to_lon": pl["to_lon"],
            "name": pl["name"],
            "voltage_kv": pl["voltage_kv"],
            "vegetation_risk": veg_risk,
            "status": "DISABLED" if is_disabled else "ACTIVE",
            "color": color,
            "height": height,
            "id": pl["id"],
        })

    return pdk.Layer(
        "ArcLayer",
        data=pd.DataFrame(data),
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_source_color="color",
        get_target_color="color",
        get_width=4,
        get_height="height",
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 0, 128],
    )


def create_substation_layer(substations_df: pd.DataFrame) -> pdk.Layer:
    """Render substations as prominent points."""
    data = substations_df.copy()
    data["color"] = [[118, 185, 0, 230]] * len(data)  # NVIDIA green
    data["size"] = 400

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["lon", "lat"],
        get_radius="size",
        get_fill_color="color",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=2,
    )


def create_critical_facilities_layer(facilities_df: pd.DataFrame, affected_ids: set = None) -> pdk.Layer:
    """
    Render critical facilities with color coding by type.
    Affected facilities (from shutoffs) are highlighted with warning colors.
    """
    affected = affected_ids or set()

    data = []
    for _, cf in facilities_df.iterrows():
        is_affected = cf["id"] in affected
        ftype = cf["type"]

        if is_affected:
            color = [255, 0, 0, 255]  # Red for affected
            size = 350
        else:
            color = COLORS.get(ftype, [200, 200, 200]) + [220]
            size = 250

        data.append({
            "lat": cf["lat"],
            "lon": cf["lon"],
            "name": cf["name"],
            "type": ftype,
            "icon": FACILITY_ICONS.get(ftype, "📍"),
            "status": "⚠️ POWER LOSS" if is_affected else "✅ Powered",
            "color": color,
            "size": size,
        })

    return pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame(data),
        get_position=["lon", "lat"],
        get_radius="size",
        get_fill_color="color",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
    )


def create_fire_spread_layer(scenarios: list[dict]) -> list[pdk.Layer]:
    """
    Render fire spread cones as semi-transparent polygons.
    Multiple time horizons shown with increasing size and darkening color.
    """
    layers = []
    for scenario in scenarios:
        data = pd.DataFrame([{
            "polygon": scenario["polygon"],
            "color": scenario["color"],
            "hours": scenario["hours"],
        }])

        layer = pdk.Layer(
            "PolygonLayer",
            data=data,
            get_polygon="polygon",
            get_fill_color="color",
            get_line_color=[255, 255, 255, 100],
            line_width_min_pixels=1,
            pickable=True,
            filled=True,
            wireframe=True,
            extruded=False,
            opacity=0.5,
        )
        layers.append(layer)

    return layers


def create_wind_field_layer(wind_df: pd.DataFrame) -> pdk.Layer:
    """
    Render wind field as directional lines (arrows).
    Line length proportional to wind speed.
    """
    data = wind_df.copy()

    # Color based on speed
    def speed_color(speed):
        if speed < 20:
            return [135, 206, 250, 150]  # light blue
        elif speed < 40:
            return [255, 165, 0, 180]    # orange
        else:
            return [255, 0, 0, 200]      # red

    data["color"] = data["speed"].apply(speed_color)

    return pdk.Layer(
        "LineLayer",
        data=data,
        get_source_position=["lon", "lat"],
        get_target_position=["end_lon", "end_lat"],
        get_color="color",
        get_width=2,
        pickable=True,
    )


def create_ignition_point_layer(lat: float, lon: float) -> pdk.Layer:
    """Render a pulsing ignition point marker."""
    data = pd.DataFrame([{
        "lat": lat,
        "lon": lon,
        "name": "Projected Ignition Point",
        "color": [255, 0, 0, 255],
        "size": 500,
    }])

    return pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["lon", "lat"],
        get_radius="size",
        get_fill_color="color",
        pickable=True,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=3,
    )


def build_full_3d_map(
    terrain_df: pd.DataFrame,
    powerlines_df: pd.DataFrame,
    substations_df: pd.DataFrame,
    facilities_df: pd.DataFrame,
    wind_df: pd.DataFrame = None,
    fire_scenarios: list[dict] = None,
    disabled_lines: set = None,
    affected_facility_ids: set = None,
    ignition_point: tuple = None,
    show_terrain: bool = False,
    show_risk_columns: bool = True,
    show_heatmap: bool = False,
    show_wind: bool = True,
    show_fire_spread: bool = True,
    view_state: pdk.ViewState = None,
) -> pdk.Deck:
    """
    Build the complete 3D visualization with all layers.

    This is the main rendering function — composes all visual layers
    into a single interactive 3D map.
    """
    layers = []

    # Terrain base (subtle)
    if show_terrain:
        layers.append(create_terrain_layer(terrain_df))

    # Risk visualization (columns OR heatmap)
    if show_risk_columns:
        layers.append(create_risk_column_layer(terrain_df))
    elif show_heatmap:
        layers.append(create_risk_heatmap_layer(terrain_df))

    # Power grid arcs
    layers.append(create_power_grid_layer(powerlines_df, disabled_lines))

    # Substations
    layers.append(create_substation_layer(substations_df))

    # Critical facilities
    layers.append(create_critical_facilities_layer(facilities_df, affected_facility_ids))

    # Wind field
    if show_wind and wind_df is not None:
        layers.append(create_wind_field_layer(wind_df))

    # Fire spread cones
    if show_fire_spread and fire_scenarios:
        layers.extend(create_fire_spread_layer(fire_scenarios))

    # Ignition point
    if ignition_point:
        layers.append(create_ignition_point_layer(*ignition_point))

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state or get_view_state(),
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={
            "html": "<b>{name}</b><br/>"
                    "Risk: {ignition_risk}<br/>"
                    "Status: {status}<br/>"
                    "Type: {type}<br/>"
                    "Voltage: {voltage_kv} kV<br/>"
                    "Vegetation Risk: {vegetation_risk}",
            "style": {
                "backgroundColor": "#1a1a2e",
                "color": "white",
                "fontSize": "12px",
                "padding": "8px",
            },
        },
    )
