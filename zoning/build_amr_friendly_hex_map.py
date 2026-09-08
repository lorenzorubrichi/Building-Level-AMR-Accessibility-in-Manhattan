"""Create a hexagonal AMR Index map without modifying the square-grid map.

This is an alternative visualization of AMR-friendly areas. It starts from the
same building-level non-AI AMR Index, aggregates buildings into smaller
hexagonal cells, and colors each hexagon by its mean AMR Index.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

BUILDING_INDEX_CSV = PROJECT_ROOT / "AMR_index_no_AI" / "output" / "building_level_no_ai_amr_index.csv"
SOURCE_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"

HEX_SUMMARY_CSV = OUTPUT_DIR / "amr_hex_cell_summary.csv"
HEX_GEOJSON = OUTPUT_DIR / "amr_hex_cells.geojson"
HEX_MAP_HTML = OUTPUT_DIR / "amr_hex_index_map.html"
AGG_HEX_SUMMARY_CSV = OUTPUT_DIR / "amr_aggregated_hex_zone_summary.csv"
AGG_HEX_GEOJSON = OUTPUT_DIR / "amr_aggregated_hex_zones.geojson"
AGG_HEX_MAP_HTML = OUTPUT_DIR / "amr_aggregated_hex_zone_map.html"
HEX_REPORT_TXT = OUTPUT_DIR / "amr_hex_index_report.txt"
SPATIAL_COMPARISON_CSV = OUTPUT_DIR / "amr_spatial_grouping_comparison.csv"

HEX_RADIUS_M = 200
MIN_BUILDINGS_PER_HEX = 20

LANDUSE_LABELS = {
    "1": "One & two family",
    "2": "Multi-family walk-up",
    "3": "Multi-family elevator",
    "4": "Mixed residential/commercial",
    "5": "Commercial & office",
    "6": "Industrial/manufacturing",
    "7": "Transportation/utility",
    "8": "Public facilities",
    "9": "Open space/recreation",
    "10": "Parking facilities",
    "11": "Vacant land",
}

BLDGCLASS_GROUP_LABELS = {
    "A": "One-family dwellings",
    "B": "Two-family dwellings",
    "C": "Walk-up apartments",
    "D": "Elevator apartments",
    "E": "Warehouses",
    "F": "Factory/industrial",
    "G": "Garages",
    "H": "Hotels",
    "I": "Hospitals/health",
    "J": "Theatres",
    "K": "Retail",
    "L": "Loft buildings",
    "M": "Religious buildings",
    "N": "Asylums/homes",
    "O": "Office buildings",
    "P": "Public assembly",
    "Q": "Outdoor recreation",
    "R": "Condominiums",
    "S": "Residence with store/office",
    "T": "Transportation",
    "U": "Utility",
    "V": "Vacant land",
    "W": "Educational structures",
    "Y": "Government/public safety",
    "Z": "Miscellaneous",
}


def readiness_class(mean_index: float) -> str:
    if mean_index < 40:
        return "low"
    if mean_index < 60:
        return "medium"
    return "high"


def color_by_index(value: float) -> str:
    if value < 40:
        return "#b2182b"
    if value < 60:
        return "#f6c85f"
    return "#1a9850"


def read_buildings() -> pd.DataFrame:
    if not BUILDING_INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing building AMR Index file: {BUILDING_INDEX_CSV}")
    df = pd.read_csv(BUILDING_INDEX_CSV, low_memory=False)
    if SOURCE_CSV.exists():
        source_cols = [
            "pluto_raw__landuse",
            "pluto_raw__bldgclass",
            "building_raw__construction_year",
            "pluto_raw__numfloors",
            "building_raw__height_roof",
        ]
        source = pd.read_csv(SOURCE_CSV, usecols=source_cols, low_memory=False)
        if len(source) == len(df):
            for col in source_cols:
                df[col] = source[col]
    for col in ["centroid_lon", "centroid_lat", "baseline_amr_index", "target_amr_accessible"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["centroid_lon", "centroid_lat", "baseline_amr_index", "target_amr_accessible"]).copy()
    df["neighborhood"] = df["neighborhood"].fillna("Unknown").astype(str)
    df["zipcode"] = df["zipcode"].fillna("Unknown").astype(str)
    df["landuse_for_summary"] = clean_landuse(df.get("pluto_raw__landuse", df.get("landuse")))
    df["bldgclass_group_for_summary"] = clean_bldgclass_group(df.get("pluto_raw__bldgclass", df.get("bldgclass")))
    df["construction_year_for_summary"] = clean_construction_year(
        df.get("building_raw__construction_year", pd.Series(index=df.index, dtype=object))
    )
    df["numfloors_for_summary"] = numeric(df.get("pluto_raw__numfloors", pd.Series(index=df.index, dtype=object)))
    return df


def numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def clean_landuse(series: pd.Series | None) -> pd.Series:
    values = numeric(series).round().astype("Int64").astype(str).replace("<NA>", "Unknown")
    return values.map(LANDUSE_LABELS).fillna("Unknown / other")


def clean_bldgclass_group(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    code = series.astype(str).str.strip().str.upper().str[0]
    return code.map(BLDGCLASS_GROUP_LABELS).fillna("Unknown / other")


def clean_construction_year(series: pd.Series | None) -> pd.Series:
    values = numeric(series)
    values = values.where(~values.between(1.0, 2.5), values * 1000)
    return values.where(values.between(1600, 2030))


def add_local_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    lat0 = math.radians(float(df["centroid_lat"].mean()))
    lon0 = float(df["centroid_lon"].mean())
    base_lat = float(df["centroid_lat"].mean())
    meters_per_deg_lon = 111_320 * math.cos(lat0)
    meters_per_deg_lat = 110_540

    out = df.copy()
    out["x_m"] = (out["centroid_lon"] - lon0) * meters_per_deg_lon
    out["y_m"] = (out["centroid_lat"] - base_lat) * meters_per_deg_lat
    meta = {
        "lon0": lon0,
        "lat0": base_lat,
        "meters_per_deg_lon": meters_per_deg_lon,
        "meters_per_deg_lat": meters_per_deg_lat,
    }
    return out, meta


def xy_to_lonlat(x: float, y: float, meta: dict[str, float]) -> tuple[float, float]:
    lon = meta["lon0"] + x / meta["meters_per_deg_lon"]
    lat = meta["lat0"] + y / meta["meters_per_deg_lat"]
    return lon, lat


def hex_polygon(cx: float, cy: float, radius: float, meta: dict[str, float]) -> Polygon:
    coords = []
    # Flat-top hexagon.
    for k in range(6):
        angle = math.radians(60 * k)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        coords.append(xy_to_lonlat(x, y, meta))
    coords.append(coords[0])
    return Polygon(coords)


def build_hex_centers(df: pd.DataFrame) -> list[tuple[str, float, float]]:
    radius = HEX_RADIUS_M
    dx = 1.5 * radius
    dy = math.sqrt(3) * radius
    min_x = float(df["x_m"].min()) - radius
    max_x = float(df["x_m"].max()) + radius
    min_y = float(df["y_m"].min()) - radius
    max_y = float(df["y_m"].max()) + radius

    centers = []
    col = 0
    x = min_x
    while x <= max_x:
        y_offset = (dy / 2) if col % 2 else 0
        row = 0
        y = min_y + y_offset
        while y <= max_y:
            centers.append((f"H{col}_{row}", x, y))
            row += 1
            y += dy
        col += 1
        x += dx
    return centers


def assign_to_nearest_hex(df: pd.DataFrame, centers: list[tuple[str, float, float]]) -> pd.DataFrame:
    center_ids = np.array([c[0] for c in centers], dtype=object)
    center_xy = np.array([(c[1], c[2]) for c in centers], dtype=float)
    point_xy = df[["x_m", "y_m"]].to_numpy(dtype=float)

    assignments = []
    for point in point_xy:
        distances = np.sum((center_xy - point) ** 2, axis=1)
        assignments.append(center_ids[int(np.argmin(distances))])

    out = df.copy()
    out["hex_id"] = assignments
    return out


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values[values.str.lower() != "unknown"]
    if values.empty:
        return "Unknown"
    return str(values.mode().iloc[0])


def top_category_summary(series: pd.Series, max_items: int = 3) -> str:
    values = series.dropna().astype(str)
    values = values[(values != "") & (values.str.lower() != "unknown")]
    if values.empty:
        return "Unknown"
    counts = values.value_counts().head(max_items)
    total = len(values)
    return "; ".join(f"{name}: {count / total * 100:.1f}%" for name, count in counts.items())


def median_or_none(series: pd.Series) -> float | None:
    value = series.dropna().median()
    return None if pd.isna(value) else float(value)


def build_hex_outputs(
    df: pd.DataFrame, centers: list[tuple[str, float, float]], meta: dict[str, float]
) -> tuple[pd.DataFrame, dict, dict[str, Polygon]]:
    center_lookup = {center_id: (x, y) for center_id, x, y in centers}
    polygon_lookup: dict[str, Polygon] = {}
    rows = []
    features = []

    for hex_id, group in df.groupby("hex_id"):
        if len(group) < MIN_BUILDINGS_PER_HEX:
            continue
        cx, cy = center_lookup[hex_id]
        mean_index = float(group["baseline_amr_index"].mean())
        actual_rate = float(group["target_amr_accessible"].mean())
        label = readiness_class(mean_index)
        polygon = hex_polygon(cx, cy, HEX_RADIUS_M, meta)
        polygon_lookup[hex_id] = polygon
        centroid_lon, centroid_lat = xy_to_lonlat(cx, cy, meta)

        row = {
            "hex_id": hex_id,
            "readiness_class": label,
            "n_buildings": int(len(group)),
            "mean_amr_index": mean_index,
            "std_amr_index": float(group["baseline_amr_index"].std(ddof=0)),
            "actual_accessibility_rate": actual_rate,
            "actual_minus_predicted_gap": actual_rate - mean_index / 100,
            "dominant_neighborhood": mode_or_unknown(group["neighborhood"]),
            "dominant_zipcode": mode_or_unknown(group["zipcode"]),
            "top_landuse": top_category_summary(group["landuse_for_summary"]),
            "top_bldgclass_groups": top_category_summary(group["bldgclass_group_for_summary"]),
            "median_construction_year": median_or_none(group["construction_year_for_summary"]),
            "median_numfloors": median_or_none(group["numfloors_for_summary"]),
            "centroid_lon": centroid_lon,
            "centroid_lat": centroid_lat,
            "share_high_index_70plus": float((group["baseline_amr_index"] >= 70).mean()),
            "share_low_index_under_30": float((group["baseline_amr_index"] < 30).mean()),
        }
        rows.append(row)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {
                    **row,
                    "color": color_by_index(mean_index),
                    "actual_accessibility_rate_pct": actual_rate * 100,
                },
            }
        )

    summary = pd.DataFrame(rows).sort_values(["centroid_lat", "centroid_lon"])
    geojson = {"type": "FeatureCollection", "features": features}
    return summary, geojson, polygon_lookup


def hex_neighbors(hex_id: str) -> list[str]:
    col_text, row_text = hex_id.replace("H", "").split("_")
    col = int(col_text)
    row = int(row_text)
    neighbors = [f"H{col}_{row - 1}", f"H{col}_{row + 1}"]
    if col % 2 == 0:
        neighbors.extend([f"H{col - 1}_{row}", f"H{col - 1}_{row - 1}", f"H{col + 1}_{row}", f"H{col + 1}_{row - 1}"])
    else:
        neighbors.extend([f"H{col - 1}_{row}", f"H{col - 1}_{row + 1}", f"H{col + 1}_{row}", f"H{col + 1}_{row + 1}"])
    return neighbors


def aggregate_adjacent_hex_zones(
    buildings: pd.DataFrame,
    hex_summary: pd.DataFrame,
    hex_polygon_lookup: dict[str, Polygon],
) -> tuple[pd.DataFrame, dict]:
    cell_class = dict(zip(hex_summary["hex_id"].astype(str), hex_summary["readiness_class"].astype(str)))
    valid_hexes = set(cell_class)
    visited: set[str] = set()
    components: list[list[str]] = []

    for hex_id in sorted(valid_hexes):
        if hex_id in visited:
            continue
        label = cell_class[hex_id]
        stack = [hex_id]
        visited.add(hex_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in hex_neighbors(current):
                if neighbor in visited or neighbor not in valid_hexes:
                    continue
                if cell_class[neighbor] == label:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    zone_counts = {"low": 0, "medium": 0, "high": 0}
    cell_to_zone: dict[str, str] = {}
    zone_polygon_lookup: dict[str, Polygon] = {}
    for component in components:
        label = cell_class[component[0]]
        zone_counts[label] += 1
        zone_id = f"AHZ_{zone_counts[label]:03d}_{label.upper()}"
        for hex_id in component:
            cell_to_zone[hex_id] = zone_id
        zone_polygon_lookup[zone_id] = unary_union([hex_polygon_lookup[hex_id] for hex_id in component]).buffer(0)

    rows = []
    features = []
    work = buildings[buildings["hex_id"].isin(cell_to_zone)].copy()
    work["aggregated_hex_zone_id"] = work["hex_id"].map(cell_to_zone)
    for zone_id, group in work.groupby("aggregated_hex_zone_id"):
        mean_index = float(group["baseline_amr_index"].mean())
        actual_rate = float(group["target_amr_accessible"].mean())
        label = readiness_class(mean_index)
        component_hexes = [hex_id for hex_id, mapped_zone in cell_to_zone.items() if mapped_zone == zone_id]
        row = {
            "zone_id": zone_id,
            "readiness_class": label,
            "n_hexagons": int(len(component_hexes)),
            "n_buildings": int(len(group)),
            "mean_amr_index": mean_index,
            "std_amr_index": float(group["baseline_amr_index"].std(ddof=0)),
            "actual_accessibility_rate": actual_rate,
            "actual_minus_predicted_gap": actual_rate - mean_index / 100,
            "dominant_neighborhood": mode_or_unknown(group["neighborhood"]),
            "dominant_zipcode": mode_or_unknown(group["zipcode"]),
            "top_landuse": top_category_summary(group["landuse_for_summary"]),
            "top_bldgclass_groups": top_category_summary(group["bldgclass_group_for_summary"]),
            "median_construction_year": median_or_none(group["construction_year_for_summary"]),
            "median_numfloors": median_or_none(group["numfloors_for_summary"]),
            "centroid_lon": float(group["centroid_lon"].mean()),
            "centroid_lat": float(group["centroid_lat"].mean()),
            "share_high_index_70plus": float((group["baseline_amr_index"] >= 70).mean()),
            "share_low_index_under_30": float((group["baseline_amr_index"] < 30).mean()),
        }
        rows.append(row)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(zone_polygon_lookup[zone_id]),
                "properties": {
                    **row,
                    "hex_id": zone_id,
                    "color": color_by_index(mean_index),
                    "actual_accessibility_rate_pct": actual_rate * 100,
                },
            }
        )

    summary = pd.DataFrame(rows).sort_values(["centroid_lat", "centroid_lon"])
    return summary, {"type": "FeatureCollection", "features": features}


def evaluate(summary: pd.DataFrame, buildings: pd.DataFrame) -> dict[str, float | int]:
    valid_ids = set(summary["hex_id"])
    work = buildings[buildings["hex_id"].isin(valid_ids)].copy()
    overall = float(work["baseline_amr_index"].mean())
    total_ss = float(((work["baseline_amr_index"] - overall) ** 2).sum())
    within_ss = 0.0
    weighted_std = 0.0

    for _, group in work.groupby("hex_id"):
        mean_index = float(group["baseline_amr_index"].mean())
        within_ss += float(((group["baseline_amr_index"] - mean_index) ** 2).sum())
        weighted_std += float(group["baseline_amr_index"].std(ddof=0)) * len(group)

    return {
        "n_hexes": int(len(summary)),
        "n_buildings": int(len(work)),
        "weighted_within_hex_std": weighted_std / len(work) if len(work) else np.nan,
        "between_hex_std_of_means": float(summary["mean_amr_index"].std(ddof=0)),
        "eta_squared_explained_variance": float(1 - within_ss / total_ss) if total_ss else np.nan,
    }


def evaluate_existing_grouping(
    buildings: pd.DataFrame,
    group_col: str,
    spatial_grouping: str,
    min_buildings: int = 20,
) -> dict[str, float | int | str]:
    work = buildings.dropna(subset=[group_col, "baseline_amr_index"]).copy()
    counts = work[group_col].value_counts()
    valid_groups = counts[counts >= min_buildings].index
    work = work[work[group_col].isin(valid_groups)].copy()

    n_zones = int(work[group_col].nunique())
    n_buildings = int(len(work))
    overall_mean = float(work["baseline_amr_index"].mean())
    total_ss = float(((work["baseline_amr_index"] - overall_mean) ** 2).sum())
    within_ss = 0.0
    weighted_std = 0.0

    for _, group in work.groupby(group_col):
        group_mean = float(group["baseline_amr_index"].mean())
        within_ss += float(((group["baseline_amr_index"] - group_mean) ** 2).sum())
        weighted_std += float(group["baseline_amr_index"].std(ddof=0)) * len(group)

    return {
        "spatial_grouping": spatial_grouping,
        "zones": n_zones,
        "mean_buildings_per_zone": n_buildings / n_zones if n_zones else np.nan,
        "within_zone_std": weighted_std / n_buildings if n_buildings else np.nan,
        "eta_squared": 1 - within_ss / total_ss if total_ss else np.nan,
    }


def evaluate_summary_grouping(
    summary: pd.DataFrame,
    spatial_grouping: str,
) -> dict[str, float | int | str]:
    n_zones = int(len(summary))
    n_buildings = int(summary["n_buildings"].sum())
    weights = summary["n_buildings"].astype(float)
    means = summary["mean_amr_index"].astype(float)
    standard_deviations = summary["std_amr_index"].astype(float)

    overall_mean = float(np.average(means, weights=weights))
    within_ss = float((weights * standard_deviations.pow(2)).sum())
    between_ss = float((weights * (means - overall_mean).pow(2)).sum())
    total_ss = within_ss + between_ss

    return {
        "spatial_grouping": spatial_grouping,
        "zones": n_zones,
        "mean_buildings_per_zone": n_buildings / n_zones if n_zones else np.nan,
        "within_zone_std": float(np.average(standard_deviations, weights=weights)),
        "eta_squared": between_ss / total_ss if total_ss else np.nan,
    }


def save_map(
    geojson: dict,
    summary: pd.DataFrame,
    output_html: Path = HEX_MAP_HTML,
    title: str = "Hexagonal AMR Index Map",
    unit_label: str = "hexagonal cells",
) -> None:
    payload = {
        "geojson": geojson,
        "labels": [
            {
                "hex_id": row.hex_id if "hex_id" in summary.columns else row.zone_id,
                "lat": float(row.centroid_lat),
                "lon": float(row.centroid_lon),
                "mean_amr_index": round(float(row.mean_amr_index), 1),
                "readiness_class": row.readiness_class,
            }
            for _, row in summary.iterrows()
        ],
    }

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; font-family: Arial, Helvetica, sans-serif; }}
    #map {{ height: 100%; width: 100%; }}
    .title-box {{
      position: absolute; top: 14px; left: 54px; z-index: 900;
      background: rgba(255,255,255,0.94); border: 1px solid #cfd6df;
      border-radius: 6px; padding: 10px 12px; max-width: 560px;
      box-shadow: 0 8px 22px rgba(23,32,42,0.15);
    }}
    .title-box h1 {{ margin: 0 0 4px; font-size: 18px; }}
    .title-box p {{ margin: 0; font-size: 13px; color: #53606b; }}
    .legend {{
      position: absolute; right: 14px; bottom: 24px; z-index: 900;
      background: rgba(255,255,255,0.94); border: 1px solid #cfd6df;
      border-radius: 6px; padding: 10px 12px; font-size: 12px;
      box-shadow: 0 8px 22px rgba(23,32,42,0.15);
    }}
    .swatch {{ display:inline-block; width:14px; height:14px; margin-right:6px; border:1px solid #172026; vertical-align:-2px; }}
    .amr-label {{
      background: rgba(255,255,255,0.92); border: 1px solid #1f2933;
      border-radius: 4px; color: #172026; font-weight: 700;
      font-size: 11px; padding: 1px 4px; white-space: nowrap;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-box">
    <h1>{title}</h1>
    <p>Buildings are aggregated by {unit_label}. Each area is colored by the mean AMR Index of buildings inside it.</p>
  </div>
  <div class="legend">
    <strong>Mean AMR Index</strong>
    <div><span class="swatch" style="background:#b2182b"></span>Low (0-40)</div>
    <div><span class="swatch" style="background:#f6c85f"></span>Medium (40-60)</div>
    <div><span class="swatch" style="background:#1a9850"></span>High (60-100)</div>
    <div style="margin-top:8px;">Hover a hexagon to fade the others.</div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const map = L.map('map');
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    function baseStyle(feature) {{
      return {{
        color: '#172026',
        weight: 1.4,
        fillColor: feature.properties.color,
        fillOpacity: 0.55,
        opacity: 0.88
      }};
    }}

    function mutedStyle() {{
      return {{
        color: '#9aa3aa',
        weight: 0.8,
        fillColor: '#c3cbd1',
        fillOpacity: 0.18,
        opacity: 0.35
      }};
    }}

    function highlightHex(activeHex) {{
      hexes.eachLayer(layer => {{
        if (layer.feature.properties.hex_id === activeHex) {{
          layer.setStyle({{
            color: '#111827',
            weight: 2.6,
            fillColor: layer.feature.properties.color,
            fillOpacity: 0.74,
            opacity: 1
          }});
          if (layer.bringToFront) layer.bringToFront();
        }} else {{
          layer.setStyle(mutedStyle());
        }}
      }});
    }}

    function resetHighlight() {{
      hexes.eachLayer(layer => layer.setStyle(baseStyle(layer.feature)));
    }}

    function bindTooltip(feature, layer) {{
      const p = feature.properties;
      layer.bindTooltip(
        `<b>${{p.hex_id}}</b><br>` +
        `Class: ${{p.readiness_class}}<br>` +
        `Mean AMR Index: ${{p.mean_amr_index.toFixed(1)}}<br>` +
        `Actual accessible rate: ${{p.actual_accessibility_rate_pct.toFixed(1)}}%<br>` +
        `Buildings: ${{p.n_buildings}}<br>` +
        `Dominant neighborhood: ${{p.dominant_neighborhood}}<br>` +
        `Dominant ZIP: ${{p.dominant_zipcode}}<br>` +
        `<br><b>Most common land use</b><br>${{p.top_landuse}}<br>` +
        `<br><b>Most common building types</b><br>${{p.top_bldgclass_groups}}<br>` +
        `<br>Median construction year: ${{p.median_construction_year === null ? 'Unknown' : p.median_construction_year.toFixed(0)}}<br>` +
        `Median floors: ${{p.median_numfloors === null ? 'Unknown' : p.median_numfloors.toFixed(1)}}`,
        {{ sticky: true }}
      );
      layer.on({{
        mouseover: () => highlightHex(p.hex_id),
        mouseout: resetHighlight
      }});
    }}

    const hexes = L.geoJSON(payload.geojson, {{ style: baseStyle, onEachFeature: bindTooltip }}).addTo(map);
    const labelsLayer = L.layerGroup().addTo(map);

    for (const item of payload.labels) {{
      const icon = L.divIcon({{ className: 'amr-label', html: `${{item.mean_amr_index.toFixed(0)}}`, iconSize: null }});
      L.marker([item.lat, item.lon], {{ icon }}).addTo(labelsLayer);
    }}
    L.control.layers(null, {{ 'AMR Index labels': labelsLayer }}, {{ collapsed: false }}).addTo(map);

    map.fitBounds(hexes.getBounds(), {{ padding: [24, 24] }});
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = read_buildings()
    buildings, meta = add_local_xy(buildings)
    centers = build_hex_centers(buildings)
    buildings = assign_to_nearest_hex(buildings, centers)
    summary, geojson, polygon_lookup = build_hex_outputs(buildings, centers, meta)
    agg_summary, agg_geojson = aggregate_adjacent_hex_zones(buildings, summary, polygon_lookup)

    summary.to_csv(HEX_SUMMARY_CSV, index=False)
    HEX_GEOJSON.write_text(json.dumps(geojson), encoding="utf-8")
    save_map(geojson, summary)
    agg_summary.to_csv(AGG_HEX_SUMMARY_CSV, index=False)
    AGG_HEX_GEOJSON.write_text(json.dumps(agg_geojson), encoding="utf-8")
    save_map(
        agg_geojson,
        agg_summary,
        AGG_HEX_MAP_HTML,
        "Aggregated Hexagonal AMR Zones",
        "adjacent hexagons with the same AMR readiness class",
    )

    metrics = evaluate(summary, buildings)
    known_zip_buildings = buildings[
        buildings["zipcode"].astype(str).str.lower() != "unknown"
    ].copy()
    spatial_comparison = pd.DataFrame(
        [
            evaluate_existing_grouping(buildings, "neighborhood", "Neighborhoods"),
            evaluate_existing_grouping(known_zip_buildings, "zipcode", "ZIP codes"),
            evaluate_summary_grouping(summary, "Hexagonal cells"),
            evaluate_summary_grouping(agg_summary, "Aggregated hexagonal zones"),
        ]
    )
    spatial_comparison = spatial_comparison.round(
        {
            "mean_buildings_per_zone": 1,
            "within_zone_std": 3,
            "eta_squared": 3,
        }
    )
    spatial_comparison.to_csv(SPATIAL_COMPARISON_CSV, index=False)

    report = [
        "Hexagonal AMR Index Map Report",
        "",
        f"Input buildings: {len(buildings):,}",
        f"Hex radius: {HEX_RADIUS_M} m",
        f"Minimum buildings per hexagon: {MIN_BUILDINGS_PER_HEX}",
        f"Retained hexagons: {metrics['n_hexes']:,}",
        f"Aggregated hex zones: {len(agg_summary):,}",
        f"Buildings inside retained hexagons: {metrics['n_buildings']:,}",
        f"Weighted within-hex AMR Index std: {metrics['weighted_within_hex_std']:.3f}",
        f"Between-hex std of mean AMR Index: {metrics['between_hex_std_of_means']:.3f}",
        f"Explained variance eta squared: {metrics['eta_squared_explained_variance']:.3f}",
        f"Spatial grouping comparison: {SPATIAL_COMPARISON_CSV.name}",
    ]
    HEX_REPORT_TXT.write_text("\n".join(report), encoding="utf-8")

    print(f"Input buildings: {len(buildings):,}")
    print(f"Retained hexagons: {metrics['n_hexes']:,}")
    print(f"Buildings inside retained hexagons: {metrics['n_buildings']:,}")
    print(f"Saved summary: {HEX_SUMMARY_CSV}")
    print(f"Saved geojson: {HEX_GEOJSON}")
    print(f"Saved map: {HEX_MAP_HTML}")
    print(f"Saved aggregated hex zones: {AGG_HEX_SUMMARY_CSV}")
    print(f"Saved aggregated hex map: {AGG_HEX_MAP_HTML}")
    print(f"Saved spatial grouping comparison: {SPATIAL_COMPARISON_CSV}")
    print(f"Saved report: {HEX_REPORT_TXT}")


if __name__ == "__main__":
    main()
