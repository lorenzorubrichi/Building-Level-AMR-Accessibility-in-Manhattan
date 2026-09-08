"""Create data-driven AMR-friendly zones from the building-level AMR Index.

Workflow:
1. Read building-level non-AI AMR Index predictions.
2. Aggregate buildings into regular spatial grid cells.
3. Classify cells as low / medium / high AMR readiness.
4. Merge adjacent cells with the same class into AMR-friendly zones.
5. Save CSV, GeoJSON, HTML map, and first evaluation metrics.

These zones are not official administrative boundaries. They are an exploratory
planning layer derived from AMR readiness patterns.
"""

from __future__ import annotations

import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

BUILDING_INDEX_CSV = PROJECT_ROOT / "AMR_index_no_AI" / "output" / "building_level_no_ai_amr_index.csv"
SOURCE_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
NEIGHBORHOODS_GEOJSON = PROJECT_ROOT / "dataset" / "rawdata" / "nyc_neighborhoods.geojson"

GRID_CELL_CSV = OUTPUT_DIR / "amr_grid_cell_summary.csv"
ZONE_SUMMARY_CSV = OUTPUT_DIR / "amr_friendly_zone_summary.csv"
ZONE_GEOJSON = OUTPUT_DIR / "amr_friendly_zones.geojson"
ZONE_METRICS_CSV = OUTPUT_DIR / "amr_friendly_zone_evaluation_metrics.csv"
ZONE_MAP_HTML = OUTPUT_DIR / "amr_friendly_zone_map.html"
REPORT_TXT = OUTPUT_DIR / "amr_friendly_zone_report.txt"

CELL_SIZE_M = 300
MIN_BUILDINGS_PER_CELL = 10

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
    df["zipcode"] = df["zipcode"].fillna("Unknown").astype(str)
    df["neighborhood"] = df["neighborhood"].fillna("Unknown").astype(str)
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


def read_neighborhood_outlines() -> dict | None:
    if not NEIGHBORHOODS_GEOJSON.exists():
        return None
    data = json.loads(NEIGHBORHOODS_GEOJSON.read_text(encoding="utf-8"))
    # Drop the duplicate WKT geometry property to keep the HTML smaller.
    for feature in data.get("features", []):
        feature.get("properties", {}).pop("the_geom", None)
    return data


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


def readiness_class(mean_index: float) -> str:
    if mean_index < 40:
        return "low"
    if mean_index < 60:
        return "medium"
    return "high"


def readiness_color(label: str) -> str:
    return {
        "low": "#d73027",
        "medium": "#fee08b",
        "high": "#1a9850",
    }[label]


def build_grid_cells(df: pd.DataFrame, meta: dict[str, float]) -> tuple[pd.DataFrame, dict[tuple[int, int], Polygon]]:
    min_x = math.floor(df["x_m"].min() / CELL_SIZE_M) * CELL_SIZE_M
    min_y = math.floor(df["y_m"].min() / CELL_SIZE_M) * CELL_SIZE_M
    work = df.copy()
    work["grid_i"] = np.floor((work["x_m"] - min_x) / CELL_SIZE_M).astype(int)
    work["grid_j"] = np.floor((work["y_m"] - min_y) / CELL_SIZE_M).astype(int)
    work["grid_id"] = work["grid_i"].astype(str) + "_" + work["grid_j"].astype(str)

    rows = []
    polygons: dict[tuple[int, int], Polygon] = {}
    for (i, j), group in work.groupby(["grid_i", "grid_j"]):
        if len(group) < MIN_BUILDINGS_PER_CELL:
            continue
        x0 = min_x + i * CELL_SIZE_M
        y0 = min_y + j * CELL_SIZE_M
        corners = [
            xy_to_lonlat(x0, y0, meta),
            xy_to_lonlat(x0 + CELL_SIZE_M, y0, meta),
            xy_to_lonlat(x0 + CELL_SIZE_M, y0 + CELL_SIZE_M, meta),
            xy_to_lonlat(x0, y0 + CELL_SIZE_M, meta),
            xy_to_lonlat(x0, y0, meta),
        ]
        polygon = Polygon(corners)
        mean_index = float(group["baseline_amr_index"].mean())
        rows.append(
            {
                "grid_id": f"{i}_{j}",
                "grid_i": int(i),
                "grid_j": int(j),
                "n_buildings": int(len(group)),
                "mean_amr_index": mean_index,
                "std_amr_index": float(group["baseline_amr_index"].std(ddof=0)),
                "actual_accessibility_rate": float(group["target_amr_accessible"].mean()),
                "readiness_class": readiness_class(mean_index),
                "centroid_lon": float(group["centroid_lon"].mean()),
                "centroid_lat": float(group["centroid_lat"].mean()),
                "dominant_neighborhood": group["neighborhood"].mode().iloc[0],
                "dominant_zipcode": group["zipcode"].mode().iloc[0],
                "top_landuse": top_category_summary(group["landuse_for_summary"]),
                "top_bldgclass_groups": top_category_summary(group["bldgclass_group_for_summary"]),
                "median_construction_year": median_or_none(group["construction_year_for_summary"]),
                "median_numfloors": median_or_none(group["numfloors_for_summary"]),
                "share_high_index_70plus": float((group["baseline_amr_index"] >= 70).mean()),
                "share_low_index_under_30": float((group["baseline_amr_index"] < 30).mean()),
            }
        )
        polygons[(int(i), int(j))] = polygon

    cells = pd.DataFrame(rows).sort_values(["grid_j", "grid_i"])
    return cells, polygons


def connected_components(cells: pd.DataFrame) -> list[list[tuple[int, int]]]:
    cell_class = {
        (int(row.grid_i), int(row.grid_j)): str(row.readiness_class)
        for _, row in cells.iterrows()
    }
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for cell, label in cell_class.items():
        if cell in visited:
            continue
        component = []
        queue = deque([cell])
        visited.add(cell)
        while queue:
            current = queue.popleft()
            component.append(current)
            i, j = current
            for nb in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if nb in visited:
                    continue
                if cell_class.get(nb) == label:
                    visited.add(nb)
                    queue.append(nb)
        components.append(component)
    return components


def summarize_zones(
    df: pd.DataFrame,
    cells: pd.DataFrame,
    polygons: dict[tuple[int, int], Polygon],
) -> tuple[pd.DataFrame, dict]:
    cell_lookup = cells.set_index(["grid_i", "grid_j"]).to_dict(orient="index")
    components = connected_components(cells)
    zone_rows = []
    features = []

    zone_counter = 1
    for component in components:
        component_cells = cells[
            cells.apply(lambda row: (int(row.grid_i), int(row.grid_j)) in component, axis=1)
        ].copy()
        grid_ids = set(component_cells["grid_id"])
        buildings = df[
            (
                df["grid_i"].astype(str)
                + "_"
                + df["grid_j"].astype(str)
            ).isin(grid_ids)
        ].copy()
        if buildings.empty:
            continue

        label = str(component_cells["readiness_class"].iloc[0])
        zone_id = f"AMRZ_{zone_counter:02d}_{label.upper()}"
        zone_counter += 1
        geom = unary_union([polygons[cell] for cell in component]).buffer(0)
        if isinstance(geom, Polygon):
            geom_out = geom
        elif isinstance(geom, MultiPolygon):
            geom_out = geom
        else:
            continue

        mean_index = float(buildings["baseline_amr_index"].mean())
        actual_rate = float(buildings["target_amr_accessible"].mean())
        std_index = float(buildings["baseline_amr_index"].std(ddof=0))
        dominant_neighborhood = Counter(buildings["neighborhood"]).most_common(1)[0][0]
        dominant_zipcode = Counter(buildings["zipcode"]).most_common(1)[0][0]

        row = {
            "zone_id": zone_id,
            "readiness_class": label,
            "n_cells": int(len(component)),
            "n_buildings": int(len(buildings)),
            "mean_amr_index": mean_index,
            "std_amr_index": std_index,
            "actual_accessibility_rate": actual_rate,
            "actual_minus_predicted_gap": actual_rate - (mean_index / 100),
            "share_high_index_70plus": float((buildings["baseline_amr_index"] >= 70).mean()),
            "share_low_index_under_30": float((buildings["baseline_amr_index"] < 30).mean()),
            "dominant_neighborhood": dominant_neighborhood,
            "dominant_zipcode": dominant_zipcode,
            "top_landuse": top_category_summary(buildings["landuse_for_summary"]),
            "top_bldgclass_groups": top_category_summary(buildings["bldgclass_group_for_summary"]),
            "median_construction_year": median_or_none(buildings["construction_year_for_summary"]),
            "median_numfloors": median_or_none(buildings["numfloors_for_summary"]),
            "centroid_lon": float(buildings["centroid_lon"].mean()),
            "centroid_lat": float(buildings["centroid_lat"].mean()),
        }
        zone_rows.append(row)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom_out),
                "properties": {
                    **row,
                    "color": readiness_color(label),
                    "actual_accessibility_rate_pct": actual_rate * 100,
                },
            }
        )

    zones = pd.DataFrame(zone_rows).sort_values(["readiness_class", "zone_id"])
    geojson = {"type": "FeatureCollection", "features": features}
    return zones, geojson


def evaluate_partition(buildings: pd.DataFrame, group_col: str, min_buildings: int = 20) -> dict[str, float | str | int]:
    work = buildings.dropna(subset=[group_col, "baseline_amr_index"]).copy()
    counts = work[group_col].value_counts()
    valid = counts[counts >= min_buildings].index
    work = work[work[group_col].isin(valid)].copy()
    if work.empty:
        return {
            "partition": group_col,
            "n_groups": 0,
            "n_buildings": 0,
            "weighted_within_zone_std": np.nan,
            "between_group_std_of_means": np.nan,
            "eta_squared_explained_variance": np.nan,
            "mean_abs_actual_predicted_gap": np.nan,
        }

    overall = float(work["baseline_amr_index"].mean())
    total_ss = float(((work["baseline_amr_index"] - overall) ** 2).sum())
    within_ss = 0.0
    weighted_std_parts = []
    gaps = []
    group_means = []

    for _, group in work.groupby(group_col):
        mean_index = float(group["baseline_amr_index"].mean())
        within_ss += float(((group["baseline_amr_index"] - mean_index) ** 2).sum())
        weighted_std_parts.append(float(group["baseline_amr_index"].std(ddof=0)) * len(group))
        gaps.append(abs(float(group["target_amr_accessible"].mean()) - mean_index / 100))
        group_means.append(mean_index)

    return {
        "partition": group_col,
        "n_groups": int(work[group_col].nunique()),
        "n_buildings": int(len(work)),
        "weighted_within_zone_std": float(sum(weighted_std_parts) / len(work)),
        "between_group_std_of_means": float(np.std(group_means, ddof=0)),
        "eta_squared_explained_variance": float(1 - within_ss / total_ss) if total_ss else np.nan,
        "mean_abs_actual_predicted_gap": float(np.mean(gaps)),
    }


def save_map(geojson: dict, zones: pd.DataFrame) -> None:
    payload = {
        "geojson": geojson,
        "neighborhoodOutlines": read_neighborhood_outlines(),
        "labels": [
            {
                "zone_id": row.zone_id,
                "lat": float(row.centroid_lat),
                "lon": float(row.centroid_lon),
                "mean_amr_index": round(float(row.mean_amr_index), 1),
                "readiness_class": row.readiness_class,
            }
            for _, row in zones.iterrows()
        ],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AMR-Friendly Data-Driven Zones</title>
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
    .swatch {{ display:inline-block; width:14px; height:14px; border:1px solid #172026; margin-right:6px; vertical-align:middle; }}
    .amr-label {{
      background: rgba(255,255,255,0.92); border: 1px solid #1f2933;
      border-radius: 4px; color: #172026; font-weight: 700;
      font-size: 12px; padding: 2px 5px; white-space: nowrap;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-box">
    <h1>Data-Driven AMR-Friendly Zones</h1>
    <p>Regular grid cells are classified by mean AMR Index and adjacent cells with the same class are merged into exploratory AMR zones.</p>
  </div>
  <div class="legend">
    <div><span class="swatch" style="background:#d73027"></span>Low AMR readiness (0-40)</div>
    <div><span class="swatch" style="background:#fee08b"></span>Medium AMR readiness (40-60)</div>
    <div><span class="swatch" style="background:#1a9850"></span>High AMR readiness (60-100)</div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const map = L.map('map');
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    function neighborhoodStyle() {{
      return {{
        color: '#334155',
        weight: 1.2,
        fillOpacity: 0,
        opacity: 0.7,
        dashArray: '4 3',
        interactive: false
      }};
    }}

    let neighborhoodOutlines = null;
    if (payload.neighborhoodOutlines) {{
      neighborhoodOutlines = L.geoJSON(payload.neighborhoodOutlines, {{ style: neighborhoodStyle }}).addTo(map);
    }}

    function styleFeature(feature) {{
      return {{
        color: '#172026',
        weight: 2,
        fillColor: feature.properties.color,
        fillOpacity: 0.48,
        opacity: 0.92
      }};
    }}

    function bindPopup(feature, layer) {{
      const p = feature.properties;
      layer.bindTooltip(
        `<b>${{p.zone_id}}</b><br>` +
        `Class: ${{p.readiness_class}}<br>` +
        `Mean AMR Index: ${{p.mean_amr_index.toFixed(1)}}<br>` +
        `Actual accessible rate: ${{p.actual_accessibility_rate_pct.toFixed(1)}}%<br>` +
        `Buildings: ${{p.n_buildings}}<br>` +
        `Cells: ${{p.n_cells}}<br>` +
        `<br><b>Most common land use</b><br>${{p.top_landuse}}<br>` +
        `<br><b>Most common building types</b><br>${{p.top_bldgclass_groups}}<br>` +
        `<br>Median construction year: ${{p.median_construction_year === null ? 'Unknown' : p.median_construction_year.toFixed(0)}}<br>` +
        `Median floors: ${{p.median_numfloors === null ? 'Unknown' : p.median_numfloors.toFixed(1)}}`,
        {{ sticky: true }}
      );
    }}

    const zones = L.geoJSON(payload.geojson, {{ style: styleFeature, onEachFeature: bindPopup }}).addTo(map);
    const labelsLayer = L.layerGroup().addTo(map);
    for (const item of payload.labels) {{
      const icon = L.divIcon({{ className: 'amr-label', html: `${{item.mean_amr_index.toFixed(1)}}%`, iconSize: null }});
      L.marker([item.lat, item.lon], {{ icon }}).addTo(labelsLayer);
    }}
    const overlays = {{ 'AMR Index labels': labelsLayer }};
    if (neighborhoodOutlines) {{
      overlays['Neighborhood outlines'] = neighborhoodOutlines;
    }}
    L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);
    map.fitBounds(zones.getBounds(), {{ padding: [24, 24] }});
  </script>
</body>
</html>
"""
    ZONE_MAP_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = read_buildings()
    buildings, meta = add_local_xy(buildings)
    cells, polygons = build_grid_cells(buildings, meta)

    # Attach grid ids to the building dataframe after valid cells are known.
    min_x = math.floor(buildings["x_m"].min() / CELL_SIZE_M) * CELL_SIZE_M
    min_y = math.floor(buildings["y_m"].min() / CELL_SIZE_M) * CELL_SIZE_M
    buildings["grid_i"] = np.floor((buildings["x_m"] - min_x) / CELL_SIZE_M).astype(int)
    buildings["grid_j"] = np.floor((buildings["y_m"] - min_y) / CELL_SIZE_M).astype(int)

    zones, geojson = summarize_zones(buildings, cells, polygons)

    cell_zone_map = {}
    for feature in geojson["features"]:
        zone_id = feature["properties"]["zone_id"]
        zone_class = feature["properties"]["readiness_class"]
        # Assign by spatial containment through the zone polygon is unnecessary for
        # metrics because zones are already summarized. Use cell-level membership
        # for exported grid summary instead.
    cells.to_csv(GRID_CELL_CSV, index=False)
    zones.to_csv(ZONE_SUMMARY_CSV, index=False)
    ZONE_GEOJSON.write_text(json.dumps(geojson), encoding="utf-8")
    save_map(geojson, zones)

    valid_grid_ids = set(cells["grid_id"])
    metric_buildings = buildings[
        (buildings["grid_i"].astype(str) + "_" + buildings["grid_j"].astype(str)).isin(valid_grid_ids)
    ].copy()
    zone_assignment = []
    for _, zone in zones.iterrows():
        matching_cells = cells[cells["readiness_class"] == zone["readiness_class"]]
    # Assign each building to a zone using polygon containment for evaluation.
    zone_polygons = []
    for feature in geojson["features"]:
        zone_polygons.append((feature["properties"]["zone_id"], unary_union([]) if False else feature))
    from shapely.geometry import Point, shape

    def find_zone_id(row: pd.Series) -> str | None:
        point = Point(float(row["centroid_lon"]), float(row["centroid_lat"]))
        for feature in geojson["features"]:
            geom = shape(feature["geometry"])
            if geom.contains(point) or geom.touches(point):
                return feature["properties"]["zone_id"]
        return None

    metric_buildings["amr_friendly_zone"] = metric_buildings.apply(find_zone_id, axis=1)
    metrics = pd.DataFrame(
        [
            evaluate_partition(metric_buildings.dropna(subset=["amr_friendly_zone"]), "amr_friendly_zone"),
            evaluate_partition(buildings, "neighborhood"),
            evaluate_partition(buildings[buildings["zipcode"].astype(str).str.lower() != "unknown"], "zipcode"),
        ]
    )
    metrics.to_csv(ZONE_METRICS_CSV, index=False)

    report = [
        "AMR-Friendly Zone Construction Report",
        "",
        f"Input buildings: {len(buildings):,}",
        f"Grid cell size: {CELL_SIZE_M} m",
        f"Minimum buildings per cell: {MIN_BUILDINGS_PER_CELL}",
        f"Retained grid cells: {len(cells):,}",
        f"Generated AMR-friendly zones: {len(zones):,}",
        "",
        "Readiness classes:",
        "- low: mean AMR Index < 40",
        "- medium: 40 <= mean AMR Index < 60",
        "- high: mean AMR Index >= 60",
        "",
        "Initial evaluation metrics were saved for the new zones and for existing neighborhood/ZIP partitions.",
    ]
    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")

    print(f"Buildings used: {len(buildings):,}")
    print(f"Retained grid cells: {len(cells):,}")
    print(f"AMR-friendly zones: {len(zones):,}")
    print(f"Saved cells: {GRID_CELL_CSV}")
    print(f"Saved zones: {ZONE_SUMMARY_CSV}")
    print(f"Saved metrics: {ZONE_METRICS_CSV}")
    print(f"Saved map: {ZONE_MAP_HTML}")


if __name__ == "__main__":
    main()
