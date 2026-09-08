"""Build polygon-style AMR Index maps using building footprints.

The available local geometry is the building footprint geometry (`the_geom`),
not official ZIP-code polygons. This script therefore creates zoning-style maps
by coloring each building footprint according to its existing geographic group
(neighborhood or ZIP code) and placing the group mean AMR Index label near the
group centroid.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Polygon


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

INDEX_CSV = OUTPUT_DIR / "building_level_no_ai_amr_index.csv"
SOURCE_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
HTML_OUTPUT = OUTPUT_DIR / "no_ai_amr_index_polygon_group_maps.html"
MIN_AREA_BUILDINGS = 20

WIDTH = 820
HEIGHT = 680
MARGIN = 32

ZONE_PALETTE = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
]


def parse_geom(value: object):
    if pd.isna(value):
        return None
    try:
        geom = wkt.loads(str(value))
    except Exception:
        return None
    if geom.is_empty:
        return None
    return geom


def read_data() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing AMR index file: {INDEX_CSV}")
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source geometry file: {SOURCE_CSV}")

    index_df = pd.read_csv(INDEX_CSV, low_memory=False)
    geom_df = pd.read_csv(SOURCE_CSV, usecols=["the_geom"], low_memory=False)
    if len(index_df) != len(geom_df):
        raise ValueError(
            f"Row mismatch: AMR index has {len(index_df)} rows, geometry source has {len(geom_df)} rows."
        )

    df = index_df.copy()
    df["the_geom"] = geom_df["the_geom"]
    df["baseline_amr_index"] = pd.to_numeric(df["baseline_amr_index"], errors="coerce")
    df["target_amr_accessible"] = pd.to_numeric(df["target_amr_accessible"], errors="coerce")
    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    df["geometry"] = df["the_geom"].map(parse_geom)
    return df.dropna(subset=["baseline_amr_index", "centroid_lon", "centroid_lat", "geometry"]).copy()


def aggregate(df: pd.DataFrame, area_col: str, exclude_unknown: bool = False) -> pd.DataFrame:
    work = df.copy()
    work[area_col] = work[area_col].fillna("Unknown").astype(str)
    if exclude_unknown:
        work = work[work[area_col].str.lower() != "unknown"]

    out = (
        work.groupby(area_col)
        .agg(
            n_buildings=("baseline_amr_index", "size"),
            mean_amr_index=("baseline_amr_index", "mean"),
            actual_accessibility_rate=("target_amr_accessible", "mean"),
            lon=("centroid_lon", "mean"),
            lat=("centroid_lat", "mean"),
        )
        .reset_index()
        .rename(columns={area_col: "area_name"})
    )
    out = out[out["n_buildings"] >= MIN_AREA_BUILDINGS].copy()
    return out


def zone_colors(groups: pd.DataFrame) -> dict[str, str]:
    ordered = groups.sort_values("area_name")["area_name"].astype(str).tolist()
    return {area: ZONE_PALETTE[i % len(ZONE_PALETTE)] for i, area in enumerate(ordered)}


def bounds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    return (
        float(df["centroid_lon"].min()),
        float(df["centroid_lat"].min()),
        float(df["centroid_lon"].max()),
        float(df["centroid_lat"].max()),
    )


def projector(map_bounds: tuple[float, float, float, float]):
    lon_min, lat_min, lon_max, lat_max = map_bounds

    def project(lon: float, lat: float) -> tuple[float, float]:
        x = MARGIN + (lon - lon_min) / (lon_max - lon_min) * (WIDTH - 2 * MARGIN)
        y = HEIGHT - MARGIN - (lat - lat_min) / (lat_max - lat_min) * (HEIGHT - 2 * MARGIN)
        return x, y

    return project


def polygon_to_paths(geom: Polygon | MultiPolygon, project) -> list[str]:
    polygons = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    paths = []
    for polygon in polygons:
        if polygon.is_empty:
            continue
        coords = list(polygon.exterior.coords)
        if len(coords) < 3:
            continue
        parts = []
        for i, (lon, lat) in enumerate(coords):
            x, y = project(float(lon), float(lat))
            parts.append(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}")
        parts.append("Z")
        paths.append(" ".join(parts))
    return paths


def panel_svg(title: str, df: pd.DataFrame, groups: pd.DataFrame, area_col: str, map_bounds) -> str:
    project = projector(map_bounds)
    group_lookup = groups.set_index("area_name")["mean_amr_index"].to_dict()
    color_lookup = zone_colors(groups)
    valid_groups = set(group_lookup)

    polygons = []
    for _, row in df.iterrows():
        area = str(row[area_col]) if pd.notna(row[area_col]) else "Unknown"
        if area not in valid_groups:
            continue
        fill = color_lookup[area]
        title_text = html.escape(
            f"{area} | building AMR Index: {float(row.baseline_amr_index):.1f} | "
            f"group mean: {float(group_lookup[area]):.1f}"
        )
        for path in polygon_to_paths(row.geometry, project):
            polygons.append(
                f'<path d="{path}" fill="{fill}" opacity="0.68" stroke="#2f3b45" stroke-width="0.45">'
                f"<title>{title_text}</title></path>"
            )

    labels = []
    for _, row in groups.sort_values("mean_amr_index", ascending=False).iterrows():
        x, y = project(float(row.lon), float(row.lat))
        area_label = html.escape(str(row.area_name))
        pct = f"{float(row.mean_amr_index):.1f}%"
        labels.append(
            f"""
            <g>
              <rect x="{x - 34:.1f}" y="{y - 18:.1f}" width="68" height="24" rx="4" fill="white" opacity="0.9" stroke="#172026" stroke-width="0.6" />
              <text x="{x:.1f}" y="{y - 1:.1f}" text-anchor="middle" class="pct-label">{pct}</text>
              <text x="{x:.1f}" y="{y + 15:.1f}" text-anchor="middle" class="area-label">{area_label}</text>
            </g>
            """
        )

    return f"""
    <section class="map-card">
      <h2>{html.escape(title)}</h2>
      <svg viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="{html.escape(title)}">
        <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" rx="10" fill="#f8fafc" stroke="#d7dde5" />
        <g>{''.join(polygons)}</g>
        <g>{''.join(labels)}</g>
      </svg>
    </section>
    """


def build_html(df: pd.DataFrame, neighborhood: pd.DataFrame, zipcode: pd.DataFrame) -> str:
    map_bounds = bounds(df)
    payload = {
        "n_buildings": int(len(df)),
        "n_neighborhoods": int(len(neighborhood)),
        "n_zipcodes": int(len(zipcode)),
        "min_area_buildings": MIN_AREA_BUILDINGS,
        "geometry_note": "Building footprints are colored categorically by group; official ZIP/neighborhood boundary polygons are not used.",
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Non-AI AMR Index Polygon Maps</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f3f5f7;
      color: #172026;
    }}
    header {{
      padding: 22px 28px;
      color: white;
      background: #1f2933;
    }}
    h1 {{ margin: 0 0 6px; font-size: 1.65rem; }}
    header p {{ margin: 0; color: #d9e1ea; max-width: 1080px; }}
    main {{
      padding: 18px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .map-card {{
      background: white;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      padding: 14px;
      box-shadow: 0 8px 22px rgba(23, 32, 42, 0.08);
    }}
    h2 {{ margin: 0 0 10px; font-size: 1.05rem; text-align: center; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .pct-label {{
      font-size: 13px;
      font-weight: 700;
      fill: #172026;
    }}
    .area-label {{
      font-size: 9px;
      fill: #172026;
      paint-order: stroke;
      stroke: white;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    .legend {{
      margin: 0 18px 18px;
      background: white;
      border: 1px solid #d7dde5;
      border-radius: 8px;
      padding: 12px 16px;
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      align-items: center;
      font-size: 0.92rem;
    }}
    .swatch {{
      width: 16px;
      height: 16px;
      display: inline-block;
      vertical-align: middle;
      margin-right: 6px;
      border: 1px solid #172026;
    }}
    @media (max-width: 1100px) {{ main {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <h1>Non-AI AMR Index: Polygon-Style Group Maps</h1>
    <p>Top-down building footprints are colored by neighborhood or ZIP-code group. Labels show the mean logistic-regression AMR Index for each group.</p>
  </header>
  <main>
    {panel_svg("Neighborhood grouping", df, neighborhood, "neighborhood", map_bounds)}
    {panel_svg("ZIP code grouping", df[df["zipcode"].astype(str).str.lower() != "unknown"], zipcode, "zipcode", map_bounds)}
  </main>
  <div class="legend">
    <strong>Color:</strong>
    <span>categorical zone/group color; all buildings in the same zone use the same color.</span>
    <strong>Label:</strong>
    <span>mean AMR Index (%) for the zone.</span>
    <span>Note: this uses building footprints, not official boundary polygons.</span>
  </div>
  <script type="application/json" id="summary">{json.dumps(payload)}</script>
</body>
</html>
"""


def main() -> None:
    df = read_data()
    neighborhood = aggregate(df, "neighborhood")
    zipcode = aggregate(df, "zipcode", exclude_unknown=True)
    HTML_OUTPUT.write_text(build_html(df, neighborhood, zipcode), encoding="utf-8")
    print(f"Buildings with footprint geometry: {len(df):,}")
    print(f"Neighborhood groups: {len(neighborhood):,}")
    print(f"ZIP code groups: {len(zipcode):,}")
    print(f"Saved: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
