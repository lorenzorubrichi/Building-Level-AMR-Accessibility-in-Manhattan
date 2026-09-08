"""Build simple self-contained HTML maps for the non-AI AMR Index.

The output is an HTML file with two SVG map panels:
1. grouping by neighborhood;
2. grouping by ZIP code.

The map is intentionally dependency-free and does not require online map tiles.
It uses building coordinates as a geographic reference and places labeled group
centroids on top.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
INPUT_CSV = OUTPUT_DIR / "building_level_no_ai_amr_index.csv"
HTML_OUTPUT = OUTPUT_DIR / "no_ai_amr_index_group_maps.html"
MIN_AREA_BUILDINGS = 20


def read_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    df["baseline_amr_index"] = pd.to_numeric(df["baseline_amr_index"], errors="coerce")
    df["target_amr_accessible"] = pd.to_numeric(df["target_amr_accessible"], errors="coerce")
    return df.dropna(subset=["centroid_lon", "centroid_lat", "baseline_amr_index"])


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
    return out.sort_values("mean_amr_index", ascending=False)


def color(value: float) -> str:
    """Color ramp from low red to mid amber to high green."""
    if value < 30:
        return "#b2182b"
    if value < 45:
        return "#ef8a62"
    if value < 60:
        return "#f6c85f"
    if value < 75:
        return "#7fbf7b"
    return "#1a9850"


def project_points(df: pd.DataFrame, width: int, height: int, margin: int = 34) -> pd.DataFrame:
    lon_min, lon_max = df["centroid_lon"].min(), df["centroid_lon"].max()
    lat_min, lat_max = df["centroid_lat"].min(), df["centroid_lat"].max()

    projected = df.copy()
    projected["x"] = margin + (projected["centroid_lon"] - lon_min) / (lon_max - lon_min) * (width - 2 * margin)
    projected["y"] = height - margin - (projected["centroid_lat"] - lat_min) / (lat_max - lat_min) * (height - 2 * margin)
    return projected


def project_groups(groups: pd.DataFrame, base: pd.DataFrame, width: int, height: int, margin: int = 34) -> pd.DataFrame:
    lon_min, lon_max = base["centroid_lon"].min(), base["centroid_lon"].max()
    lat_min, lat_max = base["centroid_lat"].min(), base["centroid_lat"].max()

    projected = groups.copy()
    projected["x"] = margin + (projected["lon"] - lon_min) / (lon_max - lon_min) * (width - 2 * margin)
    projected["y"] = height - margin - (projected["lat"] - lat_min) / (lat_max - lat_min) * (height - 2 * margin)
    return projected


def panel_svg(title: str, base_points: pd.DataFrame, groups: pd.DataFrame, width: int = 760, height: int = 590) -> str:
    base = project_points(base_points, width, height)
    agg = project_groups(groups, base_points, width, height)

    point_step = max(1, len(base) // 1600)
    building_points = []
    for _, row in base.iloc[::point_step].iterrows():
        building_points.append(
            f'<circle cx="{row.x:.1f}" cy="{row.y:.1f}" r="1.2" fill="#9aa4ae" opacity="0.28" />'
        )

    group_marks = []
    for _, row in agg.iterrows():
        r = 9 + min(26, row.n_buildings ** 0.5 * 0.75)
        label = f"{row.mean_amr_index:.1f}%"
        area_label = html.escape(str(row.area_name))
        tooltip = html.escape(
            f"{row.area_name} | buildings: {int(row.n_buildings)} | "
            f"mean AMR Index: {row.mean_amr_index:.1f} | "
            f"actual accessible: {row.actual_accessibility_rate * 100:.1f}%"
        )
        group_marks.append(
            f"""
            <g>
              <title>{tooltip}</title>
              <circle cx="{row.x:.1f}" cy="{row.y:.1f}" r="{r:.1f}" fill="{color(row.mean_amr_index)}" opacity="0.86" stroke="#172026" stroke-width="1" />
              <text x="{row.x:.1f}" y="{row.y - r - 7:.1f}" text-anchor="middle" class="pct-label">{label}</text>
              <text x="{row.x:.1f}" y="{row.y + r + 16:.1f}" text-anchor="middle" class="area-label">{area_label}</text>
            </g>
            """
        )

    return f"""
    <section class="map-card">
      <h2>{html.escape(title)}</h2>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
        <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#f8fafc" stroke="#d7dde5" />
        <g>{''.join(building_points)}</g>
        <g>{''.join(group_marks)}</g>
      </svg>
    </section>
    """


def build_html(df: pd.DataFrame, neighborhood: pd.DataFrame, zipcode: pd.DataFrame) -> str:
    payload = {
        "n_buildings": int(len(df)),
        "n_neighborhoods": int(len(neighborhood)),
        "n_zipcodes": int(len(zipcode)),
        "min_area_buildings": MIN_AREA_BUILDINGS,
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Non-AI AMR Index Group Maps</title>
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
    h1 {{
      margin: 0 0 6px;
      font-size: 1.65rem;
    }}
    header p {{
      margin: 0;
      color: #d9e1ea;
    }}
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
    h2 {{
      margin: 0 0 10px;
      font-size: 1.05rem;
      text-align: center;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .pct-label {{
      font-size: 13px;
      font-weight: 700;
      fill: #172026;
      paint-order: stroke;
      stroke: white;
      stroke-width: 3px;
      stroke-linejoin: round;
    }}
    .area-label {{
      font-size: 9.8px;
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
      border-radius: 50%;
      display: inline-block;
      vertical-align: middle;
      margin-right: 6px;
      border: 1px solid #172026;
    }}
    @media (max-width: 1100px) {{
      main {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Non-AI AMR Index by Existing Geographic Groups</h1>
    <p>Logistic-regression AMR Index aggregated by neighborhood and ZIP code. Labels show mean AMR Index.</p>
  </header>
  <main>
    {panel_svg("Neighborhood grouping", df, neighborhood)}
    {panel_svg("ZIP code grouping", df, zipcode)}
  </main>
  <div class="legend">
    <strong>Mean AMR Index:</strong>
    <span><i class="swatch" style="background:#b2182b"></i>&lt;30</span>
    <span><i class="swatch" style="background:#ef8a62"></i>30-45</span>
    <span><i class="swatch" style="background:#f6c85f"></i>45-60</span>
    <span><i class="swatch" style="background:#7fbf7b"></i>60-75</span>
    <span><i class="swatch" style="background:#1a9850"></i>75+</span>
    <span>Circle size is proportional to number of buildings.</span>
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
    print(f"Buildings mapped: {len(df):,}")
    print(f"Neighborhood groups: {len(neighborhood):,}")
    print(f"ZIP code groups: {len(zipcode):,}")
    print(f"Saved: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
