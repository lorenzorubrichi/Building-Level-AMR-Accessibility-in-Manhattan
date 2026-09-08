"""Build approximate boundary polygons for the non-AI AMR Index.

This creates data-derived boundary polygons from the spatial distribution of
buildings in each existing area. These are not official administrative
boundaries; they are concave-hull approximations useful for visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import shapely
from shapely.geometry import MultiPoint, Point, mapping


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

INDEX_CSV = OUTPUT_DIR / "building_level_no_ai_amr_index.csv"
HTML_OUTPUT = OUTPUT_DIR / "no_ai_amr_index_boundary_overlay_map.html"
MIN_AREA_BUILDINGS = 20
CONCAVE_RATIO = 0.18
BOUNDARY_BUFFER_DEGREES = 0.00055


def read_data() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing AMR index file: {INDEX_CSV}")

    df = pd.read_csv(INDEX_CSV, low_memory=False)
    df["baseline_amr_index"] = pd.to_numeric(df["baseline_amr_index"], errors="coerce")
    df["target_amr_accessible"] = pd.to_numeric(df["target_amr_accessible"], errors="coerce")
    df["centroid_lon"] = pd.to_numeric(df["centroid_lon"], errors="coerce")
    df["centroid_lat"] = pd.to_numeric(df["centroid_lat"], errors="coerce")
    return df.dropna(subset=["baseline_amr_index", "centroid_lon", "centroid_lat"]).copy()


def red_green(value: float) -> str:
    if value < 30:
        return "#b2182b"
    if value < 45:
        return "#ef8a62"
    if value < 60:
        return "#f6c85f"
    if value < 75:
        return "#7fbf7b"
    return "#1a9850"


def aggregate_with_boundaries(df: pd.DataFrame, area_col: str, exclude_unknown: bool = False) -> tuple[pd.DataFrame, dict]:
    work = df.copy()
    work[area_col] = work[area_col].fillna("Unknown").astype(str)
    if exclude_unknown:
        work = work[work[area_col].str.lower() != "unknown"]

    rows = []
    features = []
    for area, group in work.groupby(area_col, dropna=False):
        if len(group) < MIN_AREA_BUILDINGS:
            continue

        points = [Point(float(lon), float(lat)) for lon, lat in zip(group["centroid_lon"], group["centroid_lat"])]
        multipoint = MultiPoint(points)

        if len(points) >= 4:
            geom = shapely.concave_hull(multipoint, ratio=CONCAVE_RATIO, allow_holes=False)
        else:
            geom = multipoint.convex_hull

        if geom.geom_type in {"Point", "LineString"}:
            geom = geom.buffer(BOUNDARY_BUFFER_DEGREES)
        else:
            geom = geom.buffer(BOUNDARY_BUFFER_DEGREES).buffer(-BOUNDARY_BUFFER_DEGREES * 0.35)

        mean_index = float(group["baseline_amr_index"].mean())
        actual_rate = float(group["target_amr_accessible"].mean()) * 100
        lon = float(group["centroid_lon"].mean())
        lat = float(group["centroid_lat"].mean())
        area_name = str(area)

        rows.append(
            {
                "area_name": area_name,
                "n_buildings": int(len(group)),
                "mean_amr_index": mean_index,
                "actual_accessibility_rate": actual_rate,
                "lon": lon,
                "lat": lat,
            }
        )
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(geom),
                "properties": {
                    "area": area_name,
                    "n_buildings": int(len(group)),
                    "mean_amr_index": round(mean_index, 2),
                    "actual_accessibility_rate": round(actual_rate, 2),
                    "color": red_green(mean_index),
                },
            }
        )

    return pd.DataFrame(rows), {"type": "FeatureCollection", "features": features}


def labels(groups: pd.DataFrame) -> list[dict]:
    return [
        {
            "area": str(row.area_name),
            "lat": float(row.lat),
            "lon": float(row.lon),
            "mean_amr_index": round(float(row.mean_amr_index), 1),
            "n_buildings": int(row.n_buildings),
        }
        for _, row in groups.iterrows()
    ]


def build_html(df: pd.DataFrame, neighborhood: pd.DataFrame, neighborhood_geojson: dict, zipcode: pd.DataFrame, zipcode_geojson: dict) -> str:
    center_lat = float(df["centroid_lat"].mean())
    center_lon = float(df["centroid_lon"].mean())

    payload = {
        "center": [center_lat, center_lon],
        "neighborhoodGeojson": neighborhood_geojson,
        "zipcodeGeojson": zipcode_geojson,
        "neighborhoodLabels": labels(neighborhood),
        "zipcodeLabels": labels(zipcode),
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Non-AI AMR Index Boundary Map</title>
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
    .gradient {{
      width: 190px; height: 13px; margin: 6px 0;
      background: linear-gradient(90deg, #b2182b, #ef8a62, #f6c85f, #7fbf7b, #1a9850);
      border: 1px solid #172026;
    }}
    .scale-row {{ display: flex; justify-content: space-between; width: 192px; }}
    .amr-label {{
      background: rgba(255,255,255,0.9); border: 1px solid #1f2933;
      border-radius: 4px; color: #172026; font-weight: 700;
      font-size: 12px; padding: 2px 5px; white-space: nowrap;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-box">
    <h1>Non-AI AMR Index: Approximate Area Boundaries</h1>
    <p>Polygons are generated from building locations using concave hulls. They approximate the sampled area covered by each neighborhood or ZIP code and are colored by mean AMR Index.</p>
  </div>
  <div class="legend">
    <strong>Mean AMR Index</strong>
    <div class="gradient"></div>
    <div class="scale-row"><span>Low</span><span>High</span></div>
    <div style="margin-top:8px;">Layer control: switch between neighborhood and ZIP code.</div>
  </div>
  <script>
    const payload = {json.dumps(payload)};

    const map = L.map('map').setView(payload.center, 13);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    function styleFeature(feature) {{
      return {{
        color: '#172026',
        weight: 2.0,
        fillColor: feature.properties.color,
        fillOpacity: 0.46,
        opacity: 0.9
      }};
    }}

    function bindPopup(feature, layer) {{
      const p = feature.properties;
      layer.bindTooltip(
        `<b>${{p.area}}</b><br>` +
        `Mean AMR Index: ${{p.mean_amr_index.toFixed(1)}}<br>` +
        `Buildings: ${{p.n_buildings}}<br>` +
        `Actual accessible rate: ${{p.actual_accessibility_rate.toFixed(1)}}%`,
        {{ sticky: true }}
      );
    }}

    function makeLabels(items) {{
      const group = L.layerGroup();
      for (const item of items) {{
        const icon = L.divIcon({{
          className: 'amr-label',
          html: `${{item.mean_amr_index.toFixed(1)}}%`,
          iconSize: null
        }});
        L.marker([item.lat, item.lon], {{ icon }}).bindTooltip(
          `<b>${{item.area}}</b><br>Mean AMR Index: ${{item.mean_amr_index.toFixed(1)}}<br>Buildings: ${{item.n_buildings}}`
        ).addTo(group);
      }}
      return group;
    }}

    const neighborhoodPolygons = L.geoJSON(payload.neighborhoodGeojson, {{
      style: styleFeature,
      onEachFeature: bindPopup
    }});
    const zipcodePolygons = L.geoJSON(payload.zipcodeGeojson, {{
      style: styleFeature,
      onEachFeature: bindPopup
    }});
    const neighborhoodLabels = makeLabels(payload.neighborhoodLabels);
    const zipcodeLabels = makeLabels(payload.zipcodeLabels);

    const neighborhoodGroup = L.layerGroup([neighborhoodPolygons, neighborhoodLabels]).addTo(map);
    const zipcodeGroup = L.layerGroup([zipcodePolygons, zipcodeLabels]);

    L.control.layers(null, {{
      'Neighborhood approximate boundaries': neighborhoodGroup,
      'ZIP code approximate boundaries': zipcodeGroup
    }}, {{ collapsed: false }}).addTo(map);

    map.fitBounds(neighborhoodPolygons.getBounds(), {{ padding: [24, 24] }});
  </script>
</body>
</html>
"""


def main() -> None:
    df = read_data()
    neighborhood, neighborhood_geojson = aggregate_with_boundaries(df, "neighborhood")
    zipcode, zipcode_geojson = aggregate_with_boundaries(df, "zipcode", exclude_unknown=True)
    HTML_OUTPUT.write_text(build_html(df, neighborhood, neighborhood_geojson, zipcode, zipcode_geojson), encoding="utf-8")
    print(f"Buildings used: {len(df):,}")
    print(f"Neighborhood boundary polygons: {len(neighborhood):,}")
    print(f"ZIP code boundary polygons: {len(zipcode):,}")
    print(f"Saved: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
