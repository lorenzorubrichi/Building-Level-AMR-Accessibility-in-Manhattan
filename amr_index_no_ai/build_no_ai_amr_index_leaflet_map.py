"""Build a Leaflet basemap overlay for the non-AI AMR Index.

This map overlays building footprints on a simple real-world basemap. Buildings
in the same existing area receive the same color. The color follows a red to
green scale based on the area's mean AMR Index.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

INDEX_CSV = OUTPUT_DIR / "building_level_no_ai_amr_index.csv"
SOURCE_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
HTML_OUTPUT = OUTPUT_DIR / "no_ai_amr_index_leaflet_overlay_map.html"
MIN_AREA_BUILDINGS = 20


def parse_geom(value: object):
    if pd.isna(value):
        return None
    try:
        geom = wkt.loads(str(value))
    except Exception:
        return None
    return None if geom.is_empty else geom


def read_data() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing AMR index file: {INDEX_CSV}")
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Missing source geometry file: {SOURCE_CSV}")

    index_df = pd.read_csv(INDEX_CSV, low_memory=False)
    geom_df = pd.read_csv(SOURCE_CSV, usecols=["the_geom"], low_memory=False)
    if len(index_df) != len(geom_df):
        raise ValueError(f"Row mismatch: {len(index_df)} index rows vs {len(geom_df)} geometry rows")

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
    return out[out["n_buildings"] >= MIN_AREA_BUILDINGS].copy()


def feature_collection(df: pd.DataFrame, groups: pd.DataFrame, area_col: str, exclude_unknown: bool = False) -> dict:
    group_stats = groups.set_index("area_name").to_dict(orient="index")
    valid_groups = set(group_stats)
    features = []
    for _, row in df.iterrows():
        area = "Unknown" if pd.isna(row[area_col]) else str(row[area_col])
        if exclude_unknown and area.lower() == "unknown":
            continue
        if area not in valid_groups:
            continue

        stats = group_stats[area]
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "area": area,
                    "mean_amr_index": round(float(stats["mean_amr_index"]), 2),
                    "actual_accessibility_rate": round(float(stats["actual_accessibility_rate"]) * 100, 2),
                    "n_buildings": int(stats["n_buildings"]),
                    "building_amr_index": round(float(row.baseline_amr_index), 2),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def labels(groups: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in groups.iterrows():
        out.append(
            {
                "area": str(row.area_name),
                "lat": float(row.lat),
                "lon": float(row.lon),
                "mean_amr_index": round(float(row.mean_amr_index), 1),
                "n_buildings": int(row.n_buildings),
            }
        )
    return out


def build_html(df: pd.DataFrame, neighborhood: pd.DataFrame, zipcode: pd.DataFrame) -> str:
    center_lat = float(df["centroid_lat"].mean())
    center_lon = float(df["centroid_lon"].mean())
    neighborhood_geojson = feature_collection(df, neighborhood, "neighborhood")
    zipcode_geojson = feature_collection(df, zipcode, "zipcode", exclude_unknown=True)

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
  <title>Non-AI AMR Index Basemap Overlay</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{
      height: 100%;
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      color: #172026;
    }}
    #map {{
      height: 100%;
      width: 100%;
    }}
    .title-box {{
      position: absolute;
      top: 14px;
      left: 54px;
      z-index: 900;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #cfd6df;
      border-radius: 6px;
      padding: 10px 12px;
      box-shadow: 0 8px 22px rgba(23, 32, 42, 0.15);
      max-width: 520px;
    }}
    .title-box h1 {{
      margin: 0 0 4px;
      font-size: 18px;
    }}
    .title-box p {{
      margin: 0;
      font-size: 13px;
      color: #53606b;
    }}
    .legend {{
      position: absolute;
      right: 14px;
      bottom: 24px;
      z-index: 900;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #cfd6df;
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 12px;
      box-shadow: 0 8px 22px rgba(23, 32, 42, 0.15);
    }}
    .gradient {{
      width: 190px;
      height: 13px;
      margin: 6px 0;
      background: linear-gradient(90deg, #b2182b, #ef8a62, #f6c85f, #7fbf7b, #1a9850);
      border: 1px solid #172026;
    }}
    .scale-row {{
      display: flex;
      justify-content: space-between;
      width: 192px;
    }}
    .amr-label {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid #1f2933;
      border-radius: 4px;
      color: #172026;
      font-weight: 700;
      font-size: 12px;
      padding: 2px 5px;
      white-space: nowrap;
      box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-box">
    <h1>Non-AI AMR Index by Existing Areas</h1>
    <p>Building footprints are overlaid on a real basemap. All buildings in the same zone share the same red-green color based on that zone's mean AMR Index.</p>
  </div>
  <div class="legend">
    <strong>Mean AMR Index</strong>
    <div class="gradient"></div>
    <div class="scale-row"><span>Low</span><span>High</span></div>
    <div style="margin-top:8px;">Layer control: switch between neighborhood and ZIP code.</div>
  </div>
  <script>
    const payload = {json.dumps(payload)};

    function colorByIndex(value) {{
      if (value < 30) return '#b2182b';
      if (value < 45) return '#ef8a62';
      if (value < 60) return '#f6c85f';
      if (value < 75) return '#7fbf7b';
      return '#1a9850';
    }}

    const map = L.map('map', {{ preferCanvas: true }}).setView(payload.center, 13);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    function styleFeature(feature) {{
      return {{
        color: '#26313a',
        weight: 0.45,
        fillColor: colorByIndex(feature.properties.mean_amr_index),
        fillOpacity: 0.62,
        opacity: 0.72
      }};
    }}

    function mutedFeatureStyle() {{
      return {{
        color: '#8a939b',
        weight: 0.25,
        fillColor: '#b8c0c7',
        fillOpacity: 0.18,
        opacity: 0.28
      }};
    }}

    function highlightFeatureStyle(feature) {{
      return {{
        color: '#111827',
        weight: 1.4,
        fillColor: colorByIndex(feature.properties.mean_amr_index),
        fillOpacity: 0.78,
        opacity: 1
      }};
    }}

    function highlightArea(activeLayer, activeArea) {{
      activeLayer.eachLayer(layer => {{
        const area = layer.feature && layer.feature.properties.area;
        if (area === activeArea) {{
          layer.setStyle(highlightFeatureStyle(layer.feature));
          if (layer.bringToFront) layer.bringToFront();
        }} else {{
          layer.setStyle(mutedFeatureStyle());
        }}
      }});
    }}

    function resetAreaHighlight(activeLayer) {{
      activeLayer.eachLayer(layer => {{
        layer.setStyle(styleFeature(layer.feature));
      }});
    }}

    function popup(activeLayer) {{
      return function(feature, layer) {{
      const p = feature.properties;
      layer.bindTooltip(
        `<b>${{p.area}}</b><br>` +
        `Mean AMR Index: ${{p.mean_amr_index.toFixed(1)}}<br>` +
        `Buildings in area: ${{p.n_buildings}}<br>` +
        `Actual accessible rate: ${{p.actual_accessibility_rate.toFixed(1)}}%<br>` +
        `Building AMR Index: ${{p.building_amr_index.toFixed(1)}}`,
        {{ sticky: true }}
      );
      layer.on({{
        mouseover: () => highlightArea(activeLayer, p.area),
        mouseout: () => resetAreaHighlight(activeLayer)
      }});
      }};
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

    const neighborhoodLayer = L.geoJSON(null, {{ style: styleFeature }});
    neighborhoodLayer.options.onEachFeature = popup(neighborhoodLayer);
    neighborhoodLayer.addData(payload.neighborhoodGeojson);

    const zipcodeLayer = L.geoJSON(null, {{ style: styleFeature }});
    zipcodeLayer.options.onEachFeature = popup(zipcodeLayer);
    zipcodeLayer.addData(payload.zipcodeGeojson);
    const neighborhoodLabels = makeLabels(payload.neighborhoodLabels);
    const zipcodeLabels = makeLabels(payload.zipcodeLabels);

    const neighborhoodGroup = L.layerGroup([neighborhoodLayer, neighborhoodLabels]).addTo(map);
    const zipcodeGroup = L.layerGroup([zipcodeLayer, zipcodeLabels]);

    L.control.layers(
      null,
      {{
        'Neighborhood AMR Index': neighborhoodGroup,
        'ZIP code AMR Index': zipcodeGroup
      }},
      {{ collapsed: false }}
    ).addTo(map);

    map.fitBounds(neighborhoodLayer.getBounds(), {{ padding: [24, 24] }});
  </script>
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
