"""Build an interactive map of AI-sample buildings colored by PLUTO cluster."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

try:
    from shapely import wkt
    from shapely.geometry import mapping
except ImportError:  # pragma: no cover - fallback keeps the script usable
    wkt = None
    mapping = None


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

INPUT_DATASET = PARENT_DIR / "output" / "amr_accessibility_logistic_dataset.csv"
CLUSTERED_CSV = OUTPUT_DIR / "pluto_clustered_ai_buildings.csv"
CLUSTER_SUMMARY_CSV = OUTPUT_DIR / "pluto_cluster_summary.csv"
HTML_OUTPUT = OUTPUT_DIR / "pluto_cluster_building_map.html"
GEOJSON_OUTPUT = OUTPUT_DIR / "pluto_cluster_building_map.geojson"

CLUSTER_COLORS = {
    0: "#4E79A7",
    1: "#F28E2B",
    2: "#59A14F",
    3: "#E15759",
    4: "#B07AA1",
    5: "#EDC948",
}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def as_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    parsed = pd.to_numeric(str(value).replace(",", "."), errors="coerce")
    if pd.isna(parsed):
        return None
    return float(parsed)


def format_percent(value: object) -> str:
    parsed = as_float(value)
    if parsed is None:
        return "n/a"
    if parsed <= 1:
        parsed *= 100
    return f"{parsed:.1f}%"


def parse_geometry(row: pd.Series) -> dict[str, object] | None:
    raw_geometry = row.get("the_geom")
    if wkt is not None and mapping is not None and isinstance(raw_geometry, str) and raw_geometry.strip():
        try:
            return mapping(wkt.loads(raw_geometry))
        except Exception:
            pass

    lon = as_float(row.get("centroid_lon"))
    lat = as_float(row.get("centroid_lat"))
    if lon is None or lat is None:
        lon = as_float(row.get("address_lon"))
        lat = as_float(row.get("address_lat"))
    if lon is None or lat is None:
        return None
    return {"type": "Point", "coordinates": [lon, lat]}


def attach_clusters_to_source() -> pd.DataFrame:
    source = read_required_csv(INPUT_DATASET)
    clusters = read_required_csv(CLUSTERED_CSV)
    if len(source) != len(clusters):
        raise ValueError(
            "The source dataset and clustered CSV have different row counts. "
            "Regenerate pluto_clustered_ai_buildings.csv before building the map."
        )

    source = source.reset_index(drop=True)
    clusters = clusters.reset_index(drop=True)
    cluster_columns = [
        "pluto_cluster",
        "pluto_cluster_label",
        "landuse_label",
        "bldgclass_group_label",
        "bldgclass_clean",
        "target_amr_accessible",
    ]
    for column in cluster_columns:
        if column in clusters.columns:
            source[column] = clusters[column]
    return source


def build_geojson(df: pd.DataFrame) -> dict[str, object]:
    features: list[dict[str, object]] = []
    for _, row in df.iterrows():
        geometry = parse_geometry(row)
        if geometry is None:
            continue

        cluster = int(row["pluto_cluster"])
        properties = {
            "cluster": cluster,
            "cluster_label": row.get("pluto_cluster_label", f"Cluster {cluster}"),
            "color": CLUSTER_COLORS.get(cluster, "#8C8C8C"),
            "borough": row.get("borough"),
            "neighborhood": row.get("neighborhood"),
            "bbl": row.get("map_pluto_bbl"),
            "full_address": row.get("full_address"),
            "landuse": row.get("landuse_label"),
            "building_class": row.get("bldgclass_clean"),
            "building_group": row.get("bldgclass_group_label"),
            "floors": as_float(row.get("numfloors")),
            "construction_year": as_float(row.get("construction_year")),
            "amr_accessible": int(row.get("target_amr_accessible", 0)),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": geometry,
                "properties": properties,
            }
        )

    return {"type": "FeatureCollection", "features": features}


def build_html(geojson: dict[str, object], summary: pd.DataFrame) -> str:
    summary = summary.sort_values("pluto_cluster")
    legend_items = []
    for _, row in summary.iterrows():
        cluster = int(row["pluto_cluster"])
        color = CLUSTER_COLORS.get(cluster, "#8C8C8C")
        legend_items.append(
            f"""
            <label class="legend-item">
              <input type="checkbox" class="cluster-toggle" value="{cluster}" checked>
              <span class="swatch" style="background:{color}"></span>
              <span>Cluster {cluster}</span>
              <small>{int(row["rows"])} buildings, {format_percent(row["accessible_rate"])} accessible</small>
            </label>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PLUTO Cluster Building Map</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map {{
      height: 100%;
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
    }}
    .panel {{
      position: absolute;
      z-index: 1000;
      top: 14px;
      right: 14px;
      width: min(340px, calc(100vw - 48px));
      max-height: calc(100vh - 42px);
      overflow: auto;
      background: rgba(255, 255, 255, 0.94);
      border: 1px solid #cbd5df;
      border-radius: 8px;
      box-shadow: 0 10px 28px rgba(15, 23, 42, 0.18);
      padding: 14px;
    }}
    .panel h1 {{
      margin: 0 0 8px;
      font-size: 18px;
      line-height: 1.2;
    }}
    .panel p {{
      margin: 0 0 12px;
      color: #475569;
      font-size: 13px;
      line-height: 1.35;
    }}
    .legend {{
      display: grid;
      gap: 8px;
    }}
    .legend-item {{
      display: grid;
      grid-template-columns: auto 18px 1fr;
      gap: 8px;
      align-items: center;
      font-size: 13px;
      cursor: pointer;
    }}
    .legend-item small {{
      grid-column: 3;
      color: #64748b;
      line-height: 1.2;
    }}
    .swatch {{
      width: 18px;
      height: 18px;
      border-radius: 3px;
      border: 1px solid rgba(15, 23, 42, 0.35);
    }}
    .leaflet-popup-content {{
      min-width: 230px;
    }}
    .popup-title {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .muted {{
      color: #64748b;
    }}
  </style>
</head>
<body>
  <div id="map"></div>
  <aside class="panel">
    <h1>PLUTO clusters</h1>
    <p>AI-sample buildings colored by the 6 unsupervised PLUTO/building-type clusters.</p>
    <div class="legend">
      {''.join(legend_items)}
    </div>
  </aside>
  <script>
    const geojsonData = {json.dumps(geojson)};
    const activeClusters = new Set([0, 1, 2, 3, 4, 5]);
    const map = L.map('map', {{ preferCanvas: true }}).setView([40.75, -73.98], 13);

    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      maxZoom: 20,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);

    function styleFeature(feature) {{
      const isActive = activeClusters.has(Number(feature.properties.cluster));
      return {{
        color: isActive ? '#1f2937' : '#9ca3af',
        weight: isActive ? 0.8 : 0.4,
        fillColor: isActive ? feature.properties.color : '#d1d5db',
        fillOpacity: isActive ? 0.72 : 0.12,
        opacity: isActive ? 0.95 : 0.2,
        radius: isActive ? 4 : 2
      }};
    }}

    function popupHtml(properties) {{
      const accessible = properties.amr_accessible === 1 ? 'Accessible' : 'Not accessible';
      return `
        <div class="popup-title">${{properties.cluster_label}}</div>
        <div><b>AMR label:</b> ${{accessible}}</div>
        <div><b>Address:</b> ${{properties.full_address || 'n/a'}}</div>
        <div><b>Neighborhood:</b> ${{properties.neighborhood || 'n/a'}}</div>
        <div><b>Land use:</b> ${{properties.landuse || 'n/a'}}</div>
        <div><b>Building group:</b> ${{properties.building_group || 'n/a'}}</div>
        <div><b>Building class:</b> ${{properties.building_class || 'n/a'}}</div>
        <div><b>Floors:</b> ${{properties.floors ?? 'n/a'}}</div>
        <div><b>Construction year:</b> ${{properties.construction_year ?? 'n/a'}}</div>
        <div class="muted"><b>BBL:</b> ${{properties.bbl || 'n/a'}}</div>
      `;
    }}

    const buildingLayer = L.geoJSON(geojsonData, {{
      style: styleFeature,
      pointToLayer: (feature, latlng) => L.circleMarker(latlng, styleFeature(feature)),
      onEachFeature: (feature, layer) => {{
        layer.bindPopup(popupHtml(feature.properties));
        layer.on('mouseover', () => layer.setStyle({{ weight: 2, fillOpacity: 0.9 }}));
        layer.on('mouseout', () => buildingLayer.resetStyle(layer));
      }}
    }}).addTo(map);

    const bounds = buildingLayer.getBounds();
    if (bounds.isValid()) {{
      map.fitBounds(bounds.pad(0.08));
    }}

    document.querySelectorAll('.cluster-toggle').forEach(input => {{
      input.addEventListener('change', () => {{
        const cluster = Number(input.value);
        if (input.checked) {{
          activeClusters.add(cluster);
        }} else {{
          activeClusters.delete(cluster);
        }}
        buildingLayer.setStyle(styleFeature);
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = attach_clusters_to_source()
    geojson = build_geojson(df)
    summary = read_required_csv(CLUSTER_SUMMARY_CSV)
    GEOJSON_OUTPUT.write_text(json.dumps(geojson), encoding="utf-8")
    HTML_OUTPUT.write_text(build_html(geojson, summary), encoding="utf-8")
    print(f"Buildings mapped: {len(geojson['features'])}")
    print(f"Saved GeoJSON: {GEOJSON_OUTPUT}")
    print(f"Saved HTML map: {HTML_OUTPUT}")


if __name__ == "__main__":
    main()
