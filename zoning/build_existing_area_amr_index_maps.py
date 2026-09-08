"""Build comparison maps for AMR Index aggregated by existing areas.

The map uses the same building-level non-AI AMR Index used for the new
AMR-friendly zones. It colors each building footprint by the mean AMR Index of
its existing neighborhood or ZIP code, so the result can be compared directly
with the data-driven AMR zones.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely import wkt
from shapely.geometry import mapping


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
AMR_INDEX_DIR = PROJECT_ROOT / "AMR_index_no_AI"
LOGISTIC_DIR = PROJECT_ROOT / "logistic regression"
OUTPUT_DIR = SCRIPT_DIR / "output"
NTA_CSV = PROJECT_ROOT / "dataset" / "archivio" / "NTA.csv"

INDEX_CSV = AMR_INDEX_DIR / "output" / "building_level_no_ai_amr_index.csv"
SOURCE_CSV = LOGISTIC_DIR / "output" / "amr_accessibility_logistic_dataset.csv"
HTML_OUTPUT = OUTPUT_DIR / "existing_area_amr_index_map.html"
BUILDING_HTML_OUTPUT = OUTPUT_DIR / "building_amr_index_map.html"
MIN_AREA_BUILDINGS = 20
NYC_ZIP_BOUNDARIES_URL = "https://data.cityofnewyork.us/api/v3/views/35j5-n34v/query.geojson?accessType=DOWNLOAD"
SELECTED_NEIGHBORHOOD_BOUNDARIES = [
    "Midtown South-Flatiron-Union Square",
    "Midtown-Times Square",
    "Gramercy",
    "Hell's Kitchen",
    "Chelsea-Hudson Yards",
    "Upper West Side-Lincoln Square",
    "Upper West Side (Central)",
]
SELECTED_ZIP_BOUNDARIES = [
    "10018",
    "10010",
    "10016",
    "10001",
    "10019",
    "10036",
    "10003",
    "10011",
    "10023",
    "10024",
    "10025",
]


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


def selected_neighborhood_boundaries() -> list[dict]:
    if not NTA_CSV.exists():
        raise FileNotFoundError(f"Missing local NTA boundary file: {NTA_CSV}")

    selected = set(SELECTED_NEIGHBORHOOD_BOUNDARIES)
    nta = pd.read_csv(NTA_CSV, usecols=["BoroName", "NTAName", "the_geom"], low_memory=False)
    nta["BoroName"] = nta["BoroName"].fillna("").astype(str).str.strip()
    nta["NTAName"] = nta["NTAName"].fillna("").astype(str).str.strip()
    nta = nta[(nta["BoroName"] == "Manhattan") & (nta["NTAName"].isin(selected))].copy()

    boundaries = []
    for _, row in nta.iterrows():
        geom = parse_geom(row["the_geom"])
        if geom is None:
            continue
        boundaries.append(
            {
                "area": row["NTAName"],
                "geometry": geom,
            }
        )

    missing = sorted(selected - set(nta["NTAName"]))
    if missing:
        raise ValueError(f"Missing selected NTA boundaries: {missing}")

    return boundaries


def neighborhood_boundary_feature_collection(boundaries: list[dict]) -> dict:
    features = []
    for boundary in boundaries:
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(boundary["geometry"]),
                "properties": {
                    "area": boundary["area"],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def assign_selected_neighborhoods_from_boundaries(df: pd.DataFrame, boundaries: list[dict]) -> pd.DataFrame:
    work = df.copy()
    work["selected_neighborhood"] = pd.NA

    for idx, geom in work["geometry"].items():
        point = geom.representative_point()
        for boundary in boundaries:
            if boundary["geometry"].covers(point):
                work.at[idx, "selected_neighborhood"] = boundary["area"]
                break

    return work.dropna(subset=["selected_neighborhood"]).copy()


def label_payload(groups: pd.DataFrame) -> list[dict]:
    labels = []
    for _, row in groups.iterrows():
        labels.append(
            {
                "area": str(row.area_name),
                "lat": float(row.lat),
                "lon": float(row.lon),
                "mean_amr_index": round(float(row.mean_amr_index), 1),
                "actual_accessibility_rate": round(float(row.actual_accessibility_rate) * 100, 1),
                "n_buildings": int(row.n_buildings),
            }
        )
    return labels


def build_html(payload: dict, title: str, subtitle: str, single_layer: str | None = None) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
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
      max-width: 560px;
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
      background: rgba(255, 255, 255, 0.9);
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
    <h1>{title}</h1>
    <p>{subtitle}</p>
  </div>
  <div class="legend">
    <strong>Mean AMR Index</strong>
    <div class="gradient"></div>
    <div class="scale-row"><span>Low</span><span>High</span></div>
    <div style="margin-top:8px;">Hover an area to fade the others.</div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    const singleLayer = {json.dumps(single_layer)};
    const officialBoundaries = {{
      zipcode: {json.dumps(NYC_ZIP_BOUNDARIES_URL)}
    }};
    const selectedBoundaries = {{
      zipcode: {json.dumps(SELECTED_ZIP_BOUNDARIES)}
    }};

    function colorByIndex(value) {{
      if (value < 30) return '#b2182b';
      if (value < 45) return '#ef8a62';
      if (value < 60) return '#f6c85f';
      if (value < 75) return '#7fbf7b';
      return '#1a9850';
    }}

    const map = L.map('map', {{
      preferCanvas: true,
      zoomSnap: 0.25,
      zoomDelta: 0.25,
      wheelPxPerZoomLevel: 160
    }}).setView(payload.center, 13);
    map.createPane('boundaryPane');
    map.getPane('boundaryPane').style.zIndex = 625;
    map.createPane('labelPane');
    map.getPane('labelPane').style.zIndex = 700;
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 16,
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
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
        fillOpacity: 0.8,
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

    function attachTooltip(activeLayer) {{
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
          html: `${{item.mean_amr_index.toFixed(1)}}`,
          iconSize: null
        }});
        L.marker([item.lat, item.lon], {{ icon, pane: 'labelPane' }}).bindTooltip(
          `<b>${{item.area}}</b><br>` +
          `Mean AMR Index: ${{item.mean_amr_index.toFixed(1)}}<br>` +
          `Actual accessible rate: ${{item.actual_accessibility_rate.toFixed(1)}}%<br>` +
          `Buildings: ${{item.n_buildings}}`
        ).addTo(group);
      }}
      return group;
    }}

    function makeGeoLayer(geojson) {{
      const layer = L.geoJSON(null, {{ style: styleFeature }});
      layer.options.onEachFeature = attachTooltip(layer);
      layer.addData(geojson);
      return layer;
    }}

    function makeBoundaryLayer(geojson) {{
      return L.geoJSON(geojson, {{
        interactive: false,
        pane: 'boundaryPane',
        style: {{
          color: '#111827',
          weight: 3.0,
          opacity: 0.95,
          fillOpacity: 0,
          dashArray: '7 4'
        }}
      }});
    }}

    function normalizeName(value) {{
      return String(value || '')
        .replace(/[–—]/g, '-')
        .replace(/\\s+/g, ' ')
        .trim()
        .toLowerCase();
    }}

    function getBoundaryName(feature, kind) {{
      const props = feature.properties || {{}};
      if (kind === 'zipcode') return props.zcta5 || props.ZCTA5 || props.zipcode || props.ZIPCODE || props.ZipCode;
      return props.ntaname || props.NTAName || props.NTANAME || props.name || props.Name;
    }}

    function filterOfficialBoundaries(geojson, kind) {{
      const selected = new Set(selectedBoundaries[kind].map(normalizeName));
      return {{
        type: 'FeatureCollection',
        features: (geojson.features || []).filter(feature => {{
          const boundaryName = normalizeName(getBoundaryName(feature, kind));
          return selected.has(boundaryName);
        }})
      }};
    }}

    function addOfficialBoundaryLayer(url, targetGroup, kind) {{
      fetch(url)
        .then(response => {{
          if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
          return response.json();
        }})
        .then(geojson => {{
          makeBoundaryLayer(filterOfficialBoundaries(geojson, kind)).addTo(targetGroup);
        }})
        .catch(error => {{
          console.warn('Official boundary layer could not be loaded:', error);
        }});
    }}

    const neighborhoodLayer = makeGeoLayer(payload.neighborhoodGeojson);
    const zipcodeLayer = makeGeoLayer(payload.zipcodeGeojson);
    const neighborhoodBoundaryLayer = makeBoundaryLayer(payload.neighborhoodBoundaryGeojson);
    const neighborhoodLabels = makeLabels(payload.neighborhoodLabels);
    const zipcodeLabels = makeLabels(payload.zipcodeLabels);

    const neighborhoodGroup = L.layerGroup([neighborhoodLayer, neighborhoodBoundaryLayer, neighborhoodLabels]);
    const zipcodeGroup = L.layerGroup([zipcodeLayer, zipcodeLabels]);
    addOfficialBoundaryLayer(officialBoundaries.zipcode, zipcodeGroup, 'zipcode');

    if (singleLayer === 'zipcode') {{
      zipcodeGroup.addTo(map);
      map.fitBounds(zipcodeLayer.getBounds(), {{ padding: [24, 24] }});
    }} else {{
      neighborhoodGroup.addTo(map);
      map.fitBounds(neighborhoodLayer.getBounds(), {{ padding: [24, 24] }});
    }}

    if (!singleLayer) {{
      L.control.layers(
        null,
        {{
          'Neighborhood AMR Index': neighborhoodGroup,
          'ZIP code AMR Index': zipcodeGroup
        }},
        {{ collapsed: false }}
      ).addTo(map);
    }}
  </script>
</body>
</html>
"""


def building_feature_collection(df: pd.DataFrame) -> dict:
    features = []
    for _, row in df.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": {
                    "amr_index": round(float(row.baseline_amr_index), 2),
                    "amr_probability": round(float(row.baseline_amr_probability) * 100, 2),
                    "target_amr_accessible": None
                    if pd.isna(row.target_amr_accessible)
                    else int(row.target_amr_accessible),
                    "neighborhood": "Unknown" if pd.isna(row.neighborhood) else str(row.neighborhood),
                    "zipcode": "Unknown" if pd.isna(row.zipcode) else str(row.zipcode),
                    "address": "Unknown" if pd.isna(row.full_address) else str(row.full_address),
                    "bldgclass": "Unknown" if pd.isna(row.bldgclass) else str(row.bldgclass),
                    "landuse": "Unknown" if pd.isna(row.landuse) else str(row.landuse),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def build_building_html(df: pd.DataFrame) -> str:
    payload = {
        "center": [float(df["centroid_lat"].mean()), float(df["centroid_lon"].mean())],
        "geojson": building_feature_collection(df),
    }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Building-Level AMR Index Map</title>
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
      max-width: 560px;
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
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-box">
    <h1>Building-Level AMR Index</h1>
    <p>Each building footprint is colored by its own predicted AMR Index, without neighborhood or ZIP aggregation.</p>
  </div>
  <div class="legend">
    <strong>Building AMR Index</strong>
    <div class="gradient"></div>
    <div class="scale-row"><span>Low</span><span>High</span></div>
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

    const map = L.map('map', {{
      preferCanvas: true,
      zoomSnap: 0.25,
      zoomDelta: 0.25,
      wheelPxPerZoomLevel: 160
    }}).setView(payload.center, 13);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      maxZoom: 16,
      attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
    }}).addTo(map);

    const layer = L.geoJSON(payload.geojson, {{
      style: feature => ({{
        color: '#26313a',
        weight: 0.35,
        fillColor: colorByIndex(feature.properties.amr_index),
        fillOpacity: 0.66,
        opacity: 0.68
      }}),
      onEachFeature: (feature, layer) => {{
        const p = feature.properties;
        const actual = p.target_amr_accessible === null ? 'Unknown' : (p.target_amr_accessible === 1 ? 'Accessible' : 'Not accessible');
        layer.bindTooltip(
          `<b>${{p.address}}</b><br>` +
          `AMR Index: ${{p.amr_index.toFixed(1)}}<br>` +
          `Predicted probability: ${{p.amr_probability.toFixed(1)}}%<br>` +
          `Actual AI label: ${{actual}}<br>` +
          `Neighborhood: ${{p.neighborhood}}<br>` +
          `ZIP: ${{p.zipcode}}<br>` +
          `Building class: ${{p.bldgclass}}<br>` +
          `Land use: ${{p.landuse}}`,
          {{ sticky: true }}
        );
        layer.on({{
          mouseover: () => {{
            layer.setStyle({{
              color: '#111827',
              weight: 1.2,
              fillOpacity: 0.86,
              opacity: 1
            }});
            if (layer.bringToFront) layer.bringToFront();
          }},
          mouseout: () => {{
            layer.setStyle({{
              color: '#26313a',
              weight: 0.35,
              fillColor: colorByIndex(p.amr_index),
              fillOpacity: 0.66,
              opacity: 0.68
            }});
          }}
        }});
      }}
    }}).addTo(map);

    map.fitBounds(layer.getBounds(), {{ padding: [24, 24] }});
  </script>
</body>
</html>
"""


def build_payload(df: pd.DataFrame) -> dict:
    neighborhood_boundaries = selected_neighborhood_boundaries()
    neighborhood_df = assign_selected_neighborhoods_from_boundaries(df, neighborhood_boundaries)
    neighborhood = aggregate(neighborhood_df, "selected_neighborhood")
    zipcode = aggregate(df, "zipcode", exclude_unknown=True)
    return {
        "center": [float(df["centroid_lat"].mean()), float(df["centroid_lon"].mean())],
        "neighborhoodGeojson": feature_collection(neighborhood_df, neighborhood, "selected_neighborhood"),
        "zipcodeGeojson": feature_collection(df, zipcode, "zipcode", exclude_unknown=True),
        "neighborhoodBoundaryGeojson": neighborhood_boundary_feature_collection(neighborhood_boundaries),
        "neighborhoodLabels": label_payload(neighborhood),
        "zipcodeLabels": label_payload(zipcode),
        "neighborhoodCount": int(len(neighborhood)),
        "zipcodeCount": int(len(zipcode)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_data()
    payload = build_payload(df)

    HTML_OUTPUT.write_text(
        build_html(
            payload,
            "AMR Index by Existing Geographic Areas",
            "Comparison map: switch between neighborhood and ZIP code aggregations.",
        ),
        encoding="utf-8",
    )
    BUILDING_HTML_OUTPUT.write_text(build_building_html(df), encoding="utf-8")

    print(f"Buildings with footprint geometry: {len(df):,}")
    print(f"Neighborhood groups: {payload['neighborhoodCount']:,}")
    print(f"ZIP code groups: {payload['zipcodeCount']:,}")
    print(f"Saved combined map: {HTML_OUTPUT}")
    print(f"Saved building-level map: {BUILDING_HTML_OUTPUT}")


if __name__ == "__main__":
    main()
