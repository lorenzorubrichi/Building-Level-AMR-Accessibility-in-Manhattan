"""Apply the non-AI AMR Index model to Manhattan buildings.

Outputs:
- building_level_amr_index_estimated_manhattan.csv: one row per Manhattan building with the
  estimated AMR accessibility probability and AMR Index.
- square_grid_amr_index_manhattan.html: Manhattan square-cell AMR Index map.
- hex_grid_amr_index_manhattan.html: Manhattan hex-cell AMR Index map.

The model is trained on the 7,013 AI-labeled buildings, using only non-AI
features, and then applied to Manhattan buildings in the complete last-meter dataset.
"""
###C:\Users\loren\AppData\Local\Programs\Python\Python313\python.exe "C:\Users\loren\Desktop\RPI_DOC\lavori\zoning\zoning_all\build_zoning_all.py"


from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ZONING_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = ZONING_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

TRAIN_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
FULL_DATASET_CSV = PROJECT_ROOT / "dataset" / "Model" / "complete_last_meter_dataset_extended.csv"
NEIGHBORHOODS_GEOJSON = PROJECT_ROOT / "dataset" / "rawdata" / "nyc_neighborhoods.geojson"

STUDY_BOROUGH = "Manhattan"
STUDY_AREA_LABEL = "Manhattan"

BUILDING_OUTPUT_CSV = OUTPUT_DIR / "building_level_amr_index_estimated_manhattan.csv"
SQUARE_SUMMARY_CSV = OUTPUT_DIR / "square_grid_amr_index_manhattan_summary.csv"
SQUARE_GEOJSON = OUTPUT_DIR / "square_grid_amr_index_manhattan.geojson"
SQUARE_MAP_HTML = OUTPUT_DIR / "square_grid_amr_index_manhattan.html"
AGG_SQUARE_SUMMARY_CSV = OUTPUT_DIR / "aggregated_square_zones_amr_index_manhattan_summary.csv"
AGG_SQUARE_GEOJSON = OUTPUT_DIR / "aggregated_square_zones_amr_index_manhattan.geojson"
AGG_SQUARE_MAP_HTML = OUTPUT_DIR / "aggregated_square_zones_amr_index_manhattan.html"
HEX_SUMMARY_CSV = OUTPUT_DIR / "hex_grid_amr_index_manhattan_summary.csv"
HEX_GEOJSON = OUTPUT_DIR / "hex_grid_amr_index_manhattan.geojson"
HEX_MAP_HTML = OUTPUT_DIR / "hex_grid_amr_index_manhattan.html"
AGG_HEX_SUMMARY_CSV = OUTPUT_DIR / "aggregated_hex_zones_amr_index_manhattan_summary.csv"
AGG_HEX_GEOJSON = OUTPUT_DIR / "aggregated_hex_zones_amr_index_manhattan.geojson"
AGG_HEX_MAP_HTML = OUTPUT_DIR / "aggregated_hex_zones_amr_index_manhattan.html"
REPORT_TXT = OUTPUT_DIR / "zoning_manhattan_report.txt"

RANDOM_STATE = 42
CHUNK_SIZE = 100_000
SQUARE_CELL_SIZE_M = 200
HEX_RADIUS_M = 500
MIN_BUILDINGS_PER_CELL = 30

NUMERIC_FEATURES = [
    "numfloors_clean",
    "height_roof_clean",
    "construction_year_clean",
    "ground_elevation_clean",
    "shape_area_clean",
    "distance_to_sidewalk_m_clean",
]

CATEGORICAL_FEATURES = [
    "landuse_clean",
    "bldgclass_clean",
    "bldgclass_group",
]

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


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(index=df.index, dtype=object)


def clean_zipcode(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.round().astype("Int64").astype(str).replace("<NA>", "Unknown")


def clean_landuse(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.round().astype("Int64").astype(str).replace("<NA>", "Unknown")


def clean_bldgclass(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str).str.strip().str.upper().replace({"": "Unknown", "NAN": "Unknown"})


def clean_bldgclass_group(series: pd.Series) -> pd.Series:
    cleaned = clean_bldgclass(series)
    return cleaned.str[0].where(cleaned != "Unknown", "Unknown")


def read_neighborhood_outlines() -> dict | None:
    if not NEIGHBORHOODS_GEOJSON.exists():
        return None
    data = json.loads(NEIGHBORHOODS_GEOJSON.read_text(encoding="utf-8"))
    # Drop the duplicate WKT geometry property to keep the generated HTML smaller.
    for feature in data.get("features", []):
        feature.get("properties", {}).pop("the_geom", None)
    return data


def clean_construction_year(series: pd.Series) -> pd.Series:
    values = numeric(series)
    values = values.where(~values.between(1.0, 2.5), values * 1000)
    return values.where(values.between(1600, 2030))


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    landuse = first_existing(raw, ["pluto_raw__landuse", "landuse", "landuse_x", "landuse_y"])
    bldgclass = first_existing(raw, ["pluto_raw__bldgclass", "bldgclass"])
    features = pd.DataFrame(index=raw.index)
    features["borough"] = first_existing(raw, ["borough"]).fillna("Unknown").astype(str)
    features["neighborhood"] = first_existing(raw, ["neighborhood", "nta_name", "building_raw__nta_name"]).fillna(
        "Unknown"
    ).astype(str)
    features["zipcode_clean"] = clean_zipcode(first_existing(raw, ["zipcode"]))
    features["landuse_clean"] = clean_landuse(landuse)
    features["landuse_label"] = features["landuse_clean"].map(LANDUSE_LABELS).fillna("Unknown / other")
    features["bldgclass_clean"] = clean_bldgclass(bldgclass)
    features["bldgclass_group"] = clean_bldgclass_group(bldgclass)
    features["bldgclass_group_label"] = features["bldgclass_group"].map(BLDGCLASS_GROUP_LABELS).fillna(
        "Unknown / other"
    )
    features["delivery_point_source_clean"] = first_existing(raw, ["delivery_point_source"]).fillna("Unknown").astype(str)
    features["building_geometry_source_clean"] = first_existing(
        raw, ["building_raw__geometry_source", "geometry_source"]
    ).fillna("Unknown").astype(str)
    features["building_status_clean"] = first_existing(
        raw, ["building_raw__last_status_type", "last_status_type"]
    ).fillna("Unknown").astype(str)
    features["numfloors_clean"] = numeric(first_existing(raw, ["raw_numfloors", "pluto_raw__numfloors", "numfloors"]))
    features["height_roof_clean"] = numeric(first_existing(raw, ["building_raw__height_roof", "height_roof"]))
    features["construction_year_clean"] = clean_construction_year(
        first_existing(raw, ["building_raw__construction_year", "construction_year"])
    )
    features["ground_elevation_clean"] = numeric(
        first_existing(raw, ["building_raw__ground_elevation", "ground_elevation"])
    )
    features["shape_area_clean"] = numeric(first_existing(raw, ["building_raw__shape_area", "shape_area"]))
    features["distance_to_sidewalk_m_clean"] = numeric(
        first_existing(raw, ["raw_distance_to_sidewalk_m", "distance_to_sidewalk_m"])
    )
    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def output_identifiers(raw: pd.DataFrame, start_row_id: int) -> pd.DataFrame:
    landuse_clean = clean_landuse(first_existing(raw, ["pluto_raw__landuse", "landuse", "landuse_x", "landuse_y"]))
    bldgclass_clean = clean_bldgclass(first_existing(raw, ["pluto_raw__bldgclass", "bldgclass"]))
    bldgclass_group = clean_bldgclass_group(first_existing(raw, ["pluto_raw__bldgclass", "bldgclass"]))
    return pd.DataFrame(
        {
            "row_id": np.arange(start_row_id, start_row_id + len(raw)),
            "bin": first_existing(raw, ["bin"]).to_numpy(),
            "borough": first_existing(raw, ["borough"]).fillna("Unknown").astype(str).to_numpy(),
            "neighborhood": first_existing(raw, ["neighborhood", "nta_name", "building_raw__nta_name"])
            .fillna("Unknown")
            .astype(str)
            .to_numpy(),
            "zipcode": clean_zipcode(first_existing(raw, ["zipcode"])).to_numpy(),
            "map_pluto_bbl": first_existing(raw, ["map_pluto_bbl", "building_raw__map_pluto_bbl", "base_bbl"]).to_numpy(),
            "bldgclass": bldgclass_clean.to_numpy(),
            "bldgclass_group": bldgclass_group.to_numpy(),
            "landuse": landuse_clean.to_numpy(),
            "landuse_label": landuse_clean.map(LANDUSE_LABELS).fillna("Unknown / other").to_numpy(),
            "centroid_lon": numeric(first_existing(raw, ["centroid_lon"])).to_numpy(),
            "centroid_lat": numeric(first_existing(raw, ["centroid_lat"])).to_numpy(),
            "numfloors": numeric(first_existing(raw, ["raw_numfloors", "pluto_raw__numfloors", "numfloors"])).to_numpy(),
            "height_roof": numeric(first_existing(raw, ["building_raw__height_roof", "height_roof"])).to_numpy(),
            "construction_year": clean_construction_year(
                first_existing(raw, ["building_raw__construction_year", "construction_year"])
            ).to_numpy(),
            "ground_elevation": numeric(first_existing(raw, ["building_raw__ground_elevation", "ground_elevation"])).to_numpy(),
            "shape_area": numeric(first_existing(raw, ["building_raw__shape_area", "shape_area"])).to_numpy(),
            "distance_to_sidewalk_m": numeric(
                first_existing(raw, ["raw_distance_to_sidewalk_m", "distance_to_sidewalk_m"])
            ).to_numpy(),
            "delivery_point_source": first_existing(raw, ["delivery_point_source"]).fillna("Unknown").astype(str).to_numpy(),
            "geometry_source": first_existing(raw, ["building_raw__geometry_source", "geometry_source"])
            .fillna("Unknown")
            .astype(str)
            .to_numpy(),
            "last_status_type": first_existing(raw, ["building_raw__last_status_type", "last_status_type"])
            .fillna("Unknown")
            .astype(str)
            .to_numpy(),
            "car_last_meter_mean_s": numeric(first_existing(raw, ["car_last_meter_mean_s", "base_time_mean_s"])).to_numpy(),
            "amr_last_meter_mean_s": numeric(first_existing(raw, ["amr_last_meter_mean_s", "amr_time_mean_s"])).to_numpy(),
        }
    )


def build_model() -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
        ]
    )
    model = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [("num", numeric_pipe, NUMERIC_FEATURES), ("cat", categorical_pipe, CATEGORICAL_FEATURES)]
                ),
            ),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    train_raw = pd.read_csv(TRAIN_CSV, low_memory=False)
    target = numeric(train_raw["target_amr_accessible"]).astype("Int64")
    keep = target.notna()
    return model.fit(prepare_features(train_raw.loc[keep].copy()), target.loc[keep].astype(int))


def estimate_study_area_buildings(model: Pipeline) -> int:
    if BUILDING_OUTPUT_CSV.exists():
        BUILDING_OUTPUT_CSV.unlink()
    total_rows = 0
    first_chunk = True
    for chunk in pd.read_csv(FULL_DATASET_CSV, chunksize=CHUNK_SIZE, low_memory=False):
        borough = first_existing(chunk, ["borough"]).fillna("").astype(str).str.strip()
        chunk = chunk.loc[borough.str.casefold() == STUDY_BOROUGH.casefold()].copy()
        if chunk.empty:
            continue
        proba = model.predict_proba(prepare_features(chunk))[:, 1]
        out = output_identifiers(chunk, total_rows)
        out["amr_accessibility_probability_estimated"] = proba
        out["amr_index_estimated"] = proba * 100
        out["amr_class_estimated_50pct"] = (proba >= 0.5).astype(int)
        out["model_used"] = "logistic_regression_no_ai"
        out.to_csv(BUILDING_OUTPUT_CSV, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        total_rows += len(chunk)
        first_chunk = False
        print(f"Estimated {total_rows:,} {STUDY_AREA_LABEL} buildings...")
    return total_rows


def add_local_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    lat0 = math.radians(float(df["centroid_lat"].mean()))
    lon0 = float(df["centroid_lon"].mean())
    base_lat = float(df["centroid_lat"].mean())
    meters_per_deg_lon = 111_320 * math.cos(lat0)
    meters_per_deg_lat = 110_540
    out = df.copy()
    out["x_m"] = (out["centroid_lon"] - lon0) * meters_per_deg_lon
    out["y_m"] = (out["centroid_lat"] - base_lat) * meters_per_deg_lat
    return out, {
        "lon0": lon0,
        "lat0": base_lat,
        "meters_per_deg_lon": meters_per_deg_lon,
        "meters_per_deg_lat": meters_per_deg_lat,
    }


def xy_to_lonlat(x: float, y: float, meta: dict[str, float]) -> tuple[float, float]:
    return meta["lon0"] + x / meta["meters_per_deg_lon"], meta["lat0"] + y / meta["meters_per_deg_lat"]


def top_category_summary(series: pd.Series, max_items: int = 3) -> str:
    values = series.dropna().astype(str)
    values = values[(values != "") & (values.str.lower() != "unknown")]
    if values.empty:
        return "Unknown"
    counts = values.value_counts().head(max_items)
    total = len(values)
    return "; ".join(f"{name}: {count / total * 100:.1f}%" for name, count in counts.items())


def readiness_class(value: float) -> str:
    if value < 40:
        return "low"
    if value < 60:
        return "medium"
    return "high"


def color_by_index(value: float) -> str:
    if value < 40:
        return "#b2182b"   # red
    if value < 60:
        return "#f6c85f"   # yellow
    return "#1a9850"       # green

def aggregate_cells(df: pd.DataFrame, cell_col: str, polygon_lookup: dict[str, Polygon]) -> tuple[pd.DataFrame, dict]:
    rows = []
    features = []
    for cell_id, group in df.groupby(cell_col):
        if len(group) < MIN_BUILDINGS_PER_CELL or cell_id not in polygon_lookup:
            continue
        mean_index = float(group["amr_index_estimated"].mean())
        row = {
            "cell_id": str(cell_id),
            "readiness_class": readiness_class(mean_index),
            "n_buildings": int(len(group)),
            "mean_amr_index_estimated": mean_index,
            "std_amr_index_estimated": float(group["amr_index_estimated"].std(ddof=0)),
            "top_neighborhood": top_category_summary(group["neighborhood"], 3),
            "top_landuse": top_category_summary(group["landuse_label"], 3),
            "top_bldgclass_groups": top_category_summary(
                group["bldgclass_group"].map(BLDGCLASS_GROUP_LABELS).fillna("Unknown / other"), 3
            ),
            "median_construction_year": float(group["construction_year"].dropna().median())
            if group["construction_year"].notna().any()
            else None,
            "median_numfloors": float(group["numfloors"].dropna().median()) if group["numfloors"].notna().any() else None,
            "centroid_lon": float(group["centroid_lon"].mean()),
            "centroid_lat": float(group["centroid_lat"].mean()),
        }
        rows.append(row)
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon_lookup[cell_id]),
                "properties": {**row, "color": color_by_index(mean_index)},
            }
        )
    summary = pd.DataFrame(rows).sort_values(["centroid_lat", "centroid_lon"])
    return summary, {"type": "FeatureCollection", "features": features}


def aggregate_adjacent_square_zones(
    square_df: pd.DataFrame,
    square_summary: pd.DataFrame,
    square_polygon_lookup: dict[str, Polygon],
) -> tuple[pd.DataFrame, dict]:
    cell_class = dict(zip(square_summary["cell_id"].astype(str), square_summary["readiness_class"].astype(str)))
    valid_cells = set(cell_class)
    visited: set[str] = set()
    components: list[list[str]] = []

    for cell_id in sorted(valid_cells):
        if cell_id in visited:
            continue
        label = cell_class[cell_id]
        stack = [cell_id]
        visited.add(cell_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            i, j = (int(part) for part in current.split("_"))
            for neighbor in (f"{i + 1}_{j}", f"{i - 1}_{j}", f"{i}_{j + 1}", f"{i}_{j - 1}"):
                if neighbor in visited or neighbor not in valid_cells:
                    continue
                if cell_class[neighbor] == label:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    zone_polygon_lookup: dict[str, Polygon] = {}
    cell_to_zone: dict[str, str] = {}
    zone_counts = {"low": 0, "medium": 0, "high": 0}
    for component in components:
        label = cell_class[component[0]]
        zone_counts[label] += 1
        zone_id = f"ASZ_{zone_counts[label]:03d}_{label.upper()}"
        for cell_id in component:
            cell_to_zone[cell_id] = zone_id
        zone_polygon_lookup[zone_id] = unary_union([square_polygon_lookup[cell_id] for cell_id in component]).buffer(0)

    work = square_df[square_df["square_id"].isin(cell_to_zone)].copy()
    work["aggregated_square_zone_id"] = work["square_id"].map(cell_to_zone)
    summary, geojson = aggregate_cells(work, "aggregated_square_zone_id", zone_polygon_lookup)
    summary = summary.rename(columns={"cell_id": "zone_id"})
    summary["n_square_cells"] = summary["zone_id"].map(
        {zone_id: sum(1 for value in cell_to_zone.values() if value == zone_id) for zone_id in zone_polygon_lookup}
    )
    for feature in geojson["features"]:
        zone_id = feature["properties"]["cell_id"]
        feature["properties"]["zone_id"] = zone_id
        feature["properties"]["n_square_cells"] = int(summary.loc[summary["zone_id"] == zone_id, "n_square_cells"].iloc[0])
    return summary, geojson


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
    hex_df: pd.DataFrame,
    hex_summary: pd.DataFrame,
    hex_polygon_lookup: dict[str, Polygon],
) -> tuple[pd.DataFrame, dict]:
    cell_class = dict(zip(hex_summary["cell_id"].astype(str), hex_summary["readiness_class"].astype(str)))
    valid_cells = set(cell_class)
    visited: set[str] = set()
    components: list[list[str]] = []

    for cell_id in sorted(valid_cells):
        if cell_id in visited:
            continue
        label = cell_class[cell_id]
        stack = [cell_id]
        visited.add(cell_id)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in hex_neighbors(current):
                if neighbor in visited or neighbor not in valid_cells:
                    continue
                if cell_class[neighbor] == label:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    zone_polygon_lookup: dict[str, Polygon] = {}
    cell_to_zone: dict[str, str] = {}
    zone_counts = {"low": 0, "medium": 0, "high": 0}
    for component in components:
        label = cell_class[component[0]]
        zone_counts[label] += 1
        zone_id = f"AHZ_{zone_counts[label]:03d}_{label.upper()}"
        for cell_id in component:
            cell_to_zone[cell_id] = zone_id
        zone_polygon_lookup[zone_id] = unary_union([hex_polygon_lookup[cell_id] for cell_id in component]).buffer(0)

    work = hex_df[hex_df["hex_id"].isin(cell_to_zone)].copy()
    work["aggregated_hex_zone_id"] = work["hex_id"].map(cell_to_zone)
    summary, geojson = aggregate_cells(work, "aggregated_hex_zone_id", zone_polygon_lookup)
    summary = summary.rename(columns={"cell_id": "zone_id"})
    summary["n_hexagons"] = summary["zone_id"].map(
        {zone_id: sum(1 for value in cell_to_zone.values() if value == zone_id) for zone_id in zone_polygon_lookup}
    )
    for feature in geojson["features"]:
        zone_id = feature["properties"]["cell_id"]
        feature["properties"]["zone_id"] = zone_id
        feature["properties"]["n_hexagons"] = int(summary.loc[summary["zone_id"] == zone_id, "n_hexagons"].iloc[0])
    return summary, geojson


def square_polygons(df: pd.DataFrame, meta: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Polygon]]:
    min_x = math.floor(float(df["x_m"].min()) / SQUARE_CELL_SIZE_M) * SQUARE_CELL_SIZE_M
    min_y = math.floor(float(df["y_m"].min()) / SQUARE_CELL_SIZE_M) * SQUARE_CELL_SIZE_M
    out = df.copy()
    out["square_i"] = np.floor((out["x_m"] - min_x) / SQUARE_CELL_SIZE_M).astype(int)
    out["square_j"] = np.floor((out["y_m"] - min_y) / SQUARE_CELL_SIZE_M).astype(int)
    out["square_id"] = out["square_i"].astype(str) + "_" + out["square_j"].astype(str)
    polygons = {}
    for cell_id, group in out.groupby("square_id"):
        i = int(group["square_i"].iloc[0])
        j = int(group["square_j"].iloc[0])
        x0 = min_x + i * SQUARE_CELL_SIZE_M
        y0 = min_y + j * SQUARE_CELL_SIZE_M
        polygons[cell_id] = Polygon(
            [
                xy_to_lonlat(x0, y0, meta),
                xy_to_lonlat(x0 + SQUARE_CELL_SIZE_M, y0, meta),
                xy_to_lonlat(x0 + SQUARE_CELL_SIZE_M, y0 + SQUARE_CELL_SIZE_M, meta),
                xy_to_lonlat(x0, y0 + SQUARE_CELL_SIZE_M, meta),
                xy_to_lonlat(x0, y0, meta),
            ]
        )
    return out, polygons


def hex_polygon(cx: float, cy: float, radius: float, meta: dict[str, float]) -> Polygon:
    coords = []
    for k in range(6):
        angle = math.radians(60 * k)
        coords.append(xy_to_lonlat(cx + radius * math.cos(angle), cy + radius * math.sin(angle), meta))
    coords.append(coords[0])
    return Polygon(coords)


def hex_polygons(df: pd.DataFrame, meta: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Polygon]]:
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
        y_offset = dy / 2 if col % 2 else 0
        row = 0
        y = min_y + y_offset
        while y <= max_y:
            centers.append((f"H{col}_{row}", x, y))
            row += 1
            y += dy
        col += 1
        x += dx
    center_ids = np.array([c[0] for c in centers], dtype=object)
    center_xy = np.array([(c[1], c[2]) for c in centers], dtype=float)
    _, idx = cKDTree(center_xy).query(df[["x_m", "y_m"]].to_numpy(dtype=float), k=1)
    out = df.copy()
    out["hex_id"] = center_ids[idx]
    polygons = {cell_id: hex_polygon(cx, cy, radius, meta) for cell_id, cx, cy in centers}
    return out, polygons


def save_map(summary: pd.DataFrame, geojson: dict, output_html: Path, title: str, unit_label: str) -> None:
    payload = {
        "geojson": geojson,
        "neighborhoodOutlines": read_neighborhood_outlines(),
        "labels": [
            {
                "cell_id": row.cell_id,
                "lat": float(row.centroid_lat),
                "lon": float(row.centroid_lon),
                "mean_amr_index": round(float(row.mean_amr_index_estimated), 1),
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
    html, body {{ height:100%; margin:0; font-family:Arial, Helvetica, sans-serif; }}
    #map {{ height:100%; width:100%; }}
    .title-box {{ position:absolute; top:14px; left:54px; z-index:900; background:rgba(255,255,255,.94); border:1px solid #cfd6df; border-radius:6px; padding:10px 12px; max-width:620px; box-shadow:0 8px 22px rgba(23,32,42,.15); }}
    .title-box h1 {{ margin:0 0 4px; font-size:18px; }}
    .title-box p {{ margin:0; font-size:13px; color:#53606b; }}
    .legend {{ position:absolute; right:14px; bottom:24px; z-index:900; background:rgba(255,255,255,.94); border:1px solid #cfd6df; border-radius:6px; padding:10px 12px; font-size:12px; box-shadow:0 8px 22px rgba(23,32,42,.15); }}
    .swatch {{ display:inline-block; width:14px; height:14px; margin-right:6px; border:1px solid #172026; vertical-align:-2px; }}
    .amr-label {{ background:rgba(255,255,255,.92); border:1px solid #1f2933; border-radius:4px; color:#172026; font-weight:700; font-size:11px; padding:1px 4px; white-space:nowrap; box-shadow:0 2px 5px rgba(0,0,0,.2); }}
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="title-box"><h1>{title}</h1><p>{STUDY_AREA_LABEL} estimated AMR Index from the no-AI logistic model, aggregated by {unit_label}. Use the checkbox to hide labels at low zoom.</p></div>
  <div class="legend"><strong>Estimated AMR Index</strong><div><span class="swatch" style="background:#b2182b"></span>Low (0-40)</div><div><span class="swatch" style="background:#f6c85f"></span>Medium (40-60)</div><div><span class="swatch" style="background:#1a9850"></span>High (60-100)</div></div>
  <script>
    const payload = {json.dumps(payload)};
    const map = L.map('map', {{ preferCanvas:true }});
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  maxZoom:16,
  attribution:'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
}}).addTo(map);
    function neighborhoodStyle() {{
      return {{ color:'#334155', weight:1.2, fillOpacity:0, opacity:.7, dashArray:'4 3', interactive:false }};
    }}
    let neighborhoodOutlines = null;
    if (payload.neighborhoodOutlines) {{
      neighborhoodOutlines = L.geoJSON(payload.neighborhoodOutlines, {{ style: neighborhoodStyle }}).addTo(map);
    }}
    function styleFeature(feature) {{
      return {{ color:'#172026', weight:0.3, fillColor:feature.properties.color, fillOpacity:.52, opacity:.85 }};
    }}
    function mutedStyle() {{
      return {{ color:'#9aa3aa', weight:.6, fillColor:'#c3cbd1', fillOpacity:.12, opacity:.25 }};
    }}
    function highlightCell(activeId) {{
      cells.eachLayer(layer => {{
        if (layer.feature.properties.cell_id === activeId) {{
          layer.setStyle({{ color:'#111827', weight:2.4, fillColor:layer.feature.properties.color, fillOpacity:.75, opacity:1 }});
          if (layer.bringToFront) layer.bringToFront();
        }} else {{
          layer.setStyle(mutedStyle());
        }}
      }});
    }}
    function resetHighlight() {{
      cells.eachLayer(layer => layer.setStyle(styleFeature(layer.feature)));
    }}
    function bindTooltip(feature, layer) {{
      const p = feature.properties;
      layer.bindTooltip(
        `<b>${{p.cell_id}}</b><br>` +
        `Class: ${{p.readiness_class}}<br>` +
        `Estimated AMR Index: ${{p.mean_amr_index_estimated.toFixed(1)}}<br>` +
        `Buildings: ${{p.n_buildings}}<br>` +
        `<br><b>Top neighborhoods</b><br>${{p.top_neighborhood}}<br>` +
        `<br><b>Most common land use</b><br>${{p.top_landuse}}<br>` +
        `<br><b>Most common building types</b><br>${{p.top_bldgclass_groups}}<br>` +
        `<br>Median construction year: ${{p.median_construction_year === null ? 'Unknown' : p.median_construction_year.toFixed(0)}}<br>` +
        `Median floors: ${{p.median_numfloors === null ? 'Unknown' : p.median_numfloors.toFixed(1)}}`,
        {{ sticky:true }}
      );
      layer.on({{ mouseover: () => highlightCell(p.cell_id), mouseout: resetHighlight }});
    }}
    const cells = L.geoJSON(payload.geojson, {{ style:styleFeature, onEachFeature:bindTooltip }}).addTo(map);
    const labelsLayer = L.layerGroup().addTo(map);
    for (const item of payload.labels) {{
      const icon = L.divIcon({{ className:'amr-label', html:`${{item.mean_amr_index.toFixed(0)}}`, iconSize:null }});
      L.marker([item.lat, item.lon], {{ icon }}).addTo(labelsLayer);
    }}
    const overlays = {{ 'AMR Index labels': labelsLayer }};
    if (neighborhoodOutlines) {{
      overlays['Neighborhood outlines'] = neighborhoodOutlines;
    }}
    L.control.layers(null, overlays, {{ collapsed:false }}).addTo(map);
    map.fitBounds(cells.getBounds(), {{ padding:[24,24] }});
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")


def build_maps() -> tuple[int, int, int, int]:
    cols = [
        "borough",
        "neighborhood",
        "landuse_label",
        "bldgclass_group",
        "centroid_lon",
        "centroid_lat",
        "construction_year",
        "numfloors",
        "amr_index_estimated",
    ]
    df = pd.read_csv(BUILDING_OUTPUT_CSV, usecols=cols, low_memory=False)
    for col in ["centroid_lon", "centroid_lat", "construction_year", "numfloors", "amr_index_estimated"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["centroid_lon", "centroid_lat", "amr_index_estimated"]).copy()
    df, meta = add_local_xy(df)

    square_df, square_polygon_lookup = square_polygons(df, meta)
    square_summary, square_geojson = aggregate_cells(square_df, "square_id", square_polygon_lookup)
    square_summary.to_csv(SQUARE_SUMMARY_CSV, index=False)
    SQUARE_GEOJSON.write_text(json.dumps(square_geojson), encoding="utf-8")
    save_map(square_summary, square_geojson, SQUARE_MAP_HTML, f"{STUDY_AREA_LABEL} Square Grid AMR Index", f"{SQUARE_CELL_SIZE_M} m square cells")

    agg_square_summary, agg_square_geojson = aggregate_adjacent_square_zones(
        square_df, square_summary, square_polygon_lookup
    )
    agg_square_summary.to_csv(AGG_SQUARE_SUMMARY_CSV, index=False)
    AGG_SQUARE_GEOJSON.write_text(json.dumps(agg_square_geojson), encoding="utf-8")
    save_map(
        agg_square_summary.rename(columns={"zone_id": "cell_id"}),
        agg_square_geojson,
        AGG_SQUARE_MAP_HTML,
        f"{STUDY_AREA_LABEL} Aggregated Square AMR Zones",
        "adjacent square cells with the same AMR readiness class",
    )

    hex_df, hex_polygon_lookup = hex_polygons(df, meta)
    hex_summary, hex_geojson = aggregate_cells(hex_df, "hex_id", hex_polygon_lookup)
    hex_summary.to_csv(HEX_SUMMARY_CSV, index=False)
    HEX_GEOJSON.write_text(json.dumps(hex_geojson), encoding="utf-8")
    save_map(hex_summary, hex_geojson, HEX_MAP_HTML, f"{STUDY_AREA_LABEL} Hex Grid AMR Index", f"{HEX_RADIUS_M} m hexagonal cells")

    agg_hex_summary, agg_hex_geojson = aggregate_adjacent_hex_zones(hex_df, hex_summary, hex_polygon_lookup)
    agg_hex_summary.to_csv(AGG_HEX_SUMMARY_CSV, index=False)
    AGG_HEX_GEOJSON.write_text(json.dumps(agg_hex_geojson), encoding="utf-8")
    save_map(
        agg_hex_summary.rename(columns={"zone_id": "cell_id"}),
        agg_hex_geojson,
        AGG_HEX_MAP_HTML,
        f"{STUDY_AREA_LABEL} Aggregated Hexagonal AMR Zones",
        "adjacent hexagons with the same AMR readiness class",
    )
    return len(square_summary), len(hex_summary), len(agg_square_summary), len(agg_hex_summary)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Training no-AI logistic model on AI-labeled subset...")
    model = build_model()
    print(f"Applying model to {STUDY_AREA_LABEL} buildings...")
    total_rows = estimate_study_area_buildings(model)
    print(f"Building {STUDY_AREA_LABEL} maps...")
    n_squares, n_hexes, n_agg_square_zones, n_agg_hex_zones = build_maps()
    report = [
        f"{STUDY_AREA_LABEL} AMR Index Zoning Report",
        "",
        f"Training dataset: {TRAIN_CSV}",
        f"Source dataset: {FULL_DATASET_CSV}",
        f"Study area filter: borough = {STUDY_BOROUGH}",
        f"Buildings estimated: {total_rows:,}",
        f"Square cell size: {SQUARE_CELL_SIZE_M} m",
        f"Hex radius: {HEX_RADIUS_M} m",
        f"Minimum buildings per cell: {MIN_BUILDINGS_PER_CELL}",
        f"Square cells retained: {n_squares:,}",
        f"Hex cells retained: {n_hexes:,}",
        f"Aggregated square zones retained: {n_agg_square_zones:,}",
        f"Aggregated hex zones retained: {n_agg_hex_zones:,}",
        "",
        "Formula:",
        "amr_index_estimated = 100 * P(AMR accessible | non-AI building and urban features)",
        "",
        "Model predictors:",
        "- Numeric: " + ", ".join(NUMERIC_FEATURES),
        "- Categorical: " + ", ".join(CATEGORICAL_FEATURES),
        "- Geographic identifiers are excluded from the model predictors.",
    ]
    REPORT_TXT.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved building CSV: {BUILDING_OUTPUT_CSV}")
    print(f"Saved square map: {SQUARE_MAP_HTML}")
    print(f"Saved aggregated square zone map: {AGG_SQUARE_MAP_HTML}")
    print(f"Saved hex map: {HEX_MAP_HTML}")
    print(f"Saved aggregated hex zone map: {AGG_HEX_MAP_HTML}")
    print(f"Saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
