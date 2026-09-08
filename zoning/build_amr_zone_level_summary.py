"""Build AMR-friendly-zone performance summaries for the AI subset.

This mirrors the ZIP/neighborhood area-level summary, but the grouping unit is
the data-driven AMR-friendly zone created from the building-level AMR Index.
Only the AI-labeled subset is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
ZONE_GEOJSON = SCRIPT_DIR / "output" / "amr_friendly_zones.geojson"
OUTPUT_DIR = SCRIPT_DIR / "output"

ZONE_SUMMARY_CSV = OUTPUT_DIR / "area_level_amr_summary_by_amr_friendly_zone.csv"
ZONE_TOP_BOTTOM_CSV = OUTPUT_DIR / "area_level_amr_top_bottom_amr_friendly_zones.csv"
REPORT_TXT = OUTPUT_DIR / "area_level_amr_summary_by_amr_friendly_zone_report.txt"

MIN_BUILDINGS_PER_AREA = 20

LANDUSE_LABELS = {
    "1": "One & two family buildings",
    "2": "Multi-family walk-up buildings",
    "3": "Multi-family elevator buildings",
    "4": "Mixed residential & commercial buildings",
    "5": "Commercial & office buildings",
    "6": "Industrial & manufacturing",
    "7": "Transportation & utility",
    "8": "Public facilities & institutions",
    "9": "Open space & outdoor recreation",
    "10": "Parking facilities",
    "11": "Vacant land",
}

BLDGCLASS_GROUP_LABELS = {
    "A": "One-family dwellings",
    "B": "Two-family dwellings",
    "C": "Walk-up apartments",
    "D": "Elevator apartments",
    "E": "Warehouses",
    "F": "Factory / industrial",
    "G": "Garages",
    "H": "Hotels",
    "I": "Hospitals / health",
    "J": "Theatres",
    "K": "Retail",
    "L": "Loft buildings",
    "M": "Churches / religious",
    "N": "Asylums / homes",
    "O": "Office buildings",
    "P": "Places of public assembly",
    "Q": "Outdoor recreation",
    "R": "Condominiums",
    "S": "Residence with store/office",
    "T": "Transportation",
    "U": "Utility",
    "V": "Vacant land",
    "W": "Educational structures",
    "Y": "Government / public safety",
    "Z": "Miscellaneous",
}


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def bool01(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.map({"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0})


def clean_zipcode(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.round().astype("Int64").astype(str).replace("<NA>", "Unknown")


def clean_landuse(series: pd.Series) -> pd.Series:
    values = numeric(series)
    return values.round().astype("Int64").astype(str).replace("<NA>", "Unknown")


def clean_bldgclass_group(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().str[0].where(series.notna(), "Unknown")


def clean_construction_year(series: pd.Series) -> pd.Series:
    values = numeric(series)
    values = values.where(~values.between(1.0, 2.5), values * 1000)
    return values.where(values.between(1600, 2030))


def first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(index=df.index, dtype=object)


def mode_and_share(group: pd.DataFrame, column: str) -> tuple[str, float]:
    values = group[column].dropna()
    values = values[values.astype(str).str.lower() != "unknown"]
    if values.empty:
        return "Unknown", 0.0
    counts = values.astype(str).value_counts()
    return counts.index[0], float(counts.iloc[0] / len(group))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    landuse_source = work["pluto_raw__landuse"] if "pluto_raw__landuse" in work else work.get("landuse")
    work["landuse_code"] = clean_landuse(landuse_source)
    work["landuse_label"] = work["landuse_code"].map(LANDUSE_LABELS).fillna("Unknown / other")

    bldgclass_source = work["pluto_raw__bldgclass"] if "pluto_raw__bldgclass" in work else work.get("bldgclass")
    work["bldgclass_group"] = clean_bldgclass_group(bldgclass_source)
    work["bldgclass_group_label"] = work["bldgclass_group"].map(BLDGCLASS_GROUP_LABELS).fillna("Unknown / other")

    work["target_amr_accessible_num"] = numeric(work["target_amr_accessible"])
    work["amr_time_mean_s_num"] = numeric(work.get("amr_time_mean_s", work.get("amr_last_meter_mean_s")))
    work["car_time_mean_s_num"] = numeric(work.get("base_time_mean_s", work.get("car_last_meter_mean_s")))
    work["ai_barrier_score_num"] = numeric(
        work.get("ai_access_barrier_mean", work.get("streetview_access_barrier_score_R"))
    )
    work["car_ai_penalty_s_num"] = numeric(work.get("car_ai_penalty_s"))
    work["amr_ai_penalty_s_num"] = numeric(work.get("amr_ai_penalty_s"))

    work["image_usable_num"] = bool01(work.get("image_usable", work.get("streetview_image_usable_flag")))
    work["stairs_present_num"] = bool01(work.get("stairs_present", work.get("streetview_stairs_detected")))
    work["gate_present_num"] = bool01(work.get("gate_present", work.get("streetview_gate_detected")))
    work["ramp_present_num"] = bool01(work.get("ramp_present", work.get("streetview_ramp_or_barrier_detected")))
    work["amr_can_reach_door_num"] = bool01(
        work.get("amr_can_reach_door", work.get("streetview_amr_can_reach_door_flag"))
    )

    work["numfloors_num"] = numeric(work.get("raw_numfloors", work.get("numfloors")))
    work["height_roof_num"] = numeric(work.get("building_raw__height_roof", work.get("height_roof")))
    work["shape_area_num"] = numeric(work.get("building_raw__shape_area", work.get("shape_area")))
    work["ground_elevation_num"] = numeric(work.get("building_raw__ground_elevation", work.get("ground_elevation")))
    work["construction_year_num"] = clean_construction_year(
        work.get("building_raw__construction_year", work.get("construction_year"))
    )
    work["distance_to_sidewalk_m_num"] = numeric(
        work.get("raw_distance_to_sidewalk_m", work.get("distance_to_sidewalk_m"))
    )
    work["centroid_lon_num"] = numeric(first_existing(work, ["centroid_lon"]))
    work["centroid_lat_num"] = numeric(first_existing(work, ["centroid_lat"]))

    work["is_walkup"] = (work["bldgclass_group"] == "C").astype(int)
    work["is_elevator"] = (work["bldgclass_group"] == "D").astype(int)
    work["is_office"] = (work["bldgclass_group"] == "O").astype(int)
    work["is_condo"] = (work["bldgclass_group"] == "R").astype(int)
    work["is_commercial_landuse"] = (work["landuse_code"] == "5").astype(int)
    return work


def load_zones() -> list[dict[str, object]]:
    if not ZONE_GEOJSON.exists():
        raise FileNotFoundError(f"Missing AMR zone GeoJSON: {ZONE_GEOJSON}")
    data = json.loads(ZONE_GEOJSON.read_text(encoding="utf-8"))
    zones = []
    for feature in data["features"]:
        props = feature["properties"]
        zones.append(
            {
                "zone_id": props["zone_id"],
                "readiness_class": props.get("readiness_class"),
                "zone_mean_amr_index": props.get("mean_amr_index"),
                "geometry": shape(feature["geometry"]),
            }
        )
    return zones


def assign_zones(work: pd.DataFrame, zones: list[dict[str, object]]) -> pd.DataFrame:
    zone_ids = []
    zone_classes = []
    zone_mean_indexes = []
    for _, row in work.iterrows():
        if pd.isna(row["centroid_lon_num"]) or pd.isna(row["centroid_lat_num"]):
            zone_ids.append(None)
            zone_classes.append(None)
            zone_mean_indexes.append(None)
            continue
        point = Point(float(row["centroid_lon_num"]), float(row["centroid_lat_num"]))
        match = None
        for zone in zones:
            geom = zone["geometry"]
            if geom.contains(point) or geom.touches(point):
                match = zone
                break
        zone_ids.append(None if match is None else match["zone_id"])
        zone_classes.append(None if match is None else match["readiness_class"])
        zone_mean_indexes.append(None if match is None else match["zone_mean_amr_index"])

    out = work.copy()
    out["area_amr_friendly_zone"] = zone_ids
    out["zone_readiness_class"] = zone_classes
    out["zone_mean_amr_index"] = zone_mean_indexes
    return out


def aggregate_amr_zones(work: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = work.dropna(subset=["area_amr_friendly_zone"]).groupby("area_amr_friendly_zone", dropna=False)
    for zone_id, group in grouped:
        dominant_landuse, dominant_landuse_share = mode_and_share(group, "landuse_label")
        dominant_bldgclass, dominant_bldgclass_share = mode_and_share(group, "bldgclass_group_label")

        target = group["target_amr_accessible_num"]
        accessible = int((target == 1).sum())
        labeled = int(target.notna().sum())

        rows.append(
            {
                "geography_type": "amr_friendly_zone",
                "area_id": str(zone_id),
                "area_name": str(zone_id),
                "zone_readiness_class": group["zone_readiness_class"].dropna().iloc[0]
                if group["zone_readiness_class"].notna().any()
                else None,
                "zone_mean_amr_index": group["zone_mean_amr_index"].dropna().astype(float).iloc[0]
                if group["zone_mean_amr_index"].notna().any()
                else None,
                "n_buildings": int(len(group)),
                "n_with_amr_label": labeled,
                "amr_accessible_buildings": accessible,
                "amr_not_accessible_buildings": int((target == 0).sum()),
                "amr_accessibility_rate": float(target.mean()) if labeled else None,
                "avg_amr_time_s": group["amr_time_mean_s_num"].mean(),
                "median_amr_time_s": group["amr_time_mean_s_num"].median(),
                "avg_car_time_s": group["car_time_mean_s_num"].mean(),
                "median_car_time_s": group["car_time_mean_s_num"].median(),
                "avg_ai_barrier_score": group["ai_barrier_score_num"].mean(),
                "avg_car_ai_penalty_s": group["car_ai_penalty_s_num"].mean(),
                "avg_amr_ai_penalty_s": group["amr_ai_penalty_s_num"].mean(),
                "share_image_usable": group["image_usable_num"].mean(),
                "share_stairs_detected": group["stairs_present_num"].mean(),
                "share_gate_detected": group["gate_present_num"].mean(),
                "share_ramp_or_barrier_detected": group["ramp_present_num"].mean(),
                "share_ai_amr_can_reach_door": group["amr_can_reach_door_num"].mean(),
                "avg_numfloors": group["numfloors_num"].mean(),
                "median_numfloors": group["numfloors_num"].median(),
                "avg_height_roof": group["height_roof_num"].mean(),
                "avg_shape_area": group["shape_area_num"].mean(),
                "avg_ground_elevation": group["ground_elevation_num"].mean(),
                "avg_construction_year": group["construction_year_num"].mean(),
                "avg_distance_to_sidewalk_m": group["distance_to_sidewalk_m_num"].mean(),
                "dominant_landuse": dominant_landuse,
                "dominant_landuse_share": dominant_landuse_share,
                "dominant_bldgclass_group": dominant_bldgclass,
                "dominant_bldgclass_group_share": dominant_bldgclass_share,
                "share_walkup_apartments": group["is_walkup"].mean(),
                "share_elevator_apartments": group["is_elevator"].mean(),
                "share_office_buildings": group["is_office"].mean(),
                "share_condominiums": group["is_condo"].mean(),
                "share_commercial_landuse": group["is_commercial_landuse"].mean(),
            }
        )

    out = pd.DataFrame(rows)
    out = out[out["n_buildings"] >= MIN_BUILDINGS_PER_AREA].copy()
    return out.sort_values(["amr_accessibility_rate", "n_buildings"], ascending=[False, False])


def top_bottom(zones: pd.DataFrame) -> pd.DataFrame:
    high = zones.sort_values(["amr_accessibility_rate", "n_buildings"], ascending=[False, False]).head(10)
    low = zones.sort_values(["amr_accessibility_rate", "n_buildings"], ascending=[True, False]).head(10)
    return pd.concat(
        [high.assign(ranking_type="top_accessibility"), low.assign(ranking_type="bottom_accessibility")],
        ignore_index=True,
    )


def write_report(input_rows: int, assigned_rows: int, zones: pd.DataFrame) -> None:
    def fmt_rate(value: object) -> str:
        return "NA" if pd.isna(value) else f"{float(value) * 100:.1f}%"

    lines = [
        "AMR-Friendly-Zone Performance Summary",
        "",
        "Purpose:",
        "Aggregate the 7,013 AI-labeled buildings into the data-driven AMR-friendly zones.",
        f"Zones with fewer than {MIN_BUILDINGS_PER_AREA} buildings are excluded from the exported summary.",
        "",
        "Coverage:",
        f"- Input AI-labeled buildings: {input_rows:,}",
        f"- Buildings assigned to an AMR-friendly zone: {assigned_rows:,}",
        f"- AMR-friendly zones exported: {len(zones):,}",
        "",
        "Highest AMR accessibility zones:",
    ]
    for _, row in zones.head(5).iterrows():
        lines.append(
            f"- {row['area_name']}: {fmt_rate(row['amr_accessibility_rate'])} "
            f"({int(row['n_buildings'])} buildings), class: {row['zone_readiness_class']}, "
            f"dominant land use: {row['dominant_landuse']}"
        )
    lines += ["", "Lowest AMR accessibility zones:"]
    for _, row in zones.sort_values(["amr_accessibility_rate", "n_buildings"], ascending=[True, False]).head(5).iterrows():
        lines.append(
            f"- {row['area_name']}: {fmt_rate(row['amr_accessibility_rate'])} "
            f"({int(row['n_buildings'])} buildings), class: {row['zone_readiness_class']}, "
            f"dominant land use: {row['dominant_landuse']}"
        )
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(INPUT_CSV, low_memory=False)
    work = prepare(raw)
    work = assign_zones(work, load_zones())
    assigned = int(work["area_amr_friendly_zone"].notna().sum())
    zone_summary = aggregate_amr_zones(work)
    zone_summary.to_csv(ZONE_SUMMARY_CSV, index=False)
    top_bottom(zone_summary).to_csv(ZONE_TOP_BOTTOM_CSV, index=False)
    write_report(len(raw), assigned, zone_summary)

    print(f"Input buildings: {len(raw):,}")
    print(f"Assigned to AMR-friendly zones: {assigned:,}")
    print(f"Exported AMR-friendly zones: {len(zone_summary):,}")
    print(f"Saved: {ZONE_SUMMARY_CSV}")
    print(f"Saved top/bottom: {ZONE_TOP_BOTTOM_CSV}")
    print(f"Saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
