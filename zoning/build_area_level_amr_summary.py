"""Build area-level AMR accessibility summaries from the AI subset.

This is the first urban-planning step: convert the building-level AI dataset
into existing geographic aggregations and exclude very small areas from the
area-level outputs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
MIN_BUILDINGS_PER_AREA = 20

ALL_AREAS_CSV = OUTPUT_DIR / "area_level_amr_summary_all_geographies.csv"
BOROUGH_CSV = OUTPUT_DIR / "area_level_amr_summary_by_borough.csv"
NEIGHBORHOOD_CSV = OUTPUT_DIR / "area_level_amr_summary_by_neighborhood.csv"
ZIPCODE_CSV = OUTPUT_DIR / "area_level_amr_summary_by_zipcode.csv"
TOP_BOTTOM_CSV = OUTPUT_DIR / "area_level_amr_top_bottom_areas.csv"
REPORT_TXT = OUTPUT_DIR / "area_level_amr_summary_report.txt"


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


def read_input() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_CSV}")
    return pd.read_csv(INPUT_CSV, low_memory=False)


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
    return series.astype(str).str.strip().str.upper().str[0].replace({"N": "N"}).where(
        series.notna(), "Unknown"
    )


def clean_construction_year(series: pd.Series) -> pd.Series:
    values = numeric(series)
    # Some CSV exports read values such as 1,920 as 1.92. Restore the year scale.
    values = values.where(~values.between(1.0, 2.5), values * 1000)
    return values.where(values.between(1600, 2030))


def mode_and_share(group: pd.DataFrame, column: str) -> tuple[str, float]:
    values = group[column].dropna()
    values = values[values.astype(str).str.lower() != "unknown"]
    if values.empty:
        return "Unknown", 0.0
    counts = values.astype(str).value_counts()
    return counts.index[0], float(counts.iloc[0] / len(group))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    work["area_borough"] = work.get("borough", pd.Series(index=work.index, dtype=object)).fillna("Unknown")
    work["area_neighborhood"] = work.get("neighborhood", pd.Series(index=work.index, dtype=object)).fillna("Unknown")
    work["area_zipcode"] = clean_zipcode(work.get("zipcode", pd.Series(index=work.index, dtype=object)))

    landuse_source = work["pluto_raw__landuse"] if "pluto_raw__landuse" in work else work.get("landuse")
    work["landuse_code"] = clean_landuse(landuse_source)
    work["landuse_label"] = work["landuse_code"].map(LANDUSE_LABELS).fillna("Unknown / other")

    bldgclass_source = work["pluto_raw__bldgclass"] if "pluto_raw__bldgclass" in work else work.get("bldgclass")
    work["bldgclass_group"] = clean_bldgclass_group(bldgclass_source)
    work["bldgclass_group_label"] = work["bldgclass_group"].map(BLDGCLASS_GROUP_LABELS).fillna("Unknown / other")

    work["target_amr_accessible_num"] = numeric(work["target_amr_accessible"])
    work["amr_time_mean_s_num"] = numeric(work.get("amr_time_mean_s", work.get("amr_last_meter_mean_s")))
    work["car_time_mean_s_num"] = numeric(work.get("base_time_mean_s", work.get("car_last_meter_mean_s")))
    work["ai_barrier_score_num"] = numeric(work.get("ai_access_barrier_mean", work.get("streetview_access_barrier_score_R")))
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
    work["distance_to_sidewalk_m_num"] = numeric(work.get("raw_distance_to_sidewalk_m", work.get("distance_to_sidewalk_m")))

    work["is_walkup"] = (work["bldgclass_group"] == "C").astype(int)
    work["is_elevator"] = (work["bldgclass_group"] == "D").astype(int)
    work["is_office"] = (work["bldgclass_group"] == "O").astype(int)
    work["is_condo"] = (work["bldgclass_group"] == "R").astype(int)
    work["is_commercial_landuse"] = (work["landuse_code"] == "5").astype(int)

    return work


def aggregate_one(work: pd.DataFrame, area_col: str, geography_type: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for area_value, group in work.groupby(area_col, dropna=False):
        area_id = "Unknown" if pd.isna(area_value) else str(area_value)
        dominant_landuse, dominant_landuse_share = mode_and_share(group, "landuse_label")
        dominant_bldgclass, dominant_bldgclass_share = mode_and_share(group, "bldgclass_group_label")

        target = group["target_amr_accessible_num"]
        accessible = int((target == 1).sum())
        labeled = int(target.notna().sum())

        rows.append(
            {
                "geography_type": geography_type,
                "area_id": area_id,
                "area_name": area_id,
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


def top_bottom(all_areas: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for geography_type, group in all_areas.groupby("geography_type", dropna=False):
        sorted_high = group.sort_values(["amr_accessibility_rate", "n_buildings"], ascending=[False, False]).head(10)
        sorted_low = group.sort_values(["amr_accessibility_rate", "n_buildings"], ascending=[True, False]).head(10)
        parts.append(sorted_high.assign(ranking_type="top_accessibility"))
        parts.append(sorted_low.assign(ranking_type="bottom_accessibility"))
    return pd.concat(parts, ignore_index=True)


def write_report(borough: pd.DataFrame, neighborhood: pd.DataFrame, zipcode: pd.DataFrame) -> None:
    def fmt_rate(value: object) -> str:
        return "NA" if pd.isna(value) else f"{float(value) * 100:.1f}%"

    lines = [
        "Area-Level AMR Accessibility Summary",
        "",
        "Purpose:",
        "This first planning step aggregates the building-level AI subset into existing geographic units.",
        f"Areas with fewer than {MIN_BUILDINGS_PER_AREA} buildings are excluded from the area-level outputs.",
        "The input building-level dataset itself is not modified.",
        "",
        "Coverage:",
        f"- Borough areas: {len(borough)}",
        f"- Neighborhood areas: {len(neighborhood)}",
        f"- ZIP code areas: {len(zipcode)}",
        "",
        "Neighborhood results, highest AMR accessibility:",
    ]

    top_neighborhoods = neighborhood.sort_values(
        ["amr_accessibility_rate", "n_buildings"], ascending=[False, False]
    ).head(5)
    for _, row in top_neighborhoods.iterrows():
        lines.append(
            f"- {row['area_name']}: {fmt_rate(row['amr_accessibility_rate'])} "
            f"({int(row['n_buildings'])} buildings), dominant land use: {row['dominant_landuse']}"
        )

    lines += ["", "Neighborhood results, lowest AMR accessibility:"]
    bottom_neighborhoods = neighborhood.sort_values(
        ["amr_accessibility_rate", "n_buildings"], ascending=[True, False]
    ).head(5)
    for _, row in bottom_neighborhoods.iterrows():
        lines.append(
            f"- {row['area_name']}: {fmt_rate(row['amr_accessibility_rate'])} "
            f"({int(row['n_buildings'])} buildings), dominant land use: {row['dominant_landuse']}"
        )

    lines += [
        "",
        "Important interpretation note:",
        "This is a baseline using existing geographic units. The next research step is to compare these",
        "official/current boundaries with data-driven AMR-friendly zones discovered from building-level accessibility.",
    ]
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare(read_input())

    borough = aggregate_one(df, "area_borough", "borough")
    neighborhood = aggregate_one(df, "area_neighborhood", "neighborhood")
    zipcode = aggregate_one(df, "area_zipcode", "zipcode")
    all_areas = pd.concat([borough, neighborhood, zipcode], ignore_index=True)

    borough.to_csv(BOROUGH_CSV, index=False)
    neighborhood.to_csv(NEIGHBORHOOD_CSV, index=False)
    zipcode.to_csv(ZIPCODE_CSV, index=False)
    all_areas.to_csv(ALL_AREAS_CSV, index=False)
    top_bottom(all_areas).to_csv(TOP_BOTTOM_CSV, index=False)
    write_report(borough, neighborhood, zipcode)

    print(f"Input buildings: {len(df):,}")
    print(f"Borough areas: {len(borough):,}")
    print(f"Neighborhood areas: {len(neighborhood):,}")
    print(f"ZIP code areas: {len(zipcode):,}")
    print(f"Saved: {ALL_AREAS_CSV}")
    print(f"Saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
