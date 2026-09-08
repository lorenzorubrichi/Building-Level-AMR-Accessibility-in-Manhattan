"""Merge Street View AI features into the extended last-meter dataset.

Inputs:
- streetview+AI/output/streetview_visual_features_api.csv
- dataset/Model/complete_last_meter_dataset_extended.csv

Outputs:
- AI+regression/complete_last_meter_dataset_ai_subset.csv
- AI+regression/complete_last_meter_dataset_ai_subset.xlsx
- AI+regression/feature_definitions.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
AI_FEATURES_CSV = Path(
    r"C:\Users\loren\Desktop\RPI_DOC\lavori\streetview+AI\output\streetview_visual_features_api.csv"
)
BASE_DATASET_CSV = Path(
    r"C:\Users\loren\Desktop\RPI_DOC\lavori\dataset\Model\complete_last_meter_dataset_extended.csv"
)
CAR_AI_STATS_XLSX = Path(
    r"C:\Users\loren\Desktop\RPI_DOC\lavori\dataset\Model\ai_penalty_shared\car_last_meter_stats_ai_penalty.xlsx"
)
AMR_AI_STATS_XLSX = Path(
    r"C:\Users\loren\Desktop\RPI_DOC\lavori\dataset\Model\ai_penalty_shared\amr_last_meter_stats_ai_penalty.xlsx"
)
ADDRESS_POINTS_CSV = Path(
    r"C:\Users\loren\Desktop\RPI_DOC\lavori\dataset\rawdata\AddressPoint_full.csv"
)
OUTPUT_CSV = BASE_DIR / "complete_last_meter_dataset_ai_subset.csv"
OUTPUT_XLSX = BASE_DIR / "complete_last_meter_dataset_ai_subset.xlsx"
FALLBACK_OUTPUT_CSV = BASE_DIR / "complete_last_meter_dataset_ai_subset_updated.csv"
FALLBACK_OUTPUT_XLSX = BASE_DIR / "complete_last_meter_dataset_ai_subset_updated.xlsx"
FEATURES_TXT = BASE_DIR / "feature_definitions.txt"
PUBLIC_REPO_DIR = BASE_DIR.parent / "last-meter-nyc-public"
PUBLIC_SRC_DIR = PUBLIC_REPO_DIR / "src"
PUBLIC_AI_SCHEMA_CSV = PUBLIC_REPO_DIR / "data" / "schema" / "ai_fields.csv"
PUBLIC_AI_OUTPUT_CSV = BASE_DIR / "complete_last_meter_dataset_ai_subset_public.csv"
PUBLIC_AI_DICTIONARY_CSV = BASE_DIR / "complete_last_meter_dataset_ai_subset_dictionary.csv"

AMR_MODEL_FEATURES = [
    "a1_Floors_norm",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
    "b1_Population_norm",
    "b2_PedestrianPresence_norm",
    "b3_UrbanActivity_norm",
]

CAR_MODEL_FEATURES = [
    "a1_Floors_norm",
    "CurbCrowdingPenalty",
    "a2_RoadToDeliveryDistance_norm",
    "ShapePenalty_norm",
    "BuildingTypePenalty_norm",
]


def export_public_schema_view(input_csv: Path, output_csv: Path, dictionary_csv: Path, schema_csv: Path) -> None:
    """Create the filtered public AI dataset view selected in data/schema."""
    if not schema_csv.exists():
        print(f"[PUBLIC EXPORT] Skipped: missing schema {schema_csv}")
        return
    if not PUBLIC_SRC_DIR.exists():
        print(f"[PUBLIC EXPORT] Skipped: missing public source folder {PUBLIC_SRC_DIR}")
        return

    sys.path.insert(0, str(PUBLIC_SRC_DIR))
    from last_meter_nyc.feature_names import FEATURE_DESCRIPTIONS, rename_for_publication

    df = pd.read_csv(input_csv, low_memory=False)
    rename_map = rename_for_publication(list(df.columns))
    public_df = df.rename(columns=rename_map)

    schema = pd.read_csv(schema_csv)
    if "column" not in schema.columns:
        raise ValueError(f"Schema must contain a 'column' field: {schema_csv}")
    if "description" not in schema.columns:
        schema["description"] = ""
    if "order" in schema.columns:
        schema = schema.sort_values("order", kind="stable")

    schema["column"] = schema["column"].astype(str).str.strip()
    schema["description"] = schema["description"].fillna("").astype(str).str.strip()
    schema = schema[schema["column"] != ""].copy()

    requested_columns = list(schema["column"])
    missing_columns = [column for column in requested_columns if column not in public_df.columns]
    if missing_columns:
        raise ValueError(
            "Public schema columns missing from AI subset: "
            + ", ".join(missing_columns[:30])
            + (" ..." if len(missing_columns) > 30 else "")
        )

    public_df = public_df[requested_columns].copy()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    public_df.to_csv(output_csv, index=False)

    schema_descriptions = {
        str(row["column"]): str(row["description"])
        for _, row in schema.iterrows()
        if str(row["description"]).strip()
    }
    dictionary_rows = []
    for column in public_df.columns:
        dictionary_rows.append(
            {
                "column": column,
                "legacy_name": next((old for old, new in rename_map.items() if new == column), ""),
                "description": schema_descriptions.get(column, FEATURE_DESCRIPTIONS.get(column, "")),
            }
        )
    dictionary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dictionary_rows).to_csv(dictionary_csv, index=False)

    print(f"[PUBLIC EXPORT] Saved filtered CSV: {output_csv}")
    print(f"[PUBLIC EXPORT] Saved dictionary: {dictionary_csv}")


def build_ai_time_table() -> pd.DataFrame:
    if not CAR_AI_STATS_XLSX.exists():
        raise FileNotFoundError(f"Missing car AI stats workbook: {CAR_AI_STATS_XLSX}")
    if not AMR_AI_STATS_XLSX.exists():
        raise FileNotFoundError(f"Missing AMR AI stats workbook: {AMR_AI_STATS_XLSX}")

    car_df = pd.read_excel(CAR_AI_STATS_XLSX)
    amr_df = pd.read_excel(AMR_AI_STATS_XLSX)

    for df in [car_df, amr_df]:
        if "bin" not in df.columns:
            raise ValueError("AI stats workbooks must contain a 'bin' column")
        df["bin"] = normalize_bin(df["bin"])

    car_cols = [
        "bin",
        "base_time_mean_s",
        "base_time_std_s",
        "car_ai_penalty_s",
        "ai_stairs_present",
        "ai_gate_present",
        "ai_ramp_present",
        "ai_access_barrier_mean",
    ]
    amr_cols = [
        "bin",
        "amr_time_mean_s",
        "amr_time_std_s",
        "amr_ai_penalty_s",
        "ai_stairs_present",
        "ai_gate_present",
        "ai_ramp_present",
        "ai_access_barrier_mean",
    ]

    car_df = car_df[[col for col in car_cols if col in car_df.columns]].copy()
    amr_df = amr_df[[col for col in amr_cols if col in amr_df.columns]].copy()

    merged = car_df.merge(amr_df, on="bin", how="outer", suffixes=("_car", "_amr"))
    return merged


def normalize_bin(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def keep_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def first_existing(series_candidates: list[pd.Series]) -> pd.Series:
    result = series_candidates[0].copy()
    for candidate in series_candidates[1:]:
        result = result.where(result.notna() & (result.astype(str) != ""), candidate)
    return result


def canonicalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_map = {
        "raw_numfloors": ["raw_numfloors", "raw_numfloors_x", "raw_numfloors_y"],
        "raw_distance_to_sidewalk_m": ["raw_distance_to_sidewalk_m", "raw_distance_to_sidewalk_m_x", "raw_distance_to_sidewalk_m_y"],
        "landuse": ["landuse", "landuse_x", "landuse_y"],
        "a1_Floors_norm": ["a1_Floors_norm", "a1_Floors_norm_x", "a1_Floors_norm_y"],
        "a2_RoadToDeliveryDistance_norm": [
            "a2_RoadToDeliveryDistance_norm",
            "a2_RoadToDeliveryDistance_norm_x",
            "a2_RoadToDeliveryDistance_norm_y",
        ],
        "ShapePenalty_norm": ["ShapePenalty_norm", "ShapePenalty_norm_x", "ShapePenalty_norm_y"],
        "BuildingTypePenalty_norm": [
            "BuildingTypePenalty_norm",
            "BuildingTypePenalty_norm_x",
            "BuildingTypePenalty_norm_y",
        ],
    }

    for canonical_name, aliases in alias_map.items():
        present = [col for col in aliases if col in out.columns]
        if not present:
            continue
        candidate_series = [out[col] for col in present]
        out[canonical_name] = first_existing(candidate_series)

    return out


def build_primary_address_table() -> pd.DataFrame:
    if not ADDRESS_POINTS_CSV.exists():
        return pd.DataFrame(columns=["bin", "house_number", "street_name", "full_street_name", "zipcode", "full_address"])

    usecols = ["bin", "house_number", "street_name", "full_street_name", "zipcode"]
    addr_df = pd.read_csv(ADDRESS_POINTS_CSV, dtype=str, usecols=usecols)
    addr_df["bin"] = normalize_bin(addr_df["bin"])
    addr_df = addr_df.dropna(subset=["bin"])

    for col in ["house_number", "street_name", "full_street_name", "zipcode"]:
        addr_df[col] = addr_df[col].fillna("").astype(str).str.strip()

    addr_df["full_address"] = (
        addr_df["house_number"].where(addr_df["house_number"] != "", "")
        + " "
        + addr_df["full_street_name"].where(addr_df["full_street_name"] != "", addr_df["street_name"])
    ).str.strip()

    addr_df = addr_df.sort_values(
        by=["bin", "full_address", "zipcode"],
        ascending=[True, True, True],
        na_position="last",
    )
    addr_df = addr_df.drop_duplicates(subset=["bin"], keep="first")
    return addr_df[["bin", "house_number", "street_name", "full_street_name", "zipcode", "full_address"]]


def write_feature_definitions_txt() -> None:
    text = """Normalized feature definitions used in the regression models
========================================================

1. a1_Floors_norm
What it is:
- Normalized number of floors / building verticality.
How it was obtained:
- Raw source: raw_numfloors, taken from numfloors.
- Computation: min-max normalization over the working dataset.

2. a2_RoadToDeliveryDistance_norm
What it is:
- Normalized road-to-delivery distance proxy.
How it was obtained:
- Raw source: building_to_street_distance, created by matching each building delivery point to the nearest street-context geometry.
- Computation: min-max normalization of building_to_street_distance.

3. ShapePenalty_norm
What it is:
- Building shape penalty / elongation proxy.
How it was obtained:
- Raw source: building footprint geometry.
- Computation:
  - compute the minimum rotated rectangle of the building polygon
  - extract long axis and short axis
  - calculate 1 - (short_axis / long_axis)
  - clip to [0,1]

4. BuildingTypePenalty_norm
What it is:
- Rule-based building type difficulty score.
How it was obtained:
- Raw source: landuse
- Computation:
  - landuse 4,5 -> 0.00 (mixed commercial)
  - landuse 1,2,3 -> 0.33 (residential)
  - landuse 6,7 -> 0.66 (industrial / utility)
  - all other / fallback -> 1.00 (public / other)

5. b1_Population_norm
What it is:
- Normalized neighborhood population intensity proxy.
How it was obtained:
- Raw source: population_clean, from the joined population dataset via nta_key, with borough mean fallback when neighborhood population is missing.
- Computation: min-max normalization of population_clean.

6. b2_PedestrianPresence_norm
What it is:
- Pedestrian presence / corridor importance proxy.
How it was obtained:
- Raw source: ped_category from pedestrian mobility data.
- Computation:
  - category-to-score mapping:
    baseline street -> 0.00
    community connector -> 0.25
    neighborhood corridor -> 0.50
    regional corridor -> 0.75
    global corridor -> 1.00
  - with numeric fallback where needed

7. b3_UrbanActivity_norm
What it is:
- Urban activity proxy based on building land use.
How it was obtained:
- Raw source: landuse
- Computation:
  - rule-based mapping from land-use code to activity score
  - example:
    4 -> 0.90
    5 -> 1.00
    6 -> 0.80
    9 -> 0.20
    10/11 -> 0.10

8. CurbCrowdingPenalty
What it is:
- Curbside difficulty proxy used in the car regression.
How it was obtained:
- Raw ingredients:
  - raw_n_active_meters
  - raw_n_regulation_signs
  - raw_bf_vehicle_ty
- Intermediate computations:
  - ParkingScarcity_advantage = inverse min-max normalization of raw_n_active_meters
  - CurbRestriction_advantage = min-max normalization of raw_n_regulation_signs
  - CommercialCurbContext = 1 if vehicle type suggests commercial/truck context, else 0
- Final computation:
  - raw_curb_crowding_sum = ParkingScarcity_advantage + CurbRestriction_advantage + CommercialCurbContext
  - CurbCrowdingPenalty = raw_curb_crowding_sum / 3
"""
    FEATURES_TXT.write_text(text, encoding="utf-8")


def main() -> None:
    if not AI_FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing AI features CSV: {AI_FEATURES_CSV}")
    if not BASE_DATASET_CSV.exists():
        raise FileNotFoundError(f"Missing base dataset CSV: {BASE_DATASET_CSV}")

    ai_df = pd.read_csv(AI_FEATURES_CSV, dtype=str)
    base_df = pd.read_csv(BASE_DATASET_CSV, dtype=str)
    address_df = build_primary_address_table()
    ai_time_df = build_ai_time_table()

    if "bin" not in ai_df.columns:
        raise ValueError("AI features CSV must contain a 'bin' column")
    if "bin" not in base_df.columns:
        raise ValueError("Base dataset CSV must contain a 'bin' column")

    ai_df["bin"] = normalize_bin(ai_df["bin"])
    base_df["bin"] = normalize_bin(base_df["bin"])
    ai_time_df["bin"] = normalize_bin(ai_time_df["bin"])

    ai_df = ai_df.drop_duplicates(subset=["bin"], keep="last")

    merged_df = base_df.merge(ai_df, on="bin", how="inner")
    merged_df = merged_df.merge(address_df, on="bin", how="left")
    merged_df = merged_df.merge(ai_time_df, on="bin", how="left")
    merged_df = canonicalize_feature_columns(merged_df)

    if "base_time_mean_s" in merged_df.columns:
        merged_df["car_last_meter_mean_s"] = merged_df["base_time_mean_s"]
    if "amr_time_mean_s" in merged_df.columns:
        merged_df["amr_last_meter_mean_s"] = merged_df["amr_time_mean_s"]
    merged_df["car_last_meter_source"] = "ai_penalty_simulation"
    merged_df["amr_last_meter_source"] = "ai_penalty_simulation"

    ai_bool_aliases = {
        "stairs_present": ["stairs_present", "ai_stairs_present_car", "ai_stairs_present_amr"],
        "gate_present": ["gate_present", "ai_gate_present_car", "ai_gate_present_amr"],
        "ramp_present": ["ramp_present", "ai_ramp_present_car", "ai_ramp_present_amr"],
        "ai_access_barrier_mean": [
            "ai_access_barrier_mean",
            "ai_access_barrier_mean_car",
            "ai_access_barrier_mean_amr",
        ],
    }
    for canonical_name, aliases in ai_bool_aliases.items():
        present = [col for col in aliases if col in merged_df.columns]
        if present:
            merged_df[canonical_name] = first_existing([merged_df[col] for col in present])

    csv_path = OUTPUT_CSV
    try:
        merged_df.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = FALLBACK_OUTPUT_CSV
        merged_df.to_csv(csv_path, index=False)

    location_cols = keep_existing_columns(
        merged_df,
        [
            "bin",
            "borough",
            "neighborhood",
            "full_address",
            "house_number",
            "street_name",
            "full_street_name",
            "zipcode",
            "centroid_lat",
            "centroid_lon",
            "address_lat",
            "address_lon",
            "raw_numfloors",
            "car_last_meter_source",
            "amr_last_meter_source",
        ],
    )

    amr_feature_cols = [
        "bin",
        "raw_numfloors",
        "raw_distance_to_sidewalk_m",
        "landuse",
        *AMR_MODEL_FEATURES,
    ]

    car_feature_cols = [
        "bin",
        "raw_numfloors",
        "raw_distance_to_sidewalk_m",
        "landuse",
        "raw_n_active_meters",
        "raw_n_regulation_signs",
        "raw_bf_vehicle_ty",
        "raw_number_park_lanes",
        "raw_curb_crowding_sum",
        "ParkingScarcity_advantage",
        "CurbRestriction_advantage",
        "CommercialCurbContext",
        *CAR_MODEL_FEATURES,
    ]

    time_cols = keep_existing_columns(
        merged_df,
        [
            "bin",
            "car_last_meter_mean_s",
            "amr_last_meter_mean_s",
            "base_time_std_s",
            "amr_time_std_s",
            "car_ai_penalty_s",
            "amr_ai_penalty_s",
        ],
    )

    ai_cols = keep_existing_columns(
        merged_df,
        [
            "bin",
            "image_usable",
            "stairs_present",
            "gate_present",
            "ramp_present",
            "amr_can_reach_door",
            "ai_access_barrier_mean",
        ],
    )

    merged_df = ensure_columns(merged_df, amr_feature_cols + car_feature_cols)

    missing_amr = [col for col in AMR_MODEL_FEATURES if col not in merged_df.columns]
    missing_car = [col for col in CAR_MODEL_FEATURES if col not in merged_df.columns]

    workbook_path = OUTPUT_XLSX
    try:
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            merged_df[location_cols].to_excel(writer, index=False, sheet_name="building_info")
            merged_df[amr_feature_cols].to_excel(writer, index=False, sheet_name="amr_features")
            merged_df[car_feature_cols].to_excel(writer, index=False, sheet_name="car_features")
            merged_df[time_cols].to_excel(writer, index=False, sheet_name="times")
            merged_df[ai_cols].to_excel(writer, index=False, sheet_name="ai_features")
    except PermissionError:
        workbook_path = FALLBACK_OUTPUT_XLSX
        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            merged_df[location_cols].to_excel(writer, index=False, sheet_name="building_info")
            merged_df[amr_feature_cols].to_excel(writer, index=False, sheet_name="amr_features")
            merged_df[car_feature_cols].to_excel(writer, index=False, sheet_name="car_features")
            merged_df[time_cols].to_excel(writer, index=False, sheet_name="times")
            merged_df[ai_cols].to_excel(writer, index=False, sheet_name="ai_features")

    write_feature_definitions_txt()
    export_public_schema_view(csv_path, PUBLIC_AI_OUTPUT_CSV, PUBLIC_AI_DICTIONARY_CSV, PUBLIC_AI_SCHEMA_CSV)

    print(f"Saved merged dataset: {csv_path}")
    print(f"Saved merged workbook: {workbook_path}")
    print(f"Saved feature definitions: {FEATURES_TXT}")
    print(f"Base rows: {len(base_df)}")
    print(f"AI feature rows: {len(ai_df)}")
    print(f"Merged rows: {len(merged_df)}")
    print(f"Missing AMR model feature columns in base dataset: {missing_amr}")
    print(f"Missing car model feature columns in base dataset: {missing_car}")


if __name__ == "__main__":
    main()
