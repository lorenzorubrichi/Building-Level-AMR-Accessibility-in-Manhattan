from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from amr_last_meter_sim import (
    AMR_FEATURES_SHEET,
    AMR_SIDEWALK_SPEED_MPS,
    CUSTOMER_ELEVATOR_FLOOR_TIME_S,
    CUSTOMER_ELEVATOR_WAIT_RANGE_S,
    CUSTOMER_HANDOFF_TIME_S,
    CUSTOMER_RESPONSE_PROB,
    CUSTOMER_RESPONSE_TIMEOUT_S,
    CUSTOMER_WALKING_SPEED_MPS,
    HELPER_SEARCH_TIMEOUT_S,
    HELPER_WALKING_SPEED_MPS,
    MAX_HELP_CYCLES,
    OUTPUT_AMR_STATS_XLSX,
)
from building_scene_preview import (
    AMR_FEATURES_XLSX,
    CAR_FEATURES_SHEET,
    DROP_OFF_TIME_S,
    ELEVATOR_FLOOR_TIME_S,
    FT_TO_M,
    INDOOR_WALKING_SPEED_MPS,
    MAX_OCCUPANCY_RATIO,
    MODEL_DIR,
    OUTPUT_STATS_XLSX as CAR_STATS_XLSX,
    PARKING_DEPTH_FT,
    PARKING_GAP_FT,
    PARKING_LENGTH_FT,
    WALKING_SPEED_MPS,
    delivery_point_from_building,
    entrance_point_from_building,
    landuse_wait_class,
    load_buildings_and_streets,
)
from last_meter_model_utils import MODEL_TARGET_SPECS, TRAINED_MODEL_DIR, load_model_bundle, predict_with_bundle


FINAL_OUTPUT_XLSX = MODEL_DIR / "final_last_meter_database.xlsx"
FINAL_OUTPUT_CSV = MODEL_DIR / "final_last_meter_database.csv"


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    if "bin" not in df.columns:
        raise ValueError(f"Missing 'bin' column in {path.name}:{sheet_name}")
    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["bin"]).copy()
    df = df.drop_duplicates(subset=["bin"]).reset_index(drop=True)
    return df


def read_optional_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["bin"])
    return read_sheet(path, sheet_name)


def pick_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def prepare_building_reference() -> pd.DataFrame:
    buildings, _ = load_buildings_and_streets()
    buildings = buildings.copy()
    buildings["bin"] = pd.to_numeric(buildings["bin"], errors="coerce").astype("Int64")
    buildings = buildings.dropna(subset=["bin"]).copy()

    raw_rows = []
    for _, row in buildings.iterrows():
        delivery_point = delivery_point_from_building(row)
        entrance_point = entrance_point_from_building(row.geometry, delivery_point)
        raw_rows.append(
            {
                "bin": int(row["bin"]),
                "entrance_distance_m": float(entrance_point.distance(delivery_point) * FT_TO_M),
                "landuse_wait_class": landuse_wait_class(row.get("landuse")),
            }
        )
    derived = pd.DataFrame(raw_rows)

    static_candidates = [
        "bin",
        "delivery_lon",
        "delivery_lat",
        "landuse",
        "numfloors",
        "height_roof",
        "street_context_id",
        "boroname",
        "borough",
        "nta_name",
        "neighborhood",
        "stname_lab",
        "street_name",
        "street_label",
        "ped_category",
        "ped_rank",
    ]
    static_cols = pick_existing_columns(buildings.drop(columns="geometry"), static_candidates)
    static_df = pd.DataFrame(buildings[static_cols]).drop_duplicates(subset=["bin"]).reset_index(drop=True)
    return static_df.merge(derived, on="bin", how="left")


def prepare_car_outputs(car_stats: pd.DataFrame) -> pd.DataFrame:
    columns = pick_existing_columns(
        car_stats,
        [
            "bin",
            "parking_capacity",
            "base_occupancy_ratio",
            "EntranceDistance_norm",
            "ShapePenalty_norm",
            "BuildingTypePenalty_norm",
            "base_n_valid_runs",
            "base_distance_mean_m",
            "base_distance_std_m",
            "base_indoor_distance_mean_m",
            "base_indoor_distance_std_m",
            "base_floor_mean",
            "base_floor_std",
            "base_time_mean_s",
            "base_time_std_s",
        ],
    )
    df = car_stats[columns].copy()
    rename_map = {
        "base_n_valid_runs": "car_valid_runs",
        "base_distance_mean_m": "car_walk_distance_mean_m",
        "base_distance_std_m": "car_walk_distance_std_m",
        "base_indoor_distance_mean_m": "car_indoor_distance_mean_m",
        "base_indoor_distance_std_m": "car_indoor_distance_std_m",
        "base_floor_mean": "car_floor_mean",
        "base_floor_std": "car_floor_std",
        "base_time_mean_s": "car_last_meter_mean_s",
        "base_time_std_s": "car_last_meter_std_s",
    }
    return df.rename(columns=rename_map)


def prepare_amr_outputs(amr_stats: pd.DataFrame) -> pd.DataFrame:
    columns = pick_existing_columns(
        amr_stats,
        [
            "bin",
            "base_n_runs",
            "base_response_rate",
            "base_helper_rate",
            "base_responded_runs",
            "base_helper_runs",
            "base_helper_probability_base_mean",
            "base_helper_probability_used_mean",
            "base_helper_cycles_mean",
            "base_sidewalk_to_entrance_mean_m",
            "base_sidewalk_to_entrance_std_m",
            "base_amr_approach_time_mean_s",
            "base_amr_approach_time_std_s",
            "base_response_time_mean_s",
            "base_response_time_std_s",
            "base_customer_path_mean_m",
            "base_customer_path_std_m",
            "base_helper_path_mean_m",
            "base_helper_path_std_m",
            "base_customer_elevator_wait_mean_s",
            "base_customer_elevator_wait_std_s",
            "base_helper_elevator_wait_mean_s",
            "base_helper_elevator_wait_std_s",
            "base_floor_mean",
            "base_floor_std",
            "base_amr_time_mean_s",
            "base_amr_time_std_s",
        ],
    )
    df = amr_stats[columns].copy()
    rename_map = {
        "base_n_runs": "amr_runs",
        "base_response_rate": "amr_response_rate",
        "base_helper_rate": "amr_helper_rate",
        "base_responded_runs": "amr_responded_runs",
        "base_helper_runs": "amr_helper_runs",
        "base_helper_probability_base_mean": "amr_helper_probability_base_mean",
        "base_helper_probability_used_mean": "amr_helper_probability_used_mean",
        "base_helper_cycles_mean": "amr_helper_cycles_mean",
        "base_sidewalk_to_entrance_mean_m": "amr_sidewalk_to_entrance_mean_m",
        "base_sidewalk_to_entrance_std_m": "amr_sidewalk_to_entrance_std_m",
        "base_amr_approach_time_mean_s": "amr_approach_time_mean_s",
        "base_amr_approach_time_std_s": "amr_approach_time_std_s",
        "base_response_time_mean_s": "amr_response_time_mean_s",
        "base_response_time_std_s": "amr_response_time_std_s",
        "base_customer_path_mean_m": "amr_customer_path_mean_m",
        "base_customer_path_std_m": "amr_customer_path_std_m",
        "base_helper_path_mean_m": "amr_helper_path_mean_m",
        "base_helper_path_std_m": "amr_helper_path_std_m",
        "base_customer_elevator_wait_mean_s": "amr_customer_elevator_wait_mean_s",
        "base_customer_elevator_wait_std_s": "amr_customer_elevator_wait_std_s",
        "base_helper_elevator_wait_mean_s": "amr_helper_elevator_wait_mean_s",
        "base_helper_elevator_wait_std_s": "amr_helper_elevator_wait_std_s",
        "base_floor_mean": "amr_floor_mean",
        "base_floor_std": "amr_floor_std",
        "base_amr_time_mean_s": "amr_last_meter_mean_s",
        "base_amr_time_std_s": "amr_last_meter_std_s",
    }
    return df.rename(columns=rename_map)


def build_parameters_table(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {"group": "builder", "parameter": "use_pretrained_models", "value": True},
        {"group": "builder", "parameter": "trained_model_dir", "value": str(TRAINED_MODEL_DIR)},
        {"group": "car_simulation", "parameter": "walking_speed_mps", "value": WALKING_SPEED_MPS},
        {"group": "car_simulation", "parameter": "indoor_walking_speed_mps", "value": INDOOR_WALKING_SPEED_MPS},
        {"group": "car_simulation", "parameter": "drop_off_time_s", "value": DROP_OFF_TIME_S},
        {"group": "car_simulation", "parameter": "elevator_floor_time_s", "value": ELEVATOR_FLOOR_TIME_S},
        {"group": "car_simulation", "parameter": "max_occupancy_ratio", "value": MAX_OCCUPANCY_RATIO},
        {"group": "car_simulation", "parameter": "parking_length_ft", "value": PARKING_LENGTH_FT},
        {"group": "car_simulation", "parameter": "parking_gap_ft", "value": PARKING_GAP_FT},
        {"group": "car_simulation", "parameter": "parking_depth_ft", "value": PARKING_DEPTH_FT},
        {"group": "amr_simulation", "parameter": "amr_sidewalk_speed_mps", "value": AMR_SIDEWALK_SPEED_MPS},
        {"group": "amr_simulation", "parameter": "customer_response_timeout_s", "value": CUSTOMER_RESPONSE_TIMEOUT_S},
        {"group": "amr_simulation", "parameter": "helper_search_timeout_s", "value": HELPER_SEARCH_TIMEOUT_S},
        {"group": "amr_simulation", "parameter": "max_help_cycles", "value": MAX_HELP_CYCLES},
        {"group": "amr_simulation", "parameter": "customer_handoff_time_s", "value": CUSTOMER_HANDOFF_TIME_S},
        {"group": "amr_simulation", "parameter": "customer_walking_speed_mps", "value": CUSTOMER_WALKING_SPEED_MPS},
        {"group": "amr_simulation", "parameter": "helper_walking_speed_mps", "value": HELPER_WALKING_SPEED_MPS},
        {"group": "amr_simulation", "parameter": "customer_elevator_floor_time_s", "value": CUSTOMER_ELEVATOR_FLOOR_TIME_S},
        {"group": "amr_simulation", "parameter": "customer_response_probabilities", "value": json.dumps(CUSTOMER_RESPONSE_PROB)},
        {"group": "amr_simulation", "parameter": "customer_elevator_wait_ranges_s", "value": json.dumps(CUSTOMER_ELEVATOR_WAIT_RANGE_S)},
    ]
    return pd.DataFrame(rows)


def build_column_dictionary() -> pd.DataFrame:
    rows = [
        {"column": "bin", "description": "Unique building identifier (BIN).", "source": "common"},
        {"column": "address_lon", "description": "Address point longitude used by the pipeline.", "source": "buildings/amr_features"},
        {"column": "address_lat", "description": "Address point latitude used by the pipeline.", "source": "buildings/amr_features"},
        {"column": "borough", "description": "Selected borough.", "source": "buildings"},
        {"column": "neighborhood", "description": "Selected neighborhood.", "source": "buildings"},
        {"column": "landuse", "description": "PLUTO land use code.", "source": "buildings/pluto"},
        {"column": "raw_numfloors", "description": "Raw number of floors from the integrated dataset.", "source": "amr_features"},
        {"column": "height_roof", "description": "Roof height from the building dataset.", "source": "buildings"},
        {"column": "raw_distance_to_sidewalk_m", "description": "Raw distance between address point and nearest sidewalk in meters.", "source": "amr_features"},
        {"column": "car_last_meter_mean_s", "description": "Mean car last-meter time.", "source": "car_simulation"},
        {"column": "car_last_meter_source", "description": "Origin of the car time value: simulated or predicted by the selected regression model.", "source": "builder"},
        {"column": "amr_last_meter_mean_s", "description": "Mean AMR last-meter time.", "source": "amr_simulation"},
        {"column": "amr_last_meter_source", "description": "Origin of the AMR time value: simulated or predicted by the selected regression model.", "source": "builder"},
    ]
    return pd.DataFrame(rows)


def build_database() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    building_ref = prepare_building_reference()
    amr_features = read_sheet(AMR_FEATURES_XLSX, AMR_FEATURES_SHEET)
    car_features = read_sheet(AMR_FEATURES_XLSX, CAR_FEATURES_SHEET)

    amr_feature_cols = pick_existing_columns(
        amr_features,
        [
            "bin",
            "raw_numfloors",
            "raw_distance_to_sidewalk_m",
            "a1_Floors_norm",
            "a2_RoadToDeliveryDistance_norm",
            "a3_AddressUncertainty",
            "b1_Population_norm",
            "b2_PedestrianPresence_norm",
            "b3_UrbanActivity_norm",
            "c1_PedestrianPenalty",
            "c2_SidewalkAbsencePenalty",
        ],
    )
    car_feature_cols = pick_existing_columns(
        car_features,
        [
            "bin",
            "raw_n_active_meters",
            "ParkingScarcity_advantage",
            "raw_n_regulation_signs",
            "CurbRestriction_advantage",
            "raw_bf_vehicle_ty",
            "CommercialCurbContext",
            "raw_number_park_lanes",
            "raw_curb_crowding_sum",
            "CurbCrowdingPenalty",
        ],
    )

    final_df = building_ref.copy()
    final_df = final_df.merge(amr_features[amr_feature_cols], on="bin", how="left")
    final_df = final_df.merge(car_features[car_feature_cols], on="bin", how="left")
    car_stats = read_optional_sheet(CAR_STATS_XLSX, "car_last_meter")
    amr_stats = read_optional_sheet(OUTPUT_AMR_STATS_XLSX, "amr_last_meter")
    final_df = final_df.merge(prepare_car_outputs(car_stats), on="bin", how="left")
    final_df = final_df.merge(prepare_amr_outputs(amr_stats), on="bin", how="left")

    chosen_model_rows: list[dict] = []
    metric_rows: list[dict] = []

    for output_name, spec in MODEL_TARGET_SPECS.items():
        bundle_path = TRAINED_MODEL_DIR / spec["bundle_name"]
        bundle = load_model_bundle(bundle_path)
        output_col = spec["output_column"]
        source_col = spec["source_column"]

        if output_col not in final_df.columns:
            final_df[output_col] = pd.NA
        if source_col not in final_df.columns:
            final_df[source_col] = None
        final_df.loc[final_df[output_col].notna(), source_col] = "simulated"

        missing_mask = final_df[output_col].isna()
        if missing_mask.any():
            preds = predict_with_bundle(bundle, final_df.loc[missing_mask])
            final_df.loc[missing_mask, output_col] = preds.values
            final_df.loc[missing_mask, source_col] = f"predicted_{bundle['model_name']}"

        chosen_model_rows.append(
            {
                "target": output_name,
                "chosen_model": bundle["model_name"],
                "bundle_file": spec["bundle_name"],
                "training_target": bundle["target_col"],
                "n_rows": bundle["n_rows"],
            }
        )
        metric_rows.extend(bundle.get("metrics", []))

    final_df = final_df.rename(
        columns={
            "delivery_lon": "address_lon",
            "delivery_lat": "address_lat",
        }
    )

    preferred_columns = [
        "bin",
        "address_lon",
        "address_lat",
        "borough",
        "neighborhood",
        "landuse",
        "raw_numfloors",
        "height_roof",
        "raw_distance_to_sidewalk_m",
        "car_last_meter_mean_s",
        "car_last_meter_source",
        "amr_last_meter_mean_s",
        "amr_last_meter_source",
    ]
    existing_preferred = [col for col in preferred_columns if col in final_df.columns]
    final_df = final_df[existing_preferred].copy()
    final_df = final_df.sort_values("bin").reset_index(drop=True)
    chosen_models_df = pd.DataFrame(chosen_model_rows)
    metrics_df = pd.DataFrame(metric_rows)
    return final_df, chosen_models_df, metrics_df


def write_outputs(
    final_df: pd.DataFrame,
    parameters_df: pd.DataFrame,
    dictionary_df: pd.DataFrame,
    chosen_models_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_xlsx: Path,
) -> None:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_xlsx.with_suffix(".csv")

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="last_meter_database", index=False)
        parameters_df.to_excel(writer, sheet_name="scenario_parameters", index=False)
        dictionary_df.to_excel(writer, sheet_name="column_dictionary", index=False)
        chosen_models_df.to_excel(writer, sheet_name="chosen_models", index=False)
        metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)

    final_df.to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the final last-meter database using pretrained regression models."
    )
    parser.add_argument("--output", type=Path, default=FINAL_OUTPUT_XLSX, help="Output Excel path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_df, chosen_models_df, metrics_df = build_database()
    parameters_df = build_parameters_table(args)
    dictionary_df = build_column_dictionary()
    write_outputs(final_df, parameters_df, dictionary_df, chosen_models_df, metrics_df, args.output)
    print(f"Excel saved: {args.output}")
    print(f"CSV saved: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
