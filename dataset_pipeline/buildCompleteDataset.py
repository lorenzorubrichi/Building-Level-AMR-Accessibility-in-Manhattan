
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

import Dataset as ds
from last_meter_model_utils import MODEL_TARGET_SPECS, TRAINED_MODEL_DIR, load_model_bundle, predict_with_bundle


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"

COMPLETE_FEATURES_XLSX = MODEL_DIR / "OUTPUTamr_features_complete.xlsx"
COMPLETE_DATASET_XLSX = MODEL_DIR / "complete_last_meter_dataset.xlsx"
COMPLETE_DATASET_CSV = MODEL_DIR / "complete_last_meter_dataset.csv"
COMPLETE_DATASET_EXTENDED_CSV = MODEL_DIR / "complete_last_meter_dataset_extended.csv"
PUBLIC_REPO_DIR = BASE_DIR.parent / "last-meter-nyc-public"
PUBLIC_SRC_DIR = PUBLIC_REPO_DIR / "src"
PUBLIC_BASE_SCHEMA_CSV = PUBLIC_REPO_DIR / "data" / "schema" / "base_fields.csv"
PUBLIC_BASE_OUTPUT_CSV = MODEL_DIR / "complete_last_meter_dataset_public.csv"
PUBLIC_BASE_DICTIONARY_CSV = MODEL_DIR / "complete_last_meter_dataset_dictionary.csv"

AMR_SHEET = "amr_features"
CAR_SHEET = "car_features"


def export_public_schema_view(input_csv: Path, output_csv: Path, dictionary_csv: Path, schema_csv: Path) -> None:
    """Create the filtered public dataset view selected in data/schema."""
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
            "Public schema columns missing from complete dataset: "
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


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    if "bin" not in df.columns:
        raise ValueError(f"Missing 'bin' column in {path.name}:{sheet_name}")
    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["bin"]).drop_duplicates(subset=["bin"]).reset_index(drop=True)
    return df


def concat_gdfs(frames: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    valid = [gdf for gdf in frames if gdf is not None and not gdf.empty]
    if not valid:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=ds.OUTPUT_CRS)
    merged = pd.concat(valid, ignore_index=True)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=valid[0].crs or ds.OUTPUT_CRS)


def patch_dataset_module(feature_output_path: Path) -> dict[str, object]:
    original = {
        "SAVE_OUTPUTS": ds.SAVE_OUTPUTS,
        "OUTPUTamr_features_XLSX": ds.OUTPUTamr_features_XLSX,
    }
    ds.SAVE_OUTPUTS = False
    ds.OUTPUTamr_features_XLSX = feature_output_path
    return original


def restore_dataset_module(original: dict[str, object]) -> None:
    ds.SAVE_OUTPUTS = original["SAVE_OUTPUTS"]
    ds.OUTPUTamr_features_XLSX = original["OUTPUTamr_features_XLSX"]


def build_citywide_layers(feature_output_path: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    original = patch_dataset_module(feature_output_path)
    try:
        neighborhoods = ds.load_neighborhoods_geojson(ds.NTA_GEOJSON)
        building_frames: list[gpd.GeoDataFrame] = []
        street_context_frames: list[gpd.GeoDataFrame] = []
        population_frames: list[pd.DataFrame] = []

        for borough in ds.BOROUGH_TO_CODE.keys():
            print(f"\n================ CITYWIDE BOROUGH: {borough} ================")
            selected_neighborhoods = neighborhoods[neighborhoods["boroname"] == borough].copy()
            selected_polygon = selected_neighborhoods.union_all()
            selection = {
                "borough": borough,
                "neighborhoods": selected_neighborhoods["ntaname"].dropna().astype(str).tolist(),
            }

            gdf_sidewalks = ds.process_sidewalks(selected_neighborhoods, selected_polygon)
            gdf_buildings, _ = ds.process_buildings(borough, selected_neighborhoods, selected_polygon, gdf_sidewalks)
            gdf_streets = ds.process_streets(borough, selected_neighborhoods, selected_polygon)
            gdf_parking_reg = ds.process_parking_regulations(borough, selected_neighborhoods, selected_polygon)
            gdf_parking_meters = ds.process_parking_meters(borough, selected_neighborhoods, selected_polygon)
            gdf_parking_blockfaces = ds.process_parking_blockfaces(borough, selected_neighborhoods, selected_polygon)
            gdf_parking_ratezones = ds.process_parking_ratezones(borough, selected_neighborhoods, selected_polygon)
            gdf_pedestrian = ds.process_pedestrian_mobility(borough, selected_neighborhoods, selected_polygon)
            gdf_street_context = ds.process_street_context(
                gdf_streets,
                gdf_parking_reg,
                gdf_parking_meters,
                gdf_parking_blockfaces,
                gdf_parking_ratezones,
                gdf_pedestrian,
                gdf_sidewalks,
            )
            df_population = ds.process_population(selection)

            building_frames.append(gdf_buildings)
            street_context_frames.append(gdf_street_context)
            if not df_population.empty:
                population_frames.append(df_population)

        all_buildings = concat_gdfs(building_frames)
        all_street_context = concat_gdfs(street_context_frames)
        all_population = pd.concat(population_frames, ignore_index=True) if population_frames else pd.DataFrame()

        ds.export_amr_features(all_buildings, all_street_context, all_population, pd.DataFrame())
        return all_buildings, all_street_context, all_population
    finally:
        restore_dataset_module(original)


def build_complete_dataset(feature_output_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_buildings, _, _ = build_citywide_layers(feature_output_path)

    building_ref = pd.DataFrame(all_buildings.drop(columns=["geometry"], errors="ignore")).copy()
    building_ref["bin"] = pd.to_numeric(building_ref["bin"], errors="coerce").astype("Int64")
    building_ref = building_ref.dropna(subset=["bin"]).drop_duplicates(subset=["bin"]).reset_index(drop=True)

    amr_features = read_sheet(feature_output_path, AMR_SHEET)
    car_features = read_sheet(feature_output_path, CAR_SHEET)

    final_df = building_ref.merge(amr_features, on="bin", how="left")
    final_df = final_df.merge(car_features, on="bin", how="left")

    chosen_rows: list[dict[str, object]] = []
    metric_frames: list[pd.DataFrame] = []

    for target_key, spec in MODEL_TARGET_SPECS.items():
        bundle_path = TRAINED_MODEL_DIR / spec["bundle_name"]
        bundle = load_model_bundle(bundle_path)
        output_col = spec["output_column"]
        source_col = spec["source_column"]
        preds = predict_with_bundle(bundle, final_df)
        final_df[output_col] = preds
        final_df[source_col] = f"predicted_{bundle['model_name']}"
        chosen_rows.append(
            {
                "target_key": target_key,
                "training_target": bundle.get("target_col"),
                "selected_model": bundle["model_name"],
                "bundle_file": bundle_path.name if bundle_path.exists() else f"fallback:{bundle_path.name}",
                "n_training_rows": bundle.get("n_rows"),
                "feature_columns": ", ".join(bundle.get("feature_cols", [])),
            }
        )
        metric_records = bundle.get("metrics", [])
        if metric_records:
            metric_frames.append(pd.DataFrame(metric_records))

    rename_map = {
        "delivery_lon": "address_lon",
        "delivery_lat": "address_lat",
    }
    final_df = final_df.rename(columns=rename_map)
    extended_df = final_df.copy()

    preferred_cols = [
        "bin",
        "borough",
        "neighborhood",
        "address_lon",
        "address_lat",
        "landuse",
        "bldgclass",
        "height_roof",
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
        "raw_n_active_meters",
        "ParkingScarcity_advantage",
        "raw_n_regulation_signs",
        "CurbRestriction_advantage",
        "raw_bf_vehicle_ty",
        "CommercialCurbContext",
        "raw_number_park_lanes",
        "raw_curb_crowding_sum",
        "CurbCrowdingPenalty",
        "car_last_meter_mean_s",
        "car_last_meter_source",
        "amr_last_meter_mean_s",
        "amr_last_meter_source",
    ]
    existing = [col for col in preferred_cols if col in final_df.columns]
    final_df = final_df[existing].copy()

    chosen_df = pd.DataFrame(chosen_rows)
    metrics_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    return final_df, extended_df, chosen_df, metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a complete NYC building dataset with engineered features and predicted last-meter times.")
    parser.add_argument("--features-output", type=Path, default=COMPLETE_FEATURES_XLSX, help="Output workbook for citywide engineered features.")
    parser.add_argument("--dataset-output", type=Path, default=COMPLETE_DATASET_XLSX, help="Output Excel workbook for the complete predicted dataset.")
    parser.add_argument("--csv-output", type=Path, default=COMPLETE_DATASET_CSV, help="Output CSV for the complete predicted dataset.")
    parser.add_argument("--extended-csv-output", type=Path, default=COMPLETE_DATASET_EXTENDED_CSV, help="Output extended CSV with all derived/technical columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    args.dataset_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.extended_csv_output.parent.mkdir(parents=True, exist_ok=True)

    final_df, extended_df, chosen_df, metrics_df = build_complete_dataset(args.features_output)

    with pd.ExcelWriter(args.dataset_output) as writer:
        final_df.to_excel(writer, index=False, sheet_name="complete_dataset")
        chosen_df.to_excel(writer, index=False, sheet_name="chosen_models")
        metrics_df.to_excel(writer, index=False, sheet_name="model_metrics")

    final_df.to_csv(args.csv_output, index=False)
    extended_df.to_csv(args.extended_csv_output, index=False)
    export_public_schema_view(
        args.extended_csv_output,
        PUBLIC_BASE_OUTPUT_CSV,
        PUBLIC_BASE_DICTIONARY_CSV,
        PUBLIC_BASE_SCHEMA_CSV,
    )

    print(f"[COMPLETE FEATURES] Saved: {args.features_output}")
    print(f"[COMPLETE DATASET] Saved: {args.dataset_output}")
    print(f"[COMPLETE DATASET] Saved CSV: {args.csv_output}")
    print(f"[COMPLETE DATASET] Saved extended CSV: {args.extended_csv_output}")
    print(f"[COMPLETE DATASET] Rows: {len(final_df)}")


if __name__ == "__main__":
    main()
