"""Build and test a dataset for AMR accessibility logistic regression.

The script creates one row per Street View AI building target and enriches it
with available building, PLUTO, and address attributes. It then runs a first
logistic regression to estimate whether the available fields are associated
with the AI-derived AMR reachability flag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPT_DIR.parent

AI_SUBSET_CSV = WORK_DIR / "AI+regression" / "complete_last_meter_dataset_ai_subset.csv"
BUILDING_CSV = WORK_DIR / "dataset" / "rawdata" / "BUILDING.csv"
PLUTO_CSV = WORK_DIR / "dataset" / "rawdata" / "pluto_full.csv"
ADDRESS_CSV = WORK_DIR / "dataset" / "rawdata" / "AddressPoint_full.csv"

OUTPUT_DIR = SCRIPT_DIR / "output"
DATASET_OUTPUT = OUTPUT_DIR / "amr_accessibility_logistic_dataset.csv"
DICTIONARY_OUTPUT = OUTPUT_DIR / "amr_accessibility_logistic_dataset_dictionary.csv"
SELECTED_FEATURES_OUTPUT = OUTPUT_DIR / "logistic_regression_selected_features.csv"
COEFFICIENTS_OUTPUT = OUTPUT_DIR / "logistic_regression_coefficients.csv"
METRICS_OUTPUT = OUTPUT_DIR / "logistic_regression_metrics.json"

TARGET_SOURCE_COLUMN = "amr_can_reach_door"
TARGET_COLUMN = "target_amr_accessible"
JOIN_KEY = "bin"

GEOMETRY_LIKE_COLUMNS = {
    "the_geom",
    "centroid_geom",
    "address_point_geom",
    "delivery_point_geom",
}

HIGH_CARDINALITY_HINTS = (
    "address",
    "name",
    "geom",
    "date",
    "bbl",
    "nta_key",
    "source_id",
    "objectid",
    "doitt_id",
    "addresspointid",
    "a4id",
    "b7sc",
    "complexid",
)

TARGET_LEAKAGE_COLUMNS = {
    TARGET_SOURCE_COLUMN,
    TARGET_COLUMN,
}

AI_MODEL_EXCLUSION_HINTS = (
    "ai_",
    "streetview",
    "image_usable",
    "stairs",
    "gate",
    "ramp",
    "reach_door",
    "access_barrier",
    "ai_penalty",
    "amr_can_reach",
)

RAW_BASE_COLUMNS = {
    "borough",
    "neighborhood",
    "map_pluto_bbl",
    "height_roof",
    "numfloors",
    "landuse_x",
    "bldgclass",
    "centroid_lon",
    "centroid_lat",
    "has_address_point",
    "delivery_point_source",
    "address_lon",
    "address_lat",
    "construction_year",
    "feature_code",
    "last_status_type",
    "borough_nta",
    "nta_name",
    "shape_area",
    "geometry_source",
    "ground_elevation",
    "length",
    "bbl",
}

RAW_SOURCE_PREFIXES = (
    "building_raw__",
    "pluto_raw__",
    "address_raw__",
)


def normalize_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def read_selected_csv(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def prefixed_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_key: str,
    right_key: str,
    prefix: str,
) -> pd.DataFrame:
    if left_key not in left.columns or right_key not in right.columns:
        return left

    left = left.copy()
    right = right.copy()
    left[left_key] = normalize_key(left[left_key])
    right[right_key] = normalize_key(right[right_key])
    right = right.dropna(subset=[right_key]).drop_duplicates(subset=[right_key], keep="first")

    rename_map = {
        column: f"{prefix}__{column}"
        for column in right.columns
        if column != right_key
    }
    right = right.rename(columns=rename_map)
    return left.merge(right, left_on=left_key, right_on=right_key, how="left").drop(columns=[right_key], errors="ignore")


def add_building_raw(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in read_header(BUILDING_CSV) if column not in GEOMETRY_LIKE_COLUMNS]
    building = read_selected_csv(BUILDING_CSV, usecols=columns)
    return prefixed_merge(df, building, left_key=JOIN_KEY, right_key="bin", prefix="building_raw")


def add_pluto_raw(df: pd.DataFrame) -> pd.DataFrame:
    pluto = read_selected_csv(PLUTO_CSV)
    left_key = "map_pluto_bbl" if "map_pluto_bbl" in df.columns else "bbl"
    return prefixed_merge(df, pluto, left_key=left_key, right_key="bbl", prefix="pluto_raw")


def add_address_raw(df: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in read_header(ADDRESS_CSV) if column not in GEOMETRY_LIKE_COLUMNS]
    address = read_selected_csv(ADDRESS_CSV, usecols=columns)
    return prefixed_merge(df, address, left_key=JOIN_KEY, right_key="bin", prefix="address_raw")


def build_dataset() -> pd.DataFrame:
    ai = read_selected_csv(AI_SUBSET_CSV)
    if JOIN_KEY not in ai.columns:
        raise ValueError(f"AI subset must contain '{JOIN_KEY}'.")
    if TARGET_SOURCE_COLUMN not in ai.columns:
        raise ValueError(f"AI subset must contain target column '{TARGET_SOURCE_COLUMN}'.")

    ai[JOIN_KEY] = normalize_key(ai[JOIN_KEY])
    ai = ai.drop_duplicates(subset=[JOIN_KEY], keep="first").reset_index(drop=True)
    ai[TARGET_COLUMN] = ai[TARGET_SOURCE_COLUMN].astype(str).str.lower().map(
        {"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0}
    )

    if ai[TARGET_COLUMN].isna().any():
        missing = int(ai[TARGET_COLUMN].isna().sum())
        raise ValueError(f"Could not convert {missing} target values in '{TARGET_SOURCE_COLUMN}'.")

    df = ai.copy()
    df = add_building_raw(df)
    df = add_pluto_raw(df)
    df = add_address_raw(df)
    df = remove_duplicate_columns(df)
    return df


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    seen: set[str] = set()
    keep: list[str] = []
    for column in df.columns:
        if column not in seen:
            keep.append(column)
            seen.add(column)
    return df[keep].copy()


def is_model_candidate(column: str, series: pd.Series) -> bool:
    lower = column.lower()
    if column in TARGET_LEAKAGE_COLUMNS:
        return False
    if any(hint in lower for hint in AI_MODEL_EXCLUSION_HINTS):
        return False
    if not (column in RAW_BASE_COLUMNS or column.startswith(RAW_SOURCE_PREFIXES)):
        return False
    if lower == JOIN_KEY or lower.endswith("__bin"):
        return False
    if any(hint in lower for hint in HIGH_CARDINALITY_HINTS):
        return False
    if series.notna().sum() < 50:
        return False
    if series.nunique(dropna=True) <= 1:
        return False
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    return series.astype(str).nunique(dropna=True) <= 50


def prepare_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    feature_columns = [column for column in df.columns if is_model_candidate(column, df[column])]
    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)

    bool_columns = [column for column in x.columns if pd.api.types.is_bool_dtype(x[column])]
    for column in bool_columns:
        x[column] = x[column].astype("Int64")

    numeric_columns = [column for column in x.columns if pd.api.types.is_numeric_dtype(x[column])]
    categorical_columns = [column for column in x.columns if column not in numeric_columns]
    return x, y, numeric_columns, categorical_columns


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def run_logistic_regression(df: pd.DataFrame) -> dict[str, object]:
    x, y, numeric_columns, categorical_columns = prepare_feature_table(df)
    if y.nunique() != 2:
        raise ValueError("Target must have exactly two classes for logistic regression.")

    selected_features = pd.DataFrame(
        {
            "column": list(x.columns),
            "type": ["numeric" if column in numeric_columns else "categorical" for column in x.columns],
            "non_null_count": [int(x[column].notna().sum()) for column in x.columns],
            "unique_count": [int(x[column].nunique(dropna=True)) for column in x.columns],
        }
    )
    selected_features.to_csv(SELECTED_FEATURES_OUTPUT, index=False)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    transformed_feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)
    coef_df.to_csv(COEFFICIENTS_OUTPUT, index=False)

    metrics = {
        "rows": int(len(df)),
        "model_scope": "Only raw database fields and basic delivery-point matching indicators are used as predictors. Engineered features, simulations, parking-derived scores, and Street View / AI-derived predictors are excluded. The target is still the AI-derived AMR accessibility label.",
        "raw_base_columns_allowed": sorted(RAW_BASE_COLUMNS),
        "raw_source_prefixes_allowed": list(RAW_SOURCE_PREFIXES),
        "excluded_feature_hints": list(AI_MODEL_EXCLUSION_HINTS),
        "target_column": TARGET_COLUMN,
        "target_source_column": TARGET_SOURCE_COLUMN,
        "target_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
        "model_feature_count_before_encoding": int(x.shape[1]),
        "numeric_feature_count": int(len(numeric_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "encoded_feature_count": int(len(transformed_feature_names)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def write_dictionary(df: pd.DataFrame) -> None:
    rows = []
    for column in df.columns:
        if column == TARGET_COLUMN:
            source = "target"
            description = "Binary target: 1 if the AI assessment says the AMR can plausibly reach the door, 0 otherwise."
        elif column.startswith("building_raw__"):
            source = "BUILDING.csv"
            description = "Raw building attribute joined from the NYC building footprint/source file."
        elif column.startswith("pluto_raw__"):
            source = "pluto_full.csv"
            description = "Raw PLUTO building/lot attribute joined through BBL."
        elif column.startswith("address_raw__"):
            source = "AddressPoint_full.csv"
            description = "Raw address point attribute joined through BIN."
        else:
            source = "AI subset"
            description = "Existing field from the Street View AI subset dataset."
        rows.append({"column": column, "source": source, "description": description})
    pd.DataFrame(rows).to_csv(DICTIONARY_OUTPUT, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the AMR accessibility logistic-regression dataset.")
    parser.add_argument("--skip-model", action="store_true", help="Only build the enriched dataset, without fitting the logistic regression.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = build_dataset()
    df.to_csv(DATASET_OUTPUT, index=False)
    write_dictionary(df)

    print(f"Saved enriched logistic-regression dataset: {DATASET_OUTPUT}")
    print(f"Saved data dictionary: {DICTIONARY_OUTPUT}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("Target counts:")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    if not args.skip_model:
        metrics = run_logistic_regression(df)
        print(f"Saved selected model features: {SELECTED_FEATURES_OUTPUT}")
        print(f"Saved logistic coefficients: {COEFFICIENTS_OUTPUT}")
        print(f"Saved model metrics: {METRICS_OUTPUT}")
        print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
        print(f"Test ROC AUC: {metrics['test_roc_auc']:.3f}")


if __name__ == "__main__":
    main()
