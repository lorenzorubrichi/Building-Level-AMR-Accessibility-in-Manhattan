"""Run raw-database logistic regressions separately within each neighborhood.

This analysis fixes the neighborhood by fitting one model per neighborhood.
It estimates how raw building/database variables are associated with AMR
accessibility inside each local subset.
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
PARENT_DIR = SCRIPT_DIR.parent

INPUT_DATASET = PARENT_DIR / "output" / "amr_accessibility_logistic_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
SUMMARY_OUTPUT = OUTPUT_DIR / "within_neighborhood_model_summary.csv"
COEFFICIENTS_OUTPUT = OUTPUT_DIR / "within_neighborhood_coefficients.csv"
FEATURES_OUTPUT = OUTPUT_DIR / "within_neighborhood_selected_features.csv"
METRICS_OUTPUT = OUTPUT_DIR / "within_neighborhood_metrics.json"

TARGET_COLUMN = "target_amr_accessible"
GROUP_COLUMN = "neighborhood"

RAW_FEATURE_COLUMNS = [
    "numfloors",
    "landuse_x",
    "centroid_lon",
    "centroid_lat",
    "delivery_point_source",
    "construction_year",
    "ground_elevation",
    "building_raw__construction_year",
    "building_raw__ground_elevation",
    "pluto_raw__numfloors",
    "pluto_raw__landuse",
]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_feature_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    feature_columns = [
        column
        for column in RAW_FEATURE_COLUMNS
        if column in df.columns and df[column].notna().sum() >= 20 and df[column].nunique(dropna=True) > 1
    ]
    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)

    bool_columns = [column for column in x.columns if pd.api.types.is_bool_dtype(x[column])]
    for column in bool_columns:
        x[column] = x[column].astype("Int64")

    numeric_columns = [column for column in x.columns if pd.api.types.is_numeric_dtype(x[column])]
    categorical_columns = [column for column in x.columns if column not in numeric_columns]
    return x, y, numeric_columns, categorical_columns


def fit_neighborhood_model(neighborhood: str, df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame] | None:
    if len(df) < 100 or df[TARGET_COLUMN].nunique() != 2:
        return None

    x, y, numeric_columns, categorical_columns = clean_feature_table(df)
    if x.empty or y.nunique() != 2:
        return None

    min_class_count = int(y.value_counts().min())
    if min_class_count < 20:
        return None

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

    test_size = 0.30 if len(df) < 300 else 0.25
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_[0]
    coef_df = pd.DataFrame(
        {
            "neighborhood": neighborhood,
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values(["neighborhood", "abs_coefficient"], ascending=[True, False])

    features_df = pd.DataFrame(
        {
            "neighborhood": neighborhood,
            "column": list(x.columns),
            "type": ["numeric" if column in numeric_columns else "categorical" for column in x.columns],
            "non_null_count": [int(x[column].notna().sum()) for column in x.columns],
            "unique_count": [int(x[column].nunique(dropna=True)) for column in x.columns],
        }
    )

    accessible_count = int(y.sum())
    summary = {
        "neighborhood": neighborhood,
        "rows": int(len(df)),
        "accessible": accessible_count,
        "not_accessible": int(len(df) - accessible_count),
        "accessible_rate": float(y.mean()),
        "feature_count_before_encoding": int(x.shape[1]),
        "encoded_feature_count": int(len(feature_names)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return summary, coef_df, features_df


def run(min_rows: int) -> dict[str, object]:
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_DATASET}")

    df = pd.read_csv(INPUT_DATASET, low_memory=False)
    if GROUP_COLUMN not in df.columns:
        raise ValueError(f"Missing group column: {GROUP_COLUMN}")
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    coefficient_frames = []
    feature_frames = []
    skipped = []

    for neighborhood, group in df.groupby(GROUP_COLUMN, dropna=True):
        group = group.copy()
        if len(group) < min_rows:
            skipped.append({"neighborhood": neighborhood, "rows": int(len(group)), "reason": "below_min_rows"})
            continue
        result = fit_neighborhood_model(str(neighborhood), group)
        if result is None:
            skipped.append({"neighborhood": neighborhood, "rows": int(len(group)), "reason": "insufficient_data_or_single_class"})
            continue
        summary, coefficients, features = result
        summaries.append(summary)
        coefficient_frames.append(coefficients)
        feature_frames.append(features)

    summary_df = pd.DataFrame(summaries).sort_values("rows", ascending=False)
    coefficients_df = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    features_df = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()

    summary_for_csv = summary_df.drop(columns=["confusion_matrix", "classification_report"], errors="ignore")
    summary_for_csv.to_csv(SUMMARY_OUTPUT, index=False)
    coefficients_df.to_csv(COEFFICIENTS_OUTPUT, index=False)
    features_df.to_csv(FEATURES_OUTPUT, index=False)

    metrics = {
        "analysis": "within-neighborhood raw-database logistic regression",
        "input_dataset": str(INPUT_DATASET),
        "target_column": TARGET_COLUMN,
        "group_column": GROUP_COLUMN,
        "min_rows": min_rows,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "n_models_fit": int(len(summaries)),
        "skipped": skipped,
        "models": summaries,
    }
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw-database logistic regressions separately within each neighborhood.")
    parser.add_argument("--min-rows", type=int, default=150, help="Minimum rows required to fit a neighborhood model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run(min_rows=args.min_rows)
    print(f"Models fit: {metrics['n_models_fit']}")
    print(f"Saved summary: {SUMMARY_OUTPUT}")
    print(f"Saved coefficients: {COEFFICIENTS_OUTPUT}")
    print(f"Saved selected features: {FEATURES_OUTPUT}")
    print(f"Saved metrics: {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
