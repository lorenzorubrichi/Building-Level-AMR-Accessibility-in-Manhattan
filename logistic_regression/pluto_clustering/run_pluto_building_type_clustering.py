"""Cluster PLUTO/building-type attributes and analyze AMR accessibility.

The goal is to test whether raw PLUTO/building information can be used as a
pre-screening layer for Street View / AMR accessibility analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent

INPUT_DATASET = PARENT_DIR / "output" / "amr_accessibility_logistic_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"

CLUSTERED_OUTPUT = OUTPUT_DIR / "pluto_clustered_ai_buildings.csv"
CLUSTER_SUMMARY_OUTPUT = OUTPUT_DIR / "pluto_cluster_summary.csv"
BLDGCLASS_SUMMARY_OUTPUT = OUTPUT_DIR / "bldgclass_accessibility_summary.csv"
BLDGCLASS_GROUP_SUMMARY_OUTPUT = OUTPUT_DIR / "bldgclass_group_accessibility_summary.csv"
LANDUSE_SUMMARY_OUTPUT = OUTPUT_DIR / "landuse_accessibility_summary.csv"
MODEL_COMPARISON_OUTPUT = OUTPUT_DIR / "cluster_model_comparison.json"

TARGET_COLUMN = "target_amr_accessible"

NUMERIC_FEATURES = [
    "numfloors_clean",
    "height_roof_clean",
    "construction_year_clean",
    "ground_elevation_clean",
    "shape_area_clean",
]

CATEGORICAL_FEATURES = [
    "landuse_clean",
    "bldgclass_clean",
    "bldgclass_group",
]

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
    "M": "Church / mission",
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


def parse_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan}),
        errors="coerce",
    )


def clean_year(series: pd.Series) -> pd.Series:
    year = parse_number(series)
    year = year.where(~((year > 1) & (year < 3)), year * 1000)
    year = year.where((year >= 1600) & (year <= 2035), np.nan)
    return year


def clean_landuse(series: pd.Series) -> pd.Series:
    values = parse_number(series).round().astype("Int64").astype(str)
    return values.replace({"<NA>": "Unknown", "nan": "Unknown"})


def clean_bldgclass(series: pd.Series) -> pd.Series:
    return series.fillna("Unknown").astype(str).str.strip().str.upper().replace({"": "Unknown"})


def load_and_prepare() -> pd.DataFrame:
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_DATASET}")

    df = pd.read_csv(INPUT_DATASET, low_memory=False)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    df["numfloors_clean"] = parse_number(df.get("numfloors", pd.Series(index=df.index, dtype=object)))
    df["height_roof_clean"] = parse_number(df.get("height_roof", pd.Series(index=df.index, dtype=object)))
    df["construction_year_clean"] = clean_year(df.get("construction_year", pd.Series(index=df.index, dtype=object)))
    df["ground_elevation_clean"] = parse_number(df.get("ground_elevation", pd.Series(index=df.index, dtype=object)))
    df["shape_area_clean"] = parse_number(df.get("shape_area", pd.Series(index=df.index, dtype=object)))
    df["landuse_clean"] = clean_landuse(df.get("landuse_x", pd.Series(index=df.index, dtype=object)))
    df["landuse_label"] = df["landuse_clean"].map(LANDUSE_LABELS).fillna("Unknown / other")
    df["bldgclass_clean"] = clean_bldgclass(df.get("bldgclass", pd.Series(index=df.index, dtype=object)))
    df["bldgclass_group"] = df["bldgclass_clean"].str[0].replace({"U": "U"})
    df["bldgclass_group_label"] = df["bldgclass_group"].map(BLDGCLASS_GROUP_LABELS).fillna("Unknown / other")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(int)
    return df


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(categorical_features: list[str]) -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def add_clusters(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    preprocessor = build_preprocessor(CATEGORICAL_FEATURES)
    matrix = preprocessor.fit_transform(df[features])
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    out = df.copy()
    out["pluto_cluster"] = clusterer.fit_predict(matrix)
    out["pluto_cluster_label"] = out["pluto_cluster"].map(lambda value: f"Cluster {value}")
    return out


def summarize_category(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_columns, dropna=False)[TARGET_COLUMN]
        .agg(rows="count", accessible="sum", accessible_rate="mean")
        .reset_index()
    )
    summary["not_accessible"] = summary["rows"] - summary["accessible"]
    return summary.sort_values(["rows", "accessible_rate"], ascending=[False, False])


def top_values(series: pd.Series, n: int = 3) -> str:
    counts = series.fillna("Unknown").astype(str).value_counts().head(n)
    return "; ".join(f"{idx} ({count})" for idx, count in counts.items())


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in df.groupby("pluto_cluster"):
        rows.append(
            {
                "pluto_cluster": int(cluster),
                "rows": int(len(group)),
                "accessible": int(group[TARGET_COLUMN].sum()),
                "not_accessible": int(len(group) - group[TARGET_COLUMN].sum()),
                "accessible_rate": float(group[TARGET_COLUMN].mean()),
                "mean_numfloors": float(group["numfloors_clean"].mean()),
                "median_numfloors": float(group["numfloors_clean"].median()),
                "mean_height_roof": float(group["height_roof_clean"].mean()),
                "median_construction_year": float(group["construction_year_clean"].median()),
                "top_landuse": top_values(group["landuse_label"]),
                "top_bldgclass_group": top_values(group["bldgclass_group_label"]),
                "top_bldgclass": top_values(group["bldgclass_clean"]),
            }
        )
    return pd.DataFrame(rows).sort_values("accessible_rate", ascending=False)


def evaluate_model(df: pd.DataFrame, feature_columns: list[str], categorical_features: list[str]) -> dict[str, object]:
    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int)
    preprocessor = build_preprocessor(categorical_features)
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
    return {
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run(n_clusters: int) -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_prepare()
    df = add_clusters(df, n_clusters=n_clusters)

    keep_columns = [
        "bin",
        "borough",
        "neighborhood",
        "bldgclass",
        "bldgclass_clean",
        "bldgclass_group",
        "bldgclass_group_label",
        "landuse_x",
        "landuse_clean",
        "landuse_label",
        "numfloors",
        "numfloors_clean",
        "height_roof",
        "height_roof_clean",
        "construction_year",
        "construction_year_clean",
        "ground_elevation",
        "ground_elevation_clean",
        "shape_area",
        "shape_area_clean",
        "pluto_cluster",
        "pluto_cluster_label",
        TARGET_COLUMN,
    ]
    df[[column for column in keep_columns if column in df.columns]].to_csv(CLUSTERED_OUTPUT, index=False)

    summarize_clusters(df).to_csv(CLUSTER_SUMMARY_OUTPUT, index=False)
    summarize_category(df, ["bldgclass_clean"]).to_csv(BLDGCLASS_SUMMARY_OUTPUT, index=False)
    summarize_category(df, ["bldgclass_group", "bldgclass_group_label"]).to_csv(BLDGCLASS_GROUP_SUMMARY_OUTPUT, index=False)
    summarize_category(df, ["landuse_clean", "landuse_label"]).to_csv(LANDUSE_SUMMARY_OUTPUT, index=False)

    base_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    base_model = evaluate_model(df, base_features, CATEGORICAL_FEATURES)
    cluster_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["pluto_cluster_label"]
    cluster_model = evaluate_model(df, cluster_features, CATEGORICAL_FEATURES + ["pluto_cluster_label"])

    comparison = {
        "analysis": "PLUTO/building-type clustering for AMR accessibility pre-screening",
        "rows": int(len(df)),
        "n_clusters": n_clusters,
        "target_counts": {str(key): int(value) for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()},
        "base_pluto_model": base_model,
        "pluto_model_with_cluster_label": cluster_model,
        "outputs": {
            "clustered_buildings": str(CLUSTERED_OUTPUT),
            "cluster_summary": str(CLUSTER_SUMMARY_OUTPUT),
            "bldgclass_summary": str(BLDGCLASS_SUMMARY_OUTPUT),
            "bldgclass_group_summary": str(BLDGCLASS_GROUP_SUMMARY_OUTPUT),
            "landuse_summary": str(LANDUSE_SUMMARY_OUTPUT),
        },
    }
    MODEL_COMPARISON_OUTPUT.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster PLUTO/building type variables and summarize AMR accessibility.")
    parser.add_argument("--clusters", type=int, default=6, help="Number of KMeans PLUTO clusters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison = run(n_clusters=args.clusters)
    print(f"Rows: {comparison['rows']}")
    print(f"Clusters: {comparison['n_clusters']}")
    print(f"Base PLUTO ROC AUC: {comparison['base_pluto_model']['roc_auc']:.3f}")
    print(f"With cluster ROC AUC: {comparison['pluto_model_with_cluster_label']['roc_auc']:.3f}")
    print(f"Saved outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#& "C:\Users\loren\Desktop\RPI_DOC\lavori\dataset\.venv\Scripts\python.exe" "run_pluto_building_type_clustering.py" --clusters 8