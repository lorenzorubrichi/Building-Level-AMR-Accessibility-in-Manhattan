"""Train non-AI AMR accessibility models and create a baseline AMR Index.

The target comes from the AI-labeled subset, but the input features are limited
to variables that do not require Street View AI. The resulting probability is
converted into a 0-100 building-level AMR Index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_ROOT / "logistic regression" / "output" / "amr_accessibility_logistic_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"

METRICS_JSON = OUTPUT_DIR / "no_ai_model_metrics.json"
METRICS_CSV = OUTPUT_DIR / "no_ai_model_metrics.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "building_level_no_ai_amr_index.csv"
FEATURES_CSV = OUTPUT_DIR / "no_ai_selected_features.csv"
IMPORTANCE_CSV = OUTPUT_DIR / "best_model_feature_importance.csv"
AREA_NEIGHBORHOOD_CSV = OUTPUT_DIR / "no_ai_amr_index_by_neighborhood.csv"
AREA_ZIPCODE_CSV = OUTPUT_DIR / "no_ai_amr_index_by_zipcode.csv"
REPORT_TXT = OUTPUT_DIR / "no_ai_amr_index_report.txt"
SPATIAL_CV_BY_NEIGHBORHOOD_CSV = OUTPUT_DIR / "spatial_cv_leave_one_neighborhood_out.csv"
SPATIAL_CV_SUMMARY_CSV = OUTPUT_DIR / "spatial_cv_summary.csv"
SPATIAL_CV_BY_ZIPCODE_CSV = OUTPUT_DIR / "spatial_cv_leave_one_zipcode_out.csv"
SPATIAL_CV_ZIPCODE_SUMMARY_CSV = OUTPUT_DIR / "spatial_cv_zipcode_summary.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_SPLITS = 5
MIN_AREA_BUILDINGS = 20
SPATIAL_CV_MIN_ZIPCODE_BUILDINGS = 30
SELECTED_INDEX_MODEL = "logistic_regression"


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


def clean_construction_year(series: pd.Series) -> pd.Series:
    values = numeric(series)
    values = values.where(~values.between(1.0, 2.5), values * 1000)
    return values.where(values.between(1600, 2030))


def first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(index=df.index, dtype=object)


def read_and_prepare() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input dataset: {INPUT_CSV}")

    raw = pd.read_csv(INPUT_CSV, low_memory=False)
    target = numeric(raw["target_amr_accessible"]).astype("Int64")
    keep = target.notna()
    raw = raw.loc[keep].copy()
    target = target.loc[keep].astype(int)

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
    features["building_geometry_source_clean"] = first_existing(raw, ["building_raw__geometry_source", "geometry_source"]).fillna(
        "Unknown"
    ).astype(str)
    features["building_status_clean"] = first_existing(raw, ["building_raw__last_status_type", "last_status_type"]).fillna(
        "Unknown"
    ).astype(str)

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

    identifiers = pd.DataFrame(
        {
            "row_id": np.arange(len(raw)),
            "borough": features["borough"].to_numpy(),
            "neighborhood": features["neighborhood"].to_numpy(),
            "zipcode": features["zipcode_clean"].to_numpy(),
            "map_pluto_bbl": first_existing(raw, ["map_pluto_bbl", "building_raw__map_pluto_bbl", "base_bbl"]).to_numpy(),
            "bldgclass": features["bldgclass_clean"].to_numpy(),
            "landuse": features["landuse_clean"].to_numpy(),
            "target_amr_accessible": target.to_numpy(),
        }
    )
    if "centroid_lon" in raw:
        identifiers["centroid_lon"] = numeric(raw["centroid_lon"]).to_numpy()
    if "centroid_lat" in raw:
        identifiers["centroid_lat"] = numeric(raw["centroid_lat"]).to_numpy()
    if "full_address" in raw:
        identifiers["full_address"] = raw["full_address"].to_numpy()

    return features[NUMERIC_FEATURES + CATEGORICAL_FEATURES], target, identifiers


def preprocessing() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", preprocessing()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "polynomial_logistic_regression": Pipeline(
            [
                ("preprocess", preprocessing()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE, C=0.2)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", preprocessing()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        min_samples_leaf=8,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", preprocessing()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=250,
                        max_leaf_nodes=24,
                        l2_regularization=0.05,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "average_precision": average_precision_score(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def evaluate_safe(y_true: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    pred = (proba >= 0.5).astype(int)
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=labels).ravel()
    has_both_classes = pd.Series(y_true).nunique() == 2
    return {
        "roc_auc": roc_auc_score(y_true, proba) if has_both_classes else np.nan,
        "average_precision": average_precision_score(y_true, proba) if has_both_classes else np.nan,
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred) if has_both_classes else np.nan,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    preprocessor = fitted_pipeline.named_steps["preprocess"]
    names = preprocessor.get_feature_names_out()
    return [str(name) for name in names]


def save_logistic_coefficients(best_model: Pipeline) -> None:
    names = feature_names(best_model)
    coefficients = best_model.named_steps["model"].coef_[0]
    out = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False)
    out.to_csv(IMPORTANCE_CSV, index=False)


def save_tree_importance(best_model: Pipeline, model_name: str, x_test: pd.DataFrame, y_test: pd.Series) -> None:
    # Random forest has native importances after preprocessing. HistGradientBoosting
    # does not expose simple feature importances, so use permutation on original columns.
    if model_name == "random_forest":
        names = feature_names(best_model)
        importances = best_model.named_steps["model"].feature_importances_
        out = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False)
    else:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            best_model,
            x_test,
            y_test,
            n_repeats=10,
            scoring="roc_auc",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        out = pd.DataFrame(
            {
                "feature": x_test.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    out.to_csv(IMPORTANCE_CSV, index=False)


def aggregate_index(predictions: pd.DataFrame, area_col: str, geography_type: str) -> pd.DataFrame:
    rows = []
    for area, group in predictions.groupby(area_col, dropna=False):
        if len(group) < MIN_AREA_BUILDINGS:
            continue
        target = group["target_amr_accessible"]
        rows.append(
            {
                "geography_type": geography_type,
                "area_name": "Unknown" if pd.isna(area) else str(area),
                "n_buildings": int(len(group)),
                "actual_amr_accessibility_rate": float(target.mean()),
                "mean_baseline_amr_probability": float(group["baseline_amr_probability"].mean()),
                "mean_baseline_amr_index": float(group["baseline_amr_index"].mean()),
                "median_baseline_amr_index": float(group["baseline_amr_index"].median()),
                "share_predicted_high_index_70plus": float((group["baseline_amr_index"] >= 70).mean()),
                "share_predicted_low_index_under_30": float((group["baseline_amr_index"] < 30).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_baseline_amr_index", ascending=False)


def spatial_cv_leave_one_area_out(
    x: pd.DataFrame,
    y: pd.Series,
    identifiers: pd.DataFrame,
    selected_name: str,
    area_col: str,
    geography_type: str,
    min_test_buildings: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    models = build_models()
    areas = sorted(identifiers[area_col].fillna("Unknown").astype(str).unique())

    for area in areas:
        if area.lower() == "unknown":
            continue
        test_mask = identifiers[area_col].fillna("Unknown").astype(str) == area
        train_mask = ~test_mask
        if test_mask.sum() <= min_test_buildings:
            continue

        model = models[selected_name].fit(x.loc[train_mask], y.loc[train_mask])
        proba = model.predict_proba(x.loc[test_mask])[:, 1]
        fold_metrics = evaluate_safe(y.loc[test_mask], proba)
        row = {
            "geography_type": geography_type,
            "held_out_area": area,
            "n_test_buildings": int(test_mask.sum()),
            "actual_accessibility_rate": float(y.loc[test_mask].mean()),
            "mean_predicted_probability": float(proba.mean()),
            "mean_prediction_gap_actual_minus_predicted": float(y.loc[test_mask].mean() - proba.mean()),
        }
        row.update(fold_metrics)
        rows.append(row)

    by_neighborhood = pd.DataFrame(rows).sort_values("roc_auc", ascending=False, na_position="last")
    metric_cols = [
        "roc_auc",
        "average_precision",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "brier_score",
    ]
    summary_rows = []
    for metric in metric_cols:
        valid = by_neighborhood.dropna(subset=[metric])
        if valid.empty:
            mean_value = np.nan
            weighted_value = np.nan
        else:
            mean_value = float(valid[metric].mean())
            weighted_value = float(np.average(valid[metric], weights=valid["n_test_buildings"]))
        summary_rows.append(
            {
                "validation_type": f"leave_one_{geography_type}_out",
                "model": selected_name,
                "metric": metric,
                "mean_across_areas": mean_value,
                "weighted_by_test_buildings": weighted_value,
                "n_areas": int(len(valid)),
                "min_test_buildings": int(min_test_buildings),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return by_neighborhood, summary


def write_report(
    metrics: pd.DataFrame,
    selected_name: str,
    predictions: pd.DataFrame,
    spatial_summary: pd.DataFrame,
    spatial_zipcode_summary: pd.DataFrame,
) -> None:
    selected = metrics.loc[metrics["model"] == selected_name].iloc[0]
    best_by_auc = metrics.sort_values("roc_auc", ascending=False).iloc[0]
    spatial_lookup = spatial_summary.set_index("metric")
    zipcode_spatial_lookup = spatial_zipcode_summary.set_index("metric")
    lines = [
        "Non-AI AMR Index Modeling Report",
        "",
        "Goal:",
        "Train a scalable baseline model on the 7,013 AI-labeled buildings using only non-AI features.",
        "The predicted probability of AMR accessibility is converted to a 0-100 AMR Index.",
        "",
        "AMR Index formula:",
        "baseline_amr_index = 100 * P(AMR accessible | non-AI building and urban features)",
        "",
        "Features used:",
        "- Numeric: " + ", ".join(NUMERIC_FEATURES),
        "- Categorical: " + ", ".join(CATEGORICAL_FEATURES),
        "",
        "Features intentionally excluded:",
        "Street View image usability, stairs/gate/ramp/barrier indicators, AI barrier scores, AMR can reach door,",
        "AI penalties, and simulated AMR/car last-meter times.",
        "Borough, neighborhood, ZIP code, duplicate category descriptions, delivery-point source,",
        "building geometry source, and building status are also excluded from the predictors.",
        "",
        f"Selected AMR Index model: {selected_name}",
        "Reason: logistic regression provides a transparent probability formula and is easier to scale/explain.",
        f"- ROC AUC: {selected['roc_auc']:.3f}",
        f"- Accuracy: {selected['accuracy']:.3f}",
        f"- Balanced accuracy: {selected['balanced_accuracy']:.3f}",
        f"- F1: {selected['f1']:.3f}",
        f"- Brier score: {selected['brier_score']:.3f}",
        "",
        f"Best benchmark model by ROC AUC: {best_by_auc['model']} ({best_by_auc['roc_auc']:.3f})",
        "The benchmark model is kept for comparison but is not used to compute the final AMR Index.",
        "",
        "Spatial cross-validation:",
        "Leave-one-neighborhood-out validation trains the model on all neighborhoods except one and tests it on the held-out neighborhood.",
        f"- Spatial ROC AUC, weighted by test buildings: {spatial_lookup.loc['roc_auc', 'weighted_by_test_buildings']:.3f}",
        f"- Spatial accuracy, weighted by test buildings: {spatial_lookup.loc['accuracy', 'weighted_by_test_buildings']:.3f}",
        f"- Spatial F1, weighted by test buildings: {spatial_lookup.loc['f1', 'weighted_by_test_buildings']:.3f}",
        f"- Spatial Brier score, weighted by test buildings: {spatial_lookup.loc['brier_score', 'weighted_by_test_buildings']:.3f}",
        "",
        "ZIP-code spatial cross-validation:",
        f"Leave-one-ZIP-code-out validation uses ZIP codes with more than {SPATIAL_CV_MIN_ZIPCODE_BUILDINGS} test buildings.",
        f"- ZIP spatial ROC AUC, weighted by test buildings: {zipcode_spatial_lookup.loc['roc_auc', 'weighted_by_test_buildings']:.3f}",
        f"- ZIP spatial accuracy, weighted by test buildings: {zipcode_spatial_lookup.loc['accuracy', 'weighted_by_test_buildings']:.3f}",
        f"- ZIP spatial F1, weighted by test buildings: {zipcode_spatial_lookup.loc['f1', 'weighted_by_test_buildings']:.3f}",
        f"- ZIP spatial Brier score, weighted by test buildings: {zipcode_spatial_lookup.loc['brier_score', 'weighted_by_test_buildings']:.3f}",
        "",
        "Index distribution:",
        f"- Mean index: {predictions['baseline_amr_index'].mean():.1f}",
        f"- Median index: {predictions['baseline_amr_index'].median():.1f}",
        f"- Share >= 70: {(predictions['baseline_amr_index'] >= 70).mean() * 100:.1f}%",
        f"- Share < 30: {(predictions['baseline_amr_index'] < 30).mean() * 100:.1f}%",
        "",
        "Interpretation:",
        "This is the baseline scalable model. It can be applied to the complete building dataset only if the",
        "same non-AI features are available there. A later AI-enhanced model can be trained on the same subset",
        "to quantify the additional value of Street View AI features.",
    ]
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x, y, identifiers = read_and_prepare()
    pd.DataFrame({"feature": NUMERIC_FEATURES + CATEGORICAL_FEATURES, "feature_type": ["numeric"] * len(NUMERIC_FEATURES) + ["categorical"] * len(CATEGORICAL_FEATURES)}).to_csv(FEATURES_CSV, index=False)

    x_train, x_test, y_train, y_test, id_train, id_test = train_test_split(
        x,
        y,
        identifiers,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    models = build_models()
    metrics_rows = []
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        print(f"Training/evaluating {name}...")
        cv_proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
        cv_metrics = evaluate(y, cv_proba)

        fitted = model.fit(x_train, y_train)
        test_proba = fitted.predict_proba(x_test)[:, 1]
        test_metrics = evaluate(y_test, test_proba)

        row = {"model": name}
        row.update({f"cv_{k}": v for k, v in cv_metrics.items()})
        row.update(test_metrics)
        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    selected_name = SELECTED_INDEX_MODEL
    selected_model = models[selected_name].fit(x, y)
    full_proba = selected_model.predict_proba(x)[:, 1]

    predictions = identifiers.copy()
    predictions["baseline_amr_probability"] = full_proba
    predictions["baseline_amr_index"] = full_proba * 100
    predictions["baseline_amr_class_50pct"] = (full_proba >= 0.5).astype(int)
    predictions["model_used"] = selected_name

    metrics.to_csv(METRICS_CSV, index=False)
    METRICS_JSON.write_text(json.dumps(metrics.to_dict(orient="records"), indent=2), encoding="utf-8")
    predictions.to_csv(PREDICTIONS_CSV, index=False)

    aggregate_index(predictions, "neighborhood", "neighborhood").to_csv(AREA_NEIGHBORHOOD_CSV, index=False)
    aggregate_index(predictions, "zipcode", "zipcode").to_csv(AREA_ZIPCODE_CSV, index=False)

    print("Running leave-one-neighborhood-out spatial cross-validation...")
    spatial_by_neighborhood, spatial_summary = spatial_cv_leave_one_area_out(
        x,
        y,
        identifiers,
        selected_name,
        area_col="neighborhood",
        geography_type="neighborhood",
        min_test_buildings=MIN_AREA_BUILDINGS,
    )
    spatial_by_neighborhood.to_csv(SPATIAL_CV_BY_NEIGHBORHOOD_CSV, index=False)
    spatial_summary.to_csv(SPATIAL_CV_SUMMARY_CSV, index=False)

    print("Running leave-one-ZIP-code-out spatial cross-validation...")
    spatial_by_zipcode, spatial_zipcode_summary = spatial_cv_leave_one_area_out(
        x,
        y,
        identifiers,
        selected_name,
        area_col="zipcode",
        geography_type="zipcode",
        min_test_buildings=SPATIAL_CV_MIN_ZIPCODE_BUILDINGS,
    )
    spatial_by_zipcode.to_csv(SPATIAL_CV_BY_ZIPCODE_CSV, index=False)
    spatial_zipcode_summary.to_csv(SPATIAL_CV_ZIPCODE_SUMMARY_CSV, index=False)

    if selected_name in {"logistic_regression"}:
        save_logistic_coefficients(selected_model)
    elif selected_name == "random_forest":
        save_tree_importance(selected_model, selected_name, x_test, y_test)
    else:
        save_tree_importance(selected_model, selected_name, x_test, y_test)

    write_report(metrics, selected_name, predictions, spatial_summary, spatial_zipcode_summary)

    print(f"Rows used: {len(x):,}")
    print(f"Target accessible rate: {y.mean():.3f}")
    print(f"Selected AMR Index model: {selected_name}")
    print(f"Saved metrics: {METRICS_CSV}")
    print(f"Saved spatial CV: {SPATIAL_CV_BY_NEIGHBORHOOD_CSV}")
    print(f"Saved ZIP spatial CV: {SPATIAL_CV_BY_ZIPCODE_CSV}")
    print(f"Saved building AMR index: {PREDICTIONS_CSV}")
    print(f"Saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
