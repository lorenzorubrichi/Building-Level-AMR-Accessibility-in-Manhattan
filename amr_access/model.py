"""Model specification for the building-level AMR accessibility probability.

A single, pre-specified logistic regression is used (no model selection on the
test data). Probabilities are not re-weighted (no class weights), so the
predicted probability is directly interpretable and its mean matches the label
prevalence of the training sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASELINE_NUMERIC = [
    "numfloors",
    "height_roof_m",
    "footprint_area_m2",
    "construction_year",
    "ground_elevation_m",
    "distance_to_sidewalk_m",
]
BASELINE_CATEGORICAL = ["landuse", "bldgclass_group", "bldgclass"]

# Additions evaluated in pipeline/b_analysis/15_feature_set_comparison.py and adopted
# (better ROC AUC and Brier score under every validation scheme, incl. leave-one-neighbourhood-out)
EXTENDED_NUMERIC = ["distance_to_street_m", "n_address_points", "lot_units_res", "lot_bldgarea_m2",
                    "bldgfront_m", "last_alteration_year", "altered_after_1992",
                    "in_historic_district", "is_landmark"]
EXTENDED_CATEGORICAL = ["basement_code", "proximity_code"]
NUMERIC_FEATURES = BASELINE_NUMERIC + EXTENDED_NUMERIC
CATEGORICAL_FEATURES = BASELINE_CATEGORICAL + EXTENDED_CATEGORICAL
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_VERSION = "logit-v1.1"
LOG1P_FEATURES = {"footprint_area_m2", "lot_units_res", "lot_bldgarea_m2", "distance_to_street_m",
                  "distance_to_sidewalk_m"}
MIN_CATEGORY_COUNT = 5  # rarer categories are pooled into an 'infrequent' level


def feature_frame(buildings: pd.DataFrame, numeric: list[str] | None = None,
                  categorical: list[str] | None = None) -> pd.DataFrame:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    b = buildings.copy()
    if "altered_after_1992" in numeric:
        b["altered_after_1992"] = (pd.to_numeric(b["last_alteration_year"], errors="coerce") > 1992).astype(float)
    x = pd.DataFrame(index=b.index)
    for c in numeric:
        v = b[c]
        v = v.map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}) if v.dtype == bool or v.dtype == object else v
        v = pd.to_numeric(v, errors="coerce").astype(float)
        x[c] = np.log1p(v.clip(lower=0)) if c in LOG1P_FEATURES else v
    for c in categorical:
        v = b[c]
        if pd.api.types.is_numeric_dtype(v):
            v = v.astype("Int64")
        x[c] = v.astype("string").fillna("missing").astype(object)
    return x


def build_model(C: float = 1.0, seed: int = 42, numeric_features: list[str] | None = None,
                categorical_features: list[str] | None = None) -> Pipeline:
    num_cols = NUMERIC_FEATURES if numeric_features is None else numeric_features
    cat_cols = CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT,
                                sparse_output=False)
    transformers = []
    if num_cols:
        transformers.append(("num", numeric, num_cols))
    if cat_cols:
        transformers.append(("cat", categorical, cat_cols))
    pre = ColumnTransformer(transformers)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"term": names, "coefficient": coef})
    df.loc[len(df)] = ["intercept", float(model.named_steps["clf"].intercept_[0])]
    df["odds_ratio"] = np.exp(df["coefficient"])
    return df


def domain_flags(train: pd.DataFrame, target: pd.DataFrame, train_ntas: set[str]) -> pd.DataFrame:
    """Domain-of-applicability indicators for each target building.

    * ``in_training_nta``  – building lies in one of the neighbourhoods that supplied labels
    * ``categories_seen``  – land use and building class each occur >= MIN_CATEGORY_COUNT times in training
    * ``numeric_in_range`` – every observed continuous predictor within the 1st–99th percentile of training
    * ``in_domain``        – all three conditions hold
    """
    out = pd.DataFrame(index=target.index)
    out["in_training_nta"] = target["nta2020"].isin(train_ntas)
    seen = np.ones(len(target), dtype=bool)
    for c in ["landuse", "bldgclass"]:
        counts = train[c].astype("string").value_counts()
        frequent = set(counts[counts >= MIN_CATEGORY_COUNT].index)
        seen &= target[c].astype("string").isin(frequent).to_numpy()
    out["categories_seen"] = seen
    ft, fx = feature_frame(train), feature_frame(target)
    in_range = np.ones(len(target), dtype=bool)
    for c in NUMERIC_FEATURES:
        if ft[c].dropna().nunique() <= 2:      # binary indicators are always in range
            continue
        lo, hi = np.nanpercentile(ft[c], [1, 99])
        v = fx[c]
        in_range &= (v.isna() | v.between(lo, hi)).to_numpy()
    out["numeric_in_range"] = in_range
    out["in_domain"] = out["in_training_nta"] & out["categories_seen"] & out["numeric_in_range"]
    return out
