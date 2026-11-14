"""Fit the accessibility model, validate it and write the prediction record.

Runs entirely from the published records:
    data/records/manhattan_buildings.csv
    data/records/streetview_entrance_labels.csv

Two indices are fitted with the same model specification:
    primary   target accessible_rule     (gate/ramp after the targeted human review)
    AI-only   target accessible_rule_ai  (gate/ramp as reported by the multimodal model)
plus a sensitivity variant (images the reviewers could not judge keep the AI label).

Writes
    data/records/amr_accessibility_predictions.csv   (published, derived record)
    results/model/sensitivity_predictions.csv          (sensitivity variant, for the validation scorer)
    results/model/*.csv|json                          (validation outputs used in the paper)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.geo import local_xy_m  # noqa: E402
from amr_access.metrics import bootstrap_ci, evaluate, weighted_summary  # noqa: E402
from amr_access.model import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURES,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    build_model,
    coefficient_table,
    domain_flags,
    feature_frame,
)

OUT = config.RESULTS_DIR / "model"
BLOCK_SIZE_M = 500.0
SUMMARY_METRICS = ["roc_auc", "brier", "brier_skill", "accuracy", "log_loss", "ece"]


def load_training(target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = pd.read_csv(config.BUILDINGS_CSV, dtype={"bin": str, "bbl": str, "landuse": str, "zipcode": str})
    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str})
    lab = lab[lab["image_usable"] & lab["in_building_table"]].copy()
    if target_col == "accessible_rule_keep_ai":   # sensitivity: not-judgeable images keep the AI label
        nj = lab["gate_ramp_source"] == "human_review_not_judgeable"
        lab[target_col] = lab["accessible_rule"].where(~nj, lab["accessible_rule_ai"])
    lab = lab.dropna(subset=[target_col])
    train = lab[["bin", target_col]].merge(b, on="bin", how="inner", validate="one_to_one")
    train["y"] = train[target_col].astype(bool).astype(int)
    return b, train.reset_index(drop=True)


def oof_predictions(model, x, y, splitter, groups=None) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for tr, te in splitter.split(x, y, groups):
        p[te] = clone(model).fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[te])[:, 1]
    return p


def leave_one_area_out(model, x, y, areas: pd.Series, min_n: int) -> pd.DataFrame:
    rows = []
    for area in sorted(areas.dropna().unique()):
        te = (areas == area).to_numpy()
        if te.sum() < min_n or len(np.unique(y[te])) < 2:
            continue
        m = clone(model).fit(x[~te], y[~te])
        p = m.predict_proba(x[te])[:, 1]
        r = evaluate(y[te], p)
        r["area"] = area
        rows.append(r)
    return pd.DataFrame(rows)


def alternative_index(target: str, buildings: pd.DataFrame, model, bootstrap: bool) -> tuple[dict, pd.DataFrame]:
    """Same specification and validation scheme as the primary index, for another label."""
    _, tr = load_training(target)
    x, y = feature_frame(tr), tr["y"].to_numpy()
    res: dict = {"n_train": int(len(y)), "prevalence": float(y.mean())}
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.SEED)
    p_oof = oof_predictions(model, x, y, skf)
    res["cv_random"] = evaluate(y, p_oof)
    bx, by = local_xy_m(tr["centroid_lon"], tr["centroid_lat"])
    blocks = pd.Series((np.floor(bx / BLOCK_SIZE_M)).astype(int)).astype(str) + "_" + \
        pd.Series((np.floor(by / BLOCK_SIZE_M)).astype(int)).astype(str)
    res["cv_spatial_block_500m"] = evaluate(y, oof_predictions(model, x, y, GroupKFold(n_splits=config.CV_FOLDS),
                                                               groups=blocks))
    for key, col in [("cv_leave_one_neighbourhood_out", "nta_name"), ("cv_leave_one_zip_out", "zipcode")]:
        t = leave_one_area_out(model, x, y, tr[col], config.MIN_AREA_TEST_BUILDINGS)
        res[key] = {"folds": int(len(t)), **weighted_summary(t, SUMMARY_METRICS)}
    xb = feature_frame(buildings)
    out = pd.DataFrame({"bin": buildings["bin"], "p": clone(model).fit(x, y).predict_proba(xb)[:, 1]})
    if bootstrap:
        rng = np.random.default_rng(config.SEED)
        boot = np.empty((config.N_BOOTSTRAP, len(xb)), dtype=np.float32)
        for k in range(config.N_BOOTSTRAP):
            i = rng.integers(0, len(y), len(y))
            boot[k] = clone(model).fit(x.iloc[i], y[i]).predict_proba(xb)[:, 1]
        out["q05"], out["q95"] = np.percentile(boot, [5, 95], axis=0)
    out["oof"] = out["bin"].map(pd.Series(p_oof, index=tr["bin"]))
    res["mean_p"] = float(out["p"].mean())
    return res, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng_seed = config.SEED
    results: dict = {}

    buildings, train = load_training("accessible_rule")
    x, y = feature_frame(train), train["y"].to_numpy()
    model = build_model(seed=rng_seed)
    results["n_train"] = int(len(y))
    results["prevalence"] = float(y.mean())

    # 1. stratified 5-fold cross-validation (out-of-fold predictions for every labelled building)
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=rng_seed)
    p_oof = oof_predictions(model, x, y, skf)
    results["cv_random"] = evaluate(y, p_oof)
    results["cv_random"]["roc_auc_ci95"] = bootstrap_ci(y, p_oof, "roc_auc", seed=rng_seed)
    results["cv_random"]["brier_ci95"] = bootstrap_ci(y, p_oof, "brier", seed=rng_seed)

    # 2. single 80/20 hold-out (reported for comparability with earlier drafts)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=rng_seed)
    m_hold = clone(model).fit(x.iloc[tr], y[tr])
    results["holdout_20pct"] = evaluate(y[te], m_hold.predict_proba(x.iloc[te])[:, 1])

    # 3. spatially blocked CV: 500 m square blocks, 5 folds of whole blocks
    bx, by = local_xy_m(train["centroid_lon"], train["centroid_lat"])
    blocks = pd.Series((np.floor(bx / BLOCK_SIZE_M)).astype(int)).astype(str) + "_" + \
        pd.Series((np.floor(by / BLOCK_SIZE_M)).astype(int)).astype(str)
    p_block = oof_predictions(model, x, y, GroupKFold(n_splits=config.CV_FOLDS), groups=blocks)
    results["cv_spatial_block_500m"] = evaluate(y, p_block)
    results["cv_spatial_block_500m"]["n_blocks"] = int(blocks.nunique())

    # 4. leave-one-neighbourhood-out and leave-one-ZIP-out
    lono = leave_one_area_out(model, x, y, train["nta_name"], config.MIN_AREA_TEST_BUILDINGS)
    lozo = leave_one_area_out(model, x, y, train["zipcode"], config.MIN_AREA_TEST_BUILDINGS)
    lono.to_csv(OUT / "cv_leave_one_neighbourhood_out.csv", index=False)
    lozo.to_csv(OUT / "cv_leave_one_zip_out.csv", index=False)
    results["cv_leave_one_neighbourhood_out"] = {"folds": int(len(lono)), **weighted_summary(lono, SUMMARY_METRICS)}
    results["cv_leave_one_zip_out"] = {"folds": int(len(lozo)), **weighted_summary(lozo, SUMMARY_METRICS)}

    # 5. reference models (same folds): prevalence only, land use only, gradient boosting
    lu_only = build_model(seed=rng_seed, numeric_features=[], categorical_features=["landuse"])
    results["ref_prevalence_only"] = evaluate(y, np.full(len(y), y.mean()))
    results["ref_landuse_only"] = evaluate(y, oof_predictions(lu_only, x, y, skf))
    hgb = build_model(seed=rng_seed)
    hgb.steps[-1] = ("clf", HistGradientBoostingClassifier(random_state=rng_seed, max_iter=300, learning_rate=0.05))
    results["ref_gradient_boosting"] = evaluate(y, oof_predictions(hgb, x, y, skf))

    # 6. sensitivity: target = model's holistic flag instead of the rule
    _, train_flag = load_training("ai_amr_can_reach_door")
    xf, yf = feature_frame(train_flag), train_flag["y"].to_numpy()
    results["sensitivity_target_ai_flag"] = evaluate(yf, oof_predictions(model, xf, yf, skf))

    # 7. final model on all labels + bootstrap prediction intervals
    final = clone(model).fit(x, y)
    coefficient_table(final).to_csv(OUT / "coefficients.csv", index=False)
    xb = feature_frame(buildings)
    p_all = final.predict_proba(xb)[:, 1]
    rng = np.random.default_rng(rng_seed)
    boot = np.empty((config.N_BOOTSTRAP, len(xb)), dtype=np.float32)
    for k in range(config.N_BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        boot[k] = clone(model).fit(x.iloc[i], y[i]).predict_proba(xb)[:, 1]
    lo, hi = np.percentile(boot, [5, 95], axis=0)

    flags = domain_flags(train, buildings, set(train["nta2020"].dropna()))
    pred = pd.DataFrame({
        "bin": buildings["bin"],
        "p_accessible": p_all,
        "amr_index": 100 * p_all,
        "p_accessible_q05": lo,
        "p_accessible_q95": hi,
    })
    oof = pd.Series(p_oof, index=train["bin"])
    pred["p_accessible_oof"] = pred["bin"].map(oof)
    # AI-only index (same specification, label without the human review)
    results["ai_only"], alt = alternative_index("accessible_rule_ai", buildings, model, bootstrap=True)
    pred["p_accessible_ai_only"] = alt["p"].to_numpy()
    pred["amr_index_ai_only"] = 100 * pred["p_accessible_ai_only"]
    pred["p_accessible_ai_only_q05"] = alt["q05"].to_numpy()
    pred["p_accessible_ai_only_q95"] = alt["q95"].to_numpy()
    pred["p_accessible_ai_only_oof"] = alt["oof"].to_numpy()
    # sensitivity: images the reviewers could not judge keep the AI label
    results["sensitivity_not_judgeable_keep_ai"], sens = alternative_index("accessible_rule_keep_ai", buildings,
                                                                          model, bootstrap=False)
    sens.rename(columns={"p": "p_accessible", "oof": "p_accessible_oof"}).to_csv(
        OUT / "sensitivity_predictions.csv", index=False, float_format="%.5f", lineterminator="\n")
    pred = pd.concat([pred, flags.reset_index(drop=True)], axis=1)
    pred["model_version"] = MODEL_VERSION
    pred.to_csv(config.PREDICTIONS_CSV, index=False, float_format="%.5f", lineterminator="\n")

    results["prediction"] = {
        "n_buildings": int(len(pred)),
        "mean_p": float(pred["p_accessible"].mean()),
        "share_in_domain": float(pred["in_domain"].mean()),
        "share_in_training_nta": float(pred["in_training_nta"].mean()),
        "mean_interval_width": float((pred["p_accessible_q95"] - pred["p_accessible_q05"]).mean()),
        "mean_interval_width_in_domain": float((pred.loc[pred.in_domain, "p_accessible_q95"]
                                                - pred.loc[pred.in_domain, "p_accessible_q05"]).mean()),
        "mean_interval_width_out_domain": float((pred.loc[~pred.in_domain, "p_accessible_q95"]
                                                 - pred.loc[~pred.in_domain, "p_accessible_q05"]).mean()),
    }
    results["features"] = {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}

    # calibration curve data (out-of-fold)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p_oof, bins) - 1, 0, 9)
    cal = pd.DataFrame({"bin": idx, "p": p_oof, "y": y}).groupby("bin").agg(
        mean_predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
    cal.to_csv(OUT / "calibration_oof.csv", index=False)

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for k in ["cv_random", "cv_spatial_block_500m", "cv_leave_one_neighbourhood_out", "cv_leave_one_zip_out",
              "ref_landuse_only", "ref_gradient_boosting", "sensitivity_target_ai_flag"]:
        r = results[k]
        print(f"{k:34s} AUC={r['roc_auc']:.3f} Brier={r['brier']:.3f} acc={r['accuracy']:.3f}")
    for k in ["ai_only", "sensitivity_not_judgeable_keep_ai"]:
        r = results[k]["cv_random"]
        k = k + " (cv_random)"
        print(f"{k:34s} AUC={r['roc_auc']:.3f} Brier={r['brier']:.3f} acc={r['accuracy']:.3f}")
    print(json.dumps(results["prediction"], indent=1))


if __name__ == "__main__":
    main()
