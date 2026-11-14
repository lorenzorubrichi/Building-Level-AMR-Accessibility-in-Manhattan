"""Compare the baseline predictor set with the extended set (PLUTO 26v1 + new geometry fields).

Decision rule (fixed before running): the extended set replaces the baseline only if
it improves ROC AUC under leave-one-neighbourhood-out CV without worsening the Brier
score under any scheme. Both results are reported in results/model/feature_set_comparison.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.geo import local_xy_m  # noqa: E402
from amr_access.metrics import evaluate  # noqa: E402
from amr_access.model import (  # noqa: E402
    BASELINE_CATEGORICAL, BASELINE_NUMERIC, EXTENDED_CATEGORICAL, EXTENDED_NUMERIC, build_model, feature_frame)


def oof(model, x, y, splitter, groups=None):
    p = np.full(len(y), np.nan)
    for tr, te in splitter.split(x, y, groups):
        p[te] = clone(model).fit(x.iloc[tr], y[tr]).predict_proba(x.iloc[te])[:, 1]
    return p


def main() -> None:
    b = pd.read_csv(config.BUILDINGS_CSV, dtype={"bin": str, "bbl": str, "landuse": str, "zipcode": str})
    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str})
    lab = lab[lab["image_usable"] & lab["in_building_table"]].dropna(subset=["accessible_rule"])
    t = lab[["bin", "accessible_rule"]].merge(b, on="bin")
    y = t["accessible_rule"].astype(int).to_numpy()
    bx, by = local_xy_m(t["centroid_lon"], t["centroid_lat"])
    blocks = (np.floor(bx / 500).astype(int)).astype(str) + "_" + (np.floor(by / 500).astype(int)).astype(str)
    specs = {
        "baseline": (BASELINE_NUMERIC, BASELINE_CATEGORICAL),
        "extended": (BASELINE_NUMERIC + EXTENDED_NUMERIC, BASELINE_CATEGORICAL + EXTENDED_CATEGORICAL),
    }
    schemes = {
        "random_5fold": (StratifiedKFold(5, shuffle=True, random_state=config.SEED), None),
        "spatial_block_500m": (GroupKFold(5), blocks),
        "leave_one_zip_out": (GroupKFold(t["zipcode"].nunique()), t["zipcode"].fillna("NA").to_numpy()),
        "leave_one_neighbourhood_out": (GroupKFold(t["nta_name"].nunique()), t["nta_name"].fillna("unassigned").to_numpy()),
    }
    rows, preds = [], {}
    for name, (num, cat) in specs.items():
        x = feature_frame(t, num, cat)
        m = build_model(numeric_features=num, categorical_features=cat, seed=config.SEED)
        for sname, (split, groups) in schemes.items():
            p = oof(m, x, y, split, groups)
            preds[(name, sname)] = p
            r = evaluate(y, p)
            rows.append({"feature_set": name, "scheme": sname, **{k: r[k] for k in
                         ["roc_auc", "brier", "brier_skill", "log_loss", "accuracy", "calibration_slope", "ece"]}})
    res = pd.DataFrame(rows)
    # paired bootstrap of the AUC difference under leave-one-neighbourhood-out
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(config.SEED)
    pa, pb = preds[("extended", "leave_one_neighbourhood_out")], preds[("baseline", "leave_one_neighbourhood_out")]
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        diffs.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    out = config.RESULTS_DIR / "model"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "feature_set_comparison.csv", index=False)
    piv = res.pivot(index="scheme", columns="feature_set", values=["roc_auc", "brier"])
    print(piv.round(4).to_string())
    print(f"LONO AUC difference (extended - baseline): {np.mean(diffs):.4f} [{lo:.4f}, {hi:.4f}]")
    better_auc = lo > 0
    brier_ok = all(res[(res.feature_set == "extended") & (res.scheme == s)]["brier"].iloc[0]
                   <= res[(res.feature_set == "baseline") & (res.scheme == s)]["brier"].iloc[0] + 1e-4 for s in schemes)
    print("ADOPT EXTENDED" if (better_auc and brier_ok) else "KEEP BASELINE")


if __name__ == "__main__":
    main()
