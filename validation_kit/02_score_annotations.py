"""Score the human annotations against the AI labels and the model predictions.

Usage
    python validation_kit/02_score_annotations.py            # from data/records/human_validation_annotations.csv
    python validation_kit/02_score_annotations.py raw_A1.csv raw_A2.csv --primary A1

Outputs results/validation/human_validation.json and per-attribute tables.

Metrics for Set A are weighted by the inverse sampling fraction of each stratum,
so they estimate agreement over the whole annotated sample (not the stratified draw).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amr_access import config  # noqa: E402
from amr_access.metrics import evaluate  # noqa: E402

MAP = {"yes": 1.0, "no": 0.0, "unsure": np.nan, "": np.nan}
OUT = config.RESULTS_DIR / "validation"


def to01(series: pd.Series) -> np.ndarray:
    """Convert booleans / 'True'/'False' strings / 0-1 numbers to a float array with NaN."""
    m = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    return series.astype("string").str.lower().map(m).astype(float).to_numpy()


def human_rule(d: pd.DataFrame) -> pd.Series:
    """Apply the dataset rule to human answers; 'unsure' propagates only when it matters."""
    s, g, r, u = d["stairs"], d["gate"], d["ramp"], d["usable"]
    out = pd.Series(np.nan, index=d.index)
    out[(g == 0) & (s == 0)] = 1.0
    out[(g == 0) & (s == 1) & (r == 1)] = 1.0
    out[(s == 1) & (r == 0)] = 0.0
    out[g == 1] = 0.0
    out[u != 1] = np.nan
    return out


def load_annotations(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p, dtype={"bin": str, "stratum_size": float}, keep_default_na=False)
        for c in ["usable", "stairs", "gate", "ramp", "robot", "reason"]:
            d[c] = d[c].astype(str)
        for c in ["usable", "stairs", "gate", "ramp", "robot"]:
            d[c] = d[c].str.lower().map(MAP)
        # attribute answers are undefined when the annotator judged the image not usable
        d.loc[d["usable"] != 1, ["stairs", "gate", "ramp", "robot"]] = np.nan
        d["accessible_rule"] = human_rule(d)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def weighted_binary(y_true, y_pred, w) -> dict:
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred, w = y_true[ok].astype(int), y_pred[ok].astype(int), w[ok]
    if len(y_true) == 0:
        return {"n": 0}
    out = {"n": int(len(y_true)),
           "accuracy": accuracy_score(y_true, y_pred, sample_weight=w),
           "precision": precision_score(y_true, y_pred, sample_weight=w, zero_division=np.nan),
           "recall": recall_score(y_true, y_pred, sample_weight=w, zero_division=np.nan),
           "f1": f1_score(y_true, y_pred, sample_weight=w, zero_division=np.nan),
           "kappa": cohen_kappa_score(y_true, y_pred, sample_weight=w) if len(np.unique(y_true)) > 1 else np.nan,
           "prevalence_human": float(np.average(y_true, weights=w))}
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def stratified_bootstrap(df: pd.DataFrame, fn, n: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("stratum")]
    vals = []
    for _ in range(n):
        res = pd.concat([g.iloc[rng.integers(0, len(g), len(g))] for g in groups])
        v = fn(res)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", nargs="*", type=Path,
                    default=[config.RECORDS_DIR / "human_validation_annotations.csv"],
                    help="annotation files (default: the published record)")
    ap.add_argument("--primary", default="A1", help="annotator ID used as the reference labels")
    ap.add_argument("--manifest", type=Path, default=config.VALIDATION_DIR / "annotation_manifest_full.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest, dtype={"bin": str})
    ann = load_annotations(args.annotations)
    if "stratum" not in ann:                      # raw exports: attach the sampling design
        ann = ann.merge(man, on=["task_id", "image_file"], how="left", validate="many_to_one")
    prim = ann[ann["annotator"] == args.primary]
    res: dict = {"primary_annotator": args.primary, "annotators": sorted(ann["annotator"].unique().tolist())}

    # ---------------- Set A: AI vs human ----------------
    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str})
    if "gate_present_ai" not in lab:                 # label record without the human review applied
        lab = lab.rename(columns={"gate_present": "gate_present_ai", "ramp_present": "ramp_present_ai",
                                  "accessible_rule": "accessible_rule_ai"})
    # Set A measures the AI itself: compare with the AI values, not the human-revised ones
    lab = lab.drop(columns=[c for c in ["gate_present", "ramp_present", "accessible_rule"] if c in lab])
    a = prim[prim["set"] == "A"].merge(lab, on="bin", how="left")
    a["w"] = a["stratum_size"] / a.groupby("stratum")["bin"].transform("size")
    pairs = {"usable": "image_usable", "stairs": "stairs_present", "gate": "gate_present_ai",
             "ramp": "ramp_present_ai", "accessible_rule": "accessible_rule_ai", "robot": "ai_amr_can_reach_door"}
    rows = []
    for human_col, ai_col in pairs.items():
        yt = a[human_col].astype(float).to_numpy()
        yp = to01(a[ai_col])
        m = weighted_binary(yt, yp, a["w"].to_numpy(float))

        def acc_fn(d, hc=human_col, ac=ai_col):
            yt_ = d[hc].astype(float).to_numpy()
            yp_ = to01(d[ac])
            r = weighted_binary(yt_, yp_, d["w"].to_numpy(float))
            return r.get("kappa")

        m["kappa_ci95"] = stratified_bootstrap(a, acc_fn)
        m["attribute"] = human_col
        m["share_unsure_human"] = float(a[human_col].isna().mean())
        rows.append(m)
    set_a = pd.DataFrame(rows)
    set_a.to_csv(OUT / "set_a_ai_vs_human.csv", index=False)
    res["set_a"] = set_a.set_index("attribute").to_dict(orient="index")

    # ---------------- inter-annotator agreement ----------------
    others = [x for x in res["annotators"] if x != args.primary]
    if others:
        sec = ann[ann["annotator"] == others[0]]
        both = prim.merge(sec, on="task_id", suffixes=("_1", "_2"))
        ia = {}
        for c in ["usable", "stairs", "gate", "ramp", "accessible_rule", "robot"]:
            d = both[[f"{c}_1", f"{c}_2"]].dropna()
            ia[c] = {"n": int(len(d)), "agreement": float((d.iloc[:, 0] == d.iloc[:, 1]).mean()) if len(d) else np.nan,
                     "kappa": float(cohen_kappa_score(d.iloc[:, 0], d.iloc[:, 1])) if d.iloc[:, 0].nunique() > 1 else np.nan}
        res["inter_annotator"] = ia

    # ---------------- targeted review of gates and ramps ----------------
    rev_csv = config.RECORDS_DIR / "streetview_gate_ramp_review.csv"
    if rev_csv.exists():
        res["gate_ramp_review"] = score_review(pd.read_csv(rev_csv, dtype={"bin": str}), lab)

    # ---------------- Set B: human label vs predicted probability ----------------
    # out-of-fold probability for buildings that are in the training set (their own label never
    # informs their prediction), final-model probability for all others
    pred = pd.read_csv(config.PREDICTIONS_CSV, dtype={"bin": str})
    pred["p_eval"] = pred["p_accessible_oof"].fillna(pred["p_accessible"])
    indices = {"set_b": pred[["bin", "p_eval", "p_accessible_oof"]]}
    if "p_accessible_ai_only" in pred:
        indices["set_b_ai_only"] = pd.DataFrame({
            "bin": pred["bin"], "p_eval": pred["p_accessible_ai_only_oof"].fillna(pred["p_accessible_ai_only"]),
            "p_accessible_oof": pred["p_accessible_ai_only_oof"]})
    sens_csv = config.RESULTS_DIR / "model" / "sensitivity_predictions.csv"
    if sens_csv.exists():
        sp = pd.read_csv(sens_csv, dtype={"bin": str})
        indices["set_b_sensitivity_not_judgeable_keep_ai"] = pd.DataFrame({
            "bin": sp["bin"], "p_eval": sp["p_accessible_oof"].fillna(sp["p_accessible"]),
            "p_accessible_oof": sp["p_accessible_oof"]})
    for key, pr in indices.items():
        res[key] = score_set_b(prim, pr)
    rng = np.random.default_rng(config.SEED)
    if "set_b_ai_only" in indices:                     # paired bootstrap: primary minus AI-only
        res["set_b_primary_minus_ai_only"] = paired_difference(prim, indices["set_b"], indices["set_b_ai_only"], rng)

    (OUT / "human_validation.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(set_a[["attribute", "n", "accuracy", "precision", "recall", "kappa"]].round(3).to_string(index=False))
    if "inter_annotator" in res:
        print("inter-annotator kappa:", {k: round(v["kappa"], 3) for k, v in res["inter_annotator"].items()})
    for key in indices:
        for k, v in res[key].items():
            print(key, k, {m: round(v[m], 3) for m in ["n", "roc_auc", "brier", "prevalence", "mean_predicted"] if m in v})
    if "set_b_primary_minus_ai_only" in res:
        print("primary - AI-only:", res["set_b_primary_minus_ai_only"])


def score_review(rev: pd.DataFrame, lab: pd.DataFrame) -> dict:
    """Agreement between reviewers, AI precision among reviewed images, AI misses among controls."""
    out: dict = {"n_tasks": int(rev["task_id"].nunique()), "n_answers": int(len(rev)),
                 "answers": rev.groupby("annotator")["answer"].value_counts().unstack(fill_value=0).to_dict("index")}
    a1 = rev[rev["annotator"] == "A1"].merge(lab[["bin", "gate_present_ai", "ramp_present_ai"]], on="bin", how="left")
    ok = a1[a1["answer"] != "cannot_judge"]
    gate_h, ramp_h = ok["answer"].isin(["gate", "both"]), ok["answer"].isin(["ramp", "both"])
    gate_ai, ramp_ai = to01(ok["gate_present_ai"]) == 1, to01(ok["ramp_present_ai"]) == 1
    ctrl = (ok["group"] == "control").to_numpy()
    out["share_cannot_judge_A1"] = float((a1["answer"] == "cannot_judge").mean())
    out["ai_gate_confirmed"] = {"n": int(gate_ai.sum()), "share": float(gate_h[gate_ai].mean())}
    out["ai_ramp_confirmed"] = {"n": int(ramp_ai.sum()), "share": float(ramp_h[ramp_ai].mean())}
    out["controls_gate_missed_by_ai"] = {"n": int(ctrl.sum()), "share": float(gate_h[ctrl].mean())}
    out["controls_ramp_missed_by_ai"] = {"n": int(ctrl.sum()), "share": float(ramp_h[ctrl].mean())}
    w = rev.pivot(index="task_id", columns="annotator", values="answer").dropna()
    if w.shape[1] >= 2:
        x, y = w.iloc[:, 0], w.iloc[:, 1]
        ia = {"n": int(len(w)), "agreement_all_answers": float((x == y).mean()),
              "kappa_all_answers": float(cohen_kappa_score(x, y)),
              "kappa_cannot_judge": float(cohen_kappa_score(x == "cannot_judge", y == "cannot_judge"))}
        j = (x != "cannot_judge") & (y != "cannot_judge")
        for f, vals in [("gate", ["gate", "both"]), ("ramp", ["ramp", "both"])]:
            gx, gy = x[j].isin(vals), y[j].isin(vals)
            ia[f] = {"n": int(j.sum()), "agreement": float((gx == gy).mean()),
                     "kappa": float(cohen_kappa_score(gx, gy)) if gx.nunique() > 1 and gy.nunique() > 1 else None,
                     "positives": [int(gx.sum()), int(gy.sum())]}
        out["inter_annotator"] = ia
    return out


def score_set_b(prim: pd.DataFrame, pred: pd.DataFrame) -> dict:
    out = {}
    bset = prim[prim["set"].isin(["B1", "B2"])].merge(pred, on="bin", how="left")
    for name, g in [("B1_external_neighbourhood", bset[bset["set"] == "B1"]),
                    ("B2_labelled_neighbourhoods", bset[bset["set"] == "B2"]), ("B_all", bset)]:
        g = g.dropna(subset=["accessible_rule", "p_eval"])
        out[name] = evaluate(g["accessible_rule"].astype(int), g["p_eval"]) if len(g) > 20 else {"n": len(g)}
        if len(g) > 20:
            out[name]["share_in_training"] = float(g["p_accessible_oof"].notna().mean())
        out[name]["share_judgeable"] = float((prim[prim["set"].isin([name[:2]])]["usable"] == 1).mean()) \
            if name != "B_all" else float((prim[prim["set"].isin(["B1", "B2"])]["usable"] == 1).mean())
    return out


def paired_difference(prim: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame, rng, n: int = 2000) -> dict:
    """Bootstrap CI of (index a - index b) for AUC and Brier on all Set B images (paired)."""
    from sklearn.metrics import roc_auc_score
    g = prim[prim["set"].isin(["B1", "B2"])].dropna(subset=["accessible_rule"])
    y = g["accessible_rule"].astype(int).to_numpy()
    pa = g[["bin"]].merge(a, on="bin", how="left")["p_eval"].to_numpy()
    pb = g[["bin"]].merge(b, on="bin", how="left")["p_eval"].to_numpy()
    d_auc, d_brier = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        d_auc.append(roc_auc_score(y[i], pa[i]) - roc_auc_score(y[i], pb[i]))
        d_brier.append(np.mean((pa[i] - y[i]) ** 2) - np.mean((pb[i] - y[i]) ** 2))
    return {"n": int(len(y)),
            "delta_roc_auc": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "delta_roc_auc_ci95": [float(x) for x in np.percentile(d_auc, [2.5, 97.5])],
            "delta_brier": float(np.mean((pa - y) ** 2) - np.mean((pb - y) ** 2)),
            "delta_brier_ci95": [float(x) for x in np.percentile(d_brier, [2.5, 97.5])]}


if __name__ == "__main__":
    main()
