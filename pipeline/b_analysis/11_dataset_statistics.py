"""Descriptive statistics and technical-validation checks computed from the records.

Writes results/stats/*.csv and results/stats/stats.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402

OUT = config.RESULTS_DIR / "stats"


def smd_numeric(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().astype(float), b.dropna().astype(float)
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else np.nan


def smd_binary(pa: float, pb: float) -> float:
    s = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
    return float((pa - pb) / s) if s > 0 else np.nan


def balance_table(sample: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for c in ["numfloors", "height_roof_m", "construction_year", "ground_elevation_m", "distance_to_sidewalk_m"]:
        rows.append({"comparison": label, "variable": c, "level": "",
                     "sample": sample[c].mean(), "reference": reference[c].mean(),
                     "smd": smd_numeric(sample[c], reference[c])})
    levels = reference[["landuse", "landuse_label"]].dropna().drop_duplicates()
    for lvl, name in sorted(levels.itertuples(index=False), key=lambda t: int(t[0])):
        pa = (sample["landuse"] == lvl).mean()
        pb = (reference["landuse"] == lvl).mean()
        rows.append({"comparison": label, "variable": "landuse", "level": name,
                     "sample": pa, "reference": pb, "smd": smd_binary(pa, pb)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s: dict = {}
    b = pd.read_csv(config.BUILDINGS_CSV, dtype={"bin": str, "bbl": str, "landuse": str, "zipcode": str})
    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str, "zipcode": str})
    pred = pd.read_csv(config.PREDICTIONS_CSV, dtype={"bin": str})

    # ---------------- records overview ----------------
    s["n_buildings"] = len(b)
    s["n_nta"] = int(b["nta_name"].nunique())
    s["n_images_acquired"] = int(b["streetview_image_acquired"].sum())
    s["n_annotated"] = len(lab)
    s["n_annotated_ntas"] = int(lab["nta_name"].nunique())
    s["n_usable"] = int(lab["image_usable"].sum())
    s["n_unusable"] = int((~lab["image_usable"]).sum())
    s["n_usable_in_building_table"] = int((lab["image_usable"] & lab["in_building_table"]).sum())
    s["ai_model"] = lab["ai_model"].iloc[0]
    s["ai_run_start_utc"], s["ai_run_end_utc"] = lab["ai_request_utc"].min(), lab["ai_request_utc"].max()
    s["missing_share"] = b.isna().mean().round(4)[lambda x: x > 0].to_dict()
    s["delivery_point_address_share"] = float((b["delivery_point_source"] == "address_point").mean())

    # ---------------- label distribution ----------------
    # gate/ramp: final values (AI + targeted human review) and AI-only values (*_ai)
    bmap = {"True": 1.0, "False": 0.0, "true": 1.0, "false": 0.0}
    for c in ["stairs_present", "gate_present", "ramp_present", "gate_present_ai", "ramp_present_ai"]:
        lab[c] = lab[c].astype("string").map(bmap).astype(float)
    u = lab[lab["image_usable"]].copy()
    for suffix, rule in [("", "accessible_rule"), ("_ai", "accessible_rule_ai")]:
        cols = ["stairs_present", "gate_present" + suffix, "ramp_present" + suffix]
        v = u.dropna(subset=[rule])
        combo = v.groupby(cols).size().rename("n").reset_index()
        combo.columns = ["stairs_present", "gate_present", "ramp_present", "n"]
        combo["share"] = combo["n"] / combo["n"].sum()
        combo["accessible_rule"] = ((combo["gate_present"] == 0)
                                    & ((combo["stairs_present"] == 0) | (combo["ramp_present"] == 1)))
        combo = combo.sort_values(["accessible_rule", "n"], ascending=[False, False])
        combo.to_csv(OUT / f"label_combinations{suffix}.csv", index=False)
        s[f"n_labelled{suffix}"] = int(v[rule].notna().sum())
        s[f"share_accessible_rule{suffix}"] = float(v[rule].mean())
        s[f"n_accessible_rule{suffix}"] = int(v[rule].sum())
        for f in cols:
            s[f"share_{f}" if f == "stairs_present" or suffix else f"share_{f}"] = float(v[f].mean())
    s["gate_ramp_source_counts"] = u["gate_ramp_source"].value_counts().to_dict()
    same = u.dropna(subset=["accessible_rule"])          # same buildings for both labels
    s["share_accessible_rule_ai_same_buildings"] = float(same["accessible_rule_ai"].mean())
    s["share_ai_flag_true"] = float(u["ai_amr_can_reach_door"].astype(bool).mean())
    agree = (u["accessible_rule_ai"].astype(int) == u["ai_amr_can_reach_door"].astype(int))
    s["rule_vs_ai_flag_agreement"] = float(agree.mean())
    s["rule_vs_ai_flag_kappa"] = float(cohen_kappa_score(u["accessible_rule_ai"].astype(int),
                                                         u["ai_amr_can_reach_door"].astype(int)))
    pd.crosstab(u["accessible_rule_ai"].astype(int), u["ai_amr_can_reach_door"].astype(int)).to_csv(
        OUT / "rule_vs_ai_flag_crosstab.csv")

    # per-neighbourhood label summary
    nb = u.groupby("nta_name").agg(n_usable=("bin", "size"), share_accessible=("accessible_rule", "mean"),
                                   share_stairs=("stairs_present", "mean"), share_gate=("gate_present", "mean"),
                                   share_ramp=("ramp_present", "mean"))
    nb["n_annotated"] = lab.groupby("nta_name").size()
    nb["share_unusable"] = 1 - nb["n_usable"] / nb["n_annotated"]
    nb["share_accessible_ai"] = u.groupby("nta_name")["accessible_rule_ai"].mean()
    nb["share_gate_ai"] = u.groupby("nta_name")["gate_present_ai"].mean()
    oof = pred.set_index("bin")["p_accessible_oof"]
    u["p_oof"] = u["bin"].map(oof)
    nb["mean_p_oof"] = u.groupby("nta_name")["p_oof"].mean()
    nb.sort_values("share_accessible", ascending=False).to_csv(OUT / "labels_by_neighbourhood.csv")

    # ---------------- image capture dates ----------------
    yr = pd.to_numeric(lab["image_capture_date"].str[:4], errors="coerce")
    s["capture_date_available_share"] = float(yr.notna().mean())
    s["capture_year_min"], s["capture_year_max"] = int(yr.min()), int(yr.max())
    s["capture_year_median"] = float(yr.median())
    s["capture_share_2024_or_later"] = float((yr >= 2024).sum() / yr.notna().sum())
    s["capture_share_before_2020"] = float((yr < 2020).sum() / yr.notna().sum())
    yr.value_counts().sort_index().rename_axis("year").rename("n").to_csv(OUT / "capture_years.csv")
    # does the label differ with image age? (usable, dated images)
    u["year"] = pd.to_numeric(u["image_capture_date"].str[:4], errors="coerce")
    age = u.dropna(subset=["year"]).assign(recent=lambda d: d["year"] >= 2024)
    s["share_accessible_recent_images"] = float(age.loc[age.recent, "accessible_rule"].mean())
    s["share_accessible_older_images"] = float(age.loc[~age.recent, "accessible_rule"].mean())

    # ---------------- camera geometry ----------------
    from amr_access.geo import haversine_m
    d_cam = haversine_m(lab["camera_lon"], lab["camera_lat"], lab["target_lon"], lab["target_lat"])
    s["camera_target_distance_m_median"] = float(np.nanmedian(d_cam))
    s["camera_target_distance_m_p95"] = float(np.nanpercentile(d_cam, 95))
    # heading was computed as a planar bearing in degrees of lon/lat; compare with the true bearing
    dx = np.radians(lab["target_lon"] - lab["camera_lon"]) * np.cos(np.radians(lab["camera_lat"]))
    dy = np.radians(lab["target_lat"] - lab["camera_lat"])
    true_bearing = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
    dev = np.abs((lab["heading_deg"] - true_bearing + 180) % 360 - 180)
    s["heading_deviation_deg_median"] = float(np.nanmedian(dev))
    s["heading_deviation_deg_p95"] = float(np.nanpercentile(dev, 95))
    s["heading_deviation_deg_max"] = float(np.nanmax(dev))
    s["fov_deg"] = float(lab["fov_deg"].median())

    # ---------------- annotation repeatability (identical requests) ----------------
    dup = lab.dropna(subset=["duplicate_request_group"])
    rep = {}
    for f in ["image_usable", "stairs_present", "gate_present_ai", "ramp_present_ai", "accessible_rule_ai"]:
        agree, total = 0, 0
        for _, g in dup.groupby("duplicate_request_group"):
            v = g[f].dropna().astype(int).tolist() if f != "image_usable" else g[f].astype(int).tolist()
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    total += 1
                    agree += v[i] == v[j]
        rep[f] = {"pairs": total, "agreement": agree / total if total else float("nan")}
    s["repeatability_identical_requests"] = rep
    s["n_duplicate_groups"] = int(dup["duplicate_request_group"].nunique())
    s["n_qa_camera_far"] = int(lab["qa_camera_far"].sum())
    s["n_qa_capture_before_2020"] = int(lab["qa_capture_before_2020"].sum())
    for flag in ["qa_camera_far", "qa_capture_before_2020"]:
        s[f"share_unusable_if_{flag}"] = float((~lab.loc[lab[flag], "image_usable"]).mean())
        s[f"share_unusable_if_not_{flag}"] = float((~lab.loc[~lab[flag], "image_usable"]).mean())

    # ---------------- representativeness ----------------
    train_bins = set(u.loc[u["in_building_table"], "bin"])
    sample = b[b["bin"].isin(train_bins)]
    same_nta_imaged_unlabelled = b[b["nta_name"].isin(u["nta_name"].unique()) & b["streetview_image_acquired"]
                                   & ~b["in_ai_annotated_sample"]]
    bal = pd.concat([
        balance_table(sample, b, "sample_vs_all_manhattan"),
        balance_table(sample, b[b["nta_name"].isin(u["nta_name"].unique())], "sample_vs_same_neighbourhoods"),
        balance_table(sample, same_nta_imaged_unlabelled, "sample_vs_imaged_not_annotated_same_neighbourhoods"),
    ])
    bal.to_csv(OUT / "covariate_balance.csv", index=False)
    s["n_imaged_not_annotated_same_ntas"] = int(len(same_nta_imaged_unlabelled))
    for comp, g in bal.groupby("comparison"):
        s[f"max_abs_smd_{comp}"] = float(g["smd"].abs().max())

    # ---------------- predictions overview ----------------
    p = pred.merge(b[["bin", "nta_name"]], on="bin")
    byn = p.groupby("nta_name").agg(n=("bin", "size"), mean_index=("amr_index", "mean"),
                                    share_in_domain=("in_domain", "mean"),
                                    mean_interval=("p_accessible_q95", "mean"))
    byn["mean_interval"] = (p.groupby("nta_name")["p_accessible_q95"].mean()
                            - p.groupby("nta_name")["p_accessible_q05"].mean())
    byn.sort_values("mean_index").to_csv(OUT / "predictions_by_neighbourhood.csv")
    s["share_categories_seen"] = float(pred["categories_seen"].mean())
    s["share_numeric_in_range"] = float(pred["numeric_in_range"].mean())
    s["share_out_nta_but_covariates_ok"] = float((~pred["in_training_nta"] & pred["categories_seen"]
                                                  & pred["numeric_in_range"]).mean())

    (OUT / "stats.json").write_text(json.dumps(s, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k != "missing_share"}, indent=1, default=float))


if __name__ == "__main__":
    main()
