"""Generate LaTeX macros and tables so every number in the manuscript comes from the pipeline.

Writes manuscript/generated/numbers.tex and manuscript/generated/tab_*.tex.
Human-validation numbers are filled in once results/validation/human_validation.json
exists; until then they render as a visible [pending] marker.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402

GEN = config.ROOT / "manuscript" / "generated"
PENDING = r"\textbf{[pending]}"


def fmt_int(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x, d=1) -> str:
    return f"{100 * x:.{d}f}\\%"


def f3(x) -> str:
    return f"{x:.3f}"


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    m = json.loads((config.RESULTS_DIR / "model" / "metrics.json").read_text())
    s = json.loads((config.RESULTS_DIR / "stats" / "stats.json").read_text())
    hv_path = config.RESULTS_DIR / "validation" / "human_validation.json"
    hv = json.loads(hv_path.read_text()) if hv_path.exists() else None

    macros = {
        "NBuildings": fmt_int(s["n_buildings"]),
        "NNTA": str(s["n_nta"]),
        "NImagesAcquired": fmt_int(s["n_images_acquired"]),
        "NAnnotated": fmt_int(s["n_annotated"]),
        "NAnnotatedNTA": str(s["n_annotated_ntas"]),
        "NUsable": fmt_int(s["n_usable"]),
        "NUnusable": fmt_int(s["n_unusable"]),
        "NTrain": fmt_int(m["n_train"]),
        "NAccessible": fmt_int(s["n_accessible_rule"]),
        "ShareAccessible": pct(s["share_accessible_rule"]),
        "ShareStairs": pct(s["share_stairs_present"]),
        "ShareGate": pct(s["share_gate_present"]),
        "ShareRamp": pct(s["share_ramp_present"]),
        "ShareAIFlag": pct(s["share_ai_flag_true"]),
        "RuleFlagAgreement": pct(s["rule_vs_ai_flag_agreement"]),
        "RuleFlagKappa": f"{s['rule_vs_ai_flag_kappa']:.2f}",
        "AIModel": s["ai_model"].replace("_", r"\_"),
        "AIRunDate": s["ai_run_start_utc"][:10],
        "AIRunStart": s["ai_run_start_utc"][11:16],
        "AIRunEnd": s["ai_run_end_utc"][11:16],
        "ShareDeliveryAddress": pct(s["delivery_point_address_share"]),
        "CaptureDateShare": pct(s["capture_date_available_share"]),
        "CaptureYearMin": str(s["capture_year_min"]),
        "CaptureYearMax": str(s["capture_year_max"]),
        "CaptureRecentShare": pct(s["capture_share_2024_or_later"]),
        "CaptureOldShare": pct(s["capture_share_before_2020"]),
        "CamDistMedian": f"{s['camera_target_distance_m_median']:.1f}",
        "CamDistPninetyfive": f"{s['camera_target_distance_m_p95']:.1f}",
        "HeadingDevMedian": f"{s['heading_deviation_deg_median']:.1f}",
        "HeadingDevMax": f"{s['heading_deviation_deg_max']:.1f}",
        "FOV": f"{s['fov_deg']:.0f}",
        "SMDAll": f"{s['max_abs_smd_sample_vs_all_manhattan']:.2f}",
        "SMDSameNTA": f"{s['max_abs_smd_sample_vs_same_neighbourhoods']:.2f}",
        "SMDImagedNotAnnotated": f"{s['max_abs_smd_sample_vs_imaged_not_annotated_same_neighbourhoods']:.2f}",
        "NImagedNotAnnotated": fmt_int(s["n_imaged_not_annotated_same_ntas"]),
        "Prevalence": pct(m["prevalence"]),
        "CVAUC": f3(m["cv_random"]["roc_auc"]),
        "CVAUCLo": f3(m["cv_random"]["roc_auc_ci95"][0]),
        "CVAUCHi": f3(m["cv_random"]["roc_auc_ci95"][1]),
        "CVBrier": f3(m["cv_random"]["brier"]),
        "CVBrierSkill": f"{m['cv_random']['brier_skill']:.2f}",
        "CVAcc": f3(m["cv_random"]["accuracy"]),
        "CVCalSlope": f"{m['cv_random']['calibration_slope']:.2f}",
        "CVCalIntercept": f"{m['cv_random']['calibration_intercept']:.2f}",
        "CVECE": f3(m["cv_random"]["ece"]),
        "HoldAUC": f3(m["holdout_20pct"]["roc_auc"]),
        "HoldBrier": f3(m["holdout_20pct"]["brier"]),
        "HoldAcc": f3(m["holdout_20pct"]["accuracy"]),
        "BlockAUC": f3(m["cv_spatial_block_500m"]["roc_auc"]),
        "BlockBrier": f3(m["cv_spatial_block_500m"]["brier"]),
        "BlockAcc": f3(m["cv_spatial_block_500m"]["accuracy"]),
        "NBlocks": str(m["cv_spatial_block_500m"]["n_blocks"]),
        "LONOAUC": f3(m["cv_leave_one_neighbourhood_out"]["roc_auc"]),
        "LONOBrier": f3(m["cv_leave_one_neighbourhood_out"]["brier"]),
        "LONOAcc": f3(m["cv_leave_one_neighbourhood_out"]["accuracy"]),
        "LOZOAUC": f3(m["cv_leave_one_zip_out"]["roc_auc"]),
        "LOZOBrier": f3(m["cv_leave_one_zip_out"]["brier"]),
        "LOZOAcc": f3(m["cv_leave_one_zip_out"]["accuracy"]),
        "LOZOFolds": str(m["cv_leave_one_zip_out"]["folds"]),
        "LUAUC": f3(m["ref_landuse_only"]["roc_auc"]),
        "LUBrier": f3(m["ref_landuse_only"]["brier"]),
        "LUAcc": f3(m["ref_landuse_only"]["accuracy"]),
        "HGBAUC": f3(m["ref_gradient_boosting"]["roc_auc"]),
        "HGBBrier": f3(m["ref_gradient_boosting"]["brier"]),
        "HGBAcc": f3(m["ref_gradient_boosting"]["accuracy"]),
        "PrevBrier": f3(m["ref_prevalence_only"]["brier"]),
        "FlagAUC": f3(m["sensitivity_target_ai_flag"]["roc_auc"]),
        "FlagBrier": f3(m["sensitivity_target_ai_flag"]["brier"]),
        "FlagAcc": f3(m["sensitivity_target_ai_flag"]["accuracy"]),
        "NBootstrap": str(config.N_BOOTSTRAP),
        # AI-only label and index
        "NTrainAI": fmt_int(m["ai_only"]["n_train"]),
        "PrevalenceAI": pct(m["ai_only"]["prevalence"]),
        "ShareAccessibleAISameBuildings": pct(s["share_accessible_rule_ai_same_buildings"]),
        "ShareGateAI": pct(s["share_gate_present_ai"]),
        "ShareRampAI": pct(s["share_ramp_present_ai"]),
        "NNotJudgeable": fmt_int(s["gate_ramp_source_counts"].get("human_review_not_judgeable", 0)),
        "NSourceAI": fmt_int(s["gate_ramp_source_counts"].get("ai", 0)),
        "NSourceReview": fmt_int(s["gate_ramp_source_counts"].get("human_review", 0)),
        "NSourceSetA": fmt_int(s["gate_ramp_source_counts"].get("human_validation_set_a", 0)),
        "CVAUCAI": f3(m["ai_only"]["cv_random"]["roc_auc"]),
        "CVBrierAI": f3(m["ai_only"]["cv_random"]["brier"]),
        "CVAccAI": f3(m["ai_only"]["cv_random"]["accuracy"]),
        "BlockAUCAI": f3(m["ai_only"]["cv_spatial_block_500m"]["roc_auc"]),
        "BlockBrierAI": f3(m["ai_only"]["cv_spatial_block_500m"]["brier"]),
        "BlockAccAI": f3(m["ai_only"]["cv_spatial_block_500m"]["accuracy"]),
        "LONOAUCAI": f3(m["ai_only"]["cv_leave_one_neighbourhood_out"]["roc_auc"]),
        "LONOBrierAI": f3(m["ai_only"]["cv_leave_one_neighbourhood_out"]["brier"]),
        "LONOAccAI": f3(m["ai_only"]["cv_leave_one_neighbourhood_out"]["accuracy"]),
        "LOZOAUCAI": f3(m["ai_only"]["cv_leave_one_zip_out"]["roc_auc"]),
        "LOZOBrierAI": f3(m["ai_only"]["cv_leave_one_zip_out"]["brier"]),
        "LOZOAccAI": f3(m["ai_only"]["cv_leave_one_zip_out"]["accuracy"]),
        "MeanPAI": f"{m['ai_only']['mean_p']:.2f}",
        "SensCVAUC": f3(m["sensitivity_not_judgeable_keep_ai"]["cv_random"]["roc_auc"]),
        "MeanP": f"{m['prediction']['mean_p']:.2f}",
        "ShareInDomain": pct(m["prediction"]["share_in_domain"], 0),
        "ShareInTrainingNTA": pct(m["prediction"]["share_in_training_nta"], 0),
        "ShareCovariatesOkOutside": pct(s["share_out_nta_but_covariates_ok"], 0),
        "IntervalIn": f"{m['prediction']['mean_interval_width_in_domain']:.2f}",
        "IntervalOut": f"{m['prediction']['mean_interval_width_out_domain']:.2f}",
    }
    # human validation (pending until the annotation is done)
    if hv:
        a = hv["set_a"]
        for key, attr in [("Stairs", "stairs"), ("Gate", "gate"), ("Ramp", "ramp"), ("Rule", "accessible_rule"),
                          ("Usable", "usable")]:
            macros[f"HV{key}Acc"] = pct(a[attr]["accuracy"])
            macros[f"HV{key}Kappa"] = f"{a[attr]['kappa']:.2f}"
            macros[f"HV{key}Prec"] = pct(a[attr]["precision"])
            macros[f"HV{key}Rec"] = pct(a[attr]["recall"])
        ia = hv.get("inter_annotator", {})
        for key, attr in [("Stairs", "stairs"), ("Gate", "gate"), ("Ramp", "ramp"), ("Rule", "accessible_rule")]:
            macros[f"IA{key}Kappa"] = f"{ia[attr]['kappa']:.2f}" if attr in ia else PENDING
        b1, b2 = hv["set_b"]["B1_external_neighbourhood"], hv["set_b"]["B2_labelled_neighbourhoods"]
        macros.update({"BOneAUC": f3(b1["roc_auc"]), "BOneBrier": f3(b1["brier"]), "BOneN": str(b1["n"]),
                       "BTwoAUC": f3(b2["roc_auc"]), "BTwoBrier": f3(b2["brier"]), "BTwoN": str(b2["n"])})
        # both indices on Set B, their difference, and the sensitivity variant
        for tag, key in [("", "set_b"), ("AI", "set_b_ai_only"), ("Sens", "set_b_sensitivity_not_judgeable_keep_ai")]:
            if key not in hv:
                continue
            for nm, b in [("BOne", "B1_external_neighbourhood"), ("BTwo", "B2_labelled_neighbourhoods")]:
                r = hv[key][b]
                macros[f"{nm}{tag}AUC"] = f3(r["roc_auc"])
                macros[f"{nm}{tag}Brier"] = f3(r["brier"])
                macros[f"{nm}{tag}MeanP"] = f"{r['mean_predicted']:.2f}"
                macros[f"{nm}{tag}Prev"] = f"{r['prevalence']:.2f}"
        if "set_b_primary_minus_ai_only" in hv:
            d = hv["set_b_primary_minus_ai_only"]
            sg = lambda x: ("+" if x >= 0 else "$-$") + f"{abs(x):.3f}"  # noqa: E731
            macros.update({"DeltaN": str(d["n"]), "DeltaAUC": sg(d["delta_roc_auc"]),
                           "DeltaAUCLo": sg(d["delta_roc_auc_ci95"][0]), "DeltaAUCHi": sg(d["delta_roc_auc_ci95"][1]),
                           "DeltaBrier": sg(d["delta_brier"]), "DeltaBrierLo": sg(d["delta_brier_ci95"][0]),
                           "DeltaBrierHi": sg(d["delta_brier_ci95"][1])})
        if "gate_ramp_review" in hv:
            r = hv["gate_ramp_review"]
            ia = r["inter_annotator"]
            macros.update({
                "RevTasks": fmt_int(r["n_tasks"]),
                "RevAIGateN": fmt_int(r["ai_gate_confirmed"]["n"]),
                "RevAIGateConfirmed": pct(r["ai_gate_confirmed"]["share"]),
                "RevAIRampN": fmt_int(r["ai_ramp_confirmed"]["n"]),
                "RevAIRampConfirmed": pct(r["ai_ramp_confirmed"]["share"]),
                "RevControlN": fmt_int(r["controls_gate_missed_by_ai"]["n"]),
                "RevControlGateMissed": pct(r["controls_gate_missed_by_ai"]["share"]),
                "RevControlRampMissed": pct(r["controls_ramp_missed_by_ai"]["share"]),
                "RevCannotJudge": pct(r["share_cannot_judge_A1"]),
                "RevIAN": str(ia["n"]),
                "RevIAAgreeAll": pct(ia["agreement_all_answers"]),
                "RevIAKappaAll": f"{ia['kappa_all_answers']:.2f}",
                "RevIAGateN": str(ia["gate"]["n"]),
                "RevIAGateAgree": pct(ia["gate"]["agreement"]),
                "RevIAGateKappa": f"{ia['gate']['kappa']:.2f}",
                "RevIACannotKappa": f"{ia['kappa_cannot_judge']:.2f}",
            })
    else:
        for key in ["Stairs", "Gate", "Ramp", "Rule", "Usable"]:
            for suf in ["Acc", "Kappa", "Prec", "Rec"]:
                macros[f"HV{key}{suf}"] = PENDING
        for key in ["Stairs", "Gate", "Ramp", "Rule"]:
            macros[f"IA{key}Kappa"] = PENDING
        for k in ["BOneAUC", "BOneBrier", "BOneN", "BTwoAUC", "BTwoBrier", "BTwoN"]:
            macros[k] = PENDING

    lines = ["% Generated by pipeline/b_analysis/13_manuscript_numbers.py - do not edit by hand"]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------- tables ----------------
    c = pd.read_csv(config.RESULTS_DIR / "stats" / "label_combinations.csv")
    yn = lambda v: "Yes" if v else "No"  # noqa: E731
    rows = [f"{yn(r.stairs_present)} & {yn(r.gate_present)} & {yn(r.ramp_present)} & "
            f"{yn(r.accessible_rule)} & {fmt_int(r.n)} & {pct(r.share)} \\\\" for r in c.itertuples()]
    (GEN / "tab_label_combinations.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    nb = pd.read_csv(config.RESULTS_DIR / "stats" / "labels_by_neighbourhood.csv")
    rows = [f"{r.nta_name.replace('-', '--')} & {fmt_int(r.n_annotated)} & {pct(r.share_unusable)} & "
            f"{pct(r.share_stairs)} & {pct(r.share_gate)} & {pct(r.share_ramp)} & {pct(r.share_accessible)} & "
            f"{pct(r.mean_p_oof)} \\\\" for r in nb.itertuples()]
    (GEN / "tab_neighbourhoods.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    recs = []
    R = config.RECORDS_DIR
    for path, desc in [(config.BUILDINGS_CSV, "All constructed buildings in Manhattan with attributes"),
                       (config.LABELS_CSV, "Image acquisition metadata, entrance features (reviewed and AI values), "
                                           "source of each value, both labels"),
                       (config.AI_RAW_JSONL, "Verbatim model outputs with request metadata"),
                       (config.PROMPT_TXT, "Exact prompt sent with every image"),
                       (R / "streetview_gate_ramp_review.csv", "Human review answers for AI-reported gates and ramps "
                                                               "and controls"),
                       (R / "human_validation_annotations.csv", "Blind human validation annotations with sampling "
                                                                "design"),
                       (config.PREDICTIONS_CSV, "Primary and AI-only probability, 90\\% intervals, out-of-fold "
                                                "probabilities, domain flags"),
                       (R / "manhattan_buildings_with_amr_index.csv", "Join: every building with attributes, labels "
                                                                      "and both indices"),
                       (R / "training_buildings.csv", "Join: labelled buildings used for training and validation")]:
        if path.suffix == ".csv":
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
            ncol = len(open(path, encoding="utf-8").readline().split(","))
            shape = f"{fmt_int(n)} $\\times$ {ncol}"
        elif path.suffix == ".jsonl":
            shape = f"{fmt_int(sum(1 for _ in open(path, encoding='utf-8')))} lines"
        else:
            shape = "--"
        size = path.stat().st_size / 1e6
        recs.append(f"\\texttt{{{path.name.replace('_', chr(92) + '_')}}} & {shape} & {size:.1f} & {desc} \\\\")
    (GEN / "tab_records.tex").write_text("\n".join(recs) + "\n", encoding="utf-8")
    print(f"{len(macros)} macros -> {GEN / 'numbers.tex'}")


if __name__ == "__main__":
    main()
