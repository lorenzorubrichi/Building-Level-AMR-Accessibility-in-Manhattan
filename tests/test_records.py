"""Integrity checks on the published Data Records."""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amr_access import config  # noqa: E402
from amr_access.dictionary import RECORDS  # noqa: E402
from amr_access.labels import rule_label  # noqa: E402

pytestmark = pytest.mark.skipif(not config.BUILDINGS_CSV.exists(), reason="records not built")


@pytest.fixture(scope="module")
def recs():
    def load(name):
        p = config.RECORDS_DIR / name
        if name.endswith(".jsonl"):
            return pd.DataFrame([json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()])
        return pd.read_csv(p, dtype={"bin": str, "zipcode": str, "bbl": str, "landuse": str, "nta2020": str},
                           keep_default_na=name != "human_validation_annotations.csv")
    return {n: load(n) for n in RECORDS}


def test_columns_match_dictionaries(recs):
    for name, spec in RECORDS.items():
        assert list(recs[name].columns) == list(spec), name


def test_primary_keys(recs):
    for name in ["manhattan_buildings.csv", "streetview_entrance_labels.csv", "amr_accessibility_predictions.csv"]:
        b = recs[name]["bin"]
        assert b.is_unique and b.str.fullmatch(r"1\d{6}").all(), name
    assert set(recs["amr_accessibility_predictions.csv"]["bin"]) == set(recs["manhattan_buildings.csv"]["bin"])


def test_no_secrets_or_local_paths():
    for p in config.RECORDS_DIR.iterdir():
        text = p.read_text(encoding="utf-8", errors="ignore")
        assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}", text), p.name
        assert "C:\\Users" not in text and "/Users/" not in text, p.name


def test_value_ranges(recs):
    b = recs["manhattan_buildings.csv"]
    assert b["centroid_lon"].between(-74.05, -73.85).all() and b["centroid_lat"].between(40.68, 40.89).all()
    assert (b["height_roof_m"].dropna() < 600).all() and (b["height_roof_m"].dropna() > 0).all()
    assert b["construction_year"].dropna().between(1650, 2027).all()
    assert (b["footprint_area_m2"].dropna() > 0).all()
    assert (b["distance_to_sidewalk_m"].dropna() >= 0).all()
    assert b["zipcode"].dropna().str.fullmatch(r"\d{5}").all()
    p = recs["amr_accessibility_predictions.csv"]
    for c in ["p_accessible", "p_accessible_q05", "p_accessible_q95"]:
        assert p[c].between(0, 1).all()
    assert (p["p_accessible_q05"] <= p["p_accessible_q95"]).all()
    assert np.allclose(p["amr_index"], 100 * p["p_accessible"], atol=1e-3)


def _b(s):
    return s.astype("string").str.lower().map({"true": True, "false": False}).astype("boolean")


def test_label_rule_consistency(recs):
    lab = recs["streetview_entrance_labels.csv"]
    usable = lab["image_usable"].astype(bool)
    # AI-only label from the AI features
    exp_ai = rule_label(_b(lab["stairs_present"]).fillna(False), _b(lab["gate_present_ai"]).fillna(False),
                        _b(lab["ramp_present_ai"]).fillna(False), usable)
    assert exp_ai.astype("Float64").equals(lab["accessible_rule_ai"].astype("Float64"))
    # primary label from the reviewed features; empty where the reviewers could not judge
    nj = lab["gate_ramp_source"] == "human_review_not_judgeable"
    exp = rule_label(_b(lab["stairs_present"]).fillna(False), _b(lab["gate_present"]).fillna(False),
                     _b(lab["ramp_present"]).fillna(False), usable).astype("Float64")
    exp[nj] = pd.NA
    assert exp.equals(lab["accessible_rule"].astype("Float64"))
    assert lab.loc[nj, ["gate_present", "ramp_present", "accessible_rule"]].isna().all().all()
    cols = ["stairs_present", "gate_present", "ramp_present", "gate_present_ai", "ramp_present_ai", "gate_ramp_source"]
    assert lab.loc[~usable, cols].isna().all().all()


def test_review_applied_consistently(recs):
    from amr_access.review import SOURCES, apply_review
    lab = recs["streetview_entrance_labels.csv"]
    rev = recs["streetview_gate_ramp_review.csv"]
    usable = lab["image_usable"].astype(bool)
    assert set(lab.loc[usable, "gate_ramp_source"]) <= set(SOURCES)
    # features differ from the AI only where a person looked at the image
    ai_src = lab["gate_ramp_source"] == "ai"
    assert (_b(lab.loc[ai_src, "gate_present"]) == _b(lab.loc[ai_src, "gate_present_ai"])).all()
    assert (_b(lab.loc[ai_src, "ramp_present"]) == _b(lab.loc[ai_src, "ramp_present_ai"])).all()
    # every AI-positive usable image outside Set A was reviewed
    pos = usable & (_b(lab["gate_present_ai"]).fillna(False) | _b(lab["ramp_present_ai"]).fillna(False))
    assert not (pos & ai_src).any()
    assert set(rev["bin"]) <= set(lab.loc[usable, "bin"])
    # re-applying the review to the published table changes nothing (idempotent)
    human = pd.read_csv(config.RECORDS_DIR / "human_validation_annotations.csv", dtype=str, keep_default_na=False)
    again = apply_review(lab, rev, human)
    for c in ["gate_present", "ramp_present", "gate_ramp_source"]:
        assert again[c].astype("string").str.lower().fillna("").equals(lab[c].astype("string").str.lower().fillna("")), c
    assert again["accessible_rule"].astype("Float64").equals(lab["accessible_rule"].astype("Float64"))


def test_review_record(recs):
    rev = recs["streetview_gate_ramp_review.csv"]
    assert set(rev["answer"]) <= {"gate", "ramp", "both", "none", "cannot_judge"}
    assert not rev.duplicated(["annotator", "task_id"]).any()
    assert "time" not in rev.columns
    double = rev.groupby("task_id")["annotator"].nunique()
    flagged = rev.drop_duplicates("task_id").set_index("task_id")["double_annotation"].astype(bool)
    assert ((double == 2) == flagged.reindex(double.index)).all()


def test_labels_match_raw_outputs(recs):
    lab = recs["streetview_entrance_labels.csv"].set_index("bin")
    raw = recs["streetview_ai_raw_outputs.jsonl"].set_index("bin")
    assert set(lab.index) == set(raw.index)
    parsed = raw["output_text"].map(json.loads)
    stairs = parsed.map(lambda d: d["stairs_present"]).reindex(lab.index)
    usable = lab["image_usable"].astype(bool)
    assert (stairs[usable].astype(bool) == lab.loc[usable, "stairs_present"].astype(bool)).all()
    assert (raw["response_id"].reindex(lab.index) == lab["ai_response_id"]).all()


def test_oof_only_for_training(recs):
    p = recs["amr_accessibility_predictions.csv"].set_index("bin")
    lab = recs["streetview_entrance_labels.csv"]
    base = lab["image_usable"] & lab["in_building_table"]
    assert set(p.index[p["p_accessible_oof"].notna()]) == set(lab.loc[base & lab["accessible_rule"].notna(), "bin"])
    assert set(p.index[p["p_accessible_ai_only_oof"].notna()]) == set(lab.loc[base, "bin"])
    assert np.allclose(p["amr_index_ai_only"], 100 * p["p_accessible_ai_only"], atol=1e-3)
    assert (p["p_accessible_ai_only_q05"] <= p["p_accessible_ai_only_q95"]).all()


def test_new_fields(recs):
    b = recs["manhattan_buildings.csv"]
    assert (b["pluto_version"].dropna() == "26v1").all()
    assert b["basement_code"].dropna().isin([0, 1, 2, 3, 4, 5]).all()
    assert b["proximity_code"].dropna().isin([0, 1, 2, 3]).all()
    assert (b["distance_to_street_m"] >= 0).all() and (b["n_address_points"] >= 0).all()
    assert (b["last_alteration_year"].dropna() >= b[["yearalter1", "yearalter2"]].max(axis=1).dropna().min()).all()
    lab = recs["streetview_entrance_labels.csv"]
    assert (lab["qa_camera_far"] == (lab["camera_target_distance_m"] > 30)).all()
    g = lab.dropna(subset=["duplicate_request_group"]).groupby("duplicate_request_group").size()
    assert (g >= 2).all()


def test_joined_tables(recs):
    ids = {"bin": str, "bbl": str, "zipcode": str, "landuse": str, "nta2020": str}
    t = pd.read_csv(config.RECORDS_DIR / "training_buildings.csv", dtype=ids, low_memory=False)
    a = pd.read_csv(config.RECORDS_DIR / "manhattan_buildings_with_amr_index.csv", dtype=ids, low_memory=False)
    b = recs["manhattan_buildings.csv"]
    assert len(a) == len(b) and set(a["bin"]) == set(b["bin"]) and a["bin"].is_unique
    assert t["bin"].is_unique and t["accessible_rule_ai"].notna().all() and t["p_accessible_ai_only_oof"].notna().all()
    assert (t["accessible_rule"].notna() == t["p_accessible_oof"].notna()).all()
    lab = recs["streetview_entrance_labels.csv"]
    assert len(t) == int((lab["image_usable"] & lab["in_building_table"]).sum())
    merged = a.merge(recs["amr_accessibility_predictions.csv"][["bin", "p_accessible"]], on="bin")
    assert np.allclose(merged["p_accessible_x"], merged["p_accessible_y"])


def test_in_building_table_flag(recs):
    lab, b = recs["streetview_entrance_labels.csv"], recs["manhattan_buildings.csv"]
    assert (lab["in_building_table"] == lab["bin"].isin(b["bin"])).all()


def test_human_annotations(recs):
    h = recs["human_validation_annotations.csv"]
    assert set(h["annotator"]) == {"A1", "A2"}
    assert not h.duplicated(["annotator", "task_id"]).any()
    a1, a2 = h[h.annotator == "A1"], h[h.annotator == "A2"]
    assert len(a1) == 700 and set(a2["task_id"]) <= set(a1.loc[a1["double_annotation"], "task_id"])
    for c in ["stairs", "gate", "ramp", "robot"]:
        assert h[c].isin(["yes", "no", "unsure", ""]).all()
        assert (h.loc[h["usable"] != "yes", c] == "").all()
    assert "time" not in h.columns
