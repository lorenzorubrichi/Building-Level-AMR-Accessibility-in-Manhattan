"""Targeted human review of AI-reported gates and ramps.

The multimodal model over-reports gates and ramps (validation Set A: gate precision
about 0.3, ramp precision about 0.1) while it rarely misses them. Every usable image
in which the AI reported a gate or a ramp was therefore checked by a person, together
with a random sample of control images in which it reported neither.

Final values (columns ``gate_present`` / ``ramp_present``):

* image judged in the targeted review  -> human answer; with two annotators, the
  feature counts as present if at least one of them saw it;
* image in validation Set A            -> the Set A answers (same union rule);
* reviewer(s) answered "cannot judge"   -> gate/ramp and the rule label are left empty
  (the person could not see the entrance; same convention as an unusable image).
  ``inconclusive="keep_ai"`` keeps the AI values instead (sensitivity analysis);
* image not reviewed (AI reported neither feature and not drawn as a control)
                                       -> the AI value (false) is kept.

The AI values are preserved in ``gate_present_ai`` / ``ramp_present_ai`` and the
AI-only rule label in ``accessible_rule_ai``. ``gate_ramp_source`` records where the
final values come from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import rule_label

SOURCES = {
    "ai": "AI value; image not selected for human review",
    "human_review": "targeted human review (one or two annotators)",
    "human_validation_set_a": "answers from human-validation Set A",
    "human_review_not_judgeable": "selected for review but the annotator(s) could not judge the entrance; gate, ramp and label left empty",
    "ai_review_inconclusive": "(sensitivity variant only) as above, but the AI values are kept",
}


def _as_bool(s: pd.Series) -> pd.Series:
    m = {"true": True, "false": False, "1": True, "0": False, "1.0": True, "0.0": False}
    return s.astype("string").str.lower().map(m).astype("boolean")


def review_decisions(review: pd.DataFrame) -> pd.DataFrame:
    """One row per BIN from the review record: gate, ramp (boolean) or NA if not judgeable."""
    r = review.copy()
    r["bin"] = r["bin"].astype(str)
    ok = r["answer"] != "cannot_judge"
    r["gate"] = r["answer"].isin(["gate", "both"])
    r["ramp"] = r["answer"].isin(["ramp", "both"])
    judged = r[ok].groupby("bin")[["gate", "ramp"]].any()
    out = pd.DataFrame(index=pd.Index(r["bin"].unique(), name="bin"))
    out = out.join(judged)
    out["n_annotators"] = r.groupby("bin").size()
    return out.reset_index()


def validation_decisions(human: pd.DataFrame, sets=("A",)) -> pd.DataFrame:
    """Gate/ramp answers from the validation record (union of annotators; unsure ignored)."""
    h = human[human["set"].isin(sets)].copy()
    h["bin"] = h["bin"].astype(str)
    usable = h["usable"].astype(str).str.lower() == "yes"
    rows = {}
    for c in ["gate", "ramp"]:
        v = h[c].astype(str).str.lower().map({"yes": True, "no": False})
        v[~usable] = np.nan
        rows[c] = v
    d = pd.DataFrame({"bin": h["bin"], **rows})
    agg = d.groupby("bin").agg(gate=("gate", lambda x: x.dropna().any() if x.notna().any() else np.nan),
                               ramp=("ramp", lambda x: x.dropna().any() if x.notna().any() else np.nan))
    return agg.reset_index()


def apply_review(lab: pd.DataFrame, review: pd.DataFrame, human: pd.DataFrame | None,
                 inconclusive: str = "missing") -> pd.DataFrame:
    """Return the label table with AI columns preserved and gate/ramp/rule revised."""
    lab = lab.copy()
    lab["bin"] = lab["bin"].astype(str)
    if "gate_present_ai" not in lab:                         # first application
        lab["gate_present_ai"] = lab["gate_present"]
        lab["ramp_present_ai"] = lab["ramp_present"]
        lab["accessible_rule_ai"] = lab["accessible_rule"]
    usable = _as_bool(lab["image_usable"]).fillna(False).astype(bool)
    gate_ai = _as_bool(lab["gate_present_ai"])
    ramp_ai = _as_bool(lab["ramp_present_ai"])
    gate, ramp = gate_ai.copy(), ramp_ai.copy()
    source = pd.Series(np.where(usable, "ai", ""), index=lab.index, dtype=object)

    def put(dec: pd.DataFrame, name: str) -> None:
        idx = lab.reset_index().merge(dec, on="bin", how="inner")
        idx = idx[usable.to_numpy()[idx["index"].to_numpy()]]
        for _, row in idx.iterrows():
            i = row["index"]
            g_ok, r_ok = pd.notna(row["gate"]), pd.notna(row["ramp"])
            if g_ok:
                gate.at[i] = bool(row["gate"])
            if r_ok:
                ramp.at[i] = bool(row["ramp"])
            if g_ok or r_ok:
                source.at[i] = name
            elif inconclusive == "missing":
                gate.at[i], ramp.at[i] = pd.NA, pd.NA
                source.at[i] = "human_review_not_judgeable"
            else:
                source.at[i] = "ai_review_inconclusive"

    put(review_decisions(review), "human_review")
    if human is not None:
        put(validation_decisions(human), "human_validation_set_a")

    lab["gate_present"] = gate.where(usable)
    lab["ramp_present"] = ramp.where(usable)
    lab["gate_ramp_source"] = source.replace("", np.nan)
    stairs = _as_bool(lab["stairs_present"])
    lab["accessible_rule"] = rule_label(stairs.fillna(False), lab["gate_present"].fillna(False),
                                        lab["ramp_present"].fillna(False), usable)
    lab.loc[source == "human_review_not_judgeable", "accessible_rule"] = pd.NA
    return lab
