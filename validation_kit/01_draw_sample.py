"""Draw the blinded, stratified sample for human validation.

Set A  (AI agreement)      images already annotated by the AI model, stratified by
                           the stairs/gate/ramp combination (+ unusable images),
                           used to estimate per-attribute agreement AI vs human.
Set B  (model validation)  images acquired but never sent to the AI model:
                           B1 = Upper East Side-Carnegie Hill (a neighbourhood with no
                                training labels: external spatial test),
                           B2 = the seven labelled neighbourhoods (interpolation test).
                           Human labels are compared with the predicted probability
                           in amr_accessibility_predictions.csv.
Overlap                    a random subset of Set A is annotated independently by a
                           second annotator (inter-annotator agreement).

Output: validation_kit/annotation_manifest.csv  (no AI labels, random task order)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amr_access import config  # noqa: E402

EXTERNAL_NTA = "Upper East Side-Carnegie Hill"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=400, help="Set A size")
    ap.add_argument("--min-per-stratum", type=int, default=20)
    ap.add_argument("--n-b1", type=int, default=200, help="Set B1 (external neighbourhood) size")
    ap.add_argument("--n-b2", type=int, default=100, help="Set B2 (labelled neighbourhoods, unannotated images)")
    ap.add_argument("--n-overlap", type=int, default=100, help="Set A items double-annotated")
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str})
    if "gate_present_ai" in lab:                     # the samples were drawn from the AI values
        lab["gate_present"], lab["ramp_present"] = lab["gate_present_ai"], lab["ramp_present_ai"]
    b = pd.read_csv(config.BUILDINGS_CSV, dtype={"bin": str})

    # ---- Set A: stratified by AI combination, unusable images as their own stratum
    lab["stratum"] = np.where(
        ~lab["image_usable"], "unusable",
        "S" + lab["stairs_present"].astype(str).str[0] + "G" + lab["gate_present"].astype(str).str[0]
        + "R" + lab["ramp_present"].astype(str).str[0])
    sizes = lab["stratum"].value_counts()
    alloc = np.maximum(args.min_per_stratum, np.round(args.n_a * sizes / sizes.sum())).astype(int)
    alloc = np.minimum(alloc, sizes)
    while alloc.sum() > args.n_a:                     # trim the largest strata back to n_a
        alloc[alloc.idxmax()] -= 1
    parts = []
    for stratum, k in alloc.items():
        pool = lab[lab["stratum"] == stratum]
        pick = pool.sample(n=int(k), random_state=int(rng.integers(1e9)))
        parts.append(pick.assign(set="A", stratum=stratum, stratum_size=int(sizes[stratum])))
    set_a = pd.concat(parts)[["bin", "set", "stratum", "stratum_size"]]

    # ---- Set B: imaged but never annotated
    pool = b[b["streetview_image_acquired"] & ~b["in_ai_annotated_sample"]]
    b1 = pool[pool["nta_name"] == EXTERNAL_NTA].sample(n=args.n_b1, random_state=int(rng.integers(1e9)))
    b2_pool = pool[pool["nta_name"].isin(lab["nta_name"].dropna().unique())]
    b2 = b2_pool.sample(n=args.n_b2, random_state=int(rng.integers(1e9)))
    set_b = pd.concat([b1.assign(set="B1"), b2.assign(set="B2")])[["bin", "set"]]
    set_b["stratum"], set_b["stratum_size"] = set_b["set"], set_b["set"].map(
        {"B1": int((pool["nta_name"] == EXTERNAL_NTA).sum()), "B2": len(b2_pool)})

    manifest = pd.concat([set_a, set_b], ignore_index=True)
    manifest["double_annotation"] = False
    overlap = manifest[manifest["set"] == "A"].sample(n=args.n_overlap, random_state=int(rng.integers(1e9))).index
    manifest.loc[overlap, "double_annotation"] = True
    manifest = manifest.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    manifest.insert(0, "task_id", [f"T{i:04d}" for i in range(1, len(manifest) + 1)])
    manifest["image_file"] = manifest["bin"] + ".jpg"

    # strata information is kept for scoring only; annotators see task_id + image_file
    out = config.VALIDATION_DIR
    manifest.to_csv(out / "annotation_manifest_full.csv", index=False)
    manifest[["task_id", "image_file"]].to_csv(out / "tasks_annotator_1.csv", index=False)
    manifest.loc[manifest["double_annotation"], ["task_id", "image_file"]].to_csv(out / "tasks_annotator_2.csv",
                                                                                   index=False)
    print(manifest["set"].value_counts().to_dict(), "double:", int(manifest["double_annotation"].sum()))
    print(alloc.to_dict())


if __name__ == "__main__":
    main()
