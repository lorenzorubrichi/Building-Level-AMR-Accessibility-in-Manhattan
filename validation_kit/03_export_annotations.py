"""Export the human annotations as a published Data Record.

Input : raw exports of annotator.html (AMR_PRIVATE_DIR/annotations/annotations_*.csv)
Output: data/records/human_validation_annotations.csv

The published record keeps every judgement, the annotator ID and the sampling
design (set, stratum, stratum size) needed to reproduce the weighted metrics.
Working-session metadata (timestamps) are not published; the raw exports are
archived privately by the authors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amr_access import config  # noqa: E402

OUT = config.RECORDS_DIR / "human_validation_annotations.csv"
COLUMNS = ["annotator", "task_id", "bin", "set", "stratum", "stratum_size", "double_annotation",
           "usable", "reason", "stairs", "gate", "ramp", "robot", "comment"]


def main() -> None:
    raw = sorted((config.PRIVATE_DIR / "annotations").glob("annotations_*.csv"))
    if not raw:
        sys.exit("No raw annotation exports found in AMR_PRIVATE_DIR/annotations/")
    ann = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in raw], ignore_index=True)
    man = pd.read_csv(config.VALIDATION_DIR / "annotation_manifest_full.csv", dtype={"bin": str})
    out = ann.merge(man, on=["task_id", "image_file"], how="left", validate="many_to_one")
    if out["bin"].isna().any():
        raise ValueError("Annotations contain tasks that are not in the manifest")
    for c in ["stairs", "gate", "ramp", "robot"]:            # answers are undefined for non-judgeable images
        out.loc[out["usable"] != "yes", c] = ""
    out = out[COLUMNS].sort_values(["annotator", "task_id"])
    out.to_csv(OUT, index=False, lineterminator="\n")
    print(f"{OUT.name}: {len(out)} judgements from {out['annotator'].nunique()} annotators")


if __name__ == "__main__":
    main()
