"""Export the targeted gate/ramp review as a published Data Record.

Input : raw exports of review_tool.html (AMR_PRIVATE_DIR/review/review_*.csv)
Output: data/records/streetview_gate_ramp_review.csv

Keeps every answer, the annotator ID and the sampling design (group, group size,
double annotation). Working-session timestamps are not published.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amr_access import config  # noqa: E402

OUT = config.RECORDS_DIR / "streetview_gate_ramp_review.csv"
COLUMNS = ["annotator", "task_id", "bin", "group", "group_size", "double_annotation", "answer"]
ANSWERS = {"gate", "ramp", "both", "none", "cannot_judge"}


def main() -> None:
    raw = sorted((config.PRIVATE_DIR / "review").glob("review_*.csv"))
    if not raw:
        sys.exit("No raw review exports found in AMR_PRIVATE_DIR/review/")
    rev = pd.concat([pd.read_csv(p, dtype=str, keep_default_na=False) for p in raw], ignore_index=True)
    man = pd.read_csv(config.VALIDATION_DIR / "review" / "review_manifest.csv", dtype={"bin": str})
    out = rev.merge(man, on=["task_id", "image_file"], how="left", validate="many_to_one")
    if out["bin"].isna().any():
        raise ValueError("Review answers contain tasks that are not in the manifest")
    if not set(out["answer"]) <= ANSWERS:
        raise ValueError(f"Unexpected answers: {set(out['answer']) - ANSWERS}")
    if out.duplicated(["annotator", "task_id"]).any():
        raise ValueError("Duplicate answers for the same annotator and task")
    out = out[COLUMNS].sort_values(["annotator", "task_id"])
    out.to_csv(OUT, index=False, lineterminator="\n")
    print(f"wrote {OUT.relative_to(config.ROOT)}: {len(out):,} answers, annotators {sorted(out.annotator.unique())}")


if __name__ == "__main__":
    main()
