"""Apply the targeted human review of gates and ramps to the label record.

Reads   data/records/streetview_entrance_labels.csv
        data/records/streetview_gate_ramp_review.csv
        data/records/human_validation_annotations.csv (Set A answers are reused)
Writes  data/records/streetview_entrance_labels.csv (in place; idempotent: the AI
        values are kept in *_ai columns and always used as the starting point)

See amr_access/review.py for the decision rules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.review import apply_review  # noqa: E402

REVIEW_CSV = config.RECORDS_DIR / "streetview_gate_ramp_review.csv"
HUMAN_CSV = config.RECORDS_DIR / "human_validation_annotations.csv"
AI_COLS = ["gate_present_ai", "ramp_present_ai", "accessible_rule_ai"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inconclusive", choices=["missing", "keep_ai"], default="missing")
    args = ap.parse_args()
    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str, "nta2020": str, "zipcode": str, "pano_id": str})
    review = pd.read_csv(REVIEW_CSV, dtype={"bin": str})
    human = pd.read_csv(HUMAN_CSV, dtype=str, keep_default_na=False) if HUMAN_CSV.exists() else None
    out = apply_review(lab, review, human, inconclusive=args.inconclusive)

    base = [c for c in lab.columns if c not in AI_COLS + ["gate_ramp_source"]]
    i = base.index("accessible_rule") + 1
    cols = base[:i] + ["gate_ramp_source"] + AI_COLS + base[i:]
    out = out[cols]
    out.to_csv(config.LABELS_CSV, index=False, float_format="%.10g", lineterminator="\n")

    u = out[out["image_usable"].astype(str).str.lower() == "true"]
    print(u["gate_ramp_source"].value_counts().to_string())
    for c in ["gate_present", "ramp_present"]:
        ai = u[c + "_ai"].astype(str).str.lower().eq("true").mean()
        fin = u[c].astype(str).str.lower().eq("true").mean()
        print(f"{c}: AI {ai:.3f} -> final {fin:.3f}")
    print(f"accessible_rule: AI {u['accessible_rule_ai'].mean():.3f} (n={u['accessible_rule_ai'].notna().sum()}) "
          f"-> final {u['accessible_rule'].mean():.3f} (n={u['accessible_rule'].notna().sum()})")


if __name__ == "__main__":
    main()
