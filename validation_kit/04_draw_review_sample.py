"""Task lists for the targeted human review of gates and ramps (option 4).

Review set   every usable image in which the AI reported a gate or a ramp, except those
             already judged by annotator A1 in validation Set A (their answers are reused).
Controls     a random sample of usable images in which the AI reported neither, mixed in
             so that (i) the annotator cannot infer the AI output from the task list and
             (ii) gates/ramps missed by the AI can be estimated.
Double check a random subset annotated by a second person (inter-annotator agreement).

Writes validation_kit/review/{review_manifest.csv, review_tasks_A1.csv, review_tasks_A2.csv}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amr_access import config  # noqa: E402

OUT = config.VALIDATION_DIR / "review"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-controls", type=int, default=300)
    ap.add_argument("--n-double", type=int, default=300)
    ap.add_argument("--seed", type=int, default=config.SEED + 1)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str})
    if "gate_present_ai" in lab:                     # the samples were drawn from the AI values
        lab["gate_present"], lab["ramp_present"] = lab["gate_present_ai"], lab["ramp_present_ai"]
    man = pd.read_csv(config.VALIDATION_DIR / "annotation_manifest_full.csv", dtype={"bin": str})
    in_set_a = set(man.loc[man["set"] == "A", "bin"])
    u = lab[lab["image_usable"].astype(bool) & ~lab["bin"].isin(in_set_a)].copy()
    positive = u["gate_present"].astype(bool) | u["ramp_present"].astype(bool)

    review = u[positive].assign(group="ai_positive")
    controls = u[~positive].sample(n=args.n_controls, random_state=int(rng.integers(1e9))).assign(group="control")
    m = pd.concat([review, controls])[["bin", "group"]]
    m["group_size"] = m["group"].map({"ai_positive": int(positive.sum()), "control": int((~positive).sum())})
    m = m.sample(frac=1, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    m.insert(0, "task_id", [f"R{i:04d}" for i in range(1, len(m) + 1)])
    m["image_file"] = m["bin"] + ".jpg"
    m["double_annotation"] = False
    m.loc[m.sample(n=args.n_double, random_state=int(rng.integers(1e9))).index, "double_annotation"] = True

    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "review_manifest.csv", index=False)
    m[["task_id", "image_file"]].to_csv(OUT / "review_tasks_A1.csv", index=False)
    m.loc[m["double_annotation"], ["task_id", "image_file"]].to_csv(OUT / "review_tasks_A2.csv", index=False)
    print(f"review tasks: {len(m):,} ({int((m.group == 'ai_positive').sum()):,} AI-positive, "
          f"{int((m.group == 'control').sum())} controls); second annotator: {int(m.double_annotation.sum())}")
    print(f"Set A images reused (not re-annotated): {int(lab['bin'].isin(in_set_a).sum())}")


if __name__ == "__main__":
    main()
