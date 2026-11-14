"""Run the pipeline end to end (cross-platform; no make required).

    python run_all.py analysis    # from the published records only (anyone)
    python run_all.py records     # rebuild the records from the source snapshot + private inputs (authors)
    python run_all.py all         # both
    python run_all.py test        # unit and integrity tests

Stage 'records' needs data/source_snapshot/ (deposited with the records) and,
for the Street View/AI fields, the private files listed in data/private/README.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

PRIVATE_FILES = ["streetview_api_raw_responses.jsonl", "streetview_download_log.csv", "streetview_ai_prompt.txt"]
ANALYSIS = [
    ["pipeline/b_analysis/15_feature_set_comparison.py"],
    ["pipeline/b_analysis/10_model_and_validation.py"],
    ["pipeline/b_analysis/14_write_dictionaries.py"],
    ["pipeline/b_analysis/16_joined_tables.py"],
    ["validation_kit/02_score_annotations.py"],
    ["pipeline/b_analysis/11_dataset_statistics.py"],
    ["pipeline/b_analysis/12_figures.py"],
    ["pipeline/b_analysis/13_manuscript_numbers.py"],
]


def run(steps: list[list[str]], env: dict | None = None) -> None:
    for step in steps:
        print(f"\n=== {step[0]}", flush=True)
        subprocess.run([PY, *step], cwd=ROOT, check=True, env=env)


REQUIRED = {
    "records": [ROOT / "data" / "source_snapshot" / "BUILDING.csv.gz"],
    "analysis": [ROOT / "data" / "records" / "manhattan_buildings.csv",
                 ROOT / "data" / "records" / "streetview_entrance_labels.csv"],
}


def check_inputs(stage: str) -> None:
    missing = [p for p in REQUIRED.get(stage, []) if not p.exists()]
    if missing:
        sys.exit("Missing input files:\n  " + "\n  ".join(str(p.relative_to(ROOT)) for p in missing)
                 + "\nThey are part of the repository; if you downloaded only the code, get the data "
                   "archive (see README, 'Data Records').")


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "analysis"
    for s in (["records", "analysis"] if stage == "all" else [stage]):
        check_inputs(s)
    env = dict(os.environ)
    if stage in {"records", "all"}:
        env.setdefault("AMR_RAW_DIR", str(ROOT / "data" / "source_snapshot"))
        run([["pipeline/a_build_records/01_build_building_attributes.py"]], env)
        private = Path(env.get("AMR_PRIVATE_DIR", ROOT / "data" / "private"))
        if all((private / f).exists() for f in PRIVATE_FILES):
            run([["pipeline/a_build_records/02_build_records.py"]], env)
        else:
            print("\n[records] Private acquisition files not found: the published records in data/records/ "
                  "are kept;\n          the rebuilt building table is compared with the published one instead.")
            run([["pipeline/a_build_records/check_rebuild.py"]], env)
        # targeted human review of gates/ramps -> final labels (idempotent; reads published records)
        run([["pipeline/a_build_records/07_apply_human_review.py"]], env)
    if stage in {"analysis", "all"}:
        run(ANALYSIS, env)
    if stage == "test" or stage == "all":
        subprocess.run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
