"""Central configuration: paths, constants and random seed.

All paths are relative to the repository root. Two environment variables can
override the locations of inputs that are not redistributed:

* ``AMR_RAW_DIR``      – full raw downloads from NYC Open Data (see data/raw/MANIFEST.csv)
* ``AMR_PRIVATE_DIR``  – intermediate files that cannot be redistributed
                         (Street View images, download log with API URLs,
                         raw OpenAI API responses)
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RECORDS_DIR = DATA_DIR / "records"          # published Data Records (deposited on Zenodo)
DICTIONARY_DIR = DATA_DIR / "dictionaries"  # one data dictionary per record
RAW_DIR = Path(os.environ.get("AMR_RAW_DIR", DATA_DIR / "raw"))
PRIVATE_DIR = Path(os.environ.get("AMR_PRIVATE_DIR", DATA_DIR / "private"))
RESULTS_DIR = ROOT / "results"              # regenerated analysis outputs
VALIDATION_DIR = ROOT / "validation_kit"

# Published record file names -------------------------------------------------
BUILDINGS_CSV = RECORDS_DIR / "manhattan_buildings.csv"
LABELS_CSV = RECORDS_DIR / "streetview_entrance_labels.csv"
AI_RAW_JSONL = RECORDS_DIR / "streetview_ai_raw_outputs.jsonl"
PREDICTIONS_CSV = RECORDS_DIR / "amr_accessibility_predictions.csv"
PROMPT_TXT = RECORDS_DIR / "streetview_ai_prompt.txt"

# Reproducibility -------------------------------------------------------------
SEED = 42
N_BOOTSTRAP = 200          # bootstrap refits used for prediction intervals
CV_FOLDS = 5
MIN_AREA_TEST_BUILDINGS = 30  # minimum held-out buildings for a spatial CV fold

# Unit conversion ---------------------------------------------------------------
FT_TO_M = 0.3048

# Coordinate reference systems ----------------------------------------------
CRS_GEOGRAPHIC = "EPSG:4326"
CRS_PROJECTED = "EPSG:2263"  # NY State Plane Long Island (US ft), used by NYC agencies
