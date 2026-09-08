"""Scaffold for merging Street View / AI features into a building-level dataset."""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
BASE_DATASET = DATA_DIR / "base_building_dataset.csv"
VISUAL_FEATURES = OUTPUT_DIR / "streetview_visual_features.csv"
OUTPUT_CSV = OUTPUT_DIR / "building_dataset_with_visual_features.csv"


def main() -> None:
    if not BASE_DATASET.exists():
        print(f"Missing base dataset: {BASE_DATASET}")
        return
    if not VISUAL_FEATURES.exists():
        print(f"Missing visual features dataset: {VISUAL_FEATURES}")
        return

    base_df = pd.read_csv(BASE_DATASET)
    visual_df = pd.read_csv(VISUAL_FEATURES)
    if "bin" not in base_df.columns or "bin" not in visual_df.columns:
        raise ValueError("Both datasets must contain 'bin'")

    merged = base_df.merge(visual_df, on="bin", how="left")
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged dataset: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
