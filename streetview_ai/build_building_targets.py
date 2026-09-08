"""Build a Street View target list.

Preferred source:
- final_last_meter_database.csv
- complete_last_meter_dataset_extended.csv

Fallback source:
- BUILDING.csv

If address-point coordinates are available, they are used. Otherwise the script
falls back to centroid coordinates, and if those are missing it computes a
simple centroid from the building footprint WKT.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FINAL_DATASET_CSV = BASE_DIR / "final_last_meter_database.csv"
EXTENDED_DATASET_CSV = BASE_DIR / "complete_last_meter_dataset_extended.csv"
BUILDING_CSV = BASE_DIR / "BUILDING.csv"
OUTPUT_CSV = DATA_DIR / "building_targets.csv"

DEFAULT_LIMIT: int | None = None
POINT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Street View targets from the best available building dataset")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Optional row limit. By default, use all rows from the selected CSV.",
    )
    return parser.parse_args()


def polygon_centroid_from_wkt(wkt: str) -> tuple[float, float] | None:
    if not isinstance(wkt, str):
        return None
    coords = [(float(x), float(y)) for x, y in POINT_RE.findall(wkt)]
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def choose_input_csv() -> Path:
    if FINAL_DATASET_CSV.exists():
        return FINAL_DATASET_CSV
    if EXTENDED_DATASET_CSV.exists():
        return EXTENDED_DATASET_CSV
    if BUILDING_CSV.exists():
        return BUILDING_CSV
    raise FileNotFoundError(
        f"Missing input file. Expected {FINAL_DATASET_CSV}, {EXTENDED_DATASET_CSV}, or {BUILDING_CSV}"
    )


def main(limit: int | None = DEFAULT_LIMIT) -> None:
    input_csv = choose_input_csv()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    read_kwargs = {"nrows": limit} if limit is not None else {}
    df = pd.read_csv(input_csv, **read_kwargs)
    required = ["bin"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "borough" in df.columns:
        borough_col = "borough"
    else:
        borough_col = "borough_nta" if "borough_nta" in df.columns else None

    if "neighborhood" in df.columns:
        neighborhood_col = "neighborhood"
    else:
        neighborhood_col = "nta_name" if "nta_name" in df.columns else None

    centroids = None
    if "the_geom" in df.columns:
        centroids = df["the_geom"].apply(polygon_centroid_from_wkt)

    out = pd.DataFrame({"bin": df["bin"]})
    if borough_col:
        out["borough"] = df[borough_col]
    if neighborhood_col:
        out["neighborhood"] = df[neighborhood_col]

    if "address_lon" in df.columns and "address_lat" in df.columns:
        out["address_lon"] = df["address_lon"]
        out["address_lat"] = df["address_lat"]
    else:
        out["address_lon"] = None
        out["address_lat"] = None

    if "centroid_lon" in df.columns and "centroid_lat" in df.columns:
        fallback_lon = df["centroid_lon"]
        fallback_lat = df["centroid_lat"]
    elif centroids is not None:
        fallback_lon = centroids.apply(lambda p: None if p is None else p[0])
        fallback_lat = centroids.apply(lambda p: None if p is None else p[1])
    else:
        fallback_lon = pd.Series([None] * len(df))
        fallback_lat = pd.Series([None] * len(df))

    used_address_point = out["address_lon"].notna() & out["address_lat"].notna()
    needs_fallback = ~used_address_point
    if needs_fallback.any():
        out.loc[needs_fallback, "address_lon"] = fallback_lon[needs_fallback]
        out.loc[needs_fallback, "address_lat"] = fallback_lat[needs_fallback]
    out["location_source"] = used_address_point.map(
        {True: "address_point", False: "centroid_fallback"}
    )
    out = out.dropna(subset=["address_lon", "address_lat"]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    address_count = int((out["location_source"] == "address_point").sum())
    centroid_count = int((out["location_source"] == "centroid_fallback").sum())
    print(f"Saved {len(out)} building targets to: {OUTPUT_CSV}")
    print(f"Address-point targets: {address_count}")
    print(f"Centroid fallback targets: {centroid_count}")
    print(f"Source CSV: {input_csv}")


if __name__ == "__main__":
    args = parse_args()
    main(args.limit)
