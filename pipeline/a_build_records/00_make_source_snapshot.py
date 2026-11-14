"""Write a Manhattan-only snapshot of the raw NYC Open Data inputs.

NYC Open Data overwrites its datasets in place, so a later download will not
reproduce the published records exactly. This script filters the full raw
downloads (AMR_RAW_DIR) to Manhattan, keeping every original column and the
original text formatting, and writes gzip-compressed copies to
data/source_snapshot/. The snapshot is deposited with the Data Records and is
the default input of 01_build_building_attributes.py.

Street centreline and sidewalk layers are included when present (needed only
to recompute camera points and sidewalk distances).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402

SNAPSHOT_DIR = config.DATA_DIR / "source_snapshot"


def norm(c: str) -> str:
    return c.strip().lower().replace(" ", "_")


# PLUTO is reduced to the fields used (it also contains owner names, which are not needed)
COLUMN_SUBSETS = {"pluto_26v1.csv": ["BBL", "version", "numfloors", "landuse", "bldgclass", "unitsres", "unitstotal",
                                     "bldgarea", "numbldgs", "yearbuilt", "yearalter1", "yearalter2", "lotfront",
                                     "bldgfront", "bldgdepth", "bsmtcode", "proxcode", "histdist", "landmark"]}


def filter_csv(src: Path, dst: Path, keep_row) -> int:
    parts = []
    usecols = COLUMN_SUBSETS.get(src.name)
    with pd.read_csv(src, dtype=str, chunksize=200_000, keep_default_na=False, usecols=usecols) as reader:
        for chunk in reader:
            cols = {norm(c): c for c in chunk.columns}
            parts.append(chunk[keep_row(chunk, cols)])
    out = pd.concat(parts)
    out.to_csv(dst, index=False, compression={"method": "gzip", "mtime": 0})
    return len(out)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    raw = config.RAW_DIR

    def bin_manhattan(df, cols):
        return df[cols["bin"]].str.strip().str.startswith("1")

    def bbl_manhattan(df, cols):
        return df[cols["bbl"]].str.strip().str.startswith("1")

    def borough_code_1(df, cols):
        key = cols.get("borough_code") or cols.get("borocode") or cols.get("boro_code")
        return df[key].str.strip().isin(["1", "1.0"]) if key else pd.Series(True, index=df.index)

    def manhattan_bbox(df, cols):
        xy = df[cols["the_geom"]].str.extract(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)").astype(float)
        return xy[0].between(-74.06, -73.89) & xy[1].between(40.67, 40.89)

    jobs = [
        ("BUILDING.csv", bin_manhattan),
        ("AddressPoint_full.csv", bin_manhattan),
        ("pluto_26v1.csv", bbl_manhattan),
        ("Centerline.csv", borough_code_1),
        ("NYC_Sidewalk.csv", manhattan_bbox),
    ]
    for name, rule in jobs:
        src = raw / name
        if not src.exists():
            print(f"[skip] {name} not found in {raw}")
            continue
        dst = SNAPSHOT_DIR / (name + ".gz")
        n = filter_csv(src, dst, rule)
        print(f"[ok] {name}: {n:,} Manhattan rows -> {dst.name} (sha256 {sha256(dst)[:12]}…)")
    nta = raw / "nyc_neighborhoods.geojson"
    if nta.exists():
        import gzip
        import shutil
        with open(nta, "rb") as f_in, gzip.open(SNAPSHOT_DIR / "nyc_neighborhoods.geojson.gz", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("[ok] nyc_neighborhoods.geojson copied")


if __name__ == "__main__":
    main()
