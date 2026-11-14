"""Compare the rebuilt building table (data/interim) with the published record.

Prints, for every shared column, whether the values agree up to the precision
written in the published CSV. Exit code 1 if any column differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402


def main() -> None:
    ids = {"bin": str, "bbl": str, "zipcode": str, "landuse": str, "nta2020": str}
    new = pd.read_csv(config.DATA_DIR / "interim" / "building_attributes.csv", dtype=ids)
    pub = pd.read_csv(config.BUILDINGS_CSV, dtype=ids)
    cols = [c for c in pub.columns if c in new.columns and c != "bin"]
    m = pub[["bin"] + cols].merge(new[["bin"] + cols], on="bin", how="outer", suffixes=("_pub", "_new"),
                                  indicator=True)
    ok = True
    if (m["_merge"] != "both").any():
        ok = False
        print(f"BIN sets differ: {(m['_merge'] != 'both').sum()} rows")
    for c in cols:
        a, b = m[c + "_pub"], m[c + "_new"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            same = np.isclose(a, b, rtol=1e-8, atol=1e-9, equal_nan=True)
        else:
            same = (a.astype("string").fillna("") == b.astype("string").fillna("")).to_numpy()
        if not same.all():
            ok = False
            print(f"  {c}: {int((~same).sum())} values differ")
    print(f"Rebuilt building table: {len(new):,} rows, {len(cols)} columns compared -> "
          + ("IDENTICAL to data/records/manhattan_buildings.csv" if ok else "DIFFERENCES found"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
