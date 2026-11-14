"""Select the buildings for which a Street View image is requested.

Writes AMR_PRIVATE_DIR/building_targets.csv (input of 04_prepare_streetview_requests.py).
The target point is the building's delivery point (address point, else footprint centroid).

The May-2026 acquisition used the eight neighbourhoods listed in DEFAULT_NTAS.
Images were then sent to the AI model in file order until the budget was exhausted,
which yielded 6,952 annotated buildings in seven of them (see the Data Descriptor).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402

DEFAULT_NTAS = [
    "Chelsea-Hudson Yards", "Gramercy", "Hell's Kitchen", "Midtown South-Flatiron-Union Square",
    "Midtown-Times Square", "Upper East Side-Carnegie Hill", "Upper West Side (Central)",
    "Upper West Side-Lincoln Square",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nta", nargs="*", default=DEFAULT_NTAS)
    args = ap.parse_args()
    b = pd.read_csv(config.DATA_DIR / "interim" / "building_attributes.csv", dtype={"bin": str})
    b = b[b["nta_name"].isin(args.nta)]
    out = pd.DataFrame({
        "bin": b["bin"], "borough": b["borough"], "neighborhood": b["nta_name"],
        "address_lon": b["delivery_lon"], "address_lat": b["delivery_lat"],
        "location_source": b["delivery_point_source"].map({"address_point": "address_point",
                                                           "centroid": "centroid_fallback"}),
    })
    config.PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.PRIVATE_DIR / "building_targets.csv", index=False)
    print(f"{len(out):,} targets in {out['neighborhood'].nunique()} neighbourhoods")


if __name__ == "__main__":
    main()
