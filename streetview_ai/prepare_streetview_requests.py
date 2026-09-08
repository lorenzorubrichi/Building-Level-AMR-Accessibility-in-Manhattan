"""Prepare Street View requests using a road-centerline camera point.

This script keeps the building address point as the visual target, but moves
the Street View query location onto the nearest road centerline segment. The
camera heading is then set from the road point toward the building address
point, which is much more robust than querying directly from the address point.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_CSV = DATA_DIR / "building_targets.csv"
OUTPUT_CSV = DATA_DIR / "streetview_requests.csv"
CENTERLINE_CSV = Path(r"C:\Users\loren\Desktop\RPI_DOC\lavori\dataset\rawdata\Centerline.csv")

WKT_POINT_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")
GRID_SIZE_DEG = 0.01
MAX_SEARCH_RINGS = 3
DEFAULT_PITCH = 0
DEFAULT_FOV = 80

BOROUGH_CODE_MAP = {
    "manhattan": "1",
    "bronx": "2",
    "brooklyn": "3",
    "queens": "4",
    "staten island": "5",
}


def parse_linestring_segments(wkt: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if not isinstance(wkt, str):
        return []
    coords = [(float(x), float(y)) for x, y in WKT_POINT_RE.findall(wkt)]
    if len(coords) < 2:
        return []
    return list(zip(coords[:-1], coords[1:]))


def cell_for(lon: float, lat: float) -> tuple[int, int]:
    return (math.floor(lon / GRID_SIZE_DEG), math.floor(lat / GRID_SIZE_DEG))


def nearest_point_on_segment(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> tuple[float, float, float]:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom == 0:
        qx, qy = ax, ay
    else:
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        qx = ax + t * abx
        qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return qx, qy, dx * dx + dy * dy


def bearing_degrees(from_lon: float, from_lat: float, to_lon: float, to_lat: float) -> float:
    dx = to_lon - from_lon
    dy = to_lat - from_lat
    angle = math.degrees(math.atan2(dx, dy))
    return (angle + 360.0) % 360.0


def load_centerline_index() -> dict[str, dict[tuple[int, int], list[tuple[float, float, float, float]]]]:
    if not CENTERLINE_CSV.exists():
        raise FileNotFoundError(f"Missing road centerline CSV: {CENTERLINE_CSV}")

    columns = ["the_geom", "Borough Code", "RW_TYPE", "NONPED", "STATUS"]
    roads = pd.read_csv(CENTERLINE_CSV, usecols=columns, dtype=str)

    # Prefer standard street centerlines; drop obvious non-road/pedestrian-only rows.
    roads = roads[roads["STATUS"] == "2"].copy()
    roads = roads[roads["RW_TYPE"].fillna("") == "1"].copy()
    roads = roads[roads["NONPED"].fillna("") != "V"].copy()

    index: dict[str, dict[tuple[int, int], list[tuple[float, float, float, float]]]] = {}
    grouped: dict[str, defaultdict[tuple[int, int], list[tuple[float, float, float, float]]]] = {
        code: defaultdict(list) for code in BOROUGH_CODE_MAP.values()
    }

    for _, row in roads.iterrows():
        borough_code = str(row["Borough Code"]).strip()
        if borough_code not in grouped:
            continue
        for (ax, ay), (bx, by) in parse_linestring_segments(row["the_geom"]):
            min_lon, max_lon = sorted((ax, bx))
            min_lat, max_lat = sorted((ay, by))
            x0, y0 = cell_for(min_lon, min_lat)
            x1, y1 = cell_for(max_lon, max_lat)
            for gx in range(x0, x1 + 1):
                for gy in range(y0, y1 + 1):
                    grouped[borough_code][(gx, gy)].append((ax, ay, bx, by))

    for code, grid in grouped.items():
        index[code] = dict(grid)
    return index


def find_nearest_road_point(
    lon: float,
    lat: float,
    borough_code: str,
    road_index: dict[str, dict[tuple[int, int], list[tuple[float, float, float, float]]]],
) -> tuple[float, float] | None:
    borough_grid = road_index.get(borough_code)
    if not borough_grid:
        return None

    cx, cy = cell_for(lon, lat)
    best: tuple[float, float, float] | None = None

    for ring in range(MAX_SEARCH_RINGS + 1):
        segments: list[tuple[float, float, float, float]] = []
        for gx in range(cx - ring, cx + ring + 1):
            for gy in range(cy - ring, cy + ring + 1):
                segments.extend(borough_grid.get((gx, gy), []))
        if not segments:
            continue

        for ax, ay, bx, by in segments:
            qx, qy, dist2 = nearest_point_on_segment(lon, lat, ax, ay, bx, by)
            if best is None or dist2 < best[2]:
                best = (qx, qy, dist2)

        if best is not None:
            return (best[0], best[1])

    return None


def main() -> None:
    if not INPUT_CSV.exists():
        print(f"Missing input file: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    required = ["bin", "address_lon", "address_lat"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if "borough" not in df.columns:
        raise ValueError("building_targets.csv must contain a borough column for road-centerline matching")

    road_index = load_centerline_index()

    out = df.copy()
    out["camera_lon"] = out["address_lon"]
    out["camera_lat"] = out["address_lat"]
    out["heading"] = None
    out["pitch"] = DEFAULT_PITCH
    out["fov"] = DEFAULT_FOV
    out["camera_source"] = "address_fallback"

    road_hits = 0

    for idx, row in out.iterrows():
        borough_name = str(row["borough"]).strip().lower()
        borough_code = BOROUGH_CODE_MAP.get(borough_name)
        if borough_code is None:
            continue

        road_point = find_nearest_road_point(
            float(row["address_lon"]),
            float(row["address_lat"]),
            borough_code,
            road_index,
        )
        if road_point is None:
            continue

        camera_lon, camera_lat = road_point
        out.at[idx, "camera_lon"] = camera_lon
        out.at[idx, "camera_lat"] = camera_lat
        out.at[idx, "heading"] = bearing_degrees(
            camera_lon,
            camera_lat,
            float(row["address_lon"]),
            float(row["address_lat"]),
        )
        out.at[idx, "camera_source"] = "road_centerline"
        road_hits += 1

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved Street View request scaffold: {OUTPUT_CSV}")
    print(f"Road-centerline camera points: {road_hits}")
    print(f"Address-point camera fallback: {len(out) - road_hits}")


if __name__ == "__main__":
    main()
