"""Sensitivity analysis for AMR-friendly square-grid zoning.

The number and quality of AMR-friendly zones depends on two main parameters:

1. Grid cell size, in meters.
2. Minimum number of buildings required to keep a cell.

This script varies both parameters and saves a comparison table. It does not
create maps; it only produces metrics useful for choosing fairer zoning
settings and for comparing AMR-driven zones with ZIP/neighborhood boundaries.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_DIR = SCRIPT_DIR / "output"

BUILDING_INDEX_CSV = PROJECT_ROOT / "AMR_index_no_AI" / "output" / "building_level_no_ai_amr_index.csv"

SENSITIVITY_CSV = OUTPUT_DIR / "amr_grid_zone_sensitivity.csv"
SENSITIVITY_PIVOT_ETA_CSV = OUTPUT_DIR / "amr_grid_zone_sensitivity_eta_pivot.csv"
SENSITIVITY_PIVOT_ZONES_CSV = OUTPUT_DIR / "amr_grid_zone_sensitivity_nzones_pivot.csv"
REPORT_TXT = OUTPUT_DIR / "amr_grid_zone_sensitivity_report.txt"

# Edit these lists to test more/fewer scenarios.
GRID_CELL_SIZES_M = [50, 100, 200, 300, 400, 500, 600, 800, 1000]
MIN_BUILDINGS_PER_CELL_VALUES = [5, 10, 20, 30, 50, 75, 100]

LOW_THRESHOLD = 40
HIGH_THRESHOLD = 60


def read_buildings() -> pd.DataFrame:
    if not BUILDING_INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing building AMR Index file: {BUILDING_INDEX_CSV}")
    df = pd.read_csv(BUILDING_INDEX_CSV, low_memory=False)
    for col in ["centroid_lon", "centroid_lat", "baseline_amr_index", "target_amr_accessible"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["centroid_lon", "centroid_lat", "baseline_amr_index", "target_amr_accessible"]).copy()
    df["zipcode"] = df["zipcode"].fillna("Unknown").astype(str)
    df["neighborhood"] = df["neighborhood"].fillna("Unknown").astype(str)
    return df


def add_local_xy(df: pd.DataFrame) -> pd.DataFrame:
    lat0 = math.radians(float(df["centroid_lat"].mean()))
    lon0 = float(df["centroid_lon"].mean())
    base_lat = float(df["centroid_lat"].mean())
    meters_per_deg_lon = 111_320 * math.cos(lat0)
    meters_per_deg_lat = 110_540
    out = df.copy()
    out["x_m"] = (out["centroid_lon"] - lon0) * meters_per_deg_lon
    out["y_m"] = (out["centroid_lat"] - base_lat) * meters_per_deg_lat
    return out


def readiness_class(mean_index: float) -> str:
    if mean_index < LOW_THRESHOLD:
        return "low"
    if mean_index < HIGH_THRESHOLD:
        return "medium"
    return "high"


def connected_components(cells: pd.DataFrame) -> list[list[tuple[int, int]]]:
    cell_class = {
        (int(row.grid_i), int(row.grid_j)): str(row.readiness_class)
        for _, row in cells.iterrows()
    }
    visited: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for cell, label in cell_class.items():
        if cell in visited:
            continue
        queue = deque([cell])
        visited.add(cell)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            i, j = current
            for neighbor in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if neighbor in visited:
                    continue
                if cell_class.get(neighbor) != label:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append(component)
    return components


def adjusted_eta_squared(eta_squared: float, n_buildings: int, n_groups: int) -> float:
    """Adjusted explained variance, penalized for number of groups."""
    if n_buildings <= n_groups or n_groups <= 1 or np.isnan(eta_squared):
        return np.nan
    return 1 - (1 - eta_squared) * (n_buildings - 1) / (n_buildings - n_groups)


def evaluate_partition(buildings: pd.DataFrame, group_col: str, min_buildings: int = 1) -> dict[str, float | int]:
    work = buildings.dropna(subset=[group_col, "baseline_amr_index"]).copy()
    counts = work[group_col].value_counts()
    valid_groups = counts[counts >= min_buildings].index
    work = work[work[group_col].isin(valid_groups)].copy()
    if work.empty:
        return {
            "n_groups": 0,
            "n_buildings": 0,
            "weighted_within_zone_std": np.nan,
            "between_group_std_of_means": np.nan,
            "eta_squared_explained_variance": np.nan,
            "adjusted_eta_squared": np.nan,
            "mean_abs_actual_predicted_gap": np.nan,
            "min_buildings_per_zone": np.nan,
            "median_buildings_per_zone": np.nan,
            "mean_buildings_per_zone": np.nan,
        }

    overall = float(work["baseline_amr_index"].mean())
    total_ss = float(((work["baseline_amr_index"] - overall) ** 2).sum())
    within_ss = 0.0
    weighted_std_parts = []
    group_means = []
    gaps = []
    group_sizes = []

    for _, group in work.groupby(group_col):
        mean_index = float(group["baseline_amr_index"].mean())
        within_ss += float(((group["baseline_amr_index"] - mean_index) ** 2).sum())
        weighted_std_parts.append(float(group["baseline_amr_index"].std(ddof=0)) * len(group))
        group_means.append(mean_index)
        gaps.append(abs(float(group["target_amr_accessible"].mean()) - mean_index / 100))
        group_sizes.append(len(group))

    eta = float(1 - within_ss / total_ss) if total_ss else np.nan
    n_groups = int(work[group_col].nunique())
    n_buildings = int(len(work))
    return {
        "n_groups": n_groups,
        "n_buildings": n_buildings,
        "weighted_within_zone_std": float(sum(weighted_std_parts) / n_buildings),
        "between_group_std_of_means": float(np.std(group_means, ddof=0)),
        "eta_squared_explained_variance": eta,
        "adjusted_eta_squared": adjusted_eta_squared(eta, n_buildings, n_groups),
        "mean_abs_actual_predicted_gap": float(np.mean(gaps)),
        "min_buildings_per_zone": int(min(group_sizes)),
        "median_buildings_per_zone": float(np.median(group_sizes)),
        "mean_buildings_per_zone": float(np.mean(group_sizes)),
    }


def build_grid_zones(df_xy: pd.DataFrame, cell_size_m: int, min_buildings_per_cell: int) -> dict[str, float | int | str]:
    work = df_xy.copy()
    min_x = math.floor(work["x_m"].min() / cell_size_m) * cell_size_m
    min_y = math.floor(work["y_m"].min() / cell_size_m) * cell_size_m
    work["grid_i"] = np.floor((work["x_m"] - min_x) / cell_size_m).astype(int)
    work["grid_j"] = np.floor((work["y_m"] - min_y) / cell_size_m).astype(int)
    work["grid_id"] = work["grid_i"].astype(str) + "_" + work["grid_j"].astype(str)

    cell_rows = []
    for (i, j), group in work.groupby(["grid_i", "grid_j"]):
        if len(group) < min_buildings_per_cell:
            continue
        mean_index = float(group["baseline_amr_index"].mean())
        cell_rows.append(
            {
                "grid_i": int(i),
                "grid_j": int(j),
                "grid_id": f"{int(i)}_{int(j)}",
                "n_buildings": int(len(group)),
                "mean_amr_index": mean_index,
                "readiness_class": readiness_class(mean_index),
            }
        )

    cells = pd.DataFrame(cell_rows)
    if cells.empty:
        return {
            "grid_cell_size_m": cell_size_m,
            "min_buildings_per_cell": min_buildings_per_cell,
            "n_retained_cells": 0,
            "n_low_cells": 0,
            "n_medium_cells": 0,
            "n_high_cells": 0,
            "n_groups": 0,
            "n_buildings": 0,
            "share_buildings_retained": 0.0,
            "weighted_within_zone_std": np.nan,
            "between_group_std_of_means": np.nan,
            "eta_squared_explained_variance": np.nan,
            "adjusted_eta_squared": np.nan,
            "mean_abs_actual_predicted_gap": np.nan,
            "min_buildings_per_zone": np.nan,
            "median_buildings_per_zone": np.nan,
            "mean_buildings_per_zone": np.nan,
        }

    components = connected_components(cells)
    cell_to_zone = {}
    zone_classes = {}
    for zone_number, component in enumerate(components, start=1):
        label = str(
            cells[
                (cells["grid_i"] == component[0][0])
                & (cells["grid_j"] == component[0][1])
            ]["readiness_class"].iloc[0]
        )
        zone_id = f"SENS_{zone_number:03d}_{label.upper()}"
        zone_classes[zone_id] = label
        for i, j in component:
            cell_to_zone[f"{i}_{j}"] = zone_id

    valid_grid_ids = set(cell_to_zone)
    metric_buildings = work[work["grid_id"].isin(valid_grid_ids)].copy()
    metric_buildings["amr_grid_zone"] = metric_buildings["grid_id"].map(cell_to_zone)
    metrics = evaluate_partition(metric_buildings, "amr_grid_zone")

    class_counts_cells = cells["readiness_class"].value_counts()
    class_counts_zones = pd.Series(zone_classes).value_counts()
    return {
        "grid_cell_size_m": cell_size_m,
        "min_buildings_per_cell": min_buildings_per_cell,
        "n_retained_cells": int(len(cells)),
        "n_low_cells": int(class_counts_cells.get("low", 0)),
        "n_medium_cells": int(class_counts_cells.get("medium", 0)),
        "n_high_cells": int(class_counts_cells.get("high", 0)),
        "n_low_zones": int(class_counts_zones.get("low", 0)),
        "n_medium_zones": int(class_counts_zones.get("medium", 0)),
        "n_high_zones": int(class_counts_zones.get("high", 0)),
        "share_buildings_retained": float(metrics["n_buildings"] / len(df_xy)),
        **metrics,
    }


def write_report(results: pd.DataFrame, baseline: pd.DataFrame) -> None:
    best_adjusted = results.sort_values("adjusted_eta_squared", ascending=False).head(10)
    best_eta = results.sort_values("eta_squared_explained_variance", ascending=False).head(10)
    lines = [
        "AMR grid zone sensitivity analysis",
        "==================================",
        "",
        "The analysis varies:",
        f"- grid cell size: {GRID_CELL_SIZES_M}",
        f"- minimum buildings per cell: {MIN_BUILDINGS_PER_CELL_VALUES}",
        "",
        "Interpretation:",
        "- Smaller cells and lower building thresholds usually create more zones.",
        "- More zones can mechanically improve eta squared.",
        "- Adjusted eta squared penalizes excessive fragmentation.",
        "- Weighted within-zone std should be lower when zones are more internally homogeneous.",
        "",
        "Existing-boundary baselines:",
        baseline.to_string(index=False),
        "",
        "Top scenarios by adjusted eta squared:",
        best_adjusted[
            [
                "grid_cell_size_m",
                "min_buildings_per_cell",
                "n_groups",
                "n_retained_cells",
                "n_buildings",
                "share_buildings_retained",
                "weighted_within_zone_std",
                "eta_squared_explained_variance",
                "adjusted_eta_squared",
            ]
        ].to_string(index=False),
        "",
        "Top scenarios by raw eta squared:",
        best_eta[
            [
                "grid_cell_size_m",
                "min_buildings_per_cell",
                "n_groups",
                "n_retained_cells",
                "n_buildings",
                "share_buildings_retained",
                "weighted_within_zone_std",
                "eta_squared_explained_variance",
                "adjusted_eta_squared",
            ]
        ].to_string(index=False),
    ]
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = add_local_xy(read_buildings())

    baseline_rows = []
    baseline_rows.append({"partition": "neighborhood", **evaluate_partition(buildings, "neighborhood", min_buildings=20)})
    known_zip = buildings[buildings["zipcode"].astype(str).str.lower() != "unknown"].copy()
    baseline_rows.append({"partition": "zipcode", **evaluate_partition(known_zip, "zipcode", min_buildings=20)})
    baseline = pd.DataFrame(baseline_rows)

    rows = []
    total = len(GRID_CELL_SIZES_M) * len(MIN_BUILDINGS_PER_CELL_VALUES)
    run_number = 0
    for cell_size_m in GRID_CELL_SIZES_M:
        for min_buildings in MIN_BUILDINGS_PER_CELL_VALUES:
            run_number += 1
            print(f"[{run_number}/{total}] cell={cell_size_m}m min_buildings={min_buildings}")
            rows.append(build_grid_zones(buildings, cell_size_m, min_buildings))

    results = pd.DataFrame(rows).sort_values(
        ["grid_cell_size_m", "min_buildings_per_cell"]
    )
    results.to_csv(SENSITIVITY_CSV, index=False)

    eta_pivot = results.pivot(
        index="grid_cell_size_m",
        columns="min_buildings_per_cell",
        values="eta_squared_explained_variance",
    )
    eta_pivot.to_csv(SENSITIVITY_PIVOT_ETA_CSV)

    zones_pivot = results.pivot(
        index="grid_cell_size_m",
        columns="min_buildings_per_cell",
        values="n_groups",
    )
    zones_pivot.to_csv(SENSITIVITY_PIVOT_ZONES_CSV)
    write_report(results, baseline)

    print(f"Saved sensitivity table: {SENSITIVITY_CSV}")
    print(f"Saved eta pivot: {SENSITIVITY_PIVOT_ETA_CSV}")
    print(f"Saved zones pivot: {SENSITIVITY_PIVOT_ZONES_CSV}")
    print(f"Saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
