"""Sensitivity analysis for hexagonal and aggregated-hex AMR zoning."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"
HEX_SCRIPT = SCRIPT_DIR / "build_amr_friendly_hex_map.py"

SENSITIVITY_CSV = OUTPUT_DIR / "amr_hex_zone_sensitivity.csv"
ETA_HEX_PIVOT_CSV = OUTPUT_DIR / "amr_hex_sensitivity_eta_hex_pivot.csv"
ETA_AGG_PIVOT_CSV = OUTPUT_DIR / "amr_hex_sensitivity_eta_aggregated_pivot.csv"
STD_HEX_PIVOT_CSV = OUTPUT_DIR / "amr_hex_sensitivity_within_std_hex_pivot.csv"
STD_AGG_PIVOT_CSV = OUTPUT_DIR / "amr_hex_sensitivity_within_std_aggregated_pivot.csv"
ETA_FIGURE = OUTPUT_DIR / "amr_hex_sensitivity_eta_squared.png"
STD_FIGURE = OUTPUT_DIR / "amr_hex_sensitivity_within_zone_std.png"
TRADEOFF_FIGURE = OUTPUT_DIR / "amr_hex_sensitivity_retention_and_zones.png"
REPORT_TXT = OUTPUT_DIR / "amr_hex_zone_sensitivity_report.txt"

HEX_RADII_M = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
MIN_BUILDINGS_VALUES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def load_hex_module():
    spec = importlib.util.spec_from_file_location("amr_hex_map", HEX_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {HEX_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adjusted_eta_squared(eta: float, n_buildings: int, n_zones: int) -> float:
    if n_zones <= 1 or n_buildings <= n_zones or np.isnan(eta):
        return np.nan
    return 1 - (1 - eta) * (n_buildings - 1) / (n_buildings - n_zones)


def summarize_groups(
    buildings: pd.DataFrame,
    group_col: str,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    rows = []
    for group_id, group in buildings.groupby(group_col):
        values = group["baseline_amr_index"].astype(float)
        rows.append(
            {
                "group_id": str(group_id),
                "n_buildings": int(len(group)),
                "mean_amr_index": float(values.mean()),
                "std_amr_index": float(values.std(ddof=0)),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary, {
            "n_zones": 0,
            "n_buildings": 0,
            "mean_buildings_per_zone": np.nan,
            "weighted_within_zone_std": np.nan,
            "eta_squared": np.nan,
            "adjusted_eta_squared": np.nan,
        }

    weights = summary["n_buildings"].astype(float)
    means = summary["mean_amr_index"].astype(float)
    standard_deviations = summary["std_amr_index"].astype(float)
    n_buildings = int(weights.sum())
    n_zones = int(len(summary))
    overall_mean = float(np.average(means, weights=weights))
    within_ss = float((weights * standard_deviations.pow(2)).sum())
    between_ss = float((weights * (means - overall_mean).pow(2)).sum())
    total_ss = within_ss + between_ss
    eta = between_ss / total_ss if total_ss else np.nan

    return summary, {
        "n_zones": n_zones,
        "n_buildings": n_buildings,
        "mean_buildings_per_zone": n_buildings / n_zones,
        "weighted_within_zone_std": float(np.average(standard_deviations, weights=weights)),
        "eta_squared": eta,
        "adjusted_eta_squared": adjusted_eta_squared(eta, n_buildings, n_zones),
    }


def aggregate_adjacent_cells(hex_module, buildings: pd.DataFrame, cell_summary: pd.DataFrame) -> pd.DataFrame:
    if cell_summary.empty:
        return buildings.iloc[0:0].assign(aggregated_zone_id=pd.Series(dtype=str))

    cell_classes = {
        row.group_id: hex_module.readiness_class(float(row.mean_amr_index))
        for row in cell_summary.itertuples(index=False)
    }
    valid_cells = set(cell_classes)
    visited: set[str] = set()
    cell_to_zone: dict[str, str] = {}
    class_counts = {"low": 0, "medium": 0, "high": 0}

    for cell_id in sorted(valid_cells):
        if cell_id in visited:
            continue
        readiness_class = cell_classes[cell_id]
        class_counts[readiness_class] += 1
        zone_id = f"{readiness_class}_{class_counts[readiness_class]:03d}"
        stack = [cell_id]
        visited.add(cell_id)
        while stack:
            current = stack.pop()
            cell_to_zone[current] = zone_id
            for neighbor in hex_module.hex_neighbors(current):
                if (
                    neighbor not in visited
                    and neighbor in valid_cells
                    and cell_classes[neighbor] == readiness_class
                ):
                    visited.add(neighbor)
                    stack.append(neighbor)

    work = buildings[buildings["hex_id"].isin(valid_cells)].copy()
    work["aggregated_zone_id"] = work["hex_id"].map(cell_to_zone)
    return work


def run_sensitivity() -> pd.DataFrame:
    hex_module = load_hex_module()
    buildings = hex_module.read_buildings()
    buildings, _ = hex_module.add_local_xy(buildings)
    rows = []

    for radius in HEX_RADII_M:
        hex_module.HEX_RADIUS_M = radius
        centers = hex_module.build_hex_centers(buildings)
        assigned = hex_module.assign_to_nearest_hex(buildings, centers)
        counts = assigned["hex_id"].value_counts()

        for min_buildings in MIN_BUILDINGS_VALUES:
            valid_cells = set(counts[counts >= min_buildings].index.astype(str))
            retained = assigned[assigned["hex_id"].isin(valid_cells)].copy()
            cell_summary, cell_metrics = summarize_groups(retained, "hex_id")
            aggregated = aggregate_adjacent_cells(hex_module, retained, cell_summary)
            _, aggregated_metrics = summarize_groups(aggregated, "aggregated_zone_id")

            rows.append(
                {
                    "hex_radius_m": radius,
                    "min_buildings_per_hex": min_buildings,
                    "share_buildings_retained": len(retained) / len(buildings),
                    "n_buildings_retained": len(retained),
                    "n_hexagons": cell_metrics["n_zones"],
                    "mean_buildings_per_hexagon": cell_metrics["mean_buildings_per_zone"],
                    "hex_within_zone_std": cell_metrics["weighted_within_zone_std"],
                    "hex_eta_squared": cell_metrics["eta_squared"],
                    "hex_adjusted_eta_squared": cell_metrics["adjusted_eta_squared"],
                    "n_aggregated_zones": aggregated_metrics["n_zones"],
                    "mean_buildings_per_aggregated_zone": aggregated_metrics["mean_buildings_per_zone"],
                    "aggregated_within_zone_std": aggregated_metrics["weighted_within_zone_std"],
                    "aggregated_eta_squared": aggregated_metrics["eta_squared"],
                    "aggregated_adjusted_eta_squared": aggregated_metrics["adjusted_eta_squared"],
                }
            )

    return pd.DataFrame(rows)


def save_pivots(results: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pivot_specs = {
        "eta_hex": ("hex_eta_squared", ETA_HEX_PIVOT_CSV),
        "eta_aggregated": ("aggregated_eta_squared", ETA_AGG_PIVOT_CSV),
        "std_hex": ("hex_within_zone_std", STD_HEX_PIVOT_CSV),
        "std_aggregated": ("aggregated_within_zone_std", STD_AGG_PIVOT_CSV),
    }
    pivots = {}
    for name, (value_col, output_path) in pivot_specs.items():
        pivot = results.pivot(
            index="hex_radius_m",
            columns="min_buildings_per_hex",
            values=value_col,
        ).sort_index()
        pivot.to_csv(output_path)
        pivots[name] = pivot
    return pivots


def draw_heatmap(ax, pivot: pd.DataFrame, title: str, decimals: int, cmap: str) -> None:
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Minimum buildings per hexagon")
    ax.set_ylabel("Hexagon radius (m)")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(v) for v in pivot.index])
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.{decimals}f}", ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def save_figures(results: pd.DataFrame, pivots: dict[str, pd.DataFrame]) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#555555"})

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    draw_heatmap(axes[0], pivots["eta_hex"], "Individual hexagons: eta squared", 2, "YlGn")
    draw_heatmap(axes[1], pivots["eta_aggregated"], "Aggregated zones: eta squared", 2, "YlGn")
    figure.suptitle("Sensitivity of explained AMR Index variation", fontsize=14, fontweight="bold")
    figure.savefig(ETA_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    draw_heatmap(axes[0], pivots["std_hex"], "Individual hexagons: within-zone std", 1, "YlOrRd")
    draw_heatmap(axes[1], pivots["std_aggregated"], "Aggregated zones: within-zone std", 1, "YlOrRd")
    figure.suptitle("Sensitivity of within-zone AMR Index variation (lower is better)", fontsize=14, fontweight="bold")
    figure.savefig(STD_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(figure)

    retention = results.pivot(
        index="hex_radius_m", columns="min_buildings_per_hex", values="share_buildings_retained"
    ).sort_index()
    zones = results.pivot(
        index="hex_radius_m", columns="min_buildings_per_hex", values="n_aggregated_zones"
    ).sort_index()
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    draw_heatmap(axes[0], retention * 100, "Buildings retained (%)", 0, "Blues")
    draw_heatmap(axes[1], zones, "Number of aggregated AMR zones", 0, "Purples")
    figure.suptitle("Coverage and spatial fragmentation trade-off", fontsize=14, fontweight="bold")
    figure.savefig(TRADEOFF_FIGURE, dpi=220, bbox_inches="tight")
    plt.close(figure)


def save_report(results: pd.DataFrame) -> None:
    eligible = results[
        (results["share_buildings_retained"] >= 0.90)
        & (results["n_aggregated_zones"] >= 5)
    ].copy()
    best_eta = eligible.sort_values("aggregated_adjusted_eta_squared", ascending=False).head(5)
    best_std = eligible.sort_values("aggregated_within_zone_std").head(5)

    lines = [
        "Hexagonal AMR Zoning Sensitivity Analysis",
        "",
        f"Scenarios evaluated: {len(results)}",
        f"Hexagon radii: {', '.join(map(str, HEX_RADII_M))} m",
        f"Minimum-building thresholds: {', '.join(map(str, MIN_BUILDINGS_VALUES))}",
        "",
        "Interpretation:",
        "- Higher eta squared indicates that the grouping represents more AMR Index variation.",
        "- Lower within-zone standard deviation indicates more internally homogeneous zones.",
        "- More zones can mechanically improve both metrics, so coverage and fragmentation must also be considered.",
        "- Adjusted eta squared partially penalizes excessive numbers of zones.",
        "",
        "Top eligible scenarios by adjusted eta squared (coverage >= 90%, zones >= 5):",
        best_eta.to_string(index=False),
        "",
        "Top eligible scenarios by lowest within-zone std (coverage >= 90%, zones >= 5):",
        best_std.to_string(index=False),
    ]
    REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_sensitivity()
    results.to_csv(SENSITIVITY_CSV, index=False)
    pivots = save_pivots(results)
    save_figures(results, pivots)
    save_report(results)
    print(f"Saved sensitivity results: {SENSITIVITY_CSV}")
    print(f"Saved eta-squared figure: {ETA_FIGURE}")
    print(f"Saved within-zone figure: {STD_FIGURE}")
    print(f"Saved trade-off figure: {TRADEOFF_FIGURE}")
    print(f"Saved report: {REPORT_TXT}")


if __name__ == "__main__":
    main()
