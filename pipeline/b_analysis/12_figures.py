"""Figures for the Data Descriptor (PDF + PNG in results/figures/)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.geo import load_polygons  # noqa: E402

OUT = config.RESULTS_DIR / "figures"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8c8b86", "#e4e3df", "#fcfcfb"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": ["cmss10", "DejaVu Sans"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 300, "legend.frameon": False,
})


from matplotlib.ticker import FuncFormatter  # noqa: E402

THOUSANDS = FuncFormatter(lambda v, _: f"{v:,.0f}")   # journal style: 1,000


def panel(ax, letter: str, title: str = "") -> None:
    """Lower-case bold panel label (journal style), optional short title."""
    ax.set_title("")
    ax.text(0, 1.04, letter, transform=ax.transAxes, fontsize=9, fontfamily="DejaVu Sans", fontweight="bold", color=INK, va="bottom")
    if title:
        ax.text(0.07, 1.04, title, transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")


def scale_bar(ax, lon0: float, lat0: float, km: float = 1.0) -> None:
    """Scale bar with its length written on the bar (journal style)."""
    dlon = km * 1000 / (111_320 * np.cos(np.radians(lat0)))
    ax.plot([lon0, lon0 + dlon], [lat0, lat0], color=INK, lw=1.4, solid_capstyle="butt", zorder=5)
    ax.text(lon0 + dlon / 2, lat0 + 0.0012, f"{km:g} km", ha="center", va="bottom", fontsize=6.5, color=INK, zorder=5)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)


def nta_outlines(ax, polys, names=None, lw=0.3, color=MUTED):
    for name, _, paths, _, props in polys:
        if props.get("boroname") != "Manhattan" and props.get("ntaname") != "Kingsbridge-Marble Hill":
            continue
        for p in paths:
            v = p.vertices
            ax.plot(v[:, 0], v[:, 1], color=color, lw=lw if not (names and name in names) else 0.9, zorder=1)


def map_axes(ax, extent):
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.radians(40.78)))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def fig_workflow():
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.set_xlim(-1.5, 105); ax.set_ylim(0, 38); ax.axis("off")
    W, H = 18, 14
    boxes = {
        "src": (0, 22, "NYC Open Data:\nBuilding Footprints,\nPLUTO, Address Points,\nSidewalks, NTAs", "#eef4fc"),
        "bld": (21.5, 22, "Building attribute\ntable, 44,319\nbuildings", "#eef4fc"),
        "mod": (64.5, 22, "Logistic model on\nnon-visual attributes\n(two label versions)", "#eef4fc"),
        "pred": (86, 22, "Probability, 90%\ninterval, domain\nflags for every\nbuilding", "#eef4fc"),
        "sv": (0, 2, "Street View images\n6,952 buildings,\n7 neighbourhoods", "#fdf0ea"),
        "ai": (21.5, 2, "Multimodal model:\nusable image, stairs,\ngate, ramp", "#fdf0ea"),
        "rev": (43, 2, "Human review of\nevery AI-reported\ngate or ramp\n(+300 controls)", "#fdf0ea"),
        "rule": (64.5, 2, "Rule: accessible =\nno gate AND\n(no stairs OR ramp)", "#f3f3f1"),
    }
    for x, y, text, fc in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=MUTED, lw=0.6))
        ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=6, color=INK, linespacing=1.25)

    def arrow(a, b, ya=None, yb=None):
        xa, ya0 = boxes[a][0] + W + 0.6, boxes[a][1] + H / 2
        xb, yb0 = boxes[b][0] - 0.6, boxes[b][1] + H / 2
        ax.annotate("", xy=(xb, yb if yb is not None else yb0), xytext=(xa, ya if ya is not None else ya0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    arrow("src", "bld"); arrow("sv", "ai"); arrow("ai", "rev"); arrow("rev", "rule")
    arrow("bld", "mod"); arrow("mod", "pred")
    ax.annotate("", xy=(73.5, 21.4), xytext=(73.5, 16.6),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.7, shrinkA=0, shrinkB=0))
    ax.text(75, 19, "labels (with review; model only)", fontsize=6, color=MUTED, va="center")
    save(fig, "fig1_workflow")


ACC_GREEN, NOT_ACC = "#009E73", "#D55E00"   # Okabe-Ito green and vermilion: distinguishable with colour blindness


def basemap(ax, polys, extent):
    """Background map drawn from the deposited NYC Open Data snapshot (no third-party tiles)."""
    from matplotlib.collections import LineCollection, PolyCollection
    from amr_access.geo import parse_multilinestring_wkt, parse_multipolygon_wkt
    x0, x1, y0, y1 = extent
    ax.set_facecolor("#dbe7f1")                                  # water
    land = []
    for _, _, paths, _, _ in polys:                              # every NYC neighbourhood polygon = land
        land += [p.vertices for p in paths]
    ax.add_collection(PolyCollection(land, facecolors="#f4f3ef", edgecolors="none", zorder=0))
    snap = config.ROOT / "data" / "source_snapshot"

    def inside(arr):
        return (arr[:, 0].max() > x0) & (arr[:, 0].min() < x1) & (arr[:, 1].max() > y0) & (arr[:, 1].min() < y1)
    streets = []
    for w in pd.read_csv(snap / "Centerline.csv.gz", usecols=["the_geom"])["the_geom"]:
        streets += [ln for ln in parse_multilinestring_wkt(w) if len(ln) and inside(ln)]
    ax.add_collection(LineCollection(streets, colors="white", linewidths=0.9, zorder=0.5))
    fp = []
    for w in pd.read_csv(snap / "BUILDING.csv.gz", usecols=["the_geom"])["the_geom"]:
        for poly in parse_multipolygon_wkt(w):
            ring = np.column_stack(poly[0]) if isinstance(poly[0], tuple) else np.asarray(poly[0])
            if ring.ndim == 2 and len(ring) > 2 and inside(ring):
                fp.append(ring)
    ax.add_collection(PolyCollection(fp, facecolors="#dcdad3", edgecolors="none", zorder=0.6))


def fig_label_map(lab, polys):
    u = lab[lab["image_usable"]]
    extent = (-74.015, -73.955, 40.728, 40.80)
    fig, ax = plt.subplots(figsize=(4.9, 7.4))
    basemap(ax, polys, extent)
    nta_outlines(ax, polys, names=set(u["nta_name"].dropna()), color="#8a8a86")
    for val, color, name in [(0, NOT_ACC, "Not accessible"), (1, ACC_GREEN, "Accessible")]:
        d = u[u["accessible_rule"] == val]
        ax.scatter(d["target_lon"], d["target_lat"], s=4, c=color, lw=0, label=f"{name} (n={len(d):,})", zorder=3)
    nj = u[u["accessible_rule"].isna()]
    ax.scatter(nj["target_lon"], nj["target_lat"], s=4, facecolors="none", edgecolors="#6f6e69", lw=0.4,
               label=f"Not judgeable in review (n={len(nj):,})", zorder=3)
    nu = lab[~lab["image_usable"]]
    ax.scatter(nu["target_lon"], nu["target_lat"], s=8, c=INK, marker="x", lw=0.5,
               label=f"Image not usable (n={len(nu):,})", zorder=3)
    map_axes(ax, extent)
    scale_bar(ax, -73.968, 40.731)
    ax.legend(loc="upper left", markerscale=2.2, fontsize=7, handletextpad=0.3, framealpha=0.95, frameon=True,
              edgecolor="none")
    save(fig, "fig2_label_map")


def fig_combinations():
    """Feature combinations: AI only (left) and after the targeted human review (right)."""
    st = config.RESULTS_DIR / "stats"
    keys = ["stairs_present", "gate_present", "ramp_present"]
    ai = pd.read_csv(st / "label_combinations_ai.csv")
    fin = pd.read_csv(st / "label_combinations.csv")
    c = ai[keys + ["accessible_rule"]].merge(fin[keys + ["accessible_rule"]], on=keys + ["accessible_rule"], how="outer")
    c = c.merge(ai[keys + ["n", "share"]], on=keys, how="left").merge(
        fin[keys + ["n", "share"]], on=keys, how="left", suffixes=("_ai", "_final")).fillna({"n_ai": 0, "n_final": 0,
                                                                                         "share_ai": 0, "share_final": 0})
    c = c.sort_values(["accessible_rule", "n_ai"], ascending=[True, True]).reset_index(drop=True)
    yes = lambda v: "yes" if v else "no"  # noqa: E731
    names = [f"stairs {yes(s)} · gate {yes(g)} · ramp {yes(r)}"
             for s, g, r in zip(c["stairs_present"], c["gate_present"], c["ramp_present"])]
    colors = [ACC_GREEN if a else NOT_ACC for a in c["accessible_rule"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    xmax = max(c["n_ai"].max(), c["n_final"].max()) * 1.35
    for ax, col, title, letter in [(axes[0], "ai", "Model only", "a"),
                                    (axes[1], "final", "With human review of gates and ramps", "b")]:
        ax.barh(names, c[f"n_{col}"], color=colors, height=0.7, edgecolor="white", linewidth=1)
        for y, (n, sh) in enumerate(zip(c[f"n_{col}"], c[f"share_{col}"])):
            ax.text(n + 20, y, f"{int(n):,} ({sh:.1%})", va="center", fontsize=6, color=INK2)
        panel(ax, letter, title)
        ax.set_xlabel("Buildings (labelled images)")
        ax.xaxis.set_major_formatter(THOUSANDS)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.set_xlim(0, xmax)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=ACC_GREEN, label="Accessible (rule)"), Patch(color=NOT_ACC, label="Not accessible (rule)")],
                   loc="lower right", fontsize=6.5)
    save(fig, "fig3_label_combinations")


def fig_calibration_and_cv():
    m = json.loads((config.RESULTS_DIR / "model" / "metrics.json").read_text())
    cal = pd.read_csv(config.RESULTS_DIR / "model" / "calibration_oof.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.35]})
    a1.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls="--", label="Perfect calibration")
    z = 1.96; k = cal["observed"] * cal["n"]; den = cal["n"] + z**2   # Wilson 95% interval of the observed share
    mid = (k + z**2 / 2) / den; half = z * np.sqrt(k * (cal["n"] - k) / cal["n"] + z**2 / 4) / den
    a1.errorbar(cal["mean_predicted"], cal["observed"], yerr=[cal["observed"] - (mid - half), (mid + half) - cal["observed"]],
                color=C1, lw=1.5, marker="o", ms=4, capsize=2, elinewidth=0.8, label="Out-of-fold, primary index (95% CI)")
    a1.set_xlabel("Mean predicted probability"); a1.set_ylabel("Observed share accessible")
    a1.set_xlim(0, 1); a1.set_ylim(0, 1); a1.grid(True, color=GRID, lw=0.5); a1.legend(fontsize=6.5, loc="lower right")
    panel(a1, "a")
    rows = [("Random 5-fold", "cv_random"), ("500 m spatial blocks", "cv_spatial_block_500m"),
            ("Leave-one-ZIP-out", "cv_leave_one_zip_out"),
            ("Leave-one-neighbourhood-out", "cv_leave_one_neighbourhood_out")]
    y = np.arange(len(rows))[::-1]
    prim = [m[k]["roc_auc"] for _, k in rows]
    aio = [m["ai_only"][k]["roc_auc"] for _, k in rows]
    for yi, a, b in zip(y, prim, aio):
        a2.plot([b, a], [yi, yi], color=GRID, lw=2, zorder=1)
    a2.scatter(prim, y, color=C1, s=28, zorder=3, label="Primary index (with review)")
    a2.scatter(aio, y, color=C2, marker="s", s=24, zorder=3, label="Model-only index")
    for yi, v in zip(y, prim):
        a2.text(v + 0.008, yi, f"{v:.3f}", va="center", fontsize=6.5, color=INK2)
    a2.set_yticks(y, [r[0] for r in rows]); a2.set_xlim(0.5, 1.0); a2.set_xlabel("ROC AUC (0.5 = chance)")
    a2.xaxis.grid(True, color=GRID, lw=0.5); a2.set_axisbelow(True)
    a2.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.9))
    panel(a2, "b")
    fig.tight_layout()
    save(fig, "fig4_model_validation")


def fig_prediction_maps(b, pred, polys):
    p = pred.merge(b[["bin", "centroid_lon", "centroid_lat"]], on="bin")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 5.4))
    for ax in (a1, a2):
        nta_outlines(ax, polys, lw=0.25)
    sc = a1.scatter(p["centroid_lon"], p["centroid_lat"], c=p["amr_index"], cmap=SEQ, vmin=0, vmax=100, s=0.6, lw=0)
    cb = fig.colorbar(sc, ax=a1, shrink=0.5, pad=0.01); cb.set_label("Predicted AMR accessibility index (0–100)")
    cb.outline.set_visible(False)
    panel(a1, "a")
    cat = np.where(p["in_domain"], "in", np.where(p["categories_seen"] & p["numeric_in_range"], "extrap", "out"))
    for key, color, label in [("out", C2, "Covariates outside training range"),
                              ("extrap", C3, "Outside labelled neighbourhoods,\ncovariates within range"),
                              ("in", C1, "Within labelled neighbourhoods\nand covariate range")]:
        d = p[cat == key]
        a2.scatter(d["centroid_lon"], d["centroid_lat"], c=color, s=0.6, lw=0, label=f"{label} ({len(d) / len(p):.0%})")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), markerscale=6, fontsize=6.2, ncol=1)
    panel(a2, "b")
    for ax in (a1, a2):
        map_axes(ax, (-74.03, -73.90, 40.695, 40.885))
    save(fig, "fig5_prediction_maps")


def fig_capture_years():
    y = pd.read_csv(config.RESULTS_DIR / "stats" / "capture_years.csv")
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    ax.bar(y["year"].astype(int), y["n"], color=C1, width=0.75, edgecolor="white")
    ax.set_xlabel("Street View capture year"); ax.set_ylabel("Images")
    ax.set_ylim(bottom=0); ax.yaxis.set_major_formatter(THOUSANDS)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig6_capture_years")


def fig_data_layers_simple():
    """High-level version for Background & Summary: the three linked layers of the dataset."""
    st = json.loads((config.RESULTS_DIR / "stats" / "stats.json").read_text())
    f = lambda v: f"{int(v):,}"  # noqa: E731
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 140); ax.set_ylim(0, 62); ax.axis("off")

    def box(x0, y0, x1, y1, fc, lw=0.9):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=3.5",
                                    fc=fc, ec=INK, lw=lw, zorder=1))

    box(1, 1, 139, 61, "white", lw=1.1)
    ax.text(70, 59, f"(i) Building attribute table: {f(st['n_buildings'])} buildings in Manhattan",
            ha="center", va="top", fontsize=7.6, fontweight="bold", color=INK)
    ax.text(70, 54.6, "NYC Open Data: footprints, tax lots, address points, sidewalks, neighbourhoods",
            ha="center", va="top", fontsize=6.4, color=INK2)
    ax.text(70, 47.2, f"(iii) Modelled accessibility probability for all {f(st['n_buildings'])} buildings",
            ha="center", va="center", fontsize=7, fontweight="bold", color=C2)
    ax.text(70, 43.4, "uncertainty intervals and extrapolation flags; two versions (with review, model only)",
            ha="center", va="center", fontsize=6.2, color=C2)
    ax.text(3.5, 36, "Non-visual attributes", fontsize=6.8, fontweight="bold", color=INK, va="top")
    for i, t in enumerate(["Size, height, floors", "Year built, alterations", "Land use, building class",
                           "Distance to sidewalk, street", "Historic district, landmark"]):
        ax.text(4.2, 31 - i * 4.6, "\u2022 " + t, fontsize=6.1, color=INK2, va="top")

    n_img_ntas = int(pd.read_csv(config.BUILDINGS_CSV, usecols=["nta2020", "streetview_image_acquired"],
                                 dtype={"nta2020": str}).query("streetview_image_acquired")["nta2020"].nunique())
    box(38, 3, 124, 40.5, "#f6f6f4")
    ax.text(81, 39.3, f"Street View images: {f(st['n_images_acquired'])} buildings, "
                      f"{n_img_ntas} neighbourhoods", ha="center", va="top", fontsize=6.6, fontweight="bold",
            color=INK)
    box(41, 5, 121, 35, "#eef4fc")
    ax.text(81, 33.8, f"(ii) Entrance annotations: {f(st['n_annotated'])} buildings", ha="center", va="top",
            fontsize=7.2, fontweight="bold", color=INK)
    ax.text(81, 30.2, f"{st['n_annotated_ntas']} neighbourhoods", ha="center", va="top", fontsize=6.2, color=INK2)
    lines = ["Street View image of the entrance, with acquisition metadata",
             "Stairs, gates and ramps extracted by a multimodal model",
             "Every model-reported gate or ramp checked by a person",
             "Rule-based accessibility label (with review; model only)"]
    for i, t in enumerate(lines):
        ax.text(45, 25.4 - i * 4.8, "\u2022 " + t, fontsize=6.3, color=INK2, va="top")

    ax.annotate("", xy=(127, 44), xytext=(124.3, 14),
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=1.1, connectionstyle="arc3,rad=0.35",
                                shrinkA=0, shrinkB=0), zorder=4)
    ax.text(133.5, 26, "labels train a model\non the attributes", fontsize=5.6, color=C2, ha="center",
            va="center", style="italic", rotation=90)
    save(fig, "fig1_dataset_overview")


def fig_data_layers():
    """Nested data layers with the number of buildings in each (numbers read from the records)."""
    st = json.loads((config.RESULTS_DIR / "stats" / "stats.json").read_text())
    m = json.loads((config.RESULTS_DIR / "model" / "metrics.json").read_text())
    hv = pd.read_csv(config.RECORDS_DIR / "human_validation_annotations.csv", dtype=str)
    n_set = hv.drop_duplicates(["set", "bin"]).groupby("set").size()
    n_img_ntas = int(pd.read_csv(config.BUILDINGS_CSV, usecols=["nta2020", "streetview_image_acquired"],
                                 dtype={"nta2020": str}).query("streetview_image_acquired")["nta2020"].nunique())
    f = lambda v: f"{int(v):,}"  # noqa: E731

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 140); ax.set_ylim(0, 90); ax.axis("off")

    def box(x0, y0, x1, y1, fc, lw=0.9):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=3.5",
                                    fc=fc, ec=INK, lw=lw, zorder=1))

    def title(x, y, bold, rest):
        ax.text(x, y, bold, ha="center", va="top", fontsize=7.4, fontweight="bold", color=INK, zorder=3)
        ax.text(x, y - 4.2, rest, ha="center", va="top", fontsize=6.3, color=INK2, zorder=3)

    def step(x, y0, y1, label):
        ax.annotate("", xy=(x, y1), xytext=(x, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C1, lw=1.0, shrinkA=0, shrinkB=0), zorder=4)
        ax.text(x + 1.5, (y0 + y1) / 2, label, ha="left", va="center", fontsize=6, color=C1, style="italic",
                zorder=4)

    # outer layer: every building
    box(1, 1, 139, 89, "white", lw=1.1)
    title(70, 87, f"Manhattan building table: {f(st['n_buildings'])} buildings",
          "NYC Open Data: footprints, PLUTO tax lots, address points, sidewalks, neighbourhoods")

    # image-based layers
    cx = 89
    box(40, 4, 138 - 2, 69, "#f6f6f4")
    title(cx, 67, f"Street View images requested: {f(st['n_images_acquired'])} buildings",
          f"{n_img_ntas} neighbourhoods; camera on the nearest street, facing the delivery point")
    box(44, 7, 132, 54, "#eef4fc")
    title(cx, 52, f"Annotated by the multimodal model: {f(st['n_annotated'])} buildings",
          f"{st['n_annotated_ntas']} neighbourhoods; usable image, stairs, gate, ramp")
    box(48, 10, 128, 39, "#e0ebfa")
    title(cx, 37, f"Usable images: {f(st['n_usable'])} buildings", "model-only label")
    box(52, 13, 124, 25, "#cddff7")
    title(cx, 23, f"Judgeable after review: {f(m['n_train'])} buildings",
          "primary label (gates and ramps checked by a person)")
    sx = 57
    step(sx, 58.5, 54.6, "AI annotation, in file order until the budget was spent")
    step(sx, 43.4, 39.6, "image usable")
    step(sx, 29.4, 25.6, "human review of every AI-reported gate or ramp")

    # model output for every building
    ax.text(70, 77, f"Logistic model trained on the labels: accessibility probability for all "
                    f"{f(st['n_buildings'])} buildings (two versions)",
            fontsize=6.4, color=C2, ha="center", va="center", fontweight="bold", zorder=4)
    ax.annotate("", xy=(126, 75.2), xytext=(124.6, 19),
                arrowprops=dict(arrowstyle="-|>", color=C2, lw=1.1, connectionstyle="arc3,rad=0.25",
                                shrinkA=0, shrinkB=0), zorder=4)

    # left column
    ax.text(3.5, 69, "Predictors (all buildings)", fontsize=6.8, fontweight="bold", color=INK, va="top")
    preds = ["Footprint area, roof height", "Floors, construction year", "Ground elevation",
             "Land use, building class", "Distance to sidewalk, street", "Address points",
             "Units, floor area, frontage", "Alterations after 1992", "Historic district, landmark",
             "Basement, proximity codes"]
    for i, t in enumerate(preds):
        ax.text(4.2, 64 - i * 4.0, "\u2022 " + t, fontsize=6, color=INK2, va="top")
    ax.text(3.5, 21, "Human test sets", fontsize=6.8, fontweight="bold", color=INK, va="top")
    for i, t in enumerate([f"A: {f(n_set.get('A', 0))} annotated images",
                           f"B1: {f(n_set.get('B1', 0))} in the 8th area",
                           f"B2: {f(n_set.get('B2', 0))} imaged, not annotated"]):
        ax.text(4.2, 16 - i * 4.0, "\u2022 " + t, fontsize=6, color=INK2, va="top")
    save(fig, "fig1_data_layers")


def main() -> None:
    lab = pd.read_csv(config.LABELS_CSV, dtype={"bin": str})
    b = pd.read_csv(config.BUILDINGS_CSV, dtype={"bin": str})
    pred = pd.read_csv(config.PREDICTIONS_CSV, dtype={"bin": str})
    nta = next(p for p in [config.RAW_DIR / "nyc_neighborhoods.geojson",
                           config.DATA_DIR / "source_snapshot" / "nyc_neighborhoods.geojson.gz"] if p.exists())
    polys = load_polygons(nta, "ntaname", "nta2020")
    fig_workflow()
    fig_data_layers()
    fig_data_layers_simple()
    fig_label_map(lab, polys)
    fig_combinations()
    fig_calibration_and_cv()
    fig_prediction_maps(b, pred, polys)
    fig_capture_years()
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
