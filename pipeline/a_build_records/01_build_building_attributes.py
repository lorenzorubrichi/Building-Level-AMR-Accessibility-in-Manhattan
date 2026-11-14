"""Build the building attribute table for all constructed buildings in Manhattan.

Inputs (``AMR_RAW_DIR``, see data/raw/MANIFEST.csv for sources and versions):
    BUILDING.csv               NYC Building Footprints (polygon WKT; point WKT also accepted)
    pluto_26v1.csv             NYC PLUTO release 26v1 (tax-lot attributes)
    AddressPoint_full.csv      NYC Address Points
    nyc_neighborhoods.geojson  2020 Neighborhood Tabulation Areas (NTA)
    NYC_Sidewalk.csv           NYC Planimetric Database: Sidewalk (polygon WKT)
    Centerline.csv             NYC Street Centerline (CSCL) (line WKT)

Output: data/interim/building_attributes.csv (one row per BIN)

Number format: CSV exports from the NYC Open Data web interface can be
locale-formatted (e.g. '1.396' = one thousand three hundred ninety-six and
'13,06' = 13.06 with an Italian browser). The format is detected per file and
parsed explicitly; an earlier version of the pipeline parsed such files as
English-formatted, which corrupted values >= 1000 (roof heights, construction
years, footprint areas). Footprint area is recomputed from the polygon
geometry; the SHAPE_AREA field is not used because it is computed in Web
Mercator (see the dataset's metadata).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.codes import BLDGCLASS_GROUP_LABELS, LANDUSE_LABELS  # noqa: E402
from amr_access.geo import (  # noqa: E402
    distance_to_lines,
    distance_to_polygons,
    load_polygons,
    nearest_neighbour,
    parse_multipolygon_wkt,
    parse_points,
    points_in_polygons,
    polygon_centroid_area,
)

FEATURE_CODE_BUILDING = 2100
STATUS_CONSTRUCTED = "constructed"
MAX_ZIP_FALLBACK_M = 50.0


# Unambiguous evidence only: "7,520" or "1.396" alone could be either format.
_IT_EVIDENCE = r"^-?\d+,(\d{1,2}|\d{4,})$|^-?\d{1,3}(\.\d{3}){2,}(,\d+)?$|^-?\d{1,3}(\.\d{3})+,\d+$"
_EN_EVIDENCE = r"^-?\d+\.(\d{1,2}|\d{4,})$|^-?\d{1,3}(,\d{3}){2,}(\.\d+)?$|^-?\d{1,3}(,\d{3})+\.\d+$"


def detect_number_format(df: pd.DataFrame, columns: list[str]) -> str:
    """Detect the number format of a (possibly locale-formatted) CSV export.

    'it'         '.' thousands, ',' decimal  (e.g. 1.396 = 1396; 13,06 = 13.06)
    'en_grouped' ',' thousands, '.' decimal  (e.g. 1,700 = 1700; 0.99)
    'en'         no thousands separator, '.' decimal
    Only unambiguous values are used as evidence; conflicting evidence is an error.
    """
    it = en = commas = 0
    for c in columns:
        if c not in df:
            continue
        v = df[c].astype("string").str.strip()
        it += int(v.str.match(_IT_EVIDENCE).fillna(False).sum())
        en += int(v.str.match(_EN_EVIDENCE).fillna(False).sum())
        commas += int(v.str.contains(",", regex=False).fillna(False).sum())
    if it and en:
        raise ValueError(f"Conflicting number formats in {columns}: {it} 'it' vs {en} 'en' values")
    if it:
        return "it"
    if commas:
        if en:
            return "en_grouped"
        raise ValueError(f"Commas in {columns} but the format cannot be determined")
    return "en"


def parse_number(series: pd.Series, fmt: str, name: str) -> pd.Series:
    """Parse numbers in the declared format; refuse ambiguous input."""
    s = series.astype("string").str.strip()
    if fmt == "it":
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    elif fmt == "en_grouped":
        s = s.str.replace(",", "", regex=False)
    elif s.str.contains(",", regex=False).any():
        raise ValueError(f"Column '{name}' contains commas in an 'en' formatted file")
    return pd.to_numeric(s, errors="coerce")


def clean_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)


def load_buildings(path: Path) -> pd.DataFrame:
    b = pd.read_csv(path, dtype=str, low_memory=False)
    b.columns = [c.strip().lower().replace(" ", "_") for c in b.columns]
    b = b.rename(columns={"mappluto_bbl": "map_pluto_bbl", "geometry_source": "geom_source"})
    fmt = detect_number_format(b, ["construction_year", "feature_code", "height_roof"])
    print(f"[buildings] number format detected: {fmt}")
    b["bin"] = clean_id(b["bin"])
    b["feature_code"] = parse_number(b["feature_code"], fmt, "feature_code")
    b["last_status_type"] = b["last_status_type"].str.strip().str.lower()
    n0 = len(b)
    b = b[b["bin"].str.startswith("1")]                       # Manhattan
    b = b[~b["bin"].str.fullmatch(r"\d000000")]               # unassigned BIN placeholders
    b = b[(b["feature_code"] == FEATURE_CODE_BUILDING) & (b["last_status_type"] == STATUS_CONSTRUCTED)]
    print(f"[buildings] {n0:,} rows -> {len(b):,} constructed Manhattan buildings")
    geom = b["the_geom"].astype("string")
    if geom.str.upper().str.contains("POLYGON", na=False).any():
        cca = [polygon_centroid_area(parse_multipolygon_wkt(t)) for t in geom]
        b["centroid_lon"], b["centroid_lat"], b["footprint_area_m2"] = map(np.array, zip(*cca))
    else:
        pts = parse_points(b["the_geom"])
        b["centroid_lon"], b["centroid_lat"] = pts["lon"].to_numpy(), pts["lat"].to_numpy()
        b["footprint_area_m2"] = np.nan
    b["height_roof_m"] = parse_number(b["height_roof"], fmt, "height_roof") * config.FT_TO_M
    b.loc[b["height_roof_m"] <= 0, "height_roof_m"] = np.nan          # 0 = unavailable (metadata)
    b["ground_elevation_m"] = parse_number(b["ground_elevation"], fmt, "ground_elevation") * config.FT_TO_M
    year = parse_number(b["construction_year"], fmt, "construction_year")
    b["construction_year"] = year.where(year > 0)                     # 0 = unavailable (metadata)
    b["bbl"] = clean_id(b["map_pluto_bbl"])
    dup = b["bin"].duplicated(keep=False)
    if dup.any():
        # BIN is not unique in Building Footprints (e.g. multi-part structures):
        # keep the part with the largest roof height, report how many were merged.
        print(f"[buildings] {int(dup.sum()):,} rows share a BIN; keeping one row per BIN")
        b = b.sort_values("height_roof_m", ascending=False).drop_duplicates("bin")
    keep = ["bin", "bbl", "centroid_lon", "centroid_lat", "footprint_area_m2", "construction_year", "height_roof_m",
            "ground_elevation_m", "geom_source", "last_edited_date"]
    return b[keep].reset_index(drop=True)


PLUTO_NUMERIC = ["numfloors", "unitsres", "unitstotal", "bldgarea", "numbldgs", "yearalter1", "yearalter2",
                 "lotfront", "bldgfront", "bldgdepth", "bsmtcode", "proxcode"]
PLUTO_TEXT = ["bbl", "landuse", "bldgclass", "histdist", "landmark", "version"]
SQFT_TO_M2 = 0.09290304


def add_pluto(b: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Join PLUTO tax-lot attributes through MAPPLUTO_BBL (all PLUTO fields are lot-level)."""
    wanted = set(PLUTO_NUMERIC + PLUTO_TEXT)
    p = pd.read_csv(path, dtype=str, low_memory=False, usecols=lambda c: c.strip().lower() in wanted)
    p.columns = [c.strip().lower() for c in p.columns]
    fmt = detect_number_format(p, ["bldgarea", "unitstotal", "numfloors"])
    print(f"[pluto] number format detected: {fmt}; columns: {sorted(p.columns)}")
    p["bbl"] = clean_id(p["bbl"].str.replace(",", "", regex=False))
    p = p.drop_duplicates("bbl")
    for c in PLUTO_NUMERIC:
        if c in p:
            p[c] = parse_number(p[c], fmt, c)
    p.loc[p["numfloors"] <= 0, "numfloors"] = np.nan
    p["landuse"] = parse_number(p["landuse"], fmt, "landuse").astype("Int64").astype("string")
    p["bldgclass"] = p["bldgclass"].str.strip().str.upper()
    out = b.merge(p, on="bbl", how="left", validate="many_to_one")
    out["landuse_label"] = out["landuse"].map(LANDUSE_LABELS)
    out["bldgclass_group"] = out["bldgclass"].str[0]
    out["bldgclass_group_label"] = out["bldgclass_group"].map(BLDGCLASS_GROUP_LABELS)
    if "unitsres" in out:
        for c in ["yearalter1", "yearalter2"]:
            out.loc[out[c] <= 0, c] = np.nan                      # 0 = never altered
        out["last_alteration_year"] = out[["yearalter1", "yearalter2"]].max(axis=1)
        out["lot_bldgarea_m2"] = out.pop("bldgarea") * SQFT_TO_M2
        for c in ["lotfront", "bldgfront", "bldgdepth"]:
            out[c + "_m"] = out.pop(c).where(lambda v: v > 0) * config.FT_TO_M
        out["in_historic_district"] = out["histdist"].notna()
        out["is_landmark"] = out["landmark"].notna()
        out = out.rename(columns={"numbldgs": "lot_numbldgs", "version": "pluto_version", "histdist": "historic_district",
                                  "landmark": "landmark_name", "bsmtcode": "basement_code",
                                  "proxcode": "proximity_code", "unitsres": "lot_units_res",
                                  "unitstotal": "lot_units_total"})
    print(f"[pluto] matched {out['bldgclass'].notna().mean():.1%} of buildings by BBL")
    return out


def add_address_points(b: pd.DataFrame, path: Path) -> pd.DataFrame:
    cols = ["the_geom", "bin", "zipcode", "house_number", "full_street_name"]
    a = pd.read_csv(path, dtype=str, usecols=lambda c: c.lower() in cols, low_memory=False)
    a.columns = [c.lower() for c in a.columns]
    a["bin"] = clean_id(a["bin"])
    a = a[a["bin"].str.startswith("1", na=False)].copy()
    pts = parse_points(a["the_geom"])
    a["ap_lon"], a["ap_lat"] = pts["lon"].to_numpy(), pts["lat"].to_numpy()
    a["zipcode"] = a["zipcode"].str.extract(r"(\d{5})", expand=False)

    # 1) address point with the same BIN, nearest to the building centroid
    m = b[["bin", "centroid_lon", "centroid_lat"]].merge(a, on="bin", how="inner")
    from amr_access.geo import haversine_m

    m["d"] = haversine_m(m["centroid_lon"], m["centroid_lat"], m["ap_lon"], m["ap_lat"])
    best = m.sort_values(["bin", "d"]).drop_duplicates("bin")
    best = best[["bin", "ap_lon", "ap_lat", "zipcode", "house_number", "full_street_name"]]
    out = b.merge(best, on="bin", how="left")
    out["n_address_points"] = out["bin"].map(a.groupby("bin").size()).fillna(0).astype(int)
    has_ap = out["ap_lon"].notna()
    out["delivery_point_source"] = np.where(has_ap, "address_point", "centroid")
    out["delivery_lon"] = out["ap_lon"].where(has_ap, out["centroid_lon"])
    out["delivery_lat"] = out["ap_lat"].where(has_ap, out["centroid_lat"])
    out["full_address"] = (out["house_number"].fillna("") + " " + out["full_street_name"].fillna("")).str.strip()
    out.loc[~has_ap, "full_address"] = pd.NA
    out["zipcode_source"] = np.where(out["zipcode"].notna(), "address_point_same_bin", None)

    # 2) ZIP fallback: nearest Manhattan address point within 50 m of the centroid
    miss = out["zipcode"].isna().to_numpy()
    if miss.any():
        a_zip = a.dropna(subset=["zipcode", "ap_lon", "ap_lat"]).reset_index(drop=True)
        idx, _ = nearest_neighbour(out.loc[miss, "centroid_lon"].to_numpy(), out.loc[miss, "centroid_lat"].to_numpy(),
                                   a_zip["ap_lon"].to_numpy(), a_zip["ap_lat"].to_numpy(), max_dist_m=MAX_ZIP_FALLBACK_M)
        fill = np.where(idx >= 0, a_zip["zipcode"].to_numpy()[np.clip(idx, 0, None)], None)
        rows = np.flatnonzero(miss)
        out.loc[rows, "zipcode"] = fill
        out.loc[rows[idx >= 0], "zipcode_source"] = "nearest_address_point_50m"
    print(f"[address] delivery point from address point: {has_ap.mean():.1%}; "
          f"ZIP available: {out['zipcode'].notna().mean():.1%}")
    return out.drop(columns=["ap_lon", "ap_lat", "house_number", "full_street_name"])


def add_neighborhoods(b: pd.DataFrame, path: Path) -> pd.DataFrame:
    polys = load_polygons(path, "ntaname", "nta2020")
    names, codes = points_in_polygons(b["centroid_lon"].to_numpy(), b["centroid_lat"].to_numpy(), polys)
    b = b.copy()
    b["nta2020"], b["nta_name"] = codes, names
    print(f"[nta] assigned {pd.notna(names).mean():.2%} of buildings to a 2020 NTA")
    return b


def add_sidewalk_distance(b: pd.DataFrame, sidewalk_csv: Path) -> pd.DataFrame:
    """Distance (m) from the delivery point to the nearest sidewalk polygon (0 if inside)."""
    sw = pd.read_csv(sidewalk_csv, dtype=str, usecols=lambda c: c.lower() == "the_geom")
    geoms = sw.iloc[:, 0]
    # keep sidewalks within ~1 km of the buildings to limit memory
    first = geoms.str.extract(r"(-?\d+\.\d+)\s+(-?\d+\.\d+)").astype(float)
    lon0, lon1 = b["delivery_lon"].min() - 0.015, b["delivery_lon"].max() + 0.015
    lat0, lat1 = b["delivery_lat"].min() - 0.01, b["delivery_lat"].max() + 0.01
    near = first[0].between(lon0, lon1) & first[1].between(lat0, lat1)
    d = distance_to_polygons(b["delivery_lon"].to_numpy(), b["delivery_lat"].to_numpy(), geoms[near].tolist())
    b = b.copy()
    b["distance_to_sidewalk_m"] = d
    print(f"[sidewalk] {int(near.sum()):,} sidewalk polygons; distance computed for {np.isfinite(d).mean():.2%}")
    return b


def add_street_distance(b: pd.DataFrame, centerline_csv: Path) -> pd.DataFrame:
    """Distance (m) from the delivery point to the nearest vehicular street centreline.

    Same segment filter as the Street View camera points: STATUS 2 (constructed),
    RW_TYPE 1 (street), NONPED != 'V' (excludes non-vehicular segments).
    """
    c = pd.read_csv(centerline_csv, dtype=str)
    cols = {k.strip().lower(): k for k in c.columns}
    keep = (c[cols["status"]] == "2") & (c[cols["rw_type"]] == "1") & (c[cols["nonped"]].fillna("") != "V")
    d = distance_to_lines(b["delivery_lon"].to_numpy(), b["delivery_lat"].to_numpy(), c.loc[keep, cols["the_geom"]].tolist())
    b = b.copy()
    b["distance_to_street_m"] = d
    print(f"[street] {int(keep.sum()):,} street segments; distance computed for {np.isfinite(d).mean():.2%}")
    return b


def add_quality_fields(b: pd.DataFrame) -> pd.DataFrame:
    """Lot multiplicity and a plausibility flag for height vs floors."""
    b = b.copy()
    b["n_buildings_on_lot"] = b["bbl"].map(b["bbl"].value_counts()).astype("Int64")
    h_per_floor = b["height_roof_m"] / b["numfloors"]
    # PLUTO floors are lot-level: only flag single-building lots
    single = (b["n_buildings_on_lot"] == 1).fillna(False).to_numpy(dtype=bool)
    hpf = h_per_floor.to_numpy(dtype=float)
    b["qa_height_floors_inconsistent"] = single & np.isfinite(hpf) & ((hpf < 2.0) | (hpf > 8.0))
    print(f"[qa] multi-building lots: {(b['n_buildings_on_lot'] > 1).mean():.1%}; "
          f"height/floor inconsistent: {b['qa_height_floors_inconsistent'].sum():,}")
    return b


def resolve(path: Path) -> Path:
    """Accept either the plain file or its gzip-compressed snapshot copy."""
    if path.exists():
        return path
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"{path} (or {gz.name}) not found; see data/raw/MANIFEST.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--buildings", type=Path, default=config.RAW_DIR / "BUILDING.csv")
    ap.add_argument("--pluto", type=Path, default=config.RAW_DIR / "pluto_26v1.csv")
    ap.add_argument("--address-points", type=Path, default=config.RAW_DIR / "AddressPoint_full.csv")
    ap.add_argument("--nta", type=Path, default=config.RAW_DIR / "nyc_neighborhoods.geojson")
    ap.add_argument("--sidewalk", type=Path, default=config.RAW_DIR / "NYC_Sidewalk.csv")
    ap.add_argument("--centerline", type=Path, default=config.RAW_DIR / "Centerline.csv")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "interim" / "building_attributes.csv")
    args = ap.parse_args()

    b = load_buildings(resolve(args.buildings))
    b = add_pluto(b, resolve(args.pluto))
    b = add_address_points(b, resolve(args.address_points))
    b = add_neighborhoods(b, resolve(args.nta))
    b = add_sidewalk_distance(b, resolve(args.sidewalk))
    b = add_street_distance(b, resolve(args.centerline))
    b = add_quality_fields(b)
    b.insert(2, "borough", "Manhattan")
    b = b.sort_values("bin").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    b.to_csv(args.out, index=False)
    print(f"[done] {len(b):,} buildings -> {args.out}")


if __name__ == "__main__":
    main()
