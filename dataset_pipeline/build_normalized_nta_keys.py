from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import wkt


BASE_DIR = Path(__file__).resolve().parent
RAWDATA_DIR = BASE_DIR / "rawdata"

BUILDING_CSV = RAWDATA_DIR / "BUILDING.csv"
POPULATION_CSV = RAWDATA_DIR / "population.csv"
NTA_GEOJSON = RAWDATA_DIR / "nyc_neighborhoods.geojson"

DEFAULT_BUILDING_OUTPUT = RAWDATA_DIR / "BUILDING_with_nta_key.csv"
DEFAULT_POPULATION_OUTPUT = RAWDATA_DIR / "population_with_nta_key.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return out


def normalize_nta_key(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = text.lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(r"\band\b", " ", text)
    text = text.replace("'", "")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def load_buildings() -> gpd.GeoDataFrame:
    df = pd.read_csv(BUILDING_CSV, low_memory=False)
    df = normalize_columns(df)
    df["bin"] = df["bin"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["geometry"] = df["the_geom"].astype(str).apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf


def load_neighborhoods() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(NTA_GEOJSON)
    gdf = normalize_columns(gdf)
    gdf["boroname"] = gdf["boroname"].astype(str).str.strip()
    gdf["ntaname"] = gdf["ntaname"].astype(str).str.strip()
    gdf["nta_key"] = gdf["ntaname"].apply(normalize_nta_key)
    return gdf.to_crs("EPSG:4326")


def build_building_output() -> pd.DataFrame:
    buildings = load_buildings()
    neighborhoods = load_neighborhoods()

    joined = gpd.sjoin(
        buildings,
        neighborhoods[["boroname", "ntaname", "nta_key", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(
        columns={
            "boroname": "borough_nta",
            "ntaname": "nta_name",
        }
    )
    joined["nta_key"] = joined["nta_name"].apply(normalize_nta_key)

    out = pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore")).copy()
    preferred = [
        "bin",
        "name",
        "map_pluto_bbl",
        "construction_year",
        "feature_code",
        "last_status_type",
        "height_roof",
        "borough_nta",
        "nta_name",
        "nta_key",
    ]
    existing = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing]
    return out[existing + remaining]


def build_population_output() -> pd.DataFrame:
    df = pd.read_csv(POPULATION_CSV, low_memory=False)
    df = normalize_columns(df)
    df["borough"] = df["borough"].astype(str).str.strip()
    df["nta_name"] = df["nta_name"].astype(str).str.strip()
    df["nta_key"] = df["nta_name"].apply(normalize_nta_key)
    preferred = ["borough", "year", "fips_county_code", "nta_code", "nta_name", "nta_key", "population"]
    existing = [c for c in preferred if c in df.columns]
    remaining = [c for c in df.columns if c not in existing]
    return df[existing + remaining].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create normalized NTA keys for BUILDING and population datasets.")
    parser.add_argument("--building-output", type=Path, default=DEFAULT_BUILDING_OUTPUT)
    parser.add_argument("--population-output", type=Path, default=DEFAULT_POPULATION_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.building_output.parent.mkdir(parents=True, exist_ok=True)
    args.population_output.parent.mkdir(parents=True, exist_ok=True)

    building_out = build_building_output()
    population_out = build_population_output()

    building_out.to_csv(args.building_output, index=False)
    population_out.to_csv(args.population_output, index=False)

    print(f"[NTA KEY] BUILDING rows: {len(building_out)}")
    print(f"[NTA KEY] Population rows: {len(population_out)}")
    print(f"[NTA KEY] BUILDING output: {args.building_output}")
    print(f"[NTA KEY] Population output: {args.population_output}")


if __name__ == "__main__":
    main()
