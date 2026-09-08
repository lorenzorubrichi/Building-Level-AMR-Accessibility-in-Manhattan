from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import MultiPolygon, Point, Polygon


# ============================================================
# CONFIGURAZIONE GENERALE
# ============================================================

# Cartella base del progetto.
BASE_DIR = Path(__file__).resolve().parent
RAWDATA_DIR = BASE_DIR / "rawdata"
MODEL_DIR = BASE_DIR / "Model"
VISUALIZATION_DIR = BASE_DIR / "visualization"

# ----------------------------
# FILE INPUT
# ----------------------------

# Dataset principali della pipeline.
BUILDING_CSV = RAWDATA_DIR / "BUILDING.csv"
CENTERLINE_CSV = RAWDATA_DIR / "Centerline.csv"
NTA_GEOJSON = RAWDATA_DIR / "nyc_neighborhoods.geojson"
PLUTO_CSV = RAWDATA_DIR / "pluto_full.csv"
ADDRESSPOINT_CSV = RAWDATA_DIR / "AddressPoint_full.csv"
POPULATION_CSV = RAWDATA_DIR / "population.csv"
TRAFFIC_CSV = RAWDATA_DIR / "traffic.csv"
PEDESTRIAN_CSV = RAWDATA_DIR / "pedestrian_mobility.csv"
SIDEWALK_CSV = RAWDATA_DIR / "NYC_Sidewalk.csv"

# Dataset parcheggio.
PARKING_REGULATIONS_CSV = RAWDATA_DIR / "Parking_Regulation.csv"
PARKING_METERS_CSV = RAWDATA_DIR / "Parking_Meters.csv"
PARKING_BLOCKFACES_CSV = RAWDATA_DIR / "Parking_Meters_-_ParkNYC_Block_Faces.csv"
PARKING_RATEZONES_CSV = RAWDATA_DIR / "Parking_Meters_-_Citywide_Rate_Zones.csv"

# ----------------------------
# FILE OUTPUT
# ----------------------------

# Buildings.
OUTPUTbuildings_CSV = BASE_DIR / "OUTPUTbuildings.csv"
OUTPUTbuildings_XLSX = MODEL_DIR / "OUTPUTbuildings.xlsx"
OUTPUTbuildings_GEOJSON = VISUALIZATION_DIR / "OUTPUTbuildings.geojson"

# Streets.
OUTPUTcenterline_CSV = VISUALIZATION_DIR / "OUTPUTcenterline.csv"
OUTPUTcenterline_GEOJSON = VISUALIZATION_DIR / "OUTPUTcenterline.geojson"

# Service points.
OUTPUTservicepoints_CSV = VISUALIZATION_DIR / "OUTPUTservicepoints.csv"
OUTPUTservicepoints_GEOJSON = VISUALIZATION_DIR / "OUTPUTservicepoints.geojson"

# Population.
OUTPUTpopulation_TXT = BASE_DIR / "OUTPUTpopulation.txt"
OUTPUTpopulation_CSV = BASE_DIR / "OUTPUTpopulation.csv"
OUTPUTpopulation_XLSX = MODEL_DIR / "OUTPUTpopulation.xlsx"

# Traffic.
OUTPUTtraffic_CSV = BASE_DIR / "OUTPUTtraffic.csv"
OUTPUTtraffic_XLSX = MODEL_DIR / "OUTPUTtraffic.xlsx"

# Pedestrian raw filtered output.
OUTPUTpedestrian_CSV = VISUALIZATION_DIR / "OUTPUTpedestrian.csv"
OUTPUTpedestrian_GEOJSON = VISUALIZATION_DIR / "OUTPUTpedestrian.geojson"

# Sidewalk raw filtered output.
OUTPUTsidewalk_CSV = VISUALIZATION_DIR / "OUTPUTsidewalk.csv"
OUTPUTsidewalk_GEOJSON = VISUALIZATION_DIR / "OUTPUTsidewalk.geojson"

# Parking outputs.
OUTPUTparking_regulations_CSV = VISUALIZATION_DIR / "OUTPUTparking_regulations.csv"
OUTPUTparking_regulations_GEOJSON = VISUALIZATION_DIR / "OUTPUTparking_regulations.geojson"

OUTPUTparking_meters_CSV = VISUALIZATION_DIR / "OUTPUTparking_meters.csv"
OUTPUTparking_meters_GEOJSON = VISUALIZATION_DIR / "OUTPUTparking_meters.geojson"

OUTPUTparking_blockfaces_CSV = VISUALIZATION_DIR / "OUTPUTparking_blockfaces.csv"
OUTPUTparking_blockfaces_GEOJSON = VISUALIZATION_DIR / "OUTPUTparking_blockfaces.geojson"

OUTPUTparking_ratezones_CSV = VISUALIZATION_DIR / "OUTPUTparking_ratezones.csv"
OUTPUTparking_ratezones_GEOJSON = VISUALIZATION_DIR / "OUTPUTparking_ratezones.geojson"

# Output finale a livello strada:
# contiene informazioni stradali + parcheggio + pedestrian rank.
OUTPUTstreet_context_CSV = BASE_DIR / "OUTPUTstreet_context.csv"
OUTPUTstreet_context_GEOJSON = VISUALIZATION_DIR / "OUTPUTstreet_context.geojson"
OUTPUTstreet_context_XLSX = MODEL_DIR / "OUTPUTstreet_context.xlsx"

# AMR feature table for regression/scoring experiments.
OUTPUTamr_features_XLSX = MODEL_DIR / "OUTPUTamr_features.xlsx"

# ----------------------------
# FILTRI / OPZIONI
# ----------------------------

# Feature code 2.100 = edificio.
VALID_BUILDING_FEATURE_CODE = 2.100

# Teniamo solo edifici costruiti.
VALID_BUILDING_STATUS = "constructed"

# Se True, salva sempre gli output su disco.
SAVE_OUTPUTS = True

# Se True, dopo OUTPUTamr_features.xlsx richiama anche il popolamento
# dei tempi di routing OSRM. Per la pipeline finale "last meter only"
# non serve, quindi lasciamo False per evitare attese molto lunghe.
RUN_ROUTING_TIMES = False

# CRS finale lat/lon.
OUTPUT_CRS = "EPSG:4326"

# CRS metrico di lavoro per:
# - centroidi
# - distanze
# - buffer
# - nearest join
WORK_CRS = "EPSG:2263"

# EPSG:2263 lavora in piedi. Converte piedi -> metri quando vogliamo esportare distanze metriche.
FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M

# Buffer usato per associare meter e regulation signs alle strade.
PARKING_BUFFER_M = 20
PARKING_BUFFER_FT = PARKING_BUFFER_M * M_TO_FT

# Soglia in metri per dire se una strada e' raggiungibile da marciapiede.
SIDEWALK_REACHABLE_DISTANCE_M = 20
SIDEWALK_REACHABLE_DISTANCE_FT = SIDEWALK_REACHABLE_DISTANCE_M * M_TO_FT


# ============================================================
# MAPPING BOROUGH
# ============================================================

# Mapping borough -> codice usato dal dataset Centerline.
BOROUGH_TO_CODE = {
    "Manhattan": 1,
    "Bronx": 2,
    "Brooklyn": 3,
    "Queens": 4,
    "Staten Island": 5,
}

# Mapping borough -> prefisso BIN usato nel dataset BUILDING.
BOROUGH_TO_BIN_PREFIX = {
    "Manhattan": "1",
    "Bronx": "2",
    "Brooklyn": "3",
    "Queens": "4",
    "Staten Island": "5",
}


# ============================================================
# FUNZIONI GENERICHE
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizza i nomi colonna per lavorare in modo stabile:
    - lowercase
    - trim spazi
    - spazi -> underscore
    - trattini -> underscore
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def load_csv(path: Path | str) -> pd.DataFrame:
    """
    Carica un CSV e normalizza i nomi colonna.
    low_memory=False riduce warning su colonne miste.
    """
    df = pd.read_csv(path, low_memory=False)
    return normalize_columns(df)


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """
    Converte una serie in numerico gestendo:
    - virgole come separatori migliaia
    - stringhe vuote
    - spazi
    """
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


def safe_decimal_series(series: pd.Series) -> pd.Series:
    """
    Converte una serie numerica gestendo anche valori con virgola decimale.
    Esempio sidewalk:
    - "171,8796812" -> 171.8796812
    """
    s = series.astype(str).str.strip()
    comma_decimal = s.str.contains(",", regex=False) & ~s.str.contains(".", regex=False)
    normalized = s.where(~comma_decimal, s.str.replace(",", ".", regex=False))
    normalized = normalized.str.replace(",", "", regex=False)
    return pd.to_numeric(normalized, errors="coerce")


def feet_to_meters(value: Any) -> float | None:
    """
    Converte una distanza da piedi a metri.
    """
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    return float(num) * FT_TO_M


def wkt_to_gdf(df: pd.DataFrame, geom_col: str = "the_geom", crs: str = OUTPUT_CRS) -> gpd.GeoDataFrame:
    """
    Converte una colonna WKT (stringa) in geometria shapely
    e crea un GeoDataFrame.
    """
    work = df.copy()
    work[geom_col] = work[geom_col].astype(str)
    work["geometry"] = work[geom_col].apply(wkt.loads)
    return gpd.GeoDataFrame(work, geometry="geometry", crs=crs)


def xy_to_gdf(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    source_crs: str = WORK_CRS,
    target_crs: str = OUTPUT_CRS
) -> gpd.GeoDataFrame:
    """
    Crea un GeoDataFrame da colonne X/Y.
    Nel dataset parking regulations:
    - sign_x_coord / sign_y_coord sono in EPSG:2263
    """
    work = df.copy()
    work[x_col] = safe_numeric_series(work[x_col])
    work[y_col] = safe_numeric_series(work[y_col])
    work = work.dropna(subset=[x_col, y_col]).copy()

    gdf = gpd.GeoDataFrame(
        work,
        geometry=gpd.points_from_xy(work[x_col], work[y_col]),
        crs=source_crs
    )

    return gdf.to_crs(target_crs)


def save_gdf_outputs(
    gdf: gpd.GeoDataFrame,
    csv_path: Path | None = None,
    geojson_path: Path | None = None,
    xlsx_path: Path | None = None
) -> None:
    """
    Salva un GeoDataFrame in:
    - CSV
    - GeoJSON
    - Excel

    Nei formati tabellari la geometria viene convertita in WKT.
    """
    df_out = pd.DataFrame(gdf.copy())

    if "geometry" in df_out.columns:
        df_out["geometry"] = gdf.geometry.to_wkt()

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(csv_path, index=False)

    if xlsx_path is not None:
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_excel(xlsx_path, index=False)

    if geojson_path is not None:
        geojson_path.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(geojson_path, driver="GeoJSON")


def write_text_output(path: Path, text: str) -> None:
    """
    Scrive un file testuale creando la cartella di destinazione se manca.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def output_name(path: Path) -> str:
    """
    Restituisce un path relativo alla cartella dataset, utile nel riepilogo.
    """
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def require_file(path: Path, label: str) -> None:
    """
    Controlla che un file esista prima di usarlo.
    """
    if not path.exists():
        raise FileNotFoundError(f"File mancante per {label}: {path}")


def wkt_or_none(value: Any) -> str | None:
    """
    Restituisce la WKT di una geometria oppure None.
    """
    if value is None or pd.isna(value):
        return None
    try:
        return value.wkt
    except Exception:
        return None


def safe_bool(value: Any) -> bool:
    """
    Converte valori booleani/stringa in bool stabile.
    """
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ["true", "1", "yes", "y", "si", "s"]


def clean_nta_name(value: Any) -> str:
    """
    Pulisce un nome borough/quartiere:
    - stringa
    - trim
    - compatta spazi interni multipli
    """
    s = str(value).strip()
    s = " ".join(s.split())
    return s


def normalize_nta_key(value: Any) -> str:
    """
    Normalizza un nome NTA/quartiere per join robusti:
    - lowercase
    - & -> and
    - rimozione della parola and
    - rimozione apostrofi, spazi, trattini e simboli
    - solo alfanumerico finale
    """
    s = "" if value is None or pd.isna(value) else str(value)
    s = s.lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"\band\b", " ", s)
    s = s.replace("'", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def population_to_plain_int(value: Any) -> int | None:
    """
    Converte la popolazione in intero senza punti né virgole separatori.
    Esempi:
    - "28,149" -> 28149
    - "28.149" -> 28149
    """
    if value is None or pd.isna(value):
        return None

    s = str(value).strip().replace(",", "").replace(".", "")
    try:
        return int(float(s))
    except Exception:
        return None


def is_curb_restriction_sign(description: Any) -> bool:
    """
    Riconosce i segnali realmente restrittivi per la sosta/fermata.
    Include segnali tipo:
    - NO PARKING
    - NO STANDING
    - NO STOPPING

    Esclude elementi informativi o permissivi come:
    - PAY-BY-CELL
    - route/location panels
    - commercial vehicles only / loading
    """
    text = str(description).strip().upper()
    if not text:
        return False

    restrictive_tokens = ["NO PARKING", "NO STANDING", "NO STOPPING"]
    permissive_or_irrelevant_tokens = [
        "PAY-BY-CELL",
        "METER",
        "COMMERCIAL VEHICLES ONLY",
        "TRUCK LOADING",
        "LOADING",
        "UNLOADING",
        "TAXI",
        "BUS",
        "ROUTE PANEL",
        "DESTINATION PANEL",
        "LOCATION PANEL",
        "NIGHT SERVICE",
        "LOCAL MTA",
        "EXPRESS MTA",
    ]

    if any(token in text for token in permissive_or_irrelevant_tokens):
        return False

    return any(token in text for token in restrictive_tokens)


# ============================================================
# QUARTIERI / AREA SELEZIONATA
# ============================================================

def load_neighborhoods_geojson(path: Path) -> gpd.GeoDataFrame:
    """
    Carica il GeoJSON dei quartieri.
    I campi chiave devono essere:
    - boroname
    - ntaname

    ntaname conferma che i quartieri usati nella pipeline sono NTA.
    """
    require_file(path, "quartieri")

    gdf = gpd.read_file(path)
    gdf = normalize_columns(gdf)

    if "boroname" not in gdf.columns:
        raise ValueError("Nel GeoJSON manca la colonna 'boroname'")
    if "ntaname" not in gdf.columns:
        raise ValueError("Nel GeoJSON manca la colonna 'ntaname'")

    gdf["boroname"] = gdf["boroname"].astype(str).apply(clean_nta_name)
    gdf["ntaname"] = gdf["ntaname"].astype(str).apply(clean_nta_name)
    gdf["nta_key"] = gdf["ntaname"].apply(normalize_nta_key)

    print(f"[NTA] GeoJSON caricato correttamente. I quartieri usati sono NTA: {len(gdf)} record.")

    return gdf.to_crs(OUTPUT_CRS)


def get_selected_area_polygon(gdf_neighborhoods: gpd.GeoDataFrame, borough: str, neighborhoods: list[str]):
    """
    Costruisce l'area finale selezionata:
    - filtro borough
    - filtro quartieri
    - union dei poligoni selezionati
    """
    borough = clean_nta_name(borough)
    neighborhoods = [clean_nta_name(n) for n in neighborhoods]

    print("\n[AREA] Borough richiesto:", borough)
    print("[AREA] Neighborhoods richiesti:", neighborhoods)

    gdf_borough = gdf_neighborhoods[gdf_neighborhoods["boroname"] == borough].copy()
    print("[AREA] Quartieri disponibili nel borough:", len(gdf_borough))

    if not neighborhoods:
        raise ValueError("La lista dei quartieri è vuota")

    gdf_selected = gdf_borough[gdf_borough["ntaname"].isin(neighborhoods)].copy()
    print("[AREA] Quartieri trovati nel GeoJSON:", len(gdf_selected))

    if gdf_selected.empty:
        raise ValueError("Nessun quartiere trovato nel GeoJSON per la selezione ricevuta")

    return gdf_selected, gdf_selected.union_all()


# ============================================================
# BUILDINGS
# ============================================================

def prepare_buildings(df_building: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara il dataset BUILDING:
    - normalizza BIN
    - converte feature_code in numerico
    - normalizza status
    - normalizza BBL se presente
    """
    df = df_building.copy()

    df["bin"] = (
        df["bin"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    df["feature_code"] = pd.to_numeric(df["feature_code"], errors="coerce")
    df["last_status_type"] = df["last_status_type"].astype(str).str.strip().str.lower()

    if "map_pluto_bbl" in df.columns:
        df["map_pluto_bbl"] = df["map_pluto_bbl"].astype(str).str.replace(r"\.0$", "", regex=True)

    return df


def filter_buildings_by_borough(df_building: pd.DataFrame, borough_prefix_bin: str) -> pd.DataFrame:
    """
    Filtra gli edifici di un borough usando il prefisso BIN.
    """
    return df_building[df_building["bin"].str.startswith(str(borough_prefix_bin))].copy()


def filter_deliverable_buildings(df_building: pd.DataFrame) -> pd.DataFrame:
    """
    Tiene solo edifici effettivi e costruiti.
    """
    return df_building[
        (df_building["feature_code"] == VALID_BUILDING_FEATURE_CODE)
        & (df_building["last_status_type"] == VALID_BUILDING_STATUS)
    ].copy()


def add_neighborhood_info(gdf: gpd.GeoDataFrame, gdf_selected_neighborhoods: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Associa borough e neighborhood a ciascun elemento tramite spatial join.
    """
    joined = gpd.sjoin(
        gdf,
        gdf_selected_neighborhoods[["boroname", "ntaname", "nta_key", "geometry"]],
        how="left",
        predicate="intersects",
    )

    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(columns={"boroname": "borough", "ntaname": "neighborhood"})

    return joined


def add_centroid_columns(gdf_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calcola centroide edificio in CRS metrico e lo riporta in lat/lon.
    Salva:
    - centroid_geom
    - centroid_lon
    - centroid_lat
    """
    gdf_work = gdf_buildings.to_crs(WORK_CRS).copy()
    centroids_work = gdf_work.geometry.centroid
    centroids = gpd.GeoSeries(centroids_work, crs=WORK_CRS).to_crs(OUTPUT_CRS)

    gdf_buildings = gdf_buildings.copy()
    gdf_buildings["centroid_geom"] = centroids.to_wkt()
    gdf_buildings["centroid_lon"] = centroids.x
    gdf_buildings["centroid_lat"] = centroids.y

    return gdf_buildings


def enrich_with_pluto(gdf_buildings: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, bool]:
    """
    Arricchisce gli edifici con PLUTO:
    - numfloors
    - landuse
    - bldgclass

    usando il join:
    - map_pluto_bbl (buildings)
    - bbl (pluto)
    """
    if not PLUTO_CSV.exists():
        print(f"[PLUTO] File non trovato, salto merge: {PLUTO_CSV}")
        for col in ["numfloors", "landuse", "bldgclass"]:
            if col not in gdf_buildings.columns:
                gdf_buildings[col] = None
        return gdf_buildings, False

    df_pluto = load_csv(PLUTO_CSV)
    needed = [c for c in ["bbl", "numfloors", "landuse", "bldgclass"] if c in df_pluto.columns]

    if "bbl" not in needed:
        raise ValueError("Nel PLUTO manca la colonna 'bbl'")

    df_pluto = df_pluto[needed].copy()
    df_pluto["bbl"] = df_pluto["bbl"].astype(str).str.replace(r"\.0$", "", regex=True)

    merged = gdf_buildings.merge(
        df_pluto,
        how="left",
        left_on="map_pluto_bbl",
        right_on="bbl",
    )

    return merged, True


# ============================================================
# ADDRESS POINTS
# ============================================================

def load_address_points() -> gpd.GeoDataFrame | None:
    """
    Carica gli Address Point.
    Gestisce sia:
    - stringhe tipo dict GeoJSON
    - WKT classico
    """
    if not ADDRESSPOINT_CSV.exists():
        print(f"[ADDRESSPOINT] File non trovato: {ADDRESSPOINT_CSV}")
        return None

    df = load_csv(ADDRESSPOINT_CSV)

    if "the_geom" not in df.columns:
        raise ValueError("Nel file AddressPoint manca la colonna 'the_geom'")

    sample = str(df["the_geom"].iloc[0])

    if sample.startswith("{") and "coordinates" in sample:
        import ast
        from shapely.geometry import shape

        df = df.copy()
        df["geometry"] = df["the_geom"].apply(lambda x: shape(ast.literal_eval(x)))
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=OUTPUT_CRS)
    else:
        gdf = wkt_to_gdf(df)

    if "bin" in gdf.columns:
        gdf["bin"] = (
            gdf["bin"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

    return gdf.to_crs(OUTPUT_CRS)


def add_address_points(gdf_buildings: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, bool]:
    """
    Aggiunge agli edifici:
    - address_point_geom
    - has_address_point
    - delivery_point_geom
    - delivery_point_source
    - delivery_lon
    - delivery_lat

    Logica:
    - default: delivery point = centroide
    - se esiste address point: delivery point = address point
    """
    gdf_buildings = gdf_buildings.copy()
    gdf_address = load_address_points()

    # Default iniziale.
    gdf_buildings["address_point_geom"] = None
    gdf_buildings["has_address_point"] = False
    gdf_buildings["delivery_point_geom"] = gdf_buildings["centroid_geom"]
    gdf_buildings["delivery_point_source"] = "centroid_fallback"
    gdf_buildings["delivery_lon"] = gdf_buildings["centroid_lon"]
    gdf_buildings["delivery_lat"] = gdf_buildings["centroid_lat"]

    if gdf_address is None:
        print("[ADDRESSPOINT] Nessun file disponibile, uso solo il centroide.")
        return gdf_buildings, False

    if "bin" not in gdf_address.columns:
        print("[ADDRESSPOINT] Colonna 'bin' mancante, uso solo il centroide.")
        return gdf_buildings, False

    gdf_buildings["bin"] = (
        gdf_buildings["bin"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    gdf_address["bin"] = (
        gdf_address["bin"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )

    selected_bins = set(gdf_buildings["bin"])
    gdf_address = gdf_address[gdf_address["bin"].isin(selected_bins)].copy()

    print(f"[ADDRESSPOINT] Address point candidati dopo filtro BIN: {len(gdf_address)}")

    if gdf_address.empty:
        print("[ADDRESSPOINT] Nessun match BIN trovato, uso solo il centroide.")
        return gdf_buildings, False

    buildings_work = gdf_buildings.to_crs(WORK_CRS).copy()
    buildings_points = buildings_work.copy()
    buildings_points["geometry"] = buildings_work.geometry.centroid

    address_work = gdf_address.to_crs(WORK_CRS).copy()

    merged = buildings_points[["bin", "geometry"]].merge(
        address_work[["bin", "geometry"]],
        on="bin",
        how="left",
        suffixes=("_building", "_address")
    )

    merged = merged.dropna(subset=["geometry_address"]).copy()

    if merged.empty:
        print("[ADDRESSPOINT] Nessun address point associabile dopo merge BIN.")
        return gdf_buildings, False

    # Se più address point condividono lo stesso BIN,
    # scegliamo il più vicino al centroide edificio.
    merged["distance_m"] = merged.apply(
        lambda r: r["geometry_building"].distance(r["geometry_address"]),
        axis=1
    )

    best = (
        merged
        .sort_values(["bin", "distance_m"])
        .drop_duplicates(subset=["bin"], keep="first")
        .copy()
    )

    best = gpd.GeoDataFrame(best, geometry="geometry_address", crs=WORK_CRS).to_crs(OUTPUT_CRS)
    best["address_point_geom"] = best.geometry.to_wkt()
    best_geom_map = best.set_index("bin")["address_point_geom"].to_dict()

    gdf_buildings["address_point_geom"] = gdf_buildings["bin"].map(best_geom_map)
    gdf_buildings["has_address_point"] = gdf_buildings["address_point_geom"].notna()

    mask = gdf_buildings["has_address_point"]
    gdf_buildings.loc[mask, "delivery_point_geom"] = gdf_buildings.loc[mask, "address_point_geom"]
    gdf_buildings.loc[mask, "delivery_point_source"] = "addresspoint"

    delivery_series = gdf_buildings["delivery_point_geom"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) and x else None
    )
    gdf_buildings["delivery_lon"] = delivery_series.apply(lambda p: p.x if p else None)
    gdf_buildings["delivery_lat"] = delivery_series.apply(lambda p: p.y if p else None)

    print(f"[ADDRESSPOINT] Building con address point associato: {int(mask.sum())}")

    return gdf_buildings, True


def add_sidewalk_distance_to_buildings(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_sidewalks: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Per ogni edificio calcola la distanza tra delivery point/address point
    e il marciapiede piu' vicino.
    """
    gdf_buildings = gdf_buildings.copy()

    default_cols = {
        "nearest_sidewalk_source_id": None,
        "nearest_sidewalk_status": None,
        "distance_to_sidewalk_m": None,
    }

    if gdf_buildings.empty:
        for col, value in default_cols.items():
            gdf_buildings[col] = value
        return gdf_buildings

    if gdf_sidewalks is None or gdf_sidewalks.empty:
        print("[SIDEWALK] Nessun marciapiede disponibile per calcolare distanze buildings.")
        for col, value in default_cols.items():
            gdf_buildings[col] = value
        return gdf_buildings

    work = gdf_buildings.copy()
    work["delivery_geometry"] = work["delivery_point_geom"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) and x else None
    )
    work = work.dropna(subset=["delivery_geometry"]).copy()

    if work.empty:
        for col, value in default_cols.items():
            gdf_buildings[col] = value
        return gdf_buildings

    points = gpd.GeoDataFrame(
        work[["bin", "delivery_geometry"]].copy(),
        geometry="delivery_geometry",
        crs=OUTPUT_CRS
    ).to_crs(WORK_CRS)

    sidewalks = gdf_sidewalks.to_crs(WORK_CRS).copy()
    sidewalk_cols = ["geometry"]
    for col in ["source_id", "status"]:
        if col in sidewalks.columns:
            sidewalk_cols.append(col)

    nearest = gpd.sjoin_nearest(
        points,
        sidewalks[sidewalk_cols],
        how="left",
        distance_col="distance_to_sidewalk_m"
    )

    rename_map = {}
    if "source_id" in nearest.columns:
        rename_map["source_id"] = "nearest_sidewalk_source_id"
    if "status" in nearest.columns:
        rename_map["status"] = "nearest_sidewalk_status"

    nearest = nearest.rename(columns=rename_map)
    keep_cols = ["bin", "distance_to_sidewalk_m"] + list(rename_map.values())
    nearest = nearest[keep_cols].drop_duplicates(subset=["bin"], keep="first")
    if "distance_to_sidewalk_m" in nearest.columns:
        nearest["distance_to_sidewalk_m"] = nearest["distance_to_sidewalk_m"].apply(feet_to_meters)

    gdf_buildings = gdf_buildings.merge(nearest, how="left", on="bin")

    for col, value in default_cols.items():
        if col not in gdf_buildings.columns:
            gdf_buildings[col] = value

    matched = int(gdf_buildings["distance_to_sidewalk_m"].notna().sum())
    print(f"[SIDEWALK] Building con distanza al marciapiede calcolata: {matched}")

    return gdf_buildings


# ============================================================
# POPULATION / TXT OUTPUT
# ============================================================

def process_population(selection: dict) -> pd.DataFrame:
    """
    Crea un TXT con:
    - quartiere selezionato
    - popolazione

    La popolazione viene scritta senza punti o virgole.
    """
    require_file(POPULATION_CSV, "population")

    borough = clean_nta_name(selection.get("borough"))
    neighborhoods = [clean_nta_name(n) for n in selection.get("neighborhoods", [])]
    neighborhood_keys = [normalize_nta_key(n) for n in neighborhoods]

    df_pop = load_csv(POPULATION_CSV)

    required_cols = ["borough", "nta_name", "population"]
    missing_cols = [c for c in required_cols if c not in df_pop.columns]
    if missing_cols:
        raise ValueError(f"Nel file population mancano le colonne richieste: {missing_cols}")

    df_pop["borough"] = df_pop["borough"].astype(str).apply(clean_nta_name)
    df_pop["nta_name"] = df_pop["nta_name"].astype(str).apply(clean_nta_name)
    if "nta_key" not in df_pop.columns:
        df_pop["nta_key"] = df_pop["nta_name"].apply(normalize_nta_key)
    else:
        df_pop["nta_key"] = df_pop["nta_key"].apply(normalize_nta_key)

    df_pop_sel = df_pop[
        (df_pop["borough"] == borough) &
        (df_pop["nta_key"].isin(neighborhood_keys))
    ].copy()

    print(f"\n[POPULATION] Righe filtrate: {len(df_pop_sel)}")

    if "year" in df_pop_sel.columns and not df_pop_sel.empty:
        df_pop_sel["year_num"] = pd.to_numeric(df_pop_sel["year"], errors="coerce")
        df_pop_sel = (
            df_pop_sel.sort_values(["nta_name", "year_num"])
            .drop_duplicates(subset=["nta_name"], keep="last")
            .copy()
        )
        df_pop_sel = df_pop_sel.drop(columns=["year_num"], errors="ignore")

    df_pop_sel["population_clean"] = df_pop_sel["population"].apply(population_to_plain_int)
    if "nta_key" not in df_pop_sel.columns:
        df_pop_sel["nta_key"] = df_pop_sel["nta_name"].apply(normalize_nta_key)

    lines = []
    for _, row in df_pop_sel.iterrows():
        q = row["nta_name"]
        p = row["population_clean"]
        line = f"quartiere selezionato: {q}, popolazione: {p if p is not None else ''}"
        lines.append(line)

    print("[POPULATION] Nessun file intermedio salvato (cleanup output attivo).")

    return df_pop_sel


# ============================================================
# TRAFFIC
# ============================================================

def process_traffic(selection: dict) -> pd.DataFrame:
    """
    Legge traffic.csv, filtra il borough e salva:
    - OUTPUTtraffic.csv
    - OUTPUTtraffic.xlsx
    """
    require_file(TRAFFIC_CSV, "traffic")

    borough = selection.get("borough")
    if borough is None:
        raise ValueError("Nel selection manca 'borough'")

    borough = clean_nta_name(borough)

    df_traffic = load_csv(TRAFFIC_CSV)

    required_cols = ["distretto"]
    missing_cols = [c for c in required_cols if c not in df_traffic.columns]
    if missing_cols:
        raise ValueError(f"Nel file traffic mancano le colonne richieste: {missing_cols}")

    df_traffic["distretto"] = df_traffic["distretto"].astype(str).apply(clean_nta_name)

    df_filtered = df_traffic[df_traffic["distretto"] == borough].copy()
    print(f"\n[TRAFFIC] Totale righe input: {len(df_traffic)}")
    print(f"[TRAFFIC] Righe filtrate: {len(df_filtered)}")
    print("[TRAFFIC] Nessun file intermedio salvato (cleanup output attivo).")

    return df_filtered


# ============================================================
# PEDESTRIAN MOBILITY
# ============================================================

def process_pedestrian_mobility(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Carica pedestrian_mobility.csv,
    filtra borough + quartieri selezionati,
    converte in GeoDataFrame,
    salva l'output raw filtrato.

    Questo layer NON calcola score.
    Tiene solo le informazioni pedonali necessarie per essere poi
    associate alle strade.
    """
    require_file(PEDESTRIAN_CSV, "pedestrian_mobility")

    df = load_csv(PEDESTRIAN_CSV)
    print(f"\n[PEDESTRIAN] Totale righe input: {len(df)}")

    required_cols = ["boroname", "ntaname", "the_geom"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Nel file pedestrian_mobility mancano le colonne richieste: {missing_cols}")

    # Pulizia stringhe.
    df["boroname"] = df["boroname"].astype(str).apply(clean_nta_name)
    df["ntaname"] = df["ntaname"].astype(str).apply(clean_nta_name)

    # Filtro logico su borough e quartieri.
    df_filtered = df[
        (df["boroname"] == borough) &
        (df["ntaname"].isin(gdf_selected_neighborhoods["ntaname"].unique()))
    ].copy()

    # Trasformazione in layer GIS vero.
    gdf = wkt_to_gdf(df_filtered)

    # Filtro spaziale finale sull'area selezionata.
    gdf = gdf[gdf.intersects(selected_polygon)].copy()

    wanted_cols = [
        "borocode",
        "boroname",
        "borocd",
        "coundist",
        "assemdist",
        "stsendist",
        "congdist",
        "street",
        "segmentid",
        "rank",
        "pmp_id",
        "nta2020",
        "boro",
        "category",
        "ntaname",
        "femafldz",
        "femafldt",
        "hrcevac",
        "shape_leng",
        "geometry",
    ]

    existing_cols = [c for c in wanted_cols if c in gdf.columns]
    gdf = gdf[existing_cols].copy()

    print(f"[PEDESTRIAN] Righe filtrate: {len(gdf)}")
    print("[PEDESTRIAN] Nessun file intermedio salvato (cleanup output attivo).")

    return gdf


# ============================================================
# SIDEWALKS
# ============================================================

def process_sidewalks(
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Carica NYC_Sidewalk.csv, filtra i marciapiedi nell'area selezionata
    e salva un output raw filtrato.
    """
    if not SIDEWALK_CSV.exists():
        print(f"[SIDEWALK] File non trovato, salto layer marciapiedi: {SIDEWALK_CSV}")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=OUTPUT_CRS)

    df = load_csv(SIDEWALK_CSV)
    print(f"\n[SIDEWALK] Totale righe input: {len(df)}")

    if "the_geom" not in df.columns:
        raise ValueError("Nel file NYC_Sidewalk manca la colonna 'the_geom'")

    gdf = wkt_to_gdf(df)
    gdf = gdf[gdf.intersects(selected_polygon)].copy()
    gdf = add_neighborhood_info(gdf, gdf_selected_neighborhoods)

    for col in ["shape_leng", "shape_area"]:
        if col in gdf.columns:
            gdf[col] = safe_decimal_series(gdf[col])

    wanted_cols = [
        "borough",
        "neighborhood",
        "source_id",
        "sub_code",
        "feat_code",
        "status",
        "shape_leng",
        "shape_area",
        "geometry",
    ]

    existing_cols = [c for c in wanted_cols if c in gdf.columns]
    other_cols = [c for c in gdf.columns if c not in existing_cols + ["geometry"]]
    gdf = gdf[existing_cols[:-1] + other_cols + ["geometry"]].copy()

    print(f"[SIDEWALK] Marciapiedi filtrati: {len(gdf)}")
    print("[SIDEWALK] Nessun file intermedio salvato (cleanup output attivo).")

    return gdf


# ============================================================
# PARKING DATASETS
# ============================================================

def process_parking_regulations(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Carica Parking Regulation Locations and Signs,
    filtra borough + area selezionata,
    associa quartiere
    e salva output.
    """
    require_file(PARKING_REGULATIONS_CSV, "parking_regulations")

    df = load_csv(PARKING_REGULATIONS_CSV)
    print(f"\n[PARKING REG] Totale righe: {len(df)}")

    required_cols = ["borough", "sign_x_coord", "sign_y_coord"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Nel file parking regulations mancano le colonne: {missing}")

    df["borough"] = df["borough"].astype(str).apply(clean_nta_name)
    df = df[df["borough"] == borough].copy()
    print(f"[PARKING REG] Righe nel borough selezionato: {len(df)}")

    gdf = xy_to_gdf(df, "sign_x_coord", "sign_y_coord", source_crs=WORK_CRS, target_crs=OUTPUT_CRS)
    gdf = gdf[gdf.intersects(selected_polygon)].copy()
    print(f"[PARKING REG] Segnali nell'area selezionata: {len(gdf)}")

    joined = gpd.sjoin(
        gdf,
        gdf_selected_neighborhoods[["boroname", "ntaname", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(columns={"boroname": "borough_nta", "ntaname": "neighborhood"})

    keep_cols = [
        "borough", "borough_nta", "neighborhood",
        "on_street", "from_street", "to_street", "side_of_street",
        "sign_code", "sign_description", "order_number",
        "distance_from_intersection", "arrow_direction", "facing_direction",
        "sign_x_coord", "sign_y_coord", "geometry"
    ]
    existing = [c for c in keep_cols if c in joined.columns]
    joined = joined[existing].copy()


    return joined


def process_parking_meters(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Carica Parking Meters Locations and Status,
    tiene solo Active + On Street,
    filtra borough + area selezionata,
    associa quartiere
    e salva output.
    """
    require_file(PARKING_METERS_CSV, "parking_meters")

    df = load_csv(PARKING_METERS_CSV)
    print(f"\n[PARKING METERS] Totale righe: {len(df)}")

    required_cols = ["borough", "status", "facility", "longitude", "latitude"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Nel file parking meters mancano le colonne: {missing}")

    df["borough"] = df["borough"].astype(str).apply(clean_nta_name)
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["facility"] = df["facility"].astype(str).str.strip().str.lower()

    df = df[
        (df["borough"] == borough) &
        (df["status"] == "active") &
        (df["facility"] == "on street")
    ].copy()

    print(f"[PARKING METERS] Righe filtrate (borough + active + on street): {len(df)}")

    df["longitude"] = safe_numeric_series(df["longitude"])
    df["latitude"] = safe_numeric_series(df["latitude"])
    df = df.dropna(subset=["longitude", "latitude"]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=OUTPUT_CRS
    )

    gdf = gdf[gdf.intersects(selected_polygon)].copy()
    print(f"[PARKING METERS] Meter nell'area selezionata: {len(gdf)}")

    joined = gpd.sjoin(
        gdf,
        gdf_selected_neighborhoods[["boroname", "ntaname", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(columns={"boroname": "borough_nta", "ntaname": "neighborhood"})

    keep_cols = [
        "borough", "borough_nta", "neighborhood",
        "meter_number", "status", "pay_by_cell_number", "meter_hours",
        "facility", "on_street", "side_of_street", "from_street", "to_street",
        "latitude", "longitude", "geometry"
    ]
    existing = [c for c in keep_cols if c in joined.columns]
    joined = joined[existing].copy()


    return joined


def process_parking_blockfaces(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Carica ParkNYC Block Faces,
    filtra borough + area selezionata,
    associa quartiere
    e salva output.
    """
    require_file(PARKING_BLOCKFACES_CSV, "parking_blockfaces")

    df = load_csv(PARKING_BLOCKFACES_CSV)
    print(f"\n[PARKING BLOCKFACES] Totale righe: {len(df)}")

    required_cols = ["borough", "the_geom"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Nel file parking blockfaces mancano le colonne: {missing}")

    df["borough"] = df["borough"].astype(str).apply(clean_nta_name)
    df = df[df["borough"] == borough].copy()
    print(f"[PARKING BLOCKFACES] Righe nel borough selezionato: {len(df)}")

    gdf = wkt_to_gdf(df)
    gdf = gdf[gdf.intersects(selected_polygon)].copy()
    print(f"[PARKING BLOCKFACES] Block faces nell'area selezionata: {len(gdf)}")

    joined = gpd.sjoin(
        gdf,
        gdf_selected_neighborhoods[["boroname", "ntaname", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(columns={"boroname": "borough_nta", "ntaname": "neighborhood"})

    keep_cols = [
        "borough", "borough_nta", "neighborhood",
        "pay_by_cel", "vehicle_ty",
        "all_vehicl", "all_vehi_1", "all_vehi_2", "all_vehi_3",
        "commercial", "commerci_1", "commerci_2", "commerci_3",
        "on_street", "side_of_st", "from_stree", "to_street",
        "meter_rate", "shape_leng", "geometry"
    ]
    existing = [c for c in keep_cols if c in joined.columns]
    joined = joined[existing].copy()


    return joined


def process_parking_ratezones(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Carica Citywide Rate Zones,
    filtra borough + area selezionata,
    associa quartiere
    e salva output.
    """
    require_file(PARKING_RATEZONES_CSV, "parking_ratezones")

    df = load_csv(PARKING_RATEZONES_CSV)
    print(f"\n[PARKING RATEZONES] Totale righe: {len(df)}")

    required_cols = ["boro_name", "the_geom"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Nel file parking rate zones mancano le colonne: {missing}")

    df["boro_name"] = df["boro_name"].astype(str).apply(clean_nta_name)
    df = df[df["boro_name"] == borough].copy()
    print(f"[PARKING RATEZONES] Righe nel borough selezionato: {len(df)}")

    gdf = wkt_to_gdf(df)
    gdf = gdf[gdf.intersects(selected_polygon)].copy()
    print(f"[PARKING RATEZONES] Zone nell'area selezionata: {len(gdf)}")

    joined = gpd.sjoin(
        gdf,
        gdf_selected_neighborhoods[["boroname", "ntaname", "geometry"]],
        how="left",
        predicate="intersects",
    )
    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(columns={"boroname": "borough_nta", "ntaname": "neighborhood"})

    keep_cols = [
        "boro_name", "borough_nta", "neighborhood",
        "zone", "zone_name", "rate_zone",
        "shape_leng", "shape_le_1", "shape_area", "geometry"
    ]
    existing = [c for c in keep_cols if c in joined.columns]
    joined = joined[existing].copy()


    return joined


# ============================================================
# STREET CONTEXT
# ============================================================

def process_street_context(
    gdf_streets: gpd.GeoDataFrame,
    gdf_parking_reg: gpd.GeoDataFrame,
    gdf_parking_meters: gpd.GeoDataFrame,
    gdf_parking_blockfaces: gpd.GeoDataFrame,
    gdf_parking_ratezones: gpd.GeoDataFrame,
    gdf_pedestrian: gpd.GeoDataFrame,
    gdf_sidewalks: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Crea un output unico a livello strada:
    - contesto parcheggio
    - rank pedonale
    - presenza/raggiungibilita' marciapiede

    Una riga = una street centerline filtrata.

    Non vengono creati score. Vengono solo associate informazioni raw
    alle strade per i passaggi successivi.
    """
    if gdf_streets.empty:
        return gdf_streets.copy()

    streets = gdf_streets.to_crs(WORK_CRS).copy()
    streets["street_context_id"] = range(1, len(streets) + 1)

    # ---------------------------------------------------------
    # BUFFER DELLE STRADE
    # ---------------------------------------------------------
    # Il buffer serve per associare punti vicini (meter e regulation signs)
    # ai segmenti stradali.
    street_buffer = streets[["street_context_id", "geometry"]].copy()
    street_buffer["geometry"] = street_buffer.geometry.buffer(PARKING_BUFFER_FT)

    # ---------------------------------------------------------
    # REGULATION SIGNS COUNT
    # ---------------------------------------------------------
    reg_counts = pd.DataFrame(columns=["street_context_id", "n_regulation_signs"])
    if not gdf_parking_reg.empty:
        regs = gdf_parking_reg.to_crs(WORK_CRS).copy()
        if "sign_description" in regs.columns:
            regs = regs[regs["sign_description"].apply(is_curb_restriction_sign)].copy()

        reg_join = gpd.sjoin(
            regs[["geometry"]],
            street_buffer,
            how="left",
            predicate="within"
        )
        reg_counts = (
            reg_join.dropna(subset=["street_context_id"])
            .groupby("street_context_id")
            .size()
            .reset_index(name="n_regulation_signs")
        )

    # ---------------------------------------------------------
    # ACTIVE METERS COUNT
    # ---------------------------------------------------------
    meter_counts = pd.DataFrame(columns=["street_context_id", "n_active_meters"])
    if not gdf_parking_meters.empty:
        meters = gdf_parking_meters.to_crs(WORK_CRS).copy()
        meter_join = gpd.sjoin(
            meters[["geometry"]],
            street_buffer,
            how="left",
            predicate="within"
        )
        meter_counts = (
            meter_join.dropna(subset=["street_context_id"])
            .groupby("street_context_id")
            .size()
            .reset_index(name="n_active_meters")
        )

    # ---------------------------------------------------------
    # NEAREST BLOCKFACE
    # ---------------------------------------------------------
    nearest_bf = pd.DataFrame()
    if not gdf_parking_blockfaces.empty:
        bf = gdf_parking_blockfaces.to_crs(WORK_CRS).copy()

        bf_cols = ["geometry"]
        if "vehicle_ty" in bf.columns:
            bf_cols.append("vehicle_ty")
        if "meter_rate" in bf.columns:
            bf_cols.append("meter_rate")
        if "on_street" in bf.columns:
            bf_cols.append("on_street")
        if "from_stree" in bf.columns:
            bf_cols.append("from_stree")
        if "to_street" in bf.columns:
            bf_cols.append("to_street")

        nearest_bf = gpd.sjoin_nearest(
            streets[["street_context_id", "geometry"]],
            bf[bf_cols],
            how="left",
            distance_col="bf_distance_m"
        )

        rename_map = {}
        if "vehicle_ty" in nearest_bf.columns:
            rename_map["vehicle_ty"] = "bf_vehicle_ty"
        if "meter_rate" in nearest_bf.columns:
            rename_map["meter_rate"] = "bf_meter_rate"
        if "on_street" in nearest_bf.columns:
            rename_map["on_street"] = "bf_on_street"
        if "from_stree" in nearest_bf.columns:
            rename_map["from_stree"] = "bf_from_street"
        if "to_street" in nearest_bf.columns:
            rename_map["to_street"] = "bf_to_street"

        nearest_bf = nearest_bf.rename(columns=rename_map)

        keep_cols = ["street_context_id", "bf_distance_m"] + list(rename_map.values())
        nearest_bf = nearest_bf[keep_cols].copy()
        if "bf_distance_m" in nearest_bf.columns:
            nearest_bf["bf_distance_m"] = nearest_bf["bf_distance_m"].apply(feet_to_meters)

    # ---------------------------------------------------------
    # RATE ZONES
    # ---------------------------------------------------------
    # Per assegnare una rate zone alla strada usiamo il centroide del segmento.
    rate_info = pd.DataFrame()
    if not gdf_parking_ratezones.empty:
        ratez = gdf_parking_ratezones.to_crs(WORK_CRS).copy()

        street_cent = streets[["street_context_id", "geometry"]].copy()
        street_cent["geometry"] = street_cent.geometry.centroid

        rate_info = gpd.sjoin(
            street_cent,
            ratez[[c for c in ["rate_zone", "zone", "zone_name", "geometry"] if c in ratez.columns]],
            how="left",
            predicate="intersects"
        )

        keep_cols = ["street_context_id"] + [c for c in ["rate_zone", "zone", "zone_name"] if c in rate_info.columns]
        rate_info = rate_info[keep_cols].drop_duplicates(subset=["street_context_id"]).copy()

    # ---------------------------------------------------------
    # PEDESTRIAN RANK
    # ---------------------------------------------------------
    # Associa a ogni strada il segmento pedonale più vicino.
    ped_info = pd.DataFrame()
    if not gdf_pedestrian.empty:
        ped = gdf_pedestrian.to_crs(WORK_CRS).copy()

        ped_cols = ["geometry"]
        for c in ["street", "segmentid", "rank", "category", "ntaname"]:
            if c in ped.columns:
                ped_cols.append(c)

        ped_info = gpd.sjoin_nearest(
            streets[["street_context_id", "geometry"]],
            ped[ped_cols],
            how="left",
            distance_col="ped_distance_m"
        )

        rename_map = {}
        if "street" in ped_info.columns:
            rename_map["street"] = "ped_street"
        if "segmentid" in ped_info.columns:
            rename_map["segmentid"] = "ped_segmentid"
        if "rank" in ped_info.columns:
            rename_map["rank"] = "ped_rank"
        if "category" in ped_info.columns:
            rename_map["category"] = "ped_category"
        if "ntaname" in ped_info.columns:
            rename_map["ntaname"] = "ped_nta"

        ped_info = ped_info.rename(columns=rename_map)

        keep_cols = ["street_context_id", "ped_distance_m"] + list(rename_map.values())
        ped_info = ped_info[keep_cols].copy()
        if "ped_distance_m" in ped_info.columns:
            ped_info["ped_distance_m"] = ped_info["ped_distance_m"].apply(feet_to_meters)

    # ---------------------------------------------------------
    # SIDEWALKS
    # ---------------------------------------------------------
    sidewalk_info = pd.DataFrame()
    if gdf_sidewalks is not None and not gdf_sidewalks.empty:
        sidewalks = gdf_sidewalks.to_crs(WORK_CRS).copy()

        sidewalk_cols = ["geometry"]
        for c in ["source_id", "status", "shape_leng", "shape_area"]:
            if c in sidewalks.columns:
                sidewalk_cols.append(c)

        sidewalk_join = gpd.sjoin(
            sidewalks[sidewalk_cols],
            street_buffer,
            how="left",
            predicate="intersects"
        )
        sidewalk_join = sidewalk_join.dropna(subset=["street_context_id"]).copy()

        if not sidewalk_join.empty:
            agg_map = {"geometry": "count"}
            if "shape_leng" in sidewalk_join.columns:
                agg_map["shape_leng"] = "sum"
            if "shape_area" in sidewalk_join.columns:
                agg_map["shape_area"] = "sum"

            sidewalk_info = (
                sidewalk_join
                .groupby("street_context_id")
                .agg(agg_map)
                .reset_index()
                .rename(columns={
                    "geometry": "n_sidewalk_polygons",
                    "shape_leng": "sidewalk_length",
                    "shape_area": "sidewalk_area",
                })
            )
            sidewalk_info["has_sidewalk"] = sidewalk_info["n_sidewalk_polygons"] > 0

        nearest_sidewalk = gpd.sjoin_nearest(
            streets[["street_context_id", "geometry"]],
            sidewalks[sidewalk_cols],
            how="left",
            distance_col="sidewalk_distance_m"
        )

        rename_map = {}
        if "source_id" in nearest_sidewalk.columns:
            rename_map["source_id"] = "nearest_sidewalk_source_id"
        if "status" in nearest_sidewalk.columns:
            rename_map["status"] = "nearest_sidewalk_status"

        nearest_sidewalk = nearest_sidewalk.rename(columns=rename_map)
        keep_cols = ["street_context_id", "sidewalk_distance_m"] + list(rename_map.values())
        nearest_sidewalk = nearest_sidewalk[keep_cols].drop_duplicates(
            subset=["street_context_id"],
            keep="first"
        )

        if sidewalk_info.empty:
            sidewalk_info = nearest_sidewalk.copy()
        else:
            sidewalk_info = sidewalk_info.merge(nearest_sidewalk, how="left", on="street_context_id")

        if "has_sidewalk" not in sidewalk_info.columns:
            sidewalk_info["has_sidewalk"] = False
        if "n_sidewalk_polygons" not in sidewalk_info.columns:
            sidewalk_info["n_sidewalk_polygons"] = 0
        if "sidewalk_length" not in sidewalk_info.columns:
            sidewalk_info["sidewalk_length"] = 0
        if "sidewalk_area" not in sidewalk_info.columns:
            sidewalk_info["sidewalk_area"] = 0

        if "sidewalk_distance_m" in sidewalk_info.columns:
            sidewalk_info["sidewalk_distance_m"] = sidewalk_info["sidewalk_distance_m"].apply(feet_to_meters)

        sidewalk_info["sidewalk_reachable"] = (
            sidewalk_info["sidewalk_distance_m"].notna()
            & (sidewalk_info["sidewalk_distance_m"] <= SIDEWALK_REACHABLE_DISTANCE_M)
        )

    # ---------------------------------------------------------
    # MERGE FINALE SU STRADA
    # ---------------------------------------------------------
    out = streets.copy()

    if not reg_counts.empty:
        out = out.merge(reg_counts, how="left", on="street_context_id")
    else:
        out["n_regulation_signs"] = 0

    if not meter_counts.empty:
        out = out.merge(meter_counts, how="left", on="street_context_id")
    else:
        out["n_active_meters"] = 0

    if not nearest_bf.empty:
        out = out.merge(nearest_bf, how="left", on="street_context_id")

    if not rate_info.empty:
        out = out.merge(rate_info, how="left", on="street_context_id")

    if not ped_info.empty:
        out = out.merge(ped_info, how="left", on="street_context_id")

    if not sidewalk_info.empty:
        out = out.merge(sidewalk_info, how="left", on="street_context_id")
    else:
        out["has_sidewalk"] = False
        out["sidewalk_reachable"] = False
        out["n_sidewalk_polygons"] = 0
        out["sidewalk_length"] = 0
        out["sidewalk_area"] = 0
        out["sidewalk_distance_m"] = None

    # Fill per conteggi numerici.
    if "n_regulation_signs" in out.columns:
        out["n_regulation_signs"] = out["n_regulation_signs"].fillna(0).astype(int)

    if "n_active_meters" in out.columns:
        out["n_active_meters"] = out["n_active_meters"].fillna(0).astype(int)

    if "has_sidewalk" in out.columns:
        out["has_sidewalk"] = out["has_sidewalk"].fillna(False).astype(bool)

    if "sidewalk_reachable" in out.columns:
        out["sidewalk_reachable"] = out["sidewalk_reachable"].fillna(False).astype(bool)

    if "n_sidewalk_polygons" in out.columns:
        out["n_sidewalk_polygons"] = out["n_sidewalk_polygons"].fillna(0).astype(int)

    for col in ["sidewalk_length", "sidewalk_area"]:
        if col in out.columns:
            out[col] = out[col].fillna(0)

    out = out.to_crs(OUTPUT_CRS)

    preferred_cols = [
        "street_context_id",
        "borough",
        "neighborhood",
        "n_regulation_signs",
        "n_active_meters",
        "bf_vehicle_ty",
        "bf_meter_rate",
        "bf_on_street",
        "bf_from_street",
        "bf_to_street",
        "bf_distance_m",
        "rate_zone",
        "zone",
        "zone_name",
        "ped_street",
        "ped_segmentid",
        "ped_rank",
        "ped_category",
        "ped_nta",
        "ped_distance_m",
        "has_sidewalk",
        "sidewalk_reachable",
        "n_sidewalk_polygons",
        "sidewalk_length",
        "sidewalk_area",
        "sidewalk_distance_m",
        "nearest_sidewalk_source_id",
        "nearest_sidewalk_status",
    ]

    existing = [c for c in preferred_cols if c in out.columns]
    other_cols = [c for c in out.columns if c not in existing + ["geometry"]]
    out = out[existing + other_cols + ["geometry"]]

    if SAVE_OUTPUTS:
        save_gdf_outputs(
            out,
            geojson_path=OUTPUTstreet_context_GEOJSON,
        )

    print(f"\n[STREET CONTEXT] Strade con contesto parcheggio + pedonale create: {len(out)}")

    return out


# ============================================================
# BUILDINGS PIPELINE
# ============================================================

def process_buildings(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon,
    gdf_sidewalks: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, dict[str, bool]]:
    """
    Pipeline buildings:
    1. carica BUILDING
    2. filtra borough
    3. filtra solo edifici consegnabili
    4. filtra geometria sull'area selezionata
    5. aggiunge quartiere
    6. aggiunge centroidi
    7. merge PLUTO
    8. merge AddressPoint / delivery point
    9. distanza delivery point -> marciapiede
    10. salva output
    """
    require_file(BUILDING_CSV, "BUILDING")

    df_building = load_csv(BUILDING_CSV)
    print(f"\n[BUILDINGS] Totale righe: {len(df_building)}")

    df_building = prepare_buildings(df_building)
    borough_prefix_bin = BOROUGH_TO_BIN_PREFIX[borough]

    df_building_borough = filter_buildings_by_borough(df_building, borough_prefix_bin)
    print(f"[BUILDINGS] Righe nel borough selezionato: {len(df_building_borough)}")

    df_building_valid = filter_deliverable_buildings(df_building_borough)
    print(f"[BUILDINGS] Edifici consegnabili nel borough: {len(df_building_valid)}")

    gdf_building = wkt_to_gdf(df_building_valid)

    gdf_building_area = gdf_building[gdf_building.intersects(selected_polygon)].copy()
    print(f"[BUILDINGS] Edifici nell'area selezionata: {len(gdf_building_area)}")

    gdf_building_area = add_neighborhood_info(gdf_building_area, gdf_selected_neighborhoods)
    gdf_building_area = add_centroid_columns(gdf_building_area)
    gdf_building_area, pluto_used = enrich_with_pluto(gdf_building_area)
    gdf_building_area, addresspoint_used = add_address_points(gdf_building_area)
    gdf_building_area = add_sidewalk_distance_to_buildings(gdf_building_area, gdf_sidewalks)

    preferred_cols = [
        "borough", "neighborhood", "bin", "map_pluto_bbl",
        "height_roof", "numfloors", "landuse", "bldgclass",
        "centroid_geom", "centroid_lon", "centroid_lat",
        "address_point_geom", "has_address_point",
        "delivery_point_geom", "delivery_point_source",
        "delivery_lon", "delivery_lat",
        "distance_to_sidewalk_m", "nearest_sidewalk_source_id",
        "nearest_sidewalk_status"
    ]

    existing = [c for c in preferred_cols if c in gdf_building_area.columns]
    other_cols = [c for c in gdf_building_area.columns if c not in existing + ["geometry"]]
    gdf_building_area = gdf_building_area[existing + other_cols + ["geometry"]]

    if SAVE_OUTPUTS:
        save_gdf_outputs(
            gdf_building_area,
            geojson_path=OUTPUTbuildings_GEOJSON,
        )

    return gdf_building_area, {
        "pluto_used": pluto_used,
        "addresspoint_used": addresspoint_used
    }


# ============================================================
# STREETS PIPELINE
# ============================================================

def prepare_streets(df_street: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara il dataset strade:
    - borough_code numerico
    - trafdir normalizzato
    """
    df = df_street.copy()
    df["borough_code"] = pd.to_numeric(df["borough_code"], errors="coerce")

    if "trafdir" in df.columns:
        df["trafdir"] = df["trafdir"].astype(str).str.strip().str.upper()

    return df


def filter_streets_by_borough(df_street: pd.DataFrame, borough_code: int) -> pd.DataFrame:
    """
    Filtra segmenti stradali per borough.
    """
    return df_street[df_street["borough_code"] == borough_code].copy()


def process_streets(
    borough: str,
    gdf_selected_neighborhoods: gpd.GeoDataFrame,
    selected_polygon
) -> gpd.GeoDataFrame:
    """
    Pipeline strade:
    1. carica Centerline
    2. filtra borough
    3. filtra area selezionata
    4. associa quartiere
    5. salva output
    """
    require_file(CENTERLINE_CSV, "Centerline")

    df_street = load_csv(CENTERLINE_CSV)
    print(f"\n[STREETS] Totale righe: {len(df_street)}")

    df_street = prepare_streets(df_street)
    borough_code = BOROUGH_TO_CODE[borough]

    df_street_borough = filter_streets_by_borough(df_street, borough_code)
    print(f"[STREETS] Righe nel borough selezionato: {len(df_street_borough)}")

    gdf_street = wkt_to_gdf(df_street_borough)
    gdf_street_area = gdf_street[gdf_street.intersects(selected_polygon)].copy()
    print(f"[STREETS] Strade nell'area selezionata: {len(gdf_street_area)}")

    joined = gpd.sjoin(
        gdf_street_area,
        gdf_selected_neighborhoods[["boroname", "ntaname", "geometry"]],
        how="left",
        predicate="intersects",
    )

    joined = joined.drop(columns=[c for c in ["index_right"] if c in joined.columns])
    joined = joined.rename(columns={"boroname": "borough", "ntaname": "neighborhood"})


    return joined


# ============================================================
# SERVICE POINTS
# ============================================================

def export_service_points(gdf_buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Esporta i service points come layer separato.
    La geometria del layer diventa il delivery_point_geom.
    """
    work = gdf_buildings.copy()

    work["geometry"] = work["delivery_point_geom"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) and x else Point()
    )

    gdf_points = gpd.GeoDataFrame(work, geometry="geometry", crs=OUTPUT_CRS)
    gdf_points = gdf_points[gdf_points.geometry.is_valid & ~gdf_points.geometry.is_empty].copy()


    return gdf_points


# ============================================================
# AMR FEATURES
# ============================================================

def minmax_norm(series: pd.Series, inverse: bool = False) -> pd.Series:
    """
    Normalizza una serie tra 0 e 1 sul filtro corrente.
    Se inverse=True, valori alti diventano bassi e viceversa.
    """
    values = pd.to_numeric(series, errors="coerce")
    min_val = values.min(skipna=True)
    max_val = values.max(skipna=True)

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        norm = pd.Series(0.0, index=series.index)
    else:
        norm = (values - min_val) / (max_val - min_val)

    norm = norm.fillna(0.0).clip(0.0, 1.0)
    return 1.0 - norm if inverse else norm


def pedestrian_presence_value(value: Any) -> float:
    """
    Proxy 0-1 della domanda pedonale da categoria NYC DOT.
    """
    text = str(value).strip().lower()
    mapping = {
        "baseline street": 0.00,
        "community connector": 0.25,
        "neighborhood corridor": 0.50,
        "regional corridor": 0.75,
        "global corridor": 1.00,
        "community": 0.25,
        "neighborhood": 0.50,
        "regional": 0.75,
        "global": 1.00,
    }
    if text in mapping:
        return mapping[text]

    rank = pd.to_numeric(value, errors="coerce")
    if pd.isna(rank):
        return 0.0

    # Nei tuoi output rank piu' basso indica corridoio piu' importante.
    return float((5.0 - max(1.0, min(5.0, rank))) / 4.0)


def landuse_activity_value(value: Any) -> float:
    """
    Proxy semplice di attivita' urbana da landuse PLUTO.
    """
    code = str(value).strip().replace(".0", "")
    mapping = {
        "1": 0.30,  # one/two family
        "2": 0.45,  # multi-family walk-up
        "3": 0.50,  # multi-family elevator
        "4": 0.90,  # mixed commercial/residential
        "5": 1.00,  # commercial/office
        "6": 0.80,  # industrial/manufacturing
        "7": 0.70,  # transportation/utility
        "8": 0.75,  # public facilities
        "9": 0.20,  # open space
        "10": 0.10,  # parking facilities/vacant
        "11": 0.10,
    }
    return mapping.get(code, 0.40)


def building_type_penalty_value(value: Any) -> float:
    """
    Rule-based building-type penalty aligned with the simulation/training logic.
    """
    code = str(value).strip().replace(".0", "")
    if code in {"4", "5"}:
        return 0.00  # mixed_commercial
    if code in {"1", "2", "3"}:
        return 0.33  # residential
    if code in {"6", "7"}:
        return 0.66  # industrial_utility
    return 1.00      # public_other / fallback


def estimate_shape_penalty_geometry(geom: Any) -> float:
    """
    Shape penalty aligned with the simulation logic:
    elongated buildings get higher penalty, compact buildings lower penalty.
    """
    polygon: Polygon | None = None
    if geom is None:
        return 1.0
    if isinstance(geom, Polygon):
        polygon = geom
    elif isinstance(geom, MultiPolygon):
        try:
            polygon = max(list(geom.geoms), key=lambda g: g.area)
        except Exception:
            polygon = None
    if polygon is None or polygon.is_empty:
        return 1.0

    minimum_rotated = polygon.minimum_rotated_rectangle
    rect_coords = list(minimum_rotated.exterior.coords)[:-1]
    if len(rect_coords) != 4:
        return 1.0

    lengths = []
    for a, b in zip(rect_coords, rect_coords[1:] + rect_coords[:1]):
        lengths.append(Point(a).distance(Point(b)))
    lengths = sorted(lengths, reverse=True)
    long_axis = lengths[0] if lengths else 1.0
    short_axis = lengths[-1] if lengths else 1.0
    if long_axis <= 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - (short_axis / long_axis)))


def nearest_street_context_for_buildings(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_street_context: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Associa ogni edificio alla strada di contesto piu' vicina usando il delivery point.
    """
    if gdf_buildings.empty or gdf_street_context.empty:
        return pd.DataFrame()

    work = gdf_buildings.copy()
    work["delivery_geometry"] = work["delivery_point_geom"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) and x else None
    )
    work = work.dropna(subset=["delivery_geometry"]).copy()

    if work.empty:
        return pd.DataFrame()

    points = gpd.GeoDataFrame(
        work[["bin", "delivery_geometry"]].copy(),
        geometry="delivery_geometry",
        crs=OUTPUT_CRS
    ).to_crs(WORK_CRS)

    streets = gdf_street_context.to_crs(WORK_CRS).copy()
    street_cols = [
        "street_context_id",
        "neighborhood",
        "ped_rank",
        "ped_category",
        "has_sidewalk",
        "sidewalk_reachable",
        "sidewalk_distance_m",
        "n_active_meters",
        "n_regulation_signs",
        "number_park_lanes",
        "number_travel_lanes",
        "posted_speed",
        "bf_vehicle_ty",
        "bf_meter_rate",
        "geometry",
    ]
    street_cols = [c for c in street_cols if c in streets.columns]

    joined = gpd.sjoin_nearest(
        points,
        streets[street_cols],
        how="left",
        distance_col="building_to_street_distance"
    )

    return pd.DataFrame(joined.drop(columns=["geometry"], errors="ignore"))


def export_amr_features(
    gdf_buildings: gpd.GeoDataFrame,
    gdf_street_context: gpd.GeoDataFrame,
    df_population: pd.DataFrame,
    df_traffic: pd.DataFrame,
) -> pd.DataFrame:
    """
    Costruisce il workbook OUTPUTamr_features.xlsx con due fogli:
    - amr_features
    - car_features

    La versione corrente e' volutamente minimale e genera soltanto le
    feature ancora usate dal flusso last-meter, dal training e dal dataset finale.
    """
    if gdf_buildings.empty:
        df_empty = pd.DataFrame(columns=["bin"])
        OUTPUTamr_features_XLSX.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(OUTPUTamr_features_XLSX) as writer:
            df_empty.to_excel(writer, index=False, sheet_name="amr_features")
            df_empty.to_excel(writer, index=False, sheet_name="car_features")
        return df_empty

    shape_penalty_df = pd.DataFrame({
        "bin": pd.to_numeric(gdf_buildings["bin"], errors="coerce").astype("Int64"),
        "ShapePenalty_norm": gdf_buildings["geometry"].apply(estimate_shape_penalty_geometry),
    }).dropna(subset=["bin"]).drop_duplicates(subset=["bin"])

    buildings = pd.DataFrame(gdf_buildings.drop(columns=["geometry"], errors="ignore")).copy()
    buildings["bin"] = pd.to_numeric(buildings["bin"], errors="coerce").astype("Int64")
    buildings = buildings.dropna(subset=["bin"]).drop_duplicates(subset=["bin"]).reset_index(drop=True)
    buildings = buildings.merge(shape_penalty_df, on="bin", how="left")

    street_match = nearest_street_context_for_buildings(gdf_buildings, gdf_street_context)
    if not street_match.empty:
        street_match["bin"] = pd.to_numeric(street_match["bin"], errors="coerce").astype("Int64")
        street_match = street_match.dropna(subset=["bin"]).drop_duplicates(subset=["bin"]).reset_index(drop=True)
    else:
        street_match = pd.DataFrame(columns=["bin"])

    work = buildings.merge(street_match, how="left", on="bin", suffixes=("", "_street"))

    work["nta_key_join"] = pd.Series(index=work.index, dtype=object)
    if "nta_key" in work.columns:
        work["nta_key_join"] = work["nta_key"].apply(normalize_nta_key)
    elif "neighborhood" in work.columns:
        work["nta_key_join"] = work["neighborhood"].apply(normalize_nta_key)

    if not df_population.empty:
        pop = df_population.copy()
        if "population_clean" not in pop.columns and "population" in pop.columns:
            pop["population_clean"] = pop["population"].apply(population_to_plain_int)
        if "borough" in pop.columns:
            pop["borough"] = pop["borough"].astype(str).apply(clean_nta_name)
        if "nta_key" not in pop.columns:
            if "nta_name" in pop.columns:
                pop["nta_key"] = pop["nta_name"].apply(normalize_nta_key)
            elif "neighborhood" in pop.columns:
                pop["nta_key"] = pop["neighborhood"].apply(normalize_nta_key)
        pop = pop[[c for c in ["nta_key", "population_clean"] if c in pop.columns]].drop_duplicates(subset=["nta_key"])
        work = work.merge(pop, how="left", left_on="nta_key_join", right_on="nta_key")

        # Fallback: se manca il match NTA, usa la media della popolazione del borough.
        if "borough" in work.columns and "borough" in df_population.columns:
            borough_pop = df_population.copy()
            if "population_clean" not in borough_pop.columns and "population" in borough_pop.columns:
                borough_pop["population_clean"] = borough_pop["population"].apply(population_to_plain_int)
            borough_pop["borough"] = borough_pop["borough"].astype(str).apply(clean_nta_name)
            borough_mean = (
                borough_pop.groupby("borough", dropna=False)["population_clean"]
                .mean()
                .to_dict()
            )
            missing_mask = pd.to_numeric(work.get("population_clean"), errors="coerce").isna()
            if missing_mask.any():
                work.loc[missing_mask, "population_clean"] = (
                    work.loc[missing_mask, "borough"]
                    .astype(str)
                    .apply(clean_nta_name)
                    .map(borough_mean)
                )
    else:
        work["population_clean"] = None

    work["raw_numfloors"] = pd.to_numeric(work.get("numfloors"), errors="coerce")
    work["raw_distance_to_sidewalk_m"] = pd.to_numeric(work.get("distance_to_sidewalk_m"), errors="coerce")
    work["raw_n_active_meters"] = pd.to_numeric(work.get("n_active_meters"), errors="coerce").fillna(0.0)
    work["raw_n_regulation_signs"] = pd.to_numeric(work.get("n_regulation_signs"), errors="coerce").fillna(0.0)
    work["raw_number_park_lanes"] = pd.to_numeric(work.get("number_park_lanes"), errors="coerce").fillna(0.0)
    work["raw_bf_vehicle_ty"] = work.get("bf_vehicle_ty", pd.Series(index=work.index, dtype=object)).fillna("").astype(str)

    work["a1_Floors_norm"] = minmax_norm(work["raw_numfloors"])
    work["a2_RoadToDeliveryDistance_norm"] = minmax_norm(pd.to_numeric(work.get("building_to_street_distance"), errors="coerce"))
    work["a3_AddressUncertainty"] = (~work.get("has_address_point", False).fillna(False)).astype(float)

    work["b1_Population_norm"] = minmax_norm(pd.to_numeric(work.get("population_clean"), errors="coerce"))
    work["b2_PedestrianPresence_norm"] = work.get("ped_category", pd.Series(index=work.index, dtype=object)).apply(pedestrian_presence_value)
    work["b3_UrbanActivity_norm"] = work.get("landuse", pd.Series(index=work.index, dtype=object)).apply(landuse_activity_value)
    work["BuildingTypePenalty_norm"] = work.get("landuse", pd.Series(index=work.index, dtype=object)).apply(building_type_penalty_value)
    work["ShapePenalty_norm"] = pd.to_numeric(work.get("ShapePenalty_norm"), errors="coerce").fillna(1.0)

    work["c1_PedestrianPenalty"] = (1.0 - pd.to_numeric(work["b2_PedestrianPresence_norm"], errors="coerce").fillna(0.0)).clip(0.0, 1.0)
    work["c2_SidewalkAbsencePenalty"] = (~work.get("sidewalk_reachable", False).fillna(False)).astype(float)

    work["ParkingScarcity_advantage"] = minmax_norm(work["raw_n_active_meters"], inverse=True)
    work["CurbRestriction_advantage"] = minmax_norm(work["raw_n_regulation_signs"])
    work["CommercialCurbContext"] = work["raw_bf_vehicle_ty"].str.upper().str.contains("COMMERCIAL|TRUCK").astype(float)
    work["raw_curb_crowding_sum"] = (
        pd.to_numeric(work["ParkingScarcity_advantage"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["CurbRestriction_advantage"], errors="coerce").fillna(0.0)
        + pd.to_numeric(work["CommercialCurbContext"], errors="coerce").fillna(0.0)
    )
    work["CurbCrowdingPenalty"] = (work["raw_curb_crowding_sum"] / 3.0).clip(0.0, 1.0)

    amr_sheet_cols = [
        "bin",
        "raw_numfloors",
        "raw_distance_to_sidewalk_m",
        "landuse",
        "a1_Floors_norm",
        "a2_RoadToDeliveryDistance_norm",
        "a3_AddressUncertainty",
        "ShapePenalty_norm",
        "BuildingTypePenalty_norm",
        "b1_Population_norm",
        "b2_PedestrianPresence_norm",
        "b3_UrbanActivity_norm",
        "c1_PedestrianPenalty",
        "c2_SidewalkAbsencePenalty",
    ]
    car_sheet_cols = [
        "bin",
        "raw_numfloors",
        "raw_distance_to_sidewalk_m",
        "landuse",
        "a1_Floors_norm",
        "a2_RoadToDeliveryDistance_norm",
        "ShapePenalty_norm",
        "BuildingTypePenalty_norm",
        "raw_n_active_meters",
        "ParkingScarcity_advantage",
        "raw_n_regulation_signs",
        "CurbRestriction_advantage",
        "raw_bf_vehicle_ty",
        "CommercialCurbContext",
        "raw_number_park_lanes",
        "raw_curb_crowding_sum",
        "CurbCrowdingPenalty",
    ]

    amr_sheet = work[[c for c in amr_sheet_cols if c in work.columns]].copy()
    car_sheet = work[[c for c in car_sheet_cols if c in work.columns]].copy()

    OUTPUTamr_features_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUTamr_features_XLSX) as writer:
        amr_sheet.to_excel(writer, index=False, sheet_name="amr_features")
        car_sheet.to_excel(writer, index=False, sheet_name="car_features")
    print(f"[AMR FEATURES] File salvato: {OUTPUTamr_features_XLSX}")
    return amr_sheet


def populate_google_times_workbook() -> None:
    """
    Legacy hook kept only for backward compatibility.
    The current project runs in last-meter-only mode and does not populate
    external routing times anymore.
    """
    print("[ROUTING TIMES] Funzione legacy non utilizzata nella pipeline corrente.")


# ============================================================
# FUNZIONE PRINCIPALE CHIAMABILE DAL BACKEND
# ============================================================

def run_filter_pipeline(selection: dict) -> dict:
    """
    Funzione principale chiamata dal backend.

    Ordine della pipeline:
    1. quartieri NTA e area selezionata
    2. sidewalks
    3. buildings
    4. streets
    5. service points
    6. datasets parcheggio
    7. pedestrian mobility
    8. street context finale
    9. traffic
    10. population
    """
    print("\n================ FILTER PIPELINE ================")
    print("[INPUT] selection:", selection)

    borough = selection.get("borough")
    neighborhoods = selection.get("neighborhoods", [])

    if borough not in BOROUGH_TO_CODE:
        raise ValueError(f"Borough non valido: {borough}")

    if not isinstance(neighborhoods, list) or len(neighborhoods) == 0:
        raise ValueError("La lista 'neighborhoods' è vuota o non valida")

    borough = clean_nta_name(borough)
    neighborhoods = [clean_nta_name(n) for n in neighborhoods]

    # 1. Quartieri NTA e poligono finale.
    gdf_neighborhoods = load_neighborhoods_geojson(NTA_GEOJSON)
    gdf_selected_neighborhoods, selected_polygon = get_selected_area_polygon(
        gdf_neighborhoods, borough, neighborhoods
    )

    print("[AREA] Bounds area selezionata:", gdf_selected_neighborhoods.total_bounds)

    # 2. Sidewalks.
    gdf_sidewalks = process_sidewalks(
        gdf_selected_neighborhoods,
        selected_polygon
    )

    # 3. Buildings.
    gdf_buildings, enrich_flags = process_buildings(
        borough, gdf_selected_neighborhoods, selected_polygon, gdf_sidewalks
    )

    # 4. Streets.
    gdf_streets = process_streets(
        borough, gdf_selected_neighborhoods, selected_polygon
    )

    # 5. Service points.
    gdf_service_points = export_service_points(gdf_buildings)

    # 6. Datasets parcheggio.
    gdf_parking_reg = process_parking_regulations(
        borough, gdf_selected_neighborhoods, selected_polygon
    )

    gdf_parking_meters = process_parking_meters(
        borough, gdf_selected_neighborhoods, selected_polygon
    )

    gdf_parking_blockfaces = process_parking_blockfaces(
        borough, gdf_selected_neighborhoods, selected_polygon
    )

    gdf_parking_ratezones = process_parking_ratezones(
        borough, gdf_selected_neighborhoods, selected_polygon
    )

    # 7. Pedestrian mobility.
    gdf_pedestrian = process_pedestrian_mobility(
        borough,
        gdf_selected_neighborhoods,
        selected_polygon
    )

    # 8. Output finale stradale con contesto parcheggio + pedestrian rank + marciapiedi.
    gdf_street_context = process_street_context(
        gdf_streets,
        gdf_parking_reg,
        gdf_parking_meters,
        gdf_parking_blockfaces,
        gdf_parking_ratezones,
        gdf_pedestrian,
        gdf_sidewalks
    )

    # 9. Traffic.
    df_traffic = process_traffic({"borough": borough, "neighborhoods": neighborhoods})

    # 10. Population.
    df_population = process_population({"borough": borough, "neighborhoods": neighborhoods})

    # 11. AMR normalized features for scoring/regression experiments.
    df_amr_features = export_amr_features(
        gdf_buildings,
        gdf_street_context,
        df_population,
        df_traffic
    )

    result = {
        "borough": borough,
        "neighborhoods": neighborhoods,
        "n_selected_neighborhoods": int(len(gdf_selected_neighborhoods)),
        "n_buildings": int(len(gdf_buildings)),
        "n_centerline": int(len(gdf_streets)),
        "n_service_points": int(len(gdf_service_points)),
        "n_traffic_rows": int(len(df_traffic)),
        "n_population_rows": int(len(df_population)),
        "n_amr_feature_rows": int(len(df_amr_features)),
        "n_pedestrian_rows": int(len(gdf_pedestrian)),
        "n_sidewalk_rows": int(len(gdf_sidewalks)),
        "n_parking_regulations": int(len(gdf_parking_reg)),
        "n_parking_meters": int(len(gdf_parking_meters)),
        "n_parking_blockfaces": int(len(gdf_parking_blockfaces)),
        "n_parking_ratezones": int(len(gdf_parking_ratezones)),
        "n_street_context_rows": int(len(gdf_street_context)),
        "enrichment": enrich_flags,
        "outputs": {
            "OUTPUTbuildings_geojson": output_name(OUTPUTbuildings_GEOJSON),
            "OUTPUTstreet_context_geojson": output_name(OUTPUTstreet_context_GEOJSON),
            "OUTPUTamr_features_xlsx": output_name(OUTPUTamr_features_XLSX),
        },
    }

    print("\n=== RIEPILOGO FINALE ===")
    print(result)
    print("=================================================\n")

    return result


if __name__ == "__main__":
    # Test locale senza frontend.
    test_selection = {
        "borough": "Manhattan",
        "neighborhoods": ["Chelsea", "Midtown"],
    }

    print(run_filter_pipeline(test_selection))
