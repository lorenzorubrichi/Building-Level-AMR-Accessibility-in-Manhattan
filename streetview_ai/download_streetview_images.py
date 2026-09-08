"""Download one Street View image per target building.

This script reads:
- data/building_targets.csv
- data/streetview_requests.csv (optional, preferred if present)
- APIK.txt

It checks Street View metadata first, then downloads one image per target when
imagery is available.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
API_KEY_PATH = BASE_DIR / "APIK.txt"
TARGETS_CSV = DATA_DIR / "building_targets.csv"
REQUESTS_CSV = DATA_DIR / "streetview_requests.csv"
DOWNLOAD_LOG_CSV = OUTPUT_DIR / "streetview_download_log.csv"
IMAGES_DIR = OUTPUT_DIR / "images"

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

DEFAULT_SIZE = "640x640"
DEFAULT_RADIUS_M = 50
TIMEOUT_S = 30
MAX_RETRIES = 4
RETRY_DELAY_S = 1.5


def http_get_bytes(url: str) -> tuple[int, bytes]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=TIMEOUT_S) as response:
                status = getattr(response, "status", 200)
                return status, response.read()
        except HTTPError as exc:
            return exc.code, exc.read()
        except URLError as exc:
            last_error = exc
        except ConnectionResetError as exc:
            last_error = exc
        except TimeoutError as exc:
            last_error = exc

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_S * attempt)

    raise RuntimeError(f"Network error for {url}: {last_error}") from last_error


def load_api_key() -> str:
    if not API_KEY_PATH.exists():
        raise FileNotFoundError(f"Missing API key file: {API_KEY_PATH}")
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("APIK.txt is empty")
    return key


def load_requests() -> pd.DataFrame:
    source = REQUESTS_CSV if REQUESTS_CSV.exists() else TARGETS_CSV
    if not source.exists():
        raise FileNotFoundError(
            f"Missing request input. Expected {REQUESTS_CSV} or {TARGETS_CSV}"
        )

    df = pd.read_csv(source)
    required = ["bin", "address_lon", "address_lat"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "heading" not in df.columns:
        df["heading"] = None
    if "pitch" not in df.columns:
        df["pitch"] = 0
    if "fov" not in df.columns:
        df["fov"] = 90
    return df


def build_location(row: pd.Series) -> str:
    lon = row["camera_lon"] if "camera_lon" in row.index and pd.notna(row["camera_lon"]) else row["address_lon"]
    lat = row["camera_lat"] if "camera_lat" in row.index and pd.notna(row["camera_lat"]) else row["address_lat"]
    return f"{lat},{lon}"


def metadata_params(row: pd.Series, api_key: str) -> dict[str, str | int | float]:
    return {
        "location": build_location(row),
        "key": api_key,
        "radius": DEFAULT_RADIUS_M,
        "source": "outdoor",
    }


def image_params(row: pd.Series, api_key: str) -> dict[str, str | int | float]:
    params: dict[str, str | int | float] = {
        "location": build_location(row),
        "size": DEFAULT_SIZE,
        "pitch": row.get("pitch", 0),
        "fov": row.get("fov", 90),
        "radius": DEFAULT_RADIUS_M,
        "source": "outdoor",
        "return_error_code": "true",
        "key": api_key,
    }
    heading = row.get("heading")
    if pd.notna(heading):
        params["heading"] = float(heading)
    return params


def build_url(base_url: str, params: dict[str, str | int | float]) -> str:
    return f"{base_url}?{urlencode(params)}"


def load_existing_downloads() -> dict[str, dict[str, object]]:
    if not DOWNLOAD_LOG_CSV.exists():
        return {}
    df = pd.read_csv(DOWNLOAD_LOG_CSV, dtype=str).fillna("")
    existing: dict[str, dict[str, object]] = {}
    for _, row in df.iterrows():
        bin_value = str(row.get("bin", "")).strip()
        if not bin_value:
            continue
        existing[bin_value] = row.to_dict()
    return existing


def main() -> None:
    api_key = load_api_key()
    requests_df = load_requests()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    existing_log = load_existing_downloads()
    log_rows: list[dict[str, object]] = []

    for _, row in requests_df.iterrows():
        bin_value = str(row["bin"]).strip()
        image_path = IMAGES_DIR / f"{int(float(row['bin']))}.jpg"

        if image_path.exists():
            previous = existing_log.get(bin_value, {})
            resumed_record = {
                "bin": row["bin"],
                "borough": row["borough"] if "borough" in requests_df.columns else "",
                "neighborhood": (
                    row["neighborhood"] if "neighborhood" in requests_df.columns else ""
                ),
                "location_source": (
                    row["location_source"] if "location_source" in requests_df.columns else ""
                ),
                "address_lon": row["address_lon"],
                "address_lat": row["address_lat"],
                "camera_lon": row["camera_lon"] if "camera_lon" in requests_df.columns else "",
                "camera_lat": row["camera_lat"] if "camera_lat" in requests_df.columns else "",
                "camera_source": row["camera_source"] if "camera_source" in requests_df.columns else "",
                "heading": row.get("heading"),
                "pitch": row.get("pitch", 0),
                "fov": row.get("fov", 90),
                "metadata_status": previous.get("metadata_status", "existing_file"),
                "copyright": previous.get("copyright", ""),
                "image_date": previous.get("image_date", ""),
                "download_status": "downloaded",
                "image_path": str(image_path),
                "image_url": previous.get("image_url", ""),
            }
            log_rows.append(resumed_record)
            continue

        previous = existing_log.get(bin_value)
        if previous and previous.get("download_status") == "downloaded":
            resumed_record = previous.copy()
            resumed_record["image_path"] = str(image_path)
            log_rows.append(resumed_record)
            continue

        record: dict[str, object] = {
            "bin": row["bin"],
            "borough": row["borough"] if "borough" in requests_df.columns else "",
            "neighborhood": (
                row["neighborhood"] if "neighborhood" in requests_df.columns else ""
            ),
            "location_source": (
                row["location_source"] if "location_source" in requests_df.columns else ""
            ),
            "address_lon": row["address_lon"],
            "address_lat": row["address_lat"],
            "camera_lon": row["camera_lon"] if "camera_lon" in requests_df.columns else "",
            "camera_lat": row["camera_lat"] if "camera_lat" in requests_df.columns else "",
            "camera_source": row["camera_source"] if "camera_source" in requests_df.columns else "",
            "heading": row.get("heading"),
            "pitch": row.get("pitch", 0),
            "fov": row.get("fov", 90),
        }

        metadata_url = build_url(METADATA_URL, metadata_params(row, api_key))
        try:
            metadata_status_code, metadata_body = http_get_bytes(metadata_url)
        except RuntimeError as exc:
            record["metadata_status"] = "network_error"
            record["download_status"] = "skipped_network_error"
            record["image_path"] = ""
            record["error"] = str(exc)
            log_rows.append(record)
            continue
        if metadata_status_code != 200:
            record["metadata_status"] = f"http_{metadata_status_code}"
            record["download_status"] = "skipped_metadata_error"
            record["image_path"] = ""
            log_rows.append(record)
            continue

        metadata = json.loads(metadata_body.decode("utf-8"))

        record["metadata_status"] = metadata.get("status")
        record["copyright"] = metadata.get("copyright", "")
        record["image_date"] = metadata.get("date", "")

        if metadata.get("status") != "OK":
            record["download_status"] = "skipped_no_imagery"
            record["image_path"] = ""
            log_rows.append(record)
            continue

        image_url = build_url(IMAGE_URL, image_params(row, api_key))
        try:
            image_status_code, image_body = http_get_bytes(image_url)
        except RuntimeError as exc:
            record["download_status"] = "image_network_error"
            record["image_path"] = ""
            record["image_url"] = image_url
            record["error"] = str(exc)
            log_rows.append(record)
            continue

        if image_status_code != 200:
            record["download_status"] = f"http_{image_status_code}"
            record["image_path"] = ""
            record["image_url"] = image_url
            log_rows.append(record)
            continue

        image_path.write_bytes(image_body)
        record["download_status"] = "downloaded"
        record["image_path"] = str(image_path)
        record["image_url"] = image_url
        log_rows.append(record)

    pd.DataFrame(log_rows).to_csv(DOWNLOAD_LOG_CSV, index=False)
    print(f"Saved download log: {DOWNLOAD_LOG_CSV}")
    print(f"Saved images to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
