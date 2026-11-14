"""Assemble the published Data Records from the building table and the Street View / AI run.

Inputs
    data/interim/building_attributes.csv                (01_build_building_attributes.py)
    AMR_PRIVATE_DIR/streetview_download_log.csv          (download_streetview_images.py)
    AMR_PRIVATE_DIR/streetview_api_raw_responses.jsonl   (extract_visual_features.py)
    AMR_PRIVATE_DIR/streetview_ai_prompt.txt             (prompt sent with every image)

Outputs (data/records/)
    manhattan_buildings.csv            one row per constructed Manhattan building
    streetview_entrance_labels.csv     one row per building with an AI-annotated image
    streetview_ai_raw_outputs.jsonl    verbatim model outputs with request metadata
    streetview_ai_prompt.txt           exact prompt

Nothing derived from Google content other than panorama IDs, capture dates and
camera geometry is written; image URLs (which embed the API key) and local
paths are dropped. The script refuses to write if an API key pattern is found.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.labels import rule_label  # noqa: E402

QA_CAMERA_FAR_M = 30.0
KEY_PATTERNS = re.compile(r"AIza[0-9A-Za-z_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}")


def read_ai_run(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    rows, raw = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        resp = rec["response"]
        text = "".join(
            c.get("text", "")
            for item in resp.get("output", [])
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
        parsed = json.loads(text)
        created = datetime.fromtimestamp(resp["created_at"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append({
            "bin": str(rec["bin"]),
            "image_usable": bool(parsed["image_usable"]),
            "stairs_present": parsed.get("stairs_present"),
            "gate_present": parsed.get("gate_present"),
            "ramp_present": parsed.get("ramp_present"),
            "ai_amr_can_reach_door": parsed.get("amr_can_reach_door"),
            "ai_model": resp["model"],
            "ai_response_id": resp["id"],
            "ai_request_utc": created,
        })
        usage = resp.get("usage", {})
        raw.append({
            "bin": str(rec["bin"]),
            "response_id": resp["id"],
            "model": resp["model"],
            "created_utc": created,
            "status": resp.get("status"),
            "temperature": resp.get("temperature"),
            "top_p": resp.get("top_p"),
            "reasoning_effort": (resp.get("reasoning") or {}).get("effort"),
            "image_detail": "high",
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "output_text": text,
        })
    df = pd.DataFrame(rows)
    if df["bin"].duplicated().any():
        raise ValueError("Duplicate BINs in the AI run")
    for c in ["stairs_present", "gate_present", "ramp_present", "ai_amr_can_reach_door"]:
        df[c] = df[c].astype("boolean")
    return df, raw


def read_download_log(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"bin": str, "pano_id": str, "image_date": str, "image_url": str},
                    low_memory=False)
    # Images were requested with `location=<camera point>` (a point on the nearest road
    # centreline), `heading` from the camera point towards the target, `source=outdoor`.
    # A few late requests used `pano=<id>` with the panorama position as camera point.
    url = d["image_url"].fillna("")
    by_pano = url.str.contains(r"[?&]pano=", regex=True)
    pano_from_url = url.str.extract(r"[?&]pano=([^&]+)", expand=False)
    cam_lon = d["camera_lon"].where(d["camera_lon"].notna(), d["pano_lon"])
    cam_lat = d["camera_lat"].where(d["camera_lat"].notna(), d["pano_lat"])
    heading = d["heading"].where(d["heading"].notna(), d["heading_deg"])
    out = pd.DataFrame({
        "bin": d["bin"],
        "target_lon": d["address_lon"],
        "target_lat": d["address_lat"],
        "target_source": d["location_source"],
        "request_mode": np.where(by_pano, "panorama_id", "camera_location"),
        "camera_lon": cam_lon,
        "camera_lat": cam_lat,
        "camera_source": np.where(d["camera_lon"].notna(), d["camera_source"].fillna("road_centerline"),
                                  "streetview_panorama"),
        "heading_deg": heading,
        "pitch_deg": d["pitch"],
        "fov_deg": d["fov"],
        "image_size_px": "640x640",
        "pano_id": d["pano_id"].where(d["pano_id"].notna(), pano_from_url),
        "image_capture_date": d["image_date"],
        "download_status": d["download_status"],
    })
    return out


def main() -> None:
    priv = config.PRIVATE_DIR
    rec_dir = config.RECORDS_DIR
    rec_dir.mkdir(parents=True, exist_ok=True)

    needed = [priv / "streetview_api_raw_responses.jsonl", priv / "streetview_download_log.csv",
              priv / "streetview_ai_prompt.txt"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        sys.exit(f"[02] Private acquisition files not found in {priv}: {', '.join(missing)}.\n"
                 "     The building table was rebuilt in data/interim/building_attributes.csv; the published\n"
                 "     label record in data/records/ is unchanged. See data/private/README.md.")
    b = pd.read_csv(config.DATA_DIR / "interim" / "building_attributes.csv",
                    dtype={"bin": str, "bbl": str, "zipcode": str, "landuse": str, "nta2020": str})
    ai, raw = read_ai_run(priv / "streetview_api_raw_responses.jsonl")
    log = read_download_log(priv / "streetview_download_log.csv")

    # ---------------- entrance labels ----------------
    lab = ai.merge(log, on="bin", how="left", validate="one_to_one")
    missing = lab["download_status"].isna().sum()
    if missing:
        raise ValueError(f"{missing} AI-annotated BINs have no download-log entry")
    lab = lab.merge(b[["bin", "nta2020", "nta_name", "zipcode"]], on="bin", how="left")
    lab["in_building_table"] = lab["bin"].isin(set(b["bin"]))
    not_in_buildings = int((~lab["in_building_table"]).sum())
    lab["accessible_rule"] = rule_label(lab["stairs_present"].fillna(False), lab["gate_present"].fillna(False),
                                        lab["ramp_present"].fillna(False), lab["image_usable"])
    # feature fields are undefined for unusable images
    for c in ["stairs_present", "gate_present", "ramp_present", "ai_amr_can_reach_door"]:
        lab.loc[~lab["image_usable"], c] = pd.NA
    # ---------------- per-label quality fields ----------------
    from amr_access.geo import haversine_m, local_xy_m
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    lab["camera_target_distance_m"] = haversine_m(lab["camera_lon"], lab["camera_lat"],
                                                  lab["target_lon"], lab["target_lat"])
    year = pd.to_numeric(lab["image_capture_date"].str[:4], errors="coerce")
    lab["qa_camera_far"] = lab["camera_target_distance_m"] > QA_CAMERA_FAR_M
    lab["qa_capture_before_2020"] = (year < 2020).fillna(False).astype(bool)
    # near-identical requests (same camera point within 1 m, heading within 5 deg) -> same image
    x, y = local_xy_m(lab["camera_lon"], lab["camera_lat"])
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(r=1.0, output_type="ndarray")
    dh = np.abs((lab["heading_deg"].to_numpy()[pairs[:, 0]] - lab["heading_deg"].to_numpy()[pairs[:, 1]] + 180) % 360 - 180)
    pairs = pairs[dh < 5]
    n = len(lab)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    sizes = np.bincount(comp)
    lab["duplicate_request_group"] = pd.array(np.where(sizes[comp] > 1, comp, -1), dtype="Int64")
    lab.loc[lab["duplicate_request_group"] == -1, "duplicate_request_group"] = pd.NA
    # renumber groups 1..k for readability
    groups = lab["duplicate_request_group"].dropna().unique()
    lab["duplicate_request_group"] = lab["duplicate_request_group"].map({g: i + 1 for i, g in enumerate(sorted(groups))})

    cols = ["bin", "nta2020", "nta_name", "zipcode", "in_building_table",
            "target_lon", "target_lat", "target_source", "request_mode", "camera_lon", "camera_lat",
            "camera_source", "camera_target_distance_m", "heading_deg", "pitch_deg", "fov_deg", "image_size_px",
            "pano_id", "image_capture_date", "duplicate_request_group", "qa_camera_far", "qa_capture_before_2020",
            "image_usable", "stairs_present", "gate_present", "ramp_present",
            "accessible_rule", "ai_amr_can_reach_door", "ai_model", "ai_response_id", "ai_request_utc"]
    lab = lab[cols].sort_values("bin").reset_index(drop=True)

    # ---------------- building table ----------------
    downloaded = set(log.loc[log["download_status"] == "downloaded", "bin"])
    b["streetview_image_acquired"] = b["bin"].isin(downloaded)
    b["in_ai_annotated_sample"] = b["bin"].isin(set(lab["bin"]))
    bcols = ["bin", "bbl", "borough", "nta2020", "nta_name", "zipcode", "zipcode_source",
             "centroid_lon", "centroid_lat", "delivery_lon", "delivery_lat", "delivery_point_source", "full_address",
             "footprint_area_m2", "construction_year", "height_roof_m", "ground_elevation_m", "numfloors", "landuse", "landuse_label",
             "bldgclass", "bldgclass_group", "bldgclass_group_label",
             "lot_units_res", "lot_units_total", "lot_bldgarea_m2", "lot_numbldgs", "yearalter1", "yearalter2", "last_alteration_year", "lotfront_m", "bldgfront_m", "bldgdepth_m", "basement_code", "proximity_code", "in_historic_district", "historic_district", "is_landmark", "landmark_name", "pluto_version",
             "distance_to_sidewalk_m",
             "distance_to_street_m", "n_address_points", "n_buildings_on_lot", "qa_height_floors_inconsistent",
             "geom_source", "streetview_image_acquired", "in_ai_annotated_sample"]
    b = b[bcols]

    # ---------------- write ----------------
    raw_lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in sorted(raw, key=lambda r: r["bin"])) + "\n"
    outputs = {
        config.BUILDINGS_CSV: b.to_csv(index=False, float_format="%.10g", lineterminator="\n"),
        config.LABELS_CSV: lab.to_csv(index=False, float_format="%.10g", lineterminator="\n"),
        config.AI_RAW_JSONL: raw_lines,
    }
    for path, text in outputs.items():
        if KEY_PATTERNS.search(text):
            raise RuntimeError(f"API key pattern detected in {path.name}; aborting")
        path.write_text(text, encoding="utf-8", newline="\n")
    shutil.copyfile(priv / "streetview_ai_prompt.txt", config.PROMPT_TXT)

    print(f"buildings: {len(b):,} | AI-annotated: {len(lab):,} "
          f"(usable {int(lab['image_usable'].sum()):,}; outside building table {not_in_buildings})")
    print(f"rule-accessible among usable: {lab['accessible_rule'].mean():.4f}")


if __name__ == "__main__":
    main()
