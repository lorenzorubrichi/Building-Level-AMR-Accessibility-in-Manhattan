"""Analyze Street View images with the OpenAI Responses API.

This script reads downloaded Street View images, sends them to the OpenAI API,
and writes a minimal feature CSV focused on image usability plus three binary
entrance features: stairs, gate, and ramp.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[2]))
from amr_access import config  # noqa: E402



OUTPUT_DIR = config.PRIVATE_DIR
DOWNLOAD_LOG_CSV = OUTPUT_DIR / "streetview_download_log.csv"
OUTPUT_CSV = OUTPUT_DIR / "streetview_visual_features_api.csv"
SUMMARY_CSV = OUTPUT_DIR / "streetview_visual_features_summary.csv"
RAW_JSONL = OUTPUT_DIR / "streetview_api_raw_responses.jsonl"

RESPONSES_URL = "https://api.openai.com/v1/responses"
TIMEOUT_S = 120
MAX_RETRIES = 4
RETRY_DELAY_S = 2.0
FEATURE_COLUMNS = [
    "bin",
    "image_usable",
    "stairs_present",
    "gate_present",
    "ramp_present",
    "amr_can_reach_door",
]

JSON_TEMPLATE: dict[str, Any] = {
    "image_usable": "false",
    "stairs_present": "false",
    "gate_present": "false",
    "ramp_present": "false",
    "amr_can_reach_door": "false",
}

# The exact prompt used for the published annotations (2026-05-14) is stored with the
# data records; it is read from there so that code and data cannot diverge.
PROMPT = config.PROMPT_TXT.read_text(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Street View images with OpenAI API")
    parser.add_argument("--model", default="gpt-5.4-2026-03-05", help="OpenAI model snapshot (pinned)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on images to analyze")
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Reanalyze images even if output CSV already exists",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Keep existing analyzed rows and only process missing ones (default)",
    )
    return parser.parse_args()


def load_api_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    raise RuntimeError("Set the OPENAI_API_KEY environment variable (see .env.example)")


def encode_image_as_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(suffix, "image/jpeg")
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def build_payload(model: str, image_path: Path) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": PROMPT},
                    {
                        "type": "input_image",
                        "image_url": encode_image_as_data_url(image_path),
                        "detail": "high",
                    },
                ],
            }
        ],
    }


def http_post_json(url: str, api_key: str, payload: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=TIMEOUT_S) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {error_text}") from exc
        except (URLError, TimeoutError, ConnectionResetError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_S * attempt)

    raise RuntimeError(f"Network error: {last_error}") from last_error


def extract_output_text(response_json: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in response_json.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if text:
                    texts.append(text)
    return "\n".join(texts).strip()


def parse_model_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = next((part for part in parts if "{" in part), text)
        text = text[text.find("{") :]
        if text.rfind("}") != -1:
            text = text[: text.rfind("}") + 1]
    return json.loads(text)


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    out = dict(JSON_TEMPLATE)
    out.update(result)
    for key, value in list(out.items()):
        if isinstance(value, bool):
            out[key] = "true" if value else "false"
            continue
        lowered = str(value).strip().lower()
        out[key] = "true" if lowered == "true" else "false"
    return out


def save_outputs(output_rows: list[dict[str, Any]], raw_lines: list[str]) -> None:
    out_df = pd.DataFrame(output_rows, columns=FEATURE_COLUMNS)
    out_df.to_csv(OUTPUT_CSV, index=False)
    out_df.to_csv(SUMMARY_CSV, index=False)

    with RAW_JSONL.open("w", encoding="utf-8") as fh:
        for line in raw_lines:
            fh.write(line + "\n")


def main() -> None:
    args = parse_args()

    if not DOWNLOAD_LOG_CSV.exists():
        raise FileNotFoundError(f"Missing download log: {DOWNLOAD_LOG_CSV}")

    api_key = load_api_key()
    downloads = pd.read_csv(DOWNLOAD_LOG_CSV)
    if "download_status" not in downloads.columns:
        raise ValueError("Download log must contain 'download_status'")

    rows = downloads[downloads["download_status"] == "downloaded"].copy()
    if args.limit:
        rows = rows.head(args.limit)

    existing: dict[str, dict[str, Any]] = {}
    if OUTPUT_CSV.exists() and not args.overwrite:
        existing_df = pd.read_csv(OUTPUT_CSV, dtype=str).fillna("")
        existing = {
            str(row["bin"]): row.to_dict()
            for _, row in existing_df.iterrows()
            if row.get("image_usable", "") in ("true", "false")
        }

    output_rows: list[dict[str, Any]] = []
    raw_lines: list[str] = []

    if not args.overwrite:
        for bin_value in rows["bin"].astype(str):
            if bin_value in existing:
                output_rows.append(existing[bin_value])

    for _, row in rows.iterrows():
        image_path = config.PRIVATE_DIR / "images" / f"{row['bin']}.jpg"  # portable location
        bin_value = str(row["bin"])

        if bin_value in existing and not args.overwrite:
            continue

        base_row = {
            "bin": row["bin"],
        }

        if not image_path.exists():
            result = dict(JSON_TEMPLATE)
            result["image_usable"] = "false"
            normalized = normalize_result(result)
            normalized.update(base_row)
            output_rows.append(normalized)
            continue

        try:
            response_json = http_post_json(RESPONSES_URL, api_key, build_payload(args.model, image_path))
        except RuntimeError as exc:
            print(f"[skip] bin={bin_value} api error: {exc}")
            continue
        raw_lines.append(json.dumps({"bin": row["bin"], "response": response_json}, ensure_ascii=True))

        output_text = extract_output_text(response_json)
        try:
            parsed = parse_model_json(output_text)
        except json.JSONDecodeError as exc:
            print(f"[skip] bin={bin_value} invalid JSON response: {exc}")
            continue
        normalized = normalize_result(parsed)
        normalized.update(base_row)
        output_rows.append(normalized)
        save_outputs(output_rows, raw_lines)

    save_outputs(output_rows, raw_lines)

    print(f"Saved API visual features: {OUTPUT_CSV}")
    print(f"Saved summary visual features: {SUMMARY_CSV}")
    if raw_lines:
        print(f"Saved raw API responses: {RAW_JSONL}")


if __name__ == "__main__":
    main()
