# Private inputs (not redistributed)

Needed only by `pipeline/a_build_records/02_build_records.py` and the
validation kit. Location can be overridden with `AMR_PRIVATE_DIR`.

| File | Produced by | Why it is not published |
|------|-------------|-------------------------|
| `images/<BIN>.jpg` | `05_download_streetview_images.py` | Google Street View imagery may not be redistributed. Use `pano_id`/camera fields in `streetview_entrance_labels.csv` to re-request images under your own API key. |
| `streetview_download_log.csv` | `05_download_streetview_images.py` | Earlier versions stored request URLs containing the API key. The published labels record keeps every non-sensitive field. |
| `streetview_api_raw_responses.jsonl` | `06_extract_visual_features.py` | Contains full API envelopes; the published `streetview_ai_raw_outputs.jsonl` keeps the model output and all request metadata. |
| `streetview_ai_prompt.txt` | authors | Copied verbatim to `data/records/`. |
| `building_targets.csv`, `streetview_requests.csv` | `04_prepare_streetview_requests.py` | Intermediate. |
| `annotations/annotations_A*.csv` | `validation_kit/annotator.html` | Raw exports with working-session timestamps; the published `human_validation_annotations.csv` keeps every judgement (`validation_kit/03_export_annotations.py`). |
| `review/review_A*.csv` | `validation_kit/review_tool.html` | Raw exports with timestamps; the published `streetview_gate_ramp_review.csv` keeps every answer (`validation_kit/05_export_review.py`). |
