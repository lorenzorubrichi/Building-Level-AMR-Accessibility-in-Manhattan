# Street View + AI

This folder is a clean workspace for a visual-feature pipeline based on Street View imagery and AI analysis.

## Initial scripts
- `init_project.py`
- `build_building_targets.py`
- `prepare_streetview_requests.py`
- `download_streetview_images.py`
- `extract_visual_features.py`
- `merge_visual_features.py`

## Minimal workflow
1. Put `APIK.txt` in this folder, plus one of:
   - `final_last_meter_database.csv` (preferred)
   - `complete_last_meter_dataset_extended.csv`
   - `BUILDING.csv`
2. Run `build_building_targets.py` to extract Street View targets from the preferred CSV. By default it uses all rows; use `--limit N` only if you want a smaller pilot.
3. Run `prepare_streetview_requests.py` if you want to review/edit heading, pitch, or fov.
4. Run `download_streetview_images.py` to fetch one Street View image per target building.
5. Add an OpenAI API key:
   - environment variable `OPENAI_API_KEY`, or
   - `GPTKey.txt` in this folder
6. Run `extract_visual_features.py` to analyze the downloaded images via API and write:
   - `streetview_visual_features_api.csv`
   - `streetview_visual_features_summary.csv`
   - `streetview_api_raw_responses.jsonl`
   - By default, running the script with no arguments performs a full overwrite run on all available images.
   - Use `--limit N` for a smaller pilot run.
   - Use `--no-overwrite` if you want to preserve existing analyzed rows and only fill missing ones.
7. Run `merge_visual_features.py` to merge the visual features into a building dataset.

## Current visual features
- image_usable
- stairs_present
- gate_present
- ramp_present

## Usability rule
If an image is:
- inside the building
- heavily occluded

then `image_usable` should be `false`.
