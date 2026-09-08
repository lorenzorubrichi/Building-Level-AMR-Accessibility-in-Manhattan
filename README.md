# Building-Level AMR Accessibility in Manhattan

This repository contains the code and datasets used to build a
building-level last-meter accessibility dataset and estimate autonomous mobile
robot (AMR) accessibility for buildings in Manhattan, New York City.

The project combines open urban datasets, address-point information, Google
Street View imagery, visual AI assessment, logistic regression, and spatial
aggregation. The final goal is to estimate whether individual buildings are
physically accessible to sidewalk delivery robots and to translate those
building-level estimates into planning-oriented spatial layers.

## Project Workflow

The workflow has five main stages:

1. Construct a building-level dataset from open urban data.
2. Select building-level Street View targets and download entrance-oriented
   images.
3. Extract AI-derived visual features from Street View imagery.
4. Train a non-AI logistic regression model and estimate the AMR Accessibility
   Index.
5. Aggregate building-level AMR Index values into existing geographic areas,
   hexagonal cells, and AMR-oriented zones.

The AI-labeled image sample contains approximately 7,000 buildings. This sample
is used to train and validate the model. The resulting model is then applied to
Manhattan buildings using non-visual building and urban attributes.

## Repository Structure

```text
Github/
├── data/
│   ├── raw/                  GitHub-safe raw-data samples and raw-data notes
│   ├── processed/            Reserved for regenerated processed outputs
│   └── schema/               Field lists and schema documentation
├── dataset_pipeline/         Building-level dataset construction scripts
├── streetview_ai/            Street View target, download, and AI extraction scripts
├── ai_regression/            AI visual-feature merge scripts
├── logistic_regression/      Logistic modeling dataset scripts
├── amr_index_no_ai/          Main AMR Accessibility Index model scripts
├── zoning/                   Spatial aggregation and AMR-zone scripts
├── final_datasets/           Final datasets included in this repository
├── requirements.txt          Python dependencies
├── .env.example              Local API-key template
└── README.md
```

## Folder Contents

### `data/`

Contains GitHub-safe data files.

- `data/raw/` contains small schema-compatible raw-data samples. The complete
  raw municipal datasets are not included because several files exceed GitHub
  size limits.
- `data/schema/` contains field lists and schema documentation.
- `data/processed/` is reserved for processed outputs regenerated locally.

To reproduce the full pipeline, replace the sample raw files with the complete
raw files using the same file names.

### `dataset_pipeline/`

Contains scripts used to build the core building-level dataset from open urban
data. These scripts combine building footprints, PLUTO property records,
address points, sidewalk data, and selected geographic boundary files.

Important scripts include:

```text
buildCompleteDataset.py
build_final_last_meter_database.py
build_normalized_nta_keys.py
Dataset.py
```

The `archivio/` subfolder contains supporting local geography files used by the
pipeline, including selected Manhattan buildings, service points, streets, and
neighborhood boundary files.

### `streetview_ai/`

Contains scripts for preparing Street View image targets, downloading images,
and extracting AI-derived visual features.

Important scripts include:

```text
build_building_targets.py
prepare_streetview_requests.py
download_streetview_images.py
extract_visual_features.py
merge_visual_features.py
```

The complete Street View image folder is not included because it is larger than
1 GB. It can be regenerated locally after setting the Google Street View API
key.

### `ai_regression/`

Contains scripts that merge AI-derived visual features with the building-level
data. This step produces the AI-labeled image sample used for model training
and validation.

Important scripts include:

```text
merge_ai_visual_features.py
build_ai_subset_map.py
```

### `logistic_regression/`

Contains scripts for constructing the logistic-regression modeling dataset. The
target variable is the AI-derived AMR accessibility label, while predictors are
non-visual building, property, and address attributes.

Important script:

```text
build_amr_accessibility_logistic_dataset.py
```

The subfolders `pluto_clustering/` and `within_neighborhood/` contain additional
exploratory analyses used to evaluate building typologies and within-area
patterns.

### `amr_index_no_ai/`

Contains the main AMR Accessibility Index model. The selected model is a
logistic regression trained on the AI-labeled image sample using only non-visual
building and urban attributes.

Important script:

```text
train_no_ai_amr_index_models.py
```

The predicted probability is converted into the building-level AMR Accessibility
Index:

```text
AMR Index = 100 * P(AMR-accessible | non-visual building attributes)
```

### `zoning/`

Contains scripts for spatial aggregation and AMR-oriented zoning. These scripts
summarize AMR accessibility by existing geographic areas, build hexagonal
spatial cells, merge adjacent cells into AMR-oriented zones, and evaluate
alternative spatial aggregation settings.

Important scripts include:

```text
build_existing_area_amr_index_maps.py
build_area_level_amr_summary.py
build_amr_friendly_hex_map.py
build_amr_friendly_zones.py
build_amr_zone_level_summary.py
analyze_hex_zone_sensitivity.py
analyze_grid_zone_sensitivity.py
```

### `final_datasets/`

Contains the final datasets retained in the GitHub-ready version of the
project.

```text
ai_labeled_buildings_with_visual_features_and_amr_index.csv
manhattan_buildings_with_estimated_amr_index.csv
ai_labeled_buildings_dictionary.csv
```

- `ai_labeled_buildings_with_visual_features_and_amr_index.csv` contains the
  AI-labeled building sample, AI-derived visual features, and estimated AMR
  Index fields.
- `manhattan_buildings_with_estimated_amr_index.csv` contains all Manhattan
  buildings included in the final application, with estimated AMR accessibility
  probabilities and AMR Index values.
- `ai_labeled_buildings_dictionary.csv` provides field descriptions for the
  AI-labeled sample.

## Installation

Create and activate a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a local `.env` file if running the Street View download or visual AI
steps. Do not commit `.env` to GitHub.

```text
GOOGLE_STREETVIEW_API_KEY=your_google_key_here
OPENAI_API_KEY=your_openai_key_here
```

An empty template is provided in:

```text
.env.example
```

## Step-by-Step Execution

Run the scripts from the repository root unless otherwise noted.

### 1. Build the building-level dataset

```bash
cd dataset_pipeline
python buildCompleteDataset.py
python build_final_last_meter_database.py
```

This step creates the building-level dataset by linking open urban datasets at
the building/BIN level.

### 2. Create Street View targets

```bash
cd ../streetview_ai
python build_building_targets.py
python prepare_streetview_requests.py
```

This step identifies the building locations used to request Street View images.

### 3. Download Street View images

```bash
python download_streetview_images.py
```

This step requires `GOOGLE_STREETVIEW_API_KEY`. The downloaded images are
generated locally and are not included in this GitHub-ready folder.

### 4. Extract AI-derived visual features

```bash
python extract_visual_features.py
python merge_visual_features.py
```

This step requires `OPENAI_API_KEY`. It generates visual features such as image
usability, visible stairs, visible gates, visible ramps, and AMR door
reachability.

### 5. Merge AI features into the AI-labeled sample

```bash
cd ../ai_regression
python merge_ai_visual_features.py
python build_ai_subset_map.py
```

This step produces the AI-labeled image sample used for model training and
validation.

### 6. Build the logistic-regression modeling dataset

```bash
cd ../logistic_regression
python build_amr_accessibility_logistic_dataset.py
```

This step creates the model-ready dataset with the AI-derived accessibility
label and non-visual predictor variables.

### 7. Train the AMR Accessibility Index model

```bash
cd ../amr_index_no_ai
python train_no_ai_amr_index_models.py
```

This step trains the logistic regression model, evaluates model performance, and
generates building-level AMR Index estimates.

### 8. Build spatial summaries and AMR-oriented zones

```bash
cd ../zoning
python build_area_level_amr_summary.py
python build_existing_area_amr_index_maps.py
python build_amr_friendly_hex_map.py
python build_amr_friendly_zones.py
python build_amr_zone_level_summary.py
```

This step aggregates building-level AMR Index values into existing geographic
areas, hexagonal cells, and aggregated AMR-oriented zones.

### 9. Run sensitivity analysis

```bash
python analyze_hex_zone_sensitivity.py
python analyze_grid_zone_sensitivity.py
```

This step evaluates how hexagon radius and minimum-building thresholds affect
coverage, within-zone variation, number of zones, and eta squared.

## Reproducing Outputs

Intermediate outputs are intentionally excluded from the GitHub-ready folder.
Running the workflow scripts will recreate local `output/` folders as needed.
Generated maps, logs, model diagnostics, temporary files, and full image
folders should remain local unless selected for publication.

## Citation

If this repository is used in a thesis, paper, or derivative project, cite the
repository and the original NYC open-data sources used by the pipeline.
