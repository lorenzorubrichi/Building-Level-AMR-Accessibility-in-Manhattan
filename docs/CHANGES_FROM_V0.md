# Changes from the first public version (v0, single commit "Add files via upload")

Every change below was made so that a reviewer can check each published number
against data and code. Figures in brackets are measured on the records.

## Data correctness

| Issue in v0 | Fix in v1 | Effect |
|---|---|---|
| The Building Footprints and Sidewalk CSVs were exported from the NYC Open Data website with Italian number formatting ('.' thousands, ',' decimal) but parsed as English | Number format detected per file and parsed explicitly; the parser refuses ambiguous input; unit test added | 8 roof heights ≥ 1,000 ft corrected (e.g. BIN 1088817: 1.396 ft → 425.5 m); 3 missing heights recovered |
| `shape_area` used as a predictor: the source's own metadata says not to use it (Web Mercator), and values ≥ 1,000 were corrupted or missing (8,185 buildings) | Footprint area recomputed from the polygon geometry (`footprint_area_m2`, m²) | Complete for all buildings |
| `zipcode` = "Unknown" for every building of the Manhattan file, although ZIP was used for spatial CV and Table 5 | ZIP from the address point with the same BIN, else the nearest address point within 50 m (`zipcode_source`) | ZIP for 99.2 % of buildings |
| 61 AI-labelled rows had no corresponding API response (no provenance) | Labels rebuilt from the 6,952 archived API responses only | AI sample = 6,952 (matches the raw responses one-to-one) |
| Unusable images (219) kept in the training data with all features = False | Excluded from the label and the model (label = missing) | Training n = 6,733 |
| Marble Hill (Manhattan BINs inside a Bronx NTA) dropped by the neighbourhood join | Kept; borough defined by BIN prefix | +213 buildings, 211 of them in Marble Hill (44,106 → 44,319) |
| Units mixed (heights in ft, distances in m) | All lengths in metres; units in the data dictionaries | – |
| Semicolon-separated CSVs, a ZIP duplicating the CSVs, empty dictionary descriptions, dictionary columns absent from the files | Comma-separated UTF-8 CSVs; one dictionary per record, generated and checked against the files by the tests | – |

## New content (v1)

| Addition | Why |
|---|---|
| PLUTO release 26v1 (single citable version) replaces an April extract of unknown version; adds residential/total units, floor area, number of buildings, alteration years, lot/building frontage, basement and proximity codes, historic district, landmark | Demand proxy (units); ADA-relevant (alterations after 1992); stoop-relevant (above-grade basements, row houses, historic districts) |
| `distance_to_street_m`, `n_address_points`, `n_buildings_on_lot` | Setback from roadway; multiple entrances; makes lot-level PLUTO fields interpretable |
| `qa_height_floors_inconsistent` (507 buildings) | Flags likely errors in height or floors |
| Labels: `camera_target_distance_m`, `qa_camera_far`, `qa_capture_before_2020`, `duplicate_request_group` | Per-label reliability information; images flagged far or old are 2–3x more often unusable; identical requests allow a repeatability check (87.5 % agreement on the label over 16 pairs, small n) |
| `manhattan_buildings_with_amr_index.csv` (all buildings, all attributes, AMR index) and `training_buildings.csv` (the 6,733 training buildings with labels and predictions) | Ready-to-use joins, as in v0, but generated from the records |
| Model v1.1 with the extended predictors, adopted by a pre-specified rule | Leave-one-neighbourhood-out ROC AUC 0.790 → 0.807, Brier 0.149 → 0.144 |

## Method consistency

| Issue in v0 | Fix in v1 |
|---|---|
| The paper describes a rule-based label (stairs/gate/ramp), but the model was trained on the AI's holistic flag `amr_can_reach_door`. The two agree on 91 % of images, hence 48.22 % (abstract) vs 46.89 % (text) | The rule label is the primary label and the model target; the AI flag is kept as a secondary field and used as a sensitivity analysis |
| Logistic regression with `class_weight="balanced"`, so probabilities were not calibrated although they define the index | Unweighted logistic regression; calibration slope/intercept and ECE reported |
| Predicted index for labelled buildings was in-sample | Out-of-fold probability `p_accessible_oof` published for every labelled building |
| "Observed rate ≈ mean predicted index" by neighbourhood used as validation (it holds by construction for an in-sample logistic fit) | Replaced by out-of-fold comparisons, spatially blocked CV, leave-one-neighbourhood/ZIP-out and an external human-labelled test set |
| Paper says "ChatGPT Pro interface"; code used the API (`gpt-5.4`) | Methods state the API, the pinned snapshot `gpt-5.4-2026-03-05`, the date/time window, parameters and the verbatim prompt |
| "Manual validation of 100 images, ~90 % agreement": no file with human judgements exists (the 100-image file is a pilot model run) | Replaced by a blind, stratified human validation: 700 images, 2 independent annotators (100 double-annotated), published as `human_validation_annotations.csv` and scored reproducibly |
| No uncertainty or extrapolation information for the 37,000+ buildings outside the labelled area | Bootstrap 90 % interval and domain-of-applicability flags for every building |
| Validation showed that the AI over-reports gates (precision 0.29) and ramps (0.09) | Targeted human review of every AI-reported gate/ramp (3,012 images) plus 300 controls, 300 double-annotated (gate κ = 0.79); reviewed values become the primary label, AI values kept in `*_ai`; an AI-only index is published next to the primary index |
| Street View camera heading computed as a planar bearing in lon/lat degrees | Documented (median 5.2°, max 8.1° deviation vs an 80° field of view); code unchanged so that the published images are reproducible |

## Reproducibility and hygiene

* Absolute Windows paths removed; all paths relative to the repository (`amr_access/config.py`) with env overrides.
* The missing `last_meter_nyc` module dependency removed; the code is self-contained.
* Mismatched folder names fixed (`logistic regression` vs `logistic_regression`).
* No GeoPandas/Shapely requirement: geometry operations use NumPy/SciPy/Matplotlib, verified against the previous GeoPandas results (sidewalk distance: median difference 2 cm, r = 0.99999).
* Manhattan snapshot of every source dataset, with SHA-256 checksums of the original downloads (`data/raw/MANIFEST.csv`).
* API keys read from environment variables only; the download script redacts the key from logged URLs; record builder aborts if a key pattern is found.
* Pinned dependencies; `run_all.py` single entry point; 18 tests.
* Exploratory material removed from the release (PLUTO clustering, within-neighbourhood models, zoning, Leaflet/Voronoi maps, internal `.docx`).
* Every number in the manuscript is generated (`manuscript/generated/numbers.tex`).
