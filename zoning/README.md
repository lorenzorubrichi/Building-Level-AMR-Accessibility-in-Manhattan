# Zoning / Area-Level AMR Accessibility Step

This folder contains the first urban-planning extension of the AMR accessibility
project.

The goal is to move from a building-level dataset to an area-level view using
existing geographic aggregations. This step answers:

> If we use current geographic units, which areas appear more or less suitable
> for AMR delivery?

The building-level input dataset is not modified. In the area-level outputs,
geographic areas with fewer than 20 buildings are excluded to avoid unstable
accessibility rates based on very small samples.

## Input

The script starts from:

```text
../logistic regression/output/amr_accessibility_logistic_dataset.csv
```

This dataset combines building identifiers, location fields, raw PLUTO/building
attributes, calculated features, Street View AI features, estimated last-meter
times, and the AMR accessibility target.

## Outputs

Files are saved in `output/`:

```text
area_level_amr_summary_by_borough.csv
area_level_amr_summary_by_neighborhood.csv
area_level_amr_summary_by_zipcode.csv
area_level_amr_summary_all_geographies.csv
area_level_amr_top_bottom_areas.csv
```

Each row represents one existing geographic area and includes:

- number of buildings;
- AMR-accessible buildings;
- AMR accessibility rate;
- average and median AMR/car last-meter time;
- average AI barrier indicators;
- average physical building attributes;
- dominant land use;
- dominant building-class group;
- shares of selected building types.

## How To Run

From this folder:

```bash
python build_area_level_amr_summary.py
```

## How To Use This Step

This is a baseline planning analysis. It shows what the project would conclude
if existing geographic units, such as boroughs, neighborhoods, and ZIP codes,
were used as the spatial framework.

The next step can compare these official/current boundaries with new
data-driven AMR-friendly zones discovered from the building-level accessibility
data.
