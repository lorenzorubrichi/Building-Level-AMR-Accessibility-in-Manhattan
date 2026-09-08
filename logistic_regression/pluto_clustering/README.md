# PLUTO Building-Type Clustering

This analysis tests whether PLUTO/building-type attributes can be used as a
pre-screening layer for AMR accessibility analysis.

Motivation:

```text
Different building types may require different Street View interpretation or
path-finding logic. For example, a townhouse, a walk-up apartment building, and
a skyscraper may present very different entrance and delivery-path conditions.
```

The script clusters the AI-labeled buildings using raw PLUTO/building attributes:

- number of floors
- roof height
- construction year
- ground elevation
- footprint area
- land use
- building class
- building class group

It then compares AMR accessibility rates across:

- unsupervised PLUTO clusters
- PLUTO land-use categories
- detailed building classes
- broad building-class groups

Run:

```bash
python run_pluto_building_type_clustering.py
```

Outputs are saved in `output/`:

- `pluto_clustered_ai_buildings.csv`
- `pluto_cluster_summary.csv`
- `landuse_accessibility_summary.csv`
- `bldgclass_accessibility_summary.csv`
- `bldgclass_group_accessibility_summary.csv`
- `cluster_model_comparison.json`

Interpretation:

```text
If accessibility rates vary strongly by PLUTO cluster or building type, then
building type can be used to pre-screen Street View images and guide different
feature-recognition strategies for different building categories.
```
