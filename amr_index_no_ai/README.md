# AMR Index Without Street View AI

This folder trains a scalable baseline model for AMR accessibility.

The model is trained on the approximately 7,000 AI-labeled buildings, because
those buildings have the binary target:

```text
target_amr_accessible
```

However, the input features intentionally exclude Street View AI variables. The
goal is to estimate AMR accessibility using only features that can potentially
be available citywide.

## Model Features

The current model uses six numeric attributes:

- number of floors;
- roof height;
- construction year;
- ground elevation;
- building footprint area;
- distance from the delivery point to the sidewalk.

It also uses three categorical PLUTO codes:

- land-use code;
- complete building-class code;
- building-class group, obtained from the first character of the building-class code.

Borough, neighborhood, ZIP code, category descriptions, delivery-point source,
geometry source, and building status are retained where useful for outputs and
validation, but they are not model predictors. Spatial cross-validation still
uses neighborhood and ZIP code only to define held-out test areas.

## AMR Index Definition

For each building, the selected AMR Index model estimates:

```text
P(AMR accessible | non-AI building and urban features)
```

The baseline AMR Index is:

```text
baseline_amr_index = 100 * predicted probability
```

So an index of 82 means that the model estimates an 82% probability that the
building is AMR-accessible, based only on non-AI features.

## Models Compared

The script compares:

- logistic regression;
- polynomial logistic regression;
- random forest;
- histogram gradient boosting.

The AMR Index itself is computed with logistic regression because it provides a
transparent probability formula and is easier to explain in a paper:

```text
p_i = 1 / (1 + exp(-(beta_0 + beta * X_i)))
AMR_Index_i = 100 * p_i
```

The other models are kept as benchmarks to show whether more flexible
non-linear models substantially improve performance.

## Main Outputs

Files are saved in `output/`:

```text
no_ai_model_metrics.csv
no_ai_model_metrics.json
building_level_no_ai_amr_index.csv
no_ai_selected_features.csv
best_model_feature_importance.csv
no_ai_amr_index_by_neighborhood.csv
no_ai_amr_index_by_zipcode.csv
no_ai_amr_index_report.txt
```

## How To Run

From this folder:

```bash
python train_no_ai_amr_index_models.py
```

## How This Fits The Project

This is the scalable baseline. Once validated, the same model structure can be
applied to the complete building dataset if the same non-AI features are
available. A later AI-enhanced model can be trained on the same 7,000 buildings
to measure how much Street View AI improves prediction.
