# Within-Neighborhood Logistic Regression

This folder fits one raw-database logistic regression per neighborhood.

Purpose:

```text
Fix the neighborhood and estimate how raw variables such as number of floors,
land use, construction year, ground elevation, coordinates, and delivery-point
source are associated with AMR accessibility inside each neighborhood.
```

This differs from the global model:

- the global model estimates neighborhood coefficients and other variables in
  one model;
- this analysis removes the neighborhood comparison by fitting separate models
  inside each neighborhood.

Run:

```bash
python run_within_neighborhood_logistic_regression.py
```

Outputs are saved in `output/`:

- `within_neighborhood_model_summary.csv`: one row per neighborhood model.
- `within_neighborhood_coefficients.csv`: coefficients estimated within each neighborhood.
- `within_neighborhood_selected_features.csv`: features used by each local model.
- `within_neighborhood_metrics.json`: full metrics and skipped-neighborhood notes.

Interpretation:

```text
Positive coefficient  -> associated with higher AMR accessibility inside that neighborhood.
Negative coefficient  -> associated with lower AMR accessibility inside that neighborhood.
Larger absolute value -> stronger local association in the model.
```

These are associations, not causal effects.
