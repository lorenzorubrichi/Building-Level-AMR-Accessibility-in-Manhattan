# AMR Accessibility Logistic Regression

This folder builds a wide dataset for testing whether raw database fields are
associated with AMR door accessibility.

Target variable:

```text
target_amr_accessible
```

It is derived from:

```text
amr_can_reach_door
```

Run:

```bash
python build_amr_accessibility_logistic_dataset.py
```

Outputs are saved in `output/`:

- `amr_accessibility_logistic_dataset.csv`: enriched 7k-building dataset.
- `amr_accessibility_logistic_dataset_dictionary.csv`: source dictionary for the columns.
- `logistic_regression_selected_features.csv`: features automatically used by the first model.
- `logistic_regression_coefficients.csv`: logistic coefficients and odds ratios.
- `logistic_regression_metrics.json`: accuracy, ROC AUC, confusion matrix, and classification report.

The target is derived from the Street View AI assessment, but the model uses
only raw database fields and basic delivery-point matching indicators as
predictors. Engineered features, normalized variables, simulated time outputs,
parking-derived scores, and Street View / AI-derived predictor variables are
excluded from the logistic-regression feature set.

This makes the model a stricter test of whether AMR accessibility can be partly
inferred from original building/address/PLUTO attributes rather than from
features engineered during the last-meter pipeline. Coefficients should still be
interpreted as association signals rather than causal effects.
