# Publication Schemas

These CSV files define the fields included in the publication-ready datasets.

- `base_fields.csv`: selected fields for the citywide baseline dataset.
- `ai_fields.csv`: selected fields for the Street View AI subset.

Each schema is applied after legacy/internal feature names are renamed to public
names by `last_meter_nyc.feature_names`.

Expected columns:

```text
dataset,order,column,description
```

To publish a different set of fields, update the relevant schema file and rerun
`python -m last_meter_nyc.data.export_public_dataset` with the matching
`--schema` argument.
