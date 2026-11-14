"""Write one data dictionary per record (CSV) with observed missingness, and fail on schema drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amr_access import config  # noqa: E402
from amr_access.dictionary import RECORDS  # noqa: E402


def load(name: str) -> pd.DataFrame:
    path = config.RECORDS_DIR / name
    if name.endswith(".jsonl"):
        return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()])
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def present(name: str) -> bool:
    return (config.RECORDS_DIR / name).exists()


def main() -> None:
    config.DICTIONARY_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in RECORDS.items():
        if not present(name):
            print(f"{name}: not present, skipped")
            continue
        df = load(name)
        if list(df.columns) != list(spec):
            raise ValueError(f"{name}: columns {list(df.columns)} differ from dictionary {list(spec)}")
        rows = [{"column": c, "type": t, "unit": u, "description": d, "source": s,
                 "missing_share": round(float(df[c].isna().mean()), 4)} for c, (t, u, d, s) in spec.items()]
        out = config.DICTIONARY_DIR / (Path(name).stem + "_dictionary.csv")
        pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
        print(f"{out.name}: {len(rows)} columns")


if __name__ == "__main__":
    main()
