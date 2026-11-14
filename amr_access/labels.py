"""Derivation of the entrance accessibility label from AI-extracted visual features.

The primary label of the dataset is deterministic and auditable: it is computed
from three visual attributes (stairs, gate, ramp) with the rule below. The
multimodal model's own holistic judgement (``amr_can_reach_door``) is kept as a
secondary, independent field.

Rule (only for usable images):
    accessible = (no gate) AND (no stairs OR ramp present)

Images flagged as unusable receive a missing label (not 0), because no
entrance-level information could be extracted from them.
"""

from __future__ import annotations

import pandas as pd

RULE_TABLE = [
    # stairs, gate,  ramp,  accessible
    (False, False, False, True),
    (False, False, True, True),
    (True, False, True, True),
    (True, False, False, False),
    (False, True, False, False),
    (False, True, True, False),
    (True, True, False, False),
    (True, True, True, False),
]


def rule_label(stairs: pd.Series, gate: pd.Series, ramp: pd.Series, usable: pd.Series) -> pd.Series:
    """Return the rule-based accessibility label as a nullable integer (1/0/<NA>)."""
    s = stairs.astype("boolean")
    g = gate.astype("boolean")
    r = ramp.astype("boolean")
    label = (~g) & ((~s) | r)
    out = label.astype("Int64")
    out[~usable.astype(bool)] = pd.NA
    return out


def rule_table_frame() -> pd.DataFrame:
    return pd.DataFrame(RULE_TABLE, columns=["stairs", "gate", "ramp", "accessible"])
