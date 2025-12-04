"""
Shared helpers for evaluation type modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:
    import yaml

    HAVE_YAML = True
except Exception:  # pragma: no cover
    HAVE_YAML = False


def read_config(path: str) -> Dict[str, Any]:
    """Load a JSON or YAML config file."""
    ext = Path(path).suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yml", ".yaml"):
            if not HAVE_YAML:
                raise RuntimeError("PyYAML not installed.")
            return yaml.safe_load(f)
        return json.load(f)


def clean_str(value) -> Optional[str]:
    """Normalize free-form strings and treat blanks / NaNs uniformly."""
    if value is None:
        return np.nan
    if isinstance(value, float) and np.isnan(value):
        return np.nan
    text = str(value).strip()
    return text if text else np.nan


def apply_value_map(series: pd.Series, value_map: Optional[Mapping]) -> pd.Series:
    """Apply a case-insensitive mapping to a Series."""
    if not value_map:
        return series
    lower_map = {str(k).strip().lower(): v for k, v in value_map.items()}

    def _map_one(v):
        if pd.isna(v):
            return v
        return lower_map.get(str(v).strip().lower(), v)

    return series.map(_map_one)


def drop_values(series: pd.Series, drops: Optional[Iterable]) -> pd.Series:
    """Mask specific values inside a Series."""
    if not drops:
        return series
    drop_set = {str(d).strip() for d in drops if d not in (None, "", np.nan)}
    return series.mask(series.astype(str).str.strip().isin(drop_set), other=np.nan)


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric, coercing invalid entries to NaN."""
    return pd.to_numeric(series, errors="coerce")
