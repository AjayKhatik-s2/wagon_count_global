"""JSON serialization helpers for numpy/pandas types."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def json_safe(obj: Any) -> Any:
    """JSON serializer fallback for objects not handled by default encoder."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
