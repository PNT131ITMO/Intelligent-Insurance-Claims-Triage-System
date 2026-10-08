from collections.abc import Sequence

import pandas as pd

def identify_feature_types(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[list[str], list[str]]:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    numerical = [column for column in columns if pd.api.types.is_numeric_dtype(frame[column])]
    categorical = [column for column in columns if column not in numerical]
    return numerical, categorical

