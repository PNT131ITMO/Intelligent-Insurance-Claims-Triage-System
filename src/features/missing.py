import pandas as pd

def categorical_tokens(series: pd.Series, missing_token: str) -> pd.Series:
    non_missing = series.dropna().astype(str)
    if missing_token in set(non_missing.unique()):
        raise ValueError(f"Configured missing token {missing_token!r} occurs in column {series.name!r}")
    result = series.astype("string").fillna(missing_token)
    return result.astype(str)

