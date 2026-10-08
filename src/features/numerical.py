from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd

def rounded_numeric_token(value: object, missing_token: str) -> str:
    if pd.isna(value): return missing_token
    number = float(value)
    if math.isinf(number):
        return "__POS_INF__" if number > 0 else "__NEG_INF__"
    text = np.format_float_positional(number, unique=True, trim="-")
    sign = ""
    if text.startswith("-"):
        sign, text = "-", text[1:]
    integer, dot, fraction = text.partition(".")
    if not dot or len(fraction) <= 2:
        return sign + integer
    retained = fraction[:-2].rstrip("0")
    return sign + integer + (f".{retained}" if retained else "")

def rounded_numeric_tokens(series: pd.Series, missing_token: str) -> pd.Series:
    return series.map(lambda value: rounded_numeric_token(value, missing_token))

def numerical_pair_definitions(columns: list[str]) -> list[tuple[str, str]]:
    return list(combinations(columns, 2))

def numerical_pair_sums(frame: pd.DataFrame, definitions: list[tuple[str, str]]) -> pd.DataFrame:
    data = {
        f"num2sum__{left}__{right}": frame[left] + frame[right]
        for left, right in definitions
    }
    return pd.DataFrame(data, index=frame.index)