from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .missing import categorical_tokens

def interaction_definitions(columns: Sequence[str], order: int, include: Sequence[str] = ()) -> list[tuple[str, ...]]:
    include_tuple = tuple(include)

    if len(set(include_tuple)) != len(include_tuple):
        raise ValueError("include columns must be unique")
    if any(column not in columns for column in include_tuple):
        raise ValueError("include columns must be present in columns")

    choose = order - len(include_tuple)

    remaining = [column for column in columns if column not in include_tuple]

    if choose < 0 or choose > len(remaining):
        return []
    return [include_tuple + combo for combo in combinations(remaining, choose)]

def combined_tokens(frame: pd.DataFrame, columns: Sequence[str], missing_token: str) -> pd.Series:
    token_frame = pd.concat([categorical_tokens(frame[column], missing_token) for column in columns], axis=1)

    return pd.util.hash_pandas_object(
        token_frame, index=False, hash_key="0123456789123456"
    ).rename("__".join(columns))

def category_hashes(series: pd.Series, missing_token: str) -> pd.Series:
    tokens = categorical_tokens(series, missing_token)
    return hash_tokens(tokens)

def hash_tokens(tokens: pd.Series) -> pd.Series:
    return pd.util.hash_pandas_object(
        tokens, index=False, hash_key="0123456789123456"
    ).rename(tokens.name)

class StableOrdinalEncoder:
    def __init__(self, unknown_value: int = -1) -> None:
        self.unknown_value = int(unknown_value)
        self.mappings_: dict[str, np.ndarray] = {}

    def fit_series(self, name: str, values: pd.Series) -> None:
        self.mappings_[name] = np.unique(values.to_numpy(dtype=np.uint64, copy=False))

    def transform_series(self, name: str, values: pd.Series) -> pd.Series:
        if name not in self.mappings_:
            raise RuntimeError(f"Encoder has not been fitted for {name!r}")
            
        known_values = self.mappings_[name]
        raw = values.to_numpy(dtype=np.uint64, copy=False)
        positions = np.searchsorted(known_values, raw)
        in_bounds = positions < len(known_values)

        matched = np.zeros(len(raw), dtype=bool)
        matched[in_bounds] = known_values[positions[in_bounds]] == raw[in_bounds]

        encoded = np.full(len(raw), self.unknown_value, dtype=np.int32)
        encoded[matched] = positions[matched].astype(np.int32)

        return pd.Series(encoded, index=values.index, name=name)

    def fit_transform_series(self, name: str, values: pd.Series) -> pd.Series:
        self.fit_series(name, values)
        return self.transform_series(name, values)

class OOFTargetEncoder:
    def __init__(self, n_splits: int = 10, random_state: int = 42) -> None:
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)
        self.global_mean_: float | None = None
        self.mappings_: dict[str, dict[Any, float]] = {}

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> "OOFTargetEncoder":
        if len(frame) != len(y):
            raise ValueError("X and y row counts differ")

        target = pd.Series(np.asarray(y), index=frame.index, dtype=float)

        self.global_mean_ = float(target.mean())
        self.mappings_ = {}

        for column in frame.columns:
            full_map = target.groupby(frame[column]).mean()
            self.mappings_[column] = {key: float(value) for key, value in full_map.items()}
        return self

    def fit_transform(self, frame: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if len(frame) != len(y): raise ValueError("X and y row counts differ")
        self.fit(frame, y)

        target = pd.Series(np.asarray(y), index=frame.index, dtype=float)
        splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        output = pd.DataFrame(index=frame.index)
        
        for column in frame.columns:
            encoded = pd.Series(index=frame.index, dtype=np.float32)
            for fit_positions, valid_positions in splitter.split(frame, target):
                fit_index = frame.index[fit_positions]
                valid_index = frame.index[valid_positions]
                fold_map = target.loc[fit_index].groupby(frame.loc[fit_index, column]).mean()
                encoded.loc[valid_index] = frame.loc[valid_index, column].map(fold_map)
            output[column] = encoded.fillna(self.global_mean_).astype(np.float32)
        return output

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.global_mean_ is None:
            raise RuntimeError("Target encoder is not fitted")
        output = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            if column not in self.mappings_:
                raise ValueError(f"Unexpected target-encoding column: {column}")
            output[column] = (
                frame[column].map(self.mappings_[column]).fillna(self.global_mean_).astype(np.float32)
            )
        return output
