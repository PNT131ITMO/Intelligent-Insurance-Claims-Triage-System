from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import comb
from typing import Any

import numpy as np
import pandas as pd

from .categorical import (
    OOFTargetEncoder,
    StableOrdinalEncoder,
    category_hashes,
    combined_tokens,
    hash_tokens,
    interaction_definitions,
)
from .identify_types import identify_feature_types
from .numerical import numerical_pair_definitions, numerical_pair_sums, rounded_numeric_tokens


EXPERT_SELECTED_FEATURES = [
    "v10", "v12", "v14", "v21", "v22", "v24", "v30", "v31", "v34", "v38",
    "v40", "v47", "v50", "v52", "v56", "v62", "v66", "v72", "v75", "v79",
    "v91", "v112", "v113", "v114", "v129",
]


@dataclass(frozen=True)
class FeatureBlock:
    name: str
    columns: tuple[str, ...]


class BNPFeaturePipeline:
    def __init__(self, config: dict[str, Any], *, target_column: str = "target", id_column: str = "ID",) -> None:
        self.config = deepcopy(config)
        self.target_column = target_column
        self.id_column = id_column
        self.mode = str(config.get("mode", "expert"))

        if self.mode not in {"expert", "minimal"}:
            raise ValueError("mode must be 'expert' or 'minimal'")

        self.unknown_value = int(config.get("unknown_category_value", -1))
        self.missing_token = str(config.get("missing_category_token", "__MISSING__"))
        self.ordinal_encoder_ = StableOrdinalEncoder(self.unknown_value)
        self.target_encoder_: OOFTargetEncoder | None = None
        self.is_fitted_ = False
        self.fit_count_ = 0

    def _enabled(self, key: str, default: bool = True) -> bool:
        return bool(self.config.get("expert_features", {}).get(key, default))

    def _validate_input(self, frame: pd.DataFrame) -> None:
        forbidden = [column for column in (self.target_column,) if column in frame]
        if forbidden:
            raise ValueError(f"Target column must not be passed as a feature: {forbidden}")
        if self.id_column not in frame:
            raise ValueError(f"Required ID column {self.id_column!r} is missing")

    def _prepare_definitions(self, frame: pd.DataFrame) -> None:
        candidates = (
            EXPERT_SELECTED_FEATURES
            if self.mode == "expert" and self._enabled("use_selected_features")
            else [column for column in frame.columns if column != self.id_column]
        )
        self.selected_features_ = list(candidates)
        self.numerical_features_, self.categorical_features_ = identify_feature_types(
            frame, self.selected_features_
        )
        self.cat2_definitions_ = (
            interaction_definitions(self.categorical_features_, 2)
            if self.mode == "expert" and self._enabled("categorical_2way")
            else []
        )
        self.cat3_definitions_ = (
            interaction_definitions(self.categorical_features_, 3, include=["v22"])
            if self.mode == "expert"
            and self._enabled("categorical_3way_v22")
            and "v22" in self.categorical_features_
            else []
        )
        self.cat11_definitions_ = (
            interaction_definitions(self.categorical_features_, 11, include=["v22"])
            if self.mode == "expert"
            and self._enabled("high_order_v22_interactions", False)
            and "v22" in self.categorical_features_
            else []
        )
        limit = int(self.config.get("expert_features", {}).get("high_order_max_combinations", 1000))
        if len(self.cat11_definitions_) > limit:
            raise ValueError(
                f"Refusing to generate {len(self.cat11_definitions_)} high-order interactions; "
                f"configured limit is {limit}"
            )
        self.sum_definitions_ = (
            numerical_pair_definitions(self.numerical_features_)
            if self.mode == "expert" and self._enabled("numerical_pair_sums")
            else []
        )

    def _categorical_token_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        tokens: dict[str, pd.Series] = {}
        if self.mode == "minimal" or self._enabled("categorical_1way"):
            for column in self.categorical_features_:
                tokens[f"cat1__{column}"] = category_hashes(frame[column], self.missing_token)
        for prefix, definitions in (
            ("cat2", self.cat2_definitions_),
            ("cat3", self.cat3_definitions_),
            ("cat11", self.cat11_definitions_),
        ):
            for definition in definitions:
                name = f"{prefix}__{'__'.join(definition)}"
                tokens[name] = combined_tokens(frame, definition, self.missing_token)
        if self.mode == "expert" and self._enabled("rounded_numeric_categories"):
            for column in self.numerical_features_:
                tokens[f"rnum__{column}"] = hash_tokens(
                    rounded_numeric_tokens(frame[column], self.missing_token).rename(column)
                )
        return pd.DataFrame(tokens, index=frame.index)

    def _numeric_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        original = frame[self.numerical_features_].copy()
        sums = numerical_pair_sums(frame, self.sum_definitions_)
        return pd.concat([original, sums], axis=1)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series | None = None) -> "BNPFeaturePipeline":
        """Fit every learned mapping using training rows only."""
        self._validate_input(X_train)
        self.ordinal_encoder_ = StableOrdinalEncoder(self.unknown_value)
        self.target_encoder_ = None
        self._prepare_definitions(X_train)
        token_frame = self._categorical_token_frame(X_train)
        for column in token_frame.columns:
            self.ordinal_encoder_.fit_series(column, token_frame[column])
        target_cfg = self.config.get("target_encoding", {})
        self.target_encoding_enabled_ = bool(target_cfg.get("enabled", False))
        if self.target_encoding_enabled_:
            if y_train is None:
                raise ValueError("y_train is required when target encoding is enabled")
            ordinal = self._encode_tokens(token_frame)
            self.target_encoder_ = OOFTargetEncoder(
                n_splits=int(target_cfg.get("n_splits", 10)),
                random_state=int(target_cfg.get("random_state", 42)),
            )
            self.target_encoder_.fit(ordinal, y_train)
        numeric_columns = list(self._numeric_frame(X_train.iloc[:0]).columns)
        categorical_columns = list(token_frame.columns)
        self.feature_blocks_ = self._make_blocks(numeric_columns, categorical_columns)
        self.feature_names_out_ = numeric_columns + categorical_columns
        if len(self.feature_names_out_) != len(set(self.feature_names_out_)):
            raise RuntimeError("Generated duplicate feature names")
        self.is_fitted_ = True
        self.fit_count_ += 1
        return self

    def _make_blocks(
        self, numeric_columns: list[str], categorical_columns: list[str]
    ) -> list[FeatureBlock]:
        groups: list[FeatureBlock] = []
        predicates = [
            ("original_numerical", lambda name: name in self.numerical_features_),
            ("categorical_1way", lambda name: name.startswith("cat1__")),
            ("categorical_2way", lambda name: name.startswith("cat2__")),
            ("categorical_3way_v22", lambda name: name.startswith("cat3__")),
            ("rounded_numeric_categories", lambda name: name.startswith("rnum__")),
            ("numerical_pair_sums", lambda name: name.startswith("num2sum__")),
            ("high_order_v22_interactions", lambda name: name.startswith("cat11__")),
        ]
        all_columns = numeric_columns + categorical_columns
        for group, predicate in predicates:
            columns = tuple(column for column in all_columns if predicate(column))
            if columns:
                groups.append(FeatureBlock(group, columns))
        return groups

    def _encode_tokens(self, tokens: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                column: self.ordinal_encoder_.transform_series(column, tokens[column])
                for column in tokens.columns
            },
            index=tokens.index,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform without changing fitted mappings or feature definitions."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before transform")
        self._validate_input(X)
        missing = [column for column in self.selected_features_ if column not in X]
        if missing:
            raise ValueError(f"Transform input is missing fitted columns: {missing}")
        numeric = self._numeric_frame(X)
        encoded = self._encode_tokens(self._categorical_token_frame(X))
        if self.target_encoding_enabled_:
            assert self.target_encoder_ is not None
            encoded = self.target_encoder_.transform(encoded)
        result = pd.concat([numeric, encoded], axis=1)
        return result.loc[:, self.feature_names_out_]

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series | None = None
    ) -> pd.DataFrame:
        """Fit and transform training rows; target encoding, when enabled, is OOF."""
        self.fit(X_train, y_train)
        numeric = self._numeric_frame(X_train)
        encoded = self._encode_tokens(self._categorical_token_frame(X_train))
        if self.target_encoding_enabled_:
            assert self.target_encoder_ is not None and y_train is not None
            encoded = self.target_encoder_.fit_transform(encoded, y_train)
        return pd.concat([numeric, encoded], axis=1).loc[:, self.feature_names_out_]

    def get_feature_names_out(self) -> list[str]:
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted")
        return list(self.feature_names_out_)

    def feature_manifest(self) -> dict[str, Any]:
        """Return JSON-serializable lineage and feature-count metadata."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted")
        categorical = [] if self.target_encoding_enabled_ else [
            column for block in self.feature_blocks_
            if block.name in {
                "categorical_1way", "categorical_2way", "categorical_3way_v22",
                "rounded_numeric_categories", "high_order_v22_interactions",
            }
            for column in block.columns
        ]
        origin = {
            column: block.name for block in self.feature_blocks_ for column in block.columns
        }
        return {
            "preprocessing_mode": self.mode,
            "raw_selected_features": list(self.selected_features_),
            "numerical_features": list(self.numerical_features_),
            "raw_categorical_features": list(self.categorical_features_),
            "final_feature_names": list(self.feature_names_out_),
            "categorical_feature_names": categorical,
            "categorical_feature_indices": [self.feature_names_out_.index(c) for c in categorical],
            "feature_origin": origin,
            "configuration": deepcopy(self.config),
            "random_seed": int(self.config.get("random_seed", 42)),
            "generated_feature_counts": {
                block.name: len(block.columns) for block in self.feature_blocks_
            },
            "total_features": len(self.feature_names_out_),
        }

    def expected_counts(self) -> dict[str, int]:
        if not self.is_fitted_:
            raise RuntimeError("Pipeline is not fitted")
        n_cat, n_num = len(self.categorical_features_), len(self.numerical_features_)
        return {
            "categorical_base": n_cat,
            "categorical_pairs_theoretical": comb(n_cat, 2) if n_cat >= 2 else 0,
            "categorical_pairs_generated": len(self.cat2_definitions_),
            "v22_triples_theoretical": comb(n_cat - 1, 2) if n_cat >= 3 and "v22" in self.categorical_features_ else 0,
            "v22_triples_generated": len(self.cat3_definitions_),
            "v22_order11_theoretical": comb(n_cat - 1, 10) if n_cat >= 11 and "v22" in self.categorical_features_ else 0,
            "v22_order11_generated": len(self.cat11_definitions_),
            "numerical_base": n_num,
            "numerical_pairs_theoretical": comb(n_num, 2) if n_num >= 2 else 0,
            "numerical_pairs_generated": len(self.sum_definitions_),
        }
