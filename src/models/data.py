from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelingDataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    train_ids: pd.Series
    test_ids: pd.Series
    manifest: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class FeatureSpecification:
    name: str
    feature_names: tuple[str, ...]
    categorical_names: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    feature_groups: tuple[str, ...]


EXCLUDED_GROUPS = {
    "expert_without_cat11_v22": {"high_order_v22_interactions"},
    "expert_without_num_pair_sums": {"numerical_pair_sums"},
    "expert_without_rounded_numeric": {"rounded_numeric_categories"},
    "expert_without_cat3_v22": {"categorical_3way_v22"},
}


def load_modeling_dataset(processed_dir: Path) -> ModelingDataset:
    required = {
        "X_train": processed_dir / "X_train_expert.parquet",
        "y_train": processed_dir / "y_train.parquet",
        "X_test": processed_dir / "X_test_expert.parquet",
        "train_ids": processed_dir / "train_ids.parquet",
        "test_ids": processed_dir / "test_ids.parquet",
        "manifest": processed_dir / "feature_manifest.json",
        "metadata": processed_dir / "preprocessing_metadata.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing modeling artifact(s): {missing}")

    X_train = pd.read_parquet(required["X_train"])
    y_frame = pd.read_parquet(required["y_train"])
    X_test = pd.read_parquet(required["X_test"])
    train_ids_frame = pd.read_parquet(required["train_ids"])
    test_ids_frame = pd.read_parquet(required["test_ids"])
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))

    if y_frame.shape[1] != 1:
        raise ValueError("y_train must contain exactly one column")
    if train_ids_frame.shape[1] != 1 or test_ids_frame.shape[1] != 1:
        raise ValueError("ID artifacts must contain exactly one column")
    if X_train.columns.tolist() != X_test.columns.tolist():
        raise ValueError("Train and test feature schemas differ")
    if not X_train.columns.is_unique:
        raise ValueError("Feature names must be unique")
    if len(X_train) != len(y_frame) or len(X_train) != len(train_ids_frame):
        raise ValueError("Training features, target, and IDs are not aligned")
    if len(X_test) != len(test_ids_frame):
        raise ValueError("Test features and IDs are not aligned")

    expected_features = manifest.get("final_feature_names")
    if expected_features != X_train.columns.tolist():
        raise ValueError("Feature manifest does not match the processed feature order")

    categorical_names = manifest.get("categorical_feature_names", [])
    categorical_indices = manifest.get("categorical_feature_indices", [])
    if [X_train.columns[index] for index in categorical_indices] != categorical_names:
        raise ValueError("Categorical indices do not match categorical feature names")
    if X_train[categorical_names].isna().any().any():
        raise ValueError("Categorical features must not contain missing values")

    y_train = y_frame.iloc[:, 0]
    if set(y_train.unique()) != {0, 1}:
        raise ValueError("Target must contain exactly the binary labels 0 and 1")

    return ModelingDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        train_ids=train_ids_frame.iloc[:, 0],
        test_ids=test_ids_frame.iloc[:, 0],
        manifest=manifest,
        metadata=metadata,
    )


def resolve_feature_set(manifest: dict[str, Any], name: str) -> FeatureSpecification:
    feature_names = manifest["final_feature_names"]
    feature_origin = manifest["feature_origin"]
    categorical = set(manifest["categorical_feature_names"])

    if name == "expert_full":
        selected = list(feature_names)
    elif name == "expert_base_only":
        included = {"original_numerical", "categorical_1way"}
        selected = [feature for feature in feature_names if feature_origin[feature] in included]
    elif name in EXCLUDED_GROUPS:
        excluded = EXCLUDED_GROUPS[name]
        selected = [feature for feature in feature_names if feature_origin[feature] not in excluded]
    else:
        supported = ["expert_full", "expert_base_only", *EXCLUDED_GROUPS]
        raise ValueError(f"Unknown feature set {name!r}; expected one of {supported}")

    if not selected:
        raise ValueError(f"Feature set {name!r} is empty")

    categorical_names = [feature for feature in selected if feature in categorical]
    positions = {feature: index for index, feature in enumerate(selected)}
    groups = tuple(dict.fromkeys(feature_origin[feature] for feature in selected))
    return FeatureSpecification(
        name=name,
        feature_names=tuple(selected),
        categorical_names=tuple(categorical_names),
        categorical_indices=tuple(positions[feature] for feature in categorical_names),
        feature_groups=groups,
    )

