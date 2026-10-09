from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .data import ModelingDataset
from .training import BaselineRunResult, CatBoostBaselineConfig, CrossValidationConfig


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_baseline_outputs(
    sanity_result: BaselineRunResult,
    catboost_result: BaselineRunResult,
    dataset: ModelingDataset,
    cv_config: CrossValidationConfig,
    model_config: CatBoostBaselineConfig,
    processed_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics = pd.concat([sanity_result.fold_metrics, catboost_result.fold_metrics], ignore_index=True)
    summary_metrics = pd.concat([sanity_result.summary_metrics, catboost_result.summary_metrics], ignore_index=True)
    overall_metrics = pd.DataFrame([
        {
            "model": sanity_result.model_name,
            "feature_set": sanity_result.feature_specification.name,
            **sanity_result.overall_metrics,
        },
        {
            "model": catboost_result.model_name,
            "feature_set": catboost_result.feature_specification.name,
            **catboost_result.overall_metrics,
        },
    ])
    oof_predictions = sanity_result.oof_predictions.merge(
        catboost_result.oof_predictions[[dataset.train_ids.name, "catboost_probability"]],
        on=dataset.train_ids.name,
        how="inner",
        validate="one_to_one",
    )

    paths = {
        "fold_metrics": output_dir / "fold_metrics.csv",
        "summary_metrics": output_dir / "summary_metrics.csv",
        "overall_metrics": output_dir / "overall_metrics.csv",
        "oof_predictions": output_dir / "oof_predictions.parquet",
        "feature_importance": output_dir / "catboost_feature_importance.csv",
        "metadata": output_dir / "training_metadata.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    summary_metrics.to_csv(paths["summary_metrics"], index=False)
    overall_metrics.to_csv(paths["overall_metrics"], index=False)
    oof_predictions.to_parquet(paths["oof_predictions"], index=False)
    if catboost_result.feature_importance is None:
        raise ValueError("CatBoost feature importance is missing")
    catboost_result.feature_importance.to_csv(paths["feature_importance"], index=False)

    if catboost_result.test_predictions is not None:
        paths["test_predictions"] = output_dir / "test_predictions.parquet"
        catboost_result.test_predictions.to_parquet(paths["test_predictions"], index=False)

    source_names = (
        "X_train_expert.parquet",
        "y_train.parquet",
        "X_test_expert.parquet",
        "train_ids.parquet",
        "test_ids.parquet",
        "feature_manifest.json",
        "preprocessing_metadata.json",
    )
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "CatBoostClassifier",
        "primary_metric": "log_loss",
        "feature_set": catboost_result.feature_specification.name,
        "feature_count": len(catboost_result.feature_specification.feature_names),
        "categorical_feature_count": len(catboost_result.feature_specification.categorical_names),
        "feature_names": list(catboost_result.feature_specification.feature_names),
        "categorical_feature_names": list(catboost_result.feature_specification.categorical_names),
        "categorical_feature_indices": list(catboost_result.feature_specification.categorical_indices),
        "cross_validation": asdict(cv_config),
        "catboost": asdict(model_config),
        "source_checksums": {name: sha256_file(processed_dir / name) for name in source_names},
        "library_versions": {
            package: version(package)
            for package in ("catboost", "numpy", "pandas", "pyarrow", "scikit-learn")
        },
        "fold_models": [str(path) for path in catboost_result.model_paths],
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths

