from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.data import ModelingDataset, resolve_feature_set
from src.models.metrics import binary_probability_metrics
from src.models.training import (
    CatBoostBaselineConfig,
    CrossValidationConfig,
    run_catboost_baseline,
    run_sanity_baseline,
)


@pytest.fixture
def modeling_dataset():
    rows = 40
    X_train = pd.DataFrame({
        "v10": np.linspace(0, 1, rows),
        "cat1__v22": np.tile(np.arange(4, dtype=np.int32), rows // 4),
        "cat2__v22__v24": np.tile(np.arange(8, dtype=np.int32), rows // 8),
    })
    X_test = X_train.iloc[:8].reset_index(drop=True)
    y_train = pd.Series(np.tile([0, 1], rows // 2), name="target")
    manifest = {
        "final_feature_names": X_train.columns.tolist(),
        "categorical_feature_names": ["cat1__v22", "cat2__v22__v24"],
        "categorical_feature_indices": [1, 2],
        "feature_origin": {
            "v10": "original_numerical",
            "cat1__v22": "categorical_1way",
            "cat2__v22__v24": "categorical_2way",
        },
    }
    return ModelingDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        train_ids=pd.Series(np.arange(rows), name="ID"),
        test_ids=pd.Series(np.arange(100, 108), name="ID"),
        manifest=manifest,
        metadata={"target_name": "target", "id_name": "ID"},
    )


def test_feature_set_preserves_order_and_recomputes_categorical_indices(modeling_dataset):
    feature_set = resolve_feature_set(modeling_dataset.manifest, "expert_base_only")
    assert feature_set.feature_names == ("v10", "cat1__v22")
    assert feature_set.categorical_names == ("cat1__v22",)
    assert feature_set.categorical_indices == (1,)


def test_probability_metrics_reject_invalid_probabilities():
    with pytest.raises(ValueError, match="within"):
        binary_probability_metrics([0, 1], [0.2, 1.1])


def test_sanity_baseline_produces_complete_deterministic_oof(modeling_dataset):
    feature_set = resolve_feature_set(modeling_dataset.manifest, "expert_base_only")
    cv_config = CrossValidationConfig(n_splits=2, random_state=42)
    first = run_sanity_baseline(modeling_dataset, feature_set, cv_config)
    second = run_sanity_baseline(modeling_dataset, feature_set, cv_config)
    assert first.oof_predictions["fold"].between(1, 2).all()
    assert first.oof_predictions["dummy_probability"].notna().all()
    pd.testing.assert_frame_equal(first.oof_predictions, second.oof_predictions)


def test_catboost_smoke_test_uses_categorical_semantics(modeling_dataset, tmp_path: Path):
    feature_set = resolve_feature_set(modeling_dataset.manifest, "expert_base_only")
    result = run_catboost_baseline(
        modeling_dataset,
        feature_set,
        CrossValidationConfig(n_splits=2, random_state=42),
        CatBoostBaselineConfig(
            parameters={
                "iterations": 10,
                "depth": 3,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "random_seed": 42,
                "allow_writing_files": False,
                "thread_count": 1,
            },
            early_stopping_rounds=5,
            verbose=False,
        ),
        model_dir=tmp_path,
    )
    assert result.oof_predictions["catboost_probability"].between(0, 1).all()
    assert result.test_predictions is not None
    assert len(result.model_paths) == 2
    assert all(path.exists() for path in result.model_paths)
    resumed = run_catboost_baseline(
        modeling_dataset,
        feature_set,
        CrossValidationConfig(n_splits=2, random_state=42),
        CatBoostBaselineConfig(
            parameters={
                "iterations": 10,
                "depth": 3,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "random_seed": 42,
                "allow_writing_files": False,
                "thread_count": 1,
            },
            early_stopping_rounds=5,
            verbose=False,
        ),
        model_dir=tmp_path,
        resume=True,
    )
    assert resumed.fold_metrics["resumed"].all()
    np.testing.assert_allclose(
        result.oof_predictions["catboost_probability"],
        resumed.oof_predictions["catboost_probability"],
    )
