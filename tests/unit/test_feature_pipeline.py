import numpy as np
import pandas as pd
import pytest
from joblib import hash as joblib_hash

from src.features.pipeline import BNPFeaturePipeline


@pytest.fixture
def config():
    return {
        "mode": "minimal",
        "expert_features": {},
        "target_encoding": {"enabled": False, "n_splits": 2, "random_state": 42},
        "unknown_category_value": -1,
        "missing_category_token": "__MISSING__",
        "random_seed": 42,
    }


@pytest.fixture
def train_frame():
    return pd.DataFrame({"ID": [1, 2, 3, 4], "num": [1.0, 2.0, np.nan, 4.0], "cat": ["b", None, "a", "b"]})


def test_missing_and_unknown_categories(config, train_frame):
    pipeline = BNPFeaturePipeline(config).fit(train_frame)
    transformed_train = pipeline.transform(train_frame)
    transformed_future = pipeline.transform(pd.DataFrame({"ID": [5, 6], "num": [5.0, 6.0], "cat": [None, "new"]}))
    assert transformed_train.loc[1, "cat1__cat"] != -1
    assert transformed_future.loc[0, "cat1__cat"] == transformed_train.loc[1, "cat1__cat"]
    assert transformed_future.loc[1, "cat1__cat"] == -1


def test_transform_does_not_refit_and_columns_are_deterministic(config, train_frame):
    pipeline = BNPFeaturePipeline(config)
    first = pipeline.fit_transform(train_frame)
    mappings_hash = joblib_hash(pipeline.ordinal_encoder_.mappings_)
    fit_count = pipeline.fit_count_
    second = pipeline.transform(train_frame)
    pd.testing.assert_frame_equal(first, second)
    assert joblib_hash(pipeline.ordinal_encoder_.mappings_) == mappings_hash
    assert pipeline.fit_count_ == fit_count == 1
    assert first.columns.tolist() == pipeline.get_feature_names_out()


def test_expert_interactions_and_schema_consistency():
    categorical = ["v22", "v24", "v30"]
    numerical = ["v10", "v12"]
    frame = pd.DataFrame({"ID": [1, 2, 3], "v10": [1.1, 2.2, 3.3], "v12": [2.0, 3.0, 4.0], "v22": ["A", "B", "A"], "v24": ["X", "Y", "X"], "v30": ["K", None, "L"]})
    config = {
        "mode": "expert",
        "expert_features": {
            "use_selected_features": False,
            "categorical_1way": True,
            "categorical_2way": True,
            "categorical_3way_v22": True,
            "rounded_numeric_categories": True,
            "numerical_pair_sums": True,
            "high_order_v22_interactions": False,
        },
        "target_encoding": {"enabled": False},
        "unknown_category_value": -1,
        "missing_category_token": "__MISSING__",
    }
    pipeline = BNPFeaturePipeline(config)
    train = pipeline.fit_transform(frame)
    test = pipeline.transform(frame.iloc[[0, 2]].copy())
    assert "cat2__v22__v24" in train
    assert "cat3__v22__v24__v30" in train
    assert "num2sum__v10__v12" in train
    assert train.columns.tolist() == test.columns.tolist()
    assert "ID" not in train and "target" not in train


def test_target_column_is_rejected(config, train_frame):
    with pytest.raises(ValueError, match="Target column"):
        BNPFeaturePipeline(config).fit(train_frame.assign(target=[0, 1, 0, 1]))


def test_target_encoding_is_out_of_fold(train_frame, config):
    config["target_encoding"] = {"enabled": True, "n_splits": 2, "random_state": 42}
    expanded = pd.DataFrame({
        "ID": np.arange(10),
        "num": np.arange(10, dtype=float),
        "cat": [f"unique-{index}" for index in range(10)],
    })
    y = pd.Series([0, 1] * 5, name="target")
    pipeline = BNPFeaturePipeline(config)
    output = pipeline.fit_transform(expanded, y)
    inference_style = pipeline.transform(expanded)
    assert output["cat1__cat"].dtype == np.float32
    np.testing.assert_allclose(output["cat1__cat"], y.mean())
    np.testing.assert_allclose(inference_style["cat1__cat"], y.astype(float))
    assert pipeline.fit_count_ == 1
