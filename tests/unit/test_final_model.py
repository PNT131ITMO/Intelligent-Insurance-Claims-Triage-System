"""Fast contract tests for final-training selection and raw-input inference."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import BNPFeaturePipeline
from src.models.data import resolve_feature_set
from src.models.final_training import choose_final_iterations, train_final_bundle
from src.models.inference import FinalCatBoostPredictor


def test_final_iterations_use_zero_based_cv_best_iteration():
    frame = pd.DataFrame({
        "model": ["catboost"] * 5,
        "feature_set": ["expert_without_cat11_v22"] * 5,
        "fold": [1, 2, 3, 4, 5],
        "best_iteration": [971, 1596, 1188, 1780, 1986],
    })
    result, counts = choose_final_iterations(frame, "expert_without_cat11_v22", 5, 2000)
    assert result == 1597
    assert counts == (972, 1597, 1189, 1781, 1987)
    with pytest.raises(ValueError, match="every CV fold"):
        choose_final_iterations(pd.concat([frame, frame.iloc[[0]]]), "expert_without_cat11_v22", 5, 2000)


@pytest.fixture(params=[False, True], ids=["no_snapshot", "snapshot"])
def tiny_final_bundle(tmp_path, request):
    rows = 40
    raw = pd.DataFrame({
        "ID": np.arange(rows),
        "v10": np.linspace(0.1, 3.0, rows),
        "v22": np.tile(["A", "B", None, "C"], rows // 4),
    })
    y = pd.Series(np.tile([0, 1], rows // 2), name="target")
    pipeline = BNPFeaturePipeline({"mode": "minimal", "target_encoding": {"enabled": False}})
    X = pipeline.fit_transform(raw)
    specification = resolve_feature_set(pipeline.feature_manifest(), "expert_full")
    pipeline_path = tmp_path / "source_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    bundle_dir = tmp_path / "final"
    train_final_bundle(
        X, y, specification, pipeline_path, bundle_dir,
        {"iterations": 8, "depth": 2, "learning_rate": 0.1, "loss_function": "Logloss",
         "random_seed": 42, "thread_count": 1, "allow_writing_files": False},
        {"test": "tiny smoke"}, verbose=False,
        snapshot_enabled=request.param, snapshot_interval_seconds=1,
    )
    return bundle_dir, pipeline


def test_final_bundle_predicts_raw_without_refitting(tiny_final_bundle):
    bundle_dir, _ = tiny_final_bundle
    predictor = FinalCatBoostPredictor.load(bundle_dir)
    raw_new = pd.DataFrame({"ID": [100, 101, 102], "v10": [0.7, 1.1, np.nan], "v22": ["A", "UNSEEN", None]})
    before = predictor.preprocessor.fit_count_
    result = predictor.predict_raw(raw_new)
    assert result.columns.tolist() == ["ID", "probability_target_1"]
    assert result["ID"].tolist() == [100, 101, 102]
    assert result["probability_target_1"].between(0, 1).all()
    assert predictor.preprocessor.fit_count_ == before
    assert json.loads((bundle_dir / "metadata.json").read_text())["tree_count"] == 8

    with pytest.raises(ValueError, match="target"):
        predictor.predict_raw(raw_new.assign(target=0))
    with pytest.raises(ValueError, match="unique IDs"):
        predictor.predict_raw(raw_new.assign(ID=[100, 100, 102]))
    with pytest.raises(ValueError, match="missing selected features"):
        predictor.predict_raw(raw_new.drop(columns="v22"))


def test_final_bundle_rejects_tampered_checksum(tiny_final_bundle):
    bundle_dir, _ = tiny_final_bundle
    metadata_path = bundle_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum mismatch"):
        FinalCatBoostPredictor.load(bundle_dir)
