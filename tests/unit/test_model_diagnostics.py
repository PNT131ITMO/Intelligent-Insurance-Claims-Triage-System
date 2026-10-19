import numpy as np
import pandas as pd
import pytest

from src.models.diagnostics import (
    calibration_bins,
    expected_calibration_error,
    highest_loss_cases,
    segment_summary,
    threshold_summary,
    validate_oof_predictions,
)


@pytest.fixture
def small_oof():
    return pd.DataFrame({
        "ID": [10, 11, 12, 13],
        "target": [0, 0, 1, 1],
        "fold": [1, 2, 1, 2],
        "catboost_probability": [0.1, 0.8, 0.3, 0.9],
    })


def test_calibration_bins_include_empty_bins_and_one_at_upper_edge():
    table = calibration_bins([0, 1, 1], [0.0, 0.2, 1.0], n_bins=5)
    assert table["count"].tolist() == [1, 1, 0, 0, 1]
    assert table["positive_count"].tolist() == [0, 1, 0, 0, 1]
    assert np.isnan(table.loc[2, "observed_rate"])
    assert expected_calibration_error(table) == pytest.approx((0.8 + 0.0) / 3)


def test_threshold_summary_counts_all_outcomes(small_oof):
    row = threshold_summary(
        small_oof["target"], small_oof["catboost_probability"], [0.5]
    ).iloc[0]
    assert (row.true_positive, row.false_positive, row.true_negative, row.false_negative) == (
        1, 1, 1, 1
    )
    assert row.precision_class_1 == pytest.approx(0.5)
    assert row.recall_class_1 == pytest.approx(0.5)


def test_oof_validation_rejects_duplicate_ids_and_segment_misalignment(small_oof):
    validate_oof_predictions(small_oof)
    duplicated = small_oof.copy()
    duplicated.loc[1, "ID"] = 10
    with pytest.raises(ValueError, match="exactly one"):
        validate_oof_predictions(duplicated)
    with pytest.raises(ValueError, match="aligned"):
        segment_summary(small_oof, pd.Series(["a"] * 4, index=[1, 2, 3, 4]))


def test_segment_summary_and_highest_loss_case(small_oof):
    segment = pd.Series(["a", "b", "a", "b"], index=small_oof.index)
    table = segment_summary(small_oof, segment, minimum_rows=2)
    assert table["rows"].tolist() == [2, 2]
    assert table["segment"].tolist() == ["a", "b"]
    assert highest_loss_cases(small_oof, top_n=1).loc[0, "ID"] == 11
