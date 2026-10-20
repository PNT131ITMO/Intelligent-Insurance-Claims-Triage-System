from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier, Pool
import joblib
import numpy as np
import pandas as pd

from .artifacts import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "models" / "artifacts" / "final_expert_without_cat11_v22"


class FinalCatBoostPredictor:

    def __init__(self, model: CatBoostClassifier, preprocessor: Any, metadata: dict[str, Any]):
        self.model = model
        self.preprocessor = preprocessor
        self.metadata = metadata

    @classmethod
    def load(cls, bundle_dir: Path) -> "FinalCatBoostPredictor":
        bundle_dir = Path(bundle_dir)
        metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
        model_path = bundle_dir / "model.cbm"
        pipeline_path = bundle_dir / "preprocessing.joblib"
        if sha256_file(model_path) != metadata["model_sha256"]:
            raise ValueError("Final CatBoost model checksum mismatch")
        if sha256_file(pipeline_path) != metadata["preprocessing_sha256"]:
            raise ValueError("Final preprocessing artifact checksum mismatch")

        model = CatBoostClassifier()
        model.load_model(str(model_path))
        preprocessor = joblib.load(pipeline_path)
        names = metadata["feature_names"]
        cat_indices = metadata["categorical_feature_indices"]
        if len(names) != len(set(names)) or model.feature_names_ != names:
            raise ValueError("Saved model feature schema differs from bundle metadata")
        if model.tree_count_ != metadata["tree_count"] or not np.array_equal(model.classes_, [0, 1]):
            raise ValueError("Saved model tree count or class labels differ from metadata")
        if not getattr(preprocessor, "is_fitted_", False):
            raise ValueError("Saved preprocessing pipeline is not fitted")
        if getattr(preprocessor, "target_encoding_enabled_", False):
            raise ValueError("Bundle cannot infer from an OOF target-encoded pipeline")
        if preprocessor.selected_features_ != metadata["raw_selected_features"]:
            raise ValueError("Raw feature selection differs from bundle metadata")
        if preprocessor.target_column != metadata["target_name"] or preprocessor.id_column != metadata["id_name"]:
            raise ValueError("Target or ID column differs from bundle metadata")
        all_names = preprocessor.get_feature_names_out()
        selected = set(names)
        if [name for name in all_names if name in selected] != names:
            raise ValueError("Saved pipeline cannot reproduce final feature order")
        if any(not isinstance(index, int) or index < 0 or index >= len(names) for index in cat_indices):
            raise ValueError("Invalid categorical feature indices")
        if [names[index] for index in cat_indices] != metadata["categorical_feature_names"]:
            raise ValueError("Categorical feature names differ from their indices")
        if model.get_cat_feature_indices() != cat_indices:
            raise ValueError("Categorical feature indices differ from the saved model")
        return cls(model, preprocessor, metadata)

    def input_schema(self) -> dict[str, Any]:
        numeric = set(self.preprocessor.numerical_features_)
        categorical = set(self.preprocessor.categorical_features_)
        selected = self.metadata["raw_selected_features"]
        if numeric | categorical != set(selected) or numeric & categorical:
            raise ValueError("Fitted raw feature types do not match bundle metadata")
        return {
            "model_id": self.metadata["model_sha256"][:12],
            "feature_set": self.metadata["feature_set"],
            "id_field": self.metadata["id_name"],
            "target_field_is_not_input": self.metadata["target_name"],
            "features": [
                {"name": name, "type": "number" if name in numeric else "string", "nullable": True}
                for name in selected
            ],
            "probability_field": "probability_target_1",
            "decision_threshold": self.metadata["decision_threshold"],
        }

    def predict_raw(self, raw_claims: pd.DataFrame) -> pd.DataFrame:
        id_name = self.metadata["id_name"]
        target_name = self.metadata["target_name"]
        if raw_claims.empty:
            raise ValueError("At least one raw claim is required")
        if not raw_claims.columns.is_unique:
            raise ValueError("Raw claims contain duplicated column names")
        if id_name not in raw_claims or raw_claims[id_name].isna().any() or not raw_claims[id_name].is_unique:
            raise ValueError("Raw claims must have nonmissing, unique IDs")
        if target_name in raw_claims:
            raise ValueError("Inference input must not contain the target column")
        missing = [name for name in self.metadata["raw_selected_features"] if name not in raw_claims]
        if missing:
            raise ValueError(f"Raw claims are missing selected features: {missing}")

        raw = raw_claims.loc[:, [id_name, *self.metadata["raw_selected_features"]]].copy()
        for name in self.preprocessor.numerical_features_:
            values = raw[name]
            if pd.api.types.is_bool_dtype(values) or values.dropna().map(
                lambda value: isinstance(value, (bool, np.bool_))
            ).any():
                raise ValueError(f"Numerical feature {name!r} must not contain booleans")
            try:
                converted = pd.to_numeric(values, errors="raise").astype("float64")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Numerical feature {name!r} contains invalid values") from error
            if np.isinf(converted.to_numpy()).any():
                raise ValueError(f"Numerical feature {name!r} contains infinity")
            raw[name] = converted
        for name in self.preprocessor.categorical_features_:
            if not raw[name].dropna().map(lambda value: isinstance(value, str)).all():
                raise ValueError(f"Categorical feature {name!r} must contain strings or null")

        fitted_count = self.preprocessor.fit_count_
        transformed = self.preprocessor.transform(raw)
        if self.preprocessor.fit_count_ != fitted_count:
            raise RuntimeError("Preprocessing state changed during inference")
        names = self.metadata["feature_names"]
        features = transformed.loc[:, names]
        cat_names = self.metadata["categorical_feature_names"]
        if features[cat_names].isna().any().any():
            raise ValueError("Categorical inference features contain missing values")
        pool = Pool(
            features,
            cat_features=self.metadata["categorical_feature_indices"],
            feature_names=names,
        )
        probabilities = self.model.predict_proba(pool)[:, 1]
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise RuntimeError("Model returned invalid positive-class probabilities")
        return pd.DataFrame({id_name: raw[id_name].to_numpy(), "probability_target_1": probabilities})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(os.environ.get("BNP_MODEL_BUNDLE_DIR", str(DEFAULT_BUNDLE_DIR))),
        help="Model bundle directory; defaults to BNP_MODEL_BUNDLE_DIR or the project artifact path",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw CSV or single-member ZIP containing CSV")
    parser.add_argument("--output", type=Path, required=True, help="New .parquet or .csv output file")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing predictions: {args.output}")
    if args.output.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("Prediction output must have .parquet or .csv extension")
    predictor = FinalCatBoostPredictor.load(args.bundle)
    raw = pd.read_csv(args.input, compression="infer", low_memory=False)
    predictions = predictor.predict_raw(raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".parquet":
        predictions.to_parquet(args.output, index=False)
    else:
        predictions.to_csv(args.output, index=False)
    print(f"Saved {len(predictions)} probabilities to {args.output}")


if __name__ == "__main__":
    main()
