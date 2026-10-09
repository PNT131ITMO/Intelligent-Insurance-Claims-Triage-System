from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from catboost import CatBoostClassifier, Pool
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from .data import FeatureSpecification, ModelingDataset
from .metrics import binary_probability_metrics, summarize_fold_metrics


@dataclass(frozen=True)
class CrossValidationConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42

    def build_splitter(self) -> StratifiedKFold:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        return StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state if self.shuffle else None,
        )

@dataclass(frozen=True)
class CatBoostBaselineConfig:
    parameters: dict[str, Any]
    early_stopping_rounds: int = 100
    verbose: int | bool = 100


@dataclass
class BaselineRunResult:
    model_name: str
    feature_specification: FeatureSpecification
    fold_metrics: pd.DataFrame
    summary_metrics: pd.DataFrame
    overall_metrics: dict[str, float]
    oof_predictions: pd.DataFrame
    test_predictions: pd.DataFrame | None = None
    feature_importance: pd.DataFrame | None = None
    model_paths: list[Path] = field(default_factory=list)


def _folds(dataset: ModelingDataset, config: CrossValidationConfig):
    row_positions = np.arange(len(dataset.X_train))
    return config.build_splitter().split(row_positions, dataset.y_train.to_numpy())


def _prediction_frame(
    dataset: ModelingDataset,
    probability: np.ndarray,
    fold_assignment: np.ndarray,
    column_name: str,
) -> pd.DataFrame:
    return pd.DataFrame({
        dataset.train_ids.name: dataset.train_ids.to_numpy(),
        dataset.y_train.name: dataset.y_train.to_numpy(),
        "fold": fold_assignment,
        column_name: probability,
    })


def run_sanity_baseline(
    dataset: ModelingDataset,
    feature_specification: FeatureSpecification,
    cv_config: CrossValidationConfig,
    strategy: str = "prior",
) -> BaselineRunResult:
    oof_probability = np.full(len(dataset.X_train), np.nan, dtype=np.float64)
    fold_assignment = np.full(len(dataset.X_train), -1, dtype=np.int16)
    fold_rows: list[dict[str, Any]] = []
    constant_feature = np.zeros((len(dataset.X_train), 1), dtype=np.uint8)

    for fold, (train_index, valid_index) in enumerate(_folds(dataset, cv_config), start=1):
        started = perf_counter()
        model = DummyClassifier(strategy=strategy)
        model.fit(constant_feature[train_index], dataset.y_train.iloc[train_index])
        training_seconds = perf_counter() - started
        probability = model.predict_proba(constant_feature[valid_index])[:, 1]
        oof_probability[valid_index] = probability
        fold_assignment[valid_index] = fold
        fold_rows.append({
            "model": "dummy_prior",
            "feature_set": feature_specification.name,
            "fold": fold,
            "train_rows": len(train_index),
            "valid_rows": len(valid_index),
            "best_iteration": pd.NA,
            "training_seconds": training_seconds,
            "run_seconds": perf_counter() - started,
            "resumed": False,
            **binary_probability_metrics(dataset.y_train.iloc[valid_index], probability),
        })

    if np.isnan(oof_probability).any() or (fold_assignment < 1).any():
        raise RuntimeError("Sanity baseline did not produce exactly one OOF prediction per row")

    fold_metrics = pd.DataFrame(fold_rows)
    overall_metrics = binary_probability_metrics(dataset.y_train, oof_probability)
    return BaselineRunResult(
        model_name="dummy_prior",
        feature_specification=feature_specification,
        fold_metrics=fold_metrics,
        summary_metrics=summarize_fold_metrics(fold_metrics),
        overall_metrics=overall_metrics,
        oof_predictions=_prediction_frame(
            dataset,
            oof_probability,
            fold_assignment,
            "dummy_probability",
        ),
    )


def run_catboost_baseline(
    dataset: ModelingDataset,
    feature_specification: FeatureSpecification,
    cv_config: CrossValidationConfig,
    model_config: CatBoostBaselineConfig,
    model_dir: Path | None = None,
    predict_test: bool = True,
    resume: bool = True,
) -> BaselineRunResult:
    X_train = dataset.X_train.loc[:, list(feature_specification.feature_names)]
    X_test = dataset.X_test.loc[:, list(feature_specification.feature_names)]
    y_train = dataset.y_train
    oof_probability = np.full(len(X_train), np.nan, dtype=np.float64)
    fold_assignment = np.full(len(X_train), -1, dtype=np.int16)
    test_probability = np.zeros(len(X_test), dtype=np.float64) if predict_test else None
    fold_rows: list[dict[str, Any]] = []
    importance_rows: list[pd.DataFrame] = []
    model_paths: list[Path] = []

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_index, valid_index) in enumerate(_folds(dataset, cv_config), start=1):
        started = perf_counter()
        train_pool = Pool(
            X_train.iloc[train_index],
            label=y_train.iloc[train_index],
            cat_features=list(feature_specification.categorical_indices),
            feature_names=list(feature_specification.feature_names),
        )
        valid_pool = Pool(
            X_train.iloc[valid_index],
            label=y_train.iloc[valid_index],
            cat_features=list(feature_specification.categorical_indices),
            feature_names=list(feature_specification.feature_names),
        )
        parameters = dict(model_config.parameters)
        parameters["random_seed"] = int(parameters.get("random_seed", cv_config.random_state)) + fold - 1
        context = {
            "fold": fold,
            "feature_set": feature_specification.name,
            "feature_names": list(feature_specification.feature_names),
            "categorical_indices": list(feature_specification.categorical_indices),
            "cross_validation": {
                "n_splits": cv_config.n_splits,
                "shuffle": cv_config.shuffle,
                "random_state": cv_config.random_state,
            },
            "parameters": parameters,
            "early_stopping_rounds": model_config.early_stopping_rounds,
            "train_rows": len(train_index),
            "valid_rows": len(valid_index),
        }
        signature = hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        model_path = model_dir / f"fold_{fold}.cbm" if model_dir is not None else None
        context_path = model_dir / f"fold_{fold}.json" if model_dir is not None else None
        can_resume = False
        if resume and model_path is not None and context_path is not None and model_path.exists() and context_path.exists():
            stored_context = json.loads(context_path.read_text(encoding="utf-8"))
            can_resume = stored_context.get("signature") == signature

        model = CatBoostClassifier()
        if can_resume:
            model.load_model(model_path)
            training_seconds = stored_context.get("training_seconds")
        else:
            model = CatBoostClassifier(**parameters)
            fit_started = perf_counter()
            model.fit(
                train_pool,
                eval_set=valid_pool,
                early_stopping_rounds=model_config.early_stopping_rounds,
                use_best_model=True,
                verbose=model_config.verbose,
            )
            training_seconds = perf_counter() - fit_started
            if model_path is not None and context_path is not None:
                model.save_model(model_path)
                context_path.write_text(
                    json.dumps(
                        {**context, "training_seconds": training_seconds, "signature": signature},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )

        probability = model.predict_proba(valid_pool)[:, 1]
        oof_probability[valid_index] = probability
        fold_assignment[valid_index] = fold

        if test_probability is not None:
            test_pool = Pool(
                X_test,
                cat_features=list(feature_specification.categorical_indices),
                feature_names=list(feature_specification.feature_names),
            )
            test_probability += model.predict_proba(test_pool)[:, 1] / cv_config.n_splits

        importance_rows.append(pd.DataFrame({
            "model": "catboost",
            "feature_set": feature_specification.name,
            "fold": fold,
            "feature": feature_specification.feature_names,
            "importance": model.get_feature_importance(valid_pool, type="FeatureImportance"),
        }))
        fold_rows.append({
            "model": "catboost",
            "feature_set": feature_specification.name,
            "fold": fold,
            "train_rows": len(train_index),
            "valid_rows": len(valid_index),
            "best_iteration": model.get_best_iteration(),
            "training_seconds": training_seconds,
            "run_seconds": perf_counter() - started,
            "resumed": can_resume,
            **binary_probability_metrics(y_train.iloc[valid_index], probability),
        })

        if model_path is not None:
            model_paths.append(model_path)

    if np.isnan(oof_probability).any() or (fold_assignment < 1).any():
        raise RuntimeError("CatBoost did not produce exactly one OOF prediction per row")
    if test_probability is not None and (not np.isfinite(test_probability).all() or ((test_probability < 0) | (test_probability > 1)).any()):
        raise RuntimeError("CatBoost produced invalid test probabilities")

    fold_metrics = pd.DataFrame(fold_rows)
    overall_metrics = binary_probability_metrics(y_train, oof_probability)
    test_predictions = None
    if test_probability is not None:
        test_predictions = pd.DataFrame({
            dataset.test_ids.name: dataset.test_ids.to_numpy(),
            "catboost_probability": test_probability,
        })

    return BaselineRunResult(
        model_name="catboost",
        feature_specification=feature_specification,
        fold_metrics=fold_metrics,
        summary_metrics=summarize_fold_metrics(fold_metrics),
        overall_metrics=overall_metrics,
        oof_predictions=_prediction_frame(
            dataset,
            oof_probability,
            fold_assignment,
            "catboost_probability",
        ),
        test_predictions=test_predictions,
        feature_importance=pd.concat(importance_rows, ignore_index=True),
        model_paths=model_paths,
    )

