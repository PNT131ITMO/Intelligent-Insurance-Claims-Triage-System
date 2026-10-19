from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any
from uuid import uuid4

from catboost import CatBoostClassifier, Pool
import joblib
import numpy as np
import pandas as pd
import yaml

from .artifacts import sha256_file
from .data import FeatureSpecification, load_modeling_dataset, resolve_feature_set


@dataclass(frozen=True)
class FinalTrainingPlan:
    feature_specification: FeatureSpecification
    processed_dir: Path
    preprocessor_path: Path
    source_experiment: Path
    output_dir: Path
    parameters: dict[str, Any]
    cv_best_iterations: tuple[int, ...]
    source_oof_metrics: dict[str, float]
    source_checksums: dict[str, str]
    verbose: int | bool
    snapshot_enabled: bool
    snapshot_interval_seconds: int

    @property
    def iterations(self) -> int:
        return int(self.parameters["iterations"])


def _within_project(project_root: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Configured path must be relative: {relative_path}")
    resolved = (project_root / path).resolve()
    if not resolved.is_relative_to(project_root.resolve()):
        raise ValueError(f"Configured path escapes the project: {relative_path}")
    return resolved


def choose_final_iterations(
    fold_metrics: pd.DataFrame, feature_set: str, expected_folds: int, max_iterations: int
) -> tuple[int, tuple[int, ...]]:
    rows = fold_metrics.loc[
        fold_metrics["model"].eq("catboost") & fold_metrics["feature_set"].eq(feature_set)
    ].sort_values("fold")
    if rows["fold"].tolist() != list(range(1, expected_folds + 1)):
        raise ValueError("Expected exactly one CatBoost result for every CV fold")
    best = pd.to_numeric(rows["best_iteration"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(best).all() or ((best < 0) | (best >= max_iterations) | (best != np.floor(best))).any():
        raise ValueError("CV best_iteration values must be valid zero-based tree indices")
    counts = tuple(int(value) + 1 for value in best)
    return int(np.ceil(np.median(counts))), counts


def build_final_plan(project_root: Path, config_path: Path | None = None) -> FinalTrainingPlan:
    project_root = project_root.resolve()
    config_path = config_path or project_root / "configs" / "final_model.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("iteration_rule") != "median_cv_best_iteration_plus_one":
        raise ValueError("Only median_cv_best_iteration_plus_one is supported")

    feature_set = str(config["feature_set"])
    processed_dir = _within_project(project_root, str(config["processed_dir"]))
    source_dir = _within_project(project_root, str(config["source_experiment"]))
    preprocessor_path = _within_project(project_root, str(config["preprocessing_artifact"]))
    output_dir = _within_project(project_root, str(config["output_dir"]))
    if not preprocessor_path.is_file():
        raise FileNotFoundError(f"Fitted preprocessing artifact is missing: {preprocessor_path}")

    manifest = json.loads((processed_dir / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_specification = resolve_feature_set(manifest, feature_set)
    source_metadata = json.loads((source_dir / "training_metadata.json").read_text(encoding="utf-8"))
    if source_metadata["feature_set"] != feature_set:
        raise ValueError("Selected feature set differs from the source CV experiment")
    if source_metadata["feature_names"] != list(feature_specification.feature_names):
        raise ValueError("Selected feature names differ from the source CV experiment")
    if source_metadata["categorical_feature_indices"] != list(feature_specification.categorical_indices):
        raise ValueError("Categorical indices differ from the source CV experiment")
    if manifest["configuration"].get("target_encoding", {}).get("enabled", False):
        raise ValueError("This final-model plan requires target encoding to be disabled")

    source_checksums = source_metadata["source_checksums"]
    for filename in ("X_train_expert.parquet", "y_train.parquet", "feature_manifest.json"):
        if sha256_file(processed_dir / filename) != source_checksums[filename]:
            raise ValueError(f"Processed input changed since the selected CV run: {filename}")

    parameters = dict(source_metadata["catboost"]["parameters"])
    cv_folds = int(source_metadata["cross_validation"]["n_splits"])
    fold_metrics = pd.read_csv(source_dir / "fold_metrics.csv")
    iterations, cv_counts = choose_final_iterations(
        fold_metrics, feature_set, cv_folds, int(parameters["iterations"])
    )
    parameters["iterations"] = iterations
    overall = pd.read_csv(source_dir / "overall_metrics.csv")
    catboost_rows = overall.loc[
        overall["model"].eq("catboost") & overall["feature_set"].eq(feature_set)
    ]
    if len(catboost_rows) != 1:
        raise ValueError("Source experiment must have one CatBoost OOF metric row")
    source_oof_metrics = {
        key: float(catboost_rows.iloc[0][key])
        for key in ("log_loss", "roc_auc", "average_precision", "brier_score")
    }
    snapshot_config = config.get("snapshot", {})
    snapshot_interval = int(snapshot_config.get("interval_seconds", 600))
    if snapshot_interval < 1:
        raise ValueError("Snapshot interval must be positive")
    return FinalTrainingPlan(
        feature_specification=feature_specification,
        processed_dir=processed_dir,
        preprocessor_path=preprocessor_path,
        source_experiment=source_dir,
        output_dir=output_dir,
        parameters=parameters,
        cv_best_iterations=cv_counts,
        source_oof_metrics=source_oof_metrics,
        source_checksums=source_checksums,
        verbose=config.get("verbose", 100),
        snapshot_enabled=bool(snapshot_config.get("enabled", True)),
        snapshot_interval_seconds=snapshot_interval,
    )


def _validate_preprocessor(preprocessor: Any, specification: FeatureSpecification) -> None:
    if not getattr(preprocessor, "is_fitted_", False):
        raise ValueError("Preprocessing artifact is not fitted")
    if getattr(preprocessor, "target_encoding_enabled_", False):
        raise ValueError("Final inference requires target encoding to be disabled")
    selected = set(specification.feature_names)
    ordered = [name for name in preprocessor.get_feature_names_out() if name in selected]
    if ordered != list(specification.feature_names):
        raise ValueError("Preprocessing artifact cannot reproduce selected feature order")


def train_final_bundle(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    specification: FeatureSpecification,
    preprocessor_path: Path,
    output_dir: Path,
    parameters: dict[str, Any],
    provenance: dict[str, Any],
    *,
    verbose: int | bool = 100,
    snapshot_enabled: bool = False,
    snapshot_interval_seconds: int = 600,
) -> Path:
    if X_train.columns.tolist() != list(specification.feature_names):
        raise ValueError("Final training feature order does not match the selected manifest")
    if not X_train.index.equals(y_train.index) or set(y_train.unique()) != {0, 1}:
        raise ValueError("Final training labels must be aligned and binary")
    if X_train[list(specification.categorical_names)].isna().any().any():
        raise ValueError("Categorical training features contain missing values")
    if not preprocessor_path.is_file():
        raise FileNotFoundError(preprocessor_path)
    preprocessor = joblib.load(preprocessor_path)
    _validate_preprocessor(preprocessor, specification)
    train_parameters = dict(parameters)
    if snapshot_enabled:
        train_parameters["allow_writing_files"] = True
        train_parameters["train_dir"] = str(output_dir / "catboost_info")
    context = {
        "feature_names": list(specification.feature_names),
        "categorical_indices": list(specification.categorical_indices),
        "parameters": parameters,
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "provenance": provenance,
        "snapshot_enabled": snapshot_enabled,
        "snapshot_interval_seconds": snapshot_interval_seconds,
    }
    signature = hashlib.sha256(
        json.dumps(context, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    signature_path = output_dir / "run_signature.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        if (output_dir / "metadata.json").exists():
            raise FileExistsError(f"Final model already exists: {output_dir}")
        if (output_dir / "model.cbm").exists() or (output_dir / "preprocessing.joblib").exists():
            raise FileExistsError(f"Incomplete bundle requires inspection before retry: {output_dir}")
        if not signature_path.exists():
            raise FileExistsError(f"Output directory contains unrelated files: {output_dir}")
        stored = json.loads(signature_path.read_text(encoding="utf-8"))
        if stored.get("signature") != signature:
            raise ValueError("Interrupted training context differs; refusing to reuse the snapshot")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(json.dumps({"signature": signature, "context": context}, indent=2), encoding="utf-8")

    pool = Pool(
        X_train,
        label=y_train,
        cat_features=list(specification.categorical_indices),
        feature_names=list(specification.feature_names),
    )
    model = CatBoostClassifier(**train_parameters)
    started = perf_counter()
    model.fit(
        pool,
        use_best_model=False,
        verbose=verbose,
        save_snapshot=snapshot_enabled,
        snapshot_file=str(output_dir / "training.cbsnapshot") if snapshot_enabled else None,
        snapshot_interval=snapshot_interval_seconds if snapshot_enabled else None,
    )
    training_seconds = perf_counter() - started
    if model.tree_count_ != int(parameters["iterations"]):
        raise RuntimeError("Final model did not train the fixed number of trees")

    model_path = output_dir / "model.cbm"
    pipeline_path = output_dir / "preprocessing.joblib"
    suffix = uuid4().hex
    temporary_model = output_dir / f"model.{suffix}.tmp.cbm"
    temporary_pipeline = output_dir / f"preprocessing.{suffix}.tmp.joblib"
    model.save_model(str(temporary_model))
    shutil.copy2(preprocessor_path, temporary_pipeline)
    temporary_model.replace(model_path)
    temporary_pipeline.replace(pipeline_path)

    reloaded = CatBoostClassifier()
    reloaded.load_model(str(model_path))
    if reloaded.feature_names_ != list(specification.feature_names) or reloaded.tree_count_ != model.tree_count_:
        raise RuntimeError("Saved final model does not match its feature schema or tree count")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "trained_no_holdout_evaluation",
        "model_type": "CatBoostClassifier",
        "feature_set": specification.name,
        "feature_names": list(specification.feature_names),
        "categorical_feature_names": list(specification.categorical_names),
        "categorical_feature_indices": list(specification.categorical_indices),
        "raw_selected_features": list(preprocessor.selected_features_),
        "target_name": preprocessor.target_column,
        "id_name": preprocessor.id_column,
        "trained_rows": len(X_train),
        "train_positive_rate": float(y_train.mean()),
        "parameters": parameters,
        "snapshot": {
            "enabled": snapshot_enabled,
            "interval_seconds": snapshot_interval_seconds,
            "file": "training.cbsnapshot" if snapshot_enabled else None,
        },
        "tree_count": int(model.tree_count_),
        "training_seconds": training_seconds,
        "provenance": provenance,
        "source_preprocessor_sha256": sha256_file(preprocessor_path),
        "model_sha256": sha256_file(model_path),
        "preprocessing_sha256": sha256_file(pipeline_path),
        "library_versions": {
            name: version(name)
            for name in ("catboost", "numpy", "pandas", "scikit-learn", "joblib")
        },
        "decision_threshold": None,
        "calibration_applied": False,
    }
    temporary_metadata = output_dir / f"metadata.{suffix}.tmp.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(output_dir / "metadata.json")
    return output_dir


def run_final_training(plan: FinalTrainingPlan) -> Path:
    """Load validated processed data and execute the selected full-train plan."""
    dataset = load_modeling_dataset(plan.processed_dir)
    specification = resolve_feature_set(dataset.manifest, plan.feature_specification.name)
    X_train = dataset.X_train.loc[:, list(specification.feature_names)]
    provenance = {
        "selection_source": "reports/metrics/model_selection.md",
        "source_experiment": plan.source_experiment.relative_to(plan.processed_dir.parents[1]).as_posix(),
        "source_oof_metrics": plan.source_oof_metrics,
        "cv_best_tree_counts": list(plan.cv_best_iterations),
        "iteration_rule": "ceil(median(best_iteration + 1))",
        "processed_source_checksums": plan.source_checksums,
        "validation_note": "OOF metrics belong to the CV models; the final full-data model has no held-out score.",
        "cv_preprocessing_scope": "Unsupervised categorical mappings were fitted on all labeled training features before CV; no validation labels were used to fit them.",
    }
    return train_final_bundle(
        X_train,
        dataset.y_train,
        specification,
        plan.preprocessor_path,
        plan.output_dir,
        plan.parameters,
        provenance,
        verbose=plan.verbose,
        snapshot_enabled=plan.snapshot_enabled,
        snapshot_interval_seconds=plan.snapshot_interval_seconds,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train", action="store_true", help="Run full training; default only checks the plan")
    args = parser.parse_args()
    plan = build_final_plan(args.project_root, args.config)
    print(json.dumps({
        "feature_set": plan.feature_specification.name,
        "feature_count": len(plan.feature_specification.feature_names),
        "categorical_feature_count": len(plan.feature_specification.categorical_names),
        "cv_best_tree_counts": plan.cv_best_iterations,
        "final_iterations": plan.iterations,
        "snapshot_enabled": plan.snapshot_enabled,
        "output_dir": str(plan.output_dir),
        "source_oof_log_loss": plan.source_oof_metrics["log_loss"],
    }, indent=2))
    if args.train:
        output_dir = run_final_training(plan)
        print(f"Final model bundle saved to {output_dir}")


if __name__ == "__main__":
    main()
