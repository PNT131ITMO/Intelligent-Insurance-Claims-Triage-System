# Model artifacts for inference

The final CatBoost bundle is expected at `models/artifacts/final_expert_without_cat11_v22/`:

| File | Required at runtime | Git policy |
|---|---|---|
| `metadata.json` | Yes | Allowed by `.gitignore`; commit it with the source |
| `model.cbm` | Yes | No; approximately 458 MB |
| `preprocessing.joblib` | Yes | No; approximately 95 MB |
| `run_signature.json`, `training.cbsnapshot`, `catboost_info/` | No | No; training-only files |

Keep the three runtime files together. `FinalCatBoostPredictor.load(bundle_dir)` verifies the checksums in `metadata.json` and rejects a mismatched model or preprocessor. Only load `preprocessing.joblib` from a trusted source; it is a Python serialization format.

To run the project on another machine, supply the two binary files separately (for example as release assets or in a course-submission archive), or reproduce them by running `python -m src.models.final_training --train` after preparing the original data. Do **not** force-add `model.cbm` to ordinary Git: GitHub blocks files above 100 MiB. Git LFS is an alternative if repository storage/bandwidth and evaluator support have been confirmed; this repository does not enable LFS automatically.

See [SERVING.md](../../SERVING.md) for the serving contract, validation and smoke-test commands.
