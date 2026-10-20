# Model-serving handoff

This document prepares the trained ML component for a future backend and frontend. It does not add a web server or a user interface.

## Runtime bundle and environment

The final bundle is `models/artifacts/final_expert_without_cat11_v22/`. It needs exactly `metadata.json`, `model.cbm` and `preprocessing.joblib` at inference time. The binary files are intentionally excluded from ordinary Git; see [the artifact README](models/artifacts/README.md) before moving the application to another machine.

Use 64-bit Python 3.12 and install the tested inference dependencies with:

```powershell
python -m pip install -r requirements-serving.txt
```

Run the backend from the repository root (or add the repository root to `PYTHONPATH`) so the bundled `src.features` classes can be imported when `joblib` loads the fitted preprocessor. The default bundle path is resolved relative to the source tree, not the current working directory. A different location can be passed to `FinalCatBoostPredictor.load(...)` or set for the CLI with `BNP_MODEL_BUNDLE_DIR`.

## Backend contract

Load the bundle **once at application startup**:

```python
import pandas as pd
from src.models.inference import DEFAULT_BUNDLE_DIR, FinalCatBoostPredictor

predictor = FinalCatBoostPredictor.load(DEFAULT_BUNDLE_DIR)
schema = predictor.input_schema()  # Can power a future GET /model/schema endpoint.

def predict_claim(claim_record: dict[str, object]) -> dict[str, object]:
    # The backend validates/assigns an ID and accepts all 25 raw feature keys.
    result = predictor.predict_raw(pd.DataFrame([claim_record]))
    return result.to_dict("records")[0]
```

`input_schema()` lists the 25 raw features in order with `number`/`string` types, an `ID` field and the output field name. The frontend must **not** send the 247 engineered features. Every raw feature key is required, but its value may be `null` when genuinely missing. Numerical strings from CSV are converted to numbers; invalid values, booleans and infinities are rejected. Categorical values must be strings or `null`; unseen categories are encoded safely by the fitted preprocessor. The model uses `ID` only for traceability, never as a feature. The `target` column is forbidden in inference requests. Full raw BNP CSV files may contain additional columns, which are ignored after validation of the selected features.

The response is one row per input claim with `ID` and `probability_target_1` in `[0, 1]`. No business decision threshold or post-hoc calibrator is part of this bundle. Do not label the probability as approval, rejection or fraud without a separate definition and evaluation.

For a production-style HTTP layer, add request-size limits, authentication if appropriate, structured error responses and logging without sensitive raw field values. The inference method rejects duplicate/missing IDs and duplicate column names; the web layer should assign a unique ID if the user did not provide one. Temporary uploads can go under `data/uploads/` and generated prediction exports under `reports/predictions/generated/`; both are ignored by Git.

## Smoke tests

From the repository root, verify the model and one raw row without creating output files:

```powershell
python -c "import pandas as pd; from src.models.inference import FinalCatBoostPredictor, DEFAULT_BUNDLE_DIR; p=FinalCatBoostPredictor.load(DEFAULT_BUNDLE_DIR); x=pd.read_csv('data/raw/test.csv.zip',nrows=1); print(p.input_schema()); print(p.predict_raw(x))"
python -m pytest tests/unit -q
```

The first command requires the local raw test archive. On a deployment machine without Kaggle data, use a valid 25-field JSON test fixture through the future API instead. The unit tests use a tiny synthetic bundle and do not require the large final model.

The selected CatBoost model has 247 features and was trained on 114,321 labeled rows. The reported OOF Log Loss (0.43845) belongs to the cross-validation models, **not** to the final model; the unlabeled test set cannot provide an independent score. See [the OOF summary](reports/metrics/oof_diagnostics_summary.json) and [final metadata](models/artifacts/final_expert_without_cat11_v22/metadata.json).
