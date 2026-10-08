from dataclasses import dataclass
import pandas as pd

TARGET_COLUMN = "target"
ID_COLUMN = "ID"

@dataclass(frozen=True)
class SchemaSummary:
    train_shape: tuple[int, int]
    test_shape: tuple[int, int]
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    dtype_mismatches: tuple[str, ...]

def summarize_schema(train: pd.DataFrame, test: pd.DataFrame, target_column: str = TARGET_COLUMN, id_column: str = ID_COLUMN,) -> SchemaSummary:
    features = [column for column in train.columns if column not in {target_column, id_column}]
    numerical = tuple(column for column in features if pd.api.types.is_numeric_dtype(train[column]))
    categorical = tuple(column for column in features if column not in numerical)
    shared = [column for column in features if column in test.columns]
    mismatches = tuple(column for column in shared if train[column].dtype != test[column].dtype)
    return SchemaSummary(train.shape, test.shape, numerical, categorical, mismatches)

