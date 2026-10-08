import pandas as pd

def validate_raw_schema(train: pd.DataFrame, test: pd.DataFrame, target_column: str = "target", id_column: str = "ID",) -> None:
    if target_column not in train or target_column in test:
        raise ValueError(f"Expected {target_column!r} in train only")
    if id_column not in train or id_column not in test:
        raise ValueError(f"Expected {id_column!r} in both train and test")

    train_features = train.columns.drop(target_column).tolist()

    if train_features != test.columns.tolist():
        raise ValueError("Train and test columns differ (apart from the target) or are reordered")
    if train[id_column].isna().any() or test[id_column].isna().any():
        raise ValueError("ID contains missing values")
    if not train[id_column].is_unique or not test[id_column].is_unique:
        raise ValueError("ID must be unique within each split")

