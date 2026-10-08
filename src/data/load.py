from pathlib import Path

import pandas as pd


def load_raw_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists() or not test_path.exists():
        missing = [str(path) for path in (train_path, test_path) if not path.exists()]
        raise FileNotFoundError(f"Missing raw data file(s): {missing}")
    return pd.read_csv(train_path), pd.read_csv(test_path)

