from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATE_COL, ID_COL, TARGET_COL, TEST_PATH, TRAIN_PATH, VALID_FRACTION


@dataclass(frozen=True)
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def load_data(train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH) -> DatasetBundle:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    validate_schema(train, test)
    return DatasetBundle(train=train, test=test)


def validate_schema(train: pd.DataFrame, test: pd.DataFrame) -> None:
    required_train = {ID_COL, DATE_COL, TARGET_COL}
    required_test = {ID_COL, DATE_COL}

    missing_train = sorted(required_train - set(train.columns))
    missing_test = sorted(required_test - set(test.columns))
    if missing_train:
        raise ValueError(f"train.csv missing required columns: {missing_train}")
    if missing_test:
        raise ValueError(f"test.csv missing required columns: {missing_test}")
    if TARGET_COL in test.columns:
        raise ValueError("test.csv should not contain the target column cnt")


def time_order_split(
    train: pd.DataFrame,
    valid_fraction: float = VALID_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < valid_fraction < 1:
        raise ValueError("valid_fraction must be between 0 and 1")

    split_idx = int(len(train) * (1 - valid_fraction))
    if split_idx <= 0 or split_idx >= len(train):
        raise ValueError("validation split produced an empty train or valid set")

    train_part = train.iloc[:split_idx].copy()
    valid_part = train.iloc[split_idx:].copy()
    return train_part, valid_part


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = {
        TARGET_COL,
        ID_COL,
        DATE_COL,
        "casual",
        "registered",
    }
    return [col for col in df.columns if col not in drop_cols]

