from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DATE_COL, TARGET_COL

LAG_FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_24",
    "lag_48",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "rolling_std_24",
    "rolling_same_hour_7d",
]


def add_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_timestamp"] = pd.to_datetime(out[DATE_COL]) + pd.to_timedelta(out["hr"], unit="h")
    return out


@dataclass
class LagFeatureBuilder:
    global_mean: float = 0.0
    history: dict[pd.Timestamp, float] = field(default_factory=dict)

    def fit_history(self, df: pd.DataFrame) -> "LagFeatureBuilder":
        if TARGET_COL not in df.columns:
            raise ValueError(f"LagFeatureBuilder requires target column '{TARGET_COL}'")
        data = add_timestamp(df).sort_values("_timestamp")
        self.global_mean = float(data[TARGET_COL].mean())
        self.history = {
            pd.Timestamp(row["_timestamp"]): float(row[TARGET_COL])
            for _, row in data.iterrows()
        }
        return self

    def feature_values(self, timestamp: pd.Timestamp) -> dict[str, float]:
        lag_1 = self._lag(timestamp, 1)
        lag_2 = self._lag(timestamp, 2)
        lag_24 = self._lag(timestamp, 24)
        lag_48 = self._lag(timestamp, 48)
        lag_168 = self._lag(timestamp, 168)
        last_24 = self._previous_hours(timestamp, 24)
        last_168 = self._previous_hours(timestamp, 168)
        same_hour_7d = self._same_hour_days(timestamp, 7)

        return {
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_24": lag_24,
            "lag_48": lag_48,
            "lag_168": lag_168,
            "rolling_mean_24": self._mean(last_24),
            "rolling_mean_168": self._mean(last_168),
            "rolling_std_24": self._std(last_24),
            "rolling_same_hour_7d": self._mean(same_hour_7d),
        }

    def add_prediction(self, timestamp: pd.Timestamp, value: float) -> None:
        self.history[pd.Timestamp(timestamp)] = float(max(value, 0.0))

    def _lag(self, timestamp: pd.Timestamp, hours: int) -> float:
        key = timestamp - pd.Timedelta(hours=hours)
        return self.history.get(key, self.global_mean)

    def _previous_hours(self, timestamp: pd.Timestamp, hours: int) -> list[float]:
        values = [
            self.history[timestamp - pd.Timedelta(hours=offset)]
            for offset in range(1, hours + 1)
            if timestamp - pd.Timedelta(hours=offset) in self.history
        ]
        return values

    def _same_hour_days(self, timestamp: pd.Timestamp, days: int) -> list[float]:
        values = [
            self.history[timestamp - pd.Timedelta(days=offset)]
            for offset in range(1, days + 1)
            if timestamp - pd.Timedelta(days=offset) in self.history
        ]
        return values

    def _mean(self, values: list[float]) -> float:
        return float(np.mean(values)) if values else self.global_mean

    def _std(self, values: list[float]) -> float:
        return float(np.std(values)) if len(values) > 1 else 0.0


def add_training_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"training lag features require target column '{TARGET_COL}'")
    data = add_timestamp(df).sort_values("_timestamp").copy()
    builder = LagFeatureBuilder(global_mean=float(data[TARGET_COL].mean()))

    rows = []
    for _, row in data.iterrows():
        timestamp = pd.Timestamp(row["_timestamp"])
        feature_values = builder.feature_values(timestamp)
        rows.append(feature_values)
        builder.add_prediction(timestamp, float(row[TARGET_COL]))

    lag_df = pd.DataFrame(rows, index=data.index)
    for col in LAG_FEATURE_COLUMNS:
        data[col] = lag_df[col]
    return data.drop(columns=["_timestamp"])


def add_lag_features_for_row(row: pd.Series, builder: LagFeatureBuilder) -> pd.DataFrame:
    timestamp = pd.to_datetime(row[DATE_COL]) + pd.to_timedelta(row["hr"], unit="h")
    out = row.to_frame().T.copy()
    for col, value in builder.feature_values(pd.Timestamp(timestamp)).items():
        out[col] = value
    return out
