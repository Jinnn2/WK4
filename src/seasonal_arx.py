from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import DATE_COL, ID_COL, TARGET_COL
from .data_utils import select_feature_columns
from .features import GROWTH_KEY_SETS
from .lag_features import LagFeatureBuilder, add_lag_features_for_row, add_training_lag_features
from .models import (
    Regressor,
    build_model,
    fit_model,
    get_best_iteration,
    mse,
    prepare_features_for_model,
)

SEASONAL_ARX_PREFIX = "seasonal_arx_"
SEASONAL_ARX_TARGET = "_seasonal_arx_log_residual"

PROFILE_FALLBACK_KEY_SETS = [
    ["mnth", "day", "hr"],
    ["mnth", "hr", "workingday"],
    ["mnth", "hr"],
    ["hr", "workingday"],
    ["hr"],
]


def is_seasonal_arx_model(model_name: str) -> bool:
    return model_name.lower().strip().startswith(SEASONAL_ARX_PREFIX)


def seasonal_arx_base_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()
    if not normalized.startswith(SEASONAL_ARX_PREFIX):
        raise ValueError(f"Not a seasonal ARX model: {model_name}")
    base_name = normalized[len(SEASONAL_ARX_PREFIX) :]
    if not base_name:
        raise ValueError(f"Missing base model name in: {model_name}")
    return base_name


def row_timestamp(row: pd.Series) -> pd.Timestamp:
    return pd.to_datetime(row[DATE_COL]) + pd.to_timedelta(row["hr"], unit="h")


class SeasonalBaselineEncoder:
    """Year-over-year seasonal prior with growth and profile fallbacks."""

    def __init__(self, smoothing: float = 30.0) -> None:
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.global_growth_: float = 1.0
        self.last_year_table_: pd.DataFrame | None = None
        self.profile_tables_: list[tuple[str, list[str], pd.DataFrame]] = []
        self.growth_tables_: list[tuple[str, list[str], pd.DataFrame]] = []

    def fit(self, df: pd.DataFrame) -> "SeasonalBaselineEncoder":
        if TARGET_COL not in df.columns:
            raise ValueError(f"SeasonalBaselineEncoder requires target column '{TARGET_COL}'")
        required = {"yr", "mnth", "day", "hr", "workingday", TARGET_COL}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"SeasonalBaselineEncoder missing required columns: {missing}")

        data = df.copy()
        self.global_mean_ = float(data[TARGET_COL].mean())
        last_year = data[data["yr"] == 0]
        current_year = data[data["yr"] == 1]

        self.last_year_table_ = (
            last_year.groupby(["mnth", "day", "hr"], dropna=False)[TARGET_COL]
            .mean()
            .reset_index()
            .rename(columns={TARGET_COL: "last_year_same_mnth_day_hr_cnt"})
        )

        ly_mean = float(last_year[TARGET_COL].mean()) if len(last_year) else self.global_mean_
        cy_mean = float(current_year[TARGET_COL].mean()) if len(current_year) else self.global_mean_
        self.global_growth_ = cy_mean / ly_mean if ly_mean > 0 else 1.0

        self.profile_tables_ = []
        for keys in PROFILE_FALLBACK_KEY_SETS:
            if not all(key in data.columns for key in keys):
                continue
            col = "seasonal_profile_" + "_".join(keys)
            table = (
                data.groupby(keys, dropna=False)[TARGET_COL]
                .mean()
                .reset_index()
                .rename(columns={TARGET_COL: col})
            )
            self.profile_tables_.append((col, keys, table))

        self.growth_tables_ = []
        for keys in GROWTH_KEY_SETS:
            if not all(key in data.columns for key in keys):
                continue
            ly = (
                last_year.groupby(keys, dropna=False)[TARGET_COL]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "ly_mean", "count": "ly_count"})
            )
            cy = (
                current_year.groupby(keys, dropna=False)[TARGET_COL]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "cy_mean", "count": "cy_count"})
            )
            table = ly.merge(cy, on=keys, how="inner")
            if table.empty:
                continue
            raw_growth = table["cy_mean"] / table["ly_mean"].replace(0, np.nan)
            support = table["ly_count"] + table["cy_count"]
            table["growth"] = (
                raw_growth.fillna(self.global_growth_) * support
                + self.global_growth_ * self.smoothing
            ) / (support + self.smoothing)
            col = "seasonal_growth_" + "_".join(keys)
            table = table[keys + ["growth"]].rename(columns={"growth": col})
            self.growth_tables_.append((col, keys, table))

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.last_year_table_ is None:
            raise RuntimeError("SeasonalBaselineEncoder must be fitted before transform")

        out = df.copy()
        out = out.merge(self.last_year_table_, on=["mnth", "day", "hr"], how="left")
        has_current_year = out["yr"] > 0 if "yr" in out.columns else pd.Series(True, index=out.index)
        out["has_last_year_same_mnth_day_hr"] = (
            out["last_year_same_mnth_day_hr_cnt"].notna() & has_current_year
        ).astype(int)

        profile_cols: list[str] = []
        for col, keys, table in self.profile_tables_:
            out = out.merge(table, on=keys, how="left")
            profile_cols.append(col)
        if profile_cols:
            out["seasonal_profile_fallback"] = out[profile_cols].bfill(axis=1).iloc[:, 0]
        else:
            out["seasonal_profile_fallback"] = self.global_mean_
        out["seasonal_profile_fallback"] = out["seasonal_profile_fallback"].fillna(self.global_mean_)

        growth_cols: list[str] = []
        for col, keys, table in self.growth_tables_:
            out = out.merge(table, on=keys, how="left")
            growth_cols.append(col)
        if growth_cols:
            out["seasonal_growth_factor"] = out[growth_cols].bfill(axis=1).iloc[:, 0]
        else:
            out["seasonal_growth_factor"] = self.global_growth_
        out["seasonal_growth_factor"] = out["seasonal_growth_factor"].fillna(self.global_growth_)

        last_year_cnt = out["last_year_same_mnth_day_hr_cnt"].where(
            out["has_last_year_same_mnth_day_hr"].astype(bool),
            out["seasonal_profile_fallback"],
        )
        out["seasonal_baseline_cnt"] = (
            last_year_cnt.fillna(self.global_mean_) * out["seasonal_growth_factor"]
        )
        out["seasonal_baseline_cnt"] = out["seasonal_baseline_cnt"].clip(lower=0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class SeasonalArxResult:
    name: str
    model: Regressor
    baseline: SeasonalBaselineEncoder
    feature_cols: list[str]
    valid_pred: np.ndarray
    mse: float
    best_iteration: int | None = None


@dataclass
class SeasonalArxFitted:
    name: str
    model: Regressor
    baseline: SeasonalBaselineEncoder
    feature_cols: list[str]
    best_iteration: int | None = None


def select_seasonal_arx_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = select_feature_columns(df)
    excluded = {
        SEASONAL_ARX_TARGET,
        "last_year_same_mnth_day_hr_cnt",
    }
    return [col for col in cols if col not in excluded and not col.startswith("seasonal_profile_")]


def build_training_frame(
    df: pd.DataFrame,
    baseline: SeasonalBaselineEncoder,
) -> pd.DataFrame:
    data = baseline.transform(df)
    data = add_training_lag_features(data)
    baseline_values = data["seasonal_baseline_cnt"].clip(lower=0).to_numpy()
    data[SEASONAL_ARX_TARGET] = np.log1p(data[TARGET_COL].to_numpy()) - np.log1p(baseline_values)
    if "yr" in data.columns:
        current_year_rows = data["yr"] > 0
        if current_year_rows.any():
            data = data[current_year_rows].copy()
    return data


def fit_seasonal_arx_model(
    model_name: str,
    train_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    n_estimators: int | None = None,
) -> SeasonalArxFitted:
    base_model_name = seasonal_arx_base_model_name(model_name)
    baseline = SeasonalBaselineEncoder().fit(train_df)
    train_frame = build_training_frame(train_df, baseline)
    feature_cols = select_seasonal_arx_feature_columns(train_frame)
    X_train = train_frame[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_train = train_frame[SEASONAL_ARX_TARGET]

    model = build_model(base_model_name, n_estimators=n_estimators, params=params)
    fit_model(base_model_name, model, X_train, y_train)
    return SeasonalArxFitted(
        name=model_name,
        model=model,
        baseline=baseline,
        feature_cols=feature_cols,
        best_iteration=get_best_iteration(base_model_name, model),
    )


def predict_seasonal_arx(
    fitted: SeasonalArxFitted,
    rows: pd.DataFrame,
    history_df: pd.DataFrame,
) -> np.ndarray:
    base_model_name = seasonal_arx_base_model_name(fitted.name)
    builder = LagFeatureBuilder().fit_history(history_df)
    preds = pd.Series(index=rows.index, dtype=float)

    sorted_rows = rows.sort_values(["dteday", "hr", ID_COL])
    for idx, row in sorted_rows.iterrows():
        row_with_baseline = fitted.baseline.transform(row.to_frame().T).iloc[0]
        row_fe = add_lag_features_for_row(row_with_baseline, builder)
        row_x = row_fe[fitted.feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        row_x_fit = prepare_features_for_model(base_model_name, row_x)
        residual_log = float(fitted.model.predict(row_x_fit)[0])
        baseline_cnt = float(row_fe["seasonal_baseline_cnt"].iloc[0])
        pred = float(np.expm1(np.log1p(max(baseline_cnt, 0.0)) + residual_log))
        pred = max(pred, 0.0)
        preds.loc[idx] = pred
        builder.add_prediction(row_timestamp(row), pred)

    return preds.loc[rows.index].to_numpy(dtype=float)


def train_and_evaluate_seasonal_arx(
    model_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    params: dict[str, Any] | None = None,
    n_estimators: int | None = None,
) -> SeasonalArxResult:
    fitted = fit_seasonal_arx_model(
        model_name,
        train_df,
        params=params,
        n_estimators=n_estimators,
    )
    valid_pred = predict_seasonal_arx(fitted, valid_df, train_df)
    return SeasonalArxResult(
        name=model_name,
        model=fitted.model,
        baseline=fitted.baseline,
        feature_cols=fitted.feature_cols,
        valid_pred=valid_pred,
        mse=mse(valid_df[TARGET_COL], valid_pred),
        best_iteration=fitted.best_iteration,
    )
