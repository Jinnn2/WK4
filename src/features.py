from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COL, TARGET_COL

PROFILE_KEY_SETS = [
    ["mnth", "hr", "workingday", "weathersit"],
    ["mnth", "hr", "workingday"],
    ["season", "hr", "workingday"],
    ["weekday", "hr"],
    ["hr", "workingday"],
    ["hr"],
]

GROWTH_KEY_SETS = [
    ["mnth", "hr", "workingday"],
    ["mnth", "workingday"],
    ["hr", "workingday"],
    ["mnth"],
    ["hr"],
]


def add_cyclic_feature(df: pd.DataFrame, col: str, period: int) -> None:
    if col not in df.columns:
        return
    radians = 2 * np.pi * df[col].astype(float) / period
    df[f"{col}_sin"] = np.sin(radians)
    df[f"{col}_cos"] = np.cos(radians)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if DATE_COL in out.columns:
        dt = pd.to_datetime(out[DATE_COL])
        out["year_abs"] = dt.dt.year
        out["day"] = dt.dt.day
        out["dayofyear"] = dt.dt.dayofyear
        out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
        out["is_month_start"] = dt.dt.is_month_start.astype(int)
        out["is_month_end"] = dt.dt.is_month_end.astype(int)

    add_cyclic_feature(out, "hr", 24)
    add_cyclic_feature(out, "mnth", 12)
    add_cyclic_feature(out, "weekday", 7)
    add_cyclic_feature(out, "dayofyear", 366)

    if "weekday" in out.columns:
        out["is_weekend"] = out["weekday"].isin([5, 6]).astype(int)

    if "hr" in out.columns:
        out["is_morning_rush"] = out["hr"].isin([7, 8, 9]).astype(int)
        out["is_evening_rush"] = out["hr"].isin([17, 18, 19]).astype(int)
        out["is_rush_hour"] = (
            (out["is_morning_rush"] == 1) | (out["is_evening_rush"] == 1)
        ).astype(int)
        out["is_night"] = out["hr"].isin([0, 1, 2, 3, 4, 5]).astype(int)
        out["is_work_hour"] = out["hr"].between(9, 17).astype(int)

    if {"is_rush_hour", "workingday"}.issubset(out.columns):
        out["commute_rush"] = out["is_rush_hour"] * out["workingday"]

    if {"temp", "hum"}.issubset(out.columns):
        out["temp_hum"] = out["temp"] * out["hum"]
    if {"temp", "windspeed"}.issubset(out.columns):
        out["temp_windspeed"] = out["temp"] * out["windspeed"]
    if {"hum", "windspeed"}.issubset(out.columns):
        out["hum_windspeed"] = out["hum"] * out["windspeed"]
    if {"temp", "atemp"}.issubset(out.columns):
        out["feels_temp_gap"] = out["atemp"] - out["temp"]

    if "weathersit" in out.columns:
        out["bad_weather"] = (out["weathersit"] >= 3).astype(int)
    if {"weathersit", "hr"}.issubset(out.columns):
        out["weather_hour"] = out["weathersit"] * out["hr"]
    if {"season", "hr"}.issubset(out.columns):
        out["season_hour"] = out["season"] * out["hr"]
    if {"season", "temp"}.issubset(out.columns):
        out["season_temp"] = out["season"] * out["temp"]

    return out


def align_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_fe = create_features(train)
    test_fe = create_features(test)

    missing_in_test = [col for col in train_fe.columns if col not in test_fe.columns]
    missing_in_train = [col for col in test_fe.columns if col not in train_fe.columns]

    for col in missing_in_test:
        if col != "cnt":
            test_fe[col] = 0
    for col in missing_in_train:
        train_fe[col] = 0

    test_fe = test_fe[[col for col in train_fe.columns if col in test_fe.columns]]
    return train_fe, test_fe


class LastYearFeatureEncoder:
    """Adds last-year same month/day/hour priors and observed year-growth factors."""

    def __init__(self, smoothing: float = 30.0) -> None:
        self.smoothing = smoothing
        self.global_growth_: float = 1.0
        self.last_year_table_: pd.DataFrame | None = None
        self.growth_tables_: list[tuple[str, list[str], pd.DataFrame]] = []

    def fit(self, df: pd.DataFrame) -> "LastYearFeatureEncoder":
        if TARGET_COL not in df.columns:
            raise ValueError(f"LastYearFeatureEncoder requires target column '{TARGET_COL}'")
        required = {"yr", "mnth", "day", "hr", "workingday", TARGET_COL}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"LastYearFeatureEncoder missing required columns: {missing}")

        last_year = df[df["yr"] == 0]
        current_year = df[df["yr"] == 1]

        self.last_year_table_ = (
            last_year.groupby(["mnth", "day", "hr"], dropna=False)[TARGET_COL]
            .mean()
            .reset_index()
            .rename(columns={TARGET_COL: "last_year_same_mnth_day_hr_cnt"})
        )

        ly_mean = float(last_year[TARGET_COL].mean()) if len(last_year) else float(df[TARGET_COL].mean())
        cy_mean = float(current_year[TARGET_COL].mean()) if len(current_year) else float(df[TARGET_COL].mean())
        self.global_growth_ = cy_mean / ly_mean if ly_mean > 0 else 1.0

        self.growth_tables_ = []
        for keys in GROWTH_KEY_SETS:
            if not all(key in df.columns for key in keys):
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
            name = "growth_" + "_".join(keys)
            table = table[keys + ["growth"]].rename(columns={"growth": name})
            self.growth_tables_.append((name, keys, table))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.last_year_table_ is None:
            raise RuntimeError("LastYearFeatureEncoder must be fitted before transform")

        out = df.copy()
        out = out.merge(self.last_year_table_, on=["mnth", "day", "hr"], how="left")
        has_prior_year = out["yr"] > 0 if "yr" in out.columns else pd.Series(True, index=out.index)
        exact_match = out["last_year_same_mnth_day_hr_cnt"].notna() & has_prior_year

        fallback_keys = ["mnth", "hr"]
        if all(key in out.columns for key in fallback_keys):
            fallback = self.last_year_table_.groupby(fallback_keys)[
                "last_year_same_mnth_day_hr_cnt"
            ].mean().reset_index()
            fallback = fallback.rename(
                columns={"last_year_same_mnth_day_hr_cnt": "last_year_mnth_hr_cnt"}
            )
            out = out.merge(fallback, on=fallback_keys, how="left")
        else:
            out["last_year_mnth_hr_cnt"] = np.nan

        out.loc[~has_prior_year, "last_year_same_mnth_day_hr_cnt"] = np.nan
        out.loc[~has_prior_year, "last_year_mnth_hr_cnt"] = np.nan
        out["last_year_same_mnth_day_hr_cnt"] = out[
            "last_year_same_mnth_day_hr_cnt"
        ].fillna(out["last_year_mnth_hr_cnt"])
        out["last_year_same_mnth_day_hr_cnt"] = out[
            "last_year_same_mnth_day_hr_cnt"
        ].fillna(0.0)
        out["has_last_year_same_mnth_day_hr"] = exact_match.astype(int)

        growth_cols: list[str] = []
        for name, keys, table in self.growth_tables_:
            out = out.merge(table, on=keys, how="left")
            growth_cols.append(name)

        if growth_cols:
            out["year_growth_factor"] = out[growth_cols].bfill(axis=1).iloc[:, 0]
            for col in growth_cols:
                out[col] = out[col].fillna(self.global_growth_)
        else:
            out["year_growth_factor"] = self.global_growth_
        out["year_growth_factor"] = out["year_growth_factor"].fillna(self.global_growth_)
        out.loc[~has_prior_year, "year_growth_factor"] = 1.0
        out["last_year_growth_adjusted_cnt"] = (
            out["last_year_same_mnth_day_hr_cnt"] * out["year_growth_factor"]
        )
        out["last_year_residual_vs_growth"] = (
            out["last_year_growth_adjusted_cnt"] - out["last_year_same_mnth_day_hr_cnt"]
        )
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class TargetProfileEncoder:
    """Adds target mean/count features fitted only on the provided training rows."""

    def __init__(self, key_sets: list[list[str]] | None = None, smoothing: float = 20.0) -> None:
        self.key_sets = key_sets or PROFILE_KEY_SETS
        self.smoothing = smoothing
        self.global_mean_: float = 0.0
        self.tables_: list[tuple[str, list[str], pd.DataFrame]] = []

    def fit(self, df: pd.DataFrame) -> "TargetProfileEncoder":
        if TARGET_COL not in df.columns:
            raise ValueError(f"TargetProfileEncoder requires target column '{TARGET_COL}'")

        self.global_mean_ = float(df[TARGET_COL].mean())
        self.tables_ = []
        for keys in self.key_sets:
            if not all(key in df.columns for key in keys):
                continue
            name = "profile_" + "_".join(keys)
            grouped = (
                df.groupby(keys, dropna=False)[TARGET_COL]
                .agg(["mean", "count"])
                .reset_index()
            )
            grouped[f"{name}_mean"] = (
                grouped["mean"] * grouped["count"] + self.global_mean_ * self.smoothing
            ) / (grouped["count"] + self.smoothing)
            grouped[f"{name}_count"] = np.log1p(grouped["count"])
            grouped = grouped[keys + [f"{name}_mean", f"{name}_count"]]
            self.tables_.append((name, keys, grouped))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for name, keys, table in self.tables_:
            out = out.merge(table, on=keys, how="left")
            out[f"{name}_mean"] = out[f"{name}_mean"].fillna(self.global_mean_)
            out[f"{name}_count"] = out[f"{name}_count"].fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def add_target_profile_features(
    train_part: pd.DataFrame,
    valid_part: pd.DataFrame,
    test_part: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    encoder = TargetProfileEncoder().fit(train_part)
    return (
        encoder.transform(train_part),
        encoder.transform(valid_part),
        encoder.transform(test_part),
    )
