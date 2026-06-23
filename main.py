from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


RANDOM_SEED = 2026
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    target: str
    params: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fresh bike-sharing demand prediction experiment."
    )
    parser.add_argument("--train", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "test.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Last ratio of the chronologically sorted training set used as validation.",
    )
    return parser.parse_args()


def load_data(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    required_train = {"ID", "dteday", "hr", "cnt"}
    required_test = {"ID", "dteday", "hr"}
    missing_train = required_train - set(train.columns)
    missing_test = required_test - set(test.columns)
    if missing_train:
        raise ValueError(f"Training data is missing columns: {sorted(missing_train)}")
    if missing_test:
        raise ValueError(f"Test data is missing columns: {sorted(missing_test)}")
    return train.sort_values("ID").reset_index(drop=True), test.sort_values("ID").reset_index(drop=True)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dteday"] = pd.to_datetime(out["dteday"])
    out["date_ord"] = (out["dteday"] - out["dteday"].min()).dt.days
    out["day"] = out["dteday"].dt.day
    out["dayofyear"] = out["dteday"].dt.dayofyear
    out["weekofyear"] = out["dteday"].dt.isocalendar().week.astype(int)
    out["is_month_start"] = out["dteday"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["dteday"].dt.is_month_end.astype(int)

    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)
    out["is_sleep_hour"] = out["hr"].between(0, 5).astype(int)
    out["is_morning_rush"] = out["hr"].isin([7, 8, 9]).astype(int)
    out["is_evening_rush"] = out["hr"].isin([16, 17, 18, 19]).astype(int)
    out["is_rush_hour"] = ((out["is_morning_rush"] == 1) | (out["is_evening_rush"] == 1)).astype(int)
    out["workday_morning_rush"] = out["workingday"] * out["is_morning_rush"]
    out["workday_evening_rush"] = out["workingday"] * out["is_evening_rush"]
    out["nonworkday_daytime"] = (1 - out["workingday"]) * out["hr"].between(10, 18).astype(int)

    for col, period in [("hr", 24), ("weekday", 7), ("mnth", 12), ("dayofyear", 366)]:
        out[f"{col}_sin"] = np.sin(2 * np.pi * out[col] / period)
        out[f"{col}_cos"] = np.cos(2 * np.pi * out[col] / period)
    return out


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["temp_diff"] = out["atemp"] - out["temp"]
    out["temp_sq"] = out["temp"] ** 2
    out["hum_sq"] = out["hum"] ** 2
    out["windspeed_sq"] = out["windspeed"] ** 2
    out["temp_x_hum"] = out["temp"] * out["hum"]
    out["temp_x_wind"] = out["temp"] * out["windspeed"]
    out["hum_x_wind"] = out["hum"] * out["windspeed"]
    out["bad_weather"] = (out["weathersit"] >= 3).astype(int)
    out["pleasant_weather"] = (
        (out["weathersit"] == 1) & out["temp"].between(0.36, 0.74) & (out["hum"] <= 0.72)
    ).astype(int)
    out["weather_x_hour"] = out["weathersit"] * out["hr"]
    out["season_x_hour"] = out["season"] * out["hr"]
    out["season_x_temp"] = out["season"] * out["temp"]
    return out


def build_group_maps(train_fe: pd.DataFrame, y: pd.Series) -> dict[str, dict]:
    temp = train_fe.copy()
    temp["cnt"] = y.to_numpy()
    group_columns = {
        "hist_hour_mean": ["hr"],
        "hist_working_hour_mean": ["workingday", "hr"],
        "hist_month_hour_mean": ["mnth", "hr"],
        "hist_weather_hour_mean": ["weathersit", "hr"],
        "hist_season_hour_mean": ["season", "hr"],
        "hist_weekday_hour_mean": ["weekday", "hr"],
    }
    maps: dict[str, dict] = {}
    for name, cols in group_columns.items():
        grouped = temp.groupby(cols, observed=True)["cnt"].mean()
        maps[name] = {
            "columns": cols,
            "values": {tuple(k if isinstance(k, tuple) else (k,)): float(v) for k, v in grouped.items()},
            "fallback": float(y.mean()),
        }
    return maps


def apply_group_maps(df: pd.DataFrame, maps: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    for feature_name, spec in maps.items():
        cols = spec["columns"]
        values = spec["values"]
        fallback = spec["fallback"]
        keys = zip(*(out[col].to_numpy() for col in cols))
        out[feature_name] = [values.get(tuple(key), fallback) for key in keys]
    return out


def make_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stats_train: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    stats_source = train if stats_train is None else stats_train
    combined = pd.concat(
        [
            train.drop(columns=["cnt"], errors="ignore").assign(_is_train=1),
            test.drop(columns=["cnt"], errors="ignore").assign(_is_train=0),
        ],
        axis=0,
        ignore_index=True,
    )
    combined = add_weather_features(add_time_features(combined))

    train_base = combined.loc[combined["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_base = combined.loc[combined["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)

    stats_base = train_base.iloc[: len(stats_source)].copy()
    maps = build_group_maps(stats_base, stats_source["cnt"].reset_index(drop=True))
    train_fe = apply_group_maps(train_base, maps)
    test_fe = apply_group_maps(test_base, maps)

    drop_cols = {"ID", "dteday", "casual", "registered", "cnt"}
    feature_cols = [c for c in train_fe.columns if c not in drop_cols]
    categorical_cols = [
        c
        for c in [
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "is_weekend",
            "is_sleep_hour",
            "is_morning_rush",
            "is_evening_rush",
            "is_rush_hour",
            "bad_weather",
            "pleasant_weather",
            "is_month_start",
            "is_month_end",
        ]
        if c in feature_cols
    ]
    return train_fe, test_fe, feature_cols, categorical_cols


def split_train_valid(train: pd.DataFrame, valid_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.05 <= valid_ratio <= 0.5:
        raise ValueError("--valid-ratio must be between 0.05 and 0.5")
    split_idx = int(len(train) * (1 - valid_ratio))
    train_idx = np.arange(0, split_idx)
    valid_idx = np.arange(split_idx, len(train))
    return train_idx, valid_idx


def mse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, np.maximum(pred, 0)))


def rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return math.sqrt(mse(y_true, pred))


def instantiate_model(spec: ModelSpec, categorical_cols: Iterable[str]) -> object:
    if spec.kind == "lgbm":
        return LGBMRegressor(**spec.params)
    if spec.kind == "xgb":
        return XGBRegressor(**spec.params)
    if spec.kind == "cat":
        return CatBoostRegressor(**spec.params, cat_features=list(categorical_cols))
    raise ValueError(f"Unknown model kind: {spec.kind}")


def fit_predict(
    spec: ModelSpec,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[object, np.ndarray]:
    model = instantiate_model(spec, categorical_cols)
    target = np.log1p(y_train) if spec.target == "log1p" else y_train
    if spec.kind == "lgbm":
        cat_cols = [c for c in categorical_cols if c in x_train.columns]
        train_data = x_train.copy()
        valid_data = x_valid.copy()
        for col in cat_cols:
            train_data[col] = train_data[col].astype("category")
            valid_data[col] = valid_data[col].astype("category")
        model.fit(train_data, target, categorical_feature=cat_cols)
        raw_pred = model.predict(valid_data)
    else:
        model.fit(x_train, target)
        raw_pred = model.predict(x_valid)
    pred = np.expm1(raw_pred) if spec.target == "log1p" else raw_pred
    return model, np.maximum(pred, 0)


def model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="lgbm_raw",
            kind="lgbm",
            target="raw",
            params={
                "objective": "regression",
                "n_estimators": 900,
                "learning_rate": 0.035,
                "num_leaves": 80,
                "max_depth": 9,
                "min_child_samples": 24,
                "subsample": 0.86,
                "subsample_freq": 1,
                "colsample_bytree": 0.84,
                "reg_alpha": 0.05,
                "reg_lambda": 1.8,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "verbosity": -1,
            },
        ),
        ModelSpec(
            name="lgbm_log",
            kind="lgbm",
            target="log1p",
            params={
                "objective": "regression",
                "n_estimators": 850,
                "learning_rate": 0.035,
                "num_leaves": 63,
                "max_depth": 8,
                "min_child_samples": 18,
                "subsample": 0.9,
                "subsample_freq": 1,
                "colsample_bytree": 0.86,
                "reg_alpha": 0.01,
                "reg_lambda": 1.2,
                "random_state": RANDOM_SEED + 1,
                "n_jobs": -1,
                "verbosity": -1,
            },
        ),
        ModelSpec(
            name="xgb_raw",
            kind="xgb",
            target="raw",
            params={
                "objective": "reg:squarederror",
                "n_estimators": 760,
                "learning_rate": 0.035,
                "max_depth": 6,
                "min_child_weight": 3.0,
                "subsample": 0.88,
                "colsample_bytree": 0.86,
                "reg_alpha": 0.02,
                "reg_lambda": 1.5,
                "tree_method": "hist",
                "random_state": RANDOM_SEED + 2,
                "n_jobs": -1,
            },
        ),
        ModelSpec(
            name="cat_raw",
            kind="cat",
            target="raw",
            params={
                "loss_function": "RMSE",
                "iterations": 900,
                "learning_rate": 0.04,
                "depth": 7,
                "l2_leaf_reg": 6.0,
                "random_strength": 0.8,
                "bagging_temperature": 0.35,
                "allow_writing_files": False,
                "verbose": False,
                "random_seed": RANDOM_SEED + 3,
            },
        ),
    ]


def search_weights(valid_predictions: dict[str, np.ndarray], y_valid: np.ndarray) -> tuple[dict[str, float], float]:
    names = list(valid_predictions)
    pred_matrix = np.column_stack([valid_predictions[name] for name in names])
    best_score = float("inf")
    best_weights = None

    if len(names) != 4:
        weights_iter = _simplex_weights(len(names), step=0.05)
    else:
        weights_iter = (
            (a, b, c, round(1.0 - a - b - c, 10))
            for a in np.arange(0, 1.001, 0.05)
            for b in np.arange(0, 1.001 - a, 0.05)
            for c in np.arange(0, 1.001 - a - b, 0.05)
        )

    for weights in weights_iter:
        weights_arr = np.array(weights, dtype=float)
        if np.any(weights_arr < -1e-9) or abs(weights_arr.sum() - 1.0) > 1e-6:
            continue
        score = mse(y_valid, pred_matrix @ weights_arr)
        if score < best_score:
            best_score = score
            best_weights = weights_arr
    if best_weights is None:
        raise RuntimeError("Weight search failed.")
    return {name: float(weight) for name, weight in zip(names, best_weights)}, best_score


def _simplex_weights(n: int, step: float) -> Iterable[tuple[float, ...]]:
    if n == 1:
        yield (1.0,)
        return
    values = np.arange(0, 1.001, step)

    def rec(prefix: list[float], remain: float, slots: int) -> Iterable[tuple[float, ...]]:
        if slots == 1:
            yield tuple(prefix + [round(remain, 10)])
        else:
            for val in values:
                if val <= remain + 1e-9:
                    yield from rec(prefix + [float(val)], remain - float(val), slots - 1)

    yield from rec([], 1.0, n)


def evaluate_predictions(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    clipped = np.maximum(pred, 0)
    return {
        "mse": float(mean_squared_error(y_true, clipped)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, clipped))),
        "mae": float(mean_absolute_error(y_true, clipped)),
        "r2": float(r2_score(y_true, clipped)),
    }


def train_full_and_predict(
    specs: list[ModelSpec],
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, np.ndarray]:
    predictions = {}
    x_full = train[feature_cols]
    y_full = train["cnt"]
    x_test = test[feature_cols]
    for spec in specs:
        print(f"Training full {spec.name}...")
        _, pred = fit_predict(spec, x_full, y_full, x_test, categorical_cols)
        predictions[spec.name] = pred
    return predictions


def write_submission(test: pd.DataFrame, pred: np.ndarray, output_path: Path) -> None:
    final_pred = np.rint(np.maximum(pred, 0)).astype(int)
    submission = pd.DataFrame({"ID": test["ID"].astype(int), "cnt": final_pred})
    submission.to_csv(output_path, index=False)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw = load_data(args.train, args.test)
    train_idx, valid_idx = split_train_valid(train_raw, args.valid_ratio)

    stats_train = train_raw.iloc[train_idx].reset_index(drop=True)
    train_fe, test_fe, feature_cols, categorical_cols = make_features(
        train_raw, test_raw, stats_train=stats_train
    )
    x_train = train_fe.iloc[train_idx][feature_cols]
    y_train = train_raw.iloc[train_idx]["cnt"].reset_index(drop=True)
    x_valid = train_fe.iloc[valid_idx][feature_cols]
    y_valid = train_raw.iloc[valid_idx]["cnt"].to_numpy()

    specs = model_specs()
    valid_predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}

    print(f"Training rows: {len(train_idx)}, validation rows: {len(valid_idx)}")
    print(f"Feature count: {len(feature_cols)}")
    for spec in specs:
        print(f"Training validation {spec.name}...")
        _, pred = fit_predict(spec, x_train, y_train, x_valid, categorical_cols)
        valid_predictions[spec.name] = pred
        metrics[spec.name] = evaluate_predictions(y_valid, pred)
        print(
            f"  {spec.name}: MSE={metrics[spec.name]['mse']:.3f}, "
            f"RMSE={metrics[spec.name]['rmse']:.3f}, MAE={metrics[spec.name]['mae']:.3f}"
        )

    weights, ensemble_mse = search_weights(valid_predictions, y_valid)
    valid_ensemble = sum(valid_predictions[name] * weight for name, weight in weights.items())
    metrics["ensemble"] = evaluate_predictions(y_valid, valid_ensemble)
    print(f"Best weights: {weights}")
    print(
        f"  ensemble: MSE={metrics['ensemble']['mse']:.3f}, "
        f"RMSE={metrics['ensemble']['rmse']:.3f}, MAE={metrics['ensemble']['mae']:.3f}"
    )

    # Rebuild historical aggregate features from the full training set before final fitting.
    full_train_fe, full_test_fe, feature_cols, categorical_cols = make_features(
        train_raw, test_raw, stats_train=train_raw
    )
    test_predictions = train_full_and_predict(
        specs, full_train_fe.assign(cnt=train_raw["cnt"]), full_test_fe, feature_cols, categorical_cols
    )
    final_pred = sum(test_predictions[name] * weight for name, weight in weights.items())

    submission_path = args.output_dir / "submission_fresh_ensemble.csv"
    metrics_path = args.output_dir / "fresh_experiment_metrics.json"
    write_submission(test_raw, final_pred, submission_path)

    report = {
        "data": {
            "train_shape": list(train_raw.shape),
            "test_shape": list(test_raw.shape),
            "train_id_range": [int(train_raw["ID"].min()), int(train_raw["ID"].max())],
            "test_id_range": [int(test_raw["ID"].min()), int(test_raw["ID"].max())],
            "train_date_range": [str(pd.to_datetime(train_raw["dteday"]).min().date()), str(pd.to_datetime(train_raw["dteday"]).max().date())],
            "test_date_range": [str(pd.to_datetime(test_raw["dteday"]).min().date()), str(pd.to_datetime(test_raw["dteday"]).max().date())],
        },
        "validation": {
            "method": f"last {args.valid_ratio:.0%} by chronological ID order",
            "train_rows": int(len(train_idx)),
            "valid_rows": int(len(valid_idx)),
        },
        "features": {
            "count": len(feature_cols),
            "columns": feature_cols,
            "categorical_columns": categorical_cols,
        },
        "metrics": metrics,
        "ensemble_weights": weights,
        "ensemble_weight_search_mse": ensemble_mse,
        "submission": str(submission_path.relative_to(ROOT)),
    }
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {submission_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
