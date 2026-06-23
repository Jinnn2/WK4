from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any
from typing import Protocol

import numpy as np
import pandas as pd

from .config import SEED

EARLY_STOPPING_ROUNDS = 100


class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Regressor":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...


@dataclass
class TrainResult:
    name: str
    model: Regressor
    valid_pred: np.ndarray
    mse: float
    best_iteration: int | None = None


class MeanRegressor:
    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanRegressor":
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mean_, dtype=float)


class HourProfileRegressor:
    """Hierarchical mean baseline tuned for hourly bike demand."""

    def __init__(self) -> None:
        self.global_mean_: float = 0.0
        self.tables_: list[tuple[list[str], dict[tuple, float]]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HourProfileRegressor":
        data = X.copy()
        data["_target"] = y.to_numpy()
        self.global_mean_ = float(data["_target"].mean())

        key_sets = [
            ["mnth", "hr", "workingday", "weathersit"],
            ["mnth", "hr", "workingday"],
            ["hr", "workingday"],
            ["hr"],
        ]
        self.tables_ = []
        for keys in key_sets:
            if all(key in data.columns for key in keys):
                grouped = data.groupby(keys)["_target"].mean()
                table = {tuple(k if isinstance(k, tuple) else (k,)): float(v) for k, v in grouped.items()}
                self.tables_.append((keys, table))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds: list[float] = []
        for _, row in X.iterrows():
            pred = self.global_mean_
            for keys, table in self.tables_:
                key = tuple(row[k] for k in keys)
                if key in table:
                    pred = table[key]
                    break
            preds.append(pred)
        return np.asarray(preds, dtype=float)


def mse(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
    y_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_arr - pred_arr) ** 2))


def require_package(import_name: str, install_name: str | None = None) -> None:
    if find_spec(import_name) is None:
        package = install_name or import_name
        raise ImportError(
            f"Package '{package}' is required for this model. "
            f"Install dependencies with: python -m pip install -r requirements.txt"
        )


def base_model_name(name: str) -> str:
    normalized = name.lower().strip()
    if normalized.endswith("_log"):
        return normalized[: -len("_log")]
    return normalized


def uses_log_target(name: str) -> bool:
    return name.lower().strip().endswith("_log")


def transform_target(model_name: str, y: pd.Series) -> pd.Series | np.ndarray:
    if uses_log_target(model_name):
        return np.log1p(y)
    return y


def inverse_transform_predictions(model_name: str, pred: np.ndarray) -> np.ndarray:
    if uses_log_target(model_name):
        return np.expm1(pred)
    return pred


def default_model_params(name: str) -> dict[str, Any]:
    normalized = base_model_name(name)
    if normalized == "random_forest":
        return {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 2,
            "n_jobs": -1,
            "random_state": SEED,
        }
    if normalized == "hist_gradient_boosting":
        return {
            "loss": "squared_error",
            "learning_rate": 0.05,
            "max_iter": 700,
            "l2_regularization": 0.05,
            "random_state": SEED,
        }
    if normalized == "lightgbm":
        return {
            "objective": "regression",
            "n_estimators": 3000,
            "learning_rate": 0.03,
            "num_leaves": 64,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": SEED,
            "n_jobs": -1,
            "verbose": -1,
        }
    if normalized == "xgboost":
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 2500,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "random_state": SEED,
            "tree_method": "hist",
            "n_jobs": -1,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        }
    if normalized == "catboost":
        return {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "iterations": 3000,
            "learning_rate": 0.03,
            "depth": 6,
            "l2_leaf_reg": 5,
            "random_seed": SEED,
            "verbose": False,
            "allow_writing_files": False,
        }
    raise ValueError(f"Unknown model with configurable params: {name}")


def build_model(
    name: str,
    n_estimators: int | None = None,
    params: dict[str, Any] | None = None,
) -> Regressor:
    normalized = base_model_name(name)
    if normalized == "mean":
        return MeanRegressor()
    if normalized == "hour_profile":
        return HourProfileRegressor()
    if normalized == "random_forest":
        require_package("sklearn", "scikit-learn")
        from sklearn.ensemble import RandomForestRegressor

        model_params = default_model_params(normalized)
        model_params.update(params or {})
        return RandomForestRegressor(**model_params)
    if normalized == "hist_gradient_boosting":
        require_package("sklearn", "scikit-learn")
        from sklearn.ensemble import HistGradientBoostingRegressor

        model_params = default_model_params(normalized)
        model_params.update(params or {})
        if n_estimators is not None:
            model_params["max_iter"] = n_estimators
        return HistGradientBoostingRegressor(**model_params)
    if normalized == "lightgbm":
        require_package("lightgbm")
        from lightgbm import LGBMRegressor

        model_params = default_model_params(normalized)
        model_params.update(params or {})
        if n_estimators is not None:
            model_params["n_estimators"] = n_estimators
        return LGBMRegressor(**model_params)
    if normalized == "xgboost":
        require_package("xgboost")
        from xgboost import XGBRegressor

        model_params = default_model_params(normalized)
        model_params.update(params or {})
        if n_estimators is not None:
            model_params["n_estimators"] = n_estimators
            model_params.pop("early_stopping_rounds", None)
        return XGBRegressor(**model_params)
    if normalized == "catboost":
        require_package("catboost")
        from catboost import CatBoostRegressor

        model_params = default_model_params(normalized)
        model_params.update(params or {})
        if n_estimators is not None:
            model_params["iterations"] = n_estimators
        return CatBoostRegressor(**model_params)
    raise ValueError(f"Unknown model: {name}")


def supports_early_stopping(model_name: str) -> bool:
    return base_model_name(model_name) in {"lightgbm", "xgboost", "catboost"}


def fit_model(
    model_name: str,
    model: Regressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
) -> Regressor:
    normalized = base_model_name(model_name)
    if X_valid is None or y_valid is None or not supports_early_stopping(normalized):
        model.fit(X_train, y_train)
        return model

    if normalized == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            callbacks=[
                early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                log_evaluation(0),
            ],
        )
        return model

    if normalized == "xgboost":
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        return model

    if normalized == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            use_best_model=True,
            verbose=False,
        )
        return model

    model.fit(X_train, y_train)
    return model


def get_best_iteration(model_name: str, model: Regressor) -> int | None:
    normalized = base_model_name(model_name)
    if normalized == "lightgbm":
        best_iteration = getattr(model, "best_iteration_", None)
        return int(best_iteration) if best_iteration else None
    if normalized == "xgboost":
        best_iteration = getattr(model, "best_iteration", None)
        return int(best_iteration) + 1 if best_iteration is not None else None
    if normalized == "catboost" and hasattr(model, "get_best_iteration"):
        best_iteration = model.get_best_iteration()
        return int(best_iteration) + 1 if best_iteration is not None else None
    return None


def train_and_evaluate(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    params: dict[str, Any] | None = None,
) -> TrainResult:
    model = build_model(model_name, params=params)
    y_train_fit = transform_target(model_name, y_train)
    y_valid_fit = transform_target(model_name, y_valid)
    fit_model(model_name, model, X_train, y_train_fit, X_valid, y_valid_fit)
    valid_pred = inverse_transform_predictions(model_name, model.predict(X_valid))
    valid_pred = np.maximum(valid_pred, 0)
    best_iteration = get_best_iteration(model_name, model)
    return TrainResult(
        name=model_name,
        model=model,
        valid_pred=valid_pred,
        mse=mse(y_valid, valid_pred),
        best_iteration=best_iteration,
    )
