from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import OUTPUT_DIR, TARGET_COL, VALID_FRACTION
from src.data_utils import load_data, select_feature_columns, time_order_split
from src.features import align_train_test
from src.models import get_best_iteration, mse, require_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune tree models with Optuna.")
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "catboost"], default="xgboost")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--valid-fraction", type=float, default=VALID_FRACTION)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "params"))
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds")
    parser.add_argument("--show-progress", action="store_true", help="Show Optuna progress bar")
    return parser.parse_args()


def suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    if model_name == "lightgbm":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 24, 160),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 8.0, log=True),
        }
    if model_name == "xgboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 4.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 8.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }
    if model_name == "catboost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 9),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        }
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def build_trial_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "lightgbm":
        require_package("lightgbm")
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            n_estimators=3000,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **params,
        )
    if model_name == "xgboost":
        require_package("xgboost")
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=2500,
            random_state=42,
            tree_method="hist",
            n_jobs=-1,
            early_stopping_rounds=100,
            **params,
        )
    if model_name == "catboost":
        require_package("catboost")
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            iterations=3000,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            **params,
        )
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def fit_trial_model(
    model_name: str,
    model: Any,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
) -> None:
    if model_name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="l2",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        return
    if model_name == "xgboost":
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        return
    if model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=100,
            use_best_model=True,
            verbose=False,
        )
        return
    raise ValueError(f"Unsupported model for tuning: {model_name}")


def main() -> None:
    args = parse_args()
    require_package("optuna")
    import optuna

    bundle = load_data()
    train_fe, _ = align_train_test(bundle.train, bundle.test)
    train_part, valid_part = time_order_split(train_fe, valid_fraction=args.valid_fraction)
    feature_cols = select_feature_columns(train_fe)

    X_train = train_part[feature_cols]
    y_train = train_part[TARGET_COL]
    X_valid = valid_part[feature_cols]
    y_valid = valid_part[TARGET_COL]

    def objective(trial: Any) -> float:
        params = suggest_params(trial, args.model)
        model = build_trial_model(args.model, params)
        fit_trial_model(args.model, model, X_train, y_train, X_valid, y_valid)
        pred = np.maximum(model.predict(X_valid), 0)
        score = mse(y_valid, pred)
        best_iteration = get_best_iteration(args.model, model)
        if best_iteration is not None:
            trial.set_user_attr("best_iteration", best_iteration)
        return score

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=args.show_progress,
    )

    best_iteration = study.best_trial.user_attrs.get("best_iteration")
    output = {
        "model": args.model,
        "valid_mse": study.best_value,
        "best_iteration": best_iteration,
        "params": study.best_params,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model}.json"
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"best {args.model} valid MSE: {study.best_value:.4f}")
    if best_iteration is not None:
        print(f"best iteration: {best_iteration}")
    print(f"saved params: {output_path}")


if __name__ == "__main__":
    main()
