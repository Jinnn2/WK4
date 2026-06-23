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
from src.data_utils import load_data, select_feature_columns, split_train_valid
from src.features import LastYearFeatureEncoder, add_target_profile_features, align_train_test
from src.models import (
    base_model_name,
    build_model,
    core_model_name,
    fit_model,
    get_best_iteration,
    inverse_transform_predictions,
    mse,
    prepare_features_for_model,
    require_package,
    transform_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune tree models with Optuna.")
    parser.add_argument(
        "--model",
        choices=[
            "lightgbm",
            "xgboost",
            "catboost",
            "lightgbm_log",
            "xgboost_log",
            "catboost_log",
            "lightgbm_cat",
            "catboost_cat",
            "lightgbm_cat_log",
            "catboost_cat_log",
        ],
        default="xgboost",
    )
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--valid-fraction", type=float, default=VALID_FRACTION)
    parser.add_argument(
        "--valid-size",
        default="test",
        help=(
            "Chronological validation size. Use 'test' to match the test row count, "
            "'fraction' to use --valid-fraction, or an integer row count."
        ),
    )
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default="time",
        help="Validation split strategy used by the tuning objective.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / "params"))
    parser.add_argument("--timeout", type=int, default=None, help="Optional Optuna timeout in seconds")
    parser.add_argument("--show-progress", action="store_true", help="Show Optuna progress bar")
    parser.add_argument(
        "--profile-features",
        action="store_true",
        help="Enable target profile mean/count features fitted on the training window",
    )
    parser.add_argument(
        "--last-year-features",
        action="store_true",
        help="Enable experimental last-year same month/day/hour and year-growth features",
    )
    return parser.parse_args()


def resolve_valid_size(value: str, test_rows: int) -> int | None:
    normalized = value.lower().strip()
    if normalized == "test":
        return test_rows
    if normalized in {"fraction", "none"}:
        return None
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError("--valid-size must be 'test', 'fraction', or an integer") from exc


def suggest_params(trial: Any, model_name: str) -> dict[str, Any]:
    model_name = core_model_name(model_name)
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
    return build_model(model_name, params=params)


def fit_trial_model(
    model_name: str,
    model: Any,
    X_train: Any,
    y_train: Any,
    X_valid: Any,
    y_valid: Any,
) -> None:
    fit_model(model_name, model, X_train, y_train, X_valid, y_valid)


def main() -> None:
    args = parse_args()
    require_package("optuna")
    import optuna

    bundle = load_data()
    train_fe, test_fe = align_train_test(bundle.train, bundle.test)
    valid_size = resolve_valid_size(args.valid_size, len(test_fe))
    train_part, valid_part = split_train_valid(
        train_fe,
        valid_fraction=args.valid_fraction,
        strategy=args.split_strategy,
        valid_size=valid_size if args.split_strategy == "time" else None,
    )

    if args.last_year_features:
        encoder = LastYearFeatureEncoder().fit(train_part)
        train_base = encoder.transform(train_part)
        valid_base = encoder.transform(valid_part)
        test_base = encoder.transform(test_fe)
    else:
        train_base = train_part
        valid_base = valid_part
        test_base = test_fe

    if args.profile_features:
        train_model, valid_model, _ = add_target_profile_features(train_base, valid_base, test_base)
    else:
        train_model = train_base
        valid_model = valid_base

    feature_cols = select_feature_columns(train_model)

    X_train = train_model[feature_cols]
    y_train = transform_target(args.model, train_model[TARGET_COL])
    X_valid = valid_model[feature_cols]
    y_valid = transform_target(args.model, valid_model[TARGET_COL])
    y_valid_raw = valid_model[TARGET_COL]

    def objective(trial: Any) -> float:
        params = suggest_params(trial, args.model)
        model = build_trial_model(args.model, params)
        fit_trial_model(args.model, model, X_train, y_train, X_valid, y_valid)
        X_valid_fit = prepare_features_for_model(args.model, X_valid)
        pred = inverse_transform_predictions(args.model, model.predict(X_valid_fit))
        pred = np.maximum(pred, 0)
        score = mse(y_valid_raw, pred)
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
