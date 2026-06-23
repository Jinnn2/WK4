from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import OUTPUT_DIR, TARGET_COL, VALID_FRACTION
from src.data_utils import load_data, select_feature_columns, split_train_valid
from src.ensemble import search_best_weights, weighted_average
from src.features import (
    LastYearFeatureEncoder,
    TargetProfileEncoder,
    add_target_profile_features,
    align_train_test,
)
from src.lag_features import LagFeatureBuilder, add_lag_features_for_row, add_training_lag_features
from src.make_submission import save_submission
from src.models import (
    base_model_name,
    build_model,
    fit_model,
    inverse_transform_predictions,
    train_and_evaluate,
    transform_target,
)


def load_tuned_params(
    model_name: str,
    params_dir: Path | None,
) -> tuple[dict[str, Any] | None, int | None, Path | None]:
    if params_dir is None:
        return None, None, None
    path = params_dir / f"{model_name}.json"
    if not path.exists() and model_name != base_model_name(model_name):
        path = params_dir / f"{base_model_name(model_name)}.json"
    if not path.exists():
        return None, None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params")
    best_iteration = payload.get("best_iteration")
    if not isinstance(params, dict):
        raise ValueError(f"Invalid params file: {path}")
    return params, int(best_iteration) if best_iteration is not None else None, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bike sharing hourly demand prediction")
    parser.add_argument(
        "--models",
        default="hour_profile",
        help=(
            "Comma-separated model names. Available: mean,hour_profile,random_forest,"
            "hist_gradient_boosting,lightgbm,xgboost,catboost and *_log variants"
        ),
    )
    parser.add_argument("--valid-fraction", type=float, default=VALID_FRACTION)
    parser.add_argument(
        "--split-strategy",
        choices=["random", "time"],
        default="random",
        help="Validation split strategy. Use random for shuffled validation or time for chronological holdout.",
    )
    parser.add_argument("--output", default=str(OUTPUT_DIR / "submission.csv"))
    parser.add_argument(
        "--params-dir",
        default=None,
        help="Optional directory containing tuned <model>.json files from src/tune_optuna.py",
    )
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
    parser.add_argument(
        "--lag-features",
        action="store_true",
        help="Enable recursive lag and rolling cnt features for validation/test prediction",
    )
    parser.add_argument("--no-round", action="store_true", help="Keep float predictions in submission")
    return parser.parse_args()


def row_timestamp(row: pd.Series) -> pd.Timestamp:
    return pd.to_datetime(row["dteday"]) + pd.to_timedelta(row["hr"], unit="h")


def predict_sequential(
    models: list[Any],
    model_names: list[str],
    weights: list[float],
    rows: pd.DataFrame,
    feature_cols: list[str],
    history_df: pd.DataFrame,
) -> np.ndarray:
    builder = LagFeatureBuilder().fit_history(history_df)
    preds: list[float] = []

    for _, row in rows.sort_values(["dteday", "hr", "ID"]).iterrows():
        row_fe = add_lag_features_for_row(row, builder)
        row_x = row_fe[feature_cols].apply(pd.to_numeric, errors="coerce")
        row_x = row_x.fillna(0.0)
        row_pred_parts = []
        for model_name, model in zip(model_names, models):
            pred = model.predict(row_x)
            pred = inverse_transform_predictions(model_name, pred)
            row_pred_parts.append(np.maximum(pred, 0))
        pred_value = float(weighted_average(row_pred_parts, weights)[0])
        preds.append(pred_value)
        builder.add_prediction(row_timestamp(row), pred_value)

    return np.asarray(preds, dtype=float)


def main() -> None:
    args = parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("At least one model name is required")
    params_dir = Path(args.params_dir) if args.params_dir else None

    bundle = load_data()
    train_fe, test_fe = align_train_test(bundle.train, bundle.test)
    train_part, valid_part = split_train_valid(
        train_fe,
        valid_fraction=args.valid_fraction,
        strategy=args.split_strategy,
    )

    if args.last_year_features:
        valid_encoder = LastYearFeatureEncoder().fit(train_part)
        train_base = valid_encoder.transform(train_part)
        valid_base = valid_encoder.transform(valid_part)
        full_encoder = LastYearFeatureEncoder().fit(train_fe)
        full_base = full_encoder.transform(train_fe)
        test_base = full_encoder.transform(test_fe)
        print("last-year features: enabled")
    else:
        train_base = train_part
        valid_base = valid_part
        full_base = train_fe
        test_base = test_fe
        print("last-year features: disabled")

    if args.profile_features:
        train_model, valid_model, _ = add_target_profile_features(train_base, valid_base, test_base)
        profile_encoder = TargetProfileEncoder().fit(full_base)
        full_model = profile_encoder.transform(full_base)
        test_model = profile_encoder.transform(test_base)
        print("target profile features: enabled")
    else:
        train_model = train_base
        valid_model = valid_base
        full_model = full_base
        test_model = test_base
        print("target profile features: disabled")

    if args.lag_features:
        train_model = add_training_lag_features(train_model)
        full_model = add_training_lag_features(full_model)
        print("lag features: enabled")
    else:
        print("lag features: disabled")

    feature_cols = select_feature_columns(train_model)
    X_train = train_model[feature_cols]
    y_train = train_model[TARGET_COL]
    X_valid = valid_model[feature_cols] if not args.lag_features else None
    y_valid = valid_model[TARGET_COL]
    X_full = full_model[feature_cols]
    y_full = full_model[TARGET_COL]
    X_test = test_model[feature_cols] if not args.lag_features else None

    print(f"train rows: {len(train_fe)}, test rows: {len(test_fe)}")
    print(f"feature count: {len(feature_cols)}")
    print(f"validation rows: {len(valid_part)}")
    print(f"split strategy: {args.split_strategy}")

    valid_predictions = []
    trained_results = []
    tuned_params_by_model: dict[str, dict[str, Any] | None] = {}
    for model_name in model_names:
        tuned_params, _, tuned_path = load_tuned_params(model_name, params_dir)
        tuned_params_by_model[model_name] = tuned_params
        if tuned_params is not None:
            print(f"{model_name} loaded tuned params from: {tuned_path}")
        if args.lag_features:
            model = build_model(model_name, params=tuned_params)
            fit_model(model_name, model, X_train, transform_target(model_name, y_train))
            valid_pred = predict_sequential(
                [model],
                [model_name],
                [1.0],
                valid_model,
                feature_cols,
                train_model,
            )
            from src.models import TrainResult, get_best_iteration, mse

            result = TrainResult(
                name=model_name,
                model=model,
                valid_pred=valid_pred,
                mse=mse(y_valid, valid_pred),
                best_iteration=get_best_iteration(model_name, model),
            )
        else:
            result = train_and_evaluate(
                model_name,
                X_train,
                y_train,
                X_valid,
                y_valid,
                params=tuned_params,
            )
        trained_results.append(result)
        valid_predictions.append(result.valid_pred)
        print(f"{result.name} valid MSE: {result.mse:.4f}")
        if result.best_iteration is not None:
            print(f"{result.name} best iteration: {result.best_iteration}")

    weights, ensemble_mse = search_best_weights(valid_predictions, y_valid.to_numpy())
    print("ensemble weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(model_names, weights)))
    print(f"ensemble valid MSE: {ensemble_mse:.4f}")

    test_predictions = []
    final_models = []
    for model_name, result in zip(model_names, trained_results):
        final_model = build_model(
            model_name,
            n_estimators=result.best_iteration,
            params=tuned_params_by_model.get(model_name),
        )
        fit_model(model_name, final_model, X_full, transform_target(model_name, y_full))
        final_models.append(final_model)
        if not args.lag_features:
            test_pred = inverse_transform_predictions(model_name, final_model.predict(X_test))
            test_pred = np.maximum(test_pred, 0)
            test_predictions.append(test_pred)

    if args.lag_features:
        final_pred = predict_sequential(
            final_models,
            model_names,
            weights,
            test_model,
            feature_cols,
            full_model,
        )
    else:
        final_pred = weighted_average(test_predictions, weights)
    submission = save_submission(
        test_ids=bundle.test["ID"],
        pred=final_pred,
        output_path=OUTPUT_DIR / "submission.csv" if args.output is None else Path(args.output),
        round_to_int=not args.no_round,
    )
    print(f"saved submission: {args.output}")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
