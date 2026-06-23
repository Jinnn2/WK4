from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config import OUTPUT_DIR, TARGET_COL, VALID_FRACTION
from src.data_utils import load_data, select_feature_columns, time_order_split
from src.ensemble import search_best_weights, weighted_average
from src.features import align_train_test
from src.make_submission import save_submission
from src.models import build_model, fit_model, train_and_evaluate


def load_tuned_params(model_name: str, params_dir: Path | None) -> tuple[dict[str, Any] | None, int | None]:
    if params_dir is None:
        return None, None
    path = params_dir / f"{model_name}.json"
    if not path.exists():
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("params")
    best_iteration = payload.get("best_iteration")
    if not isinstance(params, dict):
        raise ValueError(f"Invalid params file: {path}")
    return params, int(best_iteration) if best_iteration is not None else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bike sharing hourly demand prediction")
    parser.add_argument(
        "--models",
        default="hour_profile",
        help=(
            "Comma-separated model names. Available: mean,hour_profile,random_forest,"
            "hist_gradient_boosting,lightgbm,xgboost,catboost"
        ),
    )
    parser.add_argument("--valid-fraction", type=float, default=VALID_FRACTION)
    parser.add_argument("--output", default=str(OUTPUT_DIR / "submission.csv"))
    parser.add_argument(
        "--params-dir",
        default=None,
        help="Optional directory containing tuned <model>.json files from src/tune_optuna.py",
    )
    parser.add_argument("--no-round", action="store_true", help="Keep float predictions in submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("At least one model name is required")
    params_dir = Path(args.params_dir) if args.params_dir else None

    bundle = load_data()
    train_fe, test_fe = align_train_test(bundle.train, bundle.test)
    train_part, valid_part = time_order_split(train_fe, valid_fraction=args.valid_fraction)

    feature_cols = select_feature_columns(train_fe)
    X_train = train_part[feature_cols]
    y_train = train_part[TARGET_COL]
    X_valid = valid_part[feature_cols]
    y_valid = valid_part[TARGET_COL]
    X_full = train_fe[feature_cols]
    y_full = train_fe[TARGET_COL]
    X_test = test_fe[feature_cols]

    print(f"train rows: {len(train_fe)}, test rows: {len(test_fe)}")
    print(f"feature count: {len(feature_cols)}")
    print(f"validation rows: {len(valid_part)}")

    valid_predictions = []
    trained_results = []
    tuned_params_by_model: dict[str, dict[str, Any] | None] = {}
    for model_name in model_names:
        tuned_params, _ = load_tuned_params(model_name, params_dir)
        tuned_params_by_model[model_name] = tuned_params
        if tuned_params is not None:
            print(f"{model_name} loaded tuned params from: {params_dir / f'{model_name}.json'}")
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
    for model_name, result in zip(model_names, trained_results):
        final_model = build_model(
            model_name,
            n_estimators=result.best_iteration,
            params=tuned_params_by_model.get(model_name),
        )
        fit_model(model_name, final_model, X_full, y_full)
        test_pred = np.maximum(final_model.predict(X_test), 0)
        test_predictions.append(test_pred)

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
