from __future__ import annotations

import itertools

import numpy as np

from .models import mse


def weighted_average(predictions: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(predictions) != len(weights):
        raise ValueError("predictions and weights must have the same length")
    total = np.zeros_like(predictions[0], dtype=float)
    for pred, weight in zip(predictions, weights):
        total += pred * weight
    return total


def search_best_weights(
    valid_predictions: list[np.ndarray],
    y_valid: np.ndarray,
    step: float = 0.05,
) -> tuple[list[float], float]:
    n_models = len(valid_predictions)
    if n_models == 1:
        return [1.0], mse(y_valid, valid_predictions[0])
    if n_models > 4:
        raise ValueError("grid weight search is intended for up to 4 models")

    grid = np.round(np.arange(0, 1 + step, step), 10)
    best_score = float("inf")
    best_weights: list[float] | None = None

    for weights_prefix in itertools.product(grid, repeat=n_models - 1):
        prefix_sum = float(sum(weights_prefix))
        if prefix_sum > 1:
            continue
        weights = list(weights_prefix) + [1 - prefix_sum]
        pred = weighted_average(valid_predictions, weights)
        score = mse(y_valid, pred)
        if score < best_score:
            best_score = score
            best_weights = weights

    if best_weights is None:
        raise RuntimeError("failed to find ensemble weights")
    return best_weights, best_score

