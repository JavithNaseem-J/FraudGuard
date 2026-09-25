from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix

DEFAULT_FALSE_POSITIVE_COST = 1.0
DEFAULT_FALSE_NEGATIVE_COST = 20.0


def cost_weighted_loss(
    y_true: np.ndarray,
    predictions: np.ndarray,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
) -> dict:
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    total_cost = (
        false_positive * false_positive_cost + false_negative * false_negative_cost
    )
    row_count = max(len(y_true), 1)
    return {
        "false_positive_cost": float(false_positive_cost),
        "false_negative_cost": float(false_negative_cost),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
        "true_negatives": int(true_negative),
        "total_cost": float(total_cost),
        "average_cost": float(total_cost / row_count),
    }


def select_cost_weighted_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    false_positive_cost: float = DEFAULT_FALSE_POSITIVE_COST,
    false_negative_cost: float = DEFAULT_FALSE_NEGATIVE_COST,
) -> dict:
    thresholds = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best: dict | None = None
    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)
        cost = cost_weighted_loss(
            y_true,
            predictions,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        candidate = {
            "optimal_threshold": float(threshold),
            "objective": "out_of_fold_cost_weighted_loss",
            **cost,
        }
        if best is None or (
            candidate["average_cost"],
            -candidate["optimal_threshold"],
        ) < (
            best["average_cost"],
            -best["optimal_threshold"],
        ):
            best = candidate
    if best is None:
        return {
            "optimal_threshold": 0.5,
            "objective": "out_of_fold_cost_weighted_loss",
            "false_positive_cost": float(false_positive_cost),
            "false_negative_cost": float(false_negative_cost),
            "total_cost": 0.0,
            "average_cost": 0.0,
        }
    return best
