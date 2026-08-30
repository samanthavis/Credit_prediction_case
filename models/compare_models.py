"""Compare saved models on the held-out test set."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

ARTIFACTS_DIR = Path("models/artifacts")
TARGET_COLUMN = "target_t+1"
MONTH_COLUMN = "yearmonth"

# Each saved model was trained on its own preprocessed dataset variant.
MODEL_DATA_DIRS = {
    "xgboost_t1": Path("data/processed/xgboost_t1"),
    "log_reg_t1": Path("data/processed/log_reg_t1"),
}

K = 90


def load_artifact(model_key: str) -> dict:
    """Load a saved model bundle (model, feature_columns, threshold)."""
    return joblib.load(ARTIFACTS_DIR / f"{model_key}.joblib")


def load_test_split(model_key: str) -> pd.DataFrame:
    """Load the test split belonging to a model's dataset variant."""
    return pd.read_parquet(MODEL_DATA_DIRS[model_key] / "test.parquet")


def precision_recall_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> tuple[float, float]:
    """Precision/recall computed on only the top-k highest-scored predictions."""

    top_k_idx = np.argsort(y_scores)[::-1][:k]

    true_positives = y_true[top_k_idx].sum()
    total_positives = y_true.sum()

    precision_at_k = true_positives / k if k else 0.0
    recall_at_k = true_positives / total_positives if total_positives else 0.0

    return precision_at_k, recall_at_k


def precision_recall_per_month(
    months: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Precision and recall computed separately for each month in the test set."""

    per_month = pd.DataFrame(
        {MONTH_COLUMN: months.to_numpy(), "y_true": y_true, "y_pred": y_pred}
    )

    rows = [
        {
            MONTH_COLUMN: month,
            "precision": precision_score(group["y_true"], group["y_pred"], zero_division=0),
            "recall": recall_score(group["y_true"], group["y_pred"], zero_division=0),
            "n_positives": int(group["y_true"].sum()),
            "n_predicted_positive": int(group["y_pred"].sum()),
        }
        for month, group in per_month.groupby(MONTH_COLUMN)
    ]

    return pd.DataFrame(rows).sort_values(MONTH_COLUMN).reset_index(drop=True)


def score_model(model_key: str, k: int = K) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate one saved model on its test split at its own stored threshold."""

    artifact = load_artifact(model_key)
    test_df = load_test_split(model_key)

    y_true = test_df[TARGET_COLUMN].to_numpy()
    y_scores = artifact["model"].predict_proba(test_df[artifact["feature_columns"]])[:, 1]
    y_pred = (y_scores >= artifact["threshold"]).astype(int)

    precision_at_k, recall_at_k = precision_recall_at_k(y_true, y_scores, k)

    # averaging per month avoids months with many observations dominating the score
    monthly = precision_recall_per_month(test_df[MONTH_COLUMN], y_true, y_pred)
    mean_precision = monthly["precision"].mean()
    mean_recall = monthly["recall"].mean()

    metrics = {
        "threshold": artifact["threshold"],
        "pr_auc": average_precision_score(y_true, y_scores),
        "mean_monthly_precision": mean_precision,
        "mean_monthly_recall": mean_recall,
        "mean_monthly_f1": (
            2 * mean_precision * mean_recall / (mean_precision + mean_recall)
            if (mean_precision + mean_recall)
            else 0.0
        ),
        "overall_precision": precision_score(y_true, y_pred, zero_division=0),
        "overall_recall": recall_score(y_true, y_pred, zero_division=0),
        "overall_f1": f1_score(y_true, y_pred, zero_division=0),
        f"precision_at_k (k={k})": precision_at_k,
        f"recall_at_k (k={k})": recall_at_k,
        "positive_rate": y_true.mean(),
    }

    return metrics, y_true, y_scores


def monthly_breakdown(model_key: str) -> pd.DataFrame:
    """Per-month precision/recall on the test set for a single model."""

    artifact = load_artifact(model_key)
    test_df = load_test_split(model_key)

    y_true = test_df[TARGET_COLUMN].to_numpy()
    y_scores = artifact["model"].predict_proba(test_df[artifact["feature_columns"]])[:, 1]
    y_pred = (y_scores >= artifact["threshold"]).astype(int)

    return precision_recall_per_month(test_df[MONTH_COLUMN], y_true, y_pred)


def compare_models(model_keys: list[str], k: int = K) -> pd.DataFrame:
    """Build a test-set metrics table for all given models."""

    return pd.DataFrame({key: score_model(key, k)[0] for key in model_keys}).T


def plot_test_precision_recall_curves(model_keys: list[str]) -> None:
    """Overlay the test-set precision-recall curves of all given models."""

    fig, ax = plt.subplots(figsize=(7, 5))

    for model_key in model_keys:
        metrics, y_true, y_scores = score_model(model_key)
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ax.plot(recall, precision, label=f"{model_key} (PR-AUC={metrics['pr_auc']:.3f})")

    # a random model scores roughly the positive rate, so it anchors how much signal was learned
    _, y_true, _ = score_model(model_keys[0])
    ax.axhline(
        y_true.mean(),
        color="grey",
        linestyle="--",
        label=f"random baseline ({y_true.mean():.3f})",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Test-set Precision-Recall Curves")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
