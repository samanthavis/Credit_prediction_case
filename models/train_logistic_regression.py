"""Train and evaluate a logistic regression model on the t+1 credit application dataset.

Hyperparameters are tuned with Optuna, maximizing PR-AUC on the validation split.
The classification threshold is deliberately NOT tuned here: it is selected manually
afterwards by inspecting the precision-recall curve (see `plot_precision_recall_curve`).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("data/processed/log_reg_t1")
TARGET_COLUMN = "target_t+1"
ID_COLUMNS = ["client_nr", "yearmonth"]

# Search space for Optuna. C controls regularization strength (smaller = stronger).
HYPERPARAMETER_SEARCH_SPACE = {
    "C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "penalty": ["l1", "l2"],
}

N_TRIALS = 50
MAX_ITER = 2000

# Number of clients used for the top-k metrics.
K = 90


def load_split(split_name: str) -> pd.DataFrame:
    """Load one split (train/val/test) of the log_reg_t1 dataset."""
    return pd.read_parquet(DATA_DIR / f"{split_name}.parquet")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Every column except target and identifier columns is used as a feature."""
    excluded = set(ID_COLUMNS + [TARGET_COLUMN])
    return [col for col in df.columns if col not in excluded]


def train_logistic_regression_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict,
) -> Pipeline:
    """Fit a scaled logistic regression; scaling is required for regularization to be fair."""

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=MAX_ITER,
                    random_state=42,
                    **params,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_columns], train_df[TARGET_COLUMN])
    return model


def tune_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_columns: list[str],
    n_trials: int = N_TRIALS,
) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Search hyperparameters with Optuna, maximizing validation PR-AUC."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: trial.suggest_categorical(name, values)
            for name, values in HYPERPARAMETER_SEARCH_SPACE.items()
        }

        model = train_logistic_regression_model(train_df, feature_columns, params)
        val_scores = model.predict_proba(val_df[feature_columns])[:, 1]

        return average_precision_score(val_df[TARGET_COLUMN], val_scores)

    # suppress per-trial logging; the summary below is more useful
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = dict(study.best_params)

    print(f"\nBest validation PR-AUC: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for name, value in best_params.items():
        print(f"  {name}: {value}")

    best_model = train_logistic_regression_model(train_df, feature_columns, best_params)

    return best_model, best_params, study.trials_dataframe()


def plot_precision_recall_curve(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float | None = None,
    title: str = "Precision-Recall Curve",
) -> pd.DataFrame:
    """Plot the precision-recall curve and return per-threshold values for manual inspection."""

    y_true = df[TARGET_COLUMN].to_numpy()
    y_scores = model.predict_proba(df[feature_columns])[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label="Precision-Recall curve")

    if threshold is not None:
        # thresholds has one fewer element than precision/recall; find the closest match
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(
            recall[threshold_idx],
            precision[threshold_idx],
            color="red",
            zorder=5,
            label=(
                f"threshold={threshold:.2f}\n"
                f"(precision={precision[threshold_idx]:.2f}, "
                f"recall={recall[threshold_idx]:.2f})"
            ),
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    # the final precision/recall pair has no corresponding threshold, so drop it
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )


def precision_recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    k: int,
) -> tuple[float, float]:
    """Precision/recall computed on only the top-k highest-scored predictions."""

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)

    top_k_idx = np.argsort(y_scores)[::-1][:k]

    true_positives = y_true[top_k_idx].sum()
    total_positives = y_true.sum()

    precision_at_k = true_positives / k if k else 0.0
    recall_at_k = true_positives / total_positives if total_positives else 0.0

    return precision_at_k, recall_at_k


def evaluate_model(
    model: Pipeline,
    df: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    k: int = K,
) -> dict[str, float]:
    """Compute model performance metrics for one dataset split."""

    y_true = df[TARGET_COLUMN].to_numpy()
    y_scores = model.predict_proba(df[feature_columns])[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    precision_at_k, recall_at_k = precision_recall_at_k(y_true, y_scores, k)

    return {
        "pr_auc": average_precision_score(y_true, y_scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        f"precision_at_k (k={k})": precision_at_k,
        f"recall_at_k (k={k})": recall_at_k,
    }


def evaluate_all_splits(
    model: Pipeline,
    feature_columns: list[str],
    threshold: float,
) -> pd.DataFrame:
    """Evaluate the model on train/validation/test with a manually chosen threshold."""

    metrics = {
        split_name: evaluate_model(
            model=model,
            df=load_split(split_name),
            feature_columns=feature_columns,
            threshold=threshold,
        )
        for split_name in ("train", "val", "test")
    }

    return pd.DataFrame(metrics).T


def train_model(n_trials: int = N_TRIALS) -> tuple[Pipeline, list[str], pd.DataFrame]:
    """Load data, tune hyperparameters on the validation split, and return the fitted model."""

    train_df = load_split("train")
    val_df = load_split("val")

    feature_columns = get_feature_columns(train_df)

    print(f"Train observations: {len(train_df):,}")
    print(f"Validation observations: {len(val_df):,}")
    print(f"Number of features: {len(feature_columns)}")

    n_positive = (train_df[TARGET_COLUMN] == 1).sum()
    n_negative = (train_df[TARGET_COLUMN] == 0).sum()
    print("\nClass distribution in training data:")
    print(f"  Negative: {n_negative:,}")
    print(f"  Positive: {n_positive:,}")

    model, _, trials_df = tune_hyperparameters(
        train_df=train_df,
        val_df=val_df,
        feature_columns=feature_columns,
        n_trials=n_trials,
    )

    return model, feature_columns, trials_df


def save_model(
    model: Pipeline,
    feature_columns: list[str],
    threshold: float,
    output_path: Path = Path("models/artifacts/log_reg_t1.joblib"),
) -> Path:
    """Persist the fitted model together with its feature list and chosen threshold."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "threshold": threshold,
            "model_name": "logistic_regression_t1",
        },
        output_path,
    )

    print(f"Saved model to {output_path.resolve()}")
    return output_path


if __name__ == "__main__":
    trained_model, features, _ = train_model()

    # default threshold only for the CLI run; choose it deliberately from the PR curve instead
    print(evaluate_all_splits(trained_model, features, threshold=0.5))
