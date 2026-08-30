"""SHAP-based explainability for the saved credit application models."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from compare_models import TARGET_COLUMN, load_artifact, load_test_split


def _transform_features(model, features: pd.DataFrame) -> pd.DataFrame:
    """Apply a pipeline's preprocessing steps, since SHAP explains the final estimator."""

    if isinstance(model, Pipeline):
        transformed = model[:-1].transform(features)
        return pd.DataFrame(transformed, columns=features.columns, index=features.index)

    return features


def compute_shap_values(
    model_key: str,
    sample_size: int | None = 2000,
    random_state: int = 42,
) -> tuple[shap.Explanation, pd.DataFrame]:
    """Compute SHAP values for a saved model on (a sample of) its test split."""

    artifact = load_artifact(model_key)
    test_df = load_test_split(model_key)

    features = test_df[artifact["feature_columns"]]
    if sample_size is not None and sample_size < len(features):
        features = features.sample(sample_size, random_state=random_state)

    model = artifact["model"]
    estimator = model[-1] if isinstance(model, Pipeline) else model
    transformed_features = _transform_features(model, features)

    # both tree and linear models have exact explainers, so no background sampling is needed
    explainer = shap.Explainer(estimator, transformed_features)
    shap_values = explainer(transformed_features)

    return shap_values, features


def plot_shap_summary(model_key: str, max_display: int = 20, **kwargs) -> shap.Explanation:
    """Beeswarm plot showing each feature's SHAP value distribution."""

    shap_values, _ = compute_shap_values(model_key, **kwargs)

    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP summary - {model_key}")
    plt.tight_layout()
    plt.show()

    return shap_values


def plot_shap_importance(model_key: str, max_display: int = 20, **kwargs) -> pd.DataFrame:
    """Bar plot of mean absolute SHAP value per feature, returned as a ranked table."""

    shap_values, _ = compute_shap_values(model_key, **kwargs)

    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP feature importance - {model_key}")
    plt.tight_layout()
    plt.show()

    importance = (
        pd.DataFrame(
            {
                "feature": shap_values.feature_names,
                "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    return importance


def plot_shap_waterfall(
    model_key: str,
    observation_index: int = 0,
    max_display: int = 15,
    **kwargs,
) -> None:
    """Explain a single prediction: how each feature pushed it away from the base value."""

    shap_values, _ = compute_shap_values(model_key, **kwargs)

    shap.plots.waterfall(shap_values[observation_index], max_display=max_display, show=False)
    plt.title(f"SHAP explanation for one prediction - {model_key}")
    plt.tight_layout()
    plt.show()


def compare_shap_importance(model_keys: list[str], top_n: int = 20, **kwargs) -> pd.DataFrame:
    """Side-by-side mean absolute SHAP importance for several models."""

    importances = {}
    for model_key in model_keys:
        shap_values, _ = compute_shap_values(model_key, **kwargs)
        importances[model_key] = pd.Series(
            np.abs(shap_values.values).mean(axis=0),
            index=shap_values.feature_names,
        )

    comparison = pd.DataFrame(importances)
    comparison["mean_rank"] = comparison.rank(ascending=False).mean(axis=1)

    return comparison.sort_values("mean_rank").head(top_n)
