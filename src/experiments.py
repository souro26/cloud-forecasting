import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

if __package__:
    from .models import MODEL_REGISTRY
else:
    from models import MODEL_REGISTRY


HORIZON = 24

FOLDS = [
    (50, 74),
    (56, 80),
    (62, 86),
    (68, 92),
    (74, 98),
    (80, 104),
    (86, 110),
]


def calculate_metrics(y_true, y_pred, peak_threshold):
    """Calculate forecast error metrics for a validation window."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional.")

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true and y_pred must contain finite values.")

    if not np.isfinite(peak_threshold):
        raise ValueError("peak_threshold must be finite.")

    peak_mask = y_true >= peak_threshold

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "bias": np.mean(y_pred - y_true),
        "peak_mae": (
            mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])
            if peak_mask.any()
            else np.nan
        ),
        "peak_observations": int(peak_mask.sum()),
    }


def run_experiments(
    series,
    peak_threshold,
    models=None,
    folds=None,
    horizon=HORIZON,
):
    """Evaluate forecasting models on rolling-origin validation folds."""
    series = np.asarray(series, dtype=float)
    models = MODEL_REGISTRY if models is None else models
    folds = FOLDS if folds is None else folds

    if series.ndim != 1 or not np.isfinite(series).all():
        raise ValueError("series must be a finite, one-dimensional array.")

    if not np.isfinite(peak_threshold):
        raise ValueError("peak_threshold must be finite.")

    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise ValueError("horizon must be a positive integer.")

    if not models:
        raise ValueError("At least one forecasting model is required.")

    results = []

    for fold_number, (train_end, validation_end) in enumerate(
        folds, start=1
    ):
        if validation_end - train_end != horizon:
            raise ValueError(
                f"Fold {fold_number} must have {horizon} validation points."
            )

        if train_end <= 0 or validation_end > len(series):
            raise ValueError(
                f"Fold {fold_number} has invalid boundaries."
            )

        history = series[:train_end]
        actual = series[train_end:validation_end]

        for model_name, forecast in models.items():
            predictions = np.asarray(
                forecast(history, horizon),
                dtype=float,
            )

            if predictions.shape != (horizon,):
                raise ValueError(
                    f"{model_name}, fold {fold_number}: expected "
                    f"{horizon} predictions, got shape "
                    f"{predictions.shape}."
                )

            if not np.isfinite(predictions).all():
                raise ValueError(
                    f"{model_name}, fold {fold_number}: "
                    "non-finite predictions."
                )

            results.append(
                {
                    "model": model_name,
                    "fold": fold_number,
                    **calculate_metrics(
                        actual,
                        predictions,
                        peak_threshold,
                    ),
                }
            )

    return pd.DataFrame(results)


def summarize_experiments(results):
    """Aggregate metrics across folds for each model."""
    required_columns = {
        "model",
        "fold",
        "mae",
        "rmse",
        "bias",
        "peak_mae",
        "peak_observations",
    }

    missing_columns = required_columns.difference(results.columns)

    if missing_columns:
        raise ValueError(
            f"Results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if results.empty:
        raise ValueError("Cannot summarize an empty results DataFrame.")

    summary = (
        results.groupby("model", sort=False)
        .agg(
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_bias=("bias", "mean"),
            peak_observations=("peak_observations", "sum"),
            folds_evaluated=("fold", "nunique"),
        )
        .reset_index()
    )

    # Weight peak MAE by the number of peak observations in each fold.
    peak_results = results.loc[
        results["peak_observations"] > 0
    ].copy()

    if not peak_results.empty:
        peak_results["weighted_peak_error"] = (
            peak_results["peak_mae"]
            * peak_results["peak_observations"]
        )

        peak_errors = (
            peak_results.groupby("model")
            .agg(
                total_peak_error=("weighted_peak_error", "sum"),
                total_peak_observations=("peak_observations", "sum"),
            )
        )

        summary = summary.merge(
            peak_errors,
            left_on="model",
            right_index=True,
            how="left",
        )

        summary["peak_mae"] = (
            summary["total_peak_error"]
            / summary["total_peak_observations"]
        )

        summary = summary.drop(
            columns=[
                "total_peak_error",
                "total_peak_observations",
            ]
        )
    else:
        summary["peak_mae"] = np.nan

    return (
        summary.sort_values("mean_mae")
        .reset_index(drop=True)
    )