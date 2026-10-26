import numpy as np
import pandas as pd
import pytest

from src.experiments import calculate_metrics, run_experiments, summarize_experiments
from src.models import forecast_persistence


def test_calculate_metrics_including_peak_subset():
    metrics = calculate_metrics(
        y_true=[0, 10, 20],
        y_pred=[1, 8, 23],
        peak_threshold=10,
    )

    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(14 / 3))
    assert metrics["bias"] == pytest.approx(2 / 3)
    assert metrics["peak_mae"] == pytest.approx(2.5)
    assert metrics["peak_observations"] == 2


def test_calculate_metrics_returns_nan_when_no_peaks():
    metrics = calculate_metrics([1, 2], [2, 3], peak_threshold=10)

    assert np.isnan(metrics["peak_mae"])
    assert metrics["peak_observations"] == 0


@pytest.mark.parametrize(
    ("actual", "predicted", "threshold"),
    [
        ([[1, 2]], [1, 2], 1),
        ([1], [1, 2], 1),
        ([1, np.nan], [1, 2], 1),
        ([1, 2], [1, 2], np.inf),
    ],
)
def test_calculate_metrics_rejects_invalid_inputs(actual, predicted, threshold):
    with pytest.raises(ValueError):
        calculate_metrics(actual, predicted, threshold)


def test_run_experiments_runs_each_model_on_each_fold():
    series = np.arange(1, 13, dtype=float)
    folds = [(4, 6), (6, 8)]
    models = {
        "persistence": forecast_persistence,
        "constant": lambda history, horizon: np.full(horizon, 2.0),
    }

    results = run_experiments(
        series,
        peak_threshold=7,
        models=models,
        folds=folds,
        horizon=2,
    )

    assert results.shape == (4, 7)
    assert results.groupby("model")["fold"].nunique().to_dict() == {
        "constant": 2,
        "persistence": 2,
    }
    assert set(results.columns) == {
        "model",
        "fold",
        "mae",
        "rmse",
        "bias",
        "peak_mae",
        "peak_observations",
    }


def test_run_experiments_rejects_invalid_model_outputs_and_folds():
    series = np.arange(1, 12, dtype=float)
    bad_length = {"bad": lambda history, horizon: np.zeros(horizon - 1)}
    with pytest.raises(ValueError, match="expected 2 predictions"):
        run_experiments(
            series,
            peak_threshold=5,
            models=bad_length,
            folds=[(4, 6)],
            horizon=2,
        )

    non_finite = {"bad": lambda history, horizon: np.full(horizon, np.nan)}
    with pytest.raises(ValueError, match="non-finite"):
        run_experiments(
            series,
            peak_threshold=5,
            models=non_finite,
            folds=[(4, 6)],
            horizon=2,
        )

    with pytest.raises(ValueError, match="validation points"):
        run_experiments(
            series,
            peak_threshold=5,
            models={"persistence": forecast_persistence},
            folds=[(4, 7)],
            horizon=2,
        )


def test_summarize_experiments_orders_models_and_weights_peak_error():
    results = pd.DataFrame(
        [
            {
                "model": "second",
                "fold": 1,
                "mae": 3.0,
                "rmse": 4.0,
                "bias": 1.0,
                "peak_mae": 2.0,
                "peak_observations": 1,
            },
            {
                "model": "first",
                "fold": 1,
                "mae": 1.0,
                "rmse": 2.0,
                "bias": -1.0,
                "peak_mae": 4.0,
                "peak_observations": 1,
            },
            {
                "model": "first",
                "fold": 2,
                "mae": 2.0,
                "rmse": 3.0,
                "bias": 0.0,
                "peak_mae": 10.0,
                "peak_observations": 3,
            },
        ]
    )

    summary = summarize_experiments(results)

    assert summary["model"].tolist() == ["first", "second"]
    first = summary.iloc[0]
    assert first["mean_mae"] == pytest.approx(1.5)
    assert first["mean_rmse"] == pytest.approx(2.5)
    assert first["mean_bias"] == pytest.approx(-0.5)
    assert first["peak_observations"] == 4
    assert first["folds_evaluated"] == 2
    assert first["peak_mae"] == pytest.approx(8.5)


def test_summarize_experiments_rejects_empty_or_incomplete_results():
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_experiments(pd.DataFrame({"model": ["one"]}))

    empty = pd.DataFrame(
        columns=[
            "model",
            "fold",
            "mae",
            "rmse",
            "bias",
            "peak_mae",
            "peak_observations",
        ]
    )
    with pytest.raises(ValueError, match="empty"):
        summarize_experiments(empty)
