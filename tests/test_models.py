import numpy as np
import pytest

from src.models import (
    MODEL_REGISTRY,
    forecast_arima,
    forecast_moving_average,
    forecast_persistence,
    make_lag_features,
    make_training_data,
)


def test_persistence_and_moving_average_forecasts():
    history = [10, 20, 30, 40]

    np.testing.assert_array_equal(forecast_persistence(history, 3), [40, 40, 40])
    np.testing.assert_array_equal(
        forecast_moving_average(history, horizon=2, window=2), [35, 35]
    )


def test_lag_features_and_supervised_training_rows():
    history = np.arange(1, 16, dtype=float)
    features = make_lag_features(history)

    assert features == {
        "lag_1": 15.0,
        "lag_2": 14.0,
        "lag_3": 13.0,
        "lag_6": 10.0,
        "lag_12": 4.0,
        "mean_3": 14.0,
        "std_3": pytest.approx(np.std([13.0, 14.0, 15.0])),
        "mean_6": 12.5,
        "std_6": pytest.approx(np.std([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])),
        "mean_12": 9.5,
        "std_12": pytest.approx(np.std(np.arange(4.0, 16.0))),
    }

    X, y = make_training_data(history)
    assert X.shape == (3, 11)
    assert y.tolist() == [13.0, 14.0, 15.0]
    assert X.iloc[0]["lag_1"] == 12.0


@pytest.mark.parametrize(
    ("forecast", "history", "horizon"),
    [
        (forecast_persistence, [], 2),
        (forecast_persistence, [[1, 2]], 2),
        (forecast_persistence, [1, np.nan], 2),
        (forecast_persistence, [-1, 2], 2),
        (forecast_persistence, [1, 101], 2),
        (forecast_persistence, [1, 2], 0),
        (forecast_persistence, [1, 2], True),
        (forecast_moving_average, [1, 2], 2),
    ],
)
def test_forecasters_reject_invalid_history_or_configuration(
    forecast, history, horizon
):
    with pytest.raises(ValueError):
        if forecast is forecast_moving_average:
            forecast(history, horizon=horizon, window=3)
        else:
            forecast(history, horizon=horizon)


def test_regression_feature_generation_requires_sufficient_history():
    with pytest.raises(ValueError, match="12 historical observations"):
        make_lag_features(np.arange(11, dtype=float))

    with pytest.raises(ValueError, match="At least 13 observations"):
        make_training_data(np.arange(12, dtype=float))


def test_arima_rejects_unsupported_differencing_order(cpu_series):
    with pytest.raises(ValueError, match="d must be either 0 or 1"):
        forecast_arima(cpu_series, horizon=2, d=2)


@pytest.mark.parametrize("model_name", list(MODEL_REGISTRY))
def test_registered_models_return_finite_forecasts(model_name, cpu_series):
    forecast = MODEL_REGISTRY[model_name](cpu_series, 3)

    assert forecast.shape == (3,)
    assert np.isfinite(forecast).all()
