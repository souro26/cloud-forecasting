from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

LAGS = (1, 2, 3, 6, 12)
ROLLING_WINDOWS = (3, 6, 12)
DEFAULT_HORIZON = 24


def _validate_horizon(horizon: int) -> None:
    if (
        isinstance(horizon, (bool, np.bool_))
        or not isinstance(horizon, (int, np.integer))
        or horizon <= 0
    ):
        raise ValueError("horizon must be a positive integer.")


def _validate_history(
    history: Sequence[float] | np.ndarray | pd.Series,
) -> np.ndarray:
    try:
        values = np.asarray(history, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("history must contain numeric observations.") from exc

    if values.ndim != 1:
        raise ValueError("history must be one-dimensional.")

    if values.size == 0:
        raise ValueError("history must not be empty.")

    if not np.isfinite(values).all():
        raise ValueError("history must contain only finite values.")

    if ((values < 0) | (values > 100)).any():
        raise ValueError("CPU utilization history must be between 0 and 100.")

    return values


def _validate_forecast(
    predictions: Sequence[float] | np.ndarray,
    horizon: int,
    model_name: str,
) -> np.ndarray:
    try:
        values = np.asarray(predictions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{model_name} returned non-numeric predictions.") from exc

    if values.shape != (horizon,):
        raise ValueError(
            f"{model_name} must return exactly {horizon} predictions; "
            f"received shape {values.shape}."
        )

    if not np.isfinite(values).all():
        raise ValueError(f"{model_name} returned non-finite predictions.")

    return values


def _prepare_inputs(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int,
) -> np.ndarray:
    _validate_horizon(horizon)
    return _validate_history(history)


def forecast_persistence(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    values = _prepare_inputs(history, horizon)
    predictions = np.full(horizon, values[-1], dtype=float)
    return _validate_forecast(predictions, horizon, "Persistence")


def forecast_moving_average(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
    window: int = 6,
) -> np.ndarray:
    values = _prepare_inputs(history, horizon)

    if (
        isinstance(window, (bool, np.bool_))
        or not isinstance(window, (int, np.integer))
        or window <= 0
    ):
        raise ValueError("window must be a positive integer.")

    if window > len(values):
        raise ValueError(
            f"window ({window}) cannot exceed history length ({len(values)})."
        )

    level = float(np.mean(values[-window:]))
    predictions = np.full(horizon, level, dtype=float)
    return _validate_forecast(predictions, horizon, "Moving average")


def forecast_ses(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    values = _prepare_inputs(history, horizon)
    model = SimpleExpSmoothing(values, initialization_method="estimated").fit(
        optimized=True
    )
    predictions = model.forecast(horizon)
    return _validate_forecast(
        predictions,
        horizon,
        "Simple Exponential Smoothing",
    )


def forecast_holt(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    values = _prepare_inputs(history, horizon)
    model = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=False,
        initialization_method="estimated",
    ).fit(optimized=True)
    predictions = model.forecast(horizon)
    return _validate_forecast(predictions, horizon, "Holt linear trend")


def forecast_arima(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
    d: int = 0,
) -> np.ndarray:
    values = _prepare_inputs(history, horizon)
    if isinstance(d, (bool, np.bool_)) or d not in (0, 1):
        raise ValueError("d must be either 0 or 1.")

    trend = "c" if d == 0 else "t"
    model = ARIMA(values, order=(1, d, 1), trend=trend).fit()
    predictions = model.forecast(steps=horizon)
    return _validate_forecast(predictions, horizon, f"ARIMA(1, {d}, 1)")


def make_lag_features(
    history: Sequence[float] | np.ndarray | pd.Series,
) -> dict[str, float]:
    values = _validate_history(history)

    if len(values) < max(LAGS):
        raise ValueError(
            f"At least {max(LAGS)} historical observations are required "
            "to construct lag features."
        )

    features: dict[str, float] = {}

    for lag in LAGS:
        features[f"lag_{lag}"] = float(values[-lag])

    for window in ROLLING_WINDOWS:
        recent = values[-window:]
        features[f"mean_{window}"] = float(np.mean(recent))
        features[f"std_{window}"] = float(np.std(recent, ddof=0))

    return features


def make_training_data(
    history: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[pd.DataFrame, np.ndarray]:
    values = _validate_history(history)
    first_target_index = max(LAGS)

    if len(values) <= first_target_index:
        raise ValueError(
            f"At least {first_target_index + 1} observations are required "
            "to construct supervised training data."
        )

    feature_rows: list[dict[str, float]] = []
    targets: list[float] = []

    for index in range(first_target_index, len(values)):
        feature_rows.append(make_lag_features(values[:index]))
        targets.append(float(values[index]))

    X = pd.DataFrame(feature_rows)
    y = np.asarray(targets, dtype=float)

    return X, y


def _build_regression_model(model_type: str) -> Pipeline | RandomForestRegressor:
    if model_type == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))

    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError("model_type must be 'ridge' or 'random_forest'.")


def forecast_lag_regression(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
    model_type: str = "ridge",
) -> np.ndarray:
    values = _prepare_inputs(history, horizon)
    X_train, y_train = make_training_data(values)
    model = _build_regression_model(model_type)
    model.fit(X_train, y_train)
    rolling_history = list(values)
    predictions: list[float] = []
    for _ in range(horizon):
        features = pd.DataFrame([make_lag_features(rolling_history)])
        prediction = float(model.predict(features)[0])

        if not np.isfinite(prediction):
            raise ValueError(
                f"{model_type} generated a non-finite prediction."
            )
        predictions.append(prediction)
        rolling_history.append(prediction)

    model_name = f"{model_type} lag regression"
    return _validate_forecast(predictions, horizon, model_name)


def forecast_ridge(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    return forecast_lag_regression(history, horizon=horizon, model_type="ridge")


def forecast_random_forest(
    history: Sequence[float] | np.ndarray | pd.Series,
    horizon: int = DEFAULT_HORIZON,
) -> np.ndarray:
    return forecast_lag_regression(
        history,
        horizon=horizon,
        model_type="random_forest",
    )

MODEL_REGISTRY: dict[str, Callable[[Sequence[float], int], np.ndarray]] = {
    "Persistence": forecast_persistence,
    "Moving Average (6)": forecast_moving_average,
    "Simple Exponential Smoothing": forecast_ses,
    "Holt Linear Trend": forecast_holt,
    "ARIMA (1,0,1)": lambda history, horizon: forecast_arima(
        history, horizon=horizon, d=0
    ),
    "ARIMA (1,1,1)": lambda history, horizon: forecast_arima(
        history, horizon=horizon, d=1
    ),
    "Ridge Lag Regression": forecast_ridge,
    "Random Forest Lag Regression": forecast_random_forest,
}
