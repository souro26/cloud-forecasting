from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

if __package__:
    from .data import load_data
    from .targets import build_aggregate_targets
    from .validation import validate_forecasting_inputs, validate_predictions
else:
    from data import load_data
    from targets import build_aggregate_targets
    from validation import validate_forecasting_inputs, validate_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "predictions.csv"

EXPECTED_CADENCE_SECONDS = 300
DEVELOPMENT_SIZE = 110
FORECAST_HORIZON = 24
EXPECTED_OBSERVATIONS = DEVELOPMENT_SIZE + FORECAST_HORIZON
ARIMA_ORDER = (2, 0, 0)


def run_pipeline(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build, validate, and write the 24-step aggregate CPU forecast."""
    cpu_readings, vm_metadata = load_data(data_dir)
    validate_forecasting_inputs(
        cpu_readings,
        vm_metadata,
        expected_cadence_seconds=EXPECTED_CADENCE_SECONDS,
        require_complete_grid=True,
    )

    aggregate = build_aggregate_targets(cpu_readings)
    if len(aggregate) != EXPECTED_OBSERVATIONS:
        raise ValueError(
            f"Expected {EXPECTED_OBSERVATIONS} aggregate observations "
            f"({DEVELOPMENT_SIZE} development and {FORECAST_HORIZON} forecast); "
            f"received {len(aggregate)}."
        )
    timestamp_differences = aggregate["timestamp"].diff().dropna()
    if not timestamp_differences.eq(EXPECTED_CADENCE_SECONDS).all():
        raise ValueError(
            "Aggregate target timestamps must be spaced exactly "
            f"{EXPECTED_CADENCE_SECONDS} seconds apart."
        )

    development = aggregate.iloc[:DEVELOPMENT_SIZE]
    holdout = aggregate.iloc[DEVELOPMENT_SIZE:]
    active_vm_count_forecast = int(development["active_vm_count"].iloc[-1])
    fitted_model = ARIMA(
        development["aggregate_cpu_utilization"].to_numpy(),
        order=ARIMA_ORDER,
    ).fit()
    forecast = fitted_model.forecast(steps=FORECAST_HORIZON)

    predictions = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"].to_numpy(),
            "active_vm_count": [active_vm_count_forecast] * FORECAST_HORIZON,
            "aggregate_cpu_utilization": forecast,
        }
    )
    validate_predictions(
        predictions,
        expected_rows=FORECAST_HORIZON,
        expected_timestamps=holdout["timestamp"].to_numpy(),
    )

    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    print(f"Wrote {len(predictions)} validated forecasts to {output_file}")
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate validated aggregate CPU forecasts from raw VM data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing vm_cpu_readings.csv.gz and vm_metadata.csv.gz.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="CSV path for the validated 24-timestamp forecast.",
    )
    args = parser.parse_args()

    print(f"Model: ARIMA{ARIMA_ORDER}")
    print(f"Training observations: {DEVELOPMENT_SIZE}")
    print(f"Forecast horizon: {FORECAST_HORIZON}")
    run_pipeline(args.data_dir, args.output)


if __name__ == "__main__":
    main()
