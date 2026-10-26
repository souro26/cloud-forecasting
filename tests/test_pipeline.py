import numpy as np
import pandas as pd

from src.experiments import run_experiments, summarize_experiments
from src.models import forecast_persistence
from src.targets import build_aggregate_targets
from src.validation import validate_forecasting_inputs, validate_predictions


def test_pipeline_from_raw_readings_through_validated_forecast(
    cpu_readings, vm_metadata
):
    validate_forecasting_inputs(
        cpu_readings,
        vm_metadata,
        expected_cadence_seconds=300,
        require_complete_grid=True,
    )
    aggregate = build_aggregate_targets(cpu_readings)
    assert len(aggregate) == 12

    development = aggregate.iloc[:8]
    experiments = run_experiments(
        development["aggregate_cpu_utilization"].to_numpy(),
        peak_threshold=30,
        models={"Persistence": forecast_persistence},
        folds=[(4, 6), (6, 8)],
        horizon=2,
    )
    summary = summarize_experiments(experiments)
    assert summary.loc[0, "model"] == "Persistence"

    holdout = aggregate.iloc[8:]
    forecast = forecast_persistence(
        development["aggregate_cpu_utilization"].to_numpy(),
        horizon=len(holdout),
    )
    predictions = pd.DataFrame(
        {
            "timestamp": holdout["timestamp"].to_numpy(),
            "active_vm_count": holdout["active_vm_count"].to_numpy(),
            "aggregate_cpu_utilization": forecast,
        }
    )
    validate_predictions(
        predictions,
        expected_rows=len(holdout),
        expected_timestamps=holdout["timestamp"].to_numpy(),
    )
    assert np.isfinite(predictions["aggregate_cpu_utilization"]).all()
