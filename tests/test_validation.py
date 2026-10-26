import numpy as np
import pandas as pd
import pytest

from src.validation import (
    validate_cpu_metadata_consistency,
    validate_cpu_readings,
    validate_forecasting_inputs,
    validate_predictions,
    validate_vm_metadata,
)


def test_valid_cpu_readings_and_metadata_pass_full_validation(
    cpu_readings, vm_metadata
):
    validate_forecasting_inputs(
        cpu_readings,
        vm_metadata,
        expected_cadence_seconds=300,
        require_complete_grid=True,
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("avg_cpu", np.nan, "missing values"),
        ("avg_cpu", np.inf, "non-finite"),
        ("avg_cpu", 101.0, "between 0.0 and 100.0"),
        ("min_cpu", 90.0, "min_cpu <= avg_cpu"),
        ("timestamp", 1.5, "integer-valued"),
    ],
)
def test_cpu_readings_reject_invalid_values(cpu_readings, column, value, message):
    invalid = cpu_readings.copy()
    if column == "timestamp":
        invalid["timestamp"] = invalid["timestamp"].astype(float)
    invalid.loc[invalid.index[0], column] = value

    with pytest.raises(ValueError, match=message):
        validate_cpu_readings(invalid)


def test_cpu_readings_reject_duplicate_vm_timestamp(cpu_readings):
    invalid = pd.concat([cpu_readings, cpu_readings.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        validate_cpu_readings(invalid)


def test_cpu_readings_validate_cadence_and_complete_grid(cpu_readings):
    validate_cpu_readings(cpu_readings, expected_cadence_seconds=300)

    with pytest.raises(ValueError, match="cadence"):
        validate_cpu_readings(cpu_readings, expected_cadence_seconds=60)

    missing_observation = cpu_readings.drop(index=cpu_readings.index[0])
    with pytest.raises(ValueError, match="do not have observations"):
        validate_cpu_readings(missing_observation, require_complete_grid=True)


@pytest.mark.parametrize("cadence", [0, -1, 1.5, True])
def test_cpu_readings_reject_invalid_expected_cadence(cpu_readings, cadence):
    with pytest.raises(ValueError, match="positive integer"):
        validate_cpu_readings(
            cpu_readings, expected_cadence_seconds=cadence
        )


def test_metadata_validation_rejects_empty_and_duplicate_ids():
    with pytest.raises(ValueError, match="must not be empty"):
        validate_vm_metadata(pd.DataFrame(columns=["vm_id"]))

    with pytest.raises(ValueError, match="one row per vm_id"):
        validate_vm_metadata(pd.DataFrame({"vm_id": [1, 1]}))


def test_cpu_metadata_consistency_options(cpu_readings, vm_metadata):
    incomplete_metadata = pd.DataFrame({"vm_id": [1, 2]})
    with pytest.raises(ValueError, match="no metadata"):
        validate_cpu_metadata_consistency(cpu_readings, incomplete_metadata)

    extra_metadata = pd.DataFrame({"vm_id": [1, 2, 3, 4]})
    validate_cpu_metadata_consistency(cpu_readings, extra_metadata)
    with pytest.raises(ValueError, match="no readings"):
        validate_cpu_metadata_consistency(
            cpu_readings,
            extra_metadata,
            allow_metadata_without_readings=False,
        )

    validate_cpu_metadata_consistency(cpu_readings, vm_metadata)


def test_prediction_validation_accepts_expected_holdout_shape():
    predictions = pd.DataFrame(
        {
            "timestamp": [1200, 1500, 1800],
            "active_vm_count": [3, 3, 2],
            "aggregate_cpu_utilization": [12.5, 13.0, 11.75],
        }
    )

    validate_predictions(
        predictions,
        expected_rows=3,
        expected_timestamps=[1200, 1500, 1800],
    )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("active_vm_count", -1, "cannot be negative"),
        ("aggregate_cpu_utilization", 101.0, "between 0 and 100"),
        ("timestamp", np.inf, "non-finite"),
        ("timestamp", 1.5, "integer-valued"),
    ],
)
def test_prediction_validation_rejects_invalid_values(column, value, message):
    predictions = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0],
            "active_vm_count": [2, 2],
            "aggregate_cpu_utilization": [10.0, 20.0],
        }
    )
    predictions.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        validate_predictions(predictions, expected_rows=2)


def test_prediction_validation_rejects_wrong_rows_order_and_timestamps():
    predictions = pd.DataFrame(
        {
            "timestamp": [2, 1],
            "active_vm_count": [1, 1],
            "aggregate_cpu_utilization": [10.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_predictions(predictions, expected_rows=2)

    predictions["timestamp"] = [1, 2]
    with pytest.raises(ValueError, match="do not match"):
        validate_predictions(
            predictions,
            expected_rows=2,
            expected_timestamps=[3, 4],
        )

    with pytest.raises(ValueError, match="Expected 3 prediction rows"):
        validate_predictions(predictions, expected_rows=3)
