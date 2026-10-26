from __future__ import annotations

import numpy as np
import pandas as pd


CPU_REQUIRED_COLUMNS = {"timestamp","vm_id","min_cpu","avg_cpu","max_cpu",}
METADATA_REQUIRED_COLUMNS = {"vm_id",}
PREDICTION_COLUMNS = ["timestamp","active_vm_count","aggregate_cpu_utilization",]
CPU_VALUE_COLUMNS = ["min_cpu","avg_cpu","max_cpu",]
CPU_BOUNDS = (0.0, 100.0)


def _require_dataframe(data: pd.DataFrame, name: str) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")


def _check_required_columns( data: pd.DataFrame, required_columns: set[str], name: str, ) -> None:
    missing = required_columns.difference(data.columns)
    if missing: raise ValueError( f"{name} is missing required columns: {sorted(missing)}" )


def _check_no_missing_values( data: pd.DataFrame, columns: list[str] | set[str], name: str, ) -> None:
    columns = list(columns)
    missing_counts = data[columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        details = missing_counts.to_dict()
        raise ValueError( f"{name} contains missing values in required columns: {details}" )

def _check_numeric_finite( data: pd.DataFrame, columns: list[str], name: str, ) -> None:
    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"{name}.{column} must be numeric.")

        values = data[column].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            raise ValueError( f"{name}.{column} contains non-finite values." )


def _check_integer_values( data: pd.DataFrame, columns: list[str], name: str, ) -> None:
    for column in columns:
        values = data[column].to_numpy(dtype=float)

        if not np.equal(values, np.floor(values)).all():
            raise ValueError( f"{name}.{column} must contain integer-valued observations." )


def validate_cpu_readings( cpu_readings: pd.DataFrame, *, expected_cadence_seconds: int | None = None, require_complete_grid: bool = False, ) -> None:
    _require_dataframe(cpu_readings, "cpu_readings")
    _check_required_columns( cpu_readings, CPU_REQUIRED_COLUMNS, "cpu_readings", )

    if cpu_readings.empty:
        raise ValueError("cpu_readings must not be empty.")

    _check_no_missing_values( cpu_readings, CPU_REQUIRED_COLUMNS, "cpu_readings", )
    _check_numeric_finite( cpu_readings, ["timestamp", *CPU_VALUE_COLUMNS], "cpu_readings", )
    _check_integer_values( cpu_readings, ["timestamp"], "cpu_readings", )

    if cpu_readings["vm_id"].nunique() == 0:
        raise ValueError("cpu_readings must contain at least one VM.")
    
    for column in CPU_VALUE_COLUMNS:
        if not cpu_readings[column].between(*CPU_BOUNDS).all():
            raise ValueError( f"cpu_readings.{column} must be between " f"{CPU_BOUNDS[0]} and {CPU_BOUNDS[1]}." )

    invalid_order = ( (cpu_readings["min_cpu"] > cpu_readings["avg_cpu"]) | (cpu_readings["avg_cpu"] > cpu_readings["max_cpu"]) )

    if invalid_order.any():
        raise ValueError( "CPU readings must satisfy min_cpu <= avg_cpu <= max_cpu." )

    duplicate_mask = cpu_readings.duplicated( subset=["vm_id", "timestamp"], keep=False, )

    if duplicate_mask.any():
        examples = (
            cpu_readings.loc[duplicate_mask, ["vm_id", "timestamp"]] .head(5) .to_dict("records")
        )
        raise ValueError( "Duplicate (vm_id, timestamp) observations found. " f"Example keys: {examples}" )

    timestamps = np.sort(cpu_readings["timestamp"].unique())

    if expected_cadence_seconds is not None:
        if (
            isinstance(expected_cadence_seconds, bool)
            or not isinstance(expected_cadence_seconds, (int, np.integer))
            or expected_cadence_seconds <= 0
        ):
            raise ValueError( "expected_cadence_seconds must be a positive integer." )

        if len(timestamps) > 1:
            differences = np.diff(timestamps)

            if not np.all(differences == expected_cadence_seconds):
                raise ValueError( "Timestamp cadence does not match " f"{expected_cadence_seconds} seconds." )

    if require_complete_grid:
        timestamp_count = len(timestamps)
        readings_per_vm = cpu_readings.groupby("vm_id")["timestamp"].nunique()

        incomplete_vms = readings_per_vm[readings_per_vm != timestamp_count]

        if not incomplete_vms.empty:
            raise ValueError( f"{len(incomplete_vms)} VMs do not have observations at every timestamp in the input." )


def validate_vm_metadata( vm_metadata: pd.DataFrame, ) -> None:
    _require_dataframe(vm_metadata, "vm_metadata")
    _check_required_columns( vm_metadata, METADATA_REQUIRED_COLUMNS, "vm_metadata", )

    if vm_metadata.empty:
        raise ValueError("vm_metadata must not be empty.")

    _check_no_missing_values( vm_metadata, METADATA_REQUIRED_COLUMNS, "vm_metadata", )

    if vm_metadata["vm_id"].duplicated().any():
        duplicate_ids = (
            vm_metadata.loc[ vm_metadata["vm_id"].duplicated(keep=False), "vm_id", ].drop_duplicates() .head(5) .tolist()
        )

        raise ValueError(
            "vm_metadata must contain one row per vm_id. "
            f"Example duplicate IDs: {duplicate_ids}"
        )


def validate_cpu_metadata_consistency( cpu_readings: pd.DataFrame, vm_metadata: pd.DataFrame, *, require_metadata_for_every_vm: bool = True, allow_metadata_without_readings: bool = True, ) -> None:
    _require_dataframe(cpu_readings, "cpu_readings")
    _require_dataframe(vm_metadata, "vm_metadata")

    _check_required_columns( cpu_readings, {"vm_id"}, "cpu_readings", )
    _check_required_columns( vm_metadata, {"vm_id"}, "vm_metadata", )

    if cpu_readings["vm_id"].isna().any():
        raise ValueError("cpu_readings.vm_id contains missing values.")

    if vm_metadata["vm_id"].isna().any():
        raise ValueError("vm_metadata.vm_id contains missing values.")

    reading_ids = set(cpu_readings["vm_id"].unique())
    metadata_ids = set(vm_metadata["vm_id"].unique())

    readings_without_metadata = reading_ids - metadata_ids
    metadata_without_readings = metadata_ids - reading_ids

    if require_metadata_for_every_vm and readings_without_metadata:
        examples = list(readings_without_metadata)[:5]
        raise ValueError(
            f"{len(readings_without_metadata)} VMs in cpu_readings have "
            f"no metadata. Example IDs: {examples}"
        )

    if not allow_metadata_without_readings and metadata_without_readings:
        examples = list(metadata_without_readings)[:5]
        raise ValueError(
            f"{len(metadata_without_readings)} metadata VMs have no readings. "
            f"Example IDs: {examples}"
        )


def validate_predictions( predictions: pd.DataFrame, *, expected_rows: int = 24, expected_timestamps: list[int] | np.ndarray | pd.Series | None = None, ) -> None:
    _require_dataframe(predictions, "predictions")
    _check_required_columns(
        predictions,
        set(PREDICTION_COLUMNS),
        "predictions",
    )

    if ( isinstance(expected_rows, bool) or not isinstance(expected_rows, (int, np.integer)) or expected_rows <= 0 ):
        raise ValueError("expected_rows must be a positive integer.")
    if len(predictions) != expected_rows: raise ValueError( f"Expected {expected_rows} prediction rows; " f"received {len(predictions)}." )

    _check_no_missing_values( predictions, PREDICTION_COLUMNS, "predictions", )
    _check_numeric_finite( predictions, PREDICTION_COLUMNS, "predictions", )
    _check_integer_values( predictions, ["timestamp", "active_vm_count"], "predictions", )

    if (predictions["active_vm_count"] < 0).any(): raise ValueError("active_vm_count cannot be negative.")
    if not predictions["aggregate_cpu_utilization"].between(0, 100).all(): raise ValueError( "aggregate_cpu_utilization must be between 0 and 100." )
    timestamps = predictions["timestamp"].to_numpy(dtype=float)
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError( "Prediction timestamps must be strictly increasing and unique." )

    if expected_timestamps is not None:
        expected = np.asarray(expected_timestamps)

        if expected.ndim != 1:
            raise ValueError("expected_timestamps must be one-dimensional.")

        if len(expected) != expected_rows:
            raise ValueError( "expected_timestamps must have exactly " f"{expected_rows} entries." )

        if not np.isfinite(expected.astype(float)).all():
            raise ValueError("expected_timestamps must contain finite values.")

        if not np.array_equal(timestamps, expected.astype(float)):
            raise ValueError( "Prediction timestamps do not match expected_timestamps." )


def validate_forecasting_inputs( cpu_readings: pd.DataFrame, vm_metadata: pd.DataFrame | None = None, *, expected_cadence_seconds: int | None = None, require_complete_grid: bool = False, ) -> None:
    validate_cpu_readings( cpu_readings, expected_cadence_seconds=expected_cadence_seconds, require_complete_grid=require_complete_grid, )

    if vm_metadata is not None:
        validate_vm_metadata(vm_metadata)
        validate_cpu_metadata_consistency( cpu_readings, vm_metadata, )
