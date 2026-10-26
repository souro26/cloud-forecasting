import pandas as pd

TARGET_COLUMNS = [
    "timestamp",
    "active_vm_count",
    "aggregate_cpu_utilization",
]


def build_aggregate_targets(cpu_readings: pd.DataFrame) -> pd.DataFrame:
    """Aggregate VM readings into one utilization target per timestamp."""
    if not isinstance(cpu_readings, pd.DataFrame):
        raise TypeError("cpu_readings must be a pandas DataFrame.")

    required_columns = {"timestamp", "vm_id", "avg_cpu"}
    missing_columns = required_columns.difference(cpu_readings.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    if cpu_readings.empty:
        raise ValueError("cpu_readings must not be empty.")

    required_data = cpu_readings[["timestamp", "vm_id", "avg_cpu"]]

    if required_data.isna().any().any():
        raise ValueError(
            "timestamp, vm_id, and avg_cpu must not contain missing values."
        )

    if not pd.api.types.is_numeric_dtype(cpu_readings["timestamp"]):
        raise ValueError("timestamp must be numeric.")

    if not pd.api.types.is_numeric_dtype(cpu_readings["avg_cpu"]):
        raise ValueError("avg_cpu must be numeric.")

    if not pd.Series(cpu_readings["timestamp"]).map(
        lambda value: pd.notna(value) and float(value).is_integer()
    ).all():
        raise ValueError("timestamp must contain integer-valued timestamps.")

    avg_cpu = cpu_readings["avg_cpu"]

    finite_avg_cpu = pd.Series(avg_cpu).map(
        lambda value: pd.notna(value) and abs(value) != float("inf")
    )
    if not finite_avg_cpu.all():
        raise ValueError("avg_cpu must contain finite values.")

    if not avg_cpu.between(0, 100).all():
        raise ValueError("avg_cpu values must be between 0 and 100.")

    duplicate_keys = cpu_readings.duplicated(subset=["vm_id", "timestamp"])

    if duplicate_keys.any():
        raise ValueError(
            "Duplicate (vm_id, timestamp) observations were found."
        )

    aggregate = (
        cpu_readings.groupby("timestamp", as_index=False)
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    aggregate["active_vm_count"] = aggregate["active_vm_count"].astype("int64")

    return aggregate[TARGET_COLUMNS]
