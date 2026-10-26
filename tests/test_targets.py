import pandas as pd
import pytest

from src.targets import TARGET_COLUMNS, build_aggregate_targets


def test_build_aggregate_targets_groups_counts_and_sorts(cpu_readings):
    shuffled = cpu_readings.sample(frac=1, random_state=7)

    targets = build_aggregate_targets(shuffled)

    assert targets.columns.tolist() == TARGET_COLUMNS
    assert targets["timestamp"].tolist() == sorted(cpu_readings["timestamp"].unique())
    assert targets["active_vm_count"].tolist() == [3] * 12
    expected = cpu_readings.groupby("timestamp")["avg_cpu"].mean().to_numpy()
    assert targets["aggregate_cpu_utilization"].to_numpy() == pytest.approx(expected)
    assert targets["active_vm_count"].dtype == "int64"


def test_aggregate_counts_unique_vms_and_averages_observations():
    readings = pd.DataFrame(
        {
            "timestamp": [10, 10, 20],
            "vm_id": [4, 8, 4],
            "avg_cpu": [10.0, 30.0, 50.0],
        }
    )

    targets = build_aggregate_targets(readings)

    assert targets.to_dict("records") == [
        {
            "timestamp": 10,
            "active_vm_count": 2,
            "aggregate_cpu_utilization": 20.0,
        },
        {
            "timestamp": 20,
            "active_vm_count": 1,
            "aggregate_cpu_utilization": 50.0,
        },
    ]


@pytest.mark.parametrize(
    ("readings", "message"),
    [
        (None, "pandas DataFrame"),
        (pd.DataFrame(), "Missing required columns"),
        (
            pd.DataFrame(columns=["timestamp", "vm_id", "avg_cpu"]),
            "must not be empty",
        ),
        (
            pd.DataFrame(
                {"timestamp": [1], "vm_id": [1], "avg_cpu": [None]}
            ),
            "must not contain missing values",
        ),
        (
            pd.DataFrame(
                {"timestamp": ["1"], "vm_id": [1], "avg_cpu": [10.0]}
            ),
            "timestamp must be numeric",
        ),
        (
            pd.DataFrame(
                {"timestamp": [1.5], "vm_id": [1], "avg_cpu": [10.0]}
            ),
            "integer-valued timestamps",
        ),
        (
            pd.DataFrame(
                {"timestamp": [1], "vm_id": [1], "avg_cpu": [float("inf")]}
            ),
            "finite values",
        ),
        (
            pd.DataFrame(
                {"timestamp": [1], "vm_id": [1], "avg_cpu": [101.0]}
            ),
            "between 0 and 100",
        ),
        (
            pd.DataFrame(
                {
                    "timestamp": [1, 1],
                    "vm_id": [1, 1],
                    "avg_cpu": [10.0, 20.0],
                }
            ),
            "Duplicate",
        ),
    ],
)
def test_build_aggregate_targets_rejects_invalid_input(readings, message):
    error = TypeError if readings is None else ValueError
    with pytest.raises(error, match=message):
        build_aggregate_targets(readings)
