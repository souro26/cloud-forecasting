from pathlib import Path

import pandas as pd
import pytest

from src.data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_FILES = (
    RAW_DATA_DIR / "vm_cpu_readings.csv.gz",
    RAW_DATA_DIR / "vm_metadata.csv.gz",
)


@pytest.fixture(scope="module")
def loaded_data():
    if not all(path.exists() for path in RAW_FILES):
        pytest.skip("The optional raw Azure dataset is not present.")
    return load_data(RAW_DATA_DIR)


def test_load_data_returns_dataframes(loaded_data):
    cpu_readings, vm_metadata = loaded_data
    assert isinstance(cpu_readings, pd.DataFrame)
    assert isinstance(vm_metadata, pd.DataFrame)


def test_load_data_returns_expected_columns(loaded_data):
    cpu_readings, vm_metadata = loaded_data
    assert {
        "timestamp",
        "vm_id",
        "min_cpu",
        "max_cpu",
        "avg_cpu",
    }.issubset(cpu_readings.columns)

    assert {
        "vm_id",
        "subscription_id",
        "deployment_id",
        "timestamp_vm_created",
        "timestamp_vm_deleted",
        "max_cpu",
        "avg_cpu",
        "p95_max_cpu",
        "vm_category",
        "virtual_core_bucket",
        "memory_gb_bucket",
    }.issubset(vm_metadata.columns)


def test_bundled_dataset_has_documented_shape_and_vm_population(loaded_data):
    cpu_readings, vm_metadata = loaded_data

    assert len(cpu_readings) == 1_340_000
    assert cpu_readings["vm_id"].nunique() == 10_000
    assert len(vm_metadata) == 10_000
    assert not cpu_readings.duplicated(["vm_id", "timestamp"]).any()
    assert set(cpu_readings["vm_id"]) == set(vm_metadata["vm_id"])