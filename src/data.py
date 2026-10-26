import pandas as pd
from pathlib import Path

def load_data(file_path : str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads data into a pandas dataframe."""
    data_path = Path(file_path)

    cpu_readings = pd.read_csv(data_path / "vm_cpu_readings.csv.gz")
    vm_metadata = pd.read_csv(data_path / "vm_metadata.csv.gz")

    return cpu_readings, vm_metadata
