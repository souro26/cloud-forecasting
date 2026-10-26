import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def cpu_readings():
    rows = []
    for timestamp in range(0, 3_600, 300):
        for vm_id in (1, 2, 3):
            avg_cpu = float((timestamp // 300 + vm_id * 7) % 70)
            rows.append(
                {
                    "timestamp": timestamp,
                    "vm_id": vm_id,
                    "min_cpu": max(0.0, avg_cpu - 2),
                    "avg_cpu": avg_cpu,
                    "max_cpu": min(100.0, avg_cpu + 3),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def vm_metadata():
    return pd.DataFrame({"vm_id": [1, 2, 3]})


@pytest.fixture
def cpu_series():
    time = np.arange(40, dtype=float)
    return 25 + 0.35 * time + 4 * np.sin(time / 2)
