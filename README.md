# Cloud Workload Forecasting

This project forecasts aggregate CPU utilization for a set of Azure virtual machines. The main workflow reads the VM traces, checks them, builds one CPU-utilization series, and forecasts its final two hours with ARIMA(2, 0, 0). Jupyter notebooks hold the separate data review and model comparison work.

## The data

The input is a curated subset of Microsoft Azure Public Dataset V2. It contains 10,000 VMs, each with 134 CPU readings at five-minute intervals. The raw files are not included in this repository. Put both files here:

```text
data/raw/
|-- vm_cpu_readings.csv.gz
`-- vm_metadata.csv.gz
```

The loader expects those exact filenames.

## Repository layout

```text
src/
|-- data.py          # Read the compressed CSV files
|-- targets.py       # Build aggregate CPU targets
|-- validation.py    # Check inputs and predictions
|-- models.py        # Forecasting models
|-- experiments.py   # Rolling-fold evaluation utilities
`-- pipeline.py      # Generate the ARIMA forecast
notebooks/
|-- data_audit.ipynb
|-- eda_workbook.ipynb
`-- model_experiments.ipynb
tests/               # Pipeline and component tests
reports/
`-- final_report.md
data/raw/            # Add the raw input files here
predictions.csv      # Created by the pipeline
.gitignore
requirements.txt
LICENSE
```

## Set up the environment

The pinned dependencies have been run with Python 3.14.3. NumPy 2.5.4 in `requirements.txt` requires Python 3.12 or newer; this project has only been verified with Python 3.14.3.

From the repository root, create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate it with:

```sh
source .venv/bin/activate
```

## How the data is checked and aggregated

Before building targets, the pipeline checks that the required reading and metadata columns are present, readings are finite and between 0 and 100, and each reading satisfies `min_cpu <= avg_cpu <= max_cpu`. It also checks for duplicate VM/timestamp pairs, five-minute spacing, complete timestamp grids, unique metadata IDs, and metadata for every VM with readings.

The supplied files contain no missing values or duplicate readings. The pipeline does not impute: it rejects missing required values or incomplete per-VM timestamp grids.

For each timestamp, `active_vm_count` is the number of distinct VMs. `aggregate_cpu_utilization` is the **unweighted arithmetic mean of the VMs' `avg_cpu` readings**. Every VM counts equally, regardless of its vCPU count or capacity; this is not a capacity-weighted measure of total CPU consumption.

The pipeline expects 134 aggregate observations with timestamps spaced exactly 300 seconds apart.

## Forecasting and evaluation

The pipeline fits ARIMA(2, 0, 0) to the first 110 aggregate observations, then forecasts the remaining 24 timestamps. Those timestamps come from the input data. The VM population is complete and constant in the supplied dataset, so the pipeline carries the last development-window `active_vm_count` forward for all 24 forecasts; it does not fit a separate VM-count model.

Model comparisons are in `notebooks/model_experiments.ipynb`. On rolling validation folds from the development data, the mean errors were:

| Model | Mean rolling MAE | Mean rolling RMSE |
| --- | ---: | ---: |
| ARIMA(2, 0, 0) | 0.19483 | 0.22458 |
| ARIMA(1, 0, 2) | 0.19500 | 0.22613 |
| Persistence baseline | 0.21524 | 0.25136 |

Errors are in CPU-utilization percentage points. ARIMA(2, 0, 0) performed slightly better than the alternatives in these folds, but its difference from ARIMA(1, 0, 2) is small. The command-line pipeline uses ARIMA(2, 0, 0) directly; it does not rerun the comparisons or select a model.

Relative to persistence, ARIMA(2, 0, 0) reduces mean rolling MAE by 9.48% and mean rolling RMSE by 10.65%. The EDA notebook also compares VM CPU distributions by category and vCPU bucket and checks recorded VM lifetimes against the observed trace; all 10,000 VMs span this trace, so there is no recorded churn within the observed window.

The final 24 observations have also been inspected in the notebook. Any scores calculated on them are retrospective, not an unbiased evaluation on untouched data, and should not be used to select a model.

For an illustrative human-review signal, the report derives a 0.374 percentage-point guard band from the 90th percentile of rolling-validation `actual - forecast` errors. The fold windows overlap, so this is a rough estimate from one short trace, not a calibrated prediction interval. Compare the buffered forecast with a utilization limit defined by the infrastructure team; do not use it as an automatic scaling trigger.

## Explore the notebooks

The data audit and exploratory analysis—including VM lifetime and workload-segment summaries—are in `notebooks/data_audit.ipynb` and `notebooks/eda_workbook.ipynb`. Model comparisons are in `notebooks/model_experiments.ipynb`. To open them, start Jupyter from the repository root:

```sh
python -m jupyter lab
```

Open a notebook in `notebooks/` and run its cells. The notebooks read the files from `data/raw/`.

## Generate predictions

After installing the dependencies and placing the raw files in `data/raw/`, run this from the repository root:

```sh
python -m src.pipeline
```

The command writes `predictions.csv` to the repository root, replacing it if it already exists. It prints the model order, training-window size, forecast horizon, and resolved output path.

You can choose different input and output paths:

```sh
python -m src.pipeline --data-dir data/raw --output output/predictions.csv
```

## Prediction file

The output contains one row per forecast timestamp, for 24 rows total:

```text
predictions.csv
|-- timestamp
|-- active_vm_count
`-- aggregate_cpu_utilization
```

`timestamp` is measured in seconds from the start of the trace. `active_vm_count` is carried forward from the development window, and `aggregate_cpu_utilization` is the ARIMA forecast in percent.

Before writing the file, `validate_predictions()` checks that the required columns are present, the row count is correct, and the required values are numeric, finite, and not missing. It also checks that timestamps and VM counts are integer-valued, VM counts are nonnegative, CPU utilization is between 0 and 100, and timestamps are strictly increasing and unique. The pipeline additionally checks that the timestamps match the expected final 24 input timestamps. Extra columns are allowed.

## Limitations

The trace covers about 11.1 hours, and the forecast covers its final two hours. That is not enough data to establish daily or weekly seasonality or show that the model generalizes to other workload patterns.

Aggregate CPU utilization is an unweighted mean of VM utilization percentages. It does not measure total CPU demand, physical capacity, or capacity-weighted utilization.

The forecast assumes the VM population stays fixed: the last development-window VM count is carried forward. VM arrivals, departures, and autoscaling are not modeled.

ARIMA(2, 0, 0) is a short-horizon statistical baseline, not a complete capacity-planning or autoscaling solution. Operational decisions would also need VM sizes, available capacity, scaling lead times, service-level objectives, and infrastructure costs.
