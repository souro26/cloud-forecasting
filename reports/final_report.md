# Short-Horizon Forecasting of Aggregate Azure VM CPU Utilization

## Executive Summary

This report evaluates a two-hour forecast of aggregate CPU utilization for a fixed population of Azure virtual machines. The input trace contains 10,000 VMs sampled every five minutes for 134 observations. The forecast target is the unweighted arithmetic mean of VM-level average CPU utilization at each timestamp.

The selected model is ARIMA(2, 0, 0), fitted to the first 110 aggregate observations and used to forecast the final 24 timestamps. It was selected on rolling validation, not on the final 24 points. Across seven rolling 24-observation folds, it achieved mean MAE of **0.19483** and mean RMSE of **0.22458** CPU-utilization percentage points. This was a marginal improvement over ARIMA(1, 0, 2) (MAE 0.19500; RMSE 0.22613) and a clearer improvement over persistence (MAE 0.21524; RMSE 0.25136). The small difference between the two ARIMA candidates does not establish decisive superiority.

The rolling results are promising for a short-horizon baseline, but peak behavior is less reassuring: ARIMA(2, 0, 0) had a pooled peak MAE of 0.31367 percentage points across 45 fold-observation instances. In the final 24 observations, which were inspected during notebook analysis, the model forecast a smoother path than the actual series. Its retrospective MAE was 0.12437, but this is **not an unbiased final evaluation** because the holdout was examined. It should not be used to make a new model-selection decision.

The practical conclusion is limited: the model can provide a near-term reference forecast for a stable population, but it does not predict VM count, capture abrupt workload changes reliably, or establish longer-term capacity needs.

## Data Understanding

### Coverage and structure

The data is a curated subset of Microsoft Azure Public Dataset V2. It contains:

- 10,000 anonymized VMs;
- 134 CPU readings per VM, for 1,340,000 VM-level rows;
- timestamps spaced 300 seconds (five minutes) apart;
- a matching metadata row for each VM.

The trace runs from timestamp 0 through 39,900 seconds, covering about 11.1 hours. The VM population was selected deterministically with stratification by VM category and virtual-core bucket. The dataset card reports 6,180 delay-insensitive VMs, 3,091 interactive VMs, and 729 VMs with unknown category.

The CPU input contains timestamp, VM identifier, and minimum, average, and maximum utilization values. The metadata includes identifiers and VM characteristics such as category, virtual-core bucket, and memory bucket. The pipeline reads the compressed files from `data/raw/`.

### Data quality

The dataset card reports that each selected VM has a complete 134-point trace, readings are five minutes apart, duplicate VM/timestamp keys were removed, and the CPU and metadata VM populations match. The pipeline checks these assumptions again at runtime. It validates required columns, finite values, CPU bounds, the ordering `min_cpu <= avg_cpu <= max_cpu`, unique VM/timestamp observations, cadence, complete per-VM timestamp grids, unique metadata IDs, and metadata coverage for VMs with readings.

In the supplied files, there are no missing values in either table, no duplicate `(vm_id, timestamp)` CPU readings, no duplicate metadata VM IDs, and the CPU and metadata VM ID sets match. The pipeline does not impute missing values: validation rejects missing required values and incomplete timestamp grids. There is consequently no imputation step that could alter the series.

The aggregate series contains exactly 134 ordered timestamps. Its active VM count is 10,000 at every timestamp, so the population is stable over this particular trace.

### VM lifetimes and workload segments

The metadata records a creation timestamp of 0 for all 10,000 VMs. The earliest recorded deletion is 40,500 seconds, later than the final CPU timestamp at 39,900 seconds. Thus every VM's recorded lifetime covers this observed trace, and there is no VM arrival or departure visible within the 134-point window. This says nothing about churn outside the sampled window.

To compare segments, each VM's mean and 95th percentile of its 134 `avg_cpu` readings were calculated first, then summarized by metadata category and virtual-core bucket. The table reports the mean of per-VM means, the median per-VM mean, and the 90th percentile of per-VM 95th-percentile CPU values. These are descriptive group summaries; they are not capacity-weighted, and they do not establish that category or core count causes workload differences.

| Segment | VMs | Mean of per-VM means (%) | Median per-VM mean (%) | 90th percentile of per-VM p95 (%) |
| --- | ---: | ---: | ---: | ---: |
| Delay-insensitive | 6,180 | 8.236 | 5.616 | 25.231 |
| Interactive | 3,091 | 11.493 | 5.565 | 52.900 |
| Unknown category | 729 | 10.233 | 3.894 | 52.098 |
| 2 vCPU bucket | 7,534 | 9.725 | 5.998 | 35.704 |
| 4 vCPU bucket | 1,908 | 7.928 | 3.218 | 30.965 |
| 8 vCPU bucket | 486 | 9.700 | 4.988 | 37.443 |
| 24 vCPU bucket | 62 | 11.397 | 8.526 | 33.179 |
| >24 vCPU bucket | 10 | 6.618 | 3.581 | 31.561 |

The gap between group means and medians, especially in the interactive and unknown categories, is consistent with a right-skewed distribution: a subset of VMs has much higher CPU readings than the typical VM. The smallest core buckets have limited sample sizes, so their summaries are particularly uncertain. This variation is one reason the aggregate mean alone should not be treated as a per-VM capacity or peak-risk measure.

### Workload shape

Across all 134 points, aggregate CPU utilization averaged **9.38810%**, with a minimum of **8.98361%** and maximum of **9.91218%**. The standard deviation was 0.19426 percentage points. The first 110 observations averaged 9.41409%; the final 24 averaged 9.26899%.

The overall series is relatively stable in level, but individual five-minute changes are not always small. In the final 24 observations, utilization ranged from 8.98361% to 9.63599%, and the largest single-step change was 0.40088 percentage points. This distinction matters: a low-variance aggregate can still include short-lived movements that a smooth multi-step forecast does not follow.

The aggregate shows short-lag dependence (lag-1 autocorrelation 0.681, falling to 0.108 at lag 24). A linear fit across the trace estimates a small decline of 0.0173 percentage points per hour (R-squared 0.082). ADF and KPSS tests give mixed evidence about stationarity (p-values 0.218 and 0.058, respectively). With only 134 observations and about 11 hours of data, these diagnostics do not establish a stable long-term trend or recurring daily/weekly pattern.

## Forecasting Methodology

### Target definition

At each timestamp, the pipeline calculates:

- **Active VM count:** the number of distinct VMs with a reading at that time.
- **Aggregate CPU utilization:** the unweighted arithmetic mean of the VM-level `avg_cpu` readings at that time.

Each VM receives equal weight, regardless of its vCPU count or capacity. The aggregate is therefore an average utilization percentage, not a measure of total CPU demand or a capacity-weighted utilization.

### Chronological split and model fit

The observations are ordered by timestamp. The first 110 aggregate points form the development window, and the final 24 points represent a two-hour horizon. The chosen model is fitted to the development-window aggregate CPU utilization and forecasts 24 steps ahead.

For the output’s `active_vm_count`, the pipeline carries forward the final count observed in the development window. This is appropriate for the supplied complete, constant-population trace, but it is an assumption rather than a separate VM-count forecast.

### Candidate models and validation design

The notebook compares ARIMA(2, 0, 0), ARIMA(1, 0, 2), the best evaluated random-forest configuration, the best evaluated ridge configuration, and a persistence baseline. The main results below are the seven rolling-origin folds on development data. The fold boundaries are `(train_end, validation_end)`: `(50, 74)`, `(56, 80)`, `(62, 86)`, `(68, 92)`, `(74, 98)`, `(80, 104)`, and `(86, 110)`. For each fold, the model trains on `development_cpu[:train_end]` and forecasts the 24 values in `development_cpu[train_end:validation_end]`. The validation windows overlap because their origins advance by six observations.

For each fold, errors are calculated as forecast minus actual. MAE measures average absolute error, RMSE gives more influence to larger errors, and bias is the mean signed error. Positive bias indicates overforecasting; negative bias indicates underforecasting. For peaks, the notebook calculates `threshold = np.quantile(history, 0.90)` separately for each fold, then selects validation points where `actual >= threshold`. It pools the absolute forecast errors for those selected fold-observation instances and takes their mean. The resulting 45 peak observations are fold-observation instances; timestamps repeated in overlapping validation windows are counted once per fold.

The command-line pipeline itself does not run candidate comparisons or select a model dynamically. It fits the configured ARIMA(2, 0, 0) model.

## Validation and Results

### Rolling validation

Mean errors across the seven rolling folds are shown below. Values are in CPU-utilization percentage points.

| Model | Mean MAE | Mean RMSE | Mean bias | Worst-fold MAE | Mean underforecast rate | Peak MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ARIMA(2, 0, 0) | 0.19483 | 0.22458 | +0.01267 | 0.30164 | 35.71% | 0.31367 |
| ARIMA(1, 0, 2) | 0.19500 | 0.22613 | +0.02656 | 0.30172 | 36.90% | 0.30137 |
| Best random forest | 0.20737 | 0.23757 | +0.02035 | 0.34152 | 39.29% | 0.32699 |
| Best ridge | 0.21253 | 0.24280 | +0.02374 | 0.30838 | 34.52% | 0.32980 |
| Persistence | 0.21524 | 0.25136 | +0.12052 | 0.30103 | 26.79% | 0.22487 |

ARIMA(2, 0, 0) has the lowest mean MAE and RMSE in this comparison. Its advantage over ARIMA(1, 0, 2) is very small: 0.00017 percentage points in MAE and 0.00155 in RMSE. It is more accurate on average than persistence, though persistence has a lower peak MAE and a lower underforecast rate in these folds. These results support choosing ARIMA(2, 0, 0) as the overall rolling-validation candidate, not as the best model for every operating concern.

Compared with persistence, ARIMA(2, 0, 0) reduces mean rolling MAE by **9.48%** and mean rolling RMSE by **10.65%**, calculated as `100 * (persistence_metric - ARIMA_metric) / persistence_metric`.

The mean signed bias for ARIMA(2, 0, 0) is +0.01267, close to zero and slightly positive. Its forecasts were below the actual value on 35.71% of fold observations and above it on the remainder. This average rate does not mean that underforecast risk is absent: behavior varied by fold, and a small number of missed high-utilization points can matter more operationally than average error suggests.

### Peak behavior

Across the seven folds, the peak definition identified 45 fold-observation instances. ARIMA(2, 0, 0) had a pooled peak MAE of 0.31367 percentage points, higher than its mean overall MAE of 0.19483. ARIMA(1, 0, 2) had a slightly lower peak MAE of 0.30137. Persistence had a lower peak MAE of 0.22487, despite weaker overall MAE and RMSE.

This trade-off is important for capacity use. A model that performs best on average may not be the model with the smallest error during high-demand periods. Peak errors should therefore be monitored alongside general accuracy, and the model choice should be revisited if peak protection becomes the primary objective.

### Retrospective holdout results

The following scores were calculated on the final 24 observations after they had been inspected in the notebook. They are included as descriptive retrospective results only, not as an untouched evaluation:

| Model | MAE | RMSE | Mean bias | Underforecast rate | Overforecast rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ARIMA(1, 0, 2) | 0.09781 | 0.12021 | +0.01968 | 37.50% | 62.50% |
| ARIMA(2, 0, 0) | 0.12437 | 0.14938 | +0.08654 | 25.00% | 75.00% |
| Persistence | 0.14678 | 0.18479 | -0.10443 | 79.17% | 20.83% |

ARIMA(1, 0, 2) scored better on this particular inspected window than the selected model. That observation does not override the rolling-validation selection: using the holdout result to switch models would tune the choice against data that has already been examined.

## Error Analysis

The seven-fold rolling results show that ARIMA(2, 0, 0) has low average signed bias but a substantially higher peak MAE than its overall MAE. The errors are not uniform over time. For example, its first two rolling folds had MAEs of 0.30164 and 0.26535 with negative biases, while later folds included positive biases. A mean near zero across folds can conceal these periods of over- and underforecasting.

The 24-point plot illustrates the smoothing issue. The actual series moved up and down, with a maximum single-step change of 0.40088 percentage points. In contrast, the ARIMA(2, 0, 0) forecast rose monotonically across the horizon; its largest single-step forecast change was 0.03450. This smooth path did not track the short-lived peak at 9.63599%: at that timestamp, the model forecast about 9.39129%, roughly 0.24470 percentage points lower.

On this 24-point window, the ARIMA(2, 0, 0) forecast was above actual for 75% of points and had positive mean bias of 0.08654. It nevertheless underpredicted some observations, including the peak. Thus “overforecasting on average” should not be mistaken for protection against every peak.

## Operational Recommendation

Treat ARIMA(2, 0, 0) as a baseline signal for near-term workload monitoring, not as an autonomous scaling trigger. For this dataset it provides a rolling-validation MAE of about 0.195 percentage points, with a small positive average bias, but its average peak error is higher and its multi-step path can smooth out abrupt changes.

For a provisional review signal, add a **0.374 percentage-point guard band** to the point forecast. To reproduce it, collect `actual - forecast` for all 24 points in each of the seven ARIMA(2, 0, 0) validation folds and take the 90th percentile of those 168 pooled fold-observation residuals. The folds overlap, so a timestamp that appears in multiple validation windows contributes more than once. Approximately 89.9% of those pooled residuals are at or below this value. Because this estimate comes from one short trace and overlapping folds, it is a rough empirical margin, not a calibrated prediction interval or a guarantee of 90% future coverage.

Compare `forecast + 0.374` with a utilization limit set by the infrastructure team. If the buffered signal approaches that limit, trigger a human review of headroom, VM mix, and available capacity; do not automatically scale from this aggregate metric. The dataset contains no capacity or service-level information from which to choose a universal utilization limit. A practical process would:

1. monitor forecast residuals as new readings arrive;
2. track peak-period errors and underforecast events separately from overall MAE;
3. revisit the guard band using additional, non-overlapping traces and the cost of under- versus over-provisioning;
4. use actual VM size, available capacity, scaling lead time, and service-level targets when translating utilization into an action;
5. require additional out-of-sample traces before automating scaling decisions.

This project does not calculate required capacity, select a headroom percentage, or model autoscaling actions. Those decisions need operational inputs that are not present in the aggregate CPU series.

## Limitations

The dataset covers approximately 11.1 hours, and the pipeline forecasts only the final two hours of this trace. This limited observation window is insufficient to establish daily or weekly seasonality or to demonstrate generalization across different workload regimes.

Aggregate CPU utilization is calculated as an unweighted arithmetic mean of VM-level utilization percentages. It is not a measure of total CPU demand, physical capacity, or capacity-weighted utilization.

The pipeline assumes a fixed VM population and carries the final development-window active VM count forward across the forecast horizon. It does not forecast VM arrivals, departures, or autoscaling events.

Finally, the selected ARIMA model is a short-horizon statistical baseline, not a complete capacity-planning or autoscaling solution. Operational deployment would require additional information about VM sizes, available capacity, scaling lead times, service-level objectives, and infrastructure costs.

## Reproducibility

With the raw files placed in `data/raw/` and dependencies installed from `requirements.txt`, generate the forecast from the repository root with:

```sh
python -m src.pipeline
```

The command validates the inputs, constructs aggregate targets, fits ARIMA(2, 0, 0) on the first 110 points, validates the 24 forecast rows, and writes `predictions.csv`. Rolling-fold analysis and the retrospective plots are in `notebooks/model_experiments.ipynb`; they are not run by the command-line pipeline.
