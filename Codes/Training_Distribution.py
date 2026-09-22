#!/usr/bin/env python3
"""
Training Distribution workflow.

This script is the Python-source equivalent of the GitHub-ready notebook.
Edit `default_data_dir` in the configuration section to point to the
directory containing the required input data. Derived input and output
paths are constructed relative to that directory.
"""


# Training workload distribution processing
#
# This notebook converts cluster job traces into 24-hour weekday/weekend workload summaries used by the downstream scenario-construction workflow.
#
# Path setup. Edit only default_data_dir in the configuration cell below. Place trace_seren.csv and trace_kalos.csv in that directory. Generated CSV/XLSX summaries are written to <default_data_dir>/outputs/.
#
# The computational procedures are retained as implemented in the original analysis: job durations are allocated to hourly bins, hourly activity is separated into weekday/weekend samples, and Gamma parameters are estimated for the positive hourly observations.


# -----------------------------------------------------------------------------
# Code block 1
# -----------------------------------------------------------------------------
from pathlib import Path

# -----------------------------------------------------------------------------
# User path configuration
# -----------------------------------------------------------------------------
# Replace only this directory with the location where the input trace files are
# stored. All input and output paths in the notebook are derived from this one
# location so no machine-specific absolute paths are required elsewhere.
default_data_dir = Path("Data Path")  # Replace with your local data directory.
output_dir = default_data_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Code block 2
# -----------------------------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import gamma

# Input traces. Both filenames are resolved relative to default_data_dir.
file_paths = {
    'Seren': default_data_dir / 'trace_seren.csv',
    'Kalos': default_data_dir / 'trace_kalos.csv',
}

# Allocate each job to the clock-hour bins that it overlaps. A job contributes
# gpu_num multiplied by the fraction of each hour during which it is active, so
# gpu_time is expressed as fractional GPU-hours within that hourly bin.
def expand_job_to_hours(row):
    start = pd.to_datetime(row['start_time'])
    end = pd.to_datetime(row['end_time'])
    if end < start:
        return []
    hours = []
    t = start.replace(minute=0, second=0, microsecond=0)
    while t <= end:
        next_hour = t + pd.Timedelta(hours=1)
        hour_start = max(t, start)
        hour_end = min(next_hour, end)
        seconds = max((hour_end - hour_start).total_seconds(), 0)
        frac = seconds / 3600
        if frac > 0:
            hours.append({
                'hour': t,
                'date': t.date(),
                'hour_of_day': t.hour,
                'gpu_time': row['gpu_num'] * frac
            })
        t += pd.Timedelta(hours=1)
    return hours

# Convert raw job records into an hourly event table. Invalid reverse-duration
# jobs contribute no rows; weekday/weekend classification is based on each
# resulting calendar date after the job is expanded across hours.
def process_file(df):
    df['gpu_num'] = df['gpu_num'].astype(int)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    rows = []
    for _, job in df.iterrows():
        rows.extend(expand_job_to_hours(job))
    hour_df = pd.DataFrame(rows)
    if hour_df.empty:
        return None
    # Assign weekday: True = weekday, False = weekend
    hour_df['weekday'] = pd.to_datetime(hour_df['date']).map(lambda d: d.weekday() < 5)
    return hour_df

# Estimate Gamma shape (alpha) and scale (beta) for positive hourly activity.
# When a direct fit is not supported by the sample, use the same method-of-
# moments fallback retained from the original workflow.
def get_gamma_params(series):
    positive = series[series > 0]
    if len(positive) < 3 or np.all(positive == positive.iloc[0]):
        mean = series.mean()
        var = series.var()
        if mean > 0 and var > 0:
            alpha = mean ** 2 / var
            beta = var / mean
        else:
            alpha = np.nan
            beta = np.nan
    else:
        fit_alpha, loc, fit_beta = gamma.fit(positive, floc=0)
        alpha = fit_alpha
        beta = fit_beta
    return alpha, beta

# For each hour-of-day, summarize the distribution of observed GPU-hours across
# the selected weekday/weekend subset and retain the number of observations.
def hourly_stats(hour_df, is_weekday):
    sel = (hour_df['weekday'] == is_weekday)
    sub = hour_df[sel]
    rows = []
    for h in range(24):
        vals = sub[sub['hour_of_day'] == h]['gpu_time']
        row = {
            'hour': h,
            'mean': vals.mean(),
            'min': vals.min(),
            'max': vals.max(),
            'q25': vals.quantile(0.25) if not vals.empty else np.nan,
            'q75': vals.quantile(0.75) if not vals.empty else np.nan,
            'request_number': len(vals)
        }
        alpha, beta = get_gamma_params(vals)
        row['alpha'] = alpha
        row['beta'] = beta
        rows.append(row)
    return pd.DataFrame(rows)

for name, path in file_paths.items():
    df = pd.read_csv(path)
    hour_df = process_file(df)
    if hour_df is None:
        print(f"No usable data in {name}")
        continue
    for is_weekday, day_label in [(True, 'weekday'), (False, 'weekend')]:
        df_stats = hourly_stats(hour_df, is_weekday)
        out_path = output_dir / f"{name.lower()}_{day_label}.xlsx"
        df_stats.to_excel(out_path, index=False)
        print(f"Saved: {out_path}")

print("All files generated with aggregated hourly stats and Gamma (alpha, beta) parameters.")


# -----------------------------------------------------------------------------
# Code block 3
# -----------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM cluster trace summarization: hourly stats (ALL jobs, no type split)
Outputs: For Seren/Kalos and weekday/weekend, per-hour .csv with columns:
    hour, mean, min, max, q25, q75, alpha, beta, request_number
"""

import pandas as pd
import numpy as np
from scipy.stats import gamma

# -------------------- File configuration --------------------
# The central default_data_dir is defined once at the top of the notebook.
# Input traces are expected directly under that directory; processed summaries
# are written to its outputs subdirectory.
FILES = {
    'Seren': default_data_dir / 'trace_seren.csv',
    'Kalos': default_data_dir / 'trace_kalos.csv',
}
OUT_CSV = {
    ("Seren", "weekday"): output_dir / "seren_weekday.csv",
    ("Seren", "weekend"): output_dir / "seren_weekend.csv",
    ("Kalos", "weekday"): output_dir / "kalos_weekday.csv",
    ("Kalos", "weekend"): output_dir / "kalos_weekend.csv",
}

# -------------------- Helpers -----------------------
# Parse job boundaries once and remove records that cannot contribute a
# positive-duration workload interval. The original timestamps are otherwise
# left unchanged, including any timezone information already present.
def parse_times(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=False)
    df["end_time"]   = pd.to_datetime(df["end_time"],   errors="coerce", utc=False)
    df = df.dropna(subset=["start_time", "end_time"])
    df = df[df["end_time"] > df["start_time"]]
    return df

# Vectorized alternative to explicit per-job hourly expansion. Partial first
# and last hours are handled directly, while complete interior hours are added
# through a difference-array accumulation for efficiency on large trace files.
def hourly_gpu_series_fast(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized allocation of GPU-hours to hourly bins (all jobs).
    Returns a Series indexed by hourly timestamps with 'gpu_hours'.
    """
    work = df.copy()
    start = work["start_time"]
    end   = work["end_time"]
    gpus  = pd.to_numeric(work["gpu_num"], errors="coerce").fillna(0.0).astype(float)

    # Floor to hour + next hour
    start_floor = start.dt.floor("h")
    end_floor   = end.dt.floor("h")
    next_hour   = start_floor + pd.Timedelta(hours=1)

    tz = start.iloc[0].tz  # may be None
    grid_start = start_floor.min()
    grid_end   = end_floor.max()
    hours      = pd.date_range(start=grid_start, end=grid_end, freq="h", tz=tz)
    n          = len(hours)

    # Indices on the hourly grid
    idx_first     = ((start_floor - grid_start) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    idx_next      = ((next_hour   - grid_start) / pd.Timedelta(hours=1)).astype(int).to_numpy()
    idx_end_floor = ((end_floor   - grid_start) / pd.Timedelta(hours=1)).astype(int).to_numpy()

    # Partial hour lengths
    len_first = (
        np.minimum(next_hour.values.astype("datetime64[ns]"), end.values.astype("datetime64[ns]"))
        - start.values.astype("datetime64[ns]")
    ) / np.timedelta64(1, "h")
    len_first = np.clip(len_first.astype(float), 0.0, 1.0)

    spans_multi = (end_floor.values > start_floor.values)
    len_last = np.where(
        spans_multi,
        (end.values.astype("datetime64[ns]") - end_floor.values.astype("datetime64[ns]")) / np.timedelta64(1, "h"),
        0.0
    )
    len_last = np.clip(len_last.astype(float), 0.0, 1.0)

    # Full hours strictly between next_hour and end_floor
    has_full = (next_hour.values < end_floor.values)
    start_full_idx = np.where(has_full, idx_next, 0)
    end_full_idx   = np.where(has_full, idx_end_floor, 0)

    # Diff array accumulation for full hours
    diff = np.zeros(n + 1, dtype=float)
    np.add.at(diff, start_full_idx[has_full], gpus.values[has_full])
    np.add.at(diff, end_full_idx[has_full],  -gpus.values[has_full])
    full_gpu = np.cumsum(diff)[:-1]

    # Add partial first/last hours
    partial = np.zeros(n, dtype=float)
    np.add.at(partial, idx_first,     gpus.values * len_first)
    np.add.at(partial, idx_end_floor, gpus.values * len_last)

    gpu_hours = full_gpu + partial
    return pd.Series(gpu_hours, index=hours, name="gpu_hours")

def fit_gamma_params(series):
    positive = series[series > 0]
    if len(positive) < 3 or np.all(positive == positive.iloc[0]):
        mean = series.mean()
        var = series.var()
        if mean > 0 and var > 0:
            alpha = mean ** 2 / var
            beta = var / mean
        else:
            alpha = np.nan
            beta = np.nan
    else:
        fit_alpha, loc, fit_beta = gamma.fit(positive, floc=0)
        alpha = fit_alpha
        beta = fit_beta
    return alpha, beta

# Reindex the hourly series to a complete time grid before calculating
# hour-of-day distributions. This retains zero-activity hours rather than
# conditioning the statistics only on hours in which jobs were observed.
def per_hour_stats_by_daytype(s: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    From hourly GPU-hours series s, compute per-hour (0–23) stats across days:
      mean, min, max, q25, q75, alpha, beta, request_number
    separately for Weekday (Mon–Fri) and Weekend (Sat–Sun).
    """
    idx = pd.Index(range(24), name="hour")
    cols = ["mean", "min", "max", "q25", "q75", "alpha", "beta", "request_number"]
    empty = pd.DataFrame({c: [np.nan]*24 for c in cols}, index=idx)

    if s.empty:
        return empty.copy(), empty.copy()

    df = pd.DataFrame({
        "date": pd.to_datetime(s.index).date,
        "hour":  s.index.hour,
        "dow":   s.index.dayofweek,  # Monday=0
        "val":   s.values,
    })

    def agg_one(x: pd.DataFrame) -> pd.DataFrame:
        if x.empty:
            return empty.copy()
        per_day = x.groupby(["date", "hour"])["val"].sum().reset_index()
        rows = []
        for h in range(24):
            vals = per_day[per_day["hour"] == h]["val"]
            mean = vals.mean()
            vmin = vals.min()
            vmax = vals.max()
            q25 = vals.quantile(0.25) if not vals.empty else np.nan
            q75 = vals.quantile(0.75) if not vals.empty else np.nan
            n = len(vals)
            alpha, beta = fit_gamma_params(vals)
            rows.append([mean, vmin, vmax, q25, q75, alpha, beta, n])
        out = pd.DataFrame(rows, columns=cols, index=idx)
        return out

    wk = df[df["dow"] <= 4]
    we = df[df["dow"] >= 5]
    return agg_one(wk), agg_one(we)

# -------------------- Main ------------------------
def main():
    for cluster, path in FILES.items():
        df = pd.read_csv(path)
        df = parse_times(df)
        df["gpu_num"] = pd.to_numeric(df["gpu_num"], errors="coerce").fillna(0.0)
        s = hourly_gpu_series_fast(df)
        wk_df, we_df = per_hour_stats_by_daytype(s)
        # Save one 24-row table for each cluster/day-type combination. These
        # files become candidate training profiles in the scenario workflow.
        wk_df.reset_index().to_csv(OUT_CSV[(cluster, "weekday")], index=False)
        we_df.reset_index().to_csv(OUT_CSV[(cluster, "weekend")], index=False)
        print(f"Written: {OUT_CSV[(cluster, 'weekday')]} and {OUT_CSV[(cluster, 'weekend')]}")

if __name__ == "__main__":
    main()
