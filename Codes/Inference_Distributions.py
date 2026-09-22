#!/usr/bin/env python3
"""
Inference Distributions workflow.

This script is the Python-source equivalent of the GitHub-ready notebook.
Edit `default_data_dir` in the configuration section to point to the
directory containing the required input data. Derived input and output
paths are constructed relative to that directory.
"""


# =============================================================================
# Notebook documentation block 1
# =============================================================================
# Inference workload distribution processing
#
# This notebook converts two inference-trace sources into hourly weekday/weekend distributions for conversation-style and API-style workloads.
#
# Path setup. Edit only default_data_dir below. Place the two BurstGPT_without_fails_*.csv files and the two AzureLLMInferenceTrace_*_1week.csv files in that directory. Generated profile CSVs are written to <default_data_dir>/outputs/.
#
# The original aggregation and fitting logic is retained. The notebook detects source columns, converts request timestamps to day/hour indices, aggregates token activity by day and hour, includes missing hourly bins where specified by the original workflow, and estimates Gamma parameters from the resulting positive samples.


# -----------------------------------------------------------------------------
# Code block 1
# -----------------------------------------------------------------------------
from pathlib import Path

# -----------------------------------------------------------------------------
# User path configuration
# -----------------------------------------------------------------------------
# Replace only this directory. All BurstGPT/Azure inputs and all generated CSVs
# are resolved relative to this location.
default_data_dir = Path("Data Path")  # Replace with your local data directory.
output_dir = default_data_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Code block 2
# -----------------------------------------------------------------------------
# BurstyGPT trace processing: original aggregation variant
import pandas as pd
import numpy as np

FIRSTDAY_OFFSET = 2  # 0..6 = Mon..Sun; 2 => first day is Wednesday
# Both trace fragments are read from the shared data directory.
file1 = default_data_dir / "BurstGPT_without_fails_1.csv"
file2 = default_data_dir / "BurstGPT_without_fails_2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

def normalize(col):
    return col.strip().lower().replace(" ", "").replace("_", "")

# Source releases can use slightly different headers. Resolve timestamp,
# token-count, and workload-type columns using normalized names before applying
# the common aggregation procedure.
def detect_columns(df):
    ts_candidates = {"timestamp","time","ts","requesttime","submissiontime","seconds"}
    token_candidates = {"totaltoken","totaltokens","tokens","token","total_tokens","total_token"}
    logtype_candidates = {"logtype","type","log","category"}
    ts_col = token_col = type_col = None
    for c in df.columns:
        cn = normalize(c)
        if ts_col is None and cn in ts_candidates: ts_col = c
        if token_col is None and cn in token_candidates: token_col = c
        if type_col is None and cn in logtype_candidates: type_col = c
    if ts_col is None:
        for c in df.columns:
            cn = normalize(c)
            if "timestamp" in cn or "time" in cn or "second" in cn: ts_col = c; break
    if token_col is None:
        for c in df.columns:
            cn = normalize(c)
            if "token" in cn: token_col = c; break
    if type_col is None:
        for c in df.columns:
            cn = normalize(c)
            if "logtype" in cn or cn.endswith("type") or "log" in cn: type_col = c; break
    return ts_col, token_col, type_col

ts1, tok1, typ1 = detect_columns(df1)
ts2, tok2, typ2 = detect_columns(df2)

df1 = df1.rename(columns={ts1:"timestamp_sec", tok1:"total_tokens", typ1:"log_type"})
df2 = df2.rename(columns={ts2:"timestamp_sec", tok2:"total_tokens", typ2:"log_type"})

# If the second file restarts its timestamp counter, shift it forward so
# the two fragments form one continuous synthetic timeline without overlap.
max1 = pd.to_numeric(df1["timestamp_sec"], errors="coerce").max()
min2 = pd.to_numeric(df2["timestamp_sec"], errors="coerce").min()
if np.isfinite(max1) and np.isfinite(min2) and min2 <= max1:
    df2["timestamp_sec"] = pd.to_numeric(df2["timestamp_sec"], errors="coerce") + (max1 - min2) + 1

df = pd.concat([df1, df2], ignore_index=True)
df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce")
df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
df = df.dropna(subset=["total_tokens","timestamp_sec"])
df = df[df["total_tokens"] > 0]

def normalize_log_type(x):
    if not isinstance(x, str): return "other"
    lx = x.strip().lower()
    if "conversation" in lx: return "Conversation"
    if "api" in lx: return "API"
    return "other"

df["log_type_norm"] = df["log_type"].apply(normalize_log_type)

# Convert elapsed seconds to day index and hour of day. FIRSTDAY_OFFSET
# anchors the synthetic first day to Wednesday, after which Saturday/Sunday are
# identified from the derived day-of-week index.
SECONDS_PER_DAY = 86400
df["day_idx"] = (df["timestamp_sec"] // SECONDS_PER_DAY).astype(int)
df["hour"] = ((df["timestamp_sec"] % SECONDS_PER_DAY) // 3600).astype(int)
df["dow"] = (df["day_idx"] + FIRSTDAY_OFFSET) % 7
df["is_weekend"] = df["dow"].isin([5,6])  # Sat, Sun

# ---------- helpers ----------
# Estimate Gamma shape/scale from positive samples using the retained
# method-of-moments formulas; nonpositive or insufficient samples return NaN.
def gamma_params_from_samples(arr):
    """Method-of-moments Gamma fit on positive values only."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size < 2: 
        return np.nan, np.nan
    m = arr.mean()
    v = arr.var(ddof=1)
    if m <= 0 or v <= 0:
        return np.nan, np.nan
    alpha = (m*m)/v
    beta  = v/m
    return alpha, beta

def summarize_by_day_hour(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    For a given subset (e.g., Conversation & Weekday):
      1) build per-day, per-hour totals (sum of total_tokens) and counts (requests)
      2) for each hour, compute stats across days on the daily totals
    """
    if df_sub.empty:
        # return 24 NaN rows with correct columns
        return pd.DataFrame({
            "hour": list(range(24)),
            "mean": [np.nan]*24,
            "min": [np.nan]*24,
            "max": [np.nan]*24,
            "q25": [np.nan]*24,
            "q75": [np.nan]*24,
            "alpha": [np.nan]*24,
            "beta": [np.nan]*24,
            "avg_requests_per_hour": [np.nan]*24,
        })

    # 1) Aggregate to day-hour
    agg = (df_sub
           .groupby(["day_idx","hour"])
           .agg(daily_tokens_sum=("total_tokens","sum"),
                request_count=("total_tokens","size"))
           .reset_index())

    all_days = pd.DataFrame({"day_idx": sorted(df_sub["day_idx"].unique())})
    rows = []
    for h in range(24):
        # Take rows for this hour
        ah = agg[agg["hour"] == h][["day_idx","daily_tokens_sum","request_count"]]
        # Ensure we include days with zero activity for this hour
        ah = all_days.merge(ah, on="day_idx", how="left").fillna({"daily_tokens_sum":0.0, "request_count":0})

        totals = ah["daily_tokens_sum"].to_numpy()
        counts = ah["request_count"].to_numpy()

        # 2) Stats across days (include zeros)
        mean = float(np.mean(totals))
        vmin = float(np.min(totals)) if totals.size else np.nan
        vmax = float(np.max(totals)) if totals.size else np.nan
        q25  = float(np.percentile(totals, 25)) if totals.size else np.nan
        q75  = float(np.percentile(totals, 75)) if totals.size else np.nan

        # Gamma on positive daily totals only
        alpha, beta = gamma_params_from_samples(totals)

        # Average request count per hour per day
        avg_requests_per_hour = float(np.mean(counts)) if counts.size else np.nan

        rows.append({
            "hour": h,
            "mean": mean,
            "min": vmin,
            "max": vmax,
            "q25": q25,
            "q75": q75,
            "alpha": alpha,
            "beta": beta,
            "avg_requests_per_hour": avg_requests_per_hour
        })

    return pd.DataFrame(rows)

# ---------- build four tables using the improved aggregation ----------
conversation = df[df["log_type_norm"]=="Conversation"]
api = df[df["log_type_norm"]=="API"]

conv_weekday = summarize_by_day_hour(conversation[~conversation["is_weekend"]])
conv_weekend = summarize_by_day_hour(conversation[conversation["is_weekend"]])
api_weekday  = summarize_by_day_hour(api[~api["is_weekend"]])
api_weekend  = summarize_by_day_hour(api[api["is_weekend"]])

# ---------- Save processed 24-hour profiles ----------
# Four tables are written separately so downstream scenario construction can
# choose workload source and day type explicitly.
conv_weekday.to_csv(output_dir / "BurstyGPT_Conversation_Weekday.csv", index=False)
conv_weekend.to_csv(output_dir / "BurstyGPT_Conversation_Weekend.csv", index=False)
api_weekday.to_csv(output_dir / "BurstyGPT_API_Weekday.csv", index=False)
api_weekend.to_csv(output_dir / "BurstyGPT_API_Weekend.csv", index=False)


# -----------------------------------------------------------------------------
# Code block 3
# -----------------------------------------------------------------------------
# BurstyGPT trace processing: revised aggregation variant
import pandas as pd
import numpy as np

FIRSTDAY_OFFSET = 2  # 0..6 = Mon..Sun; 2 => first day is Wednesday
# Both trace fragments are read from the shared data directory.
file1 = default_data_dir / "BurstGPT_without_fails_1.csv"
file2 = default_data_dir / "BurstGPT_without_fails_2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

def normalize(col):
    return col.strip().lower().replace(" ", "").replace("_", "")

# Source releases can use slightly different headers. Resolve timestamp,
# token-count, and workload-type columns using normalized names before applying
# the common aggregation procedure.
def detect_columns(df):
    ts_candidates = {"timestamp","time","ts","requesttime","submissiontime","seconds"}
    token_candidates = {"totaltoken","totaltokens","tokens","token","total_tokens","total_token"}
    logtype_candidates = {"logtype","type","log","category"}
    ts_col = token_col = type_col = None
    for c in df.columns:
        cn = normalize(c)
        if ts_col is None and cn in ts_candidates: ts_col = c
        if token_col is None and cn in token_candidates: token_col = c
        if type_col is None and cn in logtype_candidates: type_col = c
    if ts_col is None:
        for c in df.columns:
            cn = normalize(c)
            if "timestamp" in cn or "time" in cn or "second" in cn: ts_col = c; break
    if token_col is None:
        for c in df.columns:
            cn = normalize(c)
            if "token" in cn: token_col = c; break
    if type_col is None:
        for c in df.columns:
            cn = normalize(c)
            if "logtype" in cn or cn.endswith("type") or "log" in cn: type_col = c; break
    return ts_col, token_col, type_col

ts1, tok1, typ1 = detect_columns(df1)
ts2, tok2, typ2 = detect_columns(df2)

df1 = df1.rename(columns={ts1:"timestamp_sec", tok1:"total_tokens", typ1:"log_type"})
df2 = df2.rename(columns={ts2:"timestamp_sec", tok2:"total_tokens", typ2:"log_type"})

# If the second file restarts its timestamp counter, shift it forward so
# the two fragments form one continuous synthetic timeline without overlap.
max1 = pd.to_numeric(df1["timestamp_sec"], errors="coerce").max()
min2 = pd.to_numeric(df2["timestamp_sec"], errors="coerce").min()
if np.isfinite(max1) and np.isfinite(min2) and min2 <= max1:
    df2["timestamp_sec"] = pd.to_numeric(df2["timestamp_sec"], errors="coerce") + (max1 - min2) + 1

df = pd.concat([df1, df2], ignore_index=True)
df["total_tokens"] = pd.to_numeric(df["total_tokens"], errors="coerce")
df["timestamp_sec"] = pd.to_numeric(df["timestamp_sec"], errors="coerce")
df = df.dropna(subset=["total_tokens","timestamp_sec"])
df = df[df["total_tokens"] > 0]

def normalize_log_type(x):
    if not isinstance(x, str): return "other"
    lx = x.strip().lower()
    if "conversation" in lx: return "Conversation"
    if "api" in lx: return "API"
    return "other"

df["log_type_norm"] = df["log_type"].apply(normalize_log_type)

# Convert elapsed seconds to day index and hour of day. FIRSTDAY_OFFSET
# anchors the synthetic first day to Wednesday, after which Saturday/Sunday are
# identified from the derived day-of-week index.
SECONDS_PER_DAY = 86400
df["day_idx"] = (df["timestamp_sec"] // SECONDS_PER_DAY).astype(int)
df["hour"] = ((df["timestamp_sec"] % SECONDS_PER_DAY) // 3600).astype(int)
df["dow"] = (df["day_idx"] + FIRSTDAY_OFFSET) % 7
df["is_weekend"] = df["dow"].isin([5,6])  # Sat, Sun

# ---------- robust helpers ----------
# Estimate Gamma shape/scale from positive samples using the retained
# method-of-moments formulas; nonpositive or insufficient samples return NaN.
def gamma_params_from_samples(arr):
    """Method-of-moments Gamma fit on positive values only."""
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]
    if arr.size < 2:
        return np.nan, np.nan
    m = arr.mean()
    v = arr.var(ddof=1)
    if m <= 0 or v <= 0:
        return np.nan, np.nan
    alpha = (m*m)/v
    beta  = v/m
    return alpha, beta

def robust_upper_threshold(x: np.ndarray) -> float:
    """
    Build an upper cutoff for outlier rejection using the tightest of:
      - Tukey fence (Q3 + 1.5*IQR)
      - MAD cap (median + 4 * 1.4826 * MAD)
      - High percentile (p99, or p97.5 if n<40)
    Only defined for n>=3; otherwise returns +inf (no filtering).
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 3:
        return np.inf

    q1, med, q3 = np.percentile(x, [25, 50, 75])
    iqr = max(q3 - q1, 0.0)
    tukey = q3 + 1.5 * iqr

    mad = np.median(np.abs(x - med))
    mad_sigma = 1.4826 * mad  # ~sigma if normal
    mad_cap = med + 4.0 * mad_sigma

    p = 97.5 if n < 40 else 99.0
    hi_pct = np.percentile(x, p)

    # Take the tightest reasonable upper bound (but at least q3)
    candidates = [tukey, mad_cap, hi_pct]
    ub = min(c for c in candidates if np.isfinite(c))
    return max(ub, q3)

def summarize_by_day_hour(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    For a given subset (e.g., Conversation & Weekday):
      1) build per-day, per-hour totals (sum of total_tokens) and counts (requests)
      2) for each hour, compute stats across days on the daily totals
         - 'max' becomes robust (after removing extreme highs)
         - keep 'max_raw' for auditing
    """
    if df_sub.empty:
        return pd.DataFrame({
            "hour": list(range(24)),
            "mean": [np.nan]*24,
            "min": [np.nan]*24,
            "max": [np.nan]*24,                 # robust max
            "q25": [np.nan]*24,
            "q75": [np.nan]*24,
            "alpha": [np.nan]*24,
            "beta": [np.nan]*24,
            "avg_requests_per_hour": [np.nan]*24,
        })

    # 1) Aggregate to day-hour
    agg = (df_sub
           .groupby(["day_idx","hour"])
           .agg(daily_tokens_sum=("total_tokens","sum"),
                request_count=("total_tokens","size"))
           .reset_index())

    all_days = pd.DataFrame({"day_idx": sorted(df_sub["day_idx"].unique())})
    rows = []
    for h in range(24):
        ah = agg[agg["hour"] == h][["day_idx","daily_tokens_sum","request_count"]]
        ah = all_days.merge(ah, on="day_idx", how="left").fillna({"daily_tokens_sum":0.0, "request_count":0})

        totals = ah["daily_tokens_sum"].to_numpy()
        counts = ah["request_count"].to_numpy()

        mean = float(np.mean(totals))
        vmin = float(np.min(totals)) if totals.size else np.nan
        vmax_raw = float(np.max(totals)) if totals.size else np.nan
        q25  = float(np.percentile(totals, 25)) if totals.size else np.nan
        q75  = float(np.percentile(totals, 75)) if totals.size else np.nan

        # robust max
        ub = robust_upper_threshold(totals)
        normals = totals[np.isfinite(totals) & (totals <= ub)]
        if normals.size:
            vmax_robust = float(np.max(normals))
            out_hi = int(np.sum(totals > ub))
        else:
            vmax_robust = vmax_raw  # fallback if everything filtered
            out_hi = int(np.sum(totals > ub))

        alpha, beta = gamma_params_from_samples(totals)
        avg_requests_per_hour = float(np.mean(counts)) if counts.size else np.nan

        rows.append({
            "hour": h,
            "mean": mean,
            "min": vmin,
            "max": vmax_robust,          # <-- robust maximum
            "q25": q25,
            "q75": q75,
            "alpha": alpha,
            "beta": beta,
            "avg_requests_per_hour": avg_requests_per_hour
        })

    return pd.DataFrame(rows)

# ---------- build four tables using the robust aggregation ----------
conversation = df[df["log_type_norm"]=="Conversation"]
api = df[df["log_type_norm"]=="API"]

conv_weekday = summarize_by_day_hour(conversation[~conversation["is_weekend"]])
conv_weekend = summarize_by_day_hour(conversation[conversation["is_weekend"]])
api_weekday  = summarize_by_day_hour(api[~api["is_weekend"]])
api_weekend  = summarize_by_day_hour(api[api["is_weekend"]])

# ---------- Save processed 24-hour profiles ----------
# Four tables are written separately so downstream scenario construction can
# choose workload source and day type explicitly.
conv_weekday.to_csv(output_dir / "BurstyGPT_Conversation_Weekday.csv", index=False)
conv_weekend.to_csv(output_dir / "BurstyGPT_Conversation_Weekend.csv", index=False)
api_weekday.to_csv(output_dir / "BurstyGPT_API_Weekday.csv", index=False)
api_weekend.to_csv(output_dir / "BurstyGPT_API_Weekend.csv", index=False)


# -----------------------------------------------------------------------------
# Code block 4
# -----------------------------------------------------------------------------
# Azure inference-trace processing
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Inputs
# -----------------------------
# The two Azure trace files are resolved from the same central data directory.
BASE = default_data_dir
FILE_CONV = BASE / "AzureLLMInferenceTrace_conv_1week.csv"
FILE_API  = BASE / "AzureLLMInferenceTrace_code_1week.csv"

# -----------------------------
# Config
# -----------------------------
# If you want to treat a specific timezone for weekday/weekend, set tz_local (e.g., "America/New_York").
# By default we use the timestamp's own timezone (likely UTC) to determine weekday/weekend.
tz_local = None  # e.g., "America/New_York" or None to keep given tz

# -----------------------------
# Helpers
# -----------------------------
# Read one trace and convert each request to total token count. Timestamp parsing
# is performed in UTC first; tz_local can optionally convert the calendar used
# for weekday/weekend classification without changing request ordering.
def read_azure_trace(path: Path) -> pd.DataFrame:
    """
    Expected columns:
      - TIMESTAMP (e.g., "2024-05-12 00:00:00.001163+00:00")
      - ContextTokens (int)
      - GeneratedTokens (int)
    We compute: total_tokens = ContextTokens + GeneratedTokens
    """
    df = pd.read_csv(path)
    # Robust column names
    colmap = {c.strip().lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in colmap:
                return colmap[n]
        raise KeyError(f"Column not found (tried {names}) in {list(df.columns)}")

    c_ts  = pick("timestamp")
    c_ctx = pick("contexttokens")
    c_gen = pick("generatedtokens")

    # Parse timestamp with timezone
    ts = pd.to_datetime(df[c_ts], errors="coerce", utc=True)
    if tz_local:
        ts = ts.dt.tz_convert(tz_local)

    df = df.assign(
        TIMESTAMP=ts,
        ContextTokens=pd.to_numeric(df[c_ctx], errors="coerce"),
        GeneratedTokens=pd.to_numeric(df[c_gen], errors="coerce")
    ).dropna(subset=["TIMESTAMP", "ContextTokens", "GeneratedTokens"])

    df["total_tokens"] = df["ContextTokens"] + df["GeneratedTokens"]
    # Keep only positive totals
    df = df[df["total_tokens"] >= 0]
    return df[["TIMESTAMP", "total_tokens"]].copy()

def gamma_params_from_samples(arr: np.ndarray):
    """
    Method-of-moments Gamma fit on positive values only.
    mean = k*theta, var = k*theta^2 => k = mean^2/var, theta = var/mean
    Return (alpha=k, beta=theta). NaN if insufficient or invalid.
    """
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]  # strictly positive for Gamma
    if arr.size < 2:
        return np.nan, np.nan
    m = arr.mean()
    v = arr.var(ddof=1)
    if m <= 0 or v <= 0:
        return np.nan, np.nan
    alpha = (m * m) / v
    beta  = v / m
    return float(alpha), float(beta)

# Aggregate requests to daily hourly totals first, then summarize each hour of
# day across observed days. The day-by-hour grid preserves the original handling
# of hours with no requests before distribution statistics are calculated.
def summarize_by_day_hour(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Given a subset with columns: TIMESTAMP, total_tokens
    Steps:
      1) derive day (date) and hour (0..23)
      2) aggregate to per-day, per-hour totals and request counts
      3) for each hour, compute stats across days on the daily totals:
           mean, min, max, q25, q75 (zeros included for missing hours)
           Gamma alpha/beta fitted on positive daily totals only
         and avg_requests_per_hour = average requests/hour/day
    """
    if df_sub.empty:
        return pd.DataFrame({
            "hour": list(range(24)),
            "mean": [np.nan]*24,
            "min": [np.nan]*24,
            "max": [np.nan]*24,
            "q25": [np.nan]*24,
            "q75": [np.nan]*24,
            "alpha": [np.nan]*24,
            "beta": [np.nan]*24,
            "avg_requests_per_hour": [np.nan]*24,
            "days_count": [0]*24,
        })

    # Derive day (date) and hour
    df_sub = df_sub.copy()
    # Use the timestamp's own tz; .date() drops time & tz into naive date
    df_sub["day"] = df_sub["TIMESTAMP"].dt.date
    df_sub["hour"] = df_sub["TIMESTAMP"].dt.hour

    # Aggregate per day-hour
    agg = (df_sub.groupby(["day", "hour"])
                 .agg(daily_tokens_sum=("total_tokens", "sum"),
                      request_count=("total_tokens", "size"))
                 .reset_index())

    all_days = pd.DataFrame({"day": sorted(df_sub["day"].unique())})

    rows = []
    for h in range(24):
        ah = agg[agg["hour"] == h][["day", "daily_tokens_sum", "request_count"]]
        # Include days with zero activity for this hour
        ah = all_days.merge(ah, on="day", how="left").fillna(
            {"daily_tokens_sum": 0.0, "request_count": 0}
        )

        totals = ah["daily_tokens_sum"].to_numpy()
        counts = ah["request_count"].to_numpy()

        mean = float(np.mean(totals)) if totals.size else np.nan
        vmin = float(np.min(totals))  if totals.size else np.nan
        vmax = float(np.max(totals))  if totals.size else np.nan
        q25  = float(np.percentile(totals, 25)) if totals.size else np.nan
        q75  = float(np.percentile(totals, 75)) if totals.size else np.nan

        alpha, beta = gamma_params_from_samples(totals)  # positive-only inside

        avg_requests = float(np.mean(counts)) if counts.size else np.nan

        rows.append({
            "hour": h,
            "mean": mean,
            "min": vmin,
            "max": vmax,
            "q25": q25,
            "q75": q75,
            "alpha": alpha,
            "beta": beta,
            "avg_requests_per_hour": avg_requests,
            "days_count": int(len(all_days)),
        })

    return pd.DataFrame(rows)

def split_weekday_weekend(df: pd.DataFrame):
    """
    Add weekday/weekend flags based on the timestamp's timezone (or tz_local if set).
    weekend = Saturday (5) or Sunday (6)
    """
    d = df.copy()
    # If tz_local chosen, TIMESTAMP is already converted in read_azure_trace; else keep as is.
    d["dow"] = d["TIMESTAMP"].dt.weekday  # 0=Mon..6=Sun
    d["is_weekend"] = d["dow"].isin([5, 6])
    return d

def process_file(path: Path) -> dict:
    """
    Returns dict with two DataFrames:
      {
        "weekday": hourly stats across days for weekdays,
        "weekend": hourly stats across days for weekends
      }
    """
    raw = read_azure_trace(path)
    raw = split_weekday_weekend(raw)

    weekday_df = summarize_by_day_hour(raw[~raw["is_weekend"]][["TIMESTAMP", "total_tokens"]])
    weekend_df = summarize_by_day_hour(raw[ raw["is_weekend"]][["TIMESTAMP", "total_tokens"]])
    return {"weekday": weekday_df, "weekend": weekend_df}

# -----------------------------
# Run for Conversation and API
# -----------------------------
conv_res = process_file(FILE_CONV)
api_res  = process_file(FILE_API)

# -----------------------------
# Save outputs (CSV)
# -----------------------------
(conv_res["weekday"]
 .to_csv(output_dir / "Azure_Conversation_Weekday.csv", index=False))
(conv_res["weekend"]
 .to_csv(output_dir / "Azure_Conversation_Weekend.csv", index=False))
(api_res["weekday"]
 .to_csv(output_dir / "Azure_API_Weekday.csv", index=False))
(api_res["weekend"]
 .to_csv(output_dir / "Azure_API_Weekend.csv", index=False))

print("Saved:",
      output_dir / "Azure_Conversation_Weekday.csv",
      output_dir / "Azure_Conversation_Weekend.csv",
      output_dir / "Azure_API_Weekday.csv",
      output_dir / "Azure_API_Weekend.csv",
      sep="\n")
