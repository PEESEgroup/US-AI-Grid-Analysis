#!/usr/bin/env python3
"""
AI Scenarios Generation workflow.

This script is the Python-source equivalent of the GitHub-ready notebook.
Edit `default_data_dir` in the configuration section to point to the
directory containing the required input data. Derived input and output
paths are constructed relative to that directory.
"""


# =============================================================================
# Notebook documentation block 1
# =============================================================================
# Workload scenario construction and representative-profile selection
#
# This notebook combines processed training, API, and conversation distributions into the scenario bank, saves component-resolved utilization profiles, and selects representative weekday/weekend scenarios using the retained clustering workflow.
#
# Path setup. Edit only default_data_dir below. Place all processed source CSVs used by the scenario builder directly in that directory. Generated CSV, Excel, and figure outputs are written to <default_data_dir>/outputs/.
#
# The numerical procedures, ratio combinations, utilization factors, clustering features, scenario selection, and plotting logic are unchanged; the GitHub version centralizes file paths and documents the calculation flow.


# -----------------------------------------------------------------------------
# Code block 1
# -----------------------------------------------------------------------------
from pathlib import Path

# -----------------------------------------------------------------------------
# User path configuration
# -----------------------------------------------------------------------------
# Replace only this directory. All source data, intermediate products, workbooks,
# and figures are resolved relative to this location.
default_data_dir = Path("Data Path")  # Replace with your local data directory.
source_data_dir = default_data_dir
output_dir = default_data_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Code block 2
# -----------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build 40 utilization profiles (5 ratio mixes × 8 options) for both weekday and weekend.

Updates:
- Weekend normalization now uses the 24-hour peak of the "max" column from the
  *corresponding weekday file* (same category & subtype), instead of its own file.
- Row titles are short ("T:A:C = ...") and placed in figure coordinates to avoid overlap.
- Column titles show chosen sources per column (T/A/C short names).

Input and output paths are derived from the central configuration cell.
"""

import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Config
# -----------------------------
# All processed source-distribution CSVs are stored directly in the user-selected
# data directory. Generated products are written separately to outputs/.
SOURCE_DATA_DIR = source_data_dir
OUTPUT_DIR = output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Workload-composition cases. Values are relative shares for training, API,
# and conversation activity before category-specific utilization multipliers are
# applied; the five tuples span the central and directional sensitivity cases.
# Ratios (Training : API : Conversation), in this order
RATIO_COMBOS = [
    (3.0, 3.5, 3.5),
    (3.0, 6.0, 1.0),
    (3.0, 1.0, 6.0),
    (1.0, 4.5, 4.5),
    (5.0, 2.5, 2.5),
]

# Category multipliers retain the assumed relative average utilization of the
# three workload classes when their normalized source shapes are combined.
# Utilization multipliers by category
UTIL_FACTORS = {
    "training": 1.0,
    "api": 0.6,
    "conversation": 0.6,
}


# -----------------------------
# Helpers
# -----------------------------

# Infer workload category, source subtype, and day type from the source
# filename. These labels determine which source profiles may be combined in each
# scenario option while keeping weekday and weekend records paired consistently.
def parse_meta(path: Path):
    """Infer category (training/api/conversation), subtype, and daytype (weekday/weekend) from filename."""
    name = path.stem.lower()

    # Daytype
    if "weekday" in name or "weekdays" in name:
        daytype = "weekday"
    elif "weekend" in name or "weekends" in name:
        daytype = "weekend"
    else:
        daytype = "weekday"  # fallback if not denoted

    # Category & subtype
    if any(k in name for k in ["seren", "kalos", "train", "training"]):
        category = "training"
        subtype = "seren" if "seren" in name else ("kalos" if "kalos" in name else "training")
    elif "api" in name:
        category = "api"
        subtype = "api1" if re.search(r"api.*(a|1)", name) else ("api2" if re.search(r"api.*(b|2)", name) else "api")
    elif any(k in name for k in ["conv", "conversation", "chat"]):
        category = "conversation"
        subtype = "conv1" if re.search(r"(conv|conversation).*?(a|1)", name) else ("conv2" if re.search(r"(conv|conversation).*?(b|2)", name) else "conversation")
    else:
        # Try parent folder
        parent = path.parent.name.lower()
        if any(k in parent for k in ["seren", "kalos", "train", "training"]):
            category, subtype = "training", parent
        elif "api" in parent:
            category, subtype = "api", parent
        elif any(k in parent for k in ["conv", "conversation", "chat"]):
            category, subtype = "conversation", parent
        else:
            category, subtype = "unknown", "unknown"

    return category, subtype, daytype

def get_mean_max_columns(df: pd.DataFrame):
    """Find 'mean' and 'max' columns case-insensitively."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    mean_col = lower_map.get("mean")
    max_col  = lower_map.get("max")

    # Fallbacks
    if mean_col is None:
        candidates = [c for c in df.columns if c.strip().lower() == "mean"]
        mean_col = candidates[0] if candidates else None
    if max_col is None:
        candidates = [c for c in df.columns if c.strip().lower() == "max"]
        max_col = candidates[0] if candidates else None

    if mean_col is None or max_col is None:
        raise ValueError("Could not find 'mean' and/or 'max' columns.")
    return mean_col, max_col

def short_name_from_filename(filename: str, category: str) -> str:
    """
    Map filename to short display name:
      - training: 'Kalos' or 'Seren' (default to stem)
      - api/conv: 'Azure' or 'BurstyGPT' (default to stem)
    """
    fname = filename.lower()
    if category == "training":
        if "kalos" in fname:
            return "Kalos"
        if "seren" in fname:
            return "Seren"
        return Path(filename).stem
    else:
        if "azure" in fname:
            return "Azure"
        if "burst" in fname or "bursty" in fname or "gpt" in fname:
            return "BurstyGPT"
        return Path(filename).stem

def load_raw_profile(csv_path: Path):
    """Load first 24 rows, return raw arrays and meta; no normalization here."""
    df = pd.read_csv(csv_path)
    mean_col, max_col = get_mean_max_columns(df)
    df24 = df.iloc[:24].copy().reset_index(drop=True)
    means = pd.to_numeric(df24[mean_col], errors="coerce").values
    maxes = pd.to_numeric(df24[max_col], errors="coerce").values
    category, subtype, daytype = parse_meta(csv_path)
    return {
        "file": str(csv_path),
        "filename": csv_path.name,
        "category": category,
        "subtype": subtype,
        "daytype": daytype,
        "means": means,
        "maxes": maxes,
        "own_denom": np.nanmax(maxes) if len(maxes) else np.nan
    }

def normalize_with_weekday_reference(raw_rec, weekday_denoms):
    """
    For WEEKEND:
      use denom = weekday_denoms[(category, subtype)] if available,
      else fallback to own_denom.
    For WEEKDAY:
      use own_denom (and also populate weekday_denoms).
    Then apply utilization factor by category.
    """
    category = raw_rec["category"]
    subtype = raw_rec["subtype"]
    daytype = raw_rec["daytype"]

    if daytype == "weekday":
        denom = raw_rec["own_denom"]
    else:  # weekend
        denom = weekday_denoms.get((category, subtype), raw_rec["own_denom"])

    if denom is None or not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid normalization denominator for {raw_rec['filename']}: {denom}")

    base = np.clip(raw_rec["means"] / denom, 0.0, 1.0)
    factor = UTIL_FACTORS.get(category)
    if factor is None:
        raise ValueError(f"Unrecognized category in {raw_rec['filename']}: {category}")
    util = np.clip(base * factor, 0.0, 1.0)

    return {
        "file": raw_rec["file"],
        "filename": raw_rec["filename"],
        "category": category,
        "subtype": subtype,
        "daytype": daytype,
        "hour": np.arange(24),
        "util_profile": util,
    }

def combine_profiles(t_prof, a_prof, c_prof, r_t, r_a, r_c):
    """Weighted average by the given ratios."""
    w = r_t + r_a + r_c
    return (r_t * t_prof + r_a * a_prof + r_c * c_prof) / w

def option_index_triplet(opt_idx: int):
    """
    Map column index 0..7 to (ti, ai, ci) following product(range(2), range(2), range(2)):
      0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1),
      4: (1,0,0), 5: (1,0,1), 6: (1,1,0), 7: (1,1,1)
    """
    mapping = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    return mapping[opt_idx]

def add_row_labels_no_overlap(fig, axes, row_texts, xpad=0.01, fontsize=12):
    """
    Place one label per row in figure coordinates, centered vertically
    at each row's rightmost axis midpoint. This avoids overlap.
    """
    nrows, ncols = axes.shape
    assert len(row_texts) == nrows
    for r in range(nrows):
        right_ax = axes[r, ncols - 1]
        x0, y0, w, h = right_ax.get_position().bounds  # figure fraction coords
        y_mid = y0 + h / 2.0
        x_text = x0 + w + xpad
        fig.text(
            x_text, y_mid, row_texts[r],
            ha='left', va='center', fontsize=fontsize
        )

def make_40_profiles_for_daytype(records, daytype, ratio_combos, output_dir):
    """
    From per-file records, pick 2 training + 2 api + 2 conversation for the given daytype,
    build 5 combos × 8 options = 40 combined profiles.
    Save CSV and a 5x8 figure with:
      - Column headers: T: <Kalos/Seren> | A: <Azure/BurstyGPT> | C: <Azure/BurstyGPT>
      - Row headers (right side): T:A:C = <ratio> (no overlap)
    """
    # Filter candidates for this daytype and category
    train = sorted([r for r in records if r["daytype"] == daytype and r["category"] == "training"], key=lambda x: x["filename"])
    api   = sorted([r for r in records if r["daytype"] == daytype and r["category"] == "api"],       key=lambda x: x["filename"])
    conv  = sorted([r for r in records if r["daytype"] == daytype and r["category"] == "conversation"], key=lambda x: x["filename"])

    # Expect >= 2 in each category
    if len(train) < 2 or len(api) < 2 or len(conv) < 2:
        raise RuntimeError(
            f"Need >=2 files each for training/api/conversation on {daytype}. "
            f"Have: training={len(train)}, api={len(api)}, conversation={len(conv)}"
        )

    # Keep only 2 per category (deterministic order)
    train = train[:2]
    api = api[:2]
    conv = conv[:2]

    # Friendly names for headers per candidate index
    T_names = [short_name_from_filename(t["filename"], "training") for t in train]
    A_names = [short_name_from_filename(a["filename"], "api") for a in api]
    C_names = [short_name_from_filename(c["filename"], "conversation") for c in conv]

    # Precompute the 8 column headers based on (ti, ai, ci)
    col_headers = []
    for col in range(8):
        ti, ai, ci = option_index_triplet(col)
        header = f"T: {T_names[ti]} | A: {A_names[ai]} | C: {C_names[ci]}"
        col_headers.append(header)

    out_rows = []
    all_profiles = []   # for plotting per subplot
    row_ratio_texts = []  # one label per row, e.g., "T:A:C = 3:6:1"

    # Build combined profiles (5 rows × 8 cols)
    for (rt, ra, rc) in ratio_combos:
        row_ratio_texts.append(f"T:A:C = {rt}:{ra}:{rc}")
        for ti, ai, ci in product(range(2), range(2), range(2)):
            combined = combine_profiles(
                train[ti]["util_profile"], api[ai]["util_profile"], conv[ci]["util_profile"], rt, ra, rc
            )
            # Save CSV rows
            for h in range(24):
                out_rows.append({
                    "daytype": daytype,
                    "combo_ratio": f"{rt}:{ra}:{rc}",
                    "option": f"{ti+1}-{ai+1}-{ci+1}",
                    "hour": h,
                    "utilization": float(combined[h]),
                    "training_file": train[ti]["filename"],
                    "api_file": api[ai]["filename"],
                    "conversation_file": conv[ci]["filename"],
                })

            all_profiles.append(combined)

    # Save CSV
    out_df = pd.DataFrame(out_rows)
    csv_path = output_dir / f"utilization_profiles_40_{daytype}.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    # Plot: 5 rows (combos) × 8 columns (options)
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(26, 12), sharex=True, sharey=True)

    # Space: top for column headers, right for row labels
    plt.subplots_adjust(top=0.86, right=0.88, left=0.06, bottom=0.08, wspace=0.15, hspace=0.25)

    hours = np.arange(24)
    k = 0
    for r in range(5):
        for c in range(8):
            ax = axes[r, c]
            ax.plot(hours, all_profiles[k], marker='o')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 23)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

            # Top column headers (only on first row)
            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, pad=18)

            # Minimal axis labels to declutter
            if r == 4:
                ax.set_xlabel("Hour")
            if c == 0:
                ax.set_ylabel("Utilization")
            k += 1

    # Non-overlapping row labels in figure coordinates
    add_row_labels_no_overlap(fig, axes, row_ratio_texts, xpad=0.01, fontsize=12)

    # Overall title
    fig.suptitle(f"40 Combined Utilization Profiles ({daytype.capitalize()})", fontsize=16, y=0.94)
    fig_path = output_dir / f"combined_{daytype}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)
    print(f"[Saved] {fig_path}")

    return csv_path, fig_path


# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Collect processed source CSVs directly from default_data_dir.
    # The source files are intentionally limited to the top level of the data
    # directory. Generated CSVs are written to outputs/ and therefore will not
    # be mistaken for source distributions when the script is run again.
    csv_files = sorted(SOURCE_DATA_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No source CSV files found in {SOURCE_DATA_DIR}")

    # 2) First pass: load raw profiles (means, maxes) + meta
    raw_records = []
    parse_errors = []
    for f in csv_files:
        try:
            raw = load_raw_profile(f)
            raw_records.append(raw)
        except Exception as e:
            parse_errors.append((str(f), str(e)))

    print(f"Loaded raw from {len(csv_files)} CSVs; parsed: {len(raw_records)}; parse errors: {len(parse_errors)}")
    if parse_errors:
        err_df = pd.DataFrame(parse_errors, columns=["file", "error"])
        err_path = OUTPUT_DIR / "parse_issues.csv"
        err_df.to_csv(err_path, index=False)
        print(f"[Saved] parse issues to {err_path}")

    # 3) Collect weekday denominators by (category, subtype)
    weekday_denoms = {}
    for rec in raw_records:
        if rec["daytype"] == "weekday":
            key = (rec["category"], rec["subtype"])
            denom = rec["own_denom"]
            if np.isfinite(denom) and denom > 0:
                # If duplicates exist, take the max across weekday files of same (cat, subtype)
                weekday_denoms[key] = max(weekday_denoms.get(key, 0), denom)

    # 4) Second pass: normalize (weekend uses weekday_denoms when available)
    norm_records = []
    for rec in raw_records:
        try:
            norm = normalize_with_weekday_reference(rec, weekday_denoms)
            norm_records.append(norm)
        except Exception as e:
            parse_errors.append((rec["file"], f"Normalization error: {e}"))

    if len(parse_errors):
        err_df = pd.DataFrame(parse_errors, columns=["file", "error"])
        err_path = OUTPUT_DIR / "parse_issues.csv"
        err_df.to_csv(err_path, index=False)
        print(f"[Updated] parse issues to {err_path}")

    # 6) Build 40 profiles for each day type. The 5 composition ratios are
    # crossed with 8 source-profile options; weekday and weekend are generated
    # separately but use matching source identities.
    results = []
    for daytype in ["weekday", "weekend"]:
        csv_path, fig_path = make_40_profiles_for_daytype(norm_records, daytype, RATIO_COMBOS, OUTPUT_DIR)
        results.append((daytype, csv_path, fig_path))

    print("\nDone. Outputs:")
    for daytype, csvp, figp in results:
        print(f" - {daytype}: CSV={csvp}  FIG={figp}")

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Code block 3
# -----------------------------------------------------------------------------
#Improved version for present component decomposition
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build 40 utilization profiles (5 ratio mixes × 8 options) for both weekday and weekend,
using processed source distributions stored directly in the configured data directory.

All output rows (weekday + weekend) are saved to ONE CSV:
  utilization_profiles_40_all.csv

Each row corresponds to one hour (0..23) in one scenario (ratio×option) and includes:
- combined utilization
- separated component utilizations (training/api/conversation)
- normalized weights (per scenario)
- per-component contributions (sum to combined)
- provenance columns for the three selected source files

Also saves weekday/weekend overview figures under the configured outputs directory.
"""

import re
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
# All processed source-distribution CSVs are stored directly in the user-selected
# data directory. Generated products are written separately to outputs/.
SOURCE_DATA_DIR = source_data_dir
OUTPUT_DIR = output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_CSV_PATH = OUTPUT_DIR / "utilization_profiles_40_all.csv"  # single, consolidated CSV

# Workload-composition cases. Values are relative shares for training, API,
# and conversation activity before category-specific utilization multipliers are
# applied; the five tuples span the central and directional sensitivity cases.
# Ratios (Training : API : Conversation), in this order
RATIO_COMBOS = [
    (3.0, 3.5, 3.5),
    (3.0, 6.0, 1.0),
    (3.0, 1.0, 6.0),
    (1.0, 4.5, 4.5),
    (5.0, 2.5, 2.5),
]

# Category multipliers retain the assumed relative average utilization of the
# three workload classes when their normalized source shapes are combined.
# Utilization multipliers by category
UTIL_FACTORS = {
    "training": 1.0,
    "api": 0.6,
    "conversation": 0.6,
}

# -----------------------------
# Helpers
# -----------------------------

# Infer workload category, source subtype, and day type from the source
# filename. These labels determine which source profiles may be combined in each
# scenario option while keeping weekday and weekend records paired consistently.
def parse_meta(path: Path):
    """Infer category (training/api/conversation), subtype, and daytype (weekday/weekend) from filename."""
    name = path.stem.lower()

    # Daytype
    if "weekday" in name or "weekdays" in name:
        daytype = "weekday"
    elif "weekend" in name or "weekends" in name:
        daytype = "weekend"
    else:
        daytype = "weekday"  # fallback if not denoted

    # Category & subtype
    if any(k in name for k in ["seren", "kalos", "train", "training"]):
        category = "training"
        subtype = "seren" if "seren" in name else ("kalos" if "kalos" in name else "training")
    elif "api" in name:
        category = "api"
        subtype = "api1" if re.search(r"api.*(a|1)", name) else ("api2" if re.search(r"api.*(b|2)", name) else "api")
    elif any(k in name for k in ["conv", "conversation", "chat"]):
        category = "conversation"
        subtype = "conv1" if re.search(r"(conv|conversation).*?(a|1)", name) else ("conv2" if re.search(r"(conv|conversation).*?(b|2)", name) else "conversation")
    else:
        # Try parent folder
        parent = path.parent.name.lower()
        if any(k in parent for k in ["seren", "kalos", "train", "training"]):
            category, subtype = "training", parent
        elif "api" in parent:
            category, subtype = "api", parent
        elif any(k in parent for k in ["conv", "conversation", "chat"]):
            category, subtype = "conversation", parent
        else:
            category, subtype = "unknown", "unknown"

    return category, subtype, daytype

def get_mean_max_columns(df: pd.DataFrame):
    """Find 'mean' and 'max' columns case-insensitively."""
    lower_map = {c.lower().strip(): c for c in df.columns}
    mean_col = lower_map.get("mean")
    max_col  = lower_map.get("max")

    # Fallbacks
    if mean_col is None:
        candidates = [c for c in df.columns if c.strip().lower() == "mean"]
        mean_col = candidates[0] if candidates else None
    if max_col is None:
        candidates = [c for c in df.columns if c.strip().lower() == "max"]
        max_col = candidates[0] if candidates else None

    if mean_col is None or max_col is None:
        raise ValueError("Could not find 'mean' and/or 'max' columns.")
    return mean_col, max_col

def short_name_from_filename(filename: str, category: str) -> str:
    """
    Map filename to short display name:
      - training: 'Kalos' or 'Seren' (default to stem)
      - api/conv: 'Azure' or 'BurstyGPT' (default to stem)
    """
    fname = filename.lower()
    if category == "training":
        if "kalos" in fname:
            return "Kalos"
        if "seren" in fname:
            return "Seren"
        return Path(filename).stem
    else:
        if "azure" in fname:
            return "Azure"
        if "burst" in fname or "bursty" in fname or "gpt" in fname:
            return "BurstyGPT"
        return Path(filename).stem

def safe_token(name: str, lower: bool = False) -> str:
    """Keep only letters/digits and optionally lowercase."""
    cleaned = "".join(ch for ch in name if ch.isalnum())
    return cleaned.lower() if lower else cleaned

def fmt_ratio_token(x: float) -> str:
    """
    Format 3.0 -> '03' (two digits),
            3.5 -> '035' (three digits),
    mirroring the example 'kalos03_Azure035_Azure035'.
    """
    if float(x).is_integer():
        return f"{int(x):02d}"          # 3.0 -> '03'
    # assume one decimal place like *.5
    s = f"{x:.1f}".replace(".", "")     # 3.5 -> '35'
    return s.zfill(3)                   # -> '035'

def load_raw_profile(csv_path: Path):
    """Load first 24 rows, return raw arrays and meta; no normalization here."""
    df = pd.read_csv(csv_path)
    mean_col, max_col = get_mean_max_columns(df)
    df24 = df.iloc[:24].copy().reset_index(drop=True)
    means = pd.to_numeric(df24[mean_col], errors="coerce").values
    maxes = pd.to_numeric(df24[max_col], errors="coerce").values
    category, subtype, daytype = parse_meta(csv_path)
    return {
        "file": str(csv_path),
        "filename": csv_path.name,
        "category": category,
        "subtype": subtype,
        "daytype": daytype,
        "means": means,
        "maxes": maxes,
        "own_denom": np.nanmax(maxes) if len(maxes) else np.nan
    }

def normalize_with_weekday_reference(raw_rec, weekday_denoms):
    """
    For WEEKEND:
      use denom = weekday_denoms[(category, subtype)] if available,
      else fallback to own_denom.
    For WEEKDAY:
      use own_denom (and also populate weekday_denoms).
    Then apply utilization factor by category.
    """
    category = raw_rec["category"]
    subtype = raw_rec["subtype"]
    daytype = raw_rec["daytype"]

    if daytype == "weekday":
        denom = raw_rec["own_denom"]
    else:  # weekend
        denom = weekday_denoms.get((category, subtype), raw_rec["own_denom"])

    if denom is None or not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid normalization denominator for {raw_rec['filename']}: {denom}")

    base = np.clip(raw_rec["means"] / denom, 0.0, 1.0)
    factor = UTIL_FACTORS.get(category)
    if factor is None:
        raise ValueError(f"Unrecognized category in {raw_rec['filename']}: {category}")
    util = np.clip(base * factor, 0.0, 1.0)

    return {
        "file": raw_rec["file"],
        "filename": raw_rec["filename"],
        "category": category,
        "subtype": subtype,
        "daytype": daytype,
        "hour": np.arange(24),
        "util_profile": util,
    }

def option_index_triplet(opt_idx: int):
    """
    Map column index 0..7 to (ti, ai, ci) following product(range(2), range(2), range(2)):
      0: (0,0,0), 1: (0,0,1), 2: (0,1,0), 3: (0,1,1),
      4: (1,0,0), 5: (1,0,1), 6: (1,1,0), 7: (1,1,1)
    """
    mapping = [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]
    return mapping[opt_idx]

def add_row_labels_no_overlap(fig, axes, row_texts, xpad=0.01, fontsize=12):
    """Place row labels in figure coordinates (avoids overlap)."""
    nrows, ncols = axes.shape
    assert len(row_texts) == nrows
    for r in range(nrows):
        right_ax = axes[r, ncols - 1]
        x0, y0, w, h = right_ax.get_position().bounds
        y_mid = y0 + h / 2.0
        x_text = x0 + w + xpad
        fig.text(x_text, y_mid, row_texts[r], ha='left', va='center', fontsize=fontsize)

def make_40_profiles_for_daytype(records, daytype, ratio_combos, output_dir):
    """
    From per-file records, pick 2 training + 2 api + 2 conversation for the given daytype,
    build 5 combos × 8 options = 40 combined profiles.

    Returns:
      rows_for_all_csv: list of dicts for the single consolidated CSV
      fig_path: saved figure path
    """
    # Filter candidates for this daytype and category
    train = sorted([r for r in records if r["daytype"] == daytype and r["category"] == "training"], key=lambda x: x["filename"])
    api   = sorted([r for r in records if r["daytype"] == daytype and r["category"] == "api"],       key=lambda x: x["filename"])
    conv  = sorted([r for r in records if r["daytype"] == daytype and r["category"] == "conversation"], key=lambda x: x["filename"])

    # Expect >= 2 in each category
    if len(train) < 2 or len(api) < 2 or len(conv) < 2:
        raise RuntimeError(
            f"Need >=2 files each for training/api/conversation on {daytype}. "
            f"Have: training={len(train)}, api={len(api)}, conversation={len(conv)}"
        )

    # Keep only 2 per category (deterministic order)
    train = train[:2]
    api   = api[:2]
    conv  = conv[:2]

    # Friendly names for headers per candidate index
    T_names = [short_name_from_filename(t["filename"], "training") for t in train]
    A_names = [short_name_from_filename(a["filename"], "api") for a in api]
    C_names = [short_name_from_filename(c["filename"], "conversation") for c in conv]

    # Precompute the 8 column headers based on (ti, ai, ci)
    col_headers = []
    for col in range(8):
        ti, ai, ci = option_index_triplet(col)
        header = f"T: {T_names[ti]} | A: {A_names[ai]} | C: {C_names[ci]}"
        col_headers.append(header)

    all_profiles_for_plot = []
    row_ratio_texts = []  # "T:A:C = x:y:z"
    consolidated_rows = []

    # Build combined profiles (5 rows × 8 cols)
    for (rt, ra, rc) in ratio_combos:
        row_ratio_texts.append(f"T:A:C = {rt}:{ra}:{rc}")
        w = rt + ra + rc
        wt, wa, wc = rt / w, ra / w, rc / w  # normalized weights

        rt_tok = fmt_ratio_token(rt)
        ra_tok = fmt_ratio_token(ra)
        rc_tok = fmt_ratio_token(rc)

        for ti, ai, ci in product(range(2), range(2), range(2)):
            t_prof = train[ti]["util_profile"]
            a_prof = api[ai]["util_profile"]
            c_prof = conv[ci]["util_profile"]
            combined = (rt * t_prof + ra * a_prof + rc * c_prof) / w

            # raw component series
            t_raw = t_prof
            a_raw = a_prof
            c_raw = c_prof
            # weighted contributions (sum to combined)
            t_comp = wt * t_prof
            a_comp = wa * a_prof
            c_comp = wc * c_prof

            # scenario_name per your spec:
            # e.g., weekday_kalos03_Azure035_Azure035
            t_token = safe_token(T_names[ti], lower=True)    # 'kalos', 'seren'
            a_token = safe_token(A_names[ai], lower=False)   # keep case like 'Azure'/'BurstyGPT'
            c_token = safe_token(C_names[ci], lower=False)

            scenario_name = f"{daytype}_{t_token}{rt_tok}_{a_token}{ra_tok}_{c_token}{rc_tok}"

            for h in range(24):
                consolidated_rows.append({
                    "daytype": daytype,
                    "scenario_name": scenario_name,
                    "combo_ratio": f"{rt}:{ra}:{rc}",
                    "option": f"{ti+1}-{ai+1}-{ci+1}",
                    "hour": h,

                    # combined
                    "utilization": float(combined[h]),

                    # components (raw)
                    "training_util": float(t_raw[h]),
                    "api_util": float(a_raw[h]),
                    "conversation_util": float(c_raw[h]),

                    # normalized weights (per scenario)
                    "train_weight": float(wt),
                    "api_weight": float(wa),
                    "conv_weight": float(wc),

                    # weighted contributions to combined
                    "util_training_component": float(t_comp[h]),
                    "util_api_component": float(a_comp[h]),
                    "util_conv_component": float(c_comp[h]),

                    # provenance
                    "training_file": train[ti]["filename"],
                    "api_file": api[ai]["filename"],
                    "conversation_file": conv[ci]["filename"],
                })

            all_profiles_for_plot.append(combined)

    # ---- Plot (for quick QA): 5 rows (combos) × 8 columns (options) ----
    fig, axes = plt.subplots(nrows=5, ncols=8, figsize=(26, 12), sharex=True, sharey=True)
    plt.subplots_adjust(top=0.86, right=0.88, left=0.06, bottom=0.08, wspace=0.15, hspace=0.25)

    hours = np.arange(24)
    k = 0
    for r in range(5):
        for c in range(8):
            ax = axes[r, c]
            ax.plot(hours, all_profiles_for_plot[k], marker='o')
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 23)
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

            if r == 0:
                ax.set_title(col_headers[c], fontsize=10, pad=18)
            if r == 4:
                ax.set_xlabel("Hour")
            if c == 0:
                ax.set_ylabel("Utilization")
            k += 1

    add_row_labels_no_overlap(fig, axes, row_ratio_texts, xpad=0.01, fontsize=12)
    fig.suptitle(f"40 Combined Utilization Profiles ({daytype.capitalize()})", fontsize=16, y=0.94)
    fig_path = output_dir / f"combined_{daytype}.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    return consolidated_rows, fig_path

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Gather processed source CSVs directly from default_data_dir.
    # Only top-level CSV files are treated as source inputs. This prevents generated
    # CSVs in outputs/ from being re-read as source distributions on later runs.
    csv_files = sorted(SOURCE_DATA_DIR.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No source CSV files found in {SOURCE_DATA_DIR}")

    # 2) First pass: load raw profiles (means, maxes) + meta
    raw_records = []
    parse_errors = []
    for f in csv_files:
        try:
            raw = load_raw_profile(f)
            raw_records.append(raw)
        except Exception as e:
            parse_errors.append((str(f), str(e)))

    print(f"Loaded raw from {len(csv_files)} CSVs; parsed ok: {len(raw_records)}; errors: {len(parse_errors)}")
    if parse_errors:
        err_df = pd.DataFrame(parse_errors, columns=["file", "error"])
        (OUTPUT_DIR / "parse_issues.csv").write_text(err_df.to_csv(index=False))

    # 3) Collect weekday denominators by (category, subtype)
    weekday_denoms = {}
    for rec in raw_records:
        if rec["daytype"] == "weekday":
            key = (rec["category"], rec["subtype"])
            denom = rec["own_denom"]
            if np.isfinite(denom) and denom > 0:
                weekday_denoms[key] = max(weekday_denoms.get(key, 0), denom)

    # 4) Second pass: normalize (weekend uses weekday_denoms when available)
    norm_records = []
    for rec in raw_records:
        try:
            norm = normalize_with_weekday_reference(rec, weekday_denoms)
            norm_records.append(norm)
        except Exception as e:
            parse_errors.append((rec["file"], f"Normalization error: {e}"))

    if parse_errors:
        err_df = pd.DataFrame(parse_errors, columns=["file", "error"])
        (OUTPUT_DIR / "parse_issues.csv").write_text(err_df.to_csv(index=False))

    # 6) Build 40-profile outputs for weekday and weekend; collect ALL rows
    all_rows = []
    results = []
    for daytype in ["weekday", "weekend"]:
        rows, fig_path = make_40_profiles_for_daytype(norm_records, daytype, RATIO_COMBOS, OUTPUT_DIR)
        all_rows.extend(rows)
        results.append((daytype, fig_path))

    # 7) Save the complete component-resolved scenario bank to one table so the
    # downstream representative-scenario step can evaluate all 40 cases jointly.
    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(ALL_CSV_PATH, index=False)

    print("\nDone.")
    print(f" - Consolidated CSV: {ALL_CSV_PATH}")
    for daytype, figp in results:
        print(f" - {daytype} figure: {figp}")

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Code block 4
# -----------------------------------------------------------------------------
# Extract representative scenarios and draft figures (multi-metric clustering)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Paths derived from the shared default_data_dir
# ----------------------------
BASE_DIR = output_dir
IN_CSV   = BASE_DIR / "utilization_profiles_40_all.csv"
OUT_XLSX = BASE_DIR / "representative_weekday_weekend.xlsx"
OUT_FIG  = BASE_DIR / "representative_panels.png"

# ----------------------------
# Ratio-name mapping used only for interpretable labels in metadata and plots.
# The numeric ratio values remain the quantities used in feature construction.
# Ratio name mapping
# ----------------------------
CANONICAL_RATIOS = {
    (3.0, 3.5, 3.5): "Mid-Case",
    (3.0, 6.0, 1.0): "High API",
    (3.0, 1.0, 6.0): "High Conversation",
    (1.0, 4.5, 4.5): "Low Training",
    (5.0, 2.5, 2.5): "High Training",
}
_CANON_NORM = {k: np.array(k, float)/sum(k) for k in CANONICAL_RATIOS}
_CANON_NORM_ALIASES = {
    (0.3, 0.35, 0.35): "Mid-Case",
    (0.3, 0.6 , 0.1 ): "High API",
    (0.3, 0.1 , 0.6 ): "High Conversation",
    (0.1, 0.45, 0.45): "Low Training",
    (0.5, 0.25, 0.25): "High Training",
}

# ----------------------------
# Utilities
# ----------------------------
# Standardize expected column names from the consolidated scenario table and
# fail early if any component needed for clustering or reconstruction is missing.
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
        "combo_ratio",
    ]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for k in need:
        if k in lower:
            rename[lower[k]] = k
        else:
            tgt = k.replace("_", "")
            for c in df.columns:
                if c.replace("_","").lower() == tgt:
                    rename[c] = k
                    break
    df = df.rename(columns=rename)
    required = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
    ]
    miss = [k for k in required if k not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}")
    return df

def get_weekday(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekday")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekday_")].copy()

def get_weekend_for(weekday_name: str, df: pd.DataFrame):
    assert weekday_name.startswith("weekday_"), "Expected a weekday_* scenario name"
    weekend_name = "weekend_" + weekday_name[len("weekday_"):]
    return df[df["scenario_name"] == weekend_name].copy(), weekend_name

def ensure_24(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.size < 24:
        return np.pad(arr, (0, 24-arr.size), mode="edge")
    return arr[:24]

def _parse_combo_ratio_str(s: str):
    try:
        parts = [float(x) for x in str(s).strip().split(":")]
        return parts if len(parts)==3 else None
    except Exception:
        return None

def _ratio_from_name(name: str):
    # Try to infer ...03_035_035 patterns near the end of tokens
    digs = [float(x)/ (1000 if len(x)==3 else 100) for x in re.findall(r"(\d{2,3})", name)]
    if len(digs) >= 3:
        return digs[-3:]
    return None

def ratio_label_from_row(g: pd.DataFrame) -> str:
    # 1) explicit combo_ratio
    if "combo_ratio" in g.columns:
        cr = _parse_combo_ratio_str(g["combo_ratio"].iloc[0])
        if cr:
            vn = np.array(cr, float); vn = vn / vn.sum()
            for k, lab in _CANON_NORM.items():
                if np.allclose(vn, lab, atol=1e-6):
                    return CANONICAL_RATIOS[k]
            for vec, lab in _CANON_NORM_ALIASES.items():
                v2 = np.array(vec, float); v2 = v2 / v2.sum()
                if np.allclose(vn, v2, atol=1e-6):
                    return lab
    # 2) parse scenario_name
    sn = g["scenario_name"].iloc[0]
    parsed = _ratio_from_name(str(sn))
    if parsed:
        vn = np.array(parsed, float); vn = vn / vn.sum()
        for k, lab in _CANON_NORM.items():
            if np.allclose(vn, lab, atol=0.02):
                return CANONICAL_RATIOS[k]
        for vec, lab in _CANON_NORM_ALIASES.items():
            v2 = np.array(vec, float); v2 = v2 / v2.sum()
            if np.allclose(vn, v2, atol=0.02):
                return lab
    # 3) fallback: nearest to normalized weights
    tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])
    w = np.array([tw,aw,cw], float); wn = w / w.sum() if w.sum()>0 else w
    best_label, best_dist = None, 1e9
    for k, lab in _CANON_NORM.items():
        d = np.sum(np.abs(wn - lab))
        if d < best_dist:
            best_dist, best_label = d, CANONICAL_RATIOS[k]
    for vec, lab in _CANON_NORM_ALIASES.items():
        v2 = np.array(vec, float); v2 = v2 / v2.sum()
        d = np.sum(np.abs(wn - v2))
        if d < best_dist:
            best_dist, best_label = d, lab
    return best_label or "Unknown Mix"

# ----- feature builders -----
def _ramp_stats(series: np.ndarray):
    """Return (max_up, mean_abs) ramps based on consecutive hourly diffs."""
    diffs = np.diff(series.astype(float))
    if diffs.size == 0:
        return 0.0, 0.0
    max_up = float(np.maximum(diffs, 0).max(initial=0.0))
    mean_abs = float(np.mean(np.abs(diffs)))
    return max_up, mean_abs

# Construct the multi-metric feature vector used to compare scenario shapes.
# Features are standardized before K-means so quantities with larger raw scales
# do not dominate the distance calculation solely because of units.
def build_features_and_profiles(wdf: pd.DataFrame):
    total = {}
    tr_raw, ap_raw, cv_raw = {}, {}, {}
    tr_wgt, ap_wgt, cv_wgt = {}, {}, {}
    tr_w, ap_w, cv_w = {}, {}, {}
    ratio_group = {}
    rows = []

    for sid, g in wdf.groupby("scenario_name", sort=False):
        g = g.sort_values("hour")
        tot = ensure_24(g["utilization"].astype(float).to_numpy())
        tr  = ensure_24(g["training_util"].astype(float).to_numpy())
        ap  = ensure_24(g["api_util"].astype(float).to_numpy())
        cv  = ensure_24(g["conversation_util"].astype(float).to_numpy())
        tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])

        trw, apw, cvw = tr*tw, ap*aw, cv*cw

        peak = float(tot.max()) if tot.size else 0.0
        base = float(tot.min()) if tot.size else 0.0
        avg  = float(tot.mean()) if tot.size else 0.0
        lf   = (avg/peak) if peak>0 else 0.0

        # robust “time of peak” as hour of max; encode circularly
        tpk = int(np.argmax(tot)) if tot.size else 0
        tpk_sin = np.sin(2*np.pi*tpk/24.0)
        tpk_cos = np.cos(2*np.pi*tpk/24.0)

        # peak/base ratio (guard base==0)
        pbr = (peak / base) if base > 1e-9 else np.inf

        # ramp metrics
        max_up, mean_abs = _ramp_stats(tot)

        ratio_group[sid] = ratio_label_from_row(g)

        total[sid] = tot
        tr_raw[sid], ap_raw[sid], cv_raw[sid] = tr, ap, cv
        tr_wgt[sid], ap_wgt[sid], cv_wgt[sid] = tw, aw, cw
        tr_w[sid], ap_w[sid], cv_w[sid] = trw, apw, cvw

        rows.append({
            "scenario_name": sid,
            "peak": peak,
            "base": base,
            "avg": avg,
            "load_factor": lf,
            "time_peak_hr": tpk,
            "time_peak_sin": tpk_sin,
            "time_peak_cos": tpk_cos,
            "peak_base_ratio": pbr,
            "max_up_ramp": max_up,
            "mean_abs_ramp": mean_abs,
            "train_weight": tw,
            "api_weight": aw,
            "conv_weight": cw,
            "ratio_group": ratio_group[sid],
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        raise RuntimeError("No weekday scenarios found.")

    return (feat, total, tr_raw, ap_raw, cv_raw,
            tr_wgt, ap_wgt, cv_wgt, tr_w, ap_w, cv_w, ratio_group)

def pick_k_and_cluster(X: np.ndarray, kmin=3, kmax=10):
    best_k, best_labels, best_model, best_score = None, None, None, -1
    maxk = min(kmax, len(X))
    for k in range(kmin, maxk+1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_labels, best_model, best_score = k, labels, km, score
        except Exception:
            continue
    if best_k is None:
        k = min(4, len(X))
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        return k, km.labels_, km, float("nan")
    return best_k, best_labels, best_model, best_score

def choose_representatives(feat: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, X_scaled: np.ndarray):
    reps = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        memX = X_scaled[idx]
        ctr  = centers[c]
        # nearest member in scaled feature space
        d = np.linalg.norm(memX - ctr, axis=1)
        rep_idx_global = idx[d.argmin()]
        reps.append(feat.iloc[rep_idx_global].copy())
    rep_df = pd.DataFrame(reps).sort_values("cluster").reset_index(drop=True)
    return rep_df

def write_profiles_two_row_header(writer, sheet, scenario_names, disp_names, blocks):
    wb = writer.book
    ws = wb.add_worksheet(sheet)
    writer.sheets[sheet] = ws
    fmt_top = wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    fmt_sub = wb.add_format({"align": "center"})
    fmt_hour= wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    ws.merge_range(0,0,1,0,"Hour", fmt_hour)

    sub = ["total","training","api","conversation"]
    for i,key in enumerate(scenario_names):
        title = disp_names.get(key, key)
        c0 = 1 + 4*i
        ws.merge_range(0, c0, 0, c0+3, title, fmt_top)
        for j,lab in enumerate(sub):
            ws.write(1, c0+j, lab, fmt_sub)

    for h in range(24):
        ws.write(2+h, 0, h)
        for i,key in enumerate(scenario_names):
            c0 = 1 + 4*i
            ws.write(2+h, c0+0, float(blocks["total"][key][h]))
            ws.write(2+h, c0+1, float(blocks["training"][key][h]))
            ws.write(2+h, c0+2, float(blocks["api"][key][h]))
            ws.write(2+h, c0+3, float(blocks["conversation"][key][h]))
    ws.freeze_panes(2,1)
    ws.set_column(0,0,10)
    ws.set_column(1, 1+4*len(scenario_names), 15)

TITLE_FS  = 9
LEGEND_FS = 11

def stacked_area(ax, hour, trw, apw, cvw, title=None, show_ylabel=True):
    col_conv = "#D9B574"; col_api = "#F2E5B8"; col_trn = "#79C6C0"
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_linewidth(1.2)

    ax.fill_between(hour, 0, cvw, facecolor=col_conv, alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw, cvw+apw, facecolor=col_api, alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw+apw, cvw+apw+trw, facecolor=col_trn, alpha=1.0, linewidth=0)

    ax.set_xlim(0,23); ax.set_xticks([0,4,8,12,16,20])
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("Utilization Rate" if show_ylabel else "", fontsize=10)
    ymax = max(0.05, np.nanmax(cvw+apw+trw)*1.12); ax.set_ylim(0, ymax)
    if title: ax.set_title(title, fontsize=TITLE_FS, pad=3, fontweight="bold")
    ax.tick_params(axis='both', labelsize=9)

# ----------------------------
# Main
# ----------------------------
def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df = normalize_columns(df)

    # Weekday subset + features
    wdf = get_weekday(df)
    (feat, p_tot, p_tr, p_api, p_conv,
     tr_wgt, ap_wgt, cv_wgt, p_trw, p_apiw, p_convw, ratio_group) = build_features_and_profiles(wdf)

    # ----- Feature matrix for clustering (select a strong, non-redundant set)
    # Use: load_factor, peak, base, peak/base ratio, time-of-peak (sin,cos), max_up_ramp, mean_abs_ramp
    feature_cols = ["load_factor","peak","base","peak_base_ratio","time_peak_sin","time_peak_cos",
                    "max_up_ramp","mean_abs_ramp"]
    X = feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Cluster selection
    k, labels, km, sil = pick_k_and_cluster(Xs, 3, 10)
    feat["cluster"] = labels
    rep_df = choose_representatives(feat, labels, km.cluster_centers_, Xs)

    # Ratio-based names with A/B suffix for duplicates
    rep_df["ratio_group"] = rep_df["scenario_name"].map(ratio_group)
    dup_idx = rep_df.groupby("ratio_group").cumcount()
    display_names = {}
    for i,row in rep_df.iterrows():
        base = row["ratio_group"]
        display_names[row["scenario_name"]] = f"{base} {string.ascii_uppercase[dup_idx.loc[i]]}" if dup_idx.loc[i]>0 else base

    # Excel blocks (weighted components)
    wk_blocks = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    for name in rep_df["scenario_name"]:
        wk_blocks["total"][name]       = p_tot[name]
        wk_blocks["training"][name]    = p_trw[name]
        wk_blocks["api"][name]         = p_apiw[name]
        wk_blocks["conversation"][name]= p_convw[name]

    # Weekend blocks
    weekend_names = []
    weekend_blocks = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    for w_name in rep_df["scenario_name"]:
        wend_df, wend_name = get_weekend_for(w_name, df)
        weekend_names.append(wend_name)
        if wend_df.empty:
            weekend_blocks["total"][wend_name]        = np.full(24, np.nan)
            weekend_blocks["training"][wend_name]     = np.full(24, np.nan)
            weekend_blocks["api"][wend_name]          = np.full(24, np.nan)
            weekend_blocks["conversation"][wend_name] = np.full(24, np.nan)
        else:
            wend_df = wend_df.sort_values("hour")
            tot = ensure_24(wend_df["utilization"].astype(float).to_numpy())
            tr  = ensure_24(wend_df["training_util"].astype(float).to_numpy())
            ap  = ensure_24(wend_df["api_util"].astype(float).to_numpy())
            cv  = ensure_24(wend_df["conversation_util"].astype(float).to_numpy())
            tw, aw, cw = float(wend_df["train_weight"].iloc[0]), float(wend_df["api_weight"].iloc[0]), float(wend_df["conv_weight"].iloc[0])
            weekend_blocks["total"][wend_name]        = tot
            weekend_blocks["training"][wend_name]     = tr*tw
            weekend_blocks["api"][wend_name]          = ap*aw
            weekend_blocks["conversation"][wend_name] = cv*cw

    weekend_display = {wk: display_names.get("weekday_"+wk[len("weekend_"):], wk) for wk in weekend_names}

    # Save Excel
    # Save paired weekday/weekend representatives and their metadata in one
    # workbook so downstream model inputs can reproduce the selected scenario set.
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        write_profiles_two_row_header(writer, "Profiles_Weekday", list(rep_df["scenario_name"]), display_names, wk_blocks)
        write_profiles_two_row_header(writer, "Profiles_Weekend", weekend_names, weekend_display, weekend_blocks)

        # Metadata sheet (include the new metrics)
        meta = feat.copy()
        meta["ratio_group"] = meta["scenario_name"].map(ratio_group)
        meta["ratio_based_name"] = meta["scenario_name"].map(
            lambda s: display_names.get(s, meta.loc[meta["scenario_name"]==s, "ratio_group"].values[0])
        )
        meta["cluster_size"] = meta.groupby("cluster")["cluster"].transform("size")
        rep_set = set(rep_df["scenario_name"])
        meta["is_representative"] = meta["scenario_name"].isin(rep_set)

        keep_cols = ["cluster","scenario_name","ratio_group","ratio_based_name",
                     "peak","base","avg","load_factor","time_peak_hr","peak_base_ratio",
                     "max_up_ramp","mean_abs_ramp","cluster_size","is_representative",
                     "train_weight","api_weight","conv_weight"]
        meta[keep_cols].sort_values(["cluster","is_representative"], ascending=[True,False]).to_excel(
            writer, sheet_name="Metadata", index=False
        )

    # ---- Figure
    panels = []
    for w_name in rep_df["scenario_name"]:
        base = display_names[w_name]
        panels.append((f"{base} — Weekday", p_trw[w_name], p_apiw[w_name], p_convw[w_name], True))
        wend_df, wend_name = get_weekend_for(w_name, df)
        if wend_df.empty:
            trw=apw=cvw=np.full(24, np.nan)
        else:
            wend_df = wend_df.sort_values("hour")
            tr = ensure_24(wend_df["training_util"].astype(float).to_numpy())
            ap = ensure_24(wend_df["api_util"].astype(float).to_numpy())
            cv = ensure_24(wend_df["conversation_util"].astype(float).to_numpy())
            tw, aw, cw = float(wend_df["train_weight"].iloc[0]), float(wend_df["api_weight"].iloc[0]), float(wend_df["conv_weight"].iloc[0])
            trw, apw, cvw = tr*tw, ap*aw, cv*cw
        panels.append((f"{base} — Weekend", trw, apw, cvw, False))

    n_panels = len(panels)
    pairs_per_row = 2
    cols_per_row = 5
    n_rows = int(np.ceil((n_panels/2)/pairs_per_row))

    fig = plt.figure(figsize=(12.0, 2.9*n_rows))
    # Increase within-scenario spacing slightly (wspace), keep center spacer larger
    outer = gridspec.GridSpec(
        n_rows, cols_per_row, figure=fig,
        width_ratios=[1.0, 1.0, 0.36, 1.0, 1.0],  # center spacer for inter-scenario gap
        wspace=0.24,   # slightly larger than before to avoid overlap (within scenario)
        hspace=0.55,   # between rows
        left=0.06, right=0.98, top=0.92, bottom=0.09
    )

    # Legend (bold, large, close to plots)
    conv_patch = patches.Patch(facecolor="#D9B574", label="Conversation")
    api_patch  = patches.Patch(facecolor="#F2E5B8", label="API")
    trn_patch  = patches.Patch(facecolor="#79C6C0", label="Training")
    fig.legend(
        handles=[trn_patch, api_patch, conv_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
        columnspacing=1.2,
        prop={"weight":"bold","size":LEGEND_FS},
    )

    hr = np.arange(24)
    idx = 0
    for r in range(n_rows):
        for start in [0,3]:
            if idx < n_panels:
                ax = fig.add_subplot(outer[r, start])
                title,trw,apw,cvw,yl = panels[idx]; idx+=1
                stacked_area(ax, hr, trw, apw, cvw, title=title, show_ylabel=yl)
            if idx < n_panels:
                ax = fig.add_subplot(outer[r, start+1])
                title,trw,apw,cvw,yl = panels[idx]; idx+=1
                stacked_area(ax, hr, trw, apw, cvw, title=title, show_ylabel=yl)

    fig.savefig(OUT_FIG, dpi=300); plt.close(fig)

    print(f"Selected k={k} clusters on standardized multi-metric features (silhouette={sil:.3f} if defined).")
    print(f"Saved Excel: {OUT_XLSX}")
    print(f"Saved figure: {OUT_FIG}")

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Code block 5
# -----------------------------------------------------------------------------
# Extract representative scenarios and draft figures (joint weekday+weekend metrics)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

mpl.rcParams['svg.fonttype'] = 'none'   # keep text as text (not paths)

# ----------------------------
# Paths derived from the shared default_data_dir
# ----------------------------
BASE_DIR = output_dir
IN_CSV   = BASE_DIR / "utilization_profiles_40_all.csv"
OUT_XLSX = BASE_DIR / "representative_weekday_weekend.xlsx"
OUT_FIG  = BASE_DIR / "representative_panels.svg"

# ----------------------------
# Ratio-name mapping used only for interpretable labels in metadata and plots.
# The numeric ratio values remain the quantities used in feature construction.
# Ratio name mapping
# ----------------------------
CANONICAL_RATIOS = {
    (3.0, 3.5, 3.5): "Mid-Case",
    (3.0, 6.0, 1.0): "High API",
    (3.0, 1.0, 6.0): "High Conversation",
    (1.0, 4.5, 4.5): "Low Training",
    (5.0, 2.5, 2.5): "High Training",
}
_CANON_NORM = {k: np.array(k, float)/sum(k) for k in CANONICAL_RATIOS}
_CANON_NORM_ALIASES = {
    (0.3, 0.35, 0.35): "Mid-Case",
    (0.3, 0.6 , 0.1 ): "High API",
    (0.3, 0.1 , 0.6 ): "High Conversation",
    (0.1, 0.45, 0.45): "Low Training",
    (0.5, 0.25, 0.25): "High Training",
}

# ----------------------------
# Helpers
# ----------------------------
# Standardize expected column names from the consolidated scenario table and
# fail early if any component needed for clustering or reconstruction is missing.
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
        "combo_ratio",
    ]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for k in need:
        if k in lower:
            rename[lower[k]] = k
        else:
            tgt = k.replace("_", "")
            for c in df.columns:
                if c.replace("_","").lower() == tgt:
                    rename[c] = k
                    break
    df = df.rename(columns=rename)
    required = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
    ]
    miss = [k for k in required if k not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}")
    return df

def ensure_24(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    if arr.size < 24:
        return np.pad(arr, (0, 24-arr.size), mode="edge")
    return arr[:24]

def _parse_combo_ratio_str(s: str):
    try:
        parts = [float(x) for x in str(s).strip().split(":")]
        return parts if len(parts)==3 else None
    except Exception:
        return None

def _ratio_from_name(name: str):
    # Try to infer ...03_035_035 patterns near the end of tokens
    digs = [float(x)/ (1000 if len(x)==3 else 100) for x in re.findall(r"(\d{2,3})", name)]
    if len(digs) >= 3:
        return digs[-3:]
    return None

def ratio_label_from_row(g: pd.DataFrame) -> str:
    # 1) explicit combo_ratio
    if "combo_ratio" in g.columns:
        cr = _parse_combo_ratio_str(g["combo_ratio"].iloc[0])
        if cr:
            vn = np.array(cr, float); vn = vn / vn.sum()
            for k, lab in _CANON_NORM.items():
                if np.allclose(vn, lab, atol=1e-6):
                    return CANONICAL_RATIOS[k]
            for vec, lab in _CANON_NORM_ALIASES.items():
                v2 = np.array(vec, float); v2 = v2 / v2.sum()
                if np.allclose(vn, v2, atol=1e-6):
                    return lab
    # 2) parse scenario_name
    sn = g["scenario_name"].iloc[0]
    parsed = _ratio_from_name(str(sn))
    if parsed:
        vn = np.array(parsed, float); vn = vn / vn.sum()
        for k, lab in _CANON_NORM.items():
            if np.allclose(vn, lab, atol=0.02):
                return CANONICAL_RATIOS[k]
        for vec, lab in _CANON_NORM_ALIASES.items():
            v2 = np.array(vec, float); v2 = v2 / v2.sum()
            if np.allclose(vn, v2, atol=0.02):
                return lab
    # 3) fallback: nearest normalized weights
    tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])
    w = np.array([tw,aw,cw], float); wn = w / w.sum() if w.sum()>0 else w
    best_label, best_dist = None, 1e9
    for k, lab in _CANON_NORM.items():
        d = np.sum(np.abs(wn - lab))
        if d < best_dist:
            best_dist, best_label = d, CANONICAL_RATIOS[k]
    for vec, lab in _CANON_NORM_ALIASES.items():
        v2 = np.array(vec, float); v2 = v2 / v2.sum()
        d = np.sum(np.abs(wn - v2))
        if d < best_dist:
            best_dist, best_label = d, lab
    return best_label or "Unknown Mix"

def _ramp_stats(series: np.ndarray):
    diffs = np.diff(series.astype(float))
    if diffs.size == 0:
        return 0.0, 0.0
    max_up = float(np.maximum(diffs, 0).max(initial=0.0))
    mean_abs = float(np.mean(np.abs(diffs)))
    return max_up, mean_abs

# ---- compute metrics & store profiles (for one daytype) ----
# Construct the multi-metric feature vector used to compare scenario shapes.
# Features are standardized before K-means so quantities with larger raw scales
# do not dominate the distance calculation solely because of units.
def build_features_and_profiles(day_df: pd.DataFrame):
    total = {}
    tr_w, ap_w, cv_w = {}, {}, {}
    ratio_group = {}
    rows = []

    for sid, g in day_df.groupby("scenario_name", sort=False):
        g = g.sort_values("hour")
        tot = ensure_24(g["utilization"].astype(float).to_numpy())
        tr  = ensure_24(g["training_util"].astype(float).to_numpy())
        ap  = ensure_24(g["api_util"].astype(float).to_numpy())
        cv  = ensure_24(g["conversation_util"].astype(float).to_numpy())
        tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])

        trw, apw, cvw = tr*tw, ap*aw, cv*cw

        peak = float(tot.max()) if tot.size else 0.0
        base = float(tot.min()) if tot.size else 0.0
        avg  = float(tot.mean()) if tot.size else 0.0
        lf   = (avg/peak) if peak>0 else 0.0
        tpk  = int(np.argmax(tot)) if tot.size else 0
        tpk_sin = np.sin(2*np.pi*tpk/24.0)
        tpk_cos = np.cos(2*np.pi*tpk/24.0)
        pbr = (peak/base) if base > 1e-9 else np.inf
        max_up, mean_abs = _ramp_stats(tot)

        ratio_group[sid] = ratio_label_from_row(g)

        total[sid] = tot
        tr_w[sid], ap_w[sid], cv_w[sid] = trw, apw, cvw

        rows.append({
            "scenario_name": sid,
            "peak": peak,
            "base": base,
            "avg": avg,
            "load_factor": lf,
            "time_peak_hr": tpk,
            "time_peak_sin": tpk_sin,
            "time_peak_cos": tpk_cos,
            "peak_base_ratio": pbr,
            "max_up_ramp": max_up,
            "mean_abs_ramp": mean_abs,
            "train_weight": tw,
            "api_weight": aw,
            "conv_weight": cw,
            "ratio_group": ratio_group[sid],
        })

    feat = pd.DataFrame(rows)
    return feat, total, tr_w, ap_w, cv_w, ratio_group

def get_weekday(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekday")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekday_")].copy()

def get_weekend(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekend")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekend_")].copy()

def base_key(name: str) -> str:
    return name.replace("weekday_","").replace("weekend_","")

def pick_k_and_cluster(X: np.ndarray, kmin=3, kmax=10):
    best_k, best_labels, best_model, best_score = None, None, None, -1
    maxk = min(kmax, len(X))
    for k in range(kmin, maxk+1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_labels, best_model, best_score = k, labels, km, score
        except Exception:
            continue
    if best_k is None:
        k = min(4, len(X))
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        return k, km.labels_, km, float("nan")
    return best_k, best_labels, best_model, best_score

def choose_representatives(feat_wk_merge: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, X_scaled: np.ndarray):
    reps = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        memX = X_scaled[idx]
        ctr  = centers[c]
        d = np.linalg.norm(memX - ctr, axis=1)
        rep_idx_global = idx[d.argmin()]
        reps.append(feat_wk_merge.iloc[rep_idx_global].copy())
    rep_df = pd.DataFrame(reps).sort_values("cluster").reset_index(drop=True)
    return rep_df

def write_profiles_two_row_header(writer, sheet, scenario_names, disp_names, blocks):
    wb = writer.book
    ws = wb.add_worksheet(sheet)
    writer.sheets[sheet] = ws
    fmt_top = wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    fmt_sub = wb.add_format({"align": "center"})
    fmt_hour= wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    ws.merge_range(0,0,1,0,"Hour", fmt_hour)

    sub = ["total","training","api","conversation"]
    for i,key in enumerate(scenario_names):
        title = disp_names.get(key, key)
        c0 = 1 + 4*i
        ws.merge_range(0, c0, 0, c0+3, title, fmt_top)
        for j,lab in enumerate(sub):
            ws.write(1, c0+j, lab, fmt_sub)

    for h in range(24):
        ws.write(2+h, 0, h)
        for i,key in enumerate(scenario_names):
            c0 = 1 + 4*i
            ws.write(2+h, c0+0, float(blocks["total"][key][h]))
            ws.write(2+h, c0+1, float(blocks["training"][key][h]))
            ws.write(2+h, c0+2, float(blocks["api"][key][h]))
            ws.write(2+h, c0+3, float(blocks["conversation"][key][h]))
    ws.freeze_panes(2,1)
    ws.set_column(0,0,10)
    ws.set_column(1, 1+4*len(scenario_names), 15)

TITLE_FS  = 9
LEGEND_FS = 11

def stacked_area(ax, hour, trw, apw, cvw, title=None, show_ylabel=True):
    col_conv = "#D9B574"; col_api = "#F2E5B8"; col_trn = "#79C6C0"
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ["bottom","left"]: ax.spines[s].set_linewidth(1.2)

    ax.fill_between(hour, 0, cvw, facecolor=col_conv, alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw, cvw+apw, facecolor=col_api, alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw+apw, cvw+apw+trw, facecolor=col_trn, alpha=1.0, linewidth=0)

    ax.set_xlim(0,23); ax.set_xticks([0,4,8,12,16,20])
    ax.set_xlabel("Hour of Day", fontsize=10)
    ax.set_ylabel("Utilization Rate" if show_ylabel else "", fontsize=10)
    ymax = max(0.05, np.nanmax(cvw+apw+trw)*1.12); ax.set_ylim(0, ymax)
    if title: ax.set_title(title, fontsize=TITLE_FS, pad=4, fontweight="bold")
    ax.tick_params(axis='both', labelsize=9)

# ----------------------------
# Main
# ----------------------------
def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df = normalize_columns(df)

    # Split weekday / weekend
    wkdf = get_weekday(df)
    wedf = get_weekend(df)

    # Build metrics & weighted component profiles for EACH
    (feat_wk, p_tot_wk, p_trw_wk, p_apiw_wk, p_convw_wk, ratio_group_wk) = build_features_and_profiles(wkdf)
    (feat_we, p_tot_we, p_trw_we, p_apiw_we, p_convw_we, ratio_group_we) = build_features_and_profiles(wedf)

    # Merge features on base scenario key (suffix w/o day prefix)
    feat_wk["base_key"] = feat_wk["scenario_name"].map(base_key)
    feat_we["base_key"] = feat_we["scenario_name"].map(base_key)

    # Keep weekday rows as anchor, left-join weekend metrics
    sel = ["peak","base","avg","load_factor","time_peak_hr","time_peak_sin","time_peak_cos",
           "peak_base_ratio","max_up_ramp","mean_abs_ramp"]
    wk_cols = {c: f"wk_{c}" for c in sel}
    we_cols = {c: f"we_{c}" for c in sel}

    merged = (feat_wk[["scenario_name","ratio_group","base_key"] + sel]
              .rename(columns=wk_cols)
              .merge(feat_we[["base_key"] + sel].rename(columns=we_cols),
                     on="base_key", how="left"))

    # Build joint feature vector: weekday + weekend
    feature_cols = list(wk_cols.values()) + list(we_cols.values())
    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Cluster on JOINT features
    k, labels, km, sil = pick_k_and_cluster(Xs, 3, 10)
    merged["cluster"] = labels

    # Representatives: nearest to center in joint space (among weekday items)
    rep_df = choose_representatives(merged, labels, km.cluster_centers_, Xs)

    # Display names: ratio-based with A/B for duplicates
    rep_df["ratio_group"] = rep_df["ratio_group"]
    dup_idx = rep_df.groupby("ratio_group").cumcount()
    display_names = {}
    for i,row in rep_df.iterrows():
        base = row["ratio_group"]
        display_names[row["scenario_name"]] = f"{base} {string.ascii_uppercase[dup_idx.loc[i]]}" if dup_idx.loc[i]>0 else base

    # Excel blocks (weekday + weekend from dicts built above)
    wk_blocks = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    we_blocks = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    weekend_names = []

    for w_name in rep_df["scenario_name"]:
        # weekday storage
        wk_blocks["total"][w_name]        = p_tot_wk[w_name]
        wk_blocks["training"][w_name]     = p_trw_wk[w_name]
        wk_blocks["api"][w_name]          = p_apiw_wk[w_name]
        wk_blocks["conversation"][w_name] = p_convw_wk[w_name]

        # weekend pair
        wend_name = "weekend_" + base_key(w_name)
        weekend_names.append(wend_name)
        if wend_name in p_tot_we:
            we_blocks["total"][wend_name]        = p_tot_we[wend_name]
            we_blocks["training"][wend_name]     = p_trw_we[wend_name]
            we_blocks["api"][wend_name]          = p_apiw_we[wend_name]
            we_blocks["conversation"][wend_name] = p_convw_we[wend_name]
        else:
            we_blocks["total"][wend_name]        = np.full(24, np.nan)
            we_blocks["training"][wend_name]     = np.full(24, np.nan)
            we_blocks["api"][wend_name]          = np.full(24, np.nan)
            we_blocks["conversation"][wend_name] = np.full(24, np.nan)

    weekend_display = {wk: display_names.get("weekday_"+wk[len("weekend_"):], wk) for wk in weekend_names}

    # Save Excel (two-row headers)
    # Save paired weekday/weekend representatives and their metadata in one
    # workbook so downstream model inputs can reproduce the selected scenario set.
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        write_profiles_two_row_header(writer, "Profiles_Weekday",
                                      list(rep_df["scenario_name"]), display_names, wk_blocks)
        write_profiles_two_row_header(writer, "Profiles_Weekend",
                                      weekend_names, weekend_display, we_blocks)

        # Metadata: include both weekday and weekend metrics plus cluster info
        meta_cols = ["scenario_name","ratio_group","base_key","cluster"] + feature_cols
        merged[meta_cols].sort_values(["cluster","scenario_name"]).to_excel(writer, sheet_name="Metadata", index=False)

    # -------- Figure (slightly more within-scenario spacing; larger between-scenario spacer) --------
    # ---------------- Figure: 3 scenarios per row (each with Weekday + Weekend) ----------------
    panels = []
    for w_name in rep_df["scenario_name"]:
        base = display_names[w_name]
        # Weekday
        panels.append((f"{base} — Weekday",
                       wk_blocks["training"][w_name], wk_blocks["api"][w_name], wk_blocks["conversation"][w_name], True))
        # Weekend (paired by base_key)
        we_name = "weekend_" + base_key(w_name)
        panels.append((f"{base} — Weekend",
                       we_blocks["training"][we_name], we_blocks["api"][we_name], we_blocks["conversation"][we_name], False))

    n_panels = len(panels)             # 2 per scenario
    n_scen   = n_panels // 2
    scenarios_per_row = 3
    rows = int(np.ceil(n_scen / scenarios_per_row))

    # 3 scenario blocks per row → columns = [wk,we,spacer, wk,we,spacer, wk,we] = 8
    fig_width  = 16.5
    fig_height = 2.9 * rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(
        rows, 8, figure=fig,
        width_ratios=[1.0, 1.0, 0.40, 1.0, 1.0, 0.40, 1.0, 1.0],  # keep within-scenario close; larger inter-scenario gap
        wspace=0.28,   # slightly more within-scenario space to avoid overlap
        hspace=0.56,   # between rows
        left=0.06, right=0.98, top=0.92, bottom=0.09
    )

    # Legend (unchanged): bold, large, centered
    conv_patch = patches.Patch(facecolor="#D9B574", label="Conversation")
    api_patch  = patches.Patch(facecolor="#F2E5B8", label="API")
    trn_patch  = patches.Patch(facecolor="#79C6C0", label="Training")
    fig.legend(
        handles=[trn_patch, api_patch, conv_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
        columnspacing=1.2,
        prop={"weight": "bold", "size": LEGEND_FS},
    )

    # Fill grid: for each row, place 3 scenario pairs at columns (0,1), (3,4), (6,7)
    hour = np.arange(24)
    panel_idx = 0
    for r in range(rows):
        for block_start in (0, 3, 6):
            # Weekday panel of scenario
            if panel_idx < n_panels:
                ax = fig.add_subplot(outer[r, block_start])
                title, trw, apw, cvw, show_ylabel = panels[panel_idx]
                stacked_area(ax, hour, trw, apw, cvw, title=title, show_ylabel=show_ylabel)
                panel_idx += 1
            # Weekend panel of scenario
            if panel_idx < n_panels:
                ax = fig.add_subplot(outer[r, block_start + 1])
                title, trw, apw, cvw, show_ylabel = panels[panel_idx]
                stacked_area(ax, hour, trw, apw, cvw, title=title, show_ylabel=show_ylabel)  # no y-label for weekend
                panel_idx += 1
            # spacer column (block_start + 2) is left empty automatically

    fig.savefig(OUT_FIG, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"Selected k={k} on standardized JOINT features (weekday+weekend), silhouette={sil:.3f} (if defined).")
    print(f"Saved Excel: {OUT_XLSX}")
    print(f"Saved figure: {OUT_FIG}")

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Code block 6
# -----------------------------------------------------------------------------
# Extract representative scenarios and draft figures (joint weekday+weekend metrics)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- SVG & typography ---
mpl.rcParams['svg.fonttype'] = 'none'   # keep text as text (not paths)
mpl.rcParams['font.size'] = 6           # global font size = 6

# ----------------------------
# Paths derived from the shared default_data_dir
# ----------------------------
BASE_DIR = output_dir
IN_CSV   = BASE_DIR / "utilization_profiles_40_all.csv"
OUT_XLSX = BASE_DIR / "representative_weekday_weekend.xlsx"
OUT_FIG  = BASE_DIR / "representative_panels.svg"

# ----------------------------
# Ratio-name mapping used only for interpretable labels in metadata and plots.
# The numeric ratio values remain the quantities used in feature construction.
# Ratio name mapping
# ----------------------------
CANONICAL_RATIOS = {
    (3.0, 3.5, 3.5): "Mid-Case",
    (3.0, 6.0, 1.0): "High API",
    (3.0, 1.0, 6.0): "High Conversation",
    (1.0, 4.5, 4.5): "Low Training",
    (5.0, 2.5, 2.5): "High Training",
}
_CANON_NORM = {k: np.array(k, float)/sum(k) for k in CANONICAL_RATIOS}
_CANON_NORM_ALIASES = {
    (0.3, 0.35, 0.35): "Mid-Case",
    (0.3, 0.6 , 0.1 ): "High API",
    (0.3, 0.1 , 0.6 ): "High Conversation",
    (0.1, 0.45, 0.45): "Low Training",
    (0.5, 0.25, 0.25): "High Training",
}

# ----------------------------
# Helpers
# ----------------------------
# Standardize expected column names from the consolidated scenario table and
# fail early if any component needed for clustering or reconstruction is missing.
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
        "combo_ratio",
    ]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for k in need:
        if k in lower:
            rename[lower[k]] = k
        else:
            tgt = k.replace("_", "")
            for c in df.columns:
                if c.replace("_","").lower() == tgt:
                    rename[c] = k
                    break
    df = df.rename(columns=rename)
    required = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
    ]
    miss = [k for k in required if k not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}")
    return df

def ensure_24(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    if arr.size < 24:
        return np.pad(arr, (0, 24-arr.size), mode="edge")
    return arr[:24]

def _parse_combo_ratio_str(s: str):
    try:
        parts = [float(x) for x in str(s).strip().split(":")]
        return parts if len(parts)==3 else None
    except Exception:
        return None

def _ratio_from_name(name: str):
    # Try to infer ...03_035_035 patterns near the end of tokens
    digs = [float(x)/ (1000 if len(x)==3 else 100) for x in re.findall(r"(\d{2,3})", name)]
    if len(digs) >= 3:
        return digs[-3:]
    return None

def ratio_label_from_row(g: pd.DataFrame) -> str:
    # 1) explicit combo_ratio
    if "combo_ratio" in g.columns:
        cr = _parse_combo_ratio_str(g["combo_ratio"].iloc[0])
        if cr:
            vn = np.array(cr, float); vn = vn / vn.sum()
            for k, lab in _CANON_NORM.items():
                if np.allclose(vn, lab, atol=1e-6):
                    return CANONICAL_RATIOS[k]
            for vec, lab in _CANON_NORM_ALIASES.items():
                v2 = np.array(vec, float); v2 = v2 / v2.sum()
                if np.allclose(vn, v2, atol=1e-6):
                    return lab
    # 2) parse scenario_name
    sn = g["scenario_name"].iloc[0]
    parsed = _ratio_from_name(str(sn))
    if parsed:
        vn = np.array(parsed, float); vn = vn / vn.sum()
        for k, lab in _CANON_NORM.items():
            if np.allclose(vn, lab, atol=0.02):
                return CANONICAL_RATIOS[k]
        for vec, lab in _CANON_NORM_ALIASES.items():
            v2 = np.array(vec, float); v2 = v2 / v2.sum()
            if np.allclose(vn, v2, atol=0.02):
                return lab
    # 3) fallback: nearest normalized weights
    tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])
    w = np.array([tw,aw,cw], float); wn = w / w.sum() if w.sum()>0 else w
    best_label, best_dist = None, 1e9
    for k, lab in _CANON_NORM.items():
        d = np.sum(np.abs(wn - lab))
        if d < best_dist:
            best_dist, best_label = d, CANONICAL_RATIOS[k]
    for vec, lab in _CANON_NORM_ALIASES.items():
        v2 = np.array(vec, float); v2 = v2 / v2.sum()
        d = np.sum(np.abs(wn - v2))
        if d < best_dist:
            best_dist, best_label = d, lab
    return best_label or "Unknown Mix"

def _ramp_stats(series: np.ndarray):
    diffs = np.diff(series.astype(float))
    if diffs.size == 0:
        return 0.0, 0.0
    max_up = float(np.maximum(diffs, 0).max(initial=0.0))
    mean_abs = float(np.mean(np.abs(diffs)))
    return max_up, mean_abs

# ---- compute metrics & store profiles (for one daytype) ----
# Construct the multi-metric feature vector used to compare scenario shapes.
# Features are standardized before K-means so quantities with larger raw scales
# do not dominate the distance calculation solely because of units.
def build_features_and_profiles(day_df: pd.DataFrame):
    total = {}
    tr_w, ap_w, cv_w = {}, {}, {}
    ratio_group = {}
    rows = []

    for sid, g in day_df.groupby("scenario_name", sort=False):
        g = g.sort_values("hour")
        tot = ensure_24(g["utilization"].astype(float).to_numpy())
        tr  = ensure_24(g["training_util"].astype(float).to_numpy())
        ap  = ensure_24(g["api_util"].astype(float).to_numpy())
        cv  = ensure_24(g["conversation_util"].astype(float).to_numpy())
        tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])

        trw, apw, cvw = tr*tw, ap*aw, cv*cw

        peak = float(tot.max()) if tot.size else 0.0
        base = float(tot.min()) if tot.size else 0.0
        avg  = float(tot.mean()) if tot.size else 0.0
        lf   = (avg/peak) if peak>0 else 0.0
        tpk  = int(np.argmax(tot)) if tot.size else 0
        tpk_sin = np.sin(2*np.pi*tpk/24.0)
        tpk_cos = np.cos(2*np.pi*tpk/24.0)
        pbr = (peak/base) if base > 1e-9 else np.inf
        max_up, mean_abs = _ramp_stats(tot)

        ratio_group[sid] = ratio_label_from_row(g)

        total[sid] = tot
        tr_w[sid], ap_w[sid], cv_w[sid] = trw, apw, cvw

        rows.append({
            "scenario_name": sid,
            "peak": peak,
            "base": base,
            "avg": avg,
            "load_factor": lf,
            "time_peak_hr": tpk,
            "time_peak_sin": tpk_sin,
            "time_peak_cos": tpk_cos,
            "peak_base_ratio": pbr,
            "max_up_ramp": max_up,
            "mean_abs_ramp": mean_abs,
            "train_weight": tw,
            "api_weight": aw,
            "conv_weight": cw,
            "ratio_group": ratio_group[sid],
        })

    feat = pd.DataFrame(rows)
    return feat, total, tr_w, ap_w, cv_w, ratio_group

def get_weekday(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekday")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekday_")].copy()

def get_weekend(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekend")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekend_")].copy()

def base_key(name: str) -> str:
    return name.replace("weekday_","").replace("weekend_","")

def pick_k_and_cluster(X: np.ndarray, kmin=3, kmax=10):
    best_k, best_labels, best_model, best_score = None, None, None, -1
    maxk = min(kmax, len(X))
    for k in range(kmin, maxk+1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_labels, best_model, best_score = k, labels, km, score
        except Exception:
            continue
    if best_k is None:
        k = min(4, len(X))
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        return k, km.labels_, km, float("nan")
    return best_k, best_labels, best_model, best_score

def choose_representatives(feat_wk_merge: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, X_scaled: np.ndarray):
    reps = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        memX = X_scaled[idx]
        ctr  = centers[c]
        d = np.linalg.norm(memX - ctr, axis=1)
        rep_idx_global = idx[d.argmin()]
        reps.append(feat_wk_merge.iloc[rep_idx_global].copy())
    rep_df = pd.DataFrame(reps).sort_values("cluster").reset_index(drop=True)
    return rep_df

def write_profiles_two_row_header(writer, sheet, scenario_names, disp_names, blocks):
    wb = writer.book
    ws = wb.add_worksheet(sheet)
    writer.sheets[sheet] = ws
    fmt_top = wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    fmt_sub = wb.add_format({"align": "center"})
    fmt_hour= wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    ws.merge_range(0,0,1,0,"Hour", fmt_hour)

    sub = ["total","training","api","conversation"]
    for i,key in enumerate(scenario_names):
        title = disp_names.get(key, key)
        c0 = 1 + 4*i
        ws.merge_range(0, c0, 0, c0+3, title, fmt_top)
        for j,lab in enumerate(sub):
            ws.write(1, c0+j, lab, fmt_sub)

    for h in range(24):
        ws.write(2+h, 0, h)
        for i,key in enumerate(scenario_names):
            c0 = 1 + 4*i
            ws.write(2+h, c0+0, float(blocks["total"][key][h]))
            ws.write(2+h, c0+1, float(blocks["training"][key][h]))
            ws.write(2+h, c0+2, float(blocks["api"][key][h]))
            ws.write(2+h, c0+3, float(blocks["conversation"][key][h]))
    ws.freeze_panes(2,1)
    ws.set_column(0,0,10)
    ws.set_column(1, 1+4*len(scenario_names), 15)

# --- stacked area without per-panel title ---
def stacked_area(ax, hour, trw, apw, cvw, show_ylabel=True):
    col_conv = "#D9B574"; col_api = "#F2E5B8"; col_trn = "#79C6C0"
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ["bottom","left"]:
        ax.spines[s].set_linewidth(1.0)

    ax.fill_between(hour, 0, cvw,                facecolor=col_conv, alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw, cvw+apw,          facecolor=col_api,  alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw+apw, cvw+apw+trw,  facecolor=col_trn,  alpha=1.0, linewidth=0)

    ax.set_xlim(0,23); ax.set_xticks([0,4,8,12,16,20])
    ax.set_xlabel("Hour of Day", fontsize=6)
    ax.set_ylabel("Utilization Rate" if show_ylabel else "", fontsize=6)
    ymax = max(0.05, np.nanmax(cvw+apw+trw) * 1.10); ax.set_ylim(0, ymax)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=6)

# ----------------------------
# Main
# ----------------------------
def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df = normalize_columns(df)

    # Split weekday / weekend
    wkdf = get_weekday(df)
    wedf = get_weekend(df)

    # Build metrics & weighted component profiles for EACH
    (feat_wk, p_tot_wk, p_trw_wk, p_apiw_wk, p_convw_wk, ratio_group_wk) = build_features_and_profiles(wkdf)
    (feat_we, p_tot_we, p_trw_we, p_apiw_we, p_convw_we, ratio_group_we) = build_features_and_profiles(wedf)

    # Merge features on base scenario key (suffix w/o day prefix)
    feat_wk["base_key"] = feat_wk["scenario_name"].map(base_key)
    feat_we["base_key"] = feat_we["scenario_name"].map(base_key)

    # Keep weekday rows as anchor, left-join weekend metrics
    sel = ["peak","base","avg","load_factor","time_peak_hr","time_peak_sin","time_peak_cos",
           "peak_base_ratio","max_up_ramp","mean_abs_ramp"]
    wk_cols = {c: f"wk_{c}" for c in sel}
    we_cols = {c: f"we_{c}" for c in sel}

    merged = (feat_wk[["scenario_name","ratio_group","base_key"] + sel]
              .rename(columns=wk_cols)
              .merge(feat_we[["base_key"] + sel].rename(columns=we_cols),
                     on="base_key", how="left"))

    # Build joint feature vector: weekday + weekend
    feature_cols = list(wk_cols.values()) + list(we_cols.values())
    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Cluster on JOINT features
    k, labels, km, sil = pick_k_and_cluster(Xs, 3, 10)
    merged["cluster"] = labels

    # Representatives: nearest to center in joint space (among weekday items)
    merged["cluster"] = labels
    rep_df = choose_representatives(merged, labels, km.cluster_centers_, Xs)

    # Display names: ratio-based with A/B for duplicates
    rep_df["ratio_group"] = rep_df["ratio_group"]
    dup_idx = rep_df.groupby("ratio_group").cumcount()
    display_names = {}
    for i,row in rep_df.iterrows():
        base = row["ratio_group"]
        display_names[row["scenario_name"]] = f"{base} {string.ascii_uppercase[dup_idx.loc[i]]}" if dup_idx.loc[i]>0 else base

    # Excel blocks (weekday + weekend from dicts built above)
    wk_blocks = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    we_blocks = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    weekend_names = []

    for w_name in rep_df["scenario_name"]:
        # weekday storage
        wk_blocks["total"][w_name]        = p_tot_wk[w_name]
        wk_blocks["training"][w_name]     = p_trw_wk[w_name]
        wk_blocks["api"][w_name]          = p_apiw_wk[w_name]
        wk_blocks["conversation"][w_name] = p_convw_wk[w_name]

        # weekend pair
        wend_name = "weekend_" + base_key(w_name)
        weekend_names.append(wend_name)
        if wend_name in p_tot_we:
            we_blocks["total"][wend_name]        = p_tot_we[wend_name]
            we_blocks["training"][wend_name]     = p_trw_we[wend_name]
            we_blocks["api"][wend_name]          = p_apiw_we[wend_name]
            we_blocks["conversation"][wend_name] = p_convw_we[wend_name]
        else:
            we_blocks["total"][wend_name]        = np.full(24, np.nan)
            we_blocks["training"][wend_name]     = np.full(24, np.nan)
            we_blocks["api"][wend_name]          = np.full(24, np.nan)
            we_blocks["conversation"][wend_name] = np.full(24, np.nan)

    weekend_display = {wk: display_names.get("weekday_"+wk[len("weekend_"):], wk) for wk in weekend_names}

    # Save Excel (two-row headers)
    # Save paired weekday/weekend representatives and their metadata in one
    # workbook so downstream model inputs can reproduce the selected scenario set.
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        write_profiles_two_row_header(writer, "Profiles_Weekday",
                                      list(rep_df["scenario_name"]), display_names, wk_blocks)
        write_profiles_two_row_header(writer, "Profiles_Weekend",
                                      weekend_names, weekend_display, we_blocks)

        # Metadata: include both weekday and weekend metrics plus cluster info
        meta_cols = ["scenario_name","ratio_group","base_key","cluster"] + feature_cols
        merged[meta_cols].sort_values(["cluster","scenario_name"]).to_excel(writer, sheet_name="Metadata", index=False)

    # -------- SVG Figure (3 scenarios per row; pair-labelled; no subplot titles; font size 6) --------
    # Build ordered panels (weekday + weekend per scenario) and a label for each pair
    panels = []
    pair_labels = []
    for w_name in rep_df["scenario_name"]:
        pair_labels.append(display_names[w_name])
        # Weekday (show y-label)
        panels.append((wk_blocks["training"][w_name],
                       wk_blocks["api"][w_name],
                       wk_blocks["conversation"][w_name],
                       True))
        # Weekend (no y-label)
        we_name = "weekend_" + base_key(w_name)
        panels.append((we_blocks["training"][we_name],
                       we_blocks["api"][we_name],
                       we_blocks["conversation"][we_name],
                       False))

    n_panels = len(panels)           # 2 per scenario
    n_scen   = n_panels // 2
    scenarios_per_row = 3
    rows = int(np.ceil(n_scen / scenarios_per_row))

    # GridSpec: [wk, we, spacer, wk, we, spacer, wk, we]
    # Inter-scenario gap is smaller than previous, but still > inner gap
    fig_width  = 16.5
    fig_height = 2.6 * rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(
        rows, 8, figure=fig,
        width_ratios=[1.0, 1.0, 0.24, 1.0, 1.0, 0.24, 1.0, 1.0],  # smaller spacer; still > inner gap
        wspace=0.30,   # inner gap (weekday vs weekend)
        hspace=0.50,   # between rows
        left=0.06, right=0.98, top=0.90, bottom=0.10
    )

    # Legend (bold, small, close to panels)
    conv_patch = patches.Patch(facecolor="#D9B574", label="Conversation")
    api_patch  = patches.Patch(facecolor="#F2E5B8", label="API")
    trn_patch  = patches.Patch(facecolor="#79C6C0", label="Training")
    fig.legend(
        handles=[trn_patch, api_patch, conv_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
        columnspacing=1.0,
        prop={"weight": "bold", "size": 6},
    )

    # helper to draw one area plot
    def _stack(ax, hr, trw, apw, cvw, show_ylabel=True):
        stacked_area(ax, hr, trw, apw, cvw, show_ylabel=show_ylabel)

    hour = np.arange(24)
    panel_idx = 0
    label_idx = 0
    for r in range(rows):
        for block_start in (0, 3, 6):
            if label_idx >= len(pair_labels):
                break

            # Weekday
            ax_wk = fig.add_subplot(outer[r, block_start])
            trw, apw, cvw, yl = panels[panel_idx]; panel_idx += 1
            _stack(ax_wk, hour, trw, apw, cvw, show_ylabel=yl)

            # Weekend
            ax_we = fig.add_subplot(outer[r, block_start + 1])
            trw, apw, cvw, yl = panels[panel_idx]; panel_idx += 1
            _stack(ax_we, hour, trw, apw, cvw, show_ylabel=yl)  # weekend: no y-label

            # Centered pair label above the two subplots (no "Weekday/Weekend" text)
            bb_wk = ax_wk.get_position()
            bb_we = ax_we.get_position()
            x_center = (bb_wk.x0 + bb_wk.width/2 + bb_we.x0 + bb_we.width/2) / 2.0
            y_top    = max(bb_wk.y1, bb_we.y1)
            fig.text(
                x_center, y_top + 0.012,
                pair_labels[label_idx],
                ha="center", va="bottom",
                fontsize=6, fontweight="bold"
            )
            label_idx += 1

    # Save as SVG
    fig.savefig(OUT_FIG, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"Selected k={k} on standardized JOINT features (weekday+weekend), silhouette={sil:.3f} (if defined).")
    print(f"Saved Excel: {OUT_XLSX}")
    print(f"Saved figure: {OUT_FIG}")

if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# Code block 7
# -----------------------------------------------------------------------------
# Extract representative scenarios and draft figures (joint weekday+weekend metrics + ratio+resource flags)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec, patches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- SVG & typography ---
mpl.rcParams['svg.fonttype'] = 'none'   # keep text as text (not paths)
mpl.rcParams['font.size'] = 6           # global font size = 6

# ----------------------------
# Paths derived from the shared default_data_dir
# ----------------------------
BASE_DIR = output_dir
IN_CSV   = BASE_DIR / "utilization_profiles_40_all.csv"
OUT_XLSX = BASE_DIR / "representative_weekday_weekend.xlsx"
OUT_FIG  = BASE_DIR / "representative_panels.svg"
OUT_SCEN = BASE_DIR / "scenario_names_40.csv"                 # names for all 40 weekday scenarios
OUT_XLSX_ALL = BASE_DIR / "all_40_weekday_weekend.xlsx"       # <-- NEW: full 40-scenario workbook

# ----------------------------
# Ratio-name mapping used only for interpretable labels in metadata and plots.
# The numeric ratio values remain the quantities used in feature construction.
# Ratio name mapping
# ----------------------------
CANONICAL_RATIOS = {
    (3.0, 3.5, 3.5): "Baseline",          # changed from Mid-Case to Baseline
    (3.0, 6.0, 1.0): "High API",
    (3.0, 1.0, 6.0): "High Conversation",
    (1.0, 4.5, 4.5): "Low Training",
    (5.0, 2.5, 2.5): "High Training",
}
_CANON_NORM = {k: np.array(k, float)/sum(k) for k in CANONICAL_RATIOS}
_CANON_NORM_ALIASES = {
    (0.3, 0.35, 0.35): "Baseline",
    (0.3, 0.6 , 0.1 ): "High API",
    (0.3, 0.1 , 0.6 ): "High Conversation",
    (0.1, 0.45, 0.45): "Low Training",
    (0.5, 0.25, 0.25): "High Training",
}
RATIO_SORT_ORDER = {"Baseline":0, "High API":1, "High Conversation":2, "Low Training":3, "High Training":4}

# ----------------------------
# Helpers
# ----------------------------
# Standardize expected column names from the consolidated scenario table and
# fail early if any component needed for clustering or reconstruction is missing.
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    need = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
        "combo_ratio",
    ]
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for k in need:
        if k in lower:
            rename[lower[k]] = k
        else:
            tgt = k.replace("_", "")
            for c in df.columns:
                if c.replace("_","").lower() == tgt:
                    rename[c] = k
                    break
    df = df.rename(columns=rename)
    required = [
        "scenario_name", "daytype", "hour", "utilization",
        "training_util", "api_util", "conversation_util",
        "train_weight", "api_weight", "conv_weight",
    ]
    miss = [k for k in required if k not in df.columns]
    if miss:
        raise RuntimeError(f"Missing required columns: {miss}")
    return df

def ensure_24(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, float)
    if arr.size < 24:
        return np.pad(arr, (0, 24-arr.size), mode="edge")
    return arr[:24]

def _parse_combo_ratio_str(s: str):
    try:
        parts = [float(x) for x in str(s).strip().split(":")]
        return parts if len(parts)==3 else None
    except Exception:
        return None

def _ratio_from_name(name: str):
    digs = [float(x)/ (1000 if len(x)==3 else 100) for x in re.findall(r"(\d{2,3})", name)]
    if len(digs) >= 3:
        return digs[-3:]
    return None

def ratio_label_from_row(g: pd.DataFrame) -> str:
    if "combo_ratio" in g.columns:
        cr = _parse_combo_ratio_str(g["combo_ratio"].iloc[0])
        if cr:
            vn = np.array(cr, float); vn = vn / vn.sum()
            for k, lab in _CANON_NORM.items():
                if np.allclose(vn, lab, atol=1e-6):
                    return CANONICAL_RATIOS[k]
            for vec, lab in _CANON_NORM_ALIASES.items():
                v2 = np.array(vec, float); v2 = v2 / v2.sum()
                if np.allclose(vn, v2, atol=1e-6):
                    return lab
    sn = g["scenario_name"].iloc[0]
    parsed = _ratio_from_name(str(sn))
    if parsed:
        vn = np.array(parsed, float); vn = vn / vn.sum()
        for k, lab in _CANON_NORM.items():
            if np.allclose(vn, lab, atol=0.02):
                return CANONICAL_RATIOS[k]
        for vec, lab in _CANON_NORM_ALIASES.items():
            v2 = np.array(vec, float); v2 = v2 / v2.sum()
            if np.allclose(vn, v2, atol=0.02):
                return lab
    tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])
    w = np.array([tw,aw,cw], float); wn = w / w.sum() if w.sum()>0 else w
    best_label, best_dist = None, 1e9
    for k, lab in _CANON_NORM.items():
        d = np.sum(np.abs(wn - lab))
        if d < best_dist:
            best_dist, best_label = d, CANONICAL_RATIOS[k]
    for vec, lab in _CANON_NORM_ALIASES.items():
        v2 = np.array(vec, float); v2 = v2 / v2.sum()
        d = np.sum(np.abs(wn - v2))
        if d < best_dist:
            best_dist, best_label = d, lab
    return best_label or "Unknown Mix"

def _ramp_stats(series: np.ndarray):
    diffs = np.diff(series.astype(float))
    if diffs.size == 0:
        return 0.0, 0.0
    max_up = float(np.maximum(diffs, 0).max(initial=0.0))
    mean_abs = float(np.mean(np.abs(diffs)))
    return max_up, mean_abs

# ---- compute metrics & store profiles (for one daytype) ----
# Construct the multi-metric feature vector used to compare scenario shapes.
# Features are standardized before K-means so quantities with larger raw scales
# do not dominate the distance calculation solely because of units.
def build_features_and_profiles(day_df: pd.DataFrame):
    total = {}
    tr_w, ap_w, cv_w = {}, {}, {}
    ratio_group = {}
    rows = []

    for sid, g in day_df.groupby("scenario_name", sort=False):
        g = g.sort_values("hour")
        tot = ensure_24(g["utilization"].astype(float).to_numpy())
        tr  = ensure_24(g["training_util"].astype(float).to_numpy())
        ap  = ensure_24(g["api_util"].astype(float).to_numpy())
        cv  = ensure_24(g["conversation_util"].astype(float).to_numpy())
        tw, aw, cw = float(g["train_weight"].iloc[0]), float(g["api_weight"].iloc[0]), float(g["conv_weight"].iloc[0])

        trw, apw, cvw = tr*tw, ap*aw, cv*cw

        peak = float(tot.max()) if tot.size else 0.0
        base = float(tot.min()) if tot.size else 0.0
        avg  = float(tot.mean()) if tot.size else 0.0
        lf   = (avg/peak) if peak>0 else 0.0
        tpk  = int(np.argmax(tot)) if tot.size else 0
        tpk_sin = np.sin(2*np.pi*tpk/24.0)
        tpk_cos = np.cos(2*np.pi*tpk/24.0)
        pbr = (peak/base) if base > 1e-9 else np.inf
        max_up, mean_abs = _ramp_stats(tot)

        ratio_group[sid] = ratio_label_from_row(g)

        total[sid] = tot
        tr_w[sid], ap_w[sid], cv_w[sid] = trw, apw, cvw

        rows.append({
            "scenario_name": sid,
            "peak": peak,
            "base": base,
            "avg": avg,
            "load_factor": lf,
            "time_peak_hr": tpk,
            "time_peak_sin": tpk_sin,
            "time_peak_cos": tpk_cos,
            "peak_base_ratio": pbr,
            "max_up_ramp": max_up,
            "mean_abs_ramp": mean_abs,
            "train_weight": tw,
            "api_weight": aw,
            "conv_weight": cw,
            "ratio_group": ratio_group[sid],
        })

    feat = pd.DataFrame(rows)
    return feat, total, tr_w, ap_w, cv_w, ratio_group

def get_weekday(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekday")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekday_")].copy()

def get_weekend(df: pd.DataFrame) -> pd.DataFrame:
    if "daytype" in df.columns:
        return df[df["daytype"].astype(str).str.lower().eq("weekend")].copy()
    return df[df["scenario_name"].astype(str).str.startswith("weekend_")].copy()

def base_key(name: str) -> str:
    return name.replace("weekday_","").replace("weekend_","")

# --- resource columns detection (P/Q/R or by names) ---
def detect_resource_columns(df: pd.DataFrame):
    """
    Return (training_col, api_col, conversation_col) robustly.
    Prefer name-based detection; fallback to positional columns P,Q,R (16–18th).
    """
    cols = list(df.columns)
    lower = [c.lower() for c in cols]

    def _find(preds):
        for i, c in enumerate(cols):
            cl = lower[i]
            if all(p in cl for p in preds):
                return c
        return None

    tr_col = _find(["training","file"]) or _find(["train","file"]) or _find(["training"])
    api_col = _find(["api","file"]) or _find(["api"])
    cv_col  = _find(["conversation","file"]) or _find(["conv","file"]) or _find(["conversation"]) or _find(["conv"])

    # Avoid confusing util/weight columns
    def _sanitize(col):
        if col and any(k in col.lower() for k in ["util","hour","weight"]):
            return None
        return col
    tr_col = _sanitize(tr_col)
    api_col = _sanitize(api_col)
    cv_col = _sanitize(cv_col)

    # Fallback to P/Q/R (16th–18th columns)
    if tr_col is None or api_col is None or cv_col is None:
        if len(cols) >= 18:
            tr_col = tr_col or cols[15]
            api_col = api_col or cols[16]
            cv_col = cv_col or cols[17]
        else:
            raise RuntimeError("Cannot detect resource columns (P/Q/R) and not enough columns to fallback.")

    return tr_col, api_col, cv_col

def pick_k_and_cluster(X: np.ndarray, kmin=3, kmax=10):
    best_k, best_labels, best_model, best_score = None, None, None, -1
    maxk = min(kmax, len(X))
    for k in range(kmin, maxk+1):
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_labels, best_model, best_score = k, labels, km, score
        except Exception:
            continue
    if best_k is None:
        k = min(4, len(X))
        km = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
        return k, km.labels_, km, float("nan")
    return best_k, best_labels, best_model, best_score

def choose_representatives(feat_wk_merge: pd.DataFrame, labels: np.ndarray, centers: np.ndarray, X_scaled: np.ndarray):
    reps = []
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        memX = X_scaled[idx]
        ctr  = centers[c]
        d = np.linalg.norm(memX - ctr, axis=1)
        rep_idx_global = idx[d.argmin()]
        reps.append(feat_wk_merge.iloc[rep_idx_global].copy())
    rep_df = pd.DataFrame(reps).sort_values("cluster").reset_index(drop=True)
    return rep_df

def write_profiles_two_row_header(writer, sheet, scenario_names, disp_names, blocks):
    wb = writer.book
    ws = wb.add_worksheet(sheet)
    writer.sheets[sheet] = ws
    fmt_top = wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    fmt_sub = wb.add_format({"align": "center"})
    fmt_hour= wb.add_format({"bold": True, "align": "center", "valign": "vcenter"})
    ws.merge_range(0,0,1,0,"Hour", fmt_hour)

    sub = ["total","training","api","conversation"]
    for i,key in enumerate(scenario_names):
        title = disp_names.get(key, key)
        c0 = 1 + 4*i
        ws.merge_range(0, c0, 0, c0+3, title, fmt_top)
        for j,lab in enumerate(sub):
            ws.write(1, c0+j, lab, fmt_sub)

    for h in range(24):
        ws.write(2+h, 0, h)
        for i,key in enumerate(scenario_names):
            c0 = 1 + 4*i
            ws.write(2+h, c0+0, float(blocks["total"][key][h]))
            ws.write(2+h, c0+1, float(blocks["training"][key][h]))
            ws.write(2+h, c0+2, float(blocks["api"][key][h]))
            ws.write(2+h, c0+3, float(blocks["conversation"][key][h]))
    ws.freeze_panes(2,1)
    ws.set_column(0,0,10)
    ws.set_column(1, 1+4*len(scenario_names), 15)

# --- stacked area without per-panel title ---
def stacked_area(ax, hour, trw, apw, cvw, show_ylabel=True):
    col_conv = "#D9B574"; col_api = "#F2E5B8"; col_trn = "#79C6C0"
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for s in ["bottom","left"]:
        ax.spines[s].set_linewidth(1.0)
    ax.fill_between(hour, 0, cvw,                facecolor=col_conv, alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw, cvw+apw,          facecolor=col_api,  alpha=1.0, linewidth=0)
    ax.fill_between(hour, cvw+apw, cvw+apw+trw,  facecolor=col_trn,  alpha=1.0, linewidth=0)
    ax.set_xlim(0,23); ax.set_xticks([0,4,8,12,16,20])
    ax.set_xlabel("Hour of Day", fontsize=6)
    ax.set_ylabel("Utilization Rate" if show_ylabel else "", fontsize=6)
    ymax = max(0.05, np.nanmax(cvw+apw+trw) * 1.10); ax.set_ylim(0, ymax)
    ax.grid(False)
    ax.tick_params(axis='both', labelsize=6)

# ----------------------------
# Main
# ----------------------------
def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df = normalize_columns(df)

    # Detect resource columns (P/Q/R fallback)
    tr_res_col, api_res_col, conv_res_col = detect_resource_columns(df)

    # Split weekday / weekend
    wkdf = get_weekday(df)
    wedf = get_weekend(df)

    # Build metrics & weighted component profiles
    (feat_wk, p_tot_wk, p_trw_wk, p_apiw_wk, p_convw_wk, ratio_group_wk) = build_features_and_profiles(wkdf)
    (feat_we, p_tot_we, p_trw_we, p_apiw_we, p_convw_we, ratio_group_we) = build_features_and_profiles(wedf)

    # ----- Build resource-combination flags (A, B, C, ...) on weekday base scenarios -----
    res_triplets = {}
    for sid, g in wkdf.groupby("scenario_name", sort=False):
        bk = base_key(sid)
        tr_name  = str(g[tr_res_col].iloc[0]) if tr_res_col in g.columns else ""
        api_name = str(g[api_res_col].iloc[0]) if api_res_col in g.columns else ""
        cv_name  = str(g[conv_res_col].iloc[0]) if conv_res_col in g.columns else ""
        res_triplets[bk] = (tr_name, api_name, cv_name)

    unique_combos = sorted(set(res_triplets.values()))
    flag_letters = list(string.ascii_uppercase)
    combo_to_flag = {combo: flag_letters[i] for i, combo in enumerate(unique_combos)}

    # Build scenario naming map (all 40 weekday scenarios): "<RatioName> <Flag>"
    scen_name_map = {}
    scen_rows = []
    ratio_for_sid = dict(zip(feat_wk["scenario_name"], feat_wk["ratio_group"]))

    for sid in sorted(wkdf["scenario_name"].unique()):
        bk = base_key(sid)
        ratio_name = ratio_for_sid.get(sid, "Unknown Mix")
        res_combo = res_triplets.get(bk, ("", "", ""))
        flag = combo_to_flag.get(res_combo, "?")
        final_name = f"{ratio_name} {flag}"
        scen_name_map[sid] = final_name
        scen_rows.append({
            "scenario_name_weekday": sid,
            "base_key": bk,
            "ratio_name": ratio_name,
            "training_resource": res_combo[0],
            "api_resource": res_combo[1],
            "conversation_resource": res_combo[2],
            "resource_flag": flag,
            "final_name": final_name
        })

    scen_df = pd.DataFrame(scen_rows)
    scen_df.to_csv(OUT_SCEN, index=False)

    # Merge features on base scenario key (suffix w/o day prefix)
    feat_wk["base_key"] = feat_wk["scenario_name"].map(base_key)
    feat_we["base_key"] = feat_we["scenario_name"].map(base_key)

    # Keep weekday rows as anchor, left-join weekend metrics
    sel = ["peak","base","avg","load_factor","time_peak_hr","time_peak_sin","time_peak_cos",
           "peak_base_ratio","max_up_ramp","mean_abs_ramp"]
    wk_cols = {c: f"wk_{c}" for c in sel}
    we_cols = {c: f"we_{c}" for c in sel}

    merged = (feat_wk[["scenario_name","ratio_group","base_key"] + sel]
              .rename(columns=wk_cols)
              .merge(feat_we[["base_key"] + sel].rename(columns=we_cols),
                     on="base_key", how="left"))

    # Build joint feature vector: weekday + weekend
    feature_cols = list(wk_cols.values()) + list(we_cols.values())
    X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Cluster on joint features
    k, labels, km, sil = pick_k_and_cluster(Xs, 3, 10)
    merged["cluster"] = labels

    # Representatives: nearest to center
    rep_df = choose_representatives(merged, labels, km.cluster_centers_, Xs)

    # --------- Display names for representatives: RatioName + ResourceFlag ----------
    display_names_rep = {}
    for _, row in rep_df.iterrows():
        sid = row["scenario_name"]
        bk  = row["base_key"]
        ratio_name = row["ratio_group"]
        flag = combo_to_flag.get(res_triplets.get(bk, ("", "", "")), "?")
        display_names_rep[sid] = f"{ratio_name} {flag}"

    # Excel blocks (weekday + weekend) for representatives
    wk_blocks_rep = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    we_blocks_rep = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    weekend_names_rep = []

    for w_name in rep_df["scenario_name"]:
        wk_blocks_rep["total"][w_name]        = p_tot_wk[w_name]
        wk_blocks_rep["training"][w_name]     = p_trw_wk[w_name]
        wk_blocks_rep["api"][w_name]          = p_apiw_wk[w_name]
        wk_blocks_rep["conversation"][w_name] = p_convw_wk[w_name]

        wend_name = "weekend_" + base_key(w_name)
        weekend_names_rep.append(wend_name)
        if wend_name in p_tot_we:
            we_blocks_rep["total"][wend_name]        = p_tot_we[wend_name]
            we_blocks_rep["training"][wend_name]     = p_trw_we[wend_name]
            we_blocks_rep["api"][wend_name]          = p_apiw_we[wend_name]
            we_blocks_rep["conversation"][wend_name] = p_convw_we[wend_name]
        else:
            we_blocks_rep["total"][wend_name]        = np.full(24, np.nan)
            we_blocks_rep["training"][wend_name]     = np.full(24, np.nan)
            we_blocks_rep["api"][wend_name]          = np.full(24, np.nan)
            we_blocks_rep["conversation"][wend_name] = np.full(24, np.nan)

    weekend_display_rep = {wk: display_names_rep.get("weekday_"+wk[len("weekend_"):], wk) for wk in weekend_names_rep}

    # --------- Excel blocks for ALL 40 scenarios (NEW) ----------
    wk_blocks_all = {"total": {}, "training": {}, "api": {}, "conversation": {}}
    we_blocks_all = {"total": {}, "training": {}, "api": {}, "conversation": {}}

    # ordering: by ratio name (RATIO_SORT_ORDER), then by resource flag, then by base_key for stability
    scen_df["ratio_order"] = scen_df["ratio_name"].map(lambda x: RATIO_SORT_ORDER.get(x, 999))
    scen_df_sorted = scen_df.sort_values(["ratio_order", "resource_flag", "base_key"]).reset_index(drop=True)
    all_weekday_names = list(scen_df_sorted["scenario_name_weekday"])
    display_names_all = {sid: scen_df_sorted.loc[scen_df_sorted["scenario_name_weekday"]==sid, "final_name"].values[0]
                         for sid in all_weekday_names}

    for sid in all_weekday_names:
        wk_blocks_all["total"][sid]        = p_tot_wk[sid]
        wk_blocks_all["training"][sid]     = p_trw_wk[sid]
        wk_blocks_all["api"][sid]          = p_apiw_wk[sid]
        wk_blocks_all["conversation"][sid] = p_convw_wk[sid]

        we_name = "weekend_" + base_key(sid)
        if we_name in p_tot_we:
            we_blocks_all["total"][we_name]        = p_tot_we[we_name]
            we_blocks_all["training"][we_name]     = p_trw_we[we_name]
            we_blocks_all["api"][we_name]          = p_apiw_we[we_name]
            we_blocks_all["conversation"][we_name] = p_convw_we[we_name]
        else:
            we_blocks_all["total"][we_name]        = np.full(24, np.nan)
            we_blocks_all["training"][we_name]     = np.full(24, np.nan)
            we_blocks_all["api"][we_name]          = np.full(24, np.nan)
            we_blocks_all["conversation"][we_name] = np.full(24, np.nan)

    weekend_display_all = { "weekend_" + base_key(sid): display_names_all[sid] for sid in all_weekday_names }
    all_weekend_names = list(weekend_display_all.keys())

    # ---------------- Save Excel (representatives) ----------------
    # Save paired weekday/weekend representatives and their metadata in one
    # workbook so downstream model inputs can reproduce the selected scenario set.
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        write_profiles_two_row_header(writer, "Profiles_Weekday",
                                      list(rep_df["scenario_name"]), display_names_rep, wk_blocks_rep)
        write_profiles_two_row_header(writer, "Profiles_Weekend",
                                      weekend_names_rep, weekend_display_rep, we_blocks_rep)

        meta_cols = ["scenario_name","ratio_group","base_key","cluster"] + feature_cols
        merged[meta_cols].sort_values(["cluster","scenario_name"]).to_excel(writer, sheet_name="Metadata", index=False)

    # ---------------- Save Excel (ALL 40 scenarios, NEW) ----------------
    # Save the complete 40-scenario bank in parallel with the representative
    # subset to preserve the full sensitivity space used for selection.
    with pd.ExcelWriter(OUT_XLSX_ALL, engine="xlsxwriter") as writer:
        write_profiles_two_row_header(writer, "Profiles_Weekday",
                                      all_weekday_names, display_names_all, wk_blocks_all)
        write_profiles_two_row_header(writer, "Profiles_Weekend",
                                      all_weekend_names, weekend_display_all, we_blocks_all)

        # Include a small index with ratio & flag
        scen_df_sorted[[
            "scenario_name_weekday","final_name","ratio_name","resource_flag",
            "training_resource","api_resource","conversation_resource"
        ]].to_excel(writer, sheet_name="Index", index=False)

    # -------- Figure (representatives) --------
    panels = []
    pair_labels = []
    for w_name in rep_df["scenario_name"]:
        pair_labels.append(display_names_rep[w_name])
        panels.append((p_trw_wk[w_name], p_apiw_wk[w_name], p_convw_wk[w_name], True))
        we_name = "weekend_" + base_key(w_name)
        panels.append((p_trw_we.get(we_name, np.full(24, np.nan)),
                       p_apiw_we.get(we_name, np.full(24, np.nan)),
                       p_convw_we.get(we_name, np.full(24, np.nan)),
                       False))

    n_panels = len(panels)
    n_scen   = n_panels // 2
    scenarios_per_row = 3
    rows = int(np.ceil(n_scen / scenarios_per_row))

    fig_width  = 16.5
    fig_height = 2.6 * rows
    fig = plt.figure(figsize=(fig_width, fig_height))
    outer = gridspec.GridSpec(
        rows, 8, figure=fig,
        width_ratios=[1.0, 1.0, 0.24, 1.0, 1.0, 0.24, 1.0, 1.0],
        wspace=0.16, hspace=0.50,
        left=0.06, right=0.98, top=0.90, bottom=0.10
    )

    conv_patch = patches.Patch(facecolor="#D9B574", label="Conversation")
    api_patch  = patches.Patch(facecolor="#F2E5B8", label="API")
    trn_patch  = patches.Patch(facecolor="#79C6C0", label="Training")
    fig.legend(
        handles=[trn_patch, api_patch, conv_patch],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=3,
        frameon=False,
        borderaxespad=0.2,
        columnspacing=1.0,
        prop={"weight": "bold", "size": 6},
    )

    def _stack(ax, hr, trw, apw, cvw, show_ylabel=True):
        stacked_area(ax, hr, trw, apw, cvw, show_ylabel=show_ylabel)

    hour = np.arange(24)
    panel_idx = 0
    label_idx = 0
    for r in range(rows):
        for block_start in (0, 3, 6):
            if label_idx >= len(pair_labels):
                break
            ax_wk = fig.add_subplot(outer[r, block_start])
            trw, apw, cvw, yl = panels[panel_idx]; panel_idx += 1
            _stack(ax_wk, hour, trw, apw, cvw, show_ylabel=yl)

            ax_we = fig.add_subplot(outer[r, block_start + 1])
            trw, apw, cvw, yl = panels[panel_idx]; panel_idx += 1
            _stack(ax_we, hour, trw, apw, cvw, show_ylabel=yl)

            bb_wk = ax_wk.get_position()
            bb_we = ax_we.get_position()
            x_center = (bb_wk.x0 + bb_wk.width/2 + bb_we.x0 + bb_we.width/2) / 2.0
            y_top    = max(bb_wk.y1, bb_we.y1)
            fig.text(
                x_center, y_top + 0.012,
                pair_labels[label_idx],
                ha="center", va="bottom",
                fontsize=6, fontweight="bold"
            )
            label_idx += 1

    fig.savefig(OUT_FIG, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(f"Selected k={k} on standardized JOINT features (weekday+weekend), silhouette={sil:.3f} (if defined).")
    print(f"Saved Excel (representatives): {OUT_XLSX}")
    print(f"Saved Excel (ALL 40): {OUT_XLSX_ALL}")
    print(f"Saved figure: {OUT_FIG}")
    print(f"Saved 40-scenario names CSV: {OUT_SCEN}")

if __name__ == "__main__":
    main()
