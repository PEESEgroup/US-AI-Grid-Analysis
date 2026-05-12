#!/usr/bin/env python3
r"""
Generate ReEDS hourly load files for pure AI demand with ramp-minimizing
training-load shaving.

"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

STATES: List[str] = [
    "Alabama", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "District of Columbia", "Florida", "Georgia", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
    "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire",
    "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "South Carolina",
    "South Dakota", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "West Virginia", "Wisconsin", "Wyoming",
]

WEATHER_YEARS: Sequence[int] = (2007, 2008, 2009, 2010, 2011, 2012, 2013)
HOURS_PER_YEAR = 8760
N_WEATHER_YEARS = len(WEATHER_YEARS)
N_TOTAL_HOURS = HOURS_PER_YEAR * N_WEATHER_YEARS
TARGET_MODELED_YEARS: Sequence[int] = (2025, 2030, 2035)
TARGET_YEAR_TO_INDEX: Mapping[int, int] = {2025: 1, 2030: 6, 2035: 11}

LOCAL_SEARCH_ITERS = 2
LINE_SEARCH_STEPS = 9
CANDIDATE_RAMP_COUNT = 10

REDUCTION_SCENARIOS: Mapping[str, float] = {
    "train_shave5": 0.05,
    "train_shave10": 0.10,
    "train_shave20": 0.20,
}


@dataclass(frozen=True)
class Config:
    data_dir: Path
    reeds_input_dir: Path
    output_dir: Path
    renewable_generation_dir: Path
    baseline_renewable_generation_file: Path
    ai_capacity_file: Path
    pue_file: Path
    pue_improvement_file: Path
    utilization_file: Path
    training_utilization_file: Path
    ai_spatial_file: Path
    ba_area_file: Path
    base_load_file: Path
    ratio_output_file: Path
    projection_number: int = 2
    overwrite: bool = True


@dataclass
class BAMapping:
    ba_names: np.ndarray
    ba_state_ids: np.ndarray
    region_ids: np.ndarray
    ba_state_ratio: np.ndarray
    valid_ba_indices: np.ndarray


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------
def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def resolve_first_existing(candidates: Sequence[Path], description: str) -> Path:
    """Return the first existing path from a list of candidates."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{description} not found. Checked:\n  {checked}")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


def parse_args() -> Config:
    default_data_dir = "Data Path" # Replace your path here
    parser = argparse.ArgumentParser(
        description="Generate pure-AI ReEDS load files with ramp-minimizing training-load shaving."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--reeds-input-dir", type=Path, default=None, help="Optional legacy ReEDS input directory. Default: <data-dir>.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-dir>/loaddata.")
    parser.add_argument("--renewable-generation-dir", type=Path, default=None, help="Directory containing _Baseline_renewable_generation_EIA.xlsx. Default: <data-dir>.")
    parser.add_argument("--baseline-renewable-generation-file", type=Path, default=None, help="Default: <data-dir>/_Baseline_renewable_generation_EIA.xlsx.")
    parser.add_argument("--ai-capacity-file", type=Path, default=None)
    parser.add_argument("--pue-file", type=Path, default=None)
    parser.add_argument("--pue-improvement-file", type=Path, default=None)
    parser.add_argument("--utilization-file", type=Path, default=None)
    parser.add_argument("--training-utilization-file", type=Path, default=None)
    parser.add_argument("--ai-spatial-file", type=Path, default=None)
    parser.add_argument("--ba-area-file", type=Path, default=None, help="Default: <data-dir>/BA_area.csv.")
    parser.add_argument("--base-load-file", type=Path, default=None, help="Default: <data-dir>/EER_IRAlow_load_hourly.h5.")
    parser.add_argument("--ratio-output-file", type=Path, default=None)
    parser.add_argument("--projection-number", type=positive_int, default=2, help="Projection number to process. Default: 2.")
    parser.add_argument("--no-overwrite", action="store_true")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        LOGGER.debug("Ignoring unrecognized arguments from the execution environment: %s", unknown_args)

    data_dir = args.data_dir
    reeds_input_dir = args.reeds_input_dir or data_dir
    output_dir = args.output_dir or data_dir / "loaddata"
    renewable_generation_dir = args.renewable_generation_dir or data_dir
    ba_area_file = args.ba_area_file or resolve_first_existing(
        [data_dir / "BA_area.csv", data_dir / "loaddata" / "BA_area.csv", reeds_input_dir / "loaddata" / "BA_area.csv"],
        "BA area mapping CSV",
    )
    base_load_file = args.base_load_file or resolve_first_existing(
        [data_dir / "EER_IRAlow_load_hourly.h5", data_dir / "loaddata" / "EER_IRAlow_load_hourly.h5", reeds_input_dir / "loaddata" / "EER_IRAlow_load_hourly.h5"],
        "Base ReEDS hourly load HDF5",
    )
    baseline_renewable_generation_file = args.baseline_renewable_generation_file or resolve_first_existing(
        [
            renewable_generation_dir / "_Baseline_renewable_generation_EIA.xlsx",
            renewable_generation_dir / "_Baseline" / "_Baseline_renewable_generation_EIA.xlsx",
            data_dir / "_Baseline_renewable_generation_EIA.xlsx",
            data_dir / "_Baseline" / "_Baseline_renewable_generation_EIA.xlsx",
        ],
        "Baseline renewable generation workbook",
    )
    ratio_output_file = args.ratio_output_file or data_dir / "Pure_AI_Shaving_Ratios_NetDemand.xlsx"

    return Config(
        data_dir=data_dir,
        reeds_input_dir=reeds_input_dir,
        output_dir=output_dir,
        renewable_generation_dir=renewable_generation_dir,
        baseline_renewable_generation_file=baseline_renewable_generation_file,
        ai_capacity_file=args.ai_capacity_file or data_dir / "AI_capacity_projection.xlsx",
        pue_file=args.pue_file or data_dir / "RCP4.5_2035_Base_PUE.csv",
        pue_improvement_file=args.pue_improvement_file or data_dir / "RCP4.5_2035_Base_PUE_i.csv",
        utilization_file=args.utilization_file or data_dir / "reeds_7_weather_year_utilization_profiles.xlsx",
        training_utilization_file=args.training_utilization_file or data_dir / "reeds_7_weather_year_training_utilization_profiles.xlsx",
        ai_spatial_file=args.ai_spatial_file or data_dir / "AI_Spatial.txt",
        ba_area_file=ba_area_file,
        base_load_file=base_load_file,
        ratio_output_file=ratio_output_file,
        projection_number=args.projection_number,
        overwrite=not args.no_overwrite,
    )


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------
def load_ai_capacity_projection(excel_path: Path) -> Tuple[np.ndarray, List[int]]:
    ensure_file(excel_path, "AI capacity projection workbook")
    df = pd.read_excel(excel_path)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Year", "Low-Case", "Mid-Case", "High-Case"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {excel_path.name}: {missing}")
    df = df[required].copy()
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Year"]).sort_values("Year").reset_index(drop=True)
    if len(df) != 12:
        raise ValueError(f"Expected 12 modeled years in {excel_path.name}; found {len(df)}.")
    capacity = np.vstack(
        [df["Low-Case"].to_numpy(), df["Mid-Case"].to_numpy(), df["High-Case"].to_numpy()]
    ).astype(np.float32)
    years = [int(y) for y in df["Year"].to_numpy()]
    return capacity, years


def standardize_hourly_profile_df(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    df = df.copy()
    if "hour" in df.columns:
        df = df.drop(columns=["hour"])
    df.columns = [int(c) if str(c).strip().isdigit() else c for c in df.columns]
    missing_years = [year for year in WEATHER_YEARS if year not in df.columns]
    if missing_years:
        raise ValueError(f"Missing weather-year columns {missing_years} in sheet '{sheet_name}'.")
    if len(df) != HOURS_PER_YEAR:
        raise ValueError(f"Expected {HOURS_PER_YEAR} rows in sheet '{sheet_name}', found {len(df)}.")
    return df


def load_profile_workbook(workbook_path: Path) -> Dict[str, np.ndarray]:
    ensure_file(workbook_path, "Utilization workbook")
    profiles: Dict[str, np.ndarray] = {}
    xls = pd.ExcelFile(workbook_path)
    for sheet_name in xls.sheet_names:
        if sheet_name.strip().lower() == "readme":
            continue
        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        df = standardize_hourly_profile_df(df, sheet_name)
        profile = np.zeros((N_WEATHER_YEARS, HOURS_PER_YEAR), dtype=np.float32)
        for weather_idx, weather_year in enumerate(WEATHER_YEARS):
            profile[weather_idx, :] = pd.to_numeric(df[weather_year], errors="coerce").to_numpy(dtype=np.float32)
        profiles[sheet_name] = profile
    if not profiles:
        raise ValueError(f"No usable sheets found in {workbook_path}")
    return profiles


def pair_total_and_training_profiles(
    total_profiles: Mapping[str, np.ndarray],
    training_profiles: Mapping[str, np.ndarray],
) -> List[Tuple[str, str, np.ndarray, np.ndarray]]:
    pairs: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    training_names = list(training_profiles.keys())
    for idx, (total_name, total_profile) in enumerate(total_profiles.items()):
        if total_name in training_profiles:
            training_name = total_name
        elif idx < len(training_names):
            training_name = training_names[idx]
            LOGGER.warning(
                "Training sheet matching total sheet '%s' by position using '%s'.",
                total_name,
                training_name,
            )
        else:
            raise ValueError(f"No training utilization sheet available for total sheet '{total_name}'.")
        pairs.append((total_name, training_name, total_profile, training_profiles[training_name]))
    return pairs


def load_ai_spatial_weights(ai_spatial_file: Path, n_states: int) -> np.ndarray:
    ensure_file(ai_spatial_file, "AI spatial allocation file")
    weights = np.loadtxt(ai_spatial_file, delimiter="\t", dtype=np.float32).reshape(-1)
    if len(weights) != n_states:
        raise ValueError(f"AI spatial vector has {len(weights)} entries; expected {n_states}.")
    return weights


def build_ba_mapping(ba_area_file: Path, states: Sequence[str]) -> BAMapping:
    """Build BA-to-state and BA-to-EIA-region mappings from BA_area.csv.
    """
    ensure_file(ba_area_file, "BA area mapping CSV")

    raw = pd.read_csv(ba_area_file, dtype=str, keep_default_na=False)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ValueError(f"BA mapping file is empty: {ba_area_file}")

    # Trim string content and remove fully blank rows.
    raw = raw.astype(str).apply(lambda column: column.str.strip())
    raw = raw.loc[~(raw == "").all(axis=1)].reset_index(drop=True)

    columns = list(raw.columns)
    lower_columns = {col: str(col).strip().lower().replace(" ", "_") for col in columns}
    state_to_idx = {state: i for i, state in enumerate(states)}
    state_abbrev_to_name = {
        "AL": "Alabama", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
        "FL": "Florida", "GA": "Georgia", "ID": "Idaho", "IL": "Illinois",
        "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky",
        "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts",
        "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
        "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire",
        "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", "NC": "North Carolina",
        "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
        "PA": "Pennsylvania", "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
        "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
        "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    }
    region_names = {
        "ERCOT", "FRCC", "MISO", "NPCC_NE", "NPCC_NY", "PJM", "SPP",
        "SERC_C", "SERC_E", "SERC_F", "SERC_SE", "WECC_CA", "WECC_NW", "WECC_SW",
    }

    def normalized_state(value: str) -> str | None:
        value = str(value).strip()
        if value in state_to_idx:
            return value
        return state_abbrev_to_name.get(value.upper())

    def state_match_count(col: str) -> int:
        return sum(normalized_state(v) is not None for v in raw[col].tolist())

    def numeric_count(col: str) -> int:
        return int(pd.to_numeric(raw[col], errors="coerce").notna().sum())

    def region_match_count(col: str) -> int:
        values = raw[col].astype(str).str.strip().str.upper()
        exact = values.isin(region_names).sum()
        patterned = values.str.contains("_", regex=False).sum()
        return int(exact + patterned)

    # State column: prefer explicit header, otherwise the column with most state-name matches.
    state_header_candidates = [
        col for col in columns
        if lower_columns[col] in {"state", "st", "state_name", "state_name_full", "state_full"}
        or lower_columns[col].endswith("_state")
    ]
    if state_header_candidates:
        state_col = state_header_candidates[0]
    else:
        state_col = max(columns, key=state_match_count)
        if state_match_count(state_col) == 0:
            raise ValueError(
                f"Could not identify a state column in {ba_area_file}. "
                f"Available columns: {columns}"
            )

    # Area column: prefer explicit area-like headers, otherwise the numeric column with most values.
    area_header_candidates = [
        col for col in columns
        if "area" in lower_columns[col] or lower_columns[col] in {"sq_km", "km2", "sqmi", "county_area"}
    ]
    if area_header_candidates:
        area_col = area_header_candidates[0]
    else:
        numeric_candidates = [col for col in columns if col != state_col and numeric_count(col) > 0]
        if not numeric_candidates:
            raise ValueError(
                f"Could not identify an area column in {ba_area_file}. "
                f"Available columns: {columns}"
            )
        area_col = max(numeric_candidates, key=numeric_count)

    # Region column: prefer explicit EIA/ReEDS-region headers, otherwise EIA-style values.
    region_header_candidates = [
        col for col in columns
        if col not in {state_col, area_col}
        and (
            "nercr" in lower_columns[col]
            or "eia" in lower_columns[col]
            or "region" in lower_columns[col]
            or lower_columns[col] in {"r", "reeds_region", "eia_region", "reeds_ba_region"}
        )
    ]
    if region_header_candidates:
        # Avoid accidentally using a BA-name column whose header is simply "region" by
        # selecting the candidate with the most EIA-style values.
        region_col = max(region_header_candidates, key=region_match_count)
    else:
        region_candidates = [col for col in columns if col not in {state_col, area_col} and region_match_count(col) > 0]
        if region_candidates:
            region_col = max(region_candidates, key=region_match_count)
        elif len(columns) > 4:
            region_col = columns[4]  # old BA_area_2.csv fallback
        elif len(columns) > 2:
            region_col = columns[2]
        else:
            raise ValueError(
                f"Could not identify an EIA region column in {ba_area_file}. "
                f"Available columns: {columns}"
            )

    # BA/name column: prefer explicit BA-like headers; otherwise use the first non-state/non-area/non-region column.
    ba_header_candidates = [
        col for col in columns
        if col not in {state_col, area_col, region_col}
        and (
            lower_columns[col] in {"ba", "balancing_area", "ba_name", "reeds_ba", "name"}
            or "balancing" in lower_columns[col]
        )
    ]
    if ba_header_candidates:
        ba_col = ba_header_candidates[0]
    else:
        fallback_cols = [col for col in columns if col not in {state_col, area_col, region_col}]
        ba_col = fallback_cols[0] if fallback_cols else columns[0]

    LOGGER.info(
        "BA mapping columns detected: BA='%s', state='%s', area='%s', EIA region='%s'",
        ba_col,
        state_col,
        area_col,
        region_col,
    )

    n_ba = len(raw)
    ba_names = raw[ba_col].astype(str).str.strip().to_numpy(dtype=object)
    ba_state_ids = np.full(n_ba, -1, dtype=int)
    region_ids = raw[region_col].astype(str).str.strip().to_numpy(dtype=object)
    area_ba = pd.to_numeric(raw[area_col], errors="coerce").to_numpy(dtype=np.float32)

    if np.isnan(area_ba).any():
        bad_rows = np.where(np.isnan(area_ba))[0][:10]
        raise ValueError(
            f"Area column '{area_col}' in {ba_area_file} contains nonnumeric values. "
            f"Example zero-based rows: {bad_rows.tolist()}"
        )

    states_area = np.zeros(len(states), dtype=np.float32)
    for idx, value in enumerate(raw[state_col].tolist()):
        state_name = normalized_state(value)
        if state_name is None:
            continue
        state_idx = state_to_idx[state_name]
        ba_state_ids[idx] = state_idx
        states_area[state_idx] += area_ba[idx]

    valid_mask = ba_state_ids >= 0
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        raise ValueError(f"No BA rows could be matched to the configured state list in {ba_area_file}.")

    ratio = np.zeros(n_ba, dtype=np.float32)
    denominator = states_area[ba_state_ids[valid_mask]]
    if np.any(denominator <= 0):
        raise ValueError("At least one valid state has zero mapped BA area.")
    ratio[valid_mask] = area_ba[valid_mask] / denominator
    region_ids[~valid_mask] = "-1"

    return BAMapping(
        ba_names=ba_names,
        ba_state_ids=ba_state_ids,
        region_ids=region_ids,
        ba_state_ratio=ratio,
        valid_ba_indices=valid_indices,
    )


def build_region_pue_lookup(
    pue_file: Path,
    pue_improvement_file: Path,
    regions: Iterable[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ensure_file(pue_file, "Base PUE CSV")
    ensure_file(pue_improvement_file, "Improved PUE CSV")
    df_pue = pd.read_csv(pue_file)
    df_pue_improved = pd.read_csv(pue_improvement_file)
    pue_base: Dict[str, np.ndarray] = {}
    pue_improved: Dict[str, np.ndarray] = {}
    for region in sorted({str(r) for r in regions if str(r) not in {"-1", "", "nan"}}):
        if region not in df_pue.columns or region not in df_pue_improved.columns:
            raise ValueError(f"Region '{region}' is missing from one or both PUE files.")
        base = pd.to_numeric(df_pue[region], errors="coerce").to_numpy(dtype=np.float32)
        improved = pd.to_numeric(df_pue_improved[region], errors="coerce").to_numpy(dtype=np.float32)
        if len(base) != HOURS_PER_YEAR or len(improved) != HOURS_PER_YEAR:
            raise ValueError(f"PUE profile for region '{region}' must have {HOURS_PER_YEAR} rows.")
        pue_base[region] = base
        pue_improved[region] = improved
    return pue_base, pue_improved


def validate_base_load_file(base_load_file: Path, required_years: Sequence[int], n_ba: int) -> None:
    ensure_file(base_load_file, "Base load HDF5")
    with h5py.File(base_load_file, "r") as h5:
        for year in required_years:
            key = str(year)
            if key not in h5:
                raise KeyError(f"Dataset '{key}' is missing from {base_load_file}")
            if h5[key].shape != (N_TOTAL_HOURS, n_ba):
                raise ValueError(
                    f"Dataset {key} should have shape ({N_TOTAL_HOURS}, {n_ba}); got {h5[key].shape}."
                )


# -----------------------------------------------------------------------------
# Renewable generation workbook readers used by the shaving optimization
# -----------------------------------------------------------------------------
def load_region_renewable_profile(
    workbook_path: Path,
    region_name: str,
    target_year: int,
    weather_years: Sequence[int],
) -> np.ndarray:
    ensure_file(workbook_path, "Renewable generation workbook")
    df = pd.read_excel(workbook_path, sheet_name=region_name, header=[0, 1])
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    out = np.zeros((len(weather_years), HOURS_PER_YEAR), dtype=np.float32)
    for weather_idx, weather_year in enumerate(weather_years):
        matching_col = None
        for col in df.columns:
            if str(col[0]).strip() == str(target_year) and str(col[1]).strip() == str(weather_year):
                matching_col = col
                break
        if matching_col is None:
            raise ValueError(
                f"Missing renewable column ({target_year}, {weather_year}) in {workbook_path.name}, sheet {region_name}."
            )
        values = pd.to_numeric(df[matching_col], errors="coerce").to_numpy(dtype=np.float32)
        values = values[~np.isnan(values)]
        if len(values) < HOURS_PER_YEAR:
            raise ValueError(
                f"Renewable series too short in {workbook_path.name}, sheet {region_name}, "
                f"year={target_year}, weather={weather_year}: found {len(values)} rows."
            )
        out[weather_idx, :] = values[:HOURS_PER_YEAR]
    return out


def build_system_renewable_total_from_workbook(
    workbook_path: Path,
    region_names: Iterable[str],
    target_year: int,
    weather_years: Sequence[int],
) -> np.ndarray:
    total = np.zeros((len(weather_years), HOURS_PER_YEAR), dtype=np.float32)
    for region in sorted({str(r) for r in region_names}):
        if region in {"-1", "Summary", "", "nan"}:
            continue
        total += load_region_renewable_profile(workbook_path, region, target_year, weather_years)
    return total


def get_pure_ai_run_name(projection_number: int, ai_scenario_idx: int) -> str:
    return f"_Proj{projection_number}_AI{ai_scenario_idx}"


# -----------------------------------------------------------------------------
# Shaving optimization
# -----------------------------------------------------------------------------
def optimize_daily_shave_amount(
    total_load_day: np.ndarray,
    max_shave_day: np.ndarray,
    max_hours: int = 4,
    tol: float = 1.0e-8,
) -> np.ndarray:
    total_load_day = np.asarray(total_load_day, dtype=np.float32).reshape(-1)
    max_shave_day = np.asarray(max_shave_day, dtype=np.float32).reshape(-1)
    if len(total_load_day) != 24:
        raise ValueError(f"Expected 24 hourly values in one day; found {len(total_load_day)}.")

    def evaluate(shave_amount: np.ndarray) -> Tuple[float, float, int, float]:
        adjusted = total_load_day - shave_amount
        ramps = np.abs(np.diff(adjusted))
        max_ramp = float(ramps.max()) if len(ramps) > 0 else 0.0
        sum_ramp = float(ramps.sum())
        used_hours = int(np.count_nonzero(shave_amount > tol))
        total_shave = float(shave_amount.sum())
        return max_ramp, sum_ramp, used_hours, total_shave

    base_ramps = np.abs(np.diff(total_load_day)).astype(np.float32)
    top_k = min(CANDIDATE_RAMP_COUNT, len(base_ramps))
    top_idx = np.argsort(base_ramps)[-top_k:] if top_k > 0 else np.array([], dtype=int)

    candidate_hours = set()
    for ramp_idx in top_idx.tolist():
        candidate_hours.add(int(ramp_idx))
        candidate_hours.add(int(ramp_idx + 1))

    if np.any(max_shave_day > tol):
        cap_threshold = max(0.10 * float(max_shave_day.max()), tol)
        candidate_hours.update(int(h) for h in np.where(max_shave_day >= cap_threshold)[0].tolist())

    candidate_hours = sorted(h for h in candidate_hours if 0 <= h < 24 and max_shave_day[h] > tol)
    if not candidate_hours:
        return np.zeros(24, dtype=np.float32)

    line_fracs = np.linspace(0.0, 1.0, LINE_SEARCH_STEPS, dtype=np.float32)
    shave = np.zeros(24, dtype=np.float32)
    best_obj = evaluate(shave)

    available_hours = candidate_hours.copy()
    while np.count_nonzero(shave > tol) < max_hours and available_hours:
        best_hour = None
        best_hour_amount = 0.0
        best_hour_obj = best_obj

        for hour in available_hours:
            local_best_amount = 0.0
            local_best_obj = best_obj
            for frac in line_fracs:
                amount = float(frac * max_shave_day[hour])
                trial = shave.copy()
                trial[hour] = amount
                obj = evaluate(trial)
                if obj < local_best_obj:
                    local_best_obj = obj
                    local_best_amount = amount
            if local_best_obj < best_hour_obj:
                best_hour_obj = local_best_obj
                best_hour = hour
                best_hour_amount = local_best_amount

        if best_hour is None:
            break
        shave[best_hour] = best_hour_amount
        best_obj = best_hour_obj
        available_hours.remove(best_hour)

    for _ in range(LOCAL_SEARCH_ITERS):
        used_hours = [h for h in range(24) if shave[h] > tol]
        for hour in used_hours:
            local_best_amount = float(shave[hour])
            local_best_obj = best_obj
            for frac in line_fracs:
                amount = float(frac * max_shave_day[hour])
                trial = shave.copy()
                trial[hour] = amount
                obj = evaluate(trial)
                if obj < local_best_obj:
                    local_best_obj = obj
                    local_best_amount = amount
            shave[hour] = local_best_amount
            best_obj = local_best_obj

        used_hours = [h for h in range(24) if shave[h] > tol]
        unused_candidates = [h for h in candidate_hours if shave[h] <= tol]
        improved = False
        for hour_out in used_hours:
            base_trial = shave.copy()
            base_trial[hour_out] = 0.0
            for hour_in in unused_candidates:
                local_best_amount = 0.0
                local_best_obj = best_obj
                for frac in line_fracs:
                    amount = float(frac * max_shave_day[hour_in])
                    trial = base_trial.copy()
                    trial[hour_in] = amount
                    obj = evaluate(trial)
                    if obj < local_best_obj:
                        local_best_obj = obj
                        local_best_amount = amount
                if local_best_obj < best_obj:
                    shave[hour_out] = 0.0
                    shave[hour_in] = local_best_amount
                    best_obj = local_best_obj
                    improved = True
                    break
            if improved:
                break

    return shave.astype(np.float32)


def optimize_shave_amount_by_day(total_load: np.ndarray, max_shave: np.ndarray, max_hours: int = 4) -> np.ndarray:
    total_load = np.asarray(total_load, dtype=np.float32).reshape(-1)
    max_shave = np.asarray(max_shave, dtype=np.float32).reshape(-1)
    if total_load.size != max_shave.size:
        raise ValueError(f"Series lengths differ: {total_load.size} and {max_shave.size}.")
    if total_load.size % 24 != 0:
        raise ValueError(f"Hourly series length must be a multiple of 24; found {total_load.size}.")
    n_days = total_load.size // 24
    shaved = np.zeros((n_days, 24), dtype=np.float32)
    total_matrix = total_load.reshape(n_days, 24)
    shave_matrix = max_shave.reshape(n_days, 24)
    for day_idx in range(n_days):
        shaved[day_idx, :] = optimize_daily_shave_amount(total_matrix[day_idx], shave_matrix[day_idx], max_hours=max_hours)
    return shaved.reshape(-1)


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------
def load_base_system_total_by_year(base_load_file: Path, target_years: Sequence[int], n_ba: int) -> Dict[int, np.ndarray]:
    validate_base_load_file(base_load_file, target_years, n_ba)
    totals: Dict[int, np.ndarray] = {}
    with h5py.File(base_load_file, "r") as h5:
        for year in target_years:
            load = h5[str(year)][:]
            load = load.reshape(N_WEATHER_YEARS, HOURS_PER_YEAR, n_ba)
            totals[year] = load.sum(axis=2).astype(np.float32)
    return totals


def build_system_unit_ai_load(
    states_demand: np.ndarray,
    ba_mapping: BAMapping,
    pue_base: Mapping[str, np.ndarray],
    pue_improved: Mapping[str, np.ndarray],
) -> np.ndarray:
    n_years = states_demand.shape[0]
    system_unit_ai_load = np.zeros((n_years, N_TOTAL_HOURS), dtype=np.float32)
    pue_improvement_weight = np.array([0.05 * (1.2 ** year_idx) for year_idx in range(n_years)], dtype=np.float32)
    pue_base_weight = 1.0 - pue_improvement_weight

    for ba_idx in ba_mapping.valid_ba_indices:
        state_idx = ba_mapping.ba_state_ids[ba_idx]
        region = str(ba_mapping.region_ids[ba_idx])
        area_ratio = ba_mapping.ba_state_ratio[ba_idx]
        pue_year_hour = (
            pue_base_weight[:, None] * pue_base[region][None, :]
            + pue_improvement_weight[:, None] * pue_improved[region][None, :]
        )
        pue_year_hour_all_weather = np.repeat(
            pue_year_hour[:, None, :], N_WEATHER_YEARS, axis=1
        ).reshape(n_years, N_TOTAL_HOURS)
        installed_ai_capacity = (states_demand[:, state_idx] * area_ratio).astype(np.float32)
        system_unit_ai_load += installed_ai_capacity[:, None] * pue_year_hour_all_weather
    return system_unit_ai_load


def build_ba_demand_for_adjusted_profile(
    states_demand: np.ndarray,
    adjusted_profile_flat: np.ndarray,
    modeled_year: int,
    ba_mapping: BAMapping,
    pue_base: Mapping[str, np.ndarray],
    pue_improved: Mapping[str, np.ndarray],
) -> np.ndarray:
    year_idx = TARGET_YEAR_TO_INDEX[modeled_year]
    n_ba = len(ba_mapping.ba_names)
    ba_demand_year = np.zeros((n_ba, N_TOTAL_HOURS), dtype=np.float32)
    pue_improvement_weight = np.float32(0.05 * (1.2 ** year_idx))
    pue_base_weight = np.float32(1.0 - pue_improvement_weight)

    for ba_idx in ba_mapping.valid_ba_indices:
        state_idx = ba_mapping.ba_state_ids[ba_idx]
        region = str(ba_mapping.region_ids[ba_idx])
        area_ratio = ba_mapping.ba_state_ratio[ba_idx]
        pue_hour = (pue_base_weight * pue_base[region] + pue_improvement_weight * pue_improved[region]).astype(np.float32)
        pue_hour_all_weather = np.repeat(pue_hour[None, :], N_WEATHER_YEARS, axis=0).reshape(N_TOTAL_HOURS)
        installed_ai_capacity = float(states_demand[year_idx, state_idx] * area_ratio)
        ba_demand_year[ba_idx, :] = installed_ai_capacity * pue_hour_all_weather * adjusted_profile_flat
    return ba_demand_year


def write_shaved_load_file(
    base_load_file: Path,
    output_file: Path,
    ba_demand_by_year: Mapping[int, np.ndarray],
    overwrite: bool,
) -> None:
    if output_file.exists():
        if overwrite:
            output_file.unlink()
        else:
            LOGGER.info("Skipping existing file because --no-overwrite was used: %s", output_file.name)
            return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_load_file, output_file)
    with h5py.File(output_file, "r+") as h5:
        for modeled_year, ba_demand in ba_demand_by_year.items():
            h5[str(modeled_year)][:, :] = h5[str(modeled_year)][:, :] + ba_demand.T
    LOGGER.info("Saved %s", output_file)


def run(config: Config) -> None:
    projection_idx = config.projection_number - 1
    if projection_idx not in (0, 1, 2):
        raise ValueError("projection_number must be 1, 2, or 3 for the provided AI capacity workbook structure.")

    LOGGER.info("Loading inputs")
    ai_capacity, modeled_years = load_ai_capacity_projection(config.ai_capacity_file)
    for target_year in TARGET_MODELED_YEARS:
        if target_year not in modeled_years:
            raise ValueError(f"Target modeled year {target_year} is missing from AI capacity workbook.")

    total_profiles = load_profile_workbook(config.utilization_file)
    training_profiles = load_profile_workbook(config.training_utilization_file)
    profile_pairs = pair_total_and_training_profiles(total_profiles, training_profiles)

    ai_spatial_weights = load_ai_spatial_weights(config.ai_spatial_file, len(STATES))
    ba_mapping = build_ba_mapping(config.ba_area_file, STATES)
    base_total_by_year = load_base_system_total_by_year(config.base_load_file, TARGET_MODELED_YEARS, len(ba_mapping.ba_names))
    pue_base, pue_improved = build_region_pue_lookup(
        config.pue_file,
        config.pue_improvement_file,
        ba_mapping.region_ids[ba_mapping.valid_ba_indices],
    )

    load_data_mw = ai_capacity * 1.0e3
    states_demand_all = load_data_mw[:, :, None] * ai_spatial_weights[None, None, :]
    states_demand = states_demand_all[projection_idx]
    system_unit_ai_load = build_system_unit_ai_load(states_demand, ba_mapping, pue_base, pue_improved)
    report_regions = sorted({str(r) for r in ba_mapping.region_ids[ba_mapping.valid_ba_indices]})

    ratio_tables: Dict[str, pd.DataFrame] = {}
    for ai_scenario_idx, (total_name, training_name, total_profile, training_profile) in enumerate(profile_pairs, start=1):
        LOGGER.info("Processing AI%d | total='%s' | training='%s'", ai_scenario_idx, total_name, training_name)
        total_util_flat = total_profile.reshape(N_TOTAL_HOURS).astype(np.float32)
        training_util_flat = training_profile.reshape(N_TOTAL_HOURS).astype(np.float32)
        annual_total_den = float(total_util_flat.sum())
        annual_training_den = float(training_util_flat.sum())

        run_name = get_pure_ai_run_name(config.projection_number, ai_scenario_idx)
        workbook_path = config.baseline_renewable_generation_file
        ensure_file(workbook_path, "Baseline renewable generation workbook")

        for reduction_label, reduction_fraction in REDUCTION_SCENARIOS.items():
            output_file = config.output_dir / (
                f"EER_IRAlow_AI{ai_scenario_idx}_proj{config.projection_number}_{reduction_label}_load_hourly.h5"
            )
            adjusted_profile_by_year: Dict[int, np.ndarray] = {}
            ratio_rows: List[Dict[str, float | int | str]] = []

            for modeled_year in TARGET_MODELED_YEARS:
                year_idx = TARGET_YEAR_TO_INDEX[modeled_year]
                base_total_flat = base_total_by_year[modeled_year].reshape(N_TOTAL_HOURS).astype(np.float32)
                renewable_total = build_system_renewable_total_from_workbook(
                    workbook_path,
                    report_regions,
                    modeled_year,
                    WEATHER_YEARS,
                ).reshape(N_TOTAL_HOURS).astype(np.float32)

                original_ai_total = system_unit_ai_load[year_idx, :] * total_util_flat
                max_shave_total = system_unit_ai_load[year_idx, :] * reduction_fraction * training_util_flat
                net_demand_before_shave = np.maximum(base_total_flat + original_ai_total - renewable_total, 0.0).astype(np.float32)
                shaved_total_flat = optimize_shave_amount_by_day(net_demand_before_shave, max_shave_total, max_hours=4)

                shaved_util_flat = np.zeros(N_TOTAL_HOURS, dtype=np.float32)
                nonzero = system_unit_ai_load[year_idx, :] > 1.0e-8
                shaved_util_flat[nonzero] = shaved_total_flat[nonzero] / system_unit_ai_load[year_idx, nonzero]
                shaved_util_flat = np.minimum(shaved_util_flat, reduction_fraction * training_util_flat)
                adjusted_profile_by_year[modeled_year] = np.clip(total_util_flat - shaved_util_flat, 0.0, None)

                ratio_train = float(shaved_util_flat.sum()) / annual_training_den if annual_training_den > 0 else 0.0
                ratio_total = float(shaved_util_flat.sum()) / annual_total_den if annual_total_den > 0 else 0.0
                for region in report_regions:
                    ratio_rows.append(
                        {
                            "Region": region,
                            "Year": int(modeled_year),
                            "Shaved_over_Training_Utilization": ratio_train,
                            "Shaved_over_Total_Utilization": ratio_total,
                        }
                    )

                shaved_hours = int(np.count_nonzero(shaved_total_flat > 1.0e-8))
                LOGGER.info(
                    "Proj%d | AI%d | %s | %d: %d shaved hours selected",
                    config.projection_number,
                    ai_scenario_idx,
                    reduction_label,
                    modeled_year,
                    shaved_hours,
                )

            ba_demand_by_year = {
                modeled_year: build_ba_demand_for_adjusted_profile(
                    states_demand,
                    adjusted_profile_by_year[modeled_year],
                    modeled_year,
                    ba_mapping,
                    pue_base,
                    pue_improved,
                )
                for modeled_year in TARGET_MODELED_YEARS
            }
            write_shaved_load_file(config.base_load_file, output_file, ba_demand_by_year, config.overwrite)

            sheet_name = f"Proj{config.projection_number}_AI{ai_scenario_idx}_{reduction_label}"[:31]
            ratio_tables[sheet_name] = pd.DataFrame(sorted(ratio_rows, key=lambda row: (row["Year"], row["Region"])))

    config.ratio_output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(config.ratio_output_file, engine="openpyxl") as writer:
        for sheet_name, table in ratio_tables.items():
            table.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    LOGGER.info("Saved ratio summary workbook: %s", config.ratio_output_file)


def main() -> None:
    configure_logging()
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
