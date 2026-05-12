#!/usr/bin/env python3
r"""
Generate ReEDS hourly load files for AI campus cases with behind-the-meter
renewable or nuclear generation.

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
NUCLEAR_CAPACITY_FACTOR = 0.92

AI_SCENARIO_SHEET_MAP: Mapping[str, str] = {
    "Baseline_E": "Baseline E",
    "High_API_D": "High API D",
    "High_API_F": "High API F",
    "High_Conversation_A": "High Conversation A",
    "High_Conversation_H": "High Conversation H",
    "Low_Training_H": "Low Training H",
    "High_Training_C": "High Training C",
    "High_Conversation_F": "High Conversation F",
    "Low_Training_B": "Low Training B",
}

AI_SCENARIO_ORDER: Sequence[str] = (
    "Baseline_E",
    "High_API_D",
    "High_API_F",
    "High_Conversation_A",
    "High_Conversation_H",
    "Low_Training_H",
    "High_Training_C",
    "High_Conversation_F",
    "Low_Training_B",
)

AI_NAME_TO_LABEL: Mapping[str, str] = {
    name: f"AI{idx + 1}" for idx, name in enumerate(AI_SCENARIO_ORDER)
}


# includes all nine AI computing profiles.
PROJ_TO_AI_SCENARIOS: Mapping[int, Sequence[str]] = {
    0: ("High_API_F", "High_Conversation_H", "High_Conversation_A", "Low_Training_H"),
    1: AI_SCENARIO_ORDER,
    2: ("High_API_F", "Low_Training_H", "Baseline_E", "Low_Training_H"),
}

RENEWABLE_SCENARIOS: Sequence[Mapping[str, float | str]] = (
    {"label": "Wind25", "wind": 0.25, "solar": 0.00, "nuclear": 0.00},
    {"label": "Wind50", "wind": 0.50, "solar": 0.00, "nuclear": 0.00},
    {"label": "Solar25", "wind": 0.00, "solar": 0.25, "nuclear": 0.00},
    {"label": "Solar50", "wind": 0.00, "solar": 0.50, "nuclear": 0.00},
    {"label": "Wind25_Solar25", "wind": 0.25, "solar": 0.25, "nuclear": 0.00},
    {"label": "Wind25_Nuclear10", "wind": 0.25, "solar": 0.00, "nuclear": 0.10},
    {"label": "Solar25_Nuclear10", "wind": 0.00, "solar": 0.25, "nuclear": 0.10},
    {"label": "Nuclear10", "wind": 0.00, "solar": 0.00, "nuclear": 0.10},
    {"label": "Nuclear30", "wind": 0.00, "solar": 0.00, "nuclear": 0.30},
)


@dataclass(frozen=True)
class Config:
    data_dir: Path
    reeds_input_dir: Path
    output_dir: Path
    ai_capacity_file: Path
    pue_file: Path
    pue_improvement_file: Path
    utilization_file: Path
    ai_spatial_file: Path
    ba_area_file: Path
    base_load_file: Path
    solar_cf_file: Path
    wind_cf_file: Path
    overwrite: bool = True
    rename_legacy_files: bool = False


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
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"{description} not found. Checked:\n  {joined}")


def parse_args() -> Config:
    default_data_dir = "Data Path" # Replace your path here
    parser = argparse.ArgumentParser(
        description="Generate ReEDS hourly load files for AI campus cases."
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--reeds-input-dir", type=Path, default=None, help="Optional legacy ReEDS input directory. Default: <data-dir>.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <data-dir>/loaddata.")
    parser.add_argument("--ai-capacity-file", type=Path, default=None)
    parser.add_argument("--pue-file", type=Path, default=None)
    parser.add_argument("--pue-improvement-file", type=Path, default=None)
    parser.add_argument("--utilization-file", type=Path, default=None)
    parser.add_argument("--ai-spatial-file", type=Path, default=None)
    parser.add_argument("--ba-area-file", type=Path, default=None, help="Default: <data-dir>/BA_area.csv.")
    parser.add_argument("--base-load-file", type=Path, default=None, help="Default: <data-dir>/EER_IRAlow_load_hourly.h5.")
    parser.add_argument("--solar-cf-file", type=Path, default=None)
    parser.add_argument("--wind-cf-file", type=Path, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--rename-legacy-files",
        action="store_true",
        help="Optionally rename old files matching *_renewable*_proj* to descriptive renewable labels.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        LOGGER.debug("Ignoring unrecognized arguments from the execution environment: %s", unknown_args)

    data_dir = args.data_dir
    reeds_input_dir = args.reeds_input_dir or data_dir
    output_dir = args.output_dir or data_dir / "loaddata"
    ba_area_file = args.ba_area_file or resolve_first_existing(
        [data_dir / "BA_area.csv", data_dir / "loaddata" / "BA_area.csv", reeds_input_dir / "loaddata" / "BA_area.csv"],
        "BA area mapping CSV",
    )
    base_load_file = args.base_load_file or resolve_first_existing(
        [data_dir / "EER_IRAlow_load_hourly.h5", data_dir / "loaddata" / "EER_IRAlow_load_hourly.h5", reeds_input_dir / "loaddata" / "EER_IRAlow_load_hourly.h5"],
        "Base ReEDS hourly load HDF5",
    )
    solar_cf_file = args.solar_cf_file or resolve_first_existing(
        [
            data_dir / "upv-reference_ba_adjusted_regional.h5",
            data_dir / "upv-reference_ba_adjusted.h5",
            data_dir / "cf_upv_reference_ba_adjusted.h5",
            data_dir / "loaddata" / "upv-reference_ba_adjusted_regional.h5",
            reeds_input_dir / "upv-reference_ba_adjusted_regional.h5",
            reeds_input_dir / "upv-reference_ba_adjusted.h5",
            reeds_input_dir / "cf_upv_reference_ba_adjusted.h5",
        ],
        "Solar capacity-factor HDF5",
    )
    wind_cf_file = args.wind_cf_file or resolve_first_existing(
        [
            data_dir / "wind-ons-reference_ba_adjusted_regional.h5",
            data_dir / "wind-ons-reference_ba_adjusted.h5",
            data_dir / "cf_wind-ons_open_ba_adjusted.h5",
            data_dir / "loaddata" / "wind-ons-reference_ba_adjusted_regional.h5",
            reeds_input_dir / "wind-ons-reference_ba_adjusted_regional.h5",
            reeds_input_dir / "wind-ons-reference_ba_adjusted.h5",
            reeds_input_dir / "cf_wind-ons_open_ba_adjusted.h5",
        ],
        "Wind capacity-factor HDF5",
    )

    return Config(
        data_dir=data_dir,
        reeds_input_dir=reeds_input_dir,
        output_dir=output_dir,
        ai_capacity_file=args.ai_capacity_file or data_dir / "AI_capacity_projection.xlsx",
        pue_file=args.pue_file or data_dir / "RCP4.5_2035_Base_PUE.csv",
        pue_improvement_file=args.pue_improvement_file or data_dir / "RCP4.5_2035_Base_PUE_i.csv",
        utilization_file=args.utilization_file or data_dir / "reeds_7_weather_year_utilization_profiles.xlsx",
        ai_spatial_file=args.ai_spatial_file or data_dir / "AI_Spatial.txt",
        ba_area_file=ba_area_file,
        base_load_file=base_load_file,
        solar_cf_file=solar_cf_file,
        wind_cf_file=wind_cf_file,
        overwrite=not args.no_overwrite,
        rename_legacy_files=args.rename_legacy_files,
    )


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------
def decode_h5_strings(values: np.ndarray) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def load_adjusted_cf_h5(h5_path: Path, max_rows: int = N_TOTAL_HOURS) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    ensure_file(h5_path, "Capacity-factor HDF5")
    with h5py.File(h5_path, "r") as h5:
        if "columns" not in h5:
            raise ValueError(f"Dataset 'columns' is missing from {h5_path}")
        columns = decode_h5_strings(h5["columns"][:])
        if "data" in h5:
            data = h5["data"][:max_rows, :]
        elif "cf" in h5:
            data = h5["cf"][:max_rows, :]
        else:
            raise ValueError(f"Expected dataset 'data' or 'cf' in {h5_path}")

    if data.shape[0] != max_rows:
        raise ValueError(f"Expected {max_rows} rows in {h5_path}; found {data.shape[0]}.")
    return data.astype(np.float32), columns, {ba: idx for idx, ba in enumerate(columns)}


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
    ensure_file(workbook_path, "AI utilization workbook")
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
    return profiles


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


def validate_base_load_file(base_load_file: Path, modeled_years: Sequence[int], n_ba: int) -> None:
    ensure_file(base_load_file, "Base load HDF5")
    with h5py.File(base_load_file, "r") as h5:
        for year in modeled_years:
            key = str(year)
            if key not in h5:
                raise KeyError(f"Dataset '{key}' is missing from {base_load_file}")
            if h5[key].shape != (N_TOTAL_HOURS, n_ba):
                raise ValueError(
                    f"Dataset {key} should have shape ({N_TOTAL_HOURS}, {n_ba}); got {h5[key].shape}."
                )


# -----------------------------------------------------------------------------
# Core calculations
# -----------------------------------------------------------------------------
def build_ai_gross_demand(
    states_demand: np.ndarray,
    utilization_profile: np.ndarray,
    ba_mapping: BAMapping,
    pue_base: Dict[str, np.ndarray],
    pue_improved: Dict[str, np.ndarray],
) -> np.ndarray:
    n_years = states_demand.shape[0]
    n_ba = len(ba_mapping.ba_names)
    ba_demand = np.zeros((n_ba, n_years, N_TOTAL_HOURS), dtype=np.float32)
    profile_flat = utilization_profile.reshape(N_TOTAL_HOURS).astype(np.float32)
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
        demand_year = (states_demand[:, state_idx] * area_ratio).astype(np.float32)
        ba_demand[ba_idx, :, :] = demand_year[:, None] * profile_flat[None, :] * pue_year_hour_all_weather
    return ba_demand


def build_ai_net_demand_after_onsite_generation(
    gross_ba_demand: np.ndarray,
    states_demand: np.ndarray,
    ba_mapping: BAMapping,
    wind_cf_data: np.ndarray,
    wind_ba_to_col: Mapping[str, int],
    solar_cf_data: np.ndarray,
    solar_ba_to_col: Mapping[str, int],
    renewable_scenario: Mapping[str, float | str],
) -> np.ndarray:
    net_ba_demand = np.zeros_like(gross_ba_demand, dtype=np.float32)
    nuclear_cf = np.full(N_TOTAL_HOURS, NUCLEAR_CAPACITY_FACTOR, dtype=np.float32)
    wind_share = float(renewable_scenario["wind"])
    solar_share = float(renewable_scenario["solar"])
    nuclear_share = float(renewable_scenario["nuclear"])

    for ba_idx in ba_mapping.valid_ba_indices:
        ba_name = str(ba_mapping.ba_names[ba_idx]).strip()
        state_idx = ba_mapping.ba_state_ids[ba_idx]
        area_ratio = ba_mapping.ba_state_ratio[ba_idx]

        wind_cf = wind_cf_data[:, wind_ba_to_col[ba_name]].astype(np.float32) if ba_name in wind_ba_to_col else np.zeros(N_TOTAL_HOURS, dtype=np.float32)
        solar_cf = solar_cf_data[:, solar_ba_to_col[ba_name]].astype(np.float32) if ba_name in solar_ba_to_col else np.zeros(N_TOTAL_HOURS, dtype=np.float32)
        installed_ai_capacity = (states_demand[:, state_idx] * area_ratio).astype(np.float32)

        onsite_generation = (
            installed_ai_capacity[:, None] * wind_share * wind_cf[None, :]
            + installed_ai_capacity[:, None] * solar_share * solar_cf[None, :]
            + installed_ai_capacity[:, None] * nuclear_share * nuclear_cf[None, :]
        )
        net_ba_demand[ba_idx, :, :] = np.maximum(gross_ba_demand[ba_idx, :, :] - onsite_generation, 0.0)
    return net_ba_demand


def write_load_file(
    base_load_file: Path,
    output_file: Path,
    ba_demand: np.ndarray,
    modeled_years: Sequence[int],
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
        for year_idx, year in enumerate(modeled_years):
            h5[str(year)][:, :] = h5[str(year)][:, :] + ba_demand[:, year_idx, :].T
    LOGGER.info("Saved %s", output_file)


def rename_legacy_renewable_files(output_dir: Path) -> None:
    rename_map = {
        "renewable1": "wind25",
        "renewable2": "wind50",
        "renewable3": "solar25",
        "renewable4": "solar50",
        "renewable5": "wind25_solar25",
        "renewable6": "wind25_nuclear10",
        "renewable7": "solar25_nuclear10",
        "renewable8": "nuclear10",
        "renewable9": "nuclear30",
    }
    renamed = 0
    for old_file in sorted(output_dir.glob("EER_IRAlow_AI*_renewable*_proj*_load_hourly.h5")):
        old_name = old_file.name
        new_name = old_name
        for old_tag, new_tag in rename_map.items():
            token = f"_{old_tag}_"
            if token in old_name:
                new_name = old_name.replace(token, f"_{new_tag}_")
                break
        if new_name == old_name:
            continue
        new_file = old_file.with_name(new_name)
        if new_file.exists():
            LOGGER.info("Skipped rename because target exists: %s", new_file.name)
            continue
        old_file.rename(new_file)
        renamed += 1
        LOGGER.info("Renamed %s -> %s", old_name, new_name)
    LOGGER.info("Legacy rename step completed: %d files renamed", renamed)


def run(config: Config) -> None:
    LOGGER.info("Loading inputs")
    ai_capacity, modeled_years = load_ai_capacity_projection(config.ai_capacity_file)
    utilization_profiles = load_profile_workbook(config.utilization_file)
    ai_spatial_weights = load_ai_spatial_weights(config.ai_spatial_file, len(STATES))
    ba_mapping = build_ba_mapping(config.ba_area_file, STATES)
    validate_base_load_file(config.base_load_file, modeled_years, len(ba_mapping.ba_names))
    pue_base, pue_improved = build_region_pue_lookup(
        config.pue_file,
        config.pue_improvement_file,
        ba_mapping.region_ids[ba_mapping.valid_ba_indices],
    )
    solar_cf_data, _, solar_ba_to_col = load_adjusted_cf_h5(config.solar_cf_file, N_TOTAL_HOURS)
    wind_cf_data, _, wind_ba_to_col = load_adjusted_cf_h5(config.wind_cf_file, N_TOTAL_HOURS)

    for canonical_name, sheet_name in AI_SCENARIO_SHEET_MAP.items():
        if sheet_name not in utilization_profiles:
            raise ValueError(f"Required sheet '{sheet_name}' for scenario '{canonical_name}' is missing.")

    load_data_mw = ai_capacity * 1.0e3
    states_demand_all = load_data_mw[:, :, None] * ai_spatial_weights[None, None, :]

    generated_files: List[str] = []
    for projection_idx, ai_scenarios in PROJ_TO_AI_SCENARIOS.items():
        projection_label = f"proj{projection_idx + 1}"
        states_demand = states_demand_all[projection_idx]
        for canonical_ai_name in ai_scenarios:
            ai_label = AI_NAME_TO_LABEL[canonical_ai_name]
            sheet_name = AI_SCENARIO_SHEET_MAP[canonical_ai_name]
            LOGGER.info("Processing %s | %s | sheet '%s'", projection_label, ai_label, sheet_name)
            gross_ba_demand = build_ai_gross_demand(
                states_demand=states_demand,
                utilization_profile=utilization_profiles[sheet_name],
                ba_mapping=ba_mapping,
                pue_base=pue_base,
                pue_improved=pue_improved,
            )
            for renewable in RENEWABLE_SCENARIOS:
                renewable_label = str(renewable["label"])
                net_ba_demand = build_ai_net_demand_after_onsite_generation(
                    gross_ba_demand=gross_ba_demand,
                    states_demand=states_demand,
                    ba_mapping=ba_mapping,
                    wind_cf_data=wind_cf_data,
                    wind_ba_to_col=wind_ba_to_col,
                    solar_cf_data=solar_cf_data,
                    solar_ba_to_col=solar_ba_to_col,
                    renewable_scenario=renewable,
                )
                output_file = config.output_dir / f"EER_IRAlow_{ai_label}_{renewable_label}_{projection_label}_load_hourly.h5"
                write_load_file(config.base_load_file, output_file, net_ba_demand, modeled_years, config.overwrite)
                generated_files.append(output_file.name)

    LOGGER.info("Generated %d files", len(generated_files))
    if config.rename_legacy_files:
        rename_legacy_renewable_files(config.output_dir)


def main() -> None:
    configure_logging()
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
