#!/usr/bin/env python3
r"""
Generate ReEDS hourly load files for pure AI demand cases.

"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

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
def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be positive.")
    return parsed


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


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> Config:
    default_data_dir = "Data Path" # Replace your path here
    parser = argparse.ArgumentParser(
        description="Generate ReEDS hourly load files with pure AI demand additions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing all input data files.",
    )
    parser.add_argument(
        "--reeds-input-dir",
        type=Path,
        default=None,
        help="Optional legacy ReEDS input directory. Default: <data-dir>.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated HDF5 load files. Default: <data-dir>/loaddata.",
    )
    parser.add_argument("--ai-capacity-file", type=Path, default=None)
    parser.add_argument("--pue-file", type=Path, default=None)
    parser.add_argument("--pue-improvement-file", type=Path, default=None)
    parser.add_argument("--utilization-file", type=Path, default=None)
    parser.add_argument("--ai-spatial-file", type=Path, default=None)
    parser.add_argument("--ba-area-file", type=Path, default=None, help="Default: <data-dir>/BA_area.csv.")
    parser.add_argument("--base-load-file", type=Path, default=None, help="Default: <data-dir>/EER_IRAlow_load_hourly.h5.")
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip output files that already exist instead of overwriting them.",
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
        overwrite=not args.no_overwrite,
    )


# -----------------------------------------------------------------------------
# Input readers
# -----------------------------------------------------------------------------
def load_ai_capacity_projection(excel_path: Path) -> Tuple[np.ndarray, List[int]]:
    """Read AI capacity projections as [projection_case, modeled_year]."""
    ensure_file(excel_path, "AI capacity projection workbook")
    df = pd.read_excel(excel_path)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["Year", "Low-Case", "Mid-Case", "High-Case"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {excel_path.name}: {missing}")

    df = df[required_cols].copy()
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Year"]).sort_values("Year").reset_index(drop=True)

    if len(df) != 12:
        raise ValueError(f"Expected 12 modeled years in {excel_path.name}; found {len(df)}.")

    capacity = np.vstack(
        [df["Low-Case"].to_numpy(), df["Mid-Case"].to_numpy(), df["High-Case"].to_numpy()]
    ).astype(np.float32)
    modeled_years = [int(y) for y in df["Year"].to_numpy()]
    return capacity, modeled_years


def load_profile_workbook(workbook_path: Path) -> Dict[str, np.ndarray]:
    """Load utilization-profile sheets as {sheet_name: [weather_year, hour]}."""
    ensure_file(workbook_path, "AI utilization workbook")
    xls = pd.ExcelFile(workbook_path)
    profiles: Dict[str, np.ndarray] = {}

    for sheet_name in xls.sheet_names:
        if sheet_name.strip().lower() == "readme":
            continue
        df = pd.read_excel(workbook_path, sheet_name=sheet_name)
        df = standardize_hourly_profile_df(df, sheet_name)
        profile = np.zeros((N_WEATHER_YEARS, HOURS_PER_YEAR), dtype=np.float32)
        for weather_idx, weather_year in enumerate(WEATHER_YEARS):
            profile[weather_idx, :] = pd.to_numeric(
                df[weather_year], errors="coerce"
            ).to_numpy(dtype=np.float32)
        profiles[sheet_name] = profile

    if not profiles:
        raise ValueError(f"No utilization-profile sheets found in {workbook_path}")
    return profiles


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


def load_pue_profiles(pue_file: Path, pue_improvement_file: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_file(pue_file, "Base PUE CSV")
    ensure_file(pue_improvement_file, "Improved PUE CSV")
    return pd.read_csv(pue_file), pd.read_csv(pue_improvement_file)


def load_ai_spatial_weights(ai_spatial_file: Path, n_states: int) -> np.ndarray:
    ensure_file(ai_spatial_file, "AI spatial allocation file")
    weights = np.loadtxt(ai_spatial_file, delimiter="\t", dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if len(weights) != n_states:
        raise ValueError(
            f"AI spatial file length ({len(weights)}) does not match the state list length ({n_states})."
        )
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
    df_pue: pd.DataFrame,
    df_pue_improved: pd.DataFrame,
    regions: Iterable[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    pue_base: Dict[str, np.ndarray] = {}
    pue_improved: Dict[str, np.ndarray] = {}
    for region in sorted({str(r) for r in regions if str(r) not in {"-1", "", "nan"}}):
        if region not in df_pue.columns or region not in df_pue_improved.columns:
            raise ValueError(f"Region '{region}' is missing from one or both PUE CSV files.")
        base = pd.to_numeric(df_pue[region], errors="coerce").to_numpy(dtype=np.float32)
        improved = pd.to_numeric(df_pue_improved[region], errors="coerce").to_numpy(dtype=np.float32)
        if len(base) != HOURS_PER_YEAR or len(improved) != HOURS_PER_YEAR:
            raise ValueError(f"PUE profile for region '{region}' must have {HOURS_PER_YEAR} rows.")
        pue_base[region] = base
        pue_improved[region] = improved
    return pue_base, pue_improved


def validate_base_load_file(base_load_file: Path, modeled_years: Sequence[int], n_ba: int) -> None:
    ensure_file(base_load_file, "Base load HDF5 file")
    with h5py.File(base_load_file, "r") as h5:
        for year in modeled_years:
            key = str(year)
            if key not in h5:
                raise KeyError(f"Dataset '{key}' is missing from {base_load_file}")
            dataset = h5[key]
            if dataset.shape[0] != N_TOTAL_HOURS or dataset.shape[1] != n_ba:
                raise ValueError(
                    f"Dataset {key} shape should be ({N_TOTAL_HOURS}, {n_ba}); got {dataset.shape}."
                )


# -----------------------------------------------------------------------------
# Core calculation
# -----------------------------------------------------------------------------
def build_ai_demand_for_profile(
    states_demand: np.ndarray,
    utilization_profile: np.ndarray,
    ba_mapping: BAMapping,
    pue_base: Dict[str, np.ndarray],
    pue_improved: Dict[str, np.ndarray],
) -> np.ndarray:
    """Build BA AI demand as [BA, modeled_year, weather_hour]."""
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
        ba_demand[ba_idx, :, :] = demand_year[:, None] * pue_year_hour_all_weather * profile_flat[None, :]

    return ba_demand


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
            dataset = h5[str(year)]
            dataset[:, :] = dataset[:, :] + ba_demand[:, year_idx, :].T

    LOGGER.info("Saved %s", output_file)


def run(config: Config) -> None:
    LOGGER.info("Loading inputs")
    ai_capacity, modeled_years = load_ai_capacity_projection(config.ai_capacity_file)
    profiles = load_profile_workbook(config.utilization_file)
    df_pue, df_pue_improved = load_pue_profiles(config.pue_file, config.pue_improvement_file)
    ai_spatial_weights = load_ai_spatial_weights(config.ai_spatial_file, len(STATES))
    ba_mapping = build_ba_mapping(config.ba_area_file, STATES)
    validate_base_load_file(config.base_load_file, modeled_years, len(ba_mapping.ba_names))

    pue_base, pue_improved = build_region_pue_lookup(
        df_pue,
        df_pue_improved,
        ba_mapping.region_ids[ba_mapping.valid_ba_indices],
    )

    # AI capacity in GW and converted to MW before writing to load files.
    load_data_mw = ai_capacity * 1.0e3
    states_demand_all = load_data_mw[:, :, None] * ai_spatial_weights[None, None, :]

    generated_files: List[str] = []
    for projection_idx in range(states_demand_all.shape[0]):
        projection_label = f"proj{projection_idx + 1}"
        states_demand = states_demand_all[projection_idx]
        for profile_idx, (profile_name, utilization_profile) in enumerate(profiles.items(), start=1):
            LOGGER.info("Processing %s | AI%d | %s", projection_label, profile_idx, profile_name)
            ba_demand = build_ai_demand_for_profile(
                states_demand=states_demand,
                utilization_profile=utilization_profile,
                ba_mapping=ba_mapping,
                pue_base=pue_base,
                pue_improved=pue_improved,
            )
            output_file = config.output_dir / f"EER_IRAlow_AI{profile_idx}_{projection_label}_load_hourly.h5"
            write_load_file(config.base_load_file, output_file, ba_demand, modeled_years, config.overwrite)
            generated_files.append(output_file.name)

    LOGGER.info("Generated %d files", len(generated_files))
    for file_name in generated_files:
        LOGGER.info("  %s", file_name)


def main() -> None:
    configure_logging()
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
