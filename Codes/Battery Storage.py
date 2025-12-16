"""
This code provides an example case on drafting the battery storage requirements for balancing ramping power drien by AI campus
The results save process follows the contents in Figure 4(c) - (f) of the manuscript
The case setting: PJM and WECC-Southwest region, Summer, Weekday
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ------------------------ Paths & constants ------------------------ #
BASE = Path("REPLACE YOUR PATH HERE")

FILE_AI_CAP      = BASE / "Regional AI Capacity.xlsx"
FILE_ALL40       = BASE / "all_40_weekday_weekend.xlsx"
FILE_BASELOAD    = BASE / "2035_total_demand_MW.xlsx"
FILE_RENEWABLES  = BASE / "2035_renewables_MW.xlsx"
FILE_WIND_SOLAR  = BASE / "Regional_unitProfiles_wind_solar.xlsx"

PROJ_SHEET = "Mid-Case"  # select projection scenario
DAY_TYPE   = "weekday"   # select daytype
PUE        = 1.10
AI_CAP_GW  = 10.0        # fixed AI capacity per region

SEASONS = ["summer", "winter"] # select season
SEASON_LABEL_FOR_EXPORT = "max(summer,winter)"

# RE cases
RE_CASES = ["No RE", "Wind", "Solar", "Con"]

# ------------------------ Readers & Helpers ------------------------ #
def norm_region(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def canonicalize_component(raw: str) -> str | None:
    s = str(raw).strip().lower()
    if s == "total":
        return "total"
    if "train" in s:
        return "training"
    if "api" in s or "infer" in s or "service" in s:
        return "api"
    if "convers" in s or "conv" in s or "chat" in s:
        return "conversation"
    return None

def as_unit_util(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr / 100.0 if np.nanmax(arr) > 1.5 else arr

def read_ai_regions(path: Path, sheet: str) -> List[str]:
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    header = df.iloc[0].fillna("")
    regions = []
    for c in range(1, df.shape[1]):
        name = str(header[c]).strip()
        if name:
            regions.append(name)
    return regions

def read_all40_profiles(path: Path, sheet_name: str = "Profiles_Weekday") -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns utilization profiles (unit, 0-1) for training/api/conversation.
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    scen_row = df.iloc[0].astype(str).replace("nan", np.nan).ffill()
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for c in range(1, df.shape[1]):
        scen = str(scen_row.iloc[c]).strip()
        comp = canonicalize_component(df.iloc[1, c])
        if not scen or comp is None:
            continue
        vals = pd.to_numeric(df.iloc[2:26, c], errors="coerce").to_numpy(dtype=float)
        if len(vals) == 24 and not np.isnan(vals).all():
            out.setdefault(scen, {})[comp] = as_unit_util(vals)
    return out

def read_region_baseload_weekday(path: Path, region: str, season: str) -> np.ndarray:
    """Return seasonal weekday baseload in GWh."""
    xls = pd.ExcelFile(path)
    sh = None
    for nm in xls.sheet_names:
        if norm_region(nm) == norm_region(region):
            sh = nm
            break
    if sh is None:
        raise RuntimeError(f"{region} sheet not found in {path.name}")
    df = pd.read_excel(path, sheet_name=sh)

    key = f"{season.upper()}WEEKDAY"
    col = None
    for c in df.columns:
        if norm_region(c) == key:
            col = c
            break
    if col is None:
        raise RuntimeError(f"{key} column missing in {path.name} for {region}")

    arr = pd.to_numeric(df[col].iloc[:24], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(arr, nan=0.0) / 1000.0  # MWh -> GWh

def read_region_gridRE_season_MW(path: Path, region: str, season: str) -> np.ndarray:
    """Return seasonal grid RE in MW (caller may /1000 to GWh)."""
    xls = pd.ExcelFile(path)
    sh = None
    for nm in xls.sheet_names:
        if norm_region(nm) == norm_region(region):
            sh = nm
            break
    if sh is None:
        raise RuntimeError(f"{region} sheet not found in {path.name}")
    df = pd.read_excel(path, sheet_name=sh)

    wanted = season.lower()
    season_col = None
    for c in df.columns:
        if wanted == re.sub(r"[^a-z]", "", str(c).lower()):
            season_col = c
            break
    if season_col is None:
        prefix = wanted[:3]
        for c in df.columns:
            if prefix in str(c).lower():
                season_col = c
                break
    if season_col is None:
        raise RuntimeError(f"{season} column not found in {path.name} for {region}")

    arr = pd.to_numeric(df[season_col].iloc[:24], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(arr, nan=0.0)

def read_wind_solar_cf_24h(path: Path, season: str, region: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wind_cf, solar_cf) in unit CF (0-1), 24h."""
    sh_w = "Summer_Wind_CF_24h" if season.lower() == "summer" else "Winter_Wind_CF_24h"
    sh_s = "Summer_Solar_CF_24h" if season.lower() == "summer" else "Winter_Solar_CF_24h"

    def ext(sheet: str) -> np.ndarray:
        raw = pd.read_excel(path, sheet_name=sheet, header=None)
        header = raw.iloc[0].fillna("")
        first = str(header.iloc[0]).strip().lower()
        start = 1 if first in {"hour", "hours", "hr"} else 0

        col = None
        for j in range(start, raw.shape[1]):
            if norm_region(header.iloc[j]) == norm_region(region):
                col = j
                break
        if col is None:
            raise RuntimeError(f"{region} not in {sheet}")

        arr = pd.to_numeric(raw.iloc[1:25, col], errors="coerce").to_numpy(dtype=float)
        return np.nan_to_num(arr, nan=0.0)

    return ext(sh_w), ext(sh_s)

# ------------------------ Calculation Logic ------------------------ #
def ai_energy_components(components: Dict[str, np.ndarray], cap_gw: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    need = {"training", "api", "conversation"}
    if not need.issubset(components.keys()):
        missing = sorted(list(need - set(components.keys())))
        raise RuntimeError(f"Scenario missing components: {missing}")

    K = cap_gw * PUE
    t = components["training"] * K
    a = components["api"] * K
    c = components["conversation"] * K
    return t, a, c

def build_re_profile(total_ai_gwh: float, kind: str, wind_cf: np.ndarray, solar_cf: np.ndarray, adoption: float) -> np.ndarray:
    if adoption <= 0 or kind == "No RE":
        return np.zeros(24)
    target = total_ai_gwh * adoption
    if kind == "Wind":
        denom = max(1e-9, float(np.sum(wind_cf)))
        return (target / denom) * wind_cf
    if kind == "Solar":
        denom = max(1e-9, float(np.sum(solar_cf)))
        return (target / denom) * solar_cf
    if kind == "Con":
        return np.full(24, target / 24.0)
    return np.zeros(24)

def compute_ramp_stats(profile: np.ndarray) -> Tuple[float, float]:
    ramp = np.diff(profile)
    up = float(np.nanmax(np.maximum(ramp, 0.0))) if ramp.size else 0.0
    down = float(np.nanmax(-np.minimum(ramp, 0.0))) if ramp.size else 0.0
    return up, down

def identify_max_min_scenarios(
    all_profiles: Dict[str, Dict[str, np.ndarray]],
    re_case: str,
    wind_cf: np.ndarray,
    solar_cf: np.ndarray,
    ai_cap_gw: float
) -> Tuple[str, str]:
    max_scen: Optional[str] = None
    min_scen: Optional[str] = None
    max_val = -1.0
    min_val = 1e15

    for scen, comps in all_profiles.items():
        if not all(k in comps for k in ("training", "api", "conversation")):
            continue

        t, a, c = ai_energy_components(comps, ai_cap_gw)
        ai_total_load = t + a + c
        total_energy = float(np.sum(ai_total_load))

        re_gen = build_re_profile(total_energy, re_case, wind_cf, solar_cf, 1.0)
        net_ai = ai_total_load - re_gen

        ramps = np.diff(net_ai)
        metric = float(np.nanmax(np.abs(ramps))) if ramps.size else 0.0

        if metric > max_val:
            max_val = metric
            max_scen = scen
        if metric < min_val:
            min_val = metric
            min_scen = scen

    if max_scen is None or min_scen is None:
        raise RuntimeError(f"Could not identify max/min scenarios for RE case '{re_case}'.")
    return max_scen, min_scen

# --- LP 1: Standard / Reverse Storage (Rigid) ---
def solve_rigid_storage(net_profile: np.ndarray, base_up: float, base_down: float) -> float:
    n = 24
    nvar = 49
    c_vec = np.zeros(nvar)
    c_vec[0] = 1.0  # minimize C

    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    # SoC dynamics
    for t in range(n):
        row = np.zeros(nvar)
        row[25 + t] = 1.0
        if t > 0:
            row[25 + t - 1] = -1.0
        row[1 + t] = -1.0
        A_eq.append(row)
        b_eq.append(0.0)

    # S[t] <= C ; |P[t]| <= C/4
    for t in range(n):
        r_soc = np.zeros(nvar)
        r_soc[25 + t] = 1.0
        r_soc[0] = -1.0
        A_ub.append(r_soc)
        b_ub.append(0.0)

        r_p1 = np.zeros(nvar)
        r_p1[1 + t] = 1.0
        r_p1[0] = -0.25
        A_ub.append(r_p1)
        b_ub.append(0.0)

        r_p2 = np.zeros(nvar)
        r_p2[1 + t] = -1.0
        r_p2[0] = -0.25
        A_ub.append(r_p2)
        b_ub.append(0.0)

    # Ramp constraints on net + P
    for t in range(1, n):
        d_net = float(net_profile[t] - net_profile[t - 1])

        r_up = np.zeros(nvar)
        r_up[1 + t] = 1.0
        r_up[1 + t - 1] = -1.0
        A_ub.append(r_up)
        b_ub.append(base_up - d_net)

        r_dn = np.zeros(nvar)
        r_dn[1 + t] = -1.0
        r_dn[1 + t - 1] = 1.0
        A_ub.append(r_dn)
        b_ub.append(base_down + d_net)

    bounds = [(0, None)] + [(None, None)] * 24 + [(0, None)] * 24

    res = linprog(
        c_vec,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if res.success:
        return float(res.x[0])
    return np.nan

def solve_reverse_storage(base_profile: np.ndarray, target_up: float, target_down: float) -> float:
    c = solve_rigid_storage(base_profile, target_up, target_down)
    if np.isfinite(c):
        return -float(c)
    return 0.0

# --- LP 2: Flexible Training Storage (Optimal) ---
def solve_flexible_lp_core(
    base_load: np.ndarray,
    fixed_re_profile: np.ndarray,
    api_util: np.ndarray,
    conv_util: np.ndarray,
    target_train_sum: float,
    base_up: float,
    base_down: float,
    ai_cap_gw: float,
    pue: float
) -> float:
    n = 24
    nvar = 73
    c_vec = np.zeros(nvar)
    c_vec[0] = 1.0

    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    idx_C = 0
    def idx_P(t: int) -> int: return 1 + t
    def idx_S(t: int) -> int: return 25 + t
    def idx_Tr(t: int) -> int: return 49 + t

    # SoC dynamics
    for t in range(n):
        row = np.zeros(nvar)
        row[idx_S(t)] = 1.0
        if t > 0:
            row[idx_S(t - 1)] = -1.0
        row[idx_P(t)] = -1.0
        A_eq.append(row)
        b_eq.append(0.0)

    # Training sum
    row_sum = np.zeros(nvar)
    for t in range(n):
        row_sum[idx_Tr(t)] = 1.0
    A_eq.append(row_sum)
    b_eq.append(float(target_train_sum))

    # Training bounds
    train_bounds = []
    for t in range(n):
        max_tr = max(0.0, 0.9 - float(api_util[t]) - float(conv_util[t]))
        train_bounds.append((0.0, max_tr))

    # Battery bounds
    for t in range(n):
        r1 = np.zeros(nvar); r1[idx_S(t)] = 1.0; r1[idx_C] = -1.0
        A_ub.append(r1); b_ub.append(0.0)

        r2 = np.zeros(nvar); r2[idx_P(t)] = 1.0; r2[idx_C] = -0.25
        A_ub.append(r2); b_ub.append(0.0)

        r3 = np.zeros(nvar); r3[idx_P(t)] = -1.0; r3[idx_C] = -0.25
        A_ub.append(r3); b_ub.append(0.0)

    K = ai_cap_gw * pue
    fixed_net = base_load - fixed_re_profile + (api_util + conv_util) * K

    for t in range(1, n):
        d_fixed = float(fixed_net[t] - fixed_net[t - 1])

        r_up = np.zeros(nvar)
        r_up[idx_Tr(t)] = K; r_up[idx_Tr(t - 1)] = -K
        r_up[idx_P(t)] = 1.0; r_up[idx_P(t - 1)] = -1.0
        A_ub.append(r_up); b_ub.append(float(base_up - d_fixed))

        r_dn = np.zeros(nvar)
        r_dn[idx_Tr(t)] = -K; r_dn[idx_Tr(t - 1)] = K
        r_dn[idx_P(t)] = -1.0; r_dn[idx_P(t - 1)] = 1.0
        A_ub.append(r_dn); b_ub.append(float(base_down + d_fixed))

    bounds = [(0, None)] + [(None, None)] * 24 + [(0, None)] * 24 + train_bounds

    res = linprog(
        c_vec,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if res.success:
        return float(res.x[0])
    return np.nan

def solve_minimax_ramp_training(
    base_load: np.ndarray,
    fixed_re_profile: np.ndarray,
    api_util: np.ndarray,
    conv_util: np.ndarray,
    target_train_sum: float,
    ai_cap_gw: float,
    pue: float
) -> Optional[np.ndarray]:
    n = 24
    nvar = 25

    c_vec = np.zeros(nvar)
    c_vec[0] = 1.0

    A_eq: List[np.ndarray] = []
    b_eq: List[float] = []
    A_ub: List[np.ndarray] = []
    b_ub: List[float] = []

    row_sum = np.zeros(nvar)
    for t in range(n):
        row_sum[1 + t] = 1.0
    A_eq.append(row_sum)
    b_eq.append(float(target_train_sum))

    t_bounds = []
    for t in range(n):
        max_tr = max(0.0, 0.9 - float(api_util[t]) - float(conv_util[t]))
        t_bounds.append((0.0, max_tr))

    K = ai_cap_gw * pue
    fixed_net = base_load - fixed_re_profile + (api_util + conv_util) * K

    for t in range(1, n):
        d_fixed = float(fixed_net[t] - fixed_net[t - 1])

        r1 = np.zeros(nvar)
        r1[1 + t] = K; r1[1 + t - 1] = -K; r1[0] = -1.0
        A_ub.append(r1); b_ub.append(-d_fixed)

        r2 = np.zeros(nvar)
        r2[1 + t] = -K; r2[1 + t - 1] = K; r2[0] = -1.0
        A_ub.append(r2); b_ub.append(d_fixed)

    bounds = [(0, None)] + t_bounds

    res = linprog(
        c_vec,
        A_ub=np.array(A_ub) if A_ub else None,
        b_ub=np.array(b_ub) if b_ub else None,
        A_eq=np.array(A_eq) if A_eq else None,
        b_eq=np.array(b_eq) if b_eq else None,
        bounds=bounds,
        method="highs",
    )
    if res.success:
        train_opt = res.x[1:]
        net_opt = fixed_net + train_opt * K
        return net_opt
    return None

def solve_flexible_storage_with_fallback(
    base_load: np.ndarray,
    fixed_re_profile: np.ndarray,
    api_util: np.ndarray,
    conv_util: np.ndarray,
    target_train_sum: float,
    base_up: float,
    base_down: float,
    ai_cap_gw: float,
    pue: float
) -> float:
    c_std = solve_flexible_lp_core(
        base_load, fixed_re_profile,
        api_util, conv_util, target_train_sum,
        base_up, base_down, ai_cap_gw, pue
    )
    if np.isfinite(c_std) and c_std > 1e-4:
        return float(c_std)

    opt_net_profile = solve_minimax_ramp_training(
        base_load, fixed_re_profile,
        api_util, conv_util, target_train_sum,
        ai_cap_gw, pue
    )
    if opt_net_profile is None:
        return 0.0

    t_up, t_down = compute_ramp_stats(opt_net_profile)
    return float(solve_reverse_storage(base_load, t_up, t_down))

# ------------------------ SAFE aggregation helper (FIX) ------------------------ #
def safe_nanmax(vals: List[float]) -> float:
    """Return nan if all values are nan; otherwise nanmax without RuntimeWarning."""
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmax(arr))

# ------------------------ Main Logic ------------------------ #
def full_process():
    for f in (FILE_AI_CAP, FILE_ALL40, FILE_BASELOAD, FILE_RENEWABLES, FILE_WIND_SOLAR):
        if not f.exists():
            raise FileNotFoundError(f"Missing input: {f}")

    regions = read_ai_regions(FILE_AI_CAP, PROJ_SHEET)
    all40 = read_all40_profiles(FILE_ALL40)

    row_types = ["Maximum", "Maximum (Optimal)", "Minimum", "Minimum (Optimal)"]

    # data_store[season][row][region_idx][re_case] -> dict
    data_store: Dict[str, List[List[Dict[str, Dict[str, float]]]]] = {
        s: [[{} for _ in range(len(regions))] for _ in range(4)]
        for s in SEASONS
    }

    for season in SEASONS:
        for r_idx, region in enumerate(regions):
            base_tot = read_region_baseload_weekday(FILE_BASELOAD, region, season)                 # GWh
            grid_re  = read_region_gridRE_season_MW(FILE_RENEWABLES, region, season) / 1000.0     # MW -> GWh
            base_net = base_tot - grid_re
            b_up, b_down = compute_ramp_stats(base_net)

            w_cf, s_cf = read_wind_solar_cf_24h(FILE_WIND_SOLAR, season, region)
            has_wind = not np.allclose(w_cf, 0)
            has_solar = not np.allclose(s_cf, 0)

            for re_case in RE_CASES:
                if re_case == "Wind" and not has_wind:
                    continue
                if re_case == "Solar" and not has_solar:
                    continue

                max_scen, min_scen = identify_max_min_scenarios(all40, re_case, w_cf, s_cf, AI_CAP_GW)

                def process_scen(scen_name: str) -> Tuple[float, float, float, float]:
                    comps = all40[scen_name]
                    t, a, c = ai_energy_components(comps, AI_CAP_GW)
                    ai_total = t + a + c
                    tot_e = float(np.sum(ai_total))

                    # 100% Rigid
                    re100 = build_re_profile(tot_e, re_case, w_cf, s_cf, 1.0)
                    net100 = base_net + ai_total - re100
                    c100_rigid = solve_rigid_storage(net100, b_up, b_down)
                    if np.isfinite(c100_rigid) and c100_rigid < 1e-4:
                        n_up, n_down = compute_ramp_stats(net100)
                        c100_rigid = solve_reverse_storage(base_net, n_up, n_down)

                    # 50% Rigid
                    c50_rigid = np.nan
                    re50 = None
                    if re_case != "No RE":
                        re50 = build_re_profile(tot_e, re_case, w_cf, s_cf, 0.5)
                        net50 = base_net + ai_total - re50
                        c50_rigid = solve_rigid_storage(net50, b_up, b_down)
                        if np.isfinite(c50_rigid) and c50_rigid < 1e-4:
                            n_up, n_down = compute_ramp_stats(net50)
                            c50_rigid = solve_reverse_storage(base_net, n_up, n_down)

                    # Flexible (training sum in utilization units)
                    t_sum = float(np.sum(comps["training"]))

                    c100_flex = solve_flexible_storage_with_fallback(
                        base_net, re100, comps["api"], comps["conversation"], t_sum,
                        b_up, b_down, AI_CAP_GW, PUE
                    )

                    c50_flex = np.nan
                    if re_case != "No RE" and re50 is not None:
                        c50_flex = solve_flexible_storage_with_fallback(
                            base_net, re50, comps["api"], comps["conversation"], t_sum,
                            b_up, b_down, AI_CAP_GW, PUE
                        )

                    return (float(c100_rigid), float(c50_rigid), float(c100_flex), float(c50_flex))

                r100_max, r50_max, f100_max, f50_max = process_scen(max_scen)
                r100_min, r50_min, f100_min, f50_min = process_scen(min_scen)

                def norm_val(v: float) -> float:
                    if not np.isfinite(v):
                        return np.nan
                    return (v / 4.0) / AI_CAP_GW  # (C/4)/AI

                data_store[season][0][r_idx][re_case] = {"scen": max_scen,            "100": norm_val(r100_max), "50": norm_val(r50_max)}
                data_store[season][1][r_idx][re_case] = {"scen": max_scen + " (Opt)", "100": norm_val(f100_max), "50": norm_val(f50_max)}
                data_store[season][2][r_idx][re_case] = {"scen": min_scen,            "100": norm_val(r100_min), "50": norm_val(r50_min)}
                data_store[season][3][r_idx][re_case] = {"scen": min_scen + " (Opt)", "100": norm_val(f100_min), "50": norm_val(f50_min)}

    # Aggregate max(summer,winter) and export
    export_rows = []
    for row_idx in range(4):
        for r_idx, region in enumerate(regions):
            for re_case in RE_CASES:
                d_sum = data_store["summer"][row_idx][r_idx].get(re_case, {})
                d_win = data_store["winter"][row_idx][r_idx].get(re_case, {})

                v_s_100 = d_sum.get("100", np.nan)
                v_w_100 = d_win.get("100", np.nan)
                val_100 = safe_nanmax([v_s_100, v_w_100])  # FIXED

                v_s_50 = d_sum.get("50", np.nan)
                v_w_50 = d_win.get("50", np.nan)
                val_50 = safe_nanmax([v_s_50, v_w_50])     # FIXED

                used_scen = "N/A"
                if np.isfinite(v_s_100) and (np.isnan(v_w_100) or v_s_100 >= v_w_100):
                    used_scen = d_sum.get("scen", "N/A")
                elif np.isfinite(v_w_100):
                    used_scen = d_win.get("scen", "N/A")

                export_rows.append({
                    "Region": region,
                    "AI_computing_scenario": used_scen,
                    "AI_projection_scenario": PROJ_SHEET,
                    "Renewable_scenario": re_case,
                    "Season": SEASON_LABEL_FOR_EXPORT,
                    "Day_type": DAY_TYPE,
                    "Storage_type": row_types[row_idx],
                    "Adoption": "100%",
                    "Value_GW_GW": val_100,
                })

                if re_case != "No RE":
                    export_rows.append({
                        "Region": region,
                        "AI_computing_scenario": used_scen,
                        "AI_projection_scenario": PROJ_SHEET,
                        "Renewable_scenario": re_case,
                        "Season": SEASON_LABEL_FOR_EXPORT,
                        "Day_type": DAY_TYPE,
                        "Storage_type": row_types[row_idx],
                        "Adoption": "50%",
                        "Value_GW_GW": val_50,
                    })

    out_xlsx = BASE / "Figure_4(c-f)_data.xlsx"
    pd.DataFrame(export_rows).to_excel(out_xlsx, index=False)
    print("Saved data workbook:", out_xlsx)

if __name__ == "__main__":
    full_process()
