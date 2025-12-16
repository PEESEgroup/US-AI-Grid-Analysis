"""
This code provides an example case on drafting the ned demand curve considering AI demand and on-site renewable generations
The results save process follows the contents in Figure 3(a) - (d) of the manuscript
The case setting: PJM and WECC-Southwest region, Summer, Weekday
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# ------------------------------- Paths & constants ------------------------------- #
BASE = Path("REPLACE YOUR PATH HERE")

FILE_AI_CAP       = BASE / "Regional AI Capacity.xlsx"             # High-Case for capacities
FILE_REP          = BASE / "representative_weekday_weekend.xlsx"   # 9 representative scenarios (Weekday used)
FILE_BASELOAD     = BASE / "2035_total_demand_MW.xlsx"             # per-region sheet; SUMMER_WEEKDAY (MWh)
FILE_RENEWABLES   = BASE / "2035_renewables_MW.xlsx"               # per-region sheet; [hour, winter, summer] (MW)
FILE_WIND_SOLAR   = BASE / "Regional_unitProfiles_wind_solar.xlsx" # summer wind & solar CFs for 100% AI adoption

PUE               = 1.10
CAP_SHEET         = "High-Case"     # select AI projection scenario
CAP_UNITS         = "GW"            # ("MW" if your capacity cells are MW)

REGIONS           = ["PJM", "WECC_SW"] # select region
SEASON            = "summer"           # select season
DAY_KIND          = "weekday"          # select daytype

# ------------------------------- Utilities ------------------------------- #
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

def util_to_gwh(util: np.ndarray, cap_gw: float, pue: float = PUE) -> np.ndarray:
    # Utilization (0–1) × capacity (GW) × PUE -> GWh per hour bin
    return as_unit_util(util) * cap_gw * pue

def fuzzy_pick_baselineE(name_pool: List[str]) -> str:
    low = [n.lower() for n in name_pool]
    for i, s in enumerate(low):
        if "baseline-e" in s or "baseline e" in s or re.search(r"\bbaseline[- ]?e\b", s):
            return name_pool[i]
    for i, s in enumerate(low):
        if "baseline" in s:
            return name_pool[i]
    for i, s in enumerate(low):
        if "mid" in s:
            return name_pool[i]
    return name_pool[0]

# ------------------------------- Readers ------------------------------- #
def read_ai_capacity_series(path: Path, sheet: str) -> pd.Series:
    """Row 1 (idx 0) = regions; Row 13 (idx 12) = 2035 values."""
    df = pd.read_excel(path, sheet_name=sheet, header=None)
    header = df.iloc[0].fillna("")
    regs, vals = [], []
    for c in range(1, df.shape[1]):
        nm = str(header[c]).strip()
        if not nm:
            continue
        v = pd.to_numeric(df.iloc[12, c], errors="coerce")
        if pd.notna(v):
            vals.append(float(v))
            regs.append(nm)
    ser = pd.Series(vals, index=regs)
    if CAP_UNITS.upper() == "MW":
        ser = ser / 1000.0
    return ser

def read_rep_profiles(path: Path, sheet_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Row 0: scenario names (merged ok; ffill)
    Row 1: component names
    Rows 2..25: 24 hours
    returns profiles[scenario][component] -> 24-array
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
            out.setdefault(scen, {})[comp] = vals
    return out

def read_region_baseload_summer_weekday(path: Path, region: str) -> np.ndarray:
    """Return SUMMER_WEEKDAY baseload (GWh) for region."""
    xls = pd.ExcelFile(path)
    sh = None
    for nm in xls.sheet_names:
        if norm_region(nm) == norm_region(region):
            sh = nm
            break
    if sh is None:
        raise RuntimeError(f"Region '{region}' not found in {path.name}")

    df = pd.read_excel(path, sheet_name=sh)
    target = None
    for c in df.columns:
        if norm_region(c) == "SUMMERWEEKDAY":
            target = c
            break
    if target is None:
        raise RuntimeError(f"SUMMER_WEEKDAY column missing for '{region}'")

    arr = pd.to_numeric(df[target].iloc[:24], errors="coerce").to_numpy(dtype=float)  # MWh
    return np.nan_to_num(arr, nan=0.0) / 1000.0  # → GWh

def read_region_renewables_summer_MW(path: Path, region: str) -> np.ndarray:
    """Return 24h SUMMER grid renewables in MW (convert to GWh by /1000 later)."""
    xls = pd.ExcelFile(path)
    sh = None
    for nm in xls.sheet_names:
        if norm_region(nm) == norm_region(region):
            sh = nm
            break
    if sh is None:
        raise RuntimeError(f"Region '{region}' not found in {path.name}")

    df = pd.read_excel(path, sheet_name=sh)

    # Find 'summer' column robustly
    summer_col = None
    for c in df.columns:
        if "summer" == re.sub(r"[^a-z]", "", str(c).lower()):
            summer_col = c
            break
    if summer_col is None:
        for c in df.columns:
            if "sum" in str(c).lower():
                summer_col = c
                break
    if summer_col is None:
        raise RuntimeError(f"Could not find 'summer' column in {path.name}:{sh}")

    arr_mw = pd.to_numeric(df[summer_col].iloc[:24], errors="coerce").to_numpy(dtype=float)
    return np.nan_to_num(arr_mw, nan=0.0)

def _read_cf_24h(path: Path, sheet: str, region: str) -> np.ndarray:
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    header = raw.iloc[0].fillna("")
    first = str(header.iloc[0]).strip().lower()
    start_col = 1 if first in {"hour", "hours", "hr"} else 0

    col_idx = None
    for j in range(start_col, raw.shape[1]):
        name = str(header.iloc[j]).strip()
        if name and norm_region(name) == norm_region(region):
            col_idx = j
            break
    if col_idx is None:
        raise RuntimeError(f"{region} column not found in '{sheet}'")

    arr = pd.to_numeric(raw.iloc[1:25, col_idx], errors="coerce").to_numpy(dtype=float)
    if len(arr) != 24:
        raise RuntimeError(f"Expected 24 rows for {sheet}")
    return np.nan_to_num(arr, nan=0.0)

def read_solar_cf_24h(path: Path, region: str) -> np.ndarray:
    """Summer Solar CF (0–1) 24h for region."""
    return _read_cf_24h(path, "Summer_Solar_CF_24h", region)

def read_wind_cf_24h(path: Path, region: str) -> np.ndarray:
    """Summer Wind CF (0–1) 24h for region."""
    return _read_cf_24h(path, "Summer_Wind_CF_24h", region)

# ------------------------------- Scenario helpers ------------------------------- #
def ai_energy_from_profile(components: Dict[str, np.ndarray], cap_gw: float) -> np.ndarray:
    need = {"training", "api", "conversation"}
    if not need.issubset(components.keys()):
        missing = sorted(list(need - set(components.keys())))
        raise RuntimeError(f"AI profile missing components: {missing}")
    return (
        util_to_gwh(components["training"], cap_gw)
        + util_to_gwh(components["api"], cap_gw)
        + util_to_gwh(components["conversation"], cap_gw)
    )

def build_ai_on_site_from_cf(ai_gwh: np.ndarray, cf: np.ndarray) -> np.ndarray:
    """Scale CF profile so its daily energy equals the AI daily energy (100% adoption)."""
    target = float(np.nansum(ai_gwh))
    denom = float(np.nansum(cf))
    if denom <= 0:
        return np.zeros_like(ai_gwh)
    return (target / denom) * cf

# ------------------------------- Main (data generation only) ------------------------------- #
def main():
    for f in (FILE_AI_CAP, FILE_REP, FILE_BASELOAD, FILE_RENEWABLES, FILE_WIND_SOLAR):
        if not f.exists():
            raise FileNotFoundError(f"Missing input: {f}")

    # Capacities (High-Case)
    cap_ser = read_ai_capacity_series(FILE_AI_CAP, CAP_SHEET)
    caps_gw: Dict[str, float] = {}
    for r in REGIONS:
        got = None
        for nm, v in cap_ser.items():
            if norm_region(nm) == norm_region(r):
                got = float(v)
                break
        if got is None:
            raise RuntimeError(f"Capacity for region '{r}' not found in {CAP_SHEET}")
        if CAP_UNITS.upper() == "MW":
            got /= 1000.0
        caps_gw[r] = got

    # Baseloads (Summer–Weekday, GWh)
    baseloads: Dict[str, np.ndarray] = {
        r: read_region_baseload_summer_weekday(FILE_BASELOAD, r)
        for r in REGIONS
    }

    # Grid renewables (Summer; MW → GWh)
    gridRE_gwh: Dict[str, np.ndarray] = {
        r: read_region_renewables_summer_MW(FILE_RENEWABLES, r) / 1000.0
        for r in REGIONS
    }

    # Representative profiles (Weekday)
    rep_wd = read_rep_profiles(FILE_REP, "Profiles_Weekday")
    rep_names = list(rep_wd.keys())
    if not rep_names:
        raise RuntimeError("No representative scenarios found in Profiles_Weekday.")
    baselineE_name = fuzzy_pick_baselineE(rep_names)

    # Wind & solar CFs (Summer)
    solar_cf_by_region = {r: read_solar_cf_24h(FILE_WIND_SOLAR, r) for r in REGIONS}
    wind_cf_by_region  = {r: read_wind_cf_24h(FILE_WIND_SOLAR, r)  for r in REGIONS}

    # Scenario rows:
    #   kind = "base"  -> no AI
    #   kind = "none"  -> Baseline-E AI, no on-site AI RE
    #   kind = "wind"  -> Baseline-E AI + 100% on-site wind
    #   kind = "solar" -> Baseline-E AI + 100% on-site solar
    scenarios = [
        ("Base net demand",            None,           "base"),
        (f"{baselineE_name}",          baselineE_name, "none"),
        ("100% Wind adoption",         baselineE_name, "wind"),
        ("100% Solar adoption",        baselineE_name, "solar"),
    ]

    panel_ids   = ["3(a)", "3(b)", "3(c)", "3(d)"]
    sheet_names = ["Fig3a_Base", "Fig3b_BaselineE", "Fig3c_Wind100", "Fig3d_Solar100"]

    ren_label_map = {
        "base":  "Base net (no AI)",
        "none":  "No on-site AI RE",
        "wind":  "100% AI on-site wind",
        "solar": "100% AI on-site solar",
    }

    hours = np.arange(24)
    excel_sheets: List[tuple[str, pd.DataFrame]] = []

    for row_idx, (row_title, scen_name, kind) in enumerate(scenarios):
        figure_panel = panel_ids[row_idx]
        sheet_name   = sheet_names[row_idx]
        ren_scen     = ren_label_map.get(kind, kind)

        rows_data = []

        def add_region_rows(region_name: str,
                            base: np.ndarray,
                            ai: np.ndarray,
                            ai_re: np.ndarray,
                            grid: np.ndarray,
                            total: np.ndarray,
                            net: np.ndarray,
                            ai_scenario_name: str):
            fossil = net
            non_fossil = total - net  # ≈ grid renewables
            for h in range(24):
                rows_data.append({
                    "Hour": int(h),
                    "Region": region_name,
                    "Fossil_generation_GWh": float(fossil[h]),
                    "Non_fossil_generation_GWh": float(non_fossil[h]),
                    "Net_demand_GWh": float(net[h]),
                    "Total_demand_GWh": float(total[h]),
                    "Baseload_GWh": float(base[h]),
                    "AI_load_GWh": float(ai[h]),
                    "AI_on_site_RE_GWh": float(ai_re[h]),
                    "Grid_RE_GWh": float(grid[h]),
                    "AI_computing_scenario": ai_scenario_name,
                    "AI_projection_scenario": CAP_SHEET,
                    "Renewable_scenario": ren_scen,
                    "Season": SEASON,
                    "Day_type": DAY_KIND,
                    "Figure_panel": figure_panel,
                    "Scenario_row_title": row_title,
                })

        # Build data for each region, then append
        for region in REGIONS:
            base = baseloads[region]
            grid = gridRE_gwh[region]

            if kind == "base":
                ai = np.zeros_like(base)
                ai_re = np.zeros_like(base)
                ai_scenario = "No AI (Base)"
            else:
                comps = rep_wd[scen_name]
                ai = ai_energy_from_profile(comps, caps_gw[region])
                if kind == "none":
                    ai_re = np.zeros_like(ai)
                elif kind == "wind":
                    ai_re = build_ai_on_site_from_cf(ai, wind_cf_by_region[region])
                elif kind == "solar":
                    ai_re = build_ai_on_site_from_cf(ai, solar_cf_by_region[region])
                else:
                    raise ValueError(kind)
                ai_scenario = scen_name

            total = base + ai - ai_re
            net   = total - grid

            add_region_rows(region, base, ai, ai_re, grid, total, net, ai_scenario)

        df_sheet = pd.DataFrame(rows_data)
        excel_sheets.append((sheet_name, df_sheet))

    # Save Excel with all sheets
    out_xlsx = BASE / "Figure_3(a-d)_data.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for sh, df in excel_sheets:
            df.to_excel(writer, sheet_name=sh[:31], index=False)

    print("Baseline-E selected:", baselineE_name)
    print("Saved data workbook:", out_xlsx)

if __name__ == "__main__":
    main()
