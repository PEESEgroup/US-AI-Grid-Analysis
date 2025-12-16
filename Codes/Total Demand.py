"""
This code provides an example case on drafting the total demand curve considering AI demand and on-site renewable generations
The results save process follows the contents in Figure 2(a) and 2(b) of the manuscript
The case setting: PJM region, Summer, Weekday
"""

from __future__ import annotations
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#
BASE = Path("REPLACE YOUR PATH HERE") 

FILE_AI_CAP      = BASE / "Regional AI Capacity.xlsx"
FILE_REP         = BASE / "representative_weekday_weekend.xlsx"
FILE_ALL40       = BASE / "all_40_weekday_weekend.xlsx"
FILE_WIND_SOLAR  = BASE / "Regional_unitProfiles_wind_solar.xlsx"
FILE_BASELOAD    = BASE / "2035_total_demand_MW.xlsx"

REGION_FOR_PLOT  = "PJM"          # Pick region
SEASON_FOR_PLOT  = "summer"       # Pick season
PUE              = 1.10
CAP_SOURCE_SHEET = "High-Case"    # Pick AI projection case
CAP_UNITS        = "GW"           # change to "MW" if workbook stores MW

# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#
def norm_region(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())

def canonicalize_component(raw: str) -> str | None:
    s = str(raw).strip().lower()
    if s == "total":
        return "total"
    if s in {"training", "train", "pretrain", "pre-training"} or "train" in s:
        return "training"
    if s in {"api", "service", "inference"} or "api" in s or "infer" in s or "service" in s:
        return "api"
    if s in {"conversation", "conv", "chat", "assistant", "dialog"} or "convers" in s or "chat" in s or "conv" in s:
        return "conversation"
    return None

def as_unit_util(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    return arr / 100.0 if np.nanmax(arr) > 1.5 else arr

def util_to_gwh(util: np.ndarray, cap_gw: float, pue: float = PUE) -> np.ndarray:
    """Convert utilization (0-1 or 0-100) to hourly energy in GWh given capacity (GW) and PUE."""
    return as_unit_util(util) * cap_gw * pue

def compute_peak_stats(ref_series: np.ndarray, cmp_series: np.ndarray) -> Tuple[int, float]:
    """Return (Δt hours, ΔP%) comparing peaks of cmp_series vs ref_series (signed hours)."""
    if not (np.isfinite(ref_series).any() and np.isfinite(cmp_series).any()):
        return 0, np.nan
    idx_r = int(np.nanargmax(ref_series))
    val_r = float(ref_series[idx_r])
    idx_c = int(np.nanargmax(cmp_series))
    val_c = float(cmp_series[idx_c])
    dt = int(idx_c - idx_r)
    dpct = (100.0 * (val_c - val_r) / val_r) if val_r > 0 else np.nan
    return dt, dpct

# -----------------------------------------------------------------------------#
# Readers
# -----------------------------------------------------------------------------#
def read_ai_capacity_sheet(path: Path, sheet_name: str) -> pd.Series:
    """Row 1 (idx 0) = regions; Row 13 (idx 12) = 2035 values."""
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header = df.iloc[0].fillna("")
    regions, values = [], []
    for c in range(1, df.shape[1]):
        name = str(header[c]).strip()
        if not name:
            continue
        v = pd.to_numeric(df.iloc[12, c], errors="coerce")
        if pd.notna(v):
            regions.append(name)
            values.append(float(v))
    return pd.Series(values, index=regions, name="2035_AI_Capacity")

def read_rep_profiles(path: Path, sheet_name: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Row 0: scenario names (merged cells ok; ffill)
    Row 1: component names
    Rows 2..25: 24 hours
    returns profiles[scenario][component] -> 24-array
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None)
    scen_row = df.iloc[0].astype(str).replace("nan", np.nan).ffill()
    profiles: Dict[str, Dict[str, np.ndarray]] = {}
    for c in range(1, df.shape[1]):  # skip hour column
        scen = str(scen_row.iloc[c]).strip()
        comp = canonicalize_component(df.iloc[1, c])
        if not scen or comp is None:
            continue
        vals = pd.to_numeric(df.iloc[2:26, c], errors="coerce").to_numpy(dtype=float)
        if len(vals) == 24 and not np.isnan(vals).all():
            profiles.setdefault(scen, {})[comp] = vals
    return profiles

def read_region_baseload(path: Path, region: str) -> pd.DataFrame:
    """Return 24×4 DataFrame with seasonal weekday/weekend columns in MWh."""
    xls = pd.ExcelFile(path)
    sh_target = None
    for sh in xls.sheet_names:
        if norm_region(sh) == norm_region(region):
            sh_target = sh
            break
    if sh_target is None:
        raise RuntimeError(f"Region sheet '{region}' not found in {path.name}")

    df = pd.read_excel(path, sheet_name=sh_target)
    cmap = {}
    for col in df.columns:
        n = norm_region(col)
        if n.startswith("WINTERWEEKDA"):
            cmap["winter_weekday"] = col
        elif n == "WINTERWEEKEND":
            cmap["winter_weekend"] = col
        elif n == "SUMMERWEEKDAY":
            cmap["summer_weekday"] = col
        elif n == "SUMMERWEEKEND":
            cmap["summer_weekend"] = col

    req = ["winter_weekday", "winter_weekend", "summer_weekday", "summer_weekend"]
    miss = [k for k in req if k not in cmap]
    if miss:
        raise RuntimeError(f"Missing expected columns: {miss}")

    if "hour" in df.columns:
        df = df.sort_values("hour")

    out = pd.DataFrame({
        "winter_weekday": pd.to_numeric(df[cmap["winter_weekday"]].iloc[:24], errors="coerce").to_numpy(),
        "winter_weekend": pd.to_numeric(df[cmap["winter_weekend"]].iloc[:24], errors="coerce").to_numpy(),
        "summer_weekday": pd.to_numeric(df[cmap["summer_weekday"]].iloc[:24], errors="coerce").to_numpy(),
        "summer_weekend": pd.to_numeric(df[cmap["summer_weekend"]].iloc[:24], errors="coerce").to_numpy(),
    })
    return out

def read_unit_cf_profiles(path: Path, season: str, region: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (wind_cf_24, solar_cf_24) using SUMMER or WINTER sheets."""
    sheet_wind  = "Summer_Wind_CF_24h" if season.lower() == "summer" else "Winter_Wind_CF_24h"
    sheet_solar = "Summer_Solar_CF_24h" if season.lower() == "summer" else "Winter_Solar_CF_24h"

    def extract(sheet: str) -> np.ndarray:
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
            raise RuntimeError(f"{region} column not found in sheet '{sheet}'")

        arr = pd.to_numeric(raw.iloc[1:25, col_idx], errors="coerce").to_numpy(dtype=float)
        if len(arr) != 24:
            raise RuntimeError(f"Expected 24 rows in '{sheet}' for {region}")
        return arr

    return extract(sheet_wind), extract(sheet_solar)

# -----------------------------------------------------------------------------#
# Scenario selection helpers
# -----------------------------------------------------------------------------#
def fuzzy_pick_baseline(name_pool: List[str]) -> str:
    low = [n.lower() for n in name_pool]
    for i, s in enumerate(low):
        if "baseline" in s or "base-line" in s or s == "base":
            return name_pool[i]
    for i, s in enumerate(low):
        if "mid" in s:
            return name_pool[i]
    return name_pool[0]

def is_baseline_like(name: str) -> bool:
    """Robust detector for any 'Baseline*' variant."""
    s = (name or "").strip().lower()
    if not s:
        return False
    if s.startswith("baseline") or s.startswith("base "):
        return True
    s_norm = re.sub(r"[^a-z0-9]+", " ", s).strip()
    if s_norm in {"baseline", "base"}:
        return True
    if s_norm.startswith("baseline "):
        return True
    if re.match(r"^baseline[a-z0-9]?$", s_norm):
        return True
    if re.search(r"\bbaseline\b", s):
        return True
    return False

# -----------------------------------------------------------------------------#
# Main (data generation only)
# -----------------------------------------------------------------------------#
def main():
    # Validate inputs
    for f in (FILE_AI_CAP, FILE_REP, FILE_ALL40, FILE_WIND_SOLAR, FILE_BASELOAD):
        if not f.exists():
            raise FileNotFoundError(f"Missing input: {f}")

    # === Capacity (HIGH-CASE) ===
    ser_cap = read_ai_capacity_sheet(FILE_AI_CAP, CAP_SOURCE_SHEET)
    pjm_cap = None
    for rname, v in ser_cap.items():
        if norm_region(rname) == norm_region(REGION_FOR_PLOT):
            pjm_cap = float(v)
            break
    if pjm_cap is None:
        raise RuntimeError(f"{REGION_FOR_PLOT} capacity not found in '{CAP_SOURCE_SHEET}'")
    if CAP_UNITS.upper() == "MW":
        pjm_cap /= 1000.0  # → GW

    # === Baseload (SUMMER) ===
    base_df = read_region_baseload(FILE_BASELOAD, REGION_FOR_PLOT)
    base_wd = np.nan_to_num(base_df["summer_weekday"].to_numpy(), nan=0.0) / 1000.0  # MWh → GWh
    base_we = np.nan_to_num(base_df["summer_weekend"].to_numpy(), nan=0.0) / 1000.0
    peak_base_wd = int(np.nanargmax(base_wd))
    peak_base_we = int(np.nanargmax(base_we))

    # === Renewable CFs (SUMMER) ===
    wind_cf, solar_cf = read_unit_cf_profiles(FILE_WIND_SOLAR, "summer", REGION_FOR_PLOT)
    wind_cf = np.nan_to_num(wind_cf, nan=0.0)
    solar_cf = np.nan_to_num(solar_cf, nan=0.0)

    # === Representative & All-40 profiles ===
    rep_wd = read_rep_profiles(FILE_REP, "Profiles_Weekday")
    rep_we = read_rep_profiles(FILE_REP, "Profiles_Weekend")
    rep_names = [n for n in rep_wd.keys() if n in rep_we]

    all40_wd = read_rep_profiles(FILE_ALL40, "Profiles_Weekday")
    all40_we = read_rep_profiles(FILE_ALL40, "Profiles_Weekend")
    all40_names = [n for n in all40_wd.keys() if n in all40_we]

    if not rep_names:
        raise RuntimeError("Representative sheets have no overlapping scenarios.")
    if not all40_names:
        raise RuntimeError("All-40 sheets have no overlapping scenarios.")

    def ai_energy(task_dict: Dict[str, np.ndarray]) -> np.ndarray:
        return (
            util_to_gwh(task_dict["training"], pjm_cap)
            + util_to_gwh(task_dict["api"], pjm_cap)
            + util_to_gwh(task_dict["conversation"], pjm_cap)
        )

    # ---------- Figure 2(a) scenario selection ----------
    baseline_name = fuzzy_pick_baseline(rep_names)

    def has_required(src_wd: Dict[str, Dict[str, np.ndarray]],
                     src_we: Dict[str, Dict[str, np.ndarray]],
                     name: str) -> bool:
        need = {"training", "api", "conversation"}
        return (
            name in src_wd and name in src_we
            and need.issubset(src_wd[name].keys())
            and need.issubset(src_we[name].keys())
        )

    # Build NON-Baseline candidate list (exists in weekday+weekend with all 3 comps)
    seen = set()
    candidates: List[str] = []
    for name in list(all40_names) + list(rep_names):
        if name in seen:
            continue
        if is_baseline_like(name):
            continue
        if name == baseline_name:
            continue
        ok = has_required(all40_wd, all40_we, name) or has_required(rep_wd, rep_we, name)
        if ok:
            candidates.append(name)
            seen.add(name)

    if len(candidates) < 2:
        raise RuntimeError("Not enough non-Baseline scenarios to select earliest/latest.")

    def total_series_for_scenario(name: str, day: str) -> np.ndarray:
        if has_required(all40_wd, all40_we, name):
            comps = all40_wd[name] if day == "weekday" else all40_we[name]
        elif has_required(rep_wd, rep_we, name):
            comps = rep_wd[name] if day == "weekday" else rep_we[name]
        else:
            raise RuntimeError(f"Scenario '{name}' missing components for {day}.")
        ai = ai_energy(comps)
        base = base_wd if day == "weekday" else base_we
        return base + ai

    # Compute signed peak hour shifts vs baseload for each candidate
    stats = []
    for scen in candidates:
        tot_wd = total_series_for_scenario(scen, "weekday")
        tot_we = total_series_for_scenario(scen, "weekend")
        s_wd = int(np.nanargmax(tot_wd)) - peak_base_wd
        s_we = int(np.nanargmax(tot_we)) - peak_base_we
        stats.append({
            "name": scen,
            "wd": s_wd,
            "we": s_we,
            "sum": s_wd + s_we,
            "maxpos": max(s_wd, s_we),
            "minneg": min(s_wd, s_we),
        })

    # Choose EARLIEST by the most negative individual shift (<= -1 if available)
    neg_list = [d for d in stats if d["minneg"] <= -1]
    if neg_list:
        earliest = min(neg_list, key=lambda d: (d["minneg"], d["sum"]))["name"]
    else:
        earliest = min(stats, key=lambda d: (d["sum"], d["minneg"]))["name"]

    # Choose LATEST by the largest positive individual shift (>= +1 if available)
    pos_list = [d for d in stats if d["maxpos"] >= 1]
    if pos_list:
        latest = max(pos_list, key=lambda d: (d["maxpos"], d["sum"]))["name"]
    else:
        latest = max(stats, key=lambda d: (d["sum"], d["maxpos"]))["name"]

    # Ensure earliest and latest are distinct
    if earliest == latest:
        alt_pos = [
            d for d in sorted(stats, key=lambda d: (d["maxpos"], d["sum"]), reverse=True)
            if d["name"] != earliest
        ]
        if alt_pos:
            latest = alt_pos[0]["name"]

    # Console report for verification
    print("\n=== Candidate peak-shift summary (vs baseload) ===")
    for d in sorted(stats, key=lambda x: (x["maxpos"], -x["minneg"], x["sum"]), reverse=True):
        print(
            f"  {d['name']:<24s}  wd={d['wd']:+2d}h  we={d['we']:+2d}h  sum={d['sum']:+3d}  "
            f"max+={d['maxpos']:+2d}  min-={d['minneg']:+3d}"
        )

    print(f"\n[SELECT] Baseline : {baseline_name}")
    print(f"[SELECT] Earliest : {earliest}  (most negative individual shift)")
    print(f"[SELECT] Latest   : {latest}    (largest positive individual shift)")

    # =========================
    # FIGURE 2(a) DATA: 3 scenarios × {weekday, weekend}
    # =========================
    hours = np.arange(24)
    fig1_specs = [baseline_name, earliest, latest]

    panel_cache = []  # list of (scenario_name, dict_wd, dict_we)
    print("\n=== Figure 2(a) Peak Stats (vs Baseload) — printed only ===")
    for scen in fig1_specs:
        src_wd, src_we = (rep_wd, rep_we) if (scen in rep_wd and scen in rep_we) else (all40_wd, all40_we)
        comps_wd = src_wd[scen]
        comps_we = src_we[scen]

        wd_tr = util_to_gwh(comps_wd["training"], pjm_cap)
        wd_ap = util_to_gwh(comps_wd["api"], pjm_cap)
        wd_cv = util_to_gwh(comps_wd["conversation"], pjm_cap)

        we_tr = util_to_gwh(comps_we["training"], pjm_cap)
        we_ap = util_to_gwh(comps_we["api"], pjm_cap)
        we_cv = util_to_gwh(comps_we["conversation"], pjm_cap)

        tot_wd = base_wd + wd_tr + wd_ap + wd_cv
        tot_we = base_we + we_tr + we_ap + we_cv

        dt_wd, dp_wd = compute_peak_stats(base_wd, tot_wd)
        dt_we, dp_we = compute_peak_stats(base_we, tot_we)
        print(f"  {scen:<24s} — weekday: Δt={dt_wd:+d} h, ΔP={dp_wd:+.1f}%")
        print(f"  {scen:<24s} — weekend: Δt={dt_we:+d} h, ΔP={dp_we:+.1f}%")

        panel_cache.append(
            (
                scen,
                {"base": base_wd, "tr": wd_tr, "api": wd_ap, "conv": wd_cv, "total": tot_wd},
                {"base": base_we, "tr": we_tr, "api": we_ap, "conv": we_cv, "total": tot_we},
            )
        )

    # =========================
    # FIGURE 2(b) DATA: Net vs Baseload (WEEKDAY ONLY), 3×(50%,100%)
    # =========================
    src_ai = rep_wd if baseline_name in rep_wd else all40_wd
    comps_ai = src_ai[baseline_name]
    ai_train = util_to_gwh(comps_ai["training"], pjm_cap)
    ai_api   = util_to_gwh(comps_ai["api"], pjm_cap)
    ai_conv  = util_to_gwh(comps_ai["conversation"], pjm_cap)
    ai_wd = ai_train + ai_api + ai_conv

    adoptions = [0.50, 1.00]
    resources = ["Wind", "Solar", "Constant"]

    def onsite_generation(frac: float, kind: str) -> np.ndarray:
        target = frac * float(np.nansum(ai_wd))  # GWh/day
        if kind == "Wind":
            denom = max(1e-9, float(np.nansum(wind_cf)))
            gen = (target / denom) * wind_cf
        elif kind == "Solar":
            denom = max(1e-9, float(np.nansum(solar_cf)))
            gen = (target / denom) * solar_cf
        elif kind == "Constant":
            gen = np.full(24, target / 24.0)
        else:
            raise ValueError(kind)
        return gen

    panels = []
    print("\n=== Figure 2(b) Peak Stats (vs Baseload) — printed only ===")
    for res in resources:
        for frac in adoptions:
            gen = onsite_generation(frac, res)
            net = ai_wd - gen                    # + above base (extra), − below base (erosion)
            erosion = np.clip(-net, 0, base_wd)  # eats into baseload
            pos     = np.clip(net,  0, None)     # extra load above baseload
            base_remain = base_wd - erosion
            total = base_remain + pos

            dt, dpct = compute_peak_stats(base_wd, total)
            print(f"  {res:8s} — {int(frac*100):3d}%: Δt={dt:+d} h, ΔP={dpct:+.1f}%")

            panels.append((
                res,
                int(frac * 100),
                pos.astype(float),
                erosion.astype(float),
                base_remain.astype(float),
                total.astype(float),
            ))

    # -----------------------------------------------------------------
    # Save ALL data (Figure 2(a) + Figure 2(b)) to one Excel workbook
    # -----------------------------------------------------------------
    out_xlsx = BASE / "Figure 2(a)(b)_data.xlsx"
    hours_24 = hours.copy()

    with pd.ExcelWriter(out_xlsx) as writer:
        # ---- Figure 2(a): 6 panels (3 scenarios × weekday/weekend) ----
        for scen_name, day_wd, day_we in panel_cache:
            for day_type, data_dict in (("weekday", day_wd), ("weekend", day_we)):
                df_a = pd.DataFrame({
                    "Hour": hours_24,
                    "Baseload_GWh": data_dict["base"],
                    "AI_training_GWh": data_dict["tr"],
                    "AI_API_GWh": data_dict["api"],
                    "AI_conversation_GWh": data_dict["conv"],
                    "Total_GWh": data_dict["total"],
                    # metadata
                    "Region": REGION_FOR_PLOT,
                    "AI_computing_scenario": scen_name,
                    "AI_projection_scenario": CAP_SOURCE_SHEET,
                    "Renewable_resource": "None",
                    "Renewable_adoption_fraction": 0.0,
                    "Renewable_adoption_percent": 0,
                    "Renewable_scenario": "No on-site AI renewables (grid RE only)",
                    "Season": SEASON_FOR_PLOT,
                    "Day_type": day_type,
                })
                scen_tag = re.sub(r"[^A-Za-z0-9]+", "", scen_name) or "Scenario"
                day_tag = "wd" if day_type == "weekday" else "we"
                sheet_name = f"2a_{scen_tag}_{day_tag}"
                df_a.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        # ---- Figure 2(b): 6 panels (3 resources × 2 adoptions) ----
        for res, pct, pos, erosion, base_remain, total in panels:
            df_b = pd.DataFrame({
                "Hour": hours_24,
                "Base_total_GWh": base_wd,
                "Base_remain_GWh": base_remain,
                "Onsite_cover_GWh": erosion,
                "Extra_load_GWh": pos,
                "Total_GWh": total,
                # metadata
                "Region": REGION_FOR_PLOT,
                "AI_computing_scenario": baseline_name,
                "AI_projection_scenario": CAP_SOURCE_SHEET,
                "Renewable_resource": res,
                "Renewable_adoption_fraction": pct / 100.0,
                "Renewable_adoption_percent": pct,
                "Renewable_scenario": f"{res} {pct}%",
                "Season": SEASON_FOR_PLOT,
                "Day_type": "weekday",
            })
            sheet_name = f"2b_{res}_{pct}pct"
            df_b.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print("\nSaved Figure 2(a)(b) data:", out_xlsx)

if __name__ == "__main__":
    main()
