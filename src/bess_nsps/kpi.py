"""
kpi.py — derive key performance indicators (KPIs) from DP outputs


This module computes a small set of engineering KPIs from the optimal
trajectories returned by the DP solver. The KPIs are meant to compare BESS
sizing candidates on fuel, stress, and footprint proxies.


KPIs provided
-------------
    - fuel_kg: total fuel mass [kg] — populated by caller from DP.
    - c_rate_mean_per_h: average C-rate magnitude over the horizon [1/h].
    - dod_mean: mean depth-of-discharge used across the horizon (max(SoC)-min(SoC)).
    - efc_per_year: effective full cycles per year (simple proxy model).
    - energy_throughput_kwh:total absolute battery energy processed [kWh].
    - t_backup_min: backup time at rated power within the usable SoC band [min].
    - volume_m3: volumetric footprint proxy: energy x (m³/kWh).
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Optional
from .models import BESSpec

@dataclass
class KPIs:
    # Container for computed performance metrics.
    """ Notes
        -----
        fuel_kg is left as NaN by compute_kpis function and filled by the caller from the DP result (which holds the actual optimal fuel). Keeping fuel here allows writing a single CSV with both DP and BESS stress metrics.
    """
    fuel_kg: float
    c_rate_mean_per_h: float          # mean |C-rate| over the mission [1/h]
    dod_mean: float                   # SoC span over the mission [0..1]
    efc_per_year: float               # MATLAB-style annual EFD cycles
    energy_throughput_kwh: float      # Σ |P_bess| · Δt  [kWh]
    t_backup_min: float               # backup time at DoD = 0.6  [min]
    volume_m3: float                  # geometry-based volume proxy [m^3]

def compute_bess_volume_m3(
    bes,
    batt_pmax_kw: float,
    req_v: float = 660.0 * 1.35,      # grid requested voltage
    esm_nr: int = 4,                  # ESM number
    module_voltage_v: float = 54.0,   # module voltage used in MATLAB
    module_energy_kwh: float = 3.5,   # module energy (kWh)
    string_h_m: float = 2.550,        # string/pack height (m)
    string_w_m: float = 1.303,        # string/pack width  (m)
    string_l_m: float = 0.632,        # string/pack length (m)
    string_energy_kwh: float = 56.0,  # energy per pre-defined string/pack
    use_ceil: bool = True,            # mimic MATLAB ceil() behavior
) -> float:
    """
    Geometry-based BESS volume to mirror the MATLAB layout calculation.
    """
    # 1) Backup time from BESS spec (min); same as MATLAB batt.time
    t_backup_min = bes.backup_minutes()             # = E/P * DOD * 60

    # 2) Split power across ESMs
    esm_pmax_kw = batt_pmax_kw / max(esm_nr, 1)

    # 3) ESM energy sized for backup time
    esm_e_kwh = esm_pmax_kw * (t_backup_min / 60.0)

    # 4) Series/parallel module counts (MATLAB uses fractional here; old lines used ceil)
    modules_s = req_v / module_voltage_v
    modules_p = esm_e_kwh / module_energy_kwh

    # 5) Total modules per ESM: modules_g * modules_s, with modules_g = modules_p / modules_s
    modules_n = modules_p  # algebra simplifies as in MATLAB

    # 6) Number predefined "strings" (packs) per ESM
    strings_modules = string_energy_kwh / module_energy_kwh
    eff_no = modules_n / strings_modules
    if use_ceil:
        eff_no = math.ceil(eff_no)

    # 7) Volume = pack volume × effective number of strings × number of ESMs
    string_vol_m3 = string_h_m * string_w_m * string_l_m
    total_volume_m3 = string_vol_m3 * eff_no * esm_nr
    return float(total_volume_m3)

def compute_kpis(p_bess_kw: np.ndarray,
                 dt_min: np.ndarray,
                 soc: np.ndarray,
                 bes: BESSpec,
                 days_year: int = 210,          # days of manoeuvre per year
                 days_leg: int = 12,            # average navigation period
                 dod_max: float = 0.60,
                 volume_method: str = "geometry",
                 vol_density_m3_per_kwh: float = 0.0,
                 batt_pmax_kw: Optional[float] = None
                 ) -> KPIs:
    
    p_bess_kw = np.asarray(p_bess_kw, float).ravel()
    dt_min = np.asarray(dt_min, float).ravel()
    soc = np.asarray(soc, float).ravel()
    
    # Average absolute C‑rate (per hour).
    c_rate = np.abs(p_bess_kw) / bes.e_kwh
    c_rate_mean = float(np.nanmean(c_rate))

    # DoD mean (span over the mission) (robust to 1-D or 2-D)
    if soc.ndim == 1:
        span_per_leg = np.array([np.nanmax(soc) - np.nanmin(soc)], dtype=float)
    elif soc.ndim == 2:
        span_per_leg = np.nanmax(soc, axis=1) - np.nanmin(soc, axis=1)
        span_per_leg = np.asarray(span_per_leg, float).ravel()
    else:
        raise ValueError(f"Expected soc to be 1-D or 2-D, got shape {soc.shape}")

    dod_mean = float(np.nansum(span_per_leg) / dod_max)   # if band is 0–1.0
    
    # Energy throughput: integral of |P_bess| dt [kWh]
    e_through = float(np.nansum(np.abs(p_bess_kw) * (dt_min/60.0)))
    
    # Annual Equivalent Full Discharge Cycles. The factor 2 assumes a full cycle corresponds to charge + discharge between the reference DoD band.
    efc = 2.0 * (days_year / days_leg) * (dod_mean / dod_max)
    
    # Backup time at rated power, using usable SOC band
    t_backup = float(bes.backup_minutes(dod_max=0.60))
    
    # Volume proxy
    if volume_method == "geometry":
        if batt_pmax_kw is None:
            raise ValueError("batt_pmax_kw is required for geometry-based volume.")
        vol_proxy = compute_bess_volume_m3(bes, batt_pmax_kw=batt_pmax_kw)
    else:
        # simple proxy by density
        vol_proxy = float(vol_density_m3_per_kwh * bes.e_kwh)

    return KPIs(
        fuel_kg = np.nan,
        c_rate_mean_per_h = c_rate_mean,
        dod_mean = dod_mean,
        efc_per_year = efc,
        energy_throughput_kwh = e_through,
        t_backup_min = t_backup,
        volume_m3 = vol_proxy
    )