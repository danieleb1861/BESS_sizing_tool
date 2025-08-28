from dataclasses import dataclass
import numpy as np

@dataclass
class DGSpec:
    pmax_kw: float            # [kW] Rated power (per generator)
    pmin_frac: float = 0.20   # [-] lower bound fraction of pmax
    pmax_frac: float = 0.95   # [-] upper bound fraction of pmax
    ramp_up_kw_per_min: float = None
    ramp_dn_kw_per_min: float = None
    sfoc_g_per_kwh: np.ndarray = None  # [-] - shape (K,) - SFOC (specific fuel oil consumption) curve samples in g/kWh. Interpolated piecewise-linearly against p_grid_kw.
    p_grid_kw: np.ndarray = None       # [kW] - shape (K,) - Power grid where SFOC samples are defined.

    def __post_init__(self):
        # Derive default ramp limits if not provided
        # Example: reach pmax in ~20 s up, ~10 s down
        if self.ramp_up_kw_per_min is None:
            self.ramp_up_kw_per_min = self.pmax_kw / (20/60.0)  # [kW/min]
        if self.ramp_dn_kw_per_min is None:
            self.ramp_dn_kw_per_min = -self.pmax_kw / (10/60.0) # [kW/min]

@dataclass
class BESSpec:
    # Battery spec object providing energy capacity and efficiencies.
    pmax_kw: float          # [kW]
    e_kwh: float            # [kWh]
    soc_min: float = 0.20   # [-], minimum SoC
    soc_max: float = 0.80   # [-], maximum SoC
    eta_c: float = 0.97     # [-], charging efficiency
    eta_d: float = 0.94     # [-], discharging efficiency

    def backup_minutes(self) -> float:
        # Compute available backup time at rated power. It calculates how many minutes of backup are available when discharging at the battery's power limit, considering only the usable depth-of-discharge between soc_min and soc_max
        dod_max = self.soc_max - self.soc_min
        return (self.e_kwh / max(self.pmax_kw, 1e-9)) * dod_max * 60.0

def sfoc_interp_g_per_kwh(p_kw: np.ndarray, p_grid_kw: np.ndarray, sfoc: np.ndarray) -> np.ndarray:
    # Interpolate the SFOC [g/kWh] for arbitrary generator loading.
    # p_kw is the desired operating power array where SFOC is needed
    return np.interp(p_kw, p_grid_kw, sfoc)

def power_balance(p_load_kw: float, p_batt_kw: float, n_active: int) -> float:
    # Power balance: compute generator power setpoint to meet load at each instant.
    return (p_load_kw - p_batt_kw) / max(n_active, 1)

def bess_soc_step(soc_prev: float, p_bess_kw: float, dt_min: float, bes: 'BESSpec') -> float:
    # BESS state of charge
    beta = (bes.eta_c if p_bess_kw <= 0 else 1.0 / bes.eta_d)
    return soc_prev - beta * (p_bess_kw / max(bes.e_kwh, 1e-9)) * (dt_min / 60.0)