"""
Exact dynamic programming on a finite battery-power grid.

State: continuous SoC, active DG count and previous power action. Only
identical floating-point states are merged; no bucketing or pruning is used.
Power at t applies over (t-1, t]; fuel and SoC use the same interval.
Initial battery power is zero; terminal SoC equals the initial midpoint
within SOC_TOL. Ramps include generator starts/stops, with zero minimum
ON/OFF times as in the reference configuration.

Optimality is restricted to this control grid and these plant assumptions,
within floating-point arithmetic. Exceeding a resource cap raises an error
without returning an approximate solution.
"""

from dataclasses import dataclass
import numpy as np
from .models import DGSpec, BESSpec, sfoc_interp_g_per_kwh

# ------------------------------ Data containers -------------------------------

@dataclass
class DPConfig:
    ndg_installed: int            # number of installed DG units (N)
    nbatt_steps: int              # discretisation points for p_bess ∈ [−Pmax,+Pmax]
    alpha_min: float              # min fraction of Pmax allowed (−1.0 = full charge)
    alpha_max: float              # max fraction of Pmax allowed (+1.0 = full discharge)

@dataclass
class DPResult:
    cost_kg: float
    soc_opt: np.ndarray
    n_active_opt: np.ndarray
    p_bess_opt_kw: np.ndarray
    p_dg_opt_kw: np.ndarray

# ------------------------------ Numerical/resource limits ----------------------

SOC_TOL = 1e-12       # Terminal equality tolerance; states are never rounded.
STATES_CAP = 600      # Max states per (t,n,b); exceeding a cap aborts the solve.
TOTAL_STATES_CAP = 200_000
TRANSITIONS_CAP = 2_000_000

# ------------------------------ Helpers ---------------------------------------

def _soc_update(soc_prev: float, p_bess_kw: float, dt_min: float, bes: BESSpec) -> float:
    # Continuous SoC update with efficiencies.
    e = max(bes.e_kwh, 1e-12)
    if p_bess_kw >= 0:
        return soc_prev - (p_bess_kw * dt_min / 60.0) / e / max(bes.eta_d, 1e-12)
    else:
        return soc_prev - (p_bess_kw * dt_min / 60.0) / e * max(bes.eta_c, 1e-12)

def _ensure_zero(grid: np.ndarray) -> np.ndarray:
    # Ensure 0.0 is present in the control grid.
    return np.unique(np.r_[grid, 0.0])

# ------------------------------ DP solver -------------------------------------

def run_dp(p_load_kw: np.ndarray,
           t_min: np.ndarray,
           dg: DGSpec,
           bes: BESSpec,
           cfg: DPConfig) -> DPResult:

    # ---------- Horizon ----------
    p_load_kw = np.asarray(p_load_kw, float).ravel()
    T = int(len(p_load_kw))
    t_min = np.asarray(t_min, float).ravel()
    if (T < 2 or t_min.size != T or not np.all(np.isfinite(p_load_kw))
            or not np.all(np.isfinite(t_min)) or np.any(np.diff(t_min) <= 0)
            or np.any(p_load_kw < 0)):
        raise ValueError("DP needs matching finite loads and strictly increasing times (at least two samples)")
    if (cfg.ndg_installed < 1 or int(cfg.ndg_installed) != cfg.ndg_installed
            or cfg.nbatt_steps < 2 or int(cfg.nbatt_steps) != cfg.nbatt_steps
            or not (-1 <= cfg.alpha_min <= 0 <= cfg.alpha_max <= 1)):
        raise ValueError("Invalid DG count, power grid or battery power fractions")
    if not (np.isfinite(bes.e_kwh) and bes.e_kwh > 0
            and np.isfinite(bes.pmax_kw) and bes.pmax_kw >= 0
            and 0 <= bes.soc_min < bes.soc_max <= 1
            and 0 < bes.eta_c <= 1 and 0 < bes.eta_d <= 1):
        raise ValueError("Invalid battery rating, SoC bounds or efficiencies")
    if not (np.isfinite(dg.pmax_kw) and dg.pmax_kw > 0
            and 0 <= dg.pmin_frac <= dg.pmax_frac <= 1
            and np.isfinite(dg.ramp_up_kw_per_min) and dg.ramp_up_kw_per_min >= 0
            and np.isfinite(dg.ramp_dn_kw_per_min) and dg.ramp_dn_kw_per_min <= 0):
        raise ValueError("Invalid generator rating, loading bounds or ramp limits")
    dt_vec = np.diff(t_min, prepend=t_min[0])

    N = int(cfg.ndg_installed)
    if T * N * int(cfg.nbatt_steps) > TOTAL_STATES_CAP:
        raise MemoryError("Exact DP table allocation exceeds TOTAL_STATES_CAP; shorten the horizon or grid")
    p_grid = _ensure_zero(np.linspace(cfg.alpha_min*bes.pmax_kw,
                                      cfg.alpha_max*bes.pmax_kw,
                                      int(cfg.nbatt_steps)))
    B = p_grid.size
    if T * N * B > TOTAL_STATES_CAP:
        raise MemoryError("Exact DP table allocation exceeds TOTAL_STATES_CAP; shorten the horizon or grid")

    # ---------- Generator limits ----------
    pmin = dg.pmin_frac * dg.pmax_kw
    pmax = dg.pmax_frac * dg.pmax_kw

    # ---------- SoC params ----------
    soc_min, soc_max = bes.soc_min, bes.soc_max
    soc0 = 0.5*(soc_min + soc_max)

    # ---------- Precompute pdg & SFOC ----------
    pdg  = np.empty((T, N, B), float)
    sfoc = np.empty((T, N, B), float)
    for t in range(T):
        for ni in range(N):
            n = ni + 1
            pdg_tn = (p_load_kw[t] - p_grid) / n
            pdg[t, ni, :]  = pdg_tn
            sfoc[t, ni, :] = sfoc_interp_g_per_kwh(np.maximum(pdg_tn, 0.0),
                                                    dg.p_grid_kw, dg.sfoc_g_per_kwh)
    feas_gen = ((pdg >= pmin) & (pdg <= pmax)) | (pdg == 0.0)

    # ---------- DP tables ----------
    J = [[ [dict() for _ in range(B)] for _ in range(N) ] for _ in range(T)]
    states = 0
    transitions = 0

    # ---------- t = 0 ----------
    dt = float(dt_vec[0])
    for ni in range(N):
        n = ni + 1
        for b in range(B):
            if p_grid[b] != 0.0:
                continue
            if not feas_gen[0, ni, b]:
                continue
            soc1 = _soc_update(soc0, float(p_grid[b]), dt, bes)
            if not (soc_min <= soc1 <= soc_max):
                continue
            fuel_kg = n * pdg[0, ni, b] * sfoc[0, ni, b] * (dt/60.0) / 1000.0
            k = soc1
            cur = J[0][ni][b].get(k)
            if (cur is None) or (fuel_kg < cur[0]):
                J[0][ni][b][k] = (fuel_kg, soc1, None)
                states += 1

    # ---------- recursion ----------
    for t in range(1, T):
        dt = float(dt_vec[t])
        up = dg.ramp_up_kw_per_min * dt
        dn = dg.ramp_dn_kw_per_min * dt
        for ni in range(N):
            n = ni + 1
            for nip in range(N):
                for b_prev in range(B):
                    for k_prev, (cost_prev, soc_prev, _) in J[t-1][nip][b_prev].items():
                        pdg_prev = pdg[t-1, nip, b_prev]
                        for b in range(B):
                            transitions += 1
                            if transitions > TRANSITIONS_CAP:
                                raise RuntimeError("Exact DP resource limit: TRANSITIONS_CAP exceeded; no approximate result returned")
                            if not feas_gen[t, ni, b]:
                                continue
                            dp = pdg[t, ni, b] - pdg_prev
                            if dp > up or dp < dn:
                                continue
                            if ni > nip and pdg[t, ni, b] > up:
                                continue  # Newly started units ramp from zero.
                            if ni < nip and -pdg_prev < dn:
                                continue  # Stopped units must ramp down to zero.
                            soc_next = _soc_update(soc_prev, float(p_grid[b]), dt, bes)
                            if not (soc_min <= soc_next <= soc_max):
                                continue
                            if t == T - 1 and abs(soc_next - soc0) > SOC_TOL:
                                continue
                            fuel_kg = n * pdg[t, ni, b] * sfoc[t, ni, b] * (dt/60.0) / 1000.0
                            total = cost_prev + fuel_kg
                            k = soc_next
                            cur = J[t][ni][b].get(k)
                            if (cur is None) or (total < cur[0]):
                                if cur is None:
                                    states += 1
                                    if len(J[t][ni][b]) >= STATES_CAP or states > TOTAL_STATES_CAP:
                                        raise MemoryError("Exact DP state resource limit exceeded; no states discarded or approximate result returned")
                                J[t][ni][b][k] = (total, soc_next, (nip, b_prev, k_prev))

    # ---------- terminal condition: equal SoC within floating-point tolerance ----------
    best = None
    for ni in range(N):
        for b in range(B):
            for k, entry in J[T-1][ni][b].items():
                cost = entry[0]
                if (best is None) or (cost < best[0]):
                    best = (cost, T-1, ni, b, k)
    if best is None:
        raise RuntimeError("No feasible closed-SoC trajectory on the configured power grid.")

    # ---------- backtracking ----------
    best_cost, t, ni, b, k = best
    soc_opt    = np.zeros(T)
    n_opt      = np.zeros(T, dtype=int)
    p_bess_opt = np.zeros(T)
    p_dg_opt   = np.zeros(T)

    while t >= 0:
        cost, soc_val, prev = J[t][ni][b][k]
        soc_opt[t]    = soc_val
        n_opt[t]      = ni + 1
        p_bess_opt[t] = float(p_grid[b])
        p_dg_opt[t]   = pdg[t, ni, b]
        if prev is None:
            break
        nip, b_prev, k_prev = prev
        t, ni, b, k = t-1, nip, b_prev, k_prev

    # # dt vector (minutes) you already have in main/solver
    # dt = np.diff(t_min, prepend=t_min[0])

    # # reconstruct SoC purely from the returned p_bess trajectory
    # soc_recon = np.zeros_like(p_bess_opt_kw, dtype=float)
    # soc_recon[0] = soc_opt[0]
    # for t in range(1, len(p_bess_opt_kw)):
    #     pb = p_bess_opt_kw[t]
    #     if pb >= 0:
    #         ds = -(pb*dt[t]/60.0)/bes.e_kwh/max(bes.eta_d,1e-12)
    #     else:
    #         ds = -(pb*dt[t]/60.0)/bes.e_kwh*max(bes.eta_c,1e-12)
    #     soc_recon[t] = soc_recon[t-1] + ds

    # print(
    #     f"[SoC stats] min={soc_opt.min():.5f}, max={soc_opt.max():.5f}, "
    #     f"span={(soc_opt.max()-soc_opt.min())*100:.3f}%"
    # )
    # print(
    #     f"[consistency] max|soc_opt - soc_recon| = {np.max(np.abs(soc_opt - soc_recon)):.6e}"
    # )
    # print(
    #     f"[energetics] median|p_bess|={np.median(np.abs(p_bess_opt_kw)):.1f} kW, "
    #     f"median dt={np.median(dt):.3f} min, E={bes.e_kwh:.1f} kWh "
    #     f"→ typical ΔSoC per step ≈ "
    #     f"{np.median(np.abs(p_bess_opt_kw))*np.median(dt)/60.0/bes.e_kwh*100:.3f}%"
    # )

    return DPResult(best_cost,
                    soc_opt,
                    n_opt,
                    p_bess_opt,
                    p_dg_opt
                    )
