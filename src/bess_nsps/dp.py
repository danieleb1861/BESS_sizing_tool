"""
    Dynamic Programming (DP) solver for load dispatch

    Glossary
    --------
    T : number of time steps
    L (=|S|) : number of SOC grid points
    N : number of installed DG units (and max online units)
    J[t,s,n-1] : optimal cost‑to‑go up to time t at SOC index s with n active DGs
    prev_idx : backpointer storing (previous soc index, previous n)
    p_*_used : action records to reconstruct optimal control signals
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from .models import DGSpec, BESSpec, sfoc_interp_g_per_kwh, power_balance, bess_soc_step

# ------------------------------ Data containers -------------------------------

@dataclass
class DPConfig:
    # Static configuration for the DP grid.
    ndg_installed: int          # Total number of identical DG units installed (N).
    soc_grid: np.ndarray        # Discrete SOC grid (values ∈ [0,1]) used as the DP state dimension.
    alpha_min: float            # Allowed minimum loading of the BESS
    alpha_max: float            # Allowed maximum loading of the BESS

@dataclass
class DPResult:
    # Outputs reconstructed from the optimal policy.
    cost_kg: float              # [kg], total MDO consumption
    soc_traj: np.ndarray        # [-], optimal SoC trajectory
    n_active_traj: np.ndarray   # [-], optimal number of online DGs at each step
    p_bess_traj_kw: np.ndarray  # [kW], BESS power at each step
    p_dg_traj_kw: np.ndarray    # [kW], DG power set point at each step

# ------------------------------ Dynamic Programming SOLVER -------------------------------

def run_dp(p_load_kw: np.ndarray,
           t_min: np.ndarray,
           dg: DGSpec,
           bes: BESSpec,
           cfg: DPConfig) -> DPResult:
    # Solve the dispatch DP and reconstruct optimal trajectories.
    """ Parameters
        ----------
        p_load_kw : (T,) array
        Total system load at each step [kW].
        t_min : (T,) array
        Cumulative time array [minutes]. Step size is diff(t_min).
        dg : DGSpec
        Diesel generator specification (pmax, min/max fractions, SFOC curve, ramp rates, etc.).
        bes : BESSpec
        Battery specification (energy, power limits, efficiencies, SOC bounds).
        cfg : DPConfig
        DP grid configuration (SOC discretization and DG count window).

        Returns
        -------
        DPResult
        Minimal fuel cost and associated optimal trajectories.
    """

    # ---------- basic dimensions ----------
    T = len(p_load_kw)                          # horizon length
    dt_min = np.diff(t_min, prepend=t_min[0])   # [min], step durations
    L = len(cfg.soc_grid)                       # number of SOC states
    N = cfg.ndg_installed                       # max number of DGs

    # ---------- DP tables ----------
    INF = 1e30
    J = np.full((T, L, N), INF, dtype=float) # cost‑to‑go
    prev_idx = -np.ones((T, L, N, 2), dtype=int) # backpointers: (s_prev, n_prev)
    p_bess_used = np.full((T, L, N), np.nan, dtype=float) # chosen BESS dispatch
    p_dg_used = np.full((T, L, N), np.nan, dtype=float) # resulting DG setpoint

    # ---------- initial state (SOC guess and initial commitment heuristic) ----------

    # Start SOC at the middle of the allowed band unless specified otherwise
    soc0 = 0.5*(bes.soc_min + bes.soc_max)
    soc0_id = int(np.argmin(np.abs(cfg.soc_grid - soc0)))
    # Heuristic: minimum number of DGs to carry the first load at max fraction
    n0 = max(1, int(np.ceil((p_load_kw[0] / dg.pmax_kw) / dg.pmax_frac)))
    n0 = min(n0, N)

    # ---------- time t = 0: seed feasible starting points ----------
    for n in range(1, n0+1):
        p_bess0 = 0.0
        pdg = power_balance(p_load_kw[0], p_bess0, n)
        if pdg == 0 or (dg.pmin_frac*dg.pmax_kw <= pdg <= dg.pmax_frac*dg.pmax_kw):
            # SFOC interpolation expects non‑negative power
            sfoc = sfoc_interp_g_per_kwh(
                np.full((1,), max(pdg, 0.0)),
                dg.p_grid_kw,
                dg.sfoc_g_per_kwh
            )[0]                            # [g/kWh]
            # Fuel mass over the step: n * (p_dg_each [kW]) * sfoc [g/kWh] * dt[h] -> g
            trans_cost = n * pdg * sfoc * (dt_min[0]/60.0) / 1000.0  # g -> kg
            J[0, soc0_id, n-1] = trans_cost
            p_bess_used[0, soc0_id, n-1] = p_bess0
            p_dg_used[0, soc0_id, n-1] = pdg

    # ---------- DP recursion for t = 1,...,T-1 -----------
    # ---------- O(T*L2*N2) because for each (t,s,n) the code scan all (s',n')-----------
    for t in range(1, T):
        dt = dt_min[t]
        for s_idx, soc in enumerate(cfg.soc_grid):
            for n in range(1, N+1):
                best_cost = INF
                best = None # (sp, n_prev-1, p_bess, pdg)

                # Consider all previous SoC indices and previous DG counts
                for sp in range(L):
                    for npv in range(max(1, n-1), min(N, n+1)+1):
                        """ Code improvement range(max(1, n-1), min(N, n+1)+1):
                        - Limit DG commitment changes: iterate n' only near n (e.g., n-1..n+1) instead of 1..N. That typically respects ramping/operational realism and collapses the N2 factor;
                        - Reject impossible SOC jumps early: with the fix above transitions that require |p_bess| > pmax are skipped. That slashes the L2 factor."""
                        prev_cost = J[t-1, sp, npv-1]
                        if not np.isfinite(prev_cost):
                            # skip unreachable states
                            continue
                        
                        soc_prev = cfg.soc_grid[sp]

                        # --- choose battery power that realises the transition soc_prev -> soc ---
                        # --- compute required battery power from SOC change ---
                        # pbatt = -delta_soc * E / dt, with efficiency handled by
                            # if charging (soc > soc_prev): delta_soc /= eta_c
                            # if discharging (soc <= soc_prev): delta_soc *= eta_d
                        # Here SOC is in [0,1], dt in minutes, E in kWh, so power is kW via /(dt/60).
                        delta_soc = soc - soc_prev # >0 -> charging, <0 -> discharging
                        if delta_soc > 0:
                            # charging -> p_bess < 0
                            p_bess = -(delta_soc / max(bes.eta_c, 1e-9)) * bes.e_kwh / (dt/60.0)
                        else:
                            # discharging or equal -> p_bess ≥ 0
                            p_bess = -(delta_soc * bes.eta_d) * bes.e_kwh / (dt/60.0)

                        # Enforce both alpha window (if used) and bes.pmax_kw
                        p_lo = cfg.alpha_min * bes.pmax_kw
                        p_hi = cfg.alpha_max * bes.pmax_kw
                        if not (p_lo - 1e-9 <= p_bess <= p_hi + 1e-9):
                            continue  # transition infeasible at this step

                        # Validate that chosen p_bess lands on the candidate grid point
                        soc_next = bess_soc_step(soc_prev, p_bess, dt, bes)
                        if abs(soc_next - soc) > 1e-6:
                            continue

                        p_bess = float(p_bess)

                        # --- compute DG power from power balance with n active DGs ---
                        pdg = power_balance(p_load_kw[t], p_bess, n)
                        if not (pdg == 0 or (dg.pmin_frac*dg.pmax_kw <= pdg <= dg.pmax_frac*dg.pmax_kw)):
                            continue

                        # --- ramp‑rate feasibility against previous DG setpoint ---
                        pdg_prev = p_dg_used[t-1, sp, npv-1]
                        if np.isfinite(pdg_prev):
                            dp = pdg - pdg_prev
                            if dp > dg.ramp_up_kw_per_min * dt:
                                continue
                            if dp < dg.ramp_dn_kw_per_min * dt:
                                continue

                        # --- stage fuel cost for n identical DGs ---
                        sfoc = sfoc_interp_g_per_kwh(
                            np.full((1,), max(pdg, 0.0)),
                            dg.p_grid_kw,
                            dg.sfoc_g_per_kwh
                        )[0]
                        trans_cost = n * pdg * sfoc * (dt/60.0) / 1000.0
                        tot = prev_cost + trans_cost

                        # keep the best predecessor
                        if tot < best_cost:
                            best_cost = tot
                            best = (sp, npv-1, p_bess, pdg)

                # write DP table if a feasible predecessor was found
                if best is not None:
                    J[t, s_idx, n-1] = best_cost
                    prev_idx[t, s_idx, n-1] = (best[0], best[1])
                    p_bess_used[t, s_idx, n-1] = best[2]
                    p_dg_used[t, s_idx, n-1]   = best[3]

    # ---------- choose terminal n at final time and reconstruct policy ----------
    sT = soc0_id                            # target terminal SOC index
    end_slice = J[-1, sT, :]                # costs at t=T-1 with SOC=sT
    n_end = int(np.argmin(end_slice)) + 1   # best terminal DG count
    cost = float(end_slice[n_end-1])        # minimal total fuel [kg]

    # Allocate trajectories for reconstruction
    soc_traj = np.zeros(T, dtype=float)
    n_traj = np.zeros(T, dtype=int)
    p_bess_tr = np.zeros(T, dtype=float)
    p_dg_tr   = np.zeros(T, dtype=float)

    # Backtrack
    s = sT
    n = n_end - 1   # store as 0‑based index while backtracking
    for t in range(T-1, -1, -1):
        soc_traj[t] = cfg.soc_grid[s]
        n_traj[t] = n + 1
        p_bess_tr[t] = p_bess_used[t, s, n]
        p_dg_tr[t]   = p_dg_used[t, s, n]
        if t > 0:
            ps, pn = prev_idx[t, s, n]
            s, n = int(ps), int(pn)

    # Package results
    return DPResult(
        cost_kg=cost,
        soc_traj=soc_traj,
        n_active_traj=n_traj,
        p_bess_traj_kw=p_bess_tr,
        p_dg_traj_kw=p_dg_tr
    )
