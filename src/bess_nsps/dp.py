"""
    Dynamic Programming (DP) solver for load dispatch

    Glossary
    --------
    T : number of time steps
    L (=|S|) : number of SOC grid points
    N : number of installed DG units (and max online units)
    J[t,s,n-1] : optimal cost-to-go up to time t at SOC index s with n active DGs
    prev_index : backpointer storing (previous soc index, previous n)
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
    soc_opt: np.ndarray         # [-], optimal SoC trajectory
    n_active_opt: np.ndarray    # [-], optimal number of online DGs at each step
    p_bess_opt_kw: np.ndarray   # [kW], BESS power at each step
    p_dg_opt_kw: np.ndarray     # [kW], DG power set point at each step

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
        Minimum fuel cost and associated optimal trajectories.
    """

    # ---------- Basic dimensions ----------
    T = len(p_load_kw)                          # horizon length
    dt_min = np.diff(t_min, prepend=t_min[0])   # [min], step durations
    L = len(cfg.soc_grid)                       # number of SOC states
    N = cfg.ndg_installed                       # max number of DGs

    # --- Connectivity check: can we move by one grid step in one dt? ---
    band = float(bes.soc_max - bes.soc_min)                           # e.g., 0.60 for 20->80 %
    soc_step = float(np.min(np.diff(cfg.soc_grid)))                   # min grid spacing in fraction
    dt_pos = float(np.min(dt_min[1:]))                                # smallest positive step (min)
    dt_h_min = float(np.min(np.diff(t_min, prepend=t_min[0])) / 60.0) # [h]

    # ---- Feasibility diagnostic (SoC grid vs dt vs Pmax) ----
    delta_soc_maxUP  =  -cfg.alpha_min * bes.eta_c * bes.pmax_kw * (dt_pos/60.0) / bes.e_kwh        # (+) charge - max +Delta SoC per step ----- missing /60?
    delta_soc_maxDOWN = cfg.alpha_max * (1.0/bes.eta_d) * bes.pmax_kw * (dt_pos/60.0) / bes.e_kwh  # (-) discharge - max -Delta SoC per step ---- missing /60?
    # If alpha window is wide (e.g. [-1, 1]), this equals bes.pmax_kw both ways.

    # # # # dSoC_phys = min(delta_soc_maxUP, delta_soc_maxDOWN)

    # # # # # Minimum dt needed for current grid step
    # # # # # (hours; convert to s/min as you prefer)
    # # # # dt_h_needed_up = (soc_step * bes.e_kwh) / (Pchg_max * bes.eta_c)
    # # # # dt_h_needed_down = (soc_step * bes.e_kwh * bes.eta_d) / Pdis_max
    # # # # dt_h_needed = max(dt_h_needed_up, dt_h_needed_down)

    # # # # # Minimum number of grid points needed for current dt
    # # # # # Use a safety factor (e.g. 0.8) so transitions are comfortably feasible
    # # # # safety    = 0.8
    # # # # dSoC_target = safety * dSoC_phys
    # # # # L_needed  = int(np.ceil(band / dSoC_target)) + 1

    # # # # print(
    # # # #     f"[DP check] dt_min={dt_h_min*3600:.2f}s | "
    # # # #     f"grid_step={soc_step:.6f} | "
    # # # #     f"ΔSoC_phys/step={dSoC_phys:.6f} (up={delta_soc_maxUP:.6f}, down={delta_soc_maxDOWN:.6f}) | "
    # # # #     f"L_needed≈{L_needed} | dt_needed≈{dt_h_needed*60:.2f} min"
    # # # # )

    # # Hard guard (choose ONE of the two, or keep both as warnings)
    # if soc_step > dSoC_phys:
    #     raise RuntimeError(
    #         "SOC grid too coarse for current dt and Pmax.\n"
    #         f"grid_step={soc_step:.6f} > ΔSoC_phys/step={dSoC_phys:.6f}. "
    #         f"Action: increase L to ≥ {L_needed} or increase dt to ≥ {dt_h_needed*60:.2f} min."
    #     )
    # # Alternatively:
    # if dt_h_min < dt_h_needed:
    #     raise RuntimeError(
    #         "Time step too small for current SOC grid and Pmax.\n"
    #         f"dt_min={dt_h_min*60:.4f} min < dt_needed={dt_h_needed*60:.4f} min. "
    #         f"Action: increase dt or refine grid to L ≥ {L_needed}."
    #     )

    if soc_step > delta_soc_maxUP or soc_step > delta_soc_maxDOWN:
        raise RuntimeError(
            f"SOC grid too coarse for given P_batt_max and dt: "
            f"grid_step={soc_step:.5f}, max_up={delta_soc_maxUP:.5f}, max_down={delta_soc_maxDOWN:.5f}. "
            f"Use more SOC points (smaller step) or increase dt."
        )

    # ---------- DP tables (node initaliser) ----------
    INF = 1e20
    J = np.full((T, L, N), INF, dtype=float)                # cost‑to‑go
    prev_index = -np.ones((T, L, N, 2), dtype=int)          # backpointers: (s_prev, n_prev)
    p_bess_used = np.full((T, L, N), np.nan, dtype=float)   # chosen BESS dispatch
    p_dg_used = np.full((T, L, N), np.nan, dtype=float)     # resulting DG setpoint
    p_lowlimit = cfg.alpha_min * bes.pmax_kw                # BESS lower power limit
    p_highlimit = cfg.alpha_max * bes.pmax_kw               # BESS higher power limit

    # ---------- Root node (no battery usage) ----------
    # Start SOC at the middle of the allowed band unless specified otherwise
    soc0 = 0.5*(bes.soc_min + bes.soc_max)
    soc0_id = int(np.argmin(np.abs(cfg.soc_grid - soc0)))
    # Heuristic: minimum number of DGs to carry the first load at max fraction
    n0 = int(np.ceil((p_load_kw[0] / (dg.pmax_frac*dg.pmax_kw))))

    # ---------- time t = 0: seed feasible starting points ----------
    for n in range(1, N+1):
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
    # ---------- Computational complexity: O(T*L2*N2) - for each (t,s,n) the code scan all (s',n')-----------
    for t in range(1, T):
        # Stage - time index
        dt = dt_min[t]
        for s_index, soc in enumerate(cfg.soc_grid):
            for n in range(1, N+1):
                reached_cost = INF
                reached_state = None # (sp, n_prev-1, p_bess, pdg)

                # Consider all previous SoC and DG indices
                for sp in range(L):
                    # SoC index
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
                        p_bess = -(delta_soc / bes.eta_c) * bes.e_kwh / (dt/60.0) # missing /60?
                    else:
                        # discharging or equal -> p_bess ≥ 0
                        p_bess = -(delta_soc * bes.eta_d) * bes.e_kwh / (dt/60.0) # missing /60?

                    # Enforce both alpha window (if used) and bes.pmax_kw
                    if not (p_lowlimit <= p_bess <= p_highlimit):
                        continue  # transition infeasible at this step

                    p_bess = float(p_bess)

                    # --- Compute DG power from power balance with n active DGs ---
                    pdg = power_balance(p_load_kw[t], p_bess, n)

                    # --- DG power feasibility ---
                    if not (pdg == 0 or (dg.pmin_frac*dg.pmax_kw <= pdg <= dg.pmax_frac*dg.pmax_kw)):
                        continue

                    for npv in range(1,N+1):
                        # DGs index
                        """ Code improvement range(max(1, n-1), min(N, n+1)+1):
                        - Limit DG commitment changes: iterate n' only near n (e.g., n-1..n+1) instead of 1..N. That typically respects ramping/operational realism and collapses the N2 factor;
                        - Reject impossible SOC jumps early: with the fix above transitions that require |p_bess| > pmax are skipped. That slashes the L2 factor."""
                        prev_cost = J[t-1, sp, npv-1]
                        if not np.isfinite(prev_cost):
                            # skip unreachable previous states
                            continue

                        # --- Ramp‑rate feasibility against previous DG setpoint ---
                        pdg_prev = p_dg_used[t-1, sp, npv-1]
                        if np.isfinite(pdg_prev):
                            dp = pdg - pdg_prev
                            if dp > dg.ramp_up_kw_per_min * dt:
                                continue
                            if dp < dg.ramp_dn_kw_per_min * dt:
                                continue

                        # --- Stage fuel cost for n identical DGs ---
                        sfoc = sfoc_interp_g_per_kwh(
                            np.full((1,), max(pdg, 0.0)),
                            dg.p_grid_kw,
                            dg.sfoc_g_per_kwh
                        )[0]            # [g/kWh]
                        trans_cost = n * pdg * sfoc * (dt/60.0) / 1000.0 # [g/kWh] -> [kg]
                        cost = prev_cost + trans_cost

                        # Keep the best predecessor
                        if cost < reached_cost:
                            reached_cost = cost
                            reached_state = (sp, npv-1, p_bess, pdg)

                # Write DP table if a feasible predecessor was found
                if reached_state is not None:
                    J[t, s_index, n-1] = reached_cost
                    prev_index[t, s_index, n-1] = (reached_state[0], reached_state[1])
                    p_bess_used[t, s_index, n-1] = reached_state[2]
                    p_dg_used[t, s_index, n-1]   = reached_state[3]
        
        # ---------- Check that the initial state is not isolated ----------
        if not np.any(np.isfinite(J[t, soc0_id, :])):
            raise RuntimeError(
                f"No reachable states at t={t} for SoC index {soc0_id} "
                f"(SoC={cfg.soc_grid[soc0_id]:.3f}). "
                "Use more SOC points in the grid or relax constraints."
            )

    # ---------- Choose terminal n at final time and reconstruct policy ----------
    sT = soc0_id  # same SoC index at end as at start

    n_end = int(np.ceil(p_load_kw[-1] / (dg.pmax_frac*dg.pmax_kw)))
    row = J[-1, sT, :]
    if not np.isfinite(row[n_end-1]):
        # No feasible path with the required n_end at fixed SOC
        # Fall back to the nearest *reachable* n that still covers the load
        mask = np.isfinite(row)
        if not mask.any():
            raise RuntimeError("No feasible terminal state with fixed SOC.")
        # Prefer n >= n_end (cover load), otherwise the cheapest reachable n
        candidates = np.where(mask)[0] + 1
        n_ge = candidates[candidates >= n_end]
        if len(n_ge):
            n_end = int(n_ge[np.argmin(row[n_ge-1])])
        else:
            n_end = int(candidates[np.argmin(row[candidates-1])])

    # Use that terminal commitment; do NOT argmin over n
    cost = float(J[-1, sT, n_end-1])    # np indexing is 0-based
    if not np.isfinite(cost):
        raise RuntimeError(
            f"DP infeasible with fixed SoC: cannot end at s={sT} (SoC={cfg.soc_grid[sT]:.3f}) "
            f"and n={n_end}. Adjust SOC grid or dt."
        )

    # Allocate trajectories for reconstruction
    soc_opt = np.zeros(T, dtype=float)
    n_opt = np.zeros(T, dtype=int)
    p_bess_tr = np.zeros(T, dtype=float)
    p_dg_tr   = np.zeros(T, dtype=float)
    print("finite end states at sT:", np.where(np.isfinite(J[-1, soc0_id, :]))[0] + 1)

    # Backtrack
    s = sT
    n = n_end - 1   # store as 0‑based index while backtracking
    for t in range(T-1, -1, -1):
        soc_opt[t] = cfg.soc_grid[s]
        n_opt[t] = n + 1
        p_bess_tr[t] = p_bess_used[t, s, n]
        p_dg_tr[t]   = p_dg_used[t, s, n]
        if t > 0:
            ps, pn = prev_index[t, s, n]
            if ps < 0 or pn < 0:
                raise RuntimeError(f"Unreachable state at t={t}, s={s}, n={n+1}")
            s, n = int(ps), int(pn)

    # Package results
    return DPResult(
        cost_kg=cost,
        soc_opt=soc_opt,
        n_active_opt=n_opt,
        p_bess_opt_kw=p_bess_tr,
        p_dg_opt_kw=p_dg_tr
    )