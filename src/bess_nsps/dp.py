"""
Dynamic programming on a battery-power grid and a discretised SoC.

State: SoC, active DG count and previous power action. SoC stays continuous, so
every returned trajectory obeys the SoC dynamics exactly; states of one (n, b)
slice that fall in the same SoC cell keep the best path, judged on fuel plus the
fuel that restores SoC_0. SOC_BINS_PER_STEP sets the cell width (the paper's
N_SoC). States that can no longer close on SoC_0 are dropped as infeasible.
Power at t applies over (t-1, t]; fuel and SoC use the same interval.
Initial battery power is zero. The last stage takes its power from the SoC step
(paper Eq. 8), off the grid if needed, so that SoC_T equals SoC_0 (paper Eq. 10)
with no tolerance and no free energy. Ramps limit the power change per unit across
commitment changes too, as the MATLAB reference does; a unit that starts or stops
is not ramped from or to zero, and there are no minimum ON/OFF times.

Optimality is restricted to this control grid, this SoC resolution and these
plant assumptions. Exceeding a resource cap raises an error.
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

SOC_BINS_PER_STEP = 4 # SoC cells per SoC change of full power over a median stage.
STATES_CAP = 100_000  # Max states per (t,n,b); exceeding a cap aborts the solve.
TOTAL_STATES_CAP = 30_000_000
TRANSITIONS_CAP = 100_000_000_000

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

def _soc_resolution(p_grid: np.ndarray, dt_vec: np.ndarray, bes: BESSpec) -> float:
    # SoC cell width: a fraction of what full power moves over a median stage.
    step = float(np.max(np.abs(p_grid))) * float(np.median(dt_vec[1:])) / 60.0 / bes.e_kwh
    return step / SOC_BINS_PER_STEP if step > 0 else 1.0

def _best_per_cell(idx: np.ndarray, cell: np.ndarray) -> np.ndarray:
    # idx is sorted by merit, so the first state seen in each SoC cell is the best.
    low = int(cell.min())
    first = np.full(int(cell.max()) - low + 1, -1, dtype=np.int32)
    first[cell[::-1] - low] = idx[::-1]
    return first[first >= 0]

def _ramp_ok(pg, pg_prev, dt: float, dg: DGSpec):
    # Per-unit ramp limits between consecutive setpoints.
    step = pg - pg_prev
    return (step <= dg.ramp_up_kw_per_min * dt) & (step >= dg.ramp_dn_kw_per_min * dt)

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
    b_zero = int(np.flatnonzero(p_grid == 0.0)[0])
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
    if np.any(feas_gen & ~np.isfinite(sfoc)):
        raise ValueError("SFOC curve must cover all feasible generator powers")

    # ---------- t = 0 ----------
    # Each stage is a table of states sorted by merit: (cost, soc, ni, b, parent).
    keep = [ni for ni in range(N) if feas_gen[0, ni, b_zero]]
    if not keep:
        raise RuntimeError("No feasible closed-SoC trajectory on the configured power grid.")
    size = len(keep)
    if size > TOTAL_STATES_CAP:
        raise MemoryError("DP state resource limit exceeded; shorten the horizon or grid")
    stages = [(np.zeros(size), np.full(size, soc0), np.array(keep, dtype=np.int16),
               np.full(size, b_zero, dtype=np.int16), np.full(size, -1, dtype=np.int32))]
    states = size
    transitions = 0
    res = _soc_resolution(p_grid, dt_vec, bes)
    # SoC per minute at full power: a state beyond these rates can never close on SoC_0.
    rise = -cfg.alpha_min * bes.pmax_kw * bes.eta_c / 60.0 / bes.e_kwh
    fall = cfg.alpha_max * bes.pmax_kw / bes.eta_d / 60.0 / bes.e_kwh
    # Merit counts the fuel that restores SoC_0, or a state that spent more of the
    # battery would win its cell only to pay that energy back at the closure [kg].
    fuel_per_soc = bes.e_kwh * float(np.median(dg.sfoc_g_per_kwh)) / 1000.0

    # ---------- recursion ----------
    for t in range(1, T - 1):
        dt = float(dt_vec[t])
        cost_p, soc_p, ni_p, b_p, _ = stages[-1]
        left = float(t_min[T-1] - t_min[t])
        floor = max(soc_min, soc0 - rise * left - 1e-12)
        ceiling = min(soc_max, soc0 + fall * left + 1e-12)
        groups, group_of = np.unique(ni_p.astype(np.int64) * B + b_p, return_inverse=True)
        # allowed[n, b, group]: every target (n, b) against every previous (n, b) group.
        allowed = _ramp_ok(pdg[t][:, :, None], pdg[t-1, groups // B, groups % B][None, None, :], dt, dg)
        new = []
        for b in range(B):
            soc_next = soc_p + _soc_update(0.0, float(p_grid[b]), dt, bes)
            in_band = (floor <= soc_next) & (soc_next <= ceiling)
            cell = np.rint((soc_next - soc0) / res).astype(np.int64)
            cell[soc_next == soc0] = cell.min() - 1  # An idle battery never merges, so it bounds the optimum.
            for ni in np.flatnonzero(feas_gen[t, :, b]):
                idx = np.flatnonzero(in_band & allowed[ni, b, group_of])
                transitions += idx.size
                if transitions > TRANSITIONS_CAP:
                    raise RuntimeError("DP resource limit: TRANSITIONS_CAP exceeded; shorten the horizon or grid")
                if idx.size == 0:
                    continue
                pick = _best_per_cell(idx, cell[idx])
                states += pick.size
                if pick.size > STATES_CAP or states > TOTAL_STATES_CAP:
                    raise MemoryError("DP state resource limit exceeded; shorten the horizon or grid")
                fuel_kg = (ni + 1) * pdg[t, ni, b] * sfoc[t, ni, b] * (dt/60.0) / 1000.0
                new.append((cost_p[pick] + fuel_kg, soc_next[pick],
                            np.full(pick.size, ni, dtype=np.int16),
                            np.full(pick.size, b, dtype=np.int16), pick))
        if not new:
            raise RuntimeError("No feasible closed-SoC trajectory on the configured power grid.")
        cost, soc, ni_new, b_new, parent = map(np.concatenate, zip(*new))
        order = np.argsort(cost + fuel_per_soc * (soc0 - soc), kind="stable")
        stages.append((cost[order], soc[order], ni_new[order], b_new[order], parent[order]))

    # ---------- terminal stage: the SoC step back to SoC_0 sets the power (Eq. 8, Eq. 10) ----------
    dt = float(dt_vec[T-1])
    cost_p, soc_p, ni_p, b_p, _ = stages[-1]
    need = soc0 - soc_p
    p_last = -need * bes.e_kwh / (dt/60.0) * np.where(need < 0, bes.eta_d, 1.0 / bes.eta_c)
    pdg_prev = pdg[T-2, ni_p, b_p]
    usable = (p_last >= cfg.alpha_min * bes.pmax_kw) & (p_last <= cfg.alpha_max * bes.pmax_kw)
    best = None
    for ni in range(N):
        pg = (p_load_kw[T-1] - p_last) / (ni + 1)
        ok = usable & (((pg >= pmin) & (pg <= pmax)) | (pg == 0.0)) & _ramp_ok(pg, pdg_prev, dt, dg)
        idx = np.flatnonzero(ok)
        transitions += idx.size
        if transitions > TRANSITIONS_CAP:
            raise RuntimeError("DP resource limit: TRANSITIONS_CAP exceeded; shorten the horizon or grid")
        if idx.size == 0:
            continue
        rate = sfoc_interp_g_per_kwh(np.maximum(pg[idx], 0.0), dg.p_grid_kw, dg.sfoc_g_per_kwh)
        if not np.all(np.isfinite(rate)):
            raise ValueError("SFOC curve must cover all feasible generator powers")
        total = cost_p[idx] + (ni + 1) * pg[idx] * rate * (dt/60.0) / 1000.0
        i = int(np.argmin(total))
        if (best is None) or (total[i] < best[0]):
            best = (float(total[i]), ni, int(idx[i]))
    if best is None:
        raise RuntimeError("No feasible closed-SoC trajectory on the configured power grid.")

    # ---------- backtracking ----------
    best_cost, ni_end, parent = best
    p_end = float(p_last[parent])
    soc_opt    = np.zeros(T)
    n_opt      = np.zeros(T, dtype=int)
    p_bess_opt = np.zeros(T)
    p_dg_opt   = np.zeros(T)

    soc_opt[T-1]    = soc0
    n_opt[T-1]      = ni_end + 1
    p_bess_opt[T-1] = p_end
    p_dg_opt[T-1]   = (p_load_kw[T-1] - p_end) / (ni_end + 1)
    for t in range(T - 2, -1, -1):
        _, soc, ni_new, b_new, parents = stages[t]
        soc_opt[t]    = soc[parent]
        n_opt[t]      = ni_new[parent] + 1
        p_bess_opt[t] = float(p_grid[b_new[parent]])
        p_dg_opt[t]   = pdg[t, ni_new[parent], b_new[parent]]
        parent = parents[parent]

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
