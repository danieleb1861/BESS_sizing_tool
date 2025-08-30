"""
Dynamic Programming solver for load dispatch.

- State at stage t (post-decision): (SoC_cont, n, b_prev_idx)
- Control at stage t+1            : b_idx  (battery power index)
- SoC propagation                 : continuous update
- Ramp constraint                 : compare pdg(t+1,n,b_idx) vs pdg(t,n_prev,b_prev)
- Cost                            : n * pdg * SFOC * (dt/60) / 1000  [kg]

ASCII state-transition diagram

          stage t                                stage t+1
   ┌─────────────────────┐    choose control b    ┌─────────────────────┐
   │ ( SoC_t ,           |                        │ ( SoC_{t+1} ,       │
   |   n_prev ,          │ ───────────────────▶  │    n ,              │
   │   b_prev )          │                        │    b                │
   └─────────────────────┘    fuel + feasibility  └─────────────────────┘
              ▲                                          |
              └────────────── backpointer ───────────────┘


Legend:
- SoC_t    : continuous battery state of charge at time t
- n_prev   : number of DGs online at time t
- b_prev   : previous battery control index
- b        : chosen control index at time t+1
- SoC_{t+1}: next SoC after applying control b over dt
- n        : number of DGs online at time t+1

Transition checks:
  - SoC_{t+1} must stay within [SoC_min, SoC_max]
  - Generator dispatch pdg(t+1,n,b) must be feasible
  - Ramp constraint: |pdg(t+1,n,b) - pdg(t,n_prev,b_prev)| ≤ ramp limits
  - Stage cost = n * pdg * SFOC * (dt/60) / 1000 [kg]

Each node encodes:
  - current SoC value (stored continuous, bucketed only for hashing)
  - current DG count
  - previous battery action index

"""

"""
           Many SoC trajectories (continuous floats)
                      │
               [HASHING] → group by epsilon-buckets
                      │
           Buckets with best cost inside each
                      │
               [PRUNING] → keep at most 600 per slice
                      │
          Next stage DP recursion continues...

Hashing prevents explosion due to floating-point noise. Otherwise it is possible to have “infinitely many” distinct SoC values, all only microns apart.
   Continuous SoC values
      |       |       |     
   0.4999  0.5000  0.5032

   ↓ hash into ε-buckets (width=5e-4)

   [ bucket 1000 ]  [ bucket 1006 ]
     contains 3        contains 2
     states             states

All values close to 0.50 are hashed into the same bucket key 1000.
In J[t][n][b], only one entry is stored for that bucket (the cheapest cost path).

Pruning prevents explosion due to combinatorial growth. Even with hashing, every stage multiplies possibilities.
Example: T=100, N=3, B=41, 600 states per slice → ~7.4e+10 states total.

Slice J[t][n=2][b=+200]
 ┌─────────────────────────┐
 │  1000 buckets in dict   │
 └─────────────────────────┘
             │
             └─ sort by cost, keep top 600
             ↓
 ┌─────────────────────────┐
 │   600 cheapest buckets  │
 └─────────────────────────┘

All candidate buckets (cost-sorted):
 [ cheap-1 ] [ cheap-2 ] ... [ cheap-600 ] [ expensive-601 ] ... [ expensive-1000 ]

Keep only left 600 ← PRUNING
This keeps the DP tables manageable in size, at the cost of possibly discarding some paths.

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

# ------------------------------ Hardcoded hashing/pruning params ----------------------

SOC_BUCKET   = 5e-4   # SoC hashing resolution. Smaller = smoother, more states. All SoCs within +-SOC_BUCKET (+-SOC_BUCKET/100 %) end up in the same integer bucket.
STATES_CAP   = 600    # Max states kept per (t,n,b) slice for pruning.

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
    return np.sort(grid if np.min(np.abs(grid)) < 1e-12 else np.r_[grid, 0.0])

def _bucket_soc(soc: float, soc_min: float, eps: float) -> int:
    # Map continuous SoC to an integer bucket key (hash only).
    return int(round((soc - soc_min) / max(eps, 1e-12)))

# ------------------------------ DP solver -------------------------------------

def run_dp(p_load_kw: np.ndarray,
           t_min: np.ndarray,
           dg: DGSpec,
           bes: BESSpec,
           cfg: DPConfig) -> DPResult:

    # ---------- Horizon ----------
    T = int(len(p_load_kw))
    t_min = np.asarray(t_min, float).ravel()
    dt_vec = np.diff(t_min, prepend=t_min[0])

    N = int(cfg.ndg_installed)
    p_grid = _ensure_zero(np.linspace(cfg.alpha_min*bes.pmax_kw,
                                      cfg.alpha_max*bes.pmax_kw,
                                      int(cfg.nbatt_steps)))
    B = p_grid.size

    # ---------- Generator limits ----------
    pmin = dg.pmin_frac * dg.pmax_kw
    pmax = dg.pmax_frac * dg.pmax_kw

    # ---------- SoC params ----------
    soc_min, soc_max = bes.soc_min, bes.soc_max
    soc0 = 0.5*(soc_min + soc_max)
    eps = SOC_BUCKET
    s0_key = _bucket_soc(soc0, soc_min, eps)

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
    feas_gen = ((pdg >= pmin) & (pdg <= pmax)) | np.isclose(pdg, 0.0)

    # ---------- DP tables ----------
    J = [[ [dict() for _ in range(B)] for _ in range(N) ] for _ in range(T)]

    # ---------- t = 0 ----------
    dt = float(dt_vec[0])
    for ni in range(N):
        n = ni + 1
        for b in range(B):
            if not feas_gen[0, ni, b]:
                continue
            soc1 = _soc_update(soc0, float(p_grid[b]), dt, bes)
            if not (soc_min <= soc1 <= soc_max):
                continue
            fuel_kg = n * pdg[0, ni, b] * sfoc[0, ni, b] * (dt/60.0) / 1000.0
            k = _bucket_soc(soc1, soc_min, eps)
            cur = J[0][ni][b].get(k)
            if (cur is None) or (fuel_kg < cur[0]):
                J[0][ni][b][k] = (fuel_kg, soc1, None)
        # prune
        for b in range(B):
            if len(J[0][ni][b]) > STATES_CAP:
                items = sorted(J[0][ni][b].items(), key=lambda kv: kv[1][0])[:STATES_CAP]
                J[0][ni][b] = dict(items)

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
                            if not feas_gen[t, ni, b]:
                                continue
                            dp = pdg[t, ni, b] - pdg_prev
                            if dp > up or dp < dn:
                                continue
                            soc_next = _soc_update(soc_prev, float(p_grid[b]), dt, bes)
                            if not (soc_min <= soc_next <= soc_max):
                                continue
                            fuel_kg = n * pdg[t, ni, b] * sfoc[t, ni, b] * (dt/60.0) / 1000.0
                            total = cost_prev + fuel_kg
                            k = _bucket_soc(soc_next, soc_min, eps)
                            cur = J[t][ni][b].get(k)
                            if (cur is None) or (total < cur[0]):
                                J[t][ni][b][k] = (total, soc_next, (nip, b_prev, k_prev))
            # prune
            for b in range(B):
                if len(J[t][ni][b]) > STATES_CAP:
                    items = sorted(J[t][ni][b].items(), key=lambda kv: kv[1][0])[:STATES_CAP]
                    J[t][ni][b] = dict(items)

    # ---------- terminal condition: enforce SoC_end ≈ SoC_0 ----------
    best = None
    for ni in range(N):
        for b in range(B):
            entry = J[T-1][ni][b].get(s0_key)
            if entry is None:
                continue
            cost = entry[0]
            if (best is None) or (cost < best[0]):
                best = (cost, T-1, ni, b, s0_key)
    if best is None:
        raise RuntimeError("No feasible terminal state at SoC end.")

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