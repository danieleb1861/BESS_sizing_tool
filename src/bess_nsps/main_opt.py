"""
main_opt.py — CLI entrypoint to run DP sweeps and produce KPIs/plots

This script:
1) parses command line args (plant, BESS ranges, time windowing, outputs),
2) loads one or more load profiles (CSV or MAT via data.load_profile),
3) optionally windows the time series (by datetime, minutes, indices, or
   auto-detected manoeuvre windows),
4) sweeps BESS power/energy grids; for each pair runs DP and computes KPIs,
5) optionally generates plots and saves a results CSV.

It orchestrates DP (dp.run_dp), the data utilities (data.py),
plotting helpers (plots.py), and KPI computation (kpi.py).
"""

import argparse
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from pathlib import Path
from itertools import product
from datetime import datetime

# Profile I/O utilities
from .data import choose_profiles, load_profile
try:
    from .data import get_mat_window
except Exception:
    get_mat_window = None
try:
    from .data import csv_dt_bounds_to_minutes
except Exception:
    csv_dt_bounds_to_minutes = None

# Plant/BESS specs and DP interface
from .models import DGSpec, BESSpec
from .dp import DPConfig, run_dp
from .kpi import compute_kpis
from . import plots

# Windowing helpers
from .window import slice_by_indices, slice_by_time, find_manoeuvre_window


def build_dg(pmax_kw: float):
    # Create a DGSpec with a simple SFOC curve sampled on % load.
    sfoc = np.array([300, 250, 230, 215, 211, 208, 210, 213], dtype=float)  # g/kWh
    pgrid = np.array([0, 20, 40, 60, 75, 85, 95, 100], dtype=float) / 100.0 * pmax_kw
    return DGSpec(
        pmax_kw=pmax_kw,
        pmin_frac=0.20,
        pmax_frac=0.95,
        sfoc_g_per_kwh=sfoc,
        p_grid_kw=pgrid,
    )


def nondominated(F: np.ndarray) -> np.ndarray:
    # Return a boolean mask of Pareto‑nondominated rows of F.
    # Expects F with columns as objectives to *minimize*. A point i is dominated if some j has F[j] ≤ F[i] in all objectives and < in at least one.

    N = F.shape[0]
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        if not mask[i]:
            continue
        dominates = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominates[i] = False
        if np.any(dominates):
            mask[i] = False
    return mask


def _tod_to_min(s: str) -> float:
    # Convert 'HH:MM[:SS]' (or 'YYYY-MM-DD HH:MM[:SS]') to minutes from 00:00. Generic fallback when we don't have CSV base date.

    s = s.strip().replace("T", " ")
    parts = s.split()
    if len(parts) == 2 and "-" in parts[0]:
        s = parts[1]
    hh, mm, *rest = s.split(":")
    ss = float(rest[0]) if rest else 0.0
    return int(hh) * 60 + int(mm) + ss / 60.0


def _safe_load_profile(path: str):
    # Call data.load_profile() with (optionally) step_min/max_samples if supported. Returns (pow_load_kw, t_min).

    try:
        return load_profile(path, step_min=1.0, max_samples=100_000)
    except TypeError:
        # Older or simpler signature: load_profile(path) -> (pow_load, t_min)
        return load_profile(path)


def save_trace_h5(h5path, group, t_min, pow_load_kw, res, dg, bes, cfg, meta):
    Path(os.path.dirname(h5path)).mkdir(parents=True, exist_ok=True)
    with h5py.File(h5path, "a") as h5:
        g = h5.require_group(group)
        for name, arr in {
            "t_min": t_min,
            "pow_load_kw": pow_load_kw,
            "soc": res.soc_traj,
            "n_active": np.asarray(res.n_active_traj, int),
            "p_bess_kw": res.p_bess_traj_kw,
            "p_dg_kw": res.p_dg_traj_kw,
        }.items():
            if name in g:
                del g[name]
            g.create_dataset(name, data=arr, compression="lzf")  # faster than gzip
        g.attrs.update({
            "fuel_kg": float(res.cost_kg),
            "bess_pmax_kw": float(bes.pmax_kw),
            "bess_e_kwh": float(bes.e_kwh),
            "soc_min": float(bes.soc_min),
            "soc_max": float(bes.soc_max),
            "eta_c": float(bes.eta_c),
            "eta_d": float(bes.eta_d),
            "dg_pmax_kw": float(dg.pmax_kw),
            "ndg": int(cfg.ndg_installed),
            **meta
        })


def cfg_id(profile, pmax, e):
    base = os.path.splitext(os.path.basename(profile))[0]
    return f"{base}/p{int(round(pmax))}_e{int(round(e))}"


def main():
    # Command Line Interface (CLI) entrypoint: parse args, load/window data, sweep BESS, run DP, save results.

    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", required=True)

    # Plant sizing
    ap.add_argument("--dg-pmax", type=float, default=2200.0)
    ap.add_argument("--ndg", type=int, default=6)

    # Design sweep (BESS power and energy grids)
    ap.add_argument("--bess-pmin", type=float, default=660.0)
    ap.add_argument("--bess-pmax", type=float, default=1540.0)
    ap.add_argument("--bess-pesteps", type=int, default=16)
    ap.add_argument("--bess-emin", type=float, default=660.0)
    ap.add_argument("--bess-emax", type=float, default=1540.0)
    ap.add_argument("--bess-esteps", type=int, default=16)

    # BESS model
    ap.add_argument("--soc-min", type=float, default=0.20)
    ap.add_argument("--soc-max", type=float, default=0.80)
    ap.add_argument("--eta-c", type=float, default=0.97)
    ap.add_argument("--eta-d", type=float, default=0.94)
    ap.add_argument("--alpha-min", type=float, default=-1.0)
    ap.add_argument("--alpha-max", type=float, default=1.0)

    # Outputs
    ap.add_argument("--save", type=str, default="outputs/results.csv")
    ap.add_argument("--plots", action="store_true")

    # Windowing (minute-native; also supports datetime-like convenience)
    ap.add_argument("--dt-start", type=str, default=None,
                    help='Datetime start (e.g. "2024-05-01 15:04:20" or "15:04:20")')
    ap.add_argument("--dt-end", type=str, default=None,
                    help='Datetime end (e.g. "2024-05-01 15:05:10" or "15:05:10")')

    ap.add_argument("--t-start", type=float, default=None, help="Start time [minutes from first sample]")
    ap.add_argument("--t-end",   type=float, default=None, help="End time [minutes from first sample]")
    ap.add_argument("--i-start", type=int,   default=None, help="Start index (0-based)")
    ap.add_argument("--i-end",   type=int,   default=None, help="End index (exclusive)")
    ap.add_argument("--use-mat-window", action="store_true",
                    help="Read i0/i1 or t0/t1 (minutes) from the .mat file, if present")
    ap.add_argument("--window-min", type=float, default=0.0,
                    help="Auto-select window length in minutes; 0 = full profile")
    ap.add_argument("--window-method", type=str, default="ramp", choices=["ramp", "std"],
                    help="Auto window scoring: ramp=sum|dp/dt|, std=rolling std")

    args = ap.parse_args()
    
    # --- Prepare plot directories ---
    outdir_plots = os.path.join(os.path.dirname(args.save), "plots")
    os.makedirs(outdir_plots, exist_ok=True)

    from datetime import datetime
    RUN_ID = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Per-run folder so multiple runs don’t overwrite
    outdir_plots_run = os.path.join(outdir_plots, RUN_ID)
    os.makedirs(outdir_plots_run, exist_ok=True)

   
    # ---- Design grids ----

    p_grid = np.linspace(args.bess_pmin, args.bess_pmax, args.bess_pesteps)
    e_grid = np.linspace(args.bess_emin, args.bess_emax, args.bess_esteps)
    soc_grid = np.linspace(args.soc_min, args.soc_max, 31)

    # Plant specs and DP config
    dg = build_dg(args.dg_pmax)
    cfg = DPConfig(
        ndg_installed=args.ndg,
        soc_grid=soc_grid,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
    )
    
    # Prepare output dir
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    rows = []
    res_cache = {}

    # Resolve profile paths
    profile_paths = choose_profiles(args.profiles)

    for prof in profile_paths:
        # 1) Load profile
        pow_load_kw, t_min = _safe_load_profile(prof)

        # 2) Apply windowing in priority order
        if args.dt_start and args.dt_end:
            # If CSV, convert dt strings to minutes-from-first-sample using the file's first timestamp
            if os.path.splitext(prof)[1].lower() == ".csv" and csv_dt_bounds_to_minutes is not None:
                try:
                    t0m, t1m = csv_dt_bounds_to_minutes(prof, args.dt_start, args.dt_end)
                except Exception as e:
                    print(f"[warn] csv dt bounds failed: {e}; falling back to time-of-day minutes.")
                    t0m = _tod_to_min(args.dt_start)
                    t1m = _tod_to_min(args.dt_end)
            else:
                # Generic fallback: interpret as minutes-from-midnight
                t0m = _tod_to_min(args.dt_start)
                t1m = _tod_to_min(args.dt_end)

            pow_load_kw, t_min = slice_by_time(pow_load_kw, t_min, t0m, t1m)

        elif args.i_start is not None and args.i_end is not None:
            pow_load_kw, t_min = slice_by_indices(pow_load_kw, t_min, args.i_start, args.i_end)

        elif args.t_start is not None and args.t_end is not None:
            pow_load_kw, t_min = slice_by_time(pow_load_kw, t_min, args.t_start, args.t_end)

        elif args.use_mat_window and get_mat_window is not None:
            # Read i0/i1 or t0/t1 from the .mat stored alongside the profile
            meta = get_mat_window(os.path.join("data", os.path.basename(prof)))
            if meta:
                if "i0" in meta and "i1" in meta:
                    pow_load_kw, t_min = slice_by_indices(pow_load_kw, t_min, meta["i0"], meta["i1"])
                elif "t0" in meta and "t1" in meta:
                    pow_load_kw, t_min = slice_by_time(pow_load_kw, t_min, meta["t0"], meta["t1"])

        elif args.window_min and args.window_min > 0:
            # Auto‑select most "interesting" window (maneuver) by ramp/STD score
            pow_load_kw, t_min = find_manoeuvre_window(
                pow_load_kw, t_min, window_min=args.window_min, method=args.window_method
            )

        # 3) Safety: non-empty & monotonic time
        if len(t_min) == 0:
            print(f"[warn] windowing produced zero samples on {os.path.basename(prof)}; using full series.")
            pow_load_kw, t_min = _safe_load_profile(prof)

        t_min = np.asarray(t_min, float).ravel()
        pow_load_kw = np.asarray(pow_load_kw, float).ravel()
        keep = np.concatenate(([True], np.diff(t_min) > 0))
        t_min, pow_load_kw = t_min[keep], pow_load_kw[keep]
        if t_min.size < 2:
            continue

        # 4) Outer sweep + DP
        pairs = product(p_grid, e_grid)
        for p_bess, e_bess in tqdm(list(pairs), total=len(p_grid) * len(e_grid), desc=f"BESS grid ({os.path.basename(prof)})"):
            bes = BESSpec(pmax_kw=p_bess, e_kwh=e_bess, soc_min=args.soc_min, soc_max=args.soc_max, eta_c=args.eta_c, eta_d=args.eta_d)

            stem = f"p{int(round(p_bess))}_e{int(round(e_bess))}_"
            res = run_dp(pow_load_kw, t_min, dg, bes, cfg)

            k = compute_kpis(
                p_bess_kw=res.p_bess_traj_kw,
                dt_min=np.diff(t_min, prepend=t_min[0]),
                soc=res.soc_traj,
                bes=bes,
                volume_method="geometry",
                batt_pmax_kw=bes.pmax_kw,
            )
            k.fuel_kg = res.cost_kg

            rows.append({
                "profile": prof,
                "bess_pmax_kw": p_bess,
                "bess_e_kwh": e_bess,
                "fuel_kg": k.fuel_kg,
                "c_rate_mean_per_h": k.c_rate_mean_per_h,
                "dod_mean": k.dod_mean,
                "efc_per_year": k.efc_per_year,
                "energy_throughput_kwh": k.energy_throughput_kwh,
                "t_backup_min": k.t_backup_min,
                "volume_proxy_m3": k.volume_proxy_m3,
            })

            # cache result for later Pareto plotting
            res_cache[(prof, float(p_bess), float(e_bess))] = (t_min, pow_load_kw, bes, res)

    # 5) Post processing: Pareto, plots, save CSV & (optionally) traces
    df = pd.DataFrame(rows)
    cols = ["fuel_kg", "c_rate_mean_per_h", "dod_mean",
            "efc_per_year", "energy_throughput_kwh", "volume_proxy_m3"]
    df["pareto"] = nondominated(df[cols].to_numpy())

    # Use cache: plot/save only Pareto points
    pareto_df = df[df["pareto"]]
    for _, r in pareto_df.iterrows():
        key = (r["profile"], float(r["bess_pmax_kw"]), float(r["bess_e_kwh"]))
        t_min, pow_load_kw, bes, res = res_cache[key]

        prof_base = os.path.splitext(os.path.basename(r["profile"]))[0]
        cfg_dir = os.path.join(
            outdir_plots,
            prof_base,
            f"p{int(round(r['bess_pmax_kw']))}_e{int(round(r['bess_e_kwh']))}"
        )
        os.makedirs(cfg_dir, exist_ok=True)

        if args.plots and prof == profile_paths[0] and len(rows) <= 2:
            plots.plot_fuelcons(t_min,
                res.p_dg_traj_kw,
                np.array(res.n_active_traj, dtype=float),
                lambda x: np.interp(x, dg.p_grid_kw, dg.sfoc_g_per_kwh),
                outdir_plots_run,
                stem=stem,
            )
            plots.plot_soc(t_min, res.soc_traj, bes.soc_min, bes.soc_max, outdir_plots_run, stem=stem)
            plots.plot_load_sharing(
                t_min, pow_load_kw, res.p_bess_traj_kw, res.p_dg_traj_kw, res.n_active_traj,
                outdir_plots_run,
                stem=stem,
            )

        if args.save_traces:
            save_trace_h5(
                out_base,
                cfg_id(r["profile"], r["bess_pmax_kw"], r["bess_e_kwh"]),
                t_min, pow_load_kw, res, dg, bes, cfg,
                meta={"source": "pareto", "run_id": RUN_ID}
            )

    if args.plots:
        plots.plot_heatmap(
            df, "bess_pmax_kw", "bess_e_kwh", "fuel_kg",
            os.path.join(outdir_plots, "fuel_heatmap.png"),
        )

    df.to_csv(args.save, index=False)
    print(f"Saved results to: {args.save}")
    if args.save_traces:
        print(f"Saved Pareto traces to {out_base}")


if __name__ == "__main__":
    main()