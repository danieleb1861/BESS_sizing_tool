# plot_results.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Pareto helpers ----------
def _nondominated(F: np.ndarray) -> np.ndarray:
    """Mask of nondominated (Pareto) points for minimization."""
    n = F.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        dom = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Turn strings/0/1 into real booleans robustly."""
    if s.dtype == bool:
        return s
    if np.issubdtype(s.dtype, np.number):
        return s.astype(int).astype(bool)
    # assume strings
    low = s.astype(str).str.strip().str.lower()
    mapped = low.map({"true": True, "false": False, "1": True, "0": False})
    # fallback: any non-empty -> True
    return mapped.fillna(low.ne("")).astype(bool)


def _ensure_pareto(df: pd.DataFrame) -> pd.DataFrame:
    if "pareto" in df.columns:
        out = df.copy()
        out["pareto"] = _coerce_bool_series(out["pareto"])
        return out
    # recompute using the standard KPI set if available
    cand = [
        "fuel_kg",
        "c_rate_mean_per_h",
        "dod_mean",
        "efc_per_year",
        "energy_throughput_kwh",
        "volume_proxy_m3",
    ]
    cols = [c for c in cand if c in df.columns]
    out = df.copy()
    out["pareto"] = _nondominated(out[cols].to_numpy(float)) if cols else True
    return out


# ---------- Labels ----------
LABELS = {
    "fuel_kg": "Fuel [kg]",
    "fuel_pct": "Fuel [% of worst]",
    "bess_pmax_kw": r"$P_{BESS, \max}$ [kW]",
    "bess_e_kwh": r"$E_{BESS, \max}$ [kWh]",
    "volume_proxy_m3": r"Volume [m³]",
    "c_rate_mean_per_h": r"C-rate$_{\mathrm{mean}}$ [h$^{-1}$]",
    "dod_mean": r"DoD$_{\mathrm{mean}}$ [–]",
    "efc_per_year": r"EFC per year [cycles/yr]",
    "energy_throughput_kwh": r"Energy throughput [kWh]",
}

# ---------- Normalization helpers ----------
def _add_normalized_columns(df: pd.DataFrame, dg_rated: float) -> pd.DataFrame:
    """Add normalized fuel (%) and nondimensional power/energy (p.u.)."""
    out = df.copy()

    # Non-dimensional P and E
    if "bess_pmax_kw" in out.columns:
        out["bess_pmax_pu"] = out["bess_pmax_kw"].astype(float) / float(dg_rated)
    if "bess_e_kwh" in out.columns:
        out["bess_e_over_pdg_h"] = out["bess_e_kwh"].astype(float) / float(dg_rated)

    # Normalize fuel: worst = 100%, improvements < 100%
    if "fuel_kg" in out.columns and "profile" in out.columns:
        out["fuel_kg"] = out["fuel_kg"].astype(float)
        fmax = out.groupby("profile")["fuel_kg"].transform("max")
        out["fuel_pct"] = 100.0 * out["fuel_kg"] / fmax

    return out


# Which direction is "better" for the Y metric in the (fuel, Y) plane.
# We'll always minimize fuel; for Y we either maximize or minimize as listed below.
Y_MAXIMIZE = {
    "bess_pmax_kw",        # higher power is better
    "bess_e_kwh",          # higher energy is better
    "efc_per_year",        # more cycles/yr is better
}
Y_MINIMIZE = {
    "volume_proxy_m3",     # smaller volume is better
    "c_rate_mean_per_h",   # lower stress is better
    "dod_mean",            # lower average DoD is better
    "energy_throughput_kwh"  # (if used)
}

DEFAULT_VARS = [
    "bess_pmax_kw",
    "bess_e_kwh",
    "volume_proxy_m3",
    "c_rate_mean_per_h",
    "dod_mean",
    "efc_per_year",
]  # 6 subplots


# ---------- Heatmap ----------
def plot_pe_fuel_heatmap(
    df: pd.DataFrame,
    outdir: str,
    p_col="bess_pmax_kw",
    e_col="bess_e_kwh",
    fuel_col="fuel_pct",
    pareto_only=False,
    mark_min=True,
):
    os.makedirs(outdir, exist_ok=True)
    for prof, g in df.groupby("profile"):
        g = g.copy()
        if pareto_only and "pareto" in g.columns:
            g = g[_coerce_bool_series(g["pareto"])]

        if g.empty or any(c not in g.columns for c in (p_col, e_col, fuel_col)):
            continue

        piv = g.pivot_table(index=e_col, columns=p_col, values=fuel_col, aggfunc="min")
        piv = piv.sort_index().sort_index(axis=1)

        plt.figure(figsize=(6, 5))
        mesh = plt.pcolormesh(
            piv.columns.to_numpy(float),
            piv.index.to_numpy(float),
            piv.to_numpy(float),
            shading="auto",
        )
        plt.colorbar(mesh, label=LABELS.get(fuel_col, fuel_col))
        plt.xlabel(LABELS.get(p_col, p_col))
        plt.ylabel(LABELS.get(e_col, e_col))
        plt.title(
            f"{os.path.basename(str(prof))}"
            + (" (Pareto)" if pareto_only else "")
        )
        plt.grid(False)
        plt.tight_layout()

        if mark_min:
            try:
                arr = piv.to_numpy(float)
                if np.isfinite(arr).any():
                    i_min = np.nanargmin(arr)
                    i, j = np.unravel_index(i_min, arr.shape)
                    p_star = float(piv.columns[j])
                    e_star = float(piv.index[i])
                    f_star = arr[i, j]
                    plt.scatter([p_star], [e_star], s=70, marker="x", color="red", linewidths=1.6)
                    plt.text(p_star, e_star, f"  min={f_star:.2f} kg", va="center", ha="left", fontsize=9)
            except Exception:
                pass

        out = os.path.join(
            outdir,
            f"heatmap_fuel_{os.path.basename(str(prof)).replace('.','_')}{'_pareto' if pareto_only else ''}.png",
        )
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved:", out)


# ---------- Pareto grid (2×3) ----------
def plot_pareto_grid(
    df: pd.DataFrame,
    outdir: str,
    variables,
    figwidth=12.0,
    figheight=7.5,
    jitter=0.0,
):
    os.makedirs(outdir, exist_ok=True)

    for prof, g in df.groupby("profile"):
        g = g.copy()
        g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=["fuel_kg"])

        # Global min fuel for this profile
        i_min = g["fuel_pct"].idxmin()
        fuel_min = g.loc[i_min, "fuel_pct"]

        fig, axes = plt.subplots(2, 3, figsize=(figwidth, figheight))
        axes = axes.ravel()

        for ax, var in zip(axes, variables):
            # keep finite points for this plane
            gg = g.dropna(subset=["fuel_pct", var]).copy()
            if gg.empty:
                ax.set_visible(False)
                continue

            # per-plane nondominance: minimize fuel; for Y either maximize or minimize
            y = gg[var].to_numpy(float)
            fuel = gg["fuel_pct"].to_numpy(float)

            # convert to an all-minimization array for _nondominated
            if var in Y_MAXIMIZE:
                y_for_min = -y          # maximizing Y == minimizing (-Y)
            else:
                # default to minimizing (covers Y_MINIMIZE and any unknown -> conservative)
                y_for_min = y

            arr = np.column_stack([fuel, y_for_min])
            pareto_mask = _nondominated(arr)

            dom = gg[~pareto_mask]
            par = gg[pareto_mask]

            # Optional x-jitter on fuel for overlap visibility
            if jitter:
                rng = np.random.default_rng(0)
                if len(dom):
                    dom = dom.copy()
                    dom["fuel_pct"] = dom["fuel_pct"].to_numpy(float) + rng.normal(0, jitter, len(dom))
                if len(par):
                    par = par.copy()
                    par["fuel_pct"] = par["fuel_pct"].to_numpy(float) + rng.normal(0, jitter, len(par))

            # 1) dominated (blue)
            if len(dom):
                ax.scatter(dom["fuel_pct"], dom[var], s=18, alpha=0.9, color="blue", label="Dominated")

            # 2) Pareto front for this plane (red)
            if len(par):
                ax.scatter(par["fuel_pct"], par[var], s=36, alpha=0.95, color="red",
                           edgecolor="none", label="Pareto front")

            # 3) min fuel (red ×) — only if var exists for that row
            if pd.notna(g.loc[i_min, var]):
                ax.scatter([fuel_min], [g.loc[i_min, var]], s=80, marker="x",
                           color="red", linewidths=1.6, label="Min fuel")

            ax.set_xlabel(LABELS.get("fuel_pct", "Fuel [kg]"))
            ax.set_ylabel(LABELS.get(var, var))
            ax.grid(True, linestyle=":", linewidth=0.8)

            # tidy legend (unique labels)
            h, l = ax.get_legend_handles_labels()
            uniq = dict(zip(l, h))
            if uniq:
                ax.legend(uniq.values(), uniq.keys(), loc="best", frameon=False)

        # ---- these belong OUTSIDE the per-axis loop ----
        fig.suptitle(f"{os.path.basename(str(prof))}", y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out = os.path.join(outdir, f"pareto_grid_{os.path.basename(str(prof)).replace('.','_')}.png")
        plt.savefig(out, dpi=200)
        plt.close(fig)
        print("Saved:", out)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/results.csv")
    ap.add_argument("--outdir", default="outputs/plots")
    ap.add_argument("--vars", nargs="*", default=None,
                    help="Exactly 6 variable names for the 2x3 grid. "
                         "Defaults to a standard set.")
    ap.add_argument("--figwidth", type=float, default=12.0)
    ap.add_argument("--figheight", type=float, default=7.5)
    ap.add_argument("--jitter", type=float, default=0.0,
                    help="Small jitter on fuel x-values to reveal overlaps (e.g. 0.02).")

    # Heatmap options (enabled by default; disable with --no-heatmap)
    ap.add_argument("--no-heatmap", action="store_true", help="Disable P-E-Fuel heatmaps")
    ap.add_argument("--heatmap-p-col", default="bess_pmax_kw")
    ap.add_argument("--heatmap-e-col", default="bess_e_kwh")
    ap.add_argument("--heatmap-fuel-col", default="fuel_pct")
    ap.add_argument("--heatmap-pareto-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = _ensure_pareto(df)

    # --- Normalize fuel (%) and adimensionalize power/energy ---
    dg_rated = 2200.0  # adjust if different DG rating
    df = _add_normalized_columns(df, dg_rated)


    variables = args.vars if args.vars else DEFAULT_VARS
    if len(variables) != 6:
        raise SystemExit("Please provide exactly 6 variables via --vars, or use the default 6.")

    # 2×3 Pareto grid (fuel on x)
    plot_pareto_grid(
        df, outdir=args.outdir, variables=variables,
        figwidth=args.figwidth, figheight=args.figheight, jitter=args.jitter
    )

    # Heatmap per profile (on by default)
    heatmap_enabled = not args.no_heatmap  # argparse replaces '-' with '_' in attribute name
    if heatmap_enabled:
        plot_pe_fuel_heatmap(
            df,
            outdir=args.outdir,
            p_col=args.heatmap_p_col,
            e_col=args.heatmap_e_col,
            fuel_col=args.heatmap_fuel_col,
            pareto_only=args.heatmap_pareto_only,
            mark_min=True,
        )



if __name__ == "__main__":
    main()