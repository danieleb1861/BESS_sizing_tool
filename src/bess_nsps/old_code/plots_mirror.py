import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# -- Global style to emulate MATLAB LaTeX settings
def _use_latex():
    plt.rcParams.update({
        "text.usetex": False,  # keep False unless LaTeX is guaranteed; MATLAB uses LaTeX by default
        "font.size": 12,
        "axes.grid": True
    })

def _percent_xticks(ax, t_end):
    ax.set_xlim(0, t_end)
    ax.set_xticks([0, 0.25*t_end, 0.5*t_end, 0.75*t_end, t_end])
    ax.set_xticklabels(['0%','25%','50%','75%','100%'])

def _ensure_dir(outdir):
    os.makedirs(outdir, exist_ok=True)

# =======================
# plot_soc.m  -> plot_soc_mirror
# =======================
def plot_soc_mirror(t_plot, soc_opt, soc_min, soc_max, outdir, save_eps=False):
    _use_latex(); _ensure_dir(outdir)
    fig = plt.figure(figsize=(11.2, 3.15))
    ax = fig.gca()
    ax.grid(True, linestyle='--')
    ax.plot(t_plot, 100*np.asarray(soc_opt), linewidth=1.5)
    ax.axhline(100*soc_min, linestyle='--')
    ax.axhline(100*soc_max, linestyle='--')
    _percent_xticks(ax, t_plot[-1])
    ax.set_xlabel('% Mission time')
    ax.set_ylabel('SoC  (%)')
    fig.tight_layout()
    fpng = os.path.join(outdir, 'SoC.png')
    fig.savefig(fpng, dpi=200)
    if save_eps:
        fig.savefig(os.path.join(outdir, 'SoC.eps'), format='eps')
    plt.close(fig)
    return fpng

# =======================
# plot_fuelcons.m -> plot_fuelcons_mirror
# Inputs mirror MATLAB: dgc (coeffs), t_plot, pow_gen_opt, consumo_int_opt, sfoc, status
# We accept a callable sfoc_func(p)->g/kWh and arrays
# =======================
def plot_fuelcons_mirror(t_plot, consumo_int_opt, sfoc_vec, pow_gen_opt, outdir, save_eps=False):
    _use_latex(); _ensure_dir(outdir)
    fig = plt.figure(figsize=(11.2, 6.3))
    gs = fig.add_gridspec(3,1, hspace=0.25)
    ax1 = fig.add_subplot(gs[0,0]); ax2 = fig.add_subplot(gs[1,0]); ax3 = fig.add_subplot(gs[2,0])
    for ax in [ax1, ax2, ax3]: ax.grid(True); ax.box(True)

    # cumulative fuel
    ax1.plot(t_plot, consumo_int_opt, linewidth=1.5)
    _percent_xticks(ax1, t_plot[-1])
    ax1.set_ylabel('Fuel (kg)')

    # sfoc series + min/max guidelines if provided by caller upstream
    ax2.plot(t_plot, sfoc_vec[:,0] if sfoc_vec.ndim>1 else sfoc_vec, linewidth=1.5)
    _percent_xticks(ax2, t_plot[-1])
    ax2.set_ylabel('SFOC (g/kWh)')

    # DG power (first column of pow_gen_opt if 2D)
    if pow_gen_opt.ndim==2:
        pdg = pow_gen_opt[:,0]
    else:
        pdg = pow_gen_opt
    ax3.plot(t_plot, pdg, linewidth=1.5)
    _percent_xticks(ax3, t_plot[-1])
    ax3.set_ylabel('P_{DG} (kW)')
    ax3.set_xlabel('% Mission time')

    fig.tight_layout()
    fpng = os.path.join(outdir, 'cumfuel.png')
    fig.savefig(fpng, dpi=200)
    if save_eps:
        fig.savefig(os.path.join(outdir, 'cumfuel.eps'), format='eps')
    plt.close(fig)
    return fpng

# =======================
# plot_load_sharing.m -> plot_load_sharing_mirror
# =======================
def plot_load_sharing_mirror(t_plot, pow_load, pow_batt_opt, pow_gen_opt, n_active, outdir, save_eps=False):
    _use_latex(); _ensure_dir(outdir)
    fig = plt.figure(figsize=(11.2, 6.3))
    ax1 = fig.add_subplot(2,1,1)
    ax1.grid(True); ax1.box(True)
    ax1.step(t_plot, pow_load, where='post', linewidth=1.5, label='Required power')
    if pow_gen_opt.ndim==2:
        pdg_tot = np.sum(pow_gen_opt, axis=1)
    else:
        pdg_tot = pow_gen_opt
    ax1.plot(t_plot, pdg_tot, linewidth=1.2, label='DG total')
    ax1.plot(t_plot, pow_batt_opt, linewidth=1.0, label='BESS (+discharge)')
    _percent_xticks(ax1, t_plot[-1])
    ax1.set_ylabel('Power (kW)')
    ax1.legend()

    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.grid(True, linestyle='--'); ax2.box(True)
    ax2.step(t_plot, n_active, where='post', linewidth=1.2)
    _percent_xticks(ax2, t_plot[-1])
    ax2.set_xlabel('% Mission time')
    ax2.set_ylabel('Active DGs (#)')
    fig.tight_layout()
    fpng = os.path.join(outdir, 'load_sharing.png')
    fig.savefig(fpng, dpi=200)
    if save_eps:
        fig.savefig(os.path.join(outdir, 'load_sharing.eps'), format='eps')
    plt.close(fig)
    return fpng

# =======================
# plot_configs.m / plot_validation.m -> heatmap-style KPI matrices and ranked bars
# =======================
def _pastel_blue_white_red_cmap(n_colors=64):
    pastel_blue = np.array([7, 87, 152])/255.0
    pastel_red  = np.array([139, 0, 0])/255.0
    white = np.array([1,1,1])
    half = n_colors//2
    btow = np.column_stack([
        np.linspace(pastel_blue[0], white[0], half),
        np.linspace(pastel_blue[1], white[1], half),
        np.linspace(pastel_blue[2], white[2], half),
    ])
    wtored = np.column_stack([
        np.linspace(white[0], pastel_red[0], n_colors-half),
        np.linspace(white[1], pastel_red[1], n_colors-half),
        np.linspace(white[2], pastel_red[2], n_colors-half),
    ])
    colors = np.vstack([btow, wtored])
    return LinearSegmentedColormap.from_list('pbwr', colors, N=n_colors)

def plot_kpi_matrix_mirror(KPI_val, Xlabels, Ylabels, outpath, clim=None, title=None):
    _use_latex(); _ensure_dir(os.path.dirname(outpath))
    KPI_val = np.asarray(KPI_val, dtype=float)
    cmap = _pastel_blue_white_red_cmap(128)
    fig = plt.figure(figsize=(11,6))
    ax = fig.gca()
    im = ax.imshow(KPI_val, aspect='auto', cmap=cmap, origin='upper')
    if clim is not None:
        im.set_clim(*clim)
    ax.set_xticks(np.arange(len(Xlabels)))
    ax.set_xticklabels(Xlabels, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(Ylabels)))
    ax.set_yticklabels(Ylabels)
    if title: ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Score / normalized value')
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    return outpath

# =======================
# Wrapper that mirrors MATLAB entrypoints
# =======================
def mirror_from_results(df, outdir):
    """
    Create heatmaps from results dataframe:
      - fuel_heatmap
      - c_rate_heatmap
    and return paths.
    """
    _ensure_dir(outdir)
    # Example usage similar to earlier 'plot_heatmap'
    return []
