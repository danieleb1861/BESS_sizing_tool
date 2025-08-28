import os
import numpy as np
import matplotlib.pyplot as plt

def _percent_axis_data(t_min):
    """Return time relative to start (same units as t_min) and its span."""
    t = np.asarray(t_min, dtype=float)
    t_rel = t - t[0]
    return t_rel, t_rel[-1]

def _percent_xticks(ax, t_end):
    ax.set_xlim(0, t_end)
    ax.set_xticks([0, 0.25*t_end, 0.5*t_end, 0.75*t_end, t_end])
    ax.set_xticklabels(['0%','25%','50%','75%','100%'])

def plot_soc(t_min, soc, soc_min, soc_max, outdir, stem: str = ""):
    """Save SoC plot. If stem is provided, filenames are prefixed with it."""
    os.makedirs(outdir, exist_ok=True)
    t_rel, t_end = _percent_axis_data(t_min)

    fig = plt.figure(figsize=(12,3.2))
    ax = fig.gca()
    ax.grid(True, linestyle='--')
    ax.plot(t_rel, 100*np.asarray(soc), linewidth=1.6)
    ax.axhline(100*soc_min, linestyle='--')
    ax.axhline(100*soc_max, linestyle='--')
    _percent_xticks(ax, t_end)
    ax.set_xlabel('% Mission time')
    ax.set_ylabel('SoC (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}soc.png"), dpi=200)
    plt.close(fig)

def plot_load_sharing(t_min, p_load, p_bess, p_dg_per, n_active, outdir, stem: str = ""):
    """Save load sharing plot. If stem is provided, filenames are prefixed with it."""
    os.makedirs(outdir, exist_ok=True)
    t_rel, t_end = _percent_axis_data(t_min)

    total_dg = p_dg_per * n_active
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(2,1,1)
    ax1.grid(True)
    ax1.step(t_rel, p_load, where='post', linewidth=1.5, label='Load')
    ax1.plot(t_rel, total_dg, linewidth=1.2, label='DG total')
    ax1.plot(t_rel, p_bess, linewidth=1.0, label='BESS (+discharge)')
    _percent_xticks(ax1, t_end)
    ax1.set_ylabel('Power (kW)')
    ax1.legend()

    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.grid(True, linestyle='--')
    ax2.step(t_rel, n_active, where='post', linewidth=1.2)
    _percent_xticks(ax2, t_end)
    ax2.set_xlabel('% Mission time')
    ax2.set_ylabel('Active DGs (#)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}load_sharing.png"), dpi=200)
    plt.close(fig)

def plot_fuelcons(t_min, p_dg_per, n_active, sfoc_func, outdir, stem: str = ""):
    """Save fuel consumption plots. If stem is provided, filenames are prefixed with it."""
    os.makedirs(outdir, exist_ok=True)
    t_rel, t_end = _percent_axis_data(t_min)

    # timestep in hours from RELATIVE time
    dt_h = np.diff(t_rel, prepend=0.0) / 60.0

    p_dg = np.maximum(p_dg_per, 0.0)
    sfoc = sfoc_func(p_dg)                   # g/kWh
    m_dot = (p_dg * sfoc / 1000.0) * n_active  # kg/h
    m_inc = m_dot * dt_h
    m_cum = np.cumsum(m_inc)

    fig = plt.figure(figsize=(12,6.2))
    ax1 = fig.add_subplot(3,1,1)
    ax1.grid(True)
    ax1.plot(t_rel, m_cum, linewidth=1.5)
    _percent_xticks(ax1, t_end)
    ax1.set_ylabel('Fuel (kg)')

    ax2 = fig.add_subplot(3,1,2, sharex=ax1)
    ax2.grid(True)
    ax2.plot(t_rel, sfoc, linewidth=1.2)
    _percent_xticks(ax2, t_end)
    ax2.set_ylabel('SFOC (g/kWh)')

    ax3 = fig.add_subplot(3,1,3, sharex=ax1)
    ax3.grid(True)
    ax3.plot(t_rel, p_dg * n_active, linewidth=1.2)
    _percent_xticks(ax3, t_end)
    ax3.set_ylabel('DG total (kW)')
    ax3.set_xlabel('% Mission time')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}fuel_consumption.png"), dpi=200)
    plt.close(fig)