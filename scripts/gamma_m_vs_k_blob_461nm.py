"""
Compares the left-blob ROI (-0.75 to -0.5 A^-1, 0.49-0.61 eV) between the
Gamma-M and Gamma-K directions for 461 nm pump, probe S.
Tests whether the feature at that (k, E) position is direction-specific.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc

dir = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *

out_dir = os.path.join(dir, "..", "plots")
data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

BLOB_ENERGY  = (0.49, 0.61)
BLOB_L_MOM   = (-0.75, -0.5)
REF_ENERGY   = (0.0,   0.2)
REF_MOMENTUM = (-0.1,  0.1)
ENERGY_BIAS  = 31.5

TIME_AXIS_MIN, TIME_AXIS_MAX = -150, 2000

ARPES_SNAPSHOT_RANGE = (220, 280)
ARPES_ENERGY_MIN, ARPES_ENERGY_MAX = 0.2, 0.68
ARPES_MOMENTUM_CUTOFF = 1.0

_di_461_gm_s = [
    DataItem(data_root_dir, ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "31.95eV_0"], tag="31.95eV"),
    DataItem(data_root_dir, ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "32.57_sum"],  tag="32.57eV"),
    DataItem(data_root_dir, ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "33.19_sum"],  tag="33.19eV"),
    DataItem(data_root_dir, ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "33.81eV_0"], tag="33.81eV"),
]

_di_461_gk_s = [
    DataItem(data_root_dir, ["Dec", "G_Gamma_K", "31.95_summed", "ProbeRotator_37"], tag="31.95eV"),
    DataItem(data_root_dir, ["Dec", "G_Gamma_K", "32.6_summed",  "ProbeRotator_37"], tag="32.6eV"),
    DataItem(data_root_dir, ["Dec", "G_Gamma_K", "33.25_summed", "ProbeRotator_37"], tag="33.25eV"),
    DataItem(data_root_dir, ["Dec", "G_Gamma_K", "33.9_summed",  "ProbeRotator_37"], tag="33.9eV"),
]

DATASETS = [
    {"title": r"461 nm $\Gamma$-M", "label": "Γ-M", "color": "tab:blue",
     "data_items": _di_461_gm_s, "stitch_fn": concat_energy_datas},
    {"title": r"461 nm $\Gamma$-K", "label": "Γ-K", "color": "tab:orange",
     "data_items": _di_461_gk_s, "stitch_fn": concat_energy_datas},
]


def gauss_exp_model(t, B, A, t0, sigma, tau):
    arg = (sigma / tau - (t - t0) / sigma) / np.sqrt(2)
    return B + A * 0.5 * np.exp(0.5 * (sigma / tau) ** 2 - (t - t0) / tau) * erfc(arg)


def load_and_prepare(config):
    plot_data = []
    for di in config["data_items"]:
        print(f"  Loading {di.search_keywords}")
        try:
            datas = read_sweep_data(di.base_directory, di.search_keywords)
            if len(datas) <= 1:
                raise Exception()
        except Exception:
            datas = read_all_sweep_data(di.base_directory, di.search_keywords)
        plot_data.append(datas)
    for prep in [config["stitch_fn"], bin_sweep_datas, subtract_noise,
                 remove_bias, remove_e_fermi, pad_energy, sum_data_prep]:
        plot_data = prep(plot_data)
    return plot_data[0][0]


def extract_roi_ts(data, energy_range, momentum_range):
    e_mid = (energy_range[0] + energy_range[1]) / 2
    th_lo = momentum_to_angle(momentum_range[0], ENERGY_BIAS + e_mid)
    th_hi = momentum_to_angle(momentum_range[1], ENERGY_BIAS + e_mid)
    fig_tmp, ax_tmp = plt.subplots()
    display_sweep_data(data, axes_choice=[Axes.TIME],
                       summing_ranges={Axes.THETA_X: [[th_lo, th_hi]],
                                       Axes.ENERGY:  [list(energy_range)]},
                       plot_axis=ax_tmp, normalize=True, display_plot_title=False)
    t_arr = ax_tmp.lines[-1].get_xdata().copy().astype(float)
    y_arr = ax_tmp.lines[-1].get_ydata().copy().astype(float)
    plt.close(fig_tmp)
    return t_arr, y_arr


def gauss_exp_fit(t_data, y_data):
    mask = np.isfinite(t_data) & np.isfinite(y_data)
    t_fit, y_fit = t_data[mask], y_data[mask]
    try:
        popt, _ = curve_fit(gauss_exp_model, t_fit, y_fit,
                            p0=[0.05, 0.9, 0.0, 50.0, 400.0],
                            bounds=([0, 0, -300, 10, 50], [0.5, 5, 300, 300, 5000]),
                            maxfev=10000)
        return popt
    except RuntimeError as e:
        print(f"  Fit failed: {e}")
        return None


# Load
loaded = []
for config in DATASETS:
    print(f"Loading {config['title']}...")
    loaded.append(load_and_prepare(config))

# Extract time series
print("Extracting ROI time series...")
ts_blob, ts_ref = [], []
for data in loaded:
    tB, yB = extract_roi_ts(data, BLOB_ENERGY, BLOB_L_MOM)
    _,  yR = extract_roi_ts(data, REF_ENERGY,  REF_MOMENTUM)
    ts_blob.append((tB, yB))
    ts_ref.append(yR)

# Figure: 3 panels — ARPES Gamma-M, ARPES Gamma-K, time series comparison
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.8],
              left=0.06, right=0.97, top=0.88, bottom=0.14, wspace=0.28)

ax_arpes = [fig.add_subplot(gs[0, i]) for i in range(2)]
ax_ts    = fig.add_subplot(gs[0, 2])

fig.suptitle(r"461 nm: Left Blob ROI ($-0.75$ to $-0.5\ \AA^{-1}$, $0.49$–$0.61$ eV) — $\Gamma$-M vs $\Gamma$-K (probe S)",
             fontsize=12)

for idx, config in enumerate(DATASETS):
    data  = loaded[idx]
    tB, yB = ts_blob[idx]
    yR    = ts_ref[idx]
    color = config["color"]
    label = config["label"]

    # ARPES snapshot
    ax_a = ax_arpes[idx]
    has_snap = any(ARPES_SNAPSHOT_RANGE[0] <= t <= ARPES_SNAPSHOT_RANGE[1]
                   for t in sorted(data.sweep_raw.keys()))
    if has_snap:
        display_sweep_data(data, axes_choice=[Axes.THETA_X, Axes.ENERGY],
                           summing_ranges={Axes.TIME: [list(ARPES_SNAPSHOT_RANGE)]},
                           plot_axis=ax_a, color_map="terrain",
                           percentile_for_colorscale=99, angle_to_momentum=True,
                           momentum_cutoff=ARPES_MOMENTUM_CUTOFF,
                           display_plot_title=False, display_x_label=True, display_y_label=False)
    else:
        ax_a.text(0.5, 0.5, "N/A (+250 fs)", transform=ax_a.transAxes,
                  ha="center", va="center", fontsize=10, color="gray")
    ax_a.set_ylim(ARPES_ENERGY_MIN, ARPES_ENERGY_MAX)
    ax_a.set_xlim(-ARPES_MOMENTUM_CUTOFF, ARPES_MOMENTUM_CUTOFF)
    ax_a.set_ylabel(r"$E - E_F$ [eV]", fontsize=9)
    ax_a.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=9)
    ax_a.set_title(f"{config['title']}  (t = +250 fs)", fontsize=10)

    # ROI box
    ax_a.add_patch(Rectangle((BLOB_L_MOM[0], BLOB_ENERGY[0]),
                              BLOB_L_MOM[1] - BLOB_L_MOM[0],
                              BLOB_ENERGY[1] - BLOB_ENERGY[0],
                              fill=False, edgecolor=color, linewidth=1.5, linestyle="--"))
    ax_a.text(BLOB_L_MOM[1] + 0.03, (BLOB_ENERGY[0] + BLOB_ENERGY[1]) / 2,
              "L blob", color=color, fontsize=7, va="center")

    # Time series
    ax_ts.plot(tB, yB, "o-", color=color, markersize=3, label=f"{label} blob")
    ax_ts.plot(tB, yR, "v--", color=color, markersize=3, alpha=0.4, label=f"{label} ref")

    popt = gauss_exp_fit(tB, yB)
    if popt is not None:
        B, A, t0, sigma, tau = popt
        t_s = np.linspace(float(tB[np.isfinite(tB)].min()),
                          float(tB[np.isfinite(tB)].max()), 500)
        ax_ts.plot(t_s, gauss_exp_model(t_s, *popt), "-", color=color,
                   linewidth=2, alpha=0.7, label=f"fit {label}: τ={tau:.0f} fs")
        print(f"  {label} fit: τ={tau:.0f} fs, t₀={t0:.0f} fs, σ={sigma:.0f} fs")

ax_ts.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax_ts.set_xlim(TIME_AXIS_MIN, TIME_AXIS_MAX)
ax_ts.set_title("Left blob time series (onset at t = 0)", fontsize=10)
ax_ts.set_xlabel("Time delay [fs]", fontsize=9)
ax_ts.set_ylabel("Normalized intensity [a.u.]", fontsize=9)
ax_ts.legend(fontsize=8)

output_path = os.path.join(out_dir, "gamma_m_vs_k_blob_461nm.png")
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
