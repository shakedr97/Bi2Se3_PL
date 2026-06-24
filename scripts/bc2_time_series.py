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

# BC2 ROI: band sits at 2.25–2.36 eV, ±0.132 Å⁻¹
BC2_ENERGY = (2.23, 2.37)
BC2_MOMENTUM = (-0.132, 0.132)

# Reference ROI just above E_F
REF_ENERGY = (0.0, 0.2)
REF_MOMENTUM = (-0.1, 0.1)

ENERGY_BIAS = 31.5  # eV

TIME_AXIS_MIN, TIME_AXIS_MAX = -150, 750
COMPARISON_TIME_AXIS_MIN, COMPARISON_TIME_AXIS_MAX = -150, 250

# Set to a number (fs) to override auto-detected alignment offset, or None for auto
TIME_OFFSET_FS = None

ARPES_MOMENTUM_CUTOFF = 0.5
ARPES_ENERGY_MIN, ARPES_ENERGY_MAX = -0.3, 2.7
ARPES_SNAPSHOT_RANGE = (30, 80)  # time range capturing t = 50 fs key

DATASETS = [
    {
        "title": "400 nm",
        "color_bc2": "tab:blue",
        "color_ref": "tab:cyan",
        "data_items": [
            DataItem(data_root_dir, ["Sep", "837"], tag="31.95eV"),
            DataItem(data_root_dir, ["Sep", "840"], tag="32.6eV"),
            DataItem(data_root_dir, ["Sep", "841"], tag="33.25eV"),
            DataItem(data_root_dir, ["Sep", "842"], tag="33.9eV"),
        ],
        "stitch_fn": concat_energy_datas_smooth_stitching,
    },
    {
        "title": "460 nm",
        "color_bc2": "tab:orange",
        "color_ref": "tab:red",
        "data_items": [
            DataItem(
                data_root_dir,
                ["Dec", "G_Gamma_K", "ProbeRotator_82", "31.95_sum"],
                tag="31.95eV",
            ),
            DataItem(
                data_root_dir,
                ["Dec", "G_Gamma_K", "ProbeRotator_82", "32.6_sum"],
                tag="32.6eV",
            ),
            DataItem(
                data_root_dir,
                ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.25_sum"],
                tag="33.25eV",
            ),
            DataItem(
                data_root_dir,
                ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.9_sum"],
                tag="33.9eV",
            ),
        ],
        "stitch_fn": concat_energy_datas,
    },
]


def gauss_exp_model(t, B, A, t0, sigma, tau):
    arg = (sigma / tau - (t - t0) / sigma) / np.sqrt(2)
    return B + A * 0.5 * np.exp(0.5 * (sigma / tau) ** 2 - (t - t0) / tau) * erfc(arg)


def detect_onset(t_data, y_data):
    """Return the first t > -200 fs where y >= 2 * median(y[t < -200 fs])."""
    mask = t_data < -200
    if not np.any(mask):
        return float(t_data[0])
    baseline = np.nanmedian(y_data[mask])
    for t, y in zip(t_data, y_data):
        if t > -200 and np.isfinite(y) and y >= 2.0 * baseline:
            return float(t)
    return float(t_data[-1])


def load_and_prepare(config, verbose=True):
    plot_data = []
    for di in config["data_items"]:
        if verbose:
            print(f"  Loading {di.search_keywords}")
        try:
            datas = read_sweep_data(di.base_directory, di.search_keywords)
            if len(datas) <= 1:
                raise Exception()
        except Exception:
            datas = read_all_sweep_data(di.base_directory, di.search_keywords)
        plot_data.append(datas)
    for prep in [
        config["stitch_fn"],
        bin_sweep_datas,
        subtract_noise,
        remove_bias,
        remove_e_fermi,
        pad_energy,
        sum_data_prep,
    ]:
        plot_data = prep(plot_data)
    return plot_data[0][0]


def extract_roi_ts(data, energy_range, momentum_range):
    """Return (t_array, y_array) for an ROI, normalized, via display_sweep_data."""
    e_mid = (energy_range[0] + energy_range[1]) / 2
    th_lo = momentum_to_angle(momentum_range[0], ENERGY_BIAS + e_mid)
    th_hi = momentum_to_angle(momentum_range[1], ENERGY_BIAS + e_mid)
    fig_tmp, ax_tmp = plt.subplots()
    display_sweep_data(
        data,
        axes_choice=[Axes.TIME],
        summing_ranges={
            Axes.THETA_X: [[th_lo, th_hi]],
            Axes.ENERGY: [list(energy_range)],
        },
        plot_axis=ax_tmp,
        normalize=True,
        display_plot_title=False,
    )
    t_arr = ax_tmp.lines[-1].get_xdata().copy().astype(float)
    y_arr = ax_tmp.lines[-1].get_ydata().copy().astype(float)
    plt.close(fig_tmp)
    return t_arr, y_arr


def gauss_exp_fit(t_data, y_data):
    mask = np.isfinite(t_data) & np.isfinite(y_data)
    t_fit, y_fit = t_data[mask], y_data[mask]
    try:
        p0 = [0.05, 0.9, 0.0, 50.0, 400.0]
        bounds = ([0, 0, -300, 10, 10], [0.5, 5, 300, 300, 500])
        popt, _ = curve_fit(
            gauss_exp_model, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000
        )
        return popt
    except RuntimeError as e:
        print(f"  Fit failed: {e}")
        return None


# --- Load ---
loaded = []
for config in DATASETS:
    print(f"Loading {config['title']}...")
    loaded.append(load_and_prepare(config))

# --- Extract time series ---
print("Extracting ROI time series...")
ts_bc2 = []
ts_ref = []
for data in loaded:
    t, y = extract_roi_ts(data, BC2_ENERGY, BC2_MOMENTUM)
    ts_bc2.append((t, y))
    _, y_ref = extract_roi_ts(data, REF_ENERGY, REF_MOMENTUM)
    ts_ref.append(y_ref)

# --- Detect time offset for alignment ---
onsets = [detect_onset(t, y) for t, y in ts_bc2]
print(f"Detected BC2 onsets: {[f'{o:.0f} fs' for o in onsets]}")
offset = TIME_OFFSET_FS if TIME_OFFSET_FS is not None else (onsets[1] - onsets[0])
print(f"Time offset applied to 460 nm in comparison panel: {offset:+.0f} fs")

# --- Build figure ---
fig = plt.figure(figsize=(20, 9))
gs = GridSpec(
    2,
    3,
    figure=fig,
    width_ratios=[1, 1.5, 1.5],
    hspace=0.38,
    wspace=0.32,
    left=0.06,
    right=0.97,
    top=0.92,
    bottom=0.1,
)

ax_arpes = [fig.add_subplot(gs[r, 0]) for r in range(2)]
ax_ts = [fig.add_subplot(gs[r, 1]) for r in range(2)]
ax_comp = fig.add_subplot(gs[:, 2])

fig.suptitle(r"BC2 Band Dynamics — $\Gamma$-K (probe P)", fontsize=13)

for idx, config in enumerate(DATASETS):
    data = loaded[idx]
    t_bc2, y_bc2 = ts_bc2[idx]
    y_ref = ts_ref[idx]
    c_bc2 = config["color_bc2"]
    c_ref = config["color_ref"]
    title = config["title"]

    # --- ARPES snapshot at t=50 with ROI boxes ---
    ax_a = ax_arpes[idx]
    time_keys = sorted(data.sweep_raw.keys())
    has_snapshot = any(
        ARPES_SNAPSHOT_RANGE[0] <= t <= ARPES_SNAPSHOT_RANGE[1] for t in time_keys
    )
    if has_snapshot:
        display_sweep_data(
            data,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [list(ARPES_SNAPSHOT_RANGE)]},
            plot_axis=ax_a,
            color_map="terrain",
            percentile_for_colorscale=99,
            angle_to_momentum=True,
            momentum_cutoff=ARPES_MOMENTUM_CUTOFF,
            display_plot_title=False,
            display_x_label=True,
            display_y_label=False,
        )
    ax_a.set_ylim(ARPES_ENERGY_MIN, ARPES_ENERGY_MAX)
    ax_a.set_xlim(-ARPES_MOMENTUM_CUTOFF, ARPES_MOMENTUM_CUTOFF)
    ax_a.set_ylabel(r"$E - E_F$ [eV]", fontsize=9)
    ax_a.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=9)
    ax_a.set_title(f"{title}  (t = +50 fs)", fontsize=10)

    bc2_rect = Rectangle(
        (BC2_MOMENTUM[0], BC2_ENERGY[0]),
        BC2_MOMENTUM[1] - BC2_MOMENTUM[0],
        BC2_ENERGY[1] - BC2_ENERGY[0],
        fill=False,
        edgecolor=c_bc2,
        linewidth=1.5,
        linestyle="--",
    )
    ref_rect = Rectangle(
        (REF_MOMENTUM[0], REF_ENERGY[0]),
        REF_MOMENTUM[1] - REF_MOMENTUM[0],
        REF_ENERGY[1] - REF_ENERGY[0],
        fill=False,
        edgecolor=c_ref,
        linewidth=1.5,
        linestyle="--",
    )
    ax_a.add_patch(bc2_rect)
    ax_a.add_patch(ref_rect)
    ax_a.text(
        BC2_MOMENTUM[1] + 0.02,
        (BC2_ENERGY[0] + BC2_ENERGY[1]) / 2,
        "BC2",
        color=c_bc2,
        fontsize=8,
        va="center",
    )
    ax_a.text(
        REF_MOMENTUM[1] + 0.02,
        (REF_ENERGY[0] + REF_ENERGY[1]) / 2,
        "Ref",
        color=c_ref,
        fontsize=8,
        va="center",
    )

    # --- Individual time series ---
    ax_t = ax_ts[idx]
    ax_t.plot(
        t_bc2,
        y_bc2,
        "o-",
        color=c_bc2,
        markersize=3,
        label=f"BC2 ({BC2_ENERGY[0]}–{BC2_ENERGY[1]} eV)",
    )
    ax_t.plot(
        t_bc2,
        y_ref,
        "o--",
        color=c_ref,
        markersize=3,
        label=f"Ref ({REF_ENERGY[0]}–{REF_ENERGY[1]} eV)",
    )

    popt = gauss_exp_fit(t_bc2, y_bc2)
    if popt is not None:
        B, A, t0, sigma, tau = popt
        t_smooth = np.linspace(
            float(t_bc2[np.isfinite(t_bc2)].min()),
            float(t_bc2[np.isfinite(t_bc2)].max()),
            500,
        )
        ax_t.plot(
            t_smooth,
            gauss_exp_model(t_smooth, *popt),
            "-",
            color=c_bc2,
            linewidth=2,
            alpha=0.7,
            label=f"fit: τ = {tau:.0f} fs, t₀ = {t0:.0f} fs",
        )
        print(
            f"  {title} BC2 fit: τ = {tau:.0f} fs, t₀ = {t0:.0f} fs, σ = {sigma:.0f} fs"
        )

    ax_t.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_t.set_xlim(TIME_AXIS_MIN, TIME_AXIS_MAX)
    ax_t.set_title(f"{title} BC2 time series", fontsize=10)
    ax_t.set_xlabel("Time delay [fs]", fontsize=9)
    ax_t.set_ylabel("Normalized intensity [a.u.]", fontsize=9)
    ax_t.legend(fontsize=7)

# --- Comparison panel ---
for idx, config in enumerate(DATASETS):
    t_bc2, y_bc2 = ts_bc2[idx]
    t_plot = t_bc2 + (offset if idx == 1 else 0)
    shift_str = f" (t → t{offset:+.0f} fs)" if (idx == 1 and offset != 0) else ""
    ax_comp.plot(
        t_plot,
        y_bc2,
        "o-",
        color=config["color_bc2"],
        markersize=3,
        label=f"{config['title']} BC2{shift_str}",
    )

    popt = gauss_exp_fit(t_bc2, y_bc2)
    if popt is not None:
        t_smooth = np.linspace(
            float(t_bc2[np.isfinite(t_bc2)].min()),
            float(t_bc2[np.isfinite(t_bc2)].max()),
            500,
        )
        t_smooth_plot = t_smooth + (offset if idx == 1 else 0)
        ax_comp.plot(
            t_smooth_plot,
            gauss_exp_model(t_smooth, *popt),
            "-",
            color=config["color_bc2"],
            linewidth=2,
            alpha=0.7,
        )

ax_comp.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax_comp.set_xlim(TIME_AXIS_MIN, COMPARISON_TIME_AXIS_MAX)
ax_comp.set_title("BC2 comparison (onset-aligned)", fontsize=10)
ax_comp.set_xlabel("Time delay [fs]", fontsize=9)
ax_comp.set_ylabel("Normalized intensity [a.u.]", fontsize=9)
ax_comp.legend(fontsize=8)

output_path = os.path.join(out_dir, "bc2_time_series.png")
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
