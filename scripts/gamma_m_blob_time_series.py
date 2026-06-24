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

# Blob ROIs: two spherical features at 0.4-0.6 eV, ±0.4-0.8 Å⁻¹ (Gamma-M S)
BLOB_ENERGY  = (0.49, 0.61)
BLOB_L_MOM   = (-0.75, -0.5)  # left blob
BLOB_R_MOM   = ( 0.5,  0.75)  # right blob

# Reference ROI just above E_F
REF_ENERGY    = (0.0, 0.2)
REF_MOMENTUM  = (-0.1, 0.1)

ENERGY_BIAS = 31.5  # eV — kinetic energy at E_F

TIME_AXIS_MIN, TIME_AXIS_MAX = -150, 900
COMPARISON_TIME_AXIS_MAX = 2000

# Time window for ARPES snapshot shown in left column
ARPES_SNAPSHOT_RANGE = (220, 280)  # +250 fs — blobs should be prominent here
ARPES_ENERGY_MIN, ARPES_ENERGY_MAX = 0.2, 0.715
ARPES_MOMENTUM_CUTOFF = 1.0

_di_400_gm_s = [
    DataItem(data_root_dir, ["Sep", "851"], tag="31.95eV"),
    DataItem(data_root_dir, ["Sep", "852"], tag="32.6eV"),
    DataItem(data_root_dir, ["Sep", "853"], tag="33.25eV"),
    DataItem(data_root_dir, ["Sep", "854"], tag="33.9eV"),
]

_di_461_gm_s = [
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "31.95eV_0"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "32.57_sum"],
        tag="32.57eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "33.19_sum"],
        tag="33.19eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "33.81eV_0"],
        tag="33.81eV",
    ),
]

DATASETS = [
    {
        "title": "400 nm",
        "color_L":   "tab:blue",
        "color_R":   "tab:cyan",
        "color_ref": "tab:gray",
        "data_items": _di_400_gm_s,
        "stitch_fn": concat_energy_datas_smooth_stitching,
    },
    {
        "title": "461 nm",
        "color_L":   "tab:orange",
        "color_R":   "tab:red",
        "color_ref": "tab:gray",
        "data_items": _di_461_gm_s,
        "stitch_fn": concat_energy_datas,
    },
]


def gauss_exp_model(t, B, A, t0, sigma, tau):
    arg = (sigma / tau - (t - t0) / sigma) / np.sqrt(2)
    return B + A * 0.5 * np.exp(0.5 * (sigma / tau) ** 2 - (t - t0) / tau) * erfc(arg)


def detect_onset(t_data, y_data, search_start=0):
    mask = t_data < -200
    if not np.any(mask):
        return float(t_data[0])
    baseline = np.nanmedian(y_data[mask])
    for t, y in zip(t_data, y_data):
        if t >= search_start and np.isfinite(y) and y >= 2.0 * baseline:
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
        bounds = ([0, 0, -300, 10, 50], [0.5, 5, 300, 300, 5000])
        popt, _ = curve_fit(
            gauss_exp_model, t_fit, y_fit, p0=p0, bounds=bounds, maxfev=10000
        )
        return popt
    except RuntimeError as e:
        print(f"  Fit failed: {e}")
        return None


def _plot_fit(ax, t_data, popt, color, label=None):
    t_smooth = np.linspace(
        float(t_data[np.isfinite(t_data)].min()),
        float(t_data[np.isfinite(t_data)].max()),
        500,
    )
    ax.plot(
        t_smooth,
        gauss_exp_model(t_smooth, *popt),
        "-",
        color=color,
        linewidth=2,
        alpha=0.7,
        label=label,
    )


# ── Load ──────────────────────────────────────────────────────────────────────

loaded = []
for config in DATASETS:
    print(f"Loading {config['title']}...")
    loaded.append(load_and_prepare(config))

# ── Extract time series ───────────────────────────────────────────────────────

print("Extracting ROI time series...")
ts_L, ts_R, ts_ref = [], [], []
for data in loaded:
    tL, yL = extract_roi_ts(data, BLOB_ENERGY, BLOB_L_MOM)
    tR, yR = extract_roi_ts(data, BLOB_ENERGY, BLOB_R_MOM)
    _,  yref = extract_roi_ts(data, REF_ENERGY, REF_MOMENTUM)
    ts_L.append((tL, yL))
    ts_R.append((tR, yR))
    ts_ref.append(yref)

# Onset alignment on left-blob signal
onsets = [detect_onset(t, y) for t, y in ts_L]
print(f"Detected left-blob onsets: {[f'{o:.0f} fs' for o in onsets]}")
offset = onsets[0] - onsets[1]
print(f"Time offset applied to 461 nm in comparison panel: {offset:+.0f} fs")

# ── Build figure ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(22, 9))
gs = GridSpec(
    2, 3,
    figure=fig,
    width_ratios=[1, 1.8, 1.5],
    hspace=0.42,
    wspace=0.32,
    left=0.06,
    right=0.97,
    top=0.92,
    bottom=0.1,
)

ax_arpes = [fig.add_subplot(gs[r, 0]) for r in range(2)]
ax_ts    = [fig.add_subplot(gs[r, 1]) for r in range(2)]
ax_comp  = fig.add_subplot(gs[:, 2])

fig.suptitle(r"$\Gamma$-M Blob Dynamics (probe S)", fontsize=13)

for idx, config in enumerate(DATASETS):
    data   = loaded[idx]
    tL, yL = ts_L[idx]
    tR, yR = ts_R[idx]
    yref   = ts_ref[idx]
    c_L    = config["color_L"]
    c_R    = config["color_R"]
    c_ref  = config["color_ref"]
    title  = config["title"]

    # ARPES snapshot at +250 fs with ROI boxes
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
    else:
        ax_a.text(0.5, 0.5, "N/A (no +250 fs)", transform=ax_a.transAxes,
                  ha="center", va="center", fontsize=10, color="gray")
    ax_a.set_ylim(ARPES_ENERGY_MIN, ARPES_ENERGY_MAX)
    ax_a.set_xlim(-ARPES_MOMENTUM_CUTOFF, ARPES_MOMENTUM_CUTOFF)
    ax_a.set_ylabel(r"$E - E_F$ [eV]", fontsize=9)
    ax_a.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=9)
    ax_a.set_title(f"{title}  (t = +250 fs)", fontsize=10)

    for mom_range, color, label in [
        (BLOB_L_MOM, c_L, "L blob"),
        (BLOB_R_MOM, c_R, "R blob"),
        (REF_MOMENTUM, c_ref, "Ref"),
    ]:
        rect = Rectangle(
            (mom_range[0], BLOB_ENERGY[0] if label != "Ref" else REF_ENERGY[0]),
            mom_range[1] - mom_range[0],
            (BLOB_ENERGY[1] - BLOB_ENERGY[0]) if label != "Ref" else (REF_ENERGY[1] - REF_ENERGY[0]),
            fill=False,
            edgecolor=color,
            linewidth=1.5,
            linestyle="--",
        )
        ax_a.add_patch(rect)
        e_center = ((BLOB_ENERGY[0] + BLOB_ENERGY[1]) / 2
                    if label != "Ref"
                    else (REF_ENERGY[0] + REF_ENERGY[1]) / 2)
        ax_a.text(
            mom_range[1] + 0.03,
            e_center,
            label,
            color=color,
            fontsize=7,
            va="center",
        )

    # Individual time series
    ax_t = ax_ts[idx]
    ax_t.plot(tL, yL, "o-", color=c_L, markersize=3,
              label=f"L blob ({BLOB_L_MOM[0]}–{BLOB_L_MOM[1]} Å⁻¹)")
    ax_t.plot(tR, yR, "s-", color=c_R, markersize=3,
              label=f"R blob ({BLOB_R_MOM[0]}–{BLOB_R_MOM[1]} Å⁻¹)")
    ax_t.plot(tL, yref, "^--", color=c_ref, markersize=3,
              label=f"Ref ({REF_ENERGY[0]}–{REF_ENERGY[1]} eV)")

    for t_data, y_data, color, tag in [
        (tL, yL, c_L, "L"),
        (tR, yR, c_R, "R"),
    ]:
        popt = gauss_exp_fit(t_data, y_data)
        if popt is not None:
            B, A, t0, sigma, tau = popt
            _plot_fit(ax_t, t_data, popt, color,
                      label=f"fit {tag}: τ={tau:.0f} fs, t₀={t0:.0f} fs")
            print(f"  {title} {tag}-blob fit: τ={tau:.0f} fs, t₀={t0:.0f} fs, σ={sigma:.0f} fs")

    ax_t.axvline(0, color="gray", linestyle=":", linewidth=0.8)
    ax_t.set_xlim(TIME_AXIS_MIN, TIME_AXIS_MAX)
    ax_t.set_title(f"{title} — blob time series", fontsize=10)
    ax_t.set_xlabel("Time delay [fs]", fontsize=9)
    ax_t.set_ylabel("Normalized intensity [a.u.]", fontsize=9)
    ax_t.legend(fontsize=7)

# Comparison panel — left blob only, onset-aligned
for idx, config in enumerate(DATASETS):
    tL, yL = ts_L[idx]
    t_plot = tL + (offset if idx == 1 else 0)
    shift_str = f" (t→t{offset:+.0f} fs)" if (idx == 1 and abs(offset) > 5) else ""
    ax_comp.plot(t_plot, yL, "o-", color=config["color_L"], markersize=3,
                 label=f"{config['title']} L blob{shift_str}")

    popt = gauss_exp_fit(tL, yL)
    if popt is not None:
        t_smooth = np.linspace(
            float(tL[np.isfinite(tL)].min()),
            float(tL[np.isfinite(tL)].max()),
            500,
        )
        ax_comp.plot(
            t_smooth + (offset if idx == 1 else 0),
            gauss_exp_model(t_smooth, *popt),
            "-", color=config["color_L"], linewidth=2, alpha=0.7,
        )

ax_comp.axvline(0, color="gray", linestyle=":", linewidth=0.8)
ax_comp.set_xlim(TIME_AXIS_MIN, COMPARISON_TIME_AXIS_MAX)
ax_comp.set_title("L blob comparison (onset-aligned)", fontsize=10)
ax_comp.set_xlabel("Time delay [fs]", fontsize=9)
ax_comp.set_ylabel("Normalized intensity [a.u.]", fontsize=9)
ax_comp.legend(fontsize=8)

output_path = os.path.join(out_dir, "gamma_m_blob_time_series.png")
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
