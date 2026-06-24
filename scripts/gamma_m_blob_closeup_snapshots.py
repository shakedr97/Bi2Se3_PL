import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

dir = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from analysis.Bi2Se3_BandStructure import Bi2Se3_BandStructure_Gamma_M

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

# Blob display window
E_LOW, E_HIGH = 0.3, 0.715
K_CUTOFF = 1.0

time_delays = [
    (-30,  30),   # 0 fs
    (220, 280),   # +250 fs
    (370, 430),   # +400 fs
    (670, 730),   # +700 fs
]
time_labels = ["0 fs", "+250 fs", "+400 fs", "+700 fs"]

# ── DataItem definitions ───────────────────────────────────────────────────────

# 400nm Gamma-M S
_di_400_gm_s = [
    DataItem(data_root_dir, ["Sep", "851"], tag="31.95eV"),
    DataItem(data_root_dir, ["Sep", "852"], tag="32.6eV"),
    DataItem(data_root_dir, ["Sep", "853"], tag="33.25eV"),
    DataItem(data_root_dir, ["Sep", "854"], tag="33.9eV"),
]

# 461nm Gamma-M S
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

# ── Pipelines ─────────────────────────────────────────────────────────────────

_SUM_SMOOTH = [
    concat_energy_datas_smooth_stitching,
    remove_bias,
    remove_e_fermi,
    pad_energy,
    subtract_noise,
    sum_data_prep,
]
_CD_SMOOTH = [
    cd_data_prep,
    concat_energy_datas_smooth_stitching_no_filling,
    remove_bias,
    remove_e_fermi,
    pad_energy,
]
_SUM_461 = [concat_energy_datas, remove_bias, remove_e_fermi, pad_energy, sum_data_prep]
_CD_461 = [cd_data_prep, concat_energy_datas, remove_bias, remove_e_fermi, pad_energy]


def load_raw(data_items):
    plot_data = []
    for di in data_items:
        print(f"  Loading {di.search_keywords}")
        try:
            datas = read_sweep_data(di.base_directory, di.search_keywords)
            if len(datas) <= 1:
                raise Exception()
        except Exception:
            datas = read_all_sweep_data(di.base_directory, di.search_keywords)
        plot_data.append(datas)
    return plot_data


def _prep(plot_data, pipeline):
    for fn in pipeline:
        plot_data = fn(plot_data)
    return plot_data[0][0]


# ── Figure generation ──────────────────────────────────────────────────────────


def make_closeup_figure(
    row_specs,
    row_labels,
    color_map,
    percentile,
    fig_title,
    output_name,
):
    """
    row_specs: list of rows. Each row is either:
      - a single SweepData  (same dataset for all time-delay columns), or
      - a list[SweepData]   (one dataset per column, length == len(time_delays))
    """
    n_rows = len(row_specs)
    n_cols = len(time_delays)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.1, wspace=0.03, hspace=0.15
    )
    if n_rows == 1:
        axes = [axes]

    for i, (row_spec, label) in enumerate(zip(row_specs, row_labels)):
        col_data = row_spec if isinstance(row_spec, list) else [row_spec] * n_cols

        for j, (t1, t2) in enumerate(time_delays):
            ax = axes[i][j]
            data = col_data[j]
            time_keys = sorted(data.sweep_raw.keys())
            available = [t for t in time_keys if t1 <= t <= t2]

            if not available:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="gray",
                )
                ax.set_facecolor("#f4f4f4")
            else:
                display_sweep_data(
                    data,
                    axes_choice=[Axes.THETA_X, Axes.ENERGY],
                    summing_ranges={Axes.TIME: [[t1, t2]]},
                    plot_axis=ax,
                    color_map=color_map,
                    percentile_for_colorscale=percentile,
                    angle_to_momentum=True,
                    momentum_cutoff=K_CUTOFF,
                    display_plot_title=False,
                    display_x_label=(i == n_rows - 1),
                    display_y_label=False,
                )
                Bi2Se3_BandStructure_Gamma_M.plot(
                    ax,
                    momentum_low_limit=-K_CUTOFF,
                    momentum_high_limit=K_CUTOFF,
                    energy_low_limit=E_LOW,
                    energy_high_limit=E_HIGH,
                    plot_band_names=False,
                )

            ax.set_ylim(E_LOW, E_HIGH)
            ax.set_xlim(-K_CUTOFF, K_CUTOFF)
            if i == 0:
                ax.set_title(time_labels[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{label}\n" + r"$E-E_F$ [eV]", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=10)

    for row in axes:
        for ax in row:
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="both"))

    fig.suptitle(fig_title, fontsize=13)
    output_path = os.path.join(dir, output_name)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Load data ─────────────────────────────────────────────────────────────────

print("\n=== Gamma-M blob, probe S ===")
print("  Loading 400 nm SUM...")
sum_400_gm_s = _prep(load_raw(_di_400_gm_s), _SUM_SMOOTH)
print("  Loading 400 nm CD...")
cd_400_gm_s = _prep(load_raw(_di_400_gm_s), _CD_SMOOTH)
print("  Loading 461 nm SUM...")
sum_461_gm_s = _prep(load_raw(_di_461_gm_s), _SUM_461)
print("  Loading 461 nm CD...")
cd_461_gm_s = _prep(load_raw(_di_461_gm_s), _CD_461)

# ── Generate figures ──────────────────────────────────────────────────────────

make_closeup_figure(
    row_specs=[sum_461_gm_s, sum_400_gm_s],
    row_labels=["461 nm", "400 nm"],
    color_map="terrain",
    percentile=99,
    fig_title=r"$\Gamma$-M Blob Close-Up — SUM (probe S)",
    output_name="gamma_m_blob_closeup_SUM_Gamma_M_S.png",
)
make_closeup_figure(
    row_specs=[cd_461_gm_s, cd_400_gm_s],
    row_labels=["461 nm", "400 nm"],
    color_map="bwr",
    percentile=99,
    fig_title=r"$\Gamma$-M Blob Close-Up — CD (probe S)",
    output_name="gamma_m_blob_closeup_CD_Gamma_M_S.png",
)
