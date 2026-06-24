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

E_LOW, E_HIGH = 0.3, 0.715
K_CUTOFF = 1.0

time_delays = [
    (-30,  30),   # 0 fs
    (220, 280),   # +250 fs
    (370, 430),   # +400 fs
    (670, 730),   # +700 fs
]
time_labels = ["0 fs", "+250 fs", "+400 fs", "+700 fs"]

# 515nm Gamma-M — 3 energy windows (no 4th window in this dataset)
_di_515_gm = [
    DataItem(data_root_dir, ["Excitons", "515nm", "Gamma-M", "32.25"], tag="32.25eV"),
    DataItem(data_root_dir, ["Excitons", "515nm", "Gamma-M", "32.85"], tag="32.85eV"),
    DataItem(data_root_dir, ["Excitons", "515nm", "Gamma-M", "33.45"], tag="33.45eV"),
]

# 515nm uses concat_energy_datas (no smooth stitching variant available)
_SUM_515 = [concat_energy_datas, remove_bias, remove_e_fermi, pad_energy, sum_data_prep]
_CD_515  = [cd_data_prep, concat_energy_datas, remove_bias, remove_e_fermi, pad_energy]


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


def make_closeup_figure(row_data, row_label, color_map, percentile, fig_title, output_name):
    n_cols = len(time_delays)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.12, wspace=0.03)

    for j, (t1, t2) in enumerate(time_delays):
        ax = axes[j]
        time_keys = sorted(row_data.sweep_raw.keys())
        available = [t for t in time_keys if t1 <= t <= t2]

        if not available:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14, color="gray")
            ax.set_facecolor("#f4f4f4")
        else:
            display_sweep_data(
                row_data,
                axes_choice=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={Axes.TIME: [[t1, t2]]},
                plot_axis=ax,
                color_map=color_map,
                percentile_for_colorscale=percentile,
                angle_to_momentum=True,
                momentum_cutoff=K_CUTOFF,
                display_plot_title=False,
                display_x_label=True,
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
        ax.set_title(time_labels[j], fontsize=11)
        ax.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=10)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5, prune="both"))

    axes[0].set_ylabel(f"{row_label}\n" + r"$E-E_F$ [eV]", fontsize=10)
    fig.suptitle(fig_title, fontsize=13)
    output_path = os.path.join(dir, output_name)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


print("Loading 515 nm SUM...")
sum_515_gm = _prep(load_raw(_di_515_gm), _SUM_515)
print("Loading 515 nm CD...")
cd_515_gm = _prep(load_raw(_di_515_gm), _CD_515)

make_closeup_figure(
    sum_515_gm, "515 nm",
    color_map="terrain", percentile=99,
    fig_title=r"$\Gamma$-M Blob Close-Up — SUM (515 nm, probe combined)",
    output_name="gamma_m_blob_closeup_SUM_Gamma_M_515nm.png",
)
make_closeup_figure(
    cd_515_gm, "515 nm",
    color_map="bwr", percentile=99,
    fig_title=r"$\Gamma$-M Blob Close-Up — CD (515 nm, probe combined)",
    output_name="gamma_m_blob_closeup_CD_Gamma_M_515nm.png",
)
