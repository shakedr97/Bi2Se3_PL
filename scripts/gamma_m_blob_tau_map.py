import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

dir = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from analysis.Bi2Se3_BandStructure import Bi2Se3_BandStructure_Gamma_M

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

ENERGY_LOW = 0.3
ENERGY_HIGH = 0.715
MOMENTUM_LOW = -0.75
MOMENTUM_HIGH = 0.75

# Fit exponential from this time point onward (fs); avoids t=0 pulse artifact
STARTING_POINT = 0

TAU_COLOR_MIN = 50  # fs
TAU_COLOR_MAX = 2000  # fs  — covers expected blob lifetime ~330–380 fs

TAU_CONFIGS = [
    {
        "title": "400 nm (Γ-M blob, probe S)",
        "data_items": [
            DataItem(data_root_dir, ["Sep", "851"], tag="31.95eV"),
            DataItem(data_root_dir, ["Sep", "852"], tag="32.6eV"),
            DataItem(data_root_dir, ["Sep", "853"], tag="33.25eV"),
            DataItem(data_root_dir, ["Sep", "854"], tag="33.9eV"),
        ],
        "stitch_fn": concat_energy_datas_smooth_stitching,
    },
    {
        "title": "461 nm (Γ-M blob, probe S)",
        "data_items": [
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
        ],
        "stitch_fn": concat_energy_datas,
    },
]


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


# Load data
loaded = []
for config in TAU_CONFIGS:
    print(f"Loading {config['title']}...")
    loaded.append(load_and_prepare(config))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    r"$\Gamma$-M Blob — Pixel-Resolved Decay Times (probe S)"
    f"\n(energy {ENERGY_LOW}–{ENERGY_HIGH} eV, fit from t > {STARTING_POINT} fs)",
    fontsize=13,
)

for ax, config, data in zip(axes, TAU_CONFIGS, loaded):
    print(f"  Computing decay times for {config['title']}...")
    # display_pixel_resolved_decay_times expects raw KE on x_axis with energy_offset
    # converting KE → E-EF. After remove_e_fermi, x_axis_minimum is already in
    # E-EF units, so shift it back: x_min_KE = x_min_EEF - energy_offset.
    # energy_offset stays at its calibrated value; the function then computes
    # energy = KE + offset correctly and the momentum conversion uses true KE.
    data.x_axis_minimum -= data.energy_offset
    cax = ax.inset_axes([0, 1.02, 1, 0.04])
    display_pixel_resolved_decay_times(
        data,
        STARTING_POINT,
        ax=ax,
        color_min=TAU_COLOR_MIN,
        color_max=TAU_COLOR_MAX,
        energy_limit_low=ENERGY_LOW,
        energy_limit_high=ENERGY_HIGH,
        momentum_limit_low=MOMENTUM_LOW,
        momentum_limit_high=MOMENTUM_HIGH,
        target_shape=(40, 70),
        band_structure=Bi2Se3_BandStructure_Gamma_M,
    )
    ax.set_ylim(ENERGY_LOW, ENERGY_HIGH)
    ax.set_xlim(MOMENTUM_LOW, MOMENTUM_HIGH)
    ax.set_title(config["title"], fontsize=11)
    ax.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=10)
    ax.set_ylabel(r"$E - E_F\ [\mathrm{eV}]$", fontsize=10)
    cb = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("top")
    cb.locator = MaxNLocator(nbins=6)
    cb.update_ticks()
    cb.set_label("Decay Time [fs]")
    cb.ax.xaxis.set_label_position("top")

fig.subplots_adjust(top=0.82, bottom=0.1, left=0.07, right=0.97, wspace=0.3)
output_path = os.path.join(dir, "gamma_m_blob_tau_map.png")
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")
