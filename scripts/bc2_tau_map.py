import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

dir = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from analysis.Bi2Se3_BandStructure import Bi2Se3_BandStructure_Gamma_K

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

# Restrict tau map to the BC2 energy window
ENERGY_LOW = 2.0
ENERGY_HIGH = 2.6
MOMENTUM_LOW = -0.75
MOMENTUM_HIGH = 0.75

TAU_CONFIGS = [
    {
        "title": "400 nm Tau (BC2 region)",
        "pump_wavelength": 400,
        "data_items": [
            DataItem(data_root_dir, ["Sep", "837"], tag="31.95eV"),
            DataItem(data_root_dir, ["Sep", "840"], tag="32.6eV"),
            DataItem(data_root_dir, ["Sep", "841"], tag="33.25eV"),
            DataItem(data_root_dir, ["Sep", "842"], tag="33.9eV"),
        ],
        "stitch_fn": concat_energy_datas_smooth_stitching,
    },
    {
        "title": "460 nm Tau (BC2 region)",
        "pump_wavelength": 460,
        "data_items": [
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "31.95_sum"], tag="31.95eV"),
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "32.6_sum"], tag="32.6eV"),
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.25_sum"], tag="33.25eV"),
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.9_sum"], tag="33.9eV"),
        ],
        "stitch_fn": concat_energy_datas,
    },
]

TAU_COLOR_MIN, TAU_COLOR_MAX = 0, 100  # fs — focused on BC2 dynamics range
COLOR_SCALES_COUNT = 70


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
    for prep in [config["stitch_fn"], bin_sweep_datas, subtract_noise, remove_bias,
                 remove_e_fermi, pad_energy, sum_data_prep]:
        plot_data = prep(plot_data)
    return plot_data[0][0]


def compute_taus(sweep_data, config_title):
    _, _, _, taus = pixel_resolved_time_dynamics(
        sweep_data,
        bin_params=[12, 4],
        window_size=[1, 1],
        export_name=config_title.replace(" ", "_").replace("(", "").replace(")", ""),
        filter_based_on_amplitude=False,
    )
    taus = taus.transpose(1, 0)
    energy_axis = np.linspace(*get_axis_min_max(sweep_data, Axes.ENERGY), sweep_data.x_axis_count)
    angle_axis = np.linspace(*get_axis_min_max(sweep_data, Axes.THETA_X), sweep_data.y_axis_count)
    x, y, c_tau, color_scale_tau = create_scatter_plot_data(
        taus,
        energy_axis,
        angle_axis,
        bin_params=[12, 4],
        color_scales_count=COLOR_SCALES_COUNT,
        log_color_scale=False,
        energy_bias=31.5,
        angle_to_momentum=True,
    )
    extent = (
        max(MOMENTUM_LOW, min(y)),
        min(MOMENTUM_HIGH, max(y)),
        max(ENERGY_LOW, min(x)),
        min(ENERGY_HIGH, max(x)),
    )
    return x, y, c_tau, color_scale_tau, extent, sweep_data


# Load data and compute taus
scatter_results = []
for config in TAU_CONFIGS:
    print(f"Loading {config['title']}...")
    sweep_data = load_and_prepare(config)
    print(f"  Computing pixel-resolved decay times...")
    result = compute_taus(sweep_data, config["title"])
    scatter_results.append(result)

shared_color_scale = np.linspace(TAU_COLOR_MIN, TAU_COLOR_MAX, COLOR_SCALES_COUNT)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    f"BC2 Pixel-Resolved Decay Times — Γ-K probe P\n(energy {ENERGY_LOW}–{ENERGY_HIGH} eV)",
    fontsize=13,
)

for ax, config, (x, y, c_tau, _, extent, data) in zip(axes, TAU_CONFIGS, scatter_results):
    cmap = plt.colormaps.get_cmap("terrain").resampled(len(shared_color_scale))
    norm = BoundaryNorm(shared_color_scale, ncolors=cmap.N, clip=True)
    _, _, grid_z = scatter_to_grid(
        x, y,
        data.y_axis_count,
        data.x_axis_count,
        c_tau,
        extent[2], extent[3],
        extent[0], extent[1],
    )
    grid_z = np.abs(grid_z)
    sc = ax.imshow(
        grid_z,
        aspect="auto",
        extent=extent,
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(config["title"], fontsize=11)
    ax.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=10)
    ax.set_ylabel(r"$E - E_F\ [\mathrm{eV}]$", fontsize=10)
    plt.colorbar(sc, ax=ax, label="τ [fs]")
    Bi2Se3_BandStructure_Gamma_K.plot(
        ax,
        momentum_low_limit=extent[0],
        momentum_high_limit=extent[1],
        energy_low_limit=extent[2],
        energy_high_limit=extent[3],
        plot_band_names=False,
    )
    ax.set_ylim(ENERGY_LOW, ENERGY_HIGH)

output_path = os.path.join(dir, f"bc2_tau_map_{ENERGY_LOW}eV_{ENERGY_HIGH}eV.png")
fig.savefig(output_path, dpi=150)
print(f"Saved: {output_path}")

