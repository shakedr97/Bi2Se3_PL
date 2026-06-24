import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm

dir = os.path.dirname(__file__)

os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from analysis.Bi2Se3_BandStructure import (
    Bi2Se3_BandStructure_Gamma_M,
    Bi2Se3_BandStructure_Gamma_K,
)
from SweepData import SweepData

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)


def compute_taus_data(
    data: SweepData,
    bin_params=[12, 4],
    window_size=[1, 1],
    export_name="taus",
    bias=31.5,
    momentum_low_limit=-0.75,
    momentum_high_limit=0.75,
    energy_low_limit=0,
    energy_high_limit=3.0,
    taus_percentiles=(5, 50),
    color_scales_count=70,
):
    _, _, _, taus = pixel_resolved_time_dynamics(
        data,
        bin_params=bin_params,
        window_size=window_size,
        export_name=export_name.replace("-", "_").replace(" ", "_"),
        filter_based_on_amplitude=False,
        save_output=True,
        overwrite=False,
    )
    taus = taus.transpose(1, 0)
    energy_axis = np.linspace(*get_axis_min_max(data, Axes.ENERGY), data.x_axis_count)
    angle_axis = np.linspace(*get_axis_min_max(data, Axes.THETA_X), data.y_axis_count)
    x, y, c_tau, color_scale_tau = create_scatter_plot_data(
        taus,
        energy_axis,
        angle_axis,
        bin_params=bin_params,
        color_scales_count=color_scales_count,
        log_color_scale=False,
        low_percentile=taus_percentiles[0],
        high_percentile=taus_percentiles[1],
        energy_bias=bias,
        angle_to_momentum=True,
    )
    extent = (
        max(momentum_low_limit, min(y)),
        min(momentum_high_limit, max(y)),
        max(energy_low_limit, min(x)),
        min(energy_high_limit, max(x)),
    )
    return x, y, c_tau, color_scale_tau, extent, data


def plot_taus_on_axis(
    ax,
    x,
    y,
    c_tau,
    color_scale,
    extent,
    data: SweepData,
    title="Decay time values",
    taus_color_map="terrain",
    band_structure=None,
    pump_wavelength=None,
    probe_energy=6.02,
):
    cmap = cm.get_cmap(taus_color_map, len(color_scale))
    norm = BoundaryNorm(color_scale, ncolors=cmap.N, clip=True)
    _, _, grid_z = scatter_to_grid(
        x,
        y,
        data.y_axis_count,
        data.x_axis_count,
        c_tau,
        extent[2],
        extent[3],
        extent[0],
        extent[1],
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
    ax.set_title(title)
    ax.set_xlabel(r"$k_{x}\left[\AA^{-1}\right]$")
    ax.set_ylabel(r"$E-E_{F}\left[eV\right]$")
    plt.colorbar(sc, ax=ax, label="Tau [fs]")
    if band_structure is not None:
        offsets = None
        if pump_wavelength is not None:
            under_fermi_offset = 1240 / pump_wavelength
            above_fermi_offest = 1240 / pump_wavelength - probe_energy
            offsets = {
                above_fermi_offest: band_structure.bands,
            }
        band_structure.plot(
            ax,
            momentum_low_limit=extent[0],
            momentum_high_limit=extent[1],
            energy_low_limit=extent[2],
            energy_high_limit=extent[3],
            offset_bands=offsets,
        )


def all_wavelengths_tau_comparison(
    energy_low: float = 0.0,
    energy_high: float = 3.0,
    taus_percentile=(5, 50),
    verbose: bool = False,
):
    plt.ion()
    data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"
    tau_configs = [
        {
            "data_items": [
                DataItem(data_root_dir, ["Sep", "837"], tag="31.95eV"),
                DataItem(data_root_dir, ["Sep", "840"], tag="32.6eV"),
                DataItem(data_root_dir, ["Sep", "841"], tag="33.25eV"),
                DataItem(data_root_dir, ["Sep", "842"], tag="33.9eV"),
            ],
            "preparations": [
                (concat_energy_datas_smooth_stitching, {}),
                (bin_sweep_datas, {}),
                (subtract_noise, {}),
                (remove_bias, {}),
                (remove_e_fermi, {}),
                (pad_energy, {}),
                (sum_data_prep, {}),
            ],
            "title": "400nm",
            "pump_wavelength": 400,
        },
        {
            "data_items": [
                DataItem(data_root_dir, ["Nov", "423nm", "2318"], tag="31.95eV"),
                DataItem(
                    data_root_dir,
                    ["Nov", "423nm", "Gamma_K", "ProbeRotator-82", "eV_0", "32.6"],
                    tag="32.6eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Nov", "423nm", "Gamma_K", "ProbeRotator-82", "eV_1", "33.25"],
                    tag="33.25eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Nov", "423nm", "Gamma_K", "ProbeRotator-82", "eV_0", "33.9"],
                    tag="33.9eV",
                ),
            ],
            "preparations": [
                (concat_energy_datas, {}),
                (bin_sweep_datas, {}),
                (subtract_noise, {}),
                (remove_bias, {}),
                (remove_e_fermi, {}),
                (pad_energy, {}),
                (sum_data_prep, {}),
            ],
            "title": "423nm",
            "pump_wavelength": 423,
        },
        {
            "data_items": [
                # DataItem(
                #     data_root_dir,
                #     ["Excitons", "452nm", "Gamma-K", "32.25_32.95eV_CD"],
                #     tag="32.25eV",
                # ),
                # DataItem(
                #     data_root_dir,
                #     [
                #         "Excitons",
                #         "452nm",
                #         "Gamma-K",
                #         "Gamma-K_energy_sweep_32.45_34.45eV_CD_1",
                #     ],
                #     tag="32.85eV",
                # ),
                # DataItem(
                #     data_root_dir,
                #     ["Excitons", "452nm", "Gamma-K", "32.85_33.55eV_CD"],
                #     tag="32.85eV",
                # ),
                # DataItem(
                #     data_root_dir,
                #     ["Excitons", "452nm", "Gamma-K", "33.45_34.15eV_CD"],
                #     tag="33.45eV",
                # ),
                # DataItem(
                #     data_root_dir,
                #     ["Sep", "905"],  # 450nm Gamma-K probe P 33.9eV
                #     tag="33.9eV",
                # ),
                DataItem(
                    data_root_dir,
                    ["Sep", "908"],  # 450nm Gamma-K probe P 31.95eV
                    tag="31.95eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Sep", "907"],  # 450nm Gamma-K probe P 32.6eV
                    tag="32.6eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Sep", "906"],  # 450nm Gamma-K probe P 33.25eV
                    tag="33.25eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Sep", "905"],  # 450nm Gamma-K probe P 33.9eV
                    tag="33.9eV",
                ),
            ],
            "preparations": [
                (remove_bias, {}),
                (concat_energy_datas, {}),
                (
                    bin_sweep_datas,
                    {
                        # "bin_params": {Axes.ENERGY: 9, Axes.THETA_X: 2}
                    },
                ),
                (remove_e_fermi, {}),
                (pad_energy, {}),
                (sum_data_prep, {}),
            ],
            "title": "450nm",
            "pump_wavelength": 450,
        },
        {
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
            "preparations": [
                (concat_energy_datas, {}),
                (bin_sweep_datas, {}),
                (subtract_noise, {}),
                (remove_bias, {}),
                (remove_e_fermi, {}),
                (pad_energy, {}),
                (sum_data_prep, {}),
            ],
            "title": "460nm",
            "pump_wavelength": 460,
        },
        {
            "data_items": [
                DataItem(
                    data_root_dir,
                    ["Excitons", "515nm", "Gamma-K", "32.25_32.95eV_CD"],
                    tag="32.25eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Excitons", "515nm", "Gamma-K", "32.85_33.55eV_CD"],
                    tag="32.85eV",
                ),
                DataItem(
                    data_root_dir,
                    ["Excitons", "515nm", "Gamma-K", "33.45_34.15eV_CD"],
                    tag="33.45eV",
                ),
            ],
            "preparations": [
                (concat_energy_datas, {}),
                (bin_sweep_datas, {}),
                (subtract_noise, {}),
                (remove_bias, {}),
                (remove_e_fermi, {}),
                (pad_energy, {}),
                (sum_data_prep, {}),
            ],
            "title": "515nm",
            "pump_wavelength": 515,
        },
    ]

    # Compute taus for all datasets
    scatter_results = []
    for config in tau_configs:
        if verbose:
            print(f"Processing {config['title']}...")
        plot_data = []
        for data_item in config["data_items"]:
            if verbose:
                print(f"  Loading {data_item.search_keywords}")
            try:
                datas = read_sweep_data(
                    data_item.base_directory, data_item.search_keywords
                )
                if len(datas) <= 1:
                    raise Exception()
            except:
                datas = read_all_sweep_data(
                    data_item.base_directory, data_item.search_keywords
                )
                if len(datas) == 0:
                    raise
            plot_data.append(datas)
        for preparation, kwargs in config["preparations"]:
            plot_data = preparation(plot_data, **kwargs)
        sweep_data = plot_data[0][0]
        result = compute_taus_data(
            sweep_data,
            export_name=config["title"],
            energy_low_limit=energy_low,
            energy_high_limit=energy_high,
            taus_percentiles=taus_percentile,
        )
        scatter_results.append(result)

    # Build a shared color scale from the union of all ranges
    all_low = min(cs[3][0] for cs in scatter_results)
    all_high = max(cs[3][-1] for cs in scatter_results)
    shared_color_scale = np.linspace(all_low, all_high, 70)

    # Plot all with the shared color scale
    n_plots = len(tau_configs)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 6), constrained_layout=True
    )
    fig.suptitle("Tau Comparison: All Wavelengths (Gamma-K probe P)", fontsize=14)
    for ax, config, (x, y, c_tau, _, extent, data) in zip(
        axes, tau_configs, scatter_results
    ):
        plot_taus_on_axis(
            ax,
            x,
            y,
            c_tau,
            shared_color_scale,
            extent,
            data,
            title=config["title"],
            band_structure=Bi2Se3_BandStructure_Gamma_K,
            pump_wavelength=config["pump_wavelength"],
        )

    output_path = os.path.join(
        dir, f"tau_comparison_all_wavelengths_{energy_low}eV_{energy_high}eV.png"
    )
    fig.savefig(output_path)
    if verbose:
        print(f"Saved tau comparison to {output_path}")
    plt.show()


all_wavelengths_tau_comparison(
    verbose=True, energy_low=0.0, energy_high=1.0, taus_percentile=(60, 98)
)
# all_wavelengths_tau_comparison(
#     verbose=True, energy_low=1.0, energy_high=3.0, taus_percentile=(5, 70)
# )
