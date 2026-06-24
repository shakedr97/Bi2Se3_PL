import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

dir = os.path.dirname(__file__)

os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from SweepData import SweepData

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)


def load_dataset(config, verbose=False):
    """Load and prepare a single dataset from a config dict."""
    plot_data = []
    for data_item in config["data_items"]:
        if verbose:
            print(f"  Loading {data_item.search_keywords}")
        try:
            datas = read_sweep_data(data_item.base_directory, data_item.search_keywords)
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
    return plot_data[0][0]


def roi_time_series_comparison(
    dataset_configs,
    rois,
    energy_bias=31.5,
    verbose=False,
):
    """
    Plot time series of integrated counts within ROIs for multiple datasets.

    dataset_configs: list of dicts, each with "data_items", "preparations", "title"
    rois: list of dicts, each with:
        "energy": (low, high) in eV (E - E_F)
        "momentum": (low, high) in Å^-1
        "label": str for legend
    """
    plt.ion()

    # Load all datasets
    datasets = []
    for config in dataset_configs:
        if verbose:
            print(f"Processing {config['title']}...")
        data = load_dataset(config, verbose=verbose)
        datasets.append(data)

    # Create figure
    n_plots = len(dataset_configs)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 5), constrained_layout=True, sharey=True
    )
    if n_plots == 1:
        axes = [axes]

    for ax, config, data in zip(axes, dataset_configs, datasets):
        for roi in rois:
            e_low, e_high = roi["energy"]
            k_low, k_high = roi["momentum"]
            energy_midpoint = (e_low + e_high) / 2
            theta_min = momentum_to_angle(k_low, energy_bias + energy_midpoint)
            theta_max = momentum_to_angle(k_high, energy_bias + energy_midpoint)

            display_sweep_data(
                data,
                axes_choice=[Axes.TIME],
                summing_ranges={
                    Axes.THETA_X: [[theta_min, theta_max]],
                    Axes.ENERGY: [[e_low, e_high]],
                },
                plot_axis=ax,
                display_plot_title=False,
                normalize=True,
            )

        ax.set_title(config["title"])
        ax.legend([roi["label"] for roi in rois])

    fig.suptitle("ROI Time Series Comparison", fontsize=14)
    output_path = os.path.join(dir, "roi_time_series_comparison.png")
    fig.savefig(output_path)
    if verbose:
        print(f"Saved figure to {output_path}")
    plt.show()


# --- Configuration ---

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

dataset_configs = [
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
    },
    # {
    #     "data_items": [
    #         DataItem(data_root_dir, ["Nov", "423nm", "2318"], tag="31.95eV"),
    #         DataItem(
    #             data_root_dir,
    #             ["Nov", "423nm", "Gamma_K", "ProbeRotator-82", "eV_0", "32.6"],
    #             tag="32.6eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Nov", "423nm", "Gamma_K", "ProbeRotator-82", "eV_1", "33.25"],
    #             tag="33.25eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Nov", "423nm", "Gamma_K", "ProbeRotator-82", "eV_0", "33.9"],
    #             tag="33.9eV",
    #         ),
    #     ],
    #     "preparations": [
    #         (concat_energy_datas, {}),
    #         (bin_sweep_datas, {}),
    #         (subtract_noise, {}),
    #         (remove_bias, {}),
    #         (remove_e_fermi, {}),
    #         (pad_energy, {}),
    #         (sum_data_prep, {}),
    #     ],
    #     "title": "423nm",
    # },
    # {
    #     "data_items": [
    #         DataItem(
    #             data_root_dir,
    #             ["Sep", "908"],
    #             tag="31.95eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Sep", "907"],
    #             tag="32.6eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Sep", "906"],
    #             tag="33.25eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Sep", "905"],
    #             tag="33.9eV",
    #         ),
    #     ],
    #     "preparations": [
    #         (remove_bias, {}),
    #         (concat_energy_datas, {}),
    #         (bin_sweep_datas, {}),
    #         (remove_e_fermi, {}),
    #         (pad_energy, {}),
    #         (sum_data_prep, {}),
    #     ],
    #     "title": "450nm",
    # },
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
    },
    # {
    #     "data_items": [
    #         DataItem(
    #             data_root_dir,
    #             ["Excitons", "515nm", "Gamma-K", "32.25_32.95eV_CD"],
    #             tag="32.25eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Excitons", "515nm", "Gamma-K", "32.85_33.55eV_CD"],
    #             tag="32.85eV",
    #         ),
    #         DataItem(
    #             data_root_dir,
    #             ["Excitons", "515nm", "Gamma-K", "33.45_34.15eV_CD"],
    #             tag="33.45eV",
    #         ),
    #     ],
    #     "preparations": [
    #         (concat_energy_datas, {}),
    #         (bin_sweep_datas, {}),
    #         (subtract_noise, {}),
    #         (remove_bias, {}),
    #         (remove_e_fermi, {}),
    #         (pad_energy, {}),
    #         (sum_data_prep, {}),
    #     ],
    #     "title": "515nm",
    # },
]

rois = [
    {"energy": (2.25, 2.35), "momentum": (-0.15, -0.1), "label": "E=highest_band"},
    {"energy": (0, 0.1), "momentum": (-0.1, 0.1), "label": "E=straight_below"},
]

roi_time_series_comparison(dataset_configs, rois, verbose=True)
