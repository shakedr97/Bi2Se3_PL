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


def edc_comparison(
    dataset_configs,
    time_ranges,
    momentum_range=(-0.1, 0.1),
    energy_bias=31.5,
    verbose=False,
):
    """
    Plot EDCs (Energy Distribution Curves) for multiple datasets on the same plot.

    dataset_configs: list of dicts with "data_items", "preparations", "title"
    time_ranges: list of (time_low, time_high) tuples, each producing a separate subplot
    momentum_range: (k_low, k_high) in Å^-1 to sum over
    """
    plt.ion()

    # Load all datasets
    datasets = []
    for config in dataset_configs:
        if verbose:
            print(f"Processing {config['title']}...")
        data = load_dataset(config, verbose=verbose)
        datasets.append(data)

    # One subplot per time range
    n_plots = len(time_ranges)
    fig, axes = plt.subplots(
        1, n_plots, figsize=(5 * n_plots, 5), constrained_layout=True, sharey=True
    )
    if n_plots == 1:
        axes = [axes]

    for ax, (t_low, t_high) in zip(axes, time_ranges):
        # Convert momentum range to angle (use a mid-energy for conversion)
        mid_energy = 1.0  # approximate; exact value matters little for small k
        theta_min = momentum_to_angle(momentum_range[0], energy_bias + mid_energy)
        theta_max = momentum_to_angle(momentum_range[1], energy_bias + mid_energy)

        for config, data in zip(dataset_configs, datasets):
            display_sweep_data(
                data,
                axes_choice=[Axes.ENERGY],
                summing_ranges={
                    Axes.THETA_X: [[theta_min, theta_max]],
                    Axes.TIME: [[t_low, t_high]],
                },
                plot_axis=ax,
                display_plot_title=False,
            )

        ax.set_title(f"t = {t_low} to {t_high} fs")
        ax.legend([c["title"] for c in dataset_configs])
        ax.set_xlabel(r"$E - E_F$ [eV]")
        ax.set_ylabel("Counts")

    k_lo, k_hi = momentum_range
    fig.suptitle(
        rf"EDC Comparison ($k_x$ = {k_lo} to {k_hi} $\AA^{{-1}}$)", fontsize=14
    )
    output_path = os.path.join(dir, "edc_comparison.png")
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
]

time_ranges = [
    (-10, 10),
    # (50, 150),
    # (200, 400),
    # (600, 1000),
]

edc_comparison(
    dataset_configs,
    time_ranges,
    momentum_range=(-0.1, 0.1),
    verbose=True,
)
