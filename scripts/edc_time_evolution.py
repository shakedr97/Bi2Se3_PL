import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

dir = os.path.dirname(__file__)

os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from SweepData import SweepData, Axes

plots_dir = os.path.join(dir, "..", "plots")
os.chdir(plots_dir)


def load_dataset(config, verbose=False):
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


def has_time_in_window(data, t_low, t_high):
    return any(t_low <= t <= t_high for t in data.sweep_raw.keys())


def plot_edc_time_evolution(dataset_configs, target_delays, momentum_range=(-0.1, 0.1), energy_bias=31.5, verbose=False):
    datasets = []
    for config in dataset_configs:
        if verbose:
            print(f"Loading {config['title']}...")
        data = load_dataset(config, verbose=verbose)
        datasets.append(data)

    n = len(dataset_configs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    mid_energy = 1.0
    theta_min = momentum_to_angle(momentum_range[0], energy_bias + mid_energy)
    theta_max = momentum_to_angle(momentum_range[1], energy_bias + mid_energy)

    colors = cm.plasma(np.linspace(0.1, 0.9, len(target_delays)))


    for ax, config, data in zip(axes, dataset_configs, datasets):
        if verbose:
            available = sorted(data.sweep_raw.keys())
            print(f"  {config['title']} available time keys: {available}")

        ax.set_prop_cycle(color=colors)

        plotted_labels = []
        for t, hw in target_delays:
            t_low, t_high = t - hw, t + hw
            if not has_time_in_window(data, t_low, t_high):
                if verbose:
                    print(f"  Skipping t={t} fs — no data in [{t_low}, {t_high}]")
                continue

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
            plotted_labels.append(f"t = {t} fs")

        # Dirac point is at -0.24 eV; excitonic signal expected 2.25 eV above it → ~2.01 eV
        ax.axvline(2.01, color="gray", linestyle="--", linewidth=1.2, alpha=0.8, label="2.01 eV (DP+2.25)")
        ax.set_xlabel(r"$E - E_F$ [eV]")
        ax.set_ylabel("Counts")
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e2)
        ax.set_title(config["title"])
        ax.legend(plotted_labels + ["2.01 eV (DP+2.25)"], fontsize=7, loc="upper right")

    k_lo, k_hi = momentum_range
    fig.suptitle(
        rf"EDC at $k_x \approx 0$ (±{abs(k_lo):.2f} Å$^{{-1}}$) — Excitonic Population Check"
        "\n(dashed line: 2.25 eV above Dirac point @ −0.24 eV = 2.01 eV)",
        fontsize=12,
    )

    output_path = os.path.join(plots_dir, "edc_time_evolution.png")
    fig.savefig(output_path, dpi=150)
    if verbose:
        print(f"Saved to {output_path}")
    plt.show()


# --- Configuration ---

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

dataset_configs = [
    {
        "title": "400 nm Γ-K",
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
    },
    {
        "title": "461 nm Γ-K",
        "data_items": [
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "31.95_sum"], tag="31.95eV"),
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "32.6_sum"], tag="32.6eV"),
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.25_sum"], tag="33.25eV"),
            DataItem(data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.9_sum"], tag="33.9eV"),
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
    },
]

# Each entry: (target_fs, half_window_fs).
# -150 uses ±75 so it captures -100 in the 400 nm dataset (which has no point
# between -100 and -900) while still capturing -140 in the 461 nm dataset.
target_delays = [
    (-150, 75),
    (-50, 25),
    (0, 25),
    (50, 25),
    (100, 25),
    (150, 25),
    (250, 25),
    (400, 25),
    (700, 25),
]

plot_edc_time_evolution(
    dataset_configs,
    target_delays,
    momentum_range=(-0.1, 0.1),
    verbose=True,
)

# --- ARPES snapshot with EDC ROI overlay (400 nm only) ---

def plot_arpes_roi_snapshot(config, momentum_range, snapshot_time_range=(0, 50),
                             momentum_cutoff=0.7, energy_bias=31.5, verbose=False):
    """ARPES image at a single time window with the EDC momentum ROI highlighted."""
    data = load_dataset(config, verbose=verbose)

    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)

    display_sweep_data(
        data,
        axes_choice=[Axes.THETA_X, Axes.ENERGY],
        summing_ranges={Axes.TIME: [list(snapshot_time_range)]},
        plot_axis=ax,
        color_map="terrain",
        percentile_for_colorscale=99,
        angle_to_momentum=True,
        momentum_cutoff=momentum_cutoff,
        display_plot_title=False,
        display_x_label=False,
        display_y_label=False,
    )

    # Shade the EDC integration strip
    k_lo, k_hi = momentum_range
    ymin, ymax = ax.get_ylim()
    from matplotlib.patches import Rectangle
    roi_rect = Rectangle(
        (k_lo, ymin),
        k_hi - k_lo,
        ymax - ymin,
        facecolor="white",
        alpha=0.18,
        edgecolor="white",
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(roi_rect)
    ax.axvline(k_lo, color="white", linewidth=1.2, linestyle="--")
    ax.axvline(k_hi, color="white", linewidth=1.2, linestyle="--")

    # Mark the excitonic target energy
    ax.axhline(2.01, color="red", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.text(momentum_cutoff - 0.02, 2.04, "2.01 eV", color="red",
            fontsize=7, ha="right", va="bottom")

    ax.set_xlabel(r"$k_x\ [\AA^{-1}]$")
    ax.set_ylabel(r"$E - E_F$ [eV]")
    t0, t1 = snapshot_time_range
    ax.set_title(
        f"400 nm Γ-K ARPES  (t = {t0}–{t1} fs)\n"
        rf"Shaded region: EDC ROI ($k_x$ = {k_lo} to {k_hi} Å$^{{-1}}$)"
    )

    output_path = os.path.join(plots_dir, "edc_roi_arpes_snapshot.png")
    fig.savefig(output_path, dpi=150)
    if verbose:
        print(f"Saved to {output_path}")
    plt.show()
    plt.close(fig)


plot_arpes_roi_snapshot(
    dataset_configs[0],   # 400 nm
    momentum_range=(-0.1, 0.1),
    snapshot_time_range=(0, 50),
    verbose=True,
)
