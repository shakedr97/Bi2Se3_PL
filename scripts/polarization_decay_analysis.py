import copy
import os
import sys
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

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


def load_datas():
    data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"
    data_items = [
        DataItem(
            data_root_dir,
            ["Dec", "Bi2Se3_G_Gamma_K", "ProbeRotator_37", "31.95_sum"],
            tag="31.95eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Bi2Se3_G_Gamma_K", "ProbeRotator_37", "32.6_sum"],
            tag="32.6eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Bi2Se3_G_Gamma_K", "ProbeRotator_37", "33.25_sum"],
            tag="33.25eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Bi2Se3_G_Gamma_K", "ProbeRotator_37", "33.9_sum"],
            tag="33.9eV",
        ),
    ]
    all_datas_by_tag = {}

    # First, load all data items
    for data_item in data_items:
        keyword_str = data_item.tag
        try:
            datas = read_sweep_data(data_item.base_directory, data_item.search_keywords)
            if len(datas) <= 1:
                raise
        except:
            datas = read_all_sweep_data(
                data_item.base_directory, data_item.search_keywords
            )
            if len(datas) == 0:
                raise

        # Keep only the first 2 polarizations (filter out duplicates)
        datas = datas[:2]
        print(get_all_axes_limits(datas[0]))
        all_datas_by_tag[keyword_str] = datas

    # assert shapes:
    shape_reference = None
    for k in all_datas_by_tag:
        for data_piece in all_datas_by_tag[k]:
            for delay in data_piece.sweep_raw:
                if shape_reference is None:
                    shape_reference = data_piece.sweep_raw[delay].shape
                else:
                    assert data_piece.sweep_raw[delay].shape == shape_reference

    concatenated_datas = concat_energy_datas(
        [data_bunch for data_bunch in all_datas_by_tag.values()]
    )
    shape_reference = None

    # assert shapes:
    for data_group in concatenated_datas:
        for data_piece in data_group:
            for delay in data_piece.sweep_raw:
                if shape_reference is None:
                    shape_reference = data_piece.sweep_raw[delay].shape
                else:
                    assert (
                        data_piece.sweep_raw[delay].shape == shape_reference
                    ), f"Shape mismatch at {data_group}, {data_piece}, {delay}: {data_piece.sweep_raw[delay].shape} vs {shape_reference}"

    res = remove_bias(remove_e_fermi(concatenated_datas))[0]

    for data_piece in res:
        for delay in data_piece.sweep_raw:
            if shape_reference is None:
                shape_reference = data_piece.sweep_raw[delay].shape
            else:
                assert data_piece.sweep_raw[delay].shape == shape_reference

    return res


def calculate_polarization_metric(data_c1: SweepData, data_c2: SweepData):
    pol_metric = copy.deepcopy(data_c1)
    normalization_factor_c1 = np.max(
        [np.mean(data_c1.sweep_raw[k]) for k in data_c1.sweep_raw]
    )
    normalization_factor_c2 = np.max(
        [np.mean(data_c2.sweep_raw[k]) for k in data_c2.sweep_raw]
    )
    for k in data_c1.sweep_raw:
        shape_c1 = data_c1.sweep_raw[k].shape
        shape_c2 = data_c2.sweep_raw[k].shape
        if shape_c1 != shape_c2:
            raise ValueError(f"Shape mismatch for key {k}: {shape_c1} vs {shape_c2}")
        for i in range(shape_c1[0]):
            for j in range(shape_c1[1]):
                pol_metric.sweep_raw[k][i, j] = (
                    data_c1.sweep_raw[k][i, j] - data_c2.sweep_raw[k][i, j]
                ) / (
                    normalization_factor_c1
                    + normalization_factor_c2
                    + data_c1.sweep_raw[k][i, j]
                    + data_c2.sweep_raw[k][i, j]
                )
    return pol_metric


def basic_analysis_template(verbose: bool = False):
    plt.ion()
    datas = load_datas()
    data_c1, data_c2 = datas
    print(get_all_axes_limits(data_c1))

    display_data = True
    if display_data:
        print(get_all_axes_limits(data_c1))
        display_sweep_data(
            data_c1,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[-150, 150]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
        )

    initial_binning = True
    if initial_binning:
        bin_params = {Axes.ENERGY: 4, Axes.THETA_X: 1}
        binned_datas = bin_sweep_datas([[data_c1, data_c2]], bin_params=bin_params)
        data_c1, data_c2 = binned_datas[0]

    display_pixel_resolved_time_dynamics(
        data_c1,
        plot_title="Raw Data C1 Time Dynamics",
        enable_plots=[False, False, False, True],
        band_structure=Bi2Se3_BandStructure_Gamma_K,
        energy_low_limit=0,
        energy_high_limit=3,
        momentum_low_limit=-0.75,
        momentum_high_limit=0.75,
        window_size=[3, 3],
        bin_params=[8, 2],
    )

    polarization_metric = calculate_polarization_metric(data_c1, data_c2)
    display_polarization_metric = False
    if display_polarization_metric:
        display_sweep_data(
            polarization_metric,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[-10, 10]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
            color_map="bwr",
            percentile_for_colorscale=98,
        )

    # calculate the correlation for the polarization metric
    # each cell accumulates the sum of products with neighbors up to
    # n_neighbors steps away along each axis (energy and momentum).
    # cells at the boundary contribute 0 for out-of-bounds neighbors.
    n_neighbors = 2
    polarization_metric_correlation = copy.deepcopy(polarization_metric)
    for k in polarization_metric.sweep_raw:
        pm_2d = polarization_metric.sweep_raw[k]
        shape = pm_2d.shape
        corr_2d = np.zeros_like(pm_2d)
        for d in range(-n_neighbors, n_neighbors + 1):
            for m in range(-n_neighbors, n_neighbors + 1):
                try:
                    if m == 0 and d == 0:
                        continue
                    corr_2d[
                        -min(0, d) : shape[0] - max(0, d),
                        -min(0, m) : shape[1] - max(0, m),
                    ] += (
                        pm_2d[
                            -min(0, d) : shape[0] - max(0, d),
                            -min(0, m) : shape[1] - max(0, m),
                        ]
                        * pm_2d[
                            max(0, d) : shape[0] + min(0, d),
                            max(0, m) : shape[1] + min(0, m),
                        ]
                    )  # d steps forward along energy axis, k steps along angle axis
                except Exception as e:
                    print(f"indices: d:{d}, k:{m}")
                    raise
        polarization_metric_correlation.sweep_raw[k] = np.sqrt(np.abs(corr_2d))
    display_and_bin_polarization_correlation = False
    if display_and_bin_polarization_correlation:
        bin_params = {Axes.ENERGY: 1, Axes.THETA_X: 1}
        binned_datas = bin_sweep_datas([[polarization_metric_correlation]], bin_params)
        polarization_metric_correlation = binned_datas[0][0]
        display_sweep_data(
            polarization_metric_correlation,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[-60, -10]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
            color_map="terrain_r",
            color_scale_min=100,
            color_scale_max=2000,
        )
        display_sweep_data(
            polarization_metric_correlation,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[-10, 10]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
            color_map="terrain_r",
            color_scale_min=100,
            color_scale_max=2000,
        )
        display_sweep_data(
            polarization_metric_correlation,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[10, 60]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
            color_map="terrain_r",
            color_scale_min=100,
            color_scale_max=2000,
        )
        display_sweep_data(
            polarization_metric_correlation,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[60, 80]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
            color_map="terrain_r",
            color_scale_min=100,
            color_scale_max=2000,
        )
        display_sweep_data(
            polarization_metric_correlation,
            axes_choice=[Axes.THETA_X, Axes.ENERGY],
            summing_ranges={Axes.TIME: [[100, 110]]},
            angle_to_momentum=True,
            momentum_cutoff=0.75,
            color_map="terrain_r",
            color_scale_min=100,
            color_scale_max=2000,
        )
    # Time series of correlation in a specific region
    energy_bias = 31.5  # approximate kinetic energy offset [eV]
    momentum_low, momentum_high = -0.05, 0.05  # [Å^-1]
    energy_low, energy_high = 2.27, 2.345  # [eV, E - E_F]
    energy_midpoint = (energy_low + energy_high) / 2
    theta_min = momentum_to_angle(momentum_low, energy_bias + energy_midpoint)
    theta_max = momentum_to_angle(momentum_high, energy_bias + energy_midpoint)

    display_sweep_data(
        polarization_metric_correlation,
        axes_choice=[Axes.TIME],
        summing_ranges={
            Axes.THETA_X: [[theta_min, theta_max]],
            Axes.ENERGY: [[energy_low, energy_high]],
        },
    )

    # Plot tau values from pixel-resolved time dynamics
    display_pixel_resolved_time_dynamics(
        polarization_metric_correlation,
        plot_title="Polarization Correlation Decay",
        enable_plots=[False, True, False, True],
        energy_low_limit=0,
        energy_high_limit=3,
        momentum_low_limit=-0.75,
        momentum_high_limit=0.75,
        window_size=[1, 1],
        bin_params=[4, 4],
    )
    output_path = os.path.join(dir, "polarization_decay_460nm.png")
    plt.savefig(output_path)

    if verbose:
        print(f"Saved figure to {output_path}")

    plt.show()
    input("Press Enter to continue...")


basic_analysis_template(verbose=True)
