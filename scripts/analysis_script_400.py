import os
import sys

# os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from SweepData import SweepData
import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit
from scipy.special import erf
from analysis.cd_analysis import bin_data
import itertools
import matplotlib.cm as cm
import time
from analysis.analysis_tools import read_sweep_data
from analysis.analysis_tools import (
    display_sweep_data,
    sum_data_sweep,
    cd_data_sweep,
    subtract_noise_single,
    remove_outliers_single,
    DataItem,
)
from analysis.Bi2Se3_BandStructure import (
    Bi2Se3_BandStructure_Gamma_K,
    Bi2Se3_BandStructure_Gamma_M,
)
from analysis.analysis_tools import display_pixel_resolved_time_dynamics
from analysis.analysis_tools import Axes
from analysis.analysis_tools import (
    concat_energy_datas,
    concat_energy_datas_smooth_stitching,
    remove_bias,
    remove_e_fermi,
    collect_all_measurements_data,
)
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as cm

path = os.path.dirname(__file__)
os.chdir(os.path.join(path, "..", "plots"))


def cutoff_max_times(max_times: ArrayLike, max_values: ArrayLike, threshold: 0.3):
    for i in range(max_times.shape[0]):
        # values_in_i = sorted(max_values[i])
        max_value_in_i = max(max_values[i])
        threshold_value = max_value_in_i * threshold
        for j in range(max_times.shape[1]):
            if max_values[i][j] < threshold_value:
                max_times[i][j] = -500


def log_scale(data: ArrayLike):
    return np.log(1 + data)


def create_times_plot_data(
    times: np.ndarray,
    x_axis,
    y_axis,
    bin_params=[15, 5],
    low_percentile=10,
    high_percentile=90,
    color_scales_count=10,
    fwhms=None,
):
    """
    generate scatter plot data
    output - x_array, y_array, z_array, color_scale
    x_array - list of x positions for the scatter plot
    y_array - list of y positions for the scatter plot
    z_array - list of z values for the scatter plot
    color_scale - list of color scale values for the scatter plot
    """
    x = []
    y = []
    c = []
    scatter_fwhms = []
    binned_y_axis = bin_data(y_axis, bin_params[1])
    binned_x_axis = bin_data(x_axis, bin_params[0])
    shape = times.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if -200 < times[i][j] < 100:
                x.append(binned_x_axis[j])
                y.append(binned_y_axis[i])
                c.append(times[i][j])
                if fwhms is not None:
                    scatter_fwhms.append(fwhms[i][j])
    c = np.array(c)
    sorted_c = np.sort(c)
    low_value = sorted_c[int(len(sorted_c) * low_percentile / 100)]
    high_value = sorted_c[int(len(sorted_c) * high_percentile / 100)]
    color_scale = np.linspace(low_value, high_value, color_scales_count)
    if fwhms is not None:
        return x, y, c, color_scale, scatter_fwhms
    else:
        return x, y, c, color_scale


def find_earliest_time_threshold(times, plot=False, max_edge_threshold=0.1):
    for n_bins in range(5, 1000, 5):
        plt.close()
        plt.figure()
        N, bins, _ = plt.hist(times, bins=n_bins)
        highest = int(n_bins / 2)
        second_highest = int(n_bins / 2)
        for i in range(len(N)):
            if N[i] > N[highest]:
                highest = i
            elif N[highest] > N[second_highest] > N[i]:
                second_highest = i
        if abs(highest - second_highest) > 5:
            earliest = min(highest, second_highest)
            next = max(highest, second_highest)
            earliest_edge = earliest
            for i in range(n_bins - earliest):
                earliest_edge = earliest + i
                if N[earliest_edge] < max_edge_threshold * N[earliest]:
                    break
            if earliest_edge < (earliest + next) / 2:
                if plot:
                    plt.show()
                return bins[earliest_edge]
    raise Exception()


print("Loading data...")
start_time = time.time()
data_e1 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    # key_words=["461nm", "2318"],
    key_words=["843"],
)
data_e1_c1 = data_e1[0]
data_e1_c2 = data_e1[1]
data_1 = sum_data_sweep(data_e1_c1, data_e1_c2)
print("data 1 energy")
data_e2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["844"],
)
data_e2_c1 = data_e2[0]
data_e2_c2 = data_e2[1]
data_2 = sum_data_sweep(data_e2_c1, data_e2_c2)
data_e3 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["845"],
)
data_e3_c1 = data_e3[0]
data_e3_c2 = data_e3[1]
data_3 = sum_data_sweep(data_e3_c1, data_e3_c2)
data_e4 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["846"],
)
data_e4_c1 = data_e4[0]
data_e4_c2 = data_e4[1]
data_4 = sum_data_sweep(data_e4_c1, data_e4_c2)
data = concat_energy_datas_smooth_stitching([[data_1], [data_2], [data_3], [data_4]])[0]
data = remove_bias(remove_e_fermi([data]))[0][0]
# display_sweep_data(
#     data,
#     axes_choice=[Axes.THETA_X, Axes.ENERGY],
#     summing_ranges={Axes.TIME: [[-50, 250]]},
#     color_map="terrain",
#     percentile_for_colorscale=95,
#     angle_to_momentum=True,
#     momentum_cutoff=0.75,
#     band_structure=Bi2Se3_BandStructure_Gamma_K,
#     pump_wavelength_for_offset=400,
# )
# sys.exit()
# data = remove_outliers_single(
#     data, verbose=True, relative_offset_threshold=20, environment_size=5
# )
print(f"Loaded data in {time.time() - start_time} seconds.")
if True:
    display_pixel_resolved_time_dynamics(
        data,
        # bin_params=[9, 5],
        bin_params=[3, 1.5],
        window_size=[3, 3],
        plot_title="Bi2Se3 Gamma-K probe S pump 400nm",
        plot_excited_levels=True,
        pump_wavelength=400,
        optically_excited_percentile=10,
        band_structure=Bi2Se3_BandStructure_Gamma_K,
        enable_plots=[True, True, True, False],
        energy_low_limit=0.07,
        energy_high_limit=2.78,
        momentum_low_limit=-0.75,
        momentum_high_limit=0.75,
    )


print("Loading data...")
start_time = time.time()
data_e1 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    # key_words=["461nm", "2318"],
    key_words=["837"],
)
data_e1_c1 = data_e1[0]
data_e1_c2 = data_e1[1]
data_1 = sum_data_sweep(data_e1_c1, data_e1_c2)
print("data 1 energy")
data_e2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["840"],
)
data_e2_c1 = data_e2[0]
data_e2_c2 = data_e2[1]
data_2 = sum_data_sweep(data_e2_c1, data_e2_c2)
data_e3 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["841"],
)
data_e3_c1 = data_e3[0]
data_e3_c2 = data_e3[1]
data_3 = sum_data_sweep(data_e3_c1, data_e3_c2)
data_e4 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["842"],
)
data_e4_c1 = data_e4[0]
data_e4_c2 = data_e4[1]
data_4 = sum_data_sweep(data_e4_c1, data_e4_c2)
data = concat_energy_datas_smooth_stitching([[data_1], [data_2], [data_3], [data_4]])[0]
data = remove_bias(remove_e_fermi([data]))[0][0]
if False:
    data = remove_outliers_single(
        data, verbose=True, relative_offset_threshold=20, environment_size=5
    )
print(f"Loaded data in {time.time() - start_time} seconds.")
if True:
    display_pixel_resolved_time_dynamics(
        data,
        bin_params=[3, 1.5],
        window_size=[3, 3],
        plot_title="Bi2Se3 Gamma-K probe P pump 400nm",
        plot_excited_levels=True,
        pump_wavelength=400,
        optically_excited_percentile=10,
        band_structure=Bi2Se3_BandStructure_Gamma_K,
        enable_plots=[True, True, True, False],
        energy_low_limit=0.07,
        energy_high_limit=2.78,
        momentum_low_limit=-0.75,
        momentum_high_limit=0.75,
    )

gamma_m_datas = {}
data_root_dir = "/home/shakedr/Downloads/Bi2Se3"
for data_item in [
    DataItem(data_root_dir, ["Sep", "848"], tag="400nm_gamma_m_probe_p_31.95eV"),
    DataItem(data_root_dir, ["Sep", "855"], tag="400nm_gamma_m_probe_p_33.9eV"),
    DataItem(data_root_dir, ["Sep", "856"], tag="400nm_gamma_m_probe_p_33.25eV"),
    DataItem(data_root_dir, ["Sep", "857"], tag="400nm_gamma_m_probe_p_32.6eV"),
    DataItem(data_root_dir, ["Sep", "851"], tag="400nm_gamma_m_probe_s_31.95eV"),
    DataItem(data_root_dir, ["Sep", "852"], tag="400nm_gamma_m_probe_s_32.6eV"),
    DataItem(data_root_dir, ["Sep", "853"], tag="400nm_gamma_m_probe_s_33.25eV"),
    DataItem(data_root_dir, ["Sep", "854"], tag="400nm_gamma_m_probe_s_33.9eV"),
]:
    gamma_m_datas[data_item.tag] = read_sweep_data(
        data_item.base_directory, data_item.search_keywords
    )

data_e1 = gamma_m_datas["400nm_gamma_m_probe_p_31.95eV"]
data_e1_c1 = data_e1[0]
data_e1_c2 = data_e1[1]
data_1 = sum_data_sweep(data_e1_c1, data_e1_c2)

data_e2 = gamma_m_datas["400nm_gamma_m_probe_p_32.6eV"]
data_e2_c1 = data_e2[0]
data_e2_c2 = data_e2[1]
data_2 = sum_data_sweep(data_e2_c1, data_e2_c2)

data_e3 = gamma_m_datas["400nm_gamma_m_probe_p_33.25eV"]
data_e3_c1 = data_e3[0]
data_e3_c2 = data_e3[1]
data_3 = sum_data_sweep(data_e3_c1, data_e3_c2)

data_e4 = gamma_m_datas["400nm_gamma_m_probe_p_33.9eV"]
data_e4_c1 = data_e4[0]
data_e4_c2 = data_e4[1]
data_4 = sum_data_sweep(data_e4_c1, data_e4_c2)

data = concat_energy_datas_smooth_stitching([[data_1], [data_2], [data_3], [data_4]])[0]
data = remove_bias(remove_e_fermi([data]))[0][0]
if False:
    display_pixel_resolved_time_dynamics(
        data,
        bin_params=[3, 1.5],
        window_size=[3, 3],
        plot_title="Bi2Se3 Gamma-M probe P pump 400nm",
        plot_excited_levels=True,
        pump_wavelength=400,
        optically_excited_percentile=10,
        band_structure=Bi2Se3_BandStructure_Gamma_M,
        enable_plots=[True, True, True, False],
        energy_low_limit=0.07,
        energy_high_limit=2.78,
        momentum_low_limit=-0.75,
        momentum_high_limit=0.75,
    )


data_e1 = gamma_m_datas["400nm_gamma_m_probe_s_31.95eV"]
data_e1_c1 = data_e1[0]
data_e1_c2 = data_e1[1]
data_1 = sum_data_sweep(data_e1_c1, data_e1_c2)

data_e2 = gamma_m_datas["400nm_gamma_m_probe_s_32.6eV"]
data_e2_c1 = data_e2[0]
data_e2_c2 = data_e2[1]
data_2 = sum_data_sweep(data_e2_c1, data_e2_c2)

data_e3 = gamma_m_datas["400nm_gamma_m_probe_s_33.25eV"]
data_e3_c1 = data_e3[0]
data_e3_c2 = data_e3[1]
data_3 = sum_data_sweep(data_e3_c1, data_e3_c2)

data_e4 = gamma_m_datas["400nm_gamma_m_probe_s_33.9eV"]
data_e4_c1 = data_e4[0]
data_e4_c2 = data_e4[1]
data_4 = sum_data_sweep(data_e4_c1, data_e4_c2)

data = concat_energy_datas_smooth_stitching([[data_1], [data_2], [data_3], [data_4]])[0]
data = remove_bias(remove_e_fermi([data]))[0][0]
if False:
    display_pixel_resolved_time_dynamics(
        data,
        bin_params=[3, 1.5],
        window_size=[3, 3],
        plot_title="Bi2Se3 Gamma-M probe S pump 400nm",
        plot_excited_levels=True,
        pump_wavelength=400,
        optically_excited_percentile=10,
        band_structure=Bi2Se3_BandStructure_Gamma_M,
        enable_plots=[True, True, True, False],
        energy_low_limit=0.07,
        energy_high_limit=2.78,
        momentum_low_limit=-0.75,
        momentum_high_limit=0.75,
    )
