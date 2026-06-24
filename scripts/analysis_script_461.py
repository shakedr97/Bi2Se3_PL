import os
import sys
sys.path.append("/home/shakedr/git/taudaqcode")
from SweepData import SweepData
import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import sys
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
)
from analysis.Bi2Se3_BandStructure import Bi2Se3_BandStructure_Gamma_K
from analysis.analysis_tools import display_pixel_resolved_time_dynamics
from analysis.analysis_tools import Axes
from analysis.analysis_tools import (
    concat_energy_datas,
    remove_bias,
    remove_e_fermi,
    collect_all_measurements_data,
)
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as cm


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
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    # key_words=["461nm", "2318"],
    key_words=["461nm", "Gamma_K", "31.95eV", "37", "70"],
)
data_e1_c1 = data_e1[0]
data_e1_c2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "31.95eV", "37", "160"],
)
# data_e1_c2 = data_e1[1]
data_e1_c2 = data_e1_c2[0]
data_1 = sum_data_sweep(data_e1_c1, data_e1_c2)
print("data 1 energy")
data_e2_c1 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "32.6eV", "37", "70"],
)
data_e2_c1 = data_e2_c1[0]
data_e2_c2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "32.6eV", "37", "160"],
)
data_e2_c2 = data_e2_c2[0]
data_2 = sum_data_sweep(data_e2_c1, data_e2_c2)
data_e3_c1 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "33.25eV", "37", "70"],
)
data_e3_c1 = data_e3_c1[0]
data_e3_c2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "33.25eV", "37", "160"],
)
data_e3_c2 = data_e3_c2[0]
data_3 = sum_data_sweep(data_e3_c1, data_e3_c2)
data_e4_c1 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "33.9eV", "37", "70"],
)
data_e4_c1 = data_e4_c1[0]
data_e4_c2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Nov2025",
    key_words=["461nm", "Gamma_K", "33.9eV", "37", "160"],
)
data_e4_c2 = data_e4_c2[0]
data_4 = sum_data_sweep(data_e4_c1, data_e4_c2)
data = concat_energy_datas([[data_1], [data_2], [data_3], [data_4]])[0]
data = remove_bias(remove_e_fermi([data]))[0][0]
data = remove_outliers_single(
    data, verbose=True, relative_offset_threshold=20, environment_size=5
)
display_sweep_data(
    data,
    axes_choice=[Axes.THETA_X, Axes.ENERGY],
    summing_ranges={Axes.TIME: [[-100, 200]]},
    data_operations=[log_scale],
    data_preparations=[],
    color_map="binary",
    angle_to_momentum=False,
)
print(f"Loaded data in {time.time() - start_time} seconds.")
display_pixel_resolved_time_dynamics(
    data,
    bin_params=[9, 3],
    plot_title="Bi2Se3 Gamma-K probe S pump 461nm",
    plot_excited_levels=True,
    pump_wavelength=461,
    optically_excited_percentile=10,
    band_structure=Bi2Se3_BandStructure_Gamma_K,
)
