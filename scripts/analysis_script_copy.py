import os
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
)
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
# data_dir = os.path.join(
#     "/",
#     *"/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Excitons/452nm/Gamma_K/energy_sweep_21.5-24eV_423nm_0_long_delays_2/compressed_sweeps/pickle_files".split(
#         "/"
#     ),
# )
# data: SweepData = None
# for file in os.listdir(data_dir):
#     new_data: SweepData = SweepData.from_pickle(os.path.join(data_dir, file))
#     if data is None:
#         data = copy.deepcopy(new_data)
#     else:
#         for k in data.sweep_raw:
#             data.sweep_raw[k] += new_data.sweep_raw[k]
#             print(f"time {k}: {sum(sum(data.sweep_raw[k]))}")

data_e1 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["848"],
)
data_e1_c1 = data_e1[0]

data_e1_c2 = data_e1[1]
data_1 = sum_data_sweep(data_e1_c1, data_e1_c2)

data_e2 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["857"],
)
data_e2_c1 = data_e2[0]
data_e2_c2 = data_e2[1]
data_2 = sum_data_sweep(data_e2_c1, data_e2_c2)
data_e3 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["856"],
)
data_e3_c1 = data_e3[0]
data_e3_c2 = data_e3[1]
data_3 = sum_data_sweep(data_e3_c1, data_e3_c2)
data_e4 = read_sweep_data(
    "/home/shakedr/Downloads/Bi2Se3/Bi2Se3_Sep2025",
    key_words=["855"],
)
data_e4_c1 = data_e4[0]
data_e4_c2 = data_e4[1]
data_4 = sum_data_sweep(data_e4_c1, data_e4_c2)
data = concat_energy_datas([data_1, data_2, data_3, data_4])[0]
data = remove_bias(remove_e_fermi([data]))[0]
display_sweep_data(
    data,
    axes_choice=[Axes.ENERGY, Axes.THETA_X],
    summing_ranges={Axes.TIME: [[-100, 200]]},
    data_operations=[log_scale],
    data_preparations=[],
    color_map="binary",
)
print(f"Loaded data in {time.time() - start_time} seconds.")
display_pixel_resolved_time_dynamics(
    data,
    bin_params=[9, 3],
    plot_title="Bi2Se3 Gamma-M probe P pump 400nm",
    plot_excited_levels=True,
    pump_wavelength=400,
    optically_excited_percentile=15,
)

# concatenated CD
data_c1 = concat_energy_datas([data_e1_c1, data_e2_c1, data_e3_c1, data_e4_c1])[0]
data_c2 = concat_energy_datas([data_e1_c2, data_e2_c2, data_e3_c2, data_e4_c2])[0]
cd_data = cd_data_sweep(data_c1, data_c2, bin_param_energy=2, bin_param_theta_x=2)
plt.ion()
print(data_c1.sweep_raw.keys())
display_sweep_data(
    cd_data,
    axes_choice=[Axes.THETA_X, Axes.ENERGY],
    summing_ranges={Axes.TIME: [[-10, 10]]},
    color_map="bwr",
    percentile_for_colorscale=30,
)
display_sweep_data(
    cd_data,
    axes_choice=[Axes.THETA_X, Axes.ENERGY],
    summing_ranges={Axes.TIME: [[45, 55]]},
    color_map="bwr",
    percentile_for_colorscale=30,
)
display_sweep_data(
    cd_data,
    axes_choice=[Axes.THETA_X, Axes.ENERGY],
    summing_ranges={Axes.TIME: [[-55, -45]]},
    color_map="bwr",
    percentile_for_colorscale=20,
)
display_sweep_data(
    cd_data,
    axes_choice=[Axes.THETA_X, Axes.ENERGY],
    summing_ranges={Axes.TIME: [[-105, -95]]},
    color_map="bwr",
    percentile_for_colorscale=20,
)
display_sweep_data(
    cd_data,
    axes_choice=[Axes.THETA_X, Axes.ENERGY],
    summing_ranges={Axes.TIME: [[-155, -145]]},
    color_map="bwr",
    percentile_for_colorscale=20,
)
input("press enter to continue...")
# print(f"Creating rise time data matrix...")
# start_time = time.time()
# rise_time_bin_params = [6, 2]
# max_times, max_values, fwhms = pixel_resolved_time_dynamics(
#     data, bin_params=rise_time_bin_params
# )
# max_times = max_times.transpose(1, 0)
# max_values = max_values.transpose(1, 0)
# fwhms = fwhms.transpose(1, 0)
# print(f"Created rise time matrix in {time.time() - start_time} seconds.")

# x_axis = np.linspace(
#     data.x_axis_minimum,
#     data.x_axis_minimum + data.x_axis_delta * data.x_axis_count,
#     data.x_axis_count,
# )
# y_axis = np.linspace(
#     data.y_axis_minimum,
#     data.y_axis_minimum + data.y_axis_delta * data.y_axis_count,
#     data.y_axis_count,
# )

# plt.figure(1)

# print(f"Creating rise time plot data...")
# start_time = time.time()
# x, y, c, color_scale, scatter_fwhms = create_times_plot_data(
#     max_times, x_axis, y_axis, bin_params=rise_time_bin_params, fwhms=fwhms
# )
# print(f"Rise time plot data created in {time.time() - start_time} seconds.")

# print(f"Creating rise time histogram...")
# start_time = time.time()
# time_threshold = find_earliest_time_threshold(c, plot=False, max_edge_threshold=0.1)
# print(f"Rise time histogram was created in {time.time() - start_time} seconds.")
# cmap = cm.get_cmap("viridis", len(color_scale))
# norm = BoundaryNorm(color_scale, ncolors=cmap.N, clip=True)
# fig, axes = plt.subplots(1, 3, figsize=(25, 6), constrained_layout=True)
# ax = axes[0]
# sc = ax.scatter(x=y, y=x, c=c, cmap=cmap, norm=norm)
# cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("t0 (fs)")
# ax.set_title("t0s from gauss-exp fit")

# ax = axes[2]
# color_scame_min = np.percentile(scatter_fwhms, 10)
# color_scale_max = np.percentile(scatter_fwhms, 90)
# color_scale = np.linspace(color_scame_min, color_scale_max, 10)
# cmap = cm.get_cmap("viridis", len(color_scale))
# norm = BoundaryNorm(color_scale, ncolors=cmap.N, clip=True)
# sc = ax.scatter(
#     x=y,
#     y=x,
#     c=scatter_fwhms,
#     cmap=cmap,
#     norm=norm,
# )
# cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
# cbar.set_label("FWHM (fs)")
# ax.set_title("FWHM from gauss-exp fit")


# ax = axes[1]
# display_sweep_data(
#     data,
#     plot_axis=ax,
#     axes_choice=[Axes.THETA_X, Axes.ENERGY],
#     summing_ranges={Axes.TIME: [[-100, 50]]},
# )

# print("Plotting time series...")
# start_time = time.time()
# # plot a few time series together
# plt.figure(3)
# bias = 30
# # angle = 0, energy = 23.6eV_423nm_0
# delays = list(data.sweep_raw.keys())
# time_series_1 = extract_time_series_at_angle_energy(data, angle=0, energy=3.6 + bias)
# time_series_1 = np.array(time_series_1) / np.max(time_series_1)
# # angle = 0, energy = 23.15eV_423nm_0
# time_series_2 = extract_time_series_at_angle_energy(data, angle=0, energy=3.15 + bias)
# time_series_2 = np.array(time_series_2) / np.max(time_series_2)
# # angle = 0, energy = 21.782eV_423nm_0
# time_series_3 = extract_time_series_at_angle_energy(data, angle=0, energy=1.782 + bias)
# time_series_3 = np.array(time_series_3) / np.max(time_series_3)

# # angle = 0, energy = 23.97eV_423nm_0
# time_series_4 = extract_time_series_at_angle_energy(data, angle=0, energy=3.92 + bias)
# time_series_4 = np.array(time_series_4) / np.max(time_series_4)

# # angle = 15, energy = 23.97eV_423nm_0
# time_series_5 = extract_time_series_at_angle_energy(data, angle=15, energy=3.92 + bias)
# time_series_5 = np.array(time_series_5) / np.max(time_series_5)


# # plot them, no lines, just markers
# plt.plot(
#     delays, time_series_1, label="23.6 eV_423nm_0", color="red", marker="o", linestyle="None"
# )
# plt.plot(
#     delays, time_series_2, label="23.15 eV_423nm_0", color="green", marker="o", linestyle="None"
# )
# plt.plot(
#     delays, time_series_3, label="21.782 eV_423nm_0", color="blue", marker="o", linestyle="None"
# )
# plt.plot(
#     delays,
#     time_series_4,
#     label="23.97 eV_423nm_0",
#     color="purple",
#     marker="o",
#     linestyle="None",
# )
# plt.plot(
#     delays,
#     time_series_5,
#     label="23.97 eV_423nm_0, 15 deg",
#     color="orange",
#     marker="o",
#     linestyle="None",
# )
# print(f"Time series plotted in {time.time() - start_time} seconds.")
# plt.show()
