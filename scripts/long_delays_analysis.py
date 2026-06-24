import os
import sys
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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


class SubPlotType:
    SUM = "sum"
    CD = "cd"


def normalize_max(data: ArrayLike):
    max_val = np.max(data)
    if max_val != 0:
        return data / max_val
    return data


def log_scale(data: ArrayLike):
    return np.log(1 + data)


class SubPlot:
    def __init__(
        self,
        plot_type: SubPlotType,
        keywords: List[str],
        axis_choices: list,
        summing_ranges: dict,
        plot_name: str = None,
        legend: List[str] = None,
        data_operations: List[Callable] = None,
        data_preparations: List[Callable] = None,
        color_scale_percentile: int = 95,
        color_scale_min: int = None,
        color_scale_max: int = None,
    ):
        self.plot_type = plot_type
        self.keywords = keywords
        self.axis_choices = axis_choices
        self.summing_ranges = summing_ranges
        self.plot_name = plot_name if plot_name else keywords[0]
        self.legend = legend
        self.data_operations = data_operations if data_operations else []
        self.data_preparations = data_preparations if data_preparations else []
        self.color_scale_percentile = color_scale_percentile
        self.color_scale_min = color_scale_min
        self.color_scale_max = color_scale_max


class Plot:
    def __init__(self, subplots: list, plot_name: str = None):
        self.subplots = subplots
        self.plot_name = plot_name if plot_name else "plot"


def basic_analysis_template(verbose: bool = False):
    # directory which contains all the data for the relevant analysis
    plt.ion()
    data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"
    # list of lists of keywords to search for in the filenames
    # each list of keywords will be treated as a separate dataset
    # e.g. datas_keywords = [["pump_on", "delay_0"], ["pump_off", "delay_0"]]
    data_items = [
        DataItem(
            data_root_dir,
            ["Excitons", "452nm", "Gamma-M", "long_delays_1"],
            tag="long_delays",
        ),
    ]

    # list to hold all the generated plots
    # each plot can be a 2D array of datasets to be plotted together
    # e.g. plots = [ [[data1, data2], [data3, data4]], [[data5, data6]] ]
    plots = []
    times = [
        # (-1010, -990),
        # (-160, -140),
        # (-110, -90),
        # (-60, -40),
        # (-10, 10),
        (49, 51),
        (199, 201),
        # (395, 405),
        # (690, 710),
        # (1490, 1510),
        (1990, 2010),
        # (2990, 3010),
        # (4950, 5050),
    ]
    # gamma-K probe P
    subplots = [[]]
    for time_1, time_2 in times:
        subplots[0].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=["long_delays"],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas_smooth_stitching,
                    bin_sweep_datas,
                    # subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                ],
                plot_name=f"{50 * int((time_1 + time_2) / 100)}fs",
                color_scale_min=5,
                color_scale_max=1000,
            )
        )
    plots.append(
        Plot(
            subplots=subplots,
            plot_name="Excitation Dynamics Gamma-M probe S 452nm",
        )
    )

    # load all data items by keyword
    all_datas = (
        {}
    )  # dictionary to hold the datasets, dict keys are the last keyword in each keywords list
    for data_item in data_items:
        keyword_str = data_item.tag
        if verbose:
            print(f"loading data for keywords: {data_item.search_keywords}")
        try:
            datas = read_sweep_data(data_item.base_directory, data_item.search_keywords)

            if len(datas) <= 1:
                raise
        except:
            datas = read_all_sweep_data(
                data_item.base_directory, data_item.search_keywords
            )
            print(datas)
            if len(datas) == 0:
                raise
        if verbose:
            data_ex = datas[0]
            print(get_all_axes_limits(data_ex))
            print(
                f"times: [{', '.join([str(t) for t in sorted(data_ex.sweep_raw.keys())])}]"
            )
        all_datas[keyword_str] = datas
        if verbose:
            print(
                f"loaded {len(datas)} datasets for keywords: {data_item.search_keywords}"
            )
    if False:
        for plot_plan in plots:
            # arrange data for plotting
            plot_plan: Plot
            file_name = f"{plot_plan.plot_name}__-_"
            plot_datas = []
            for row in plot_plan.subplots:
                plot_datas_row = []
                for plot_req in row:
                    plot_req: SubPlot
                    plot_data = []
                    for keyword in plot_req.keywords:
                        data = None
                        basic_data = all_datas[keyword]
                        plot_data.append(copy.deepcopy(basic_data))
                    for preparation in plot_req.data_preparations:
                        if verbose:
                            print(f"{preparation} - {plot_data}")
                        plot_data = preparation(plot_data)
                    plot_datas_row.append(plot_data[0])
                plot_datas.append(plot_datas_row)

            plot_shape = (len(plot_datas), len(plot_datas[0]) if plot_datas else 1)
            fig, axes = plt.subplots(*plot_shape, sharex=True, sharey=True)
            fig.subplots_adjust(hspace=0)
            counter = 0
            if plot_shape[0] > 1:
                for i in range(plot_shape[0]):
                    axes[i][0].set_ylabel(r"$E-E_{F}\left[eV\right]$", fontsize=8)
                    for j in range(plot_shape[1]):
                        marker = chr(ord("a") + counter)
                        counter += 1
                        if plot_shape[1] == 1:
                            ax = axes[i]
                            # ax.sharey(original_ax)
                        else:
                            ax = axes[i, j]

                        plot = plot_plan.subplots[i][j]
                        plot: SubPlot
                        if len(plot.axis_choices) > 1:
                            datas = plot_datas[i][j]
                        else:
                            datas = plot_datas[i][j]
                        color_map = (
                            "bwr" if plot.plot_type == SubPlotType.CD else "terrain"
                        )
                        for data in datas:
                            display_sweep_data(
                                data,
                                axes_choice=plot.axis_choices,
                                summing_ranges=plot.summing_ranges,
                                plot_axis=ax,
                                color_map=color_map,
                                # percentile_for_colorscale=(90),
                                color_scale_min=plot.color_scale_min,
                                color_scale_max=plot.color_scale_max,
                                data_operations=plot.data_operations,
                                angle_to_momentum=True,
                                momentum_cutoff=0.75,
                            )
                        ax.text(-0.65, 2.65, marker)
                        title = plot.plot_name
                        # ax.set_title(f"{plot.plot_name}", fontsize=5)

            else:
                color_scale_percentiles = [97, 97.5, 97.5, 98.1, 98.1, 98.1, 98.1, 98.1]
                for j in range(plot_shape[1]):
                    ax = axes[j] if plot_shape[1] > 1 else axes
                    plot = plot_plan.subplots[0][j]
                    plot: SubPlot
                    if len(plot.axis_choices) > 1:
                        datas = plot_datas[0][j]
                    else:
                        datas = plot_datas[0][j]

                    color_map = "bwr" if plot.plot_type == SubPlotType.CD else "terrain"
                    for data in datas:
                        display_sweep_data(
                            data,
                            axes_choice=plot.axis_choices,
                            summing_ranges=plot.summing_ranges,
                            plot_axis=ax,
                            color_map=color_map,
                            percentile_for_colorscale=color_scale_percentiles[j],
                            data_operations=plot.data_operations,
                            angle_to_momentum=True,
                            momentum_cutoff=0.75,
                        )
                        ax: plt.Axes
                        ax.set_xlabel(r"$k_{x}\left[\AA^{-1}\right]$", fontsize=8)
                        if j == 0:
                            ax.set_ylabel(r"$E-E_{F}\left[eV\right]$", fontsize=8)
                        else:
                            ax.set_ylabel("")
                        ax.text(-0.65, 2.85, plot.plot_name)
                        cax = ax.inset_axes([0.1, 0.9, 0.8, 0.03])
                        fig.colorbar(ax.images[0], cax=cax, orientation="horizontal")
                        ax.set_title("")
                    if plot.legend is not None:
                        ax.legend(plot.legend)
                    title = plot.plot_name
                    for operation in plot.data_operations:
                        title += f" + {operation.__name__}"
                    file_name += f"_{title}"
            # fig.suptitle(f"{plot_plan.plot_name}")
            fig.subplots_adjust(hspace=0)
            # fig.tight_layout()
            output_path = os.path.join(dir, f"{file_name}.png".replace(" ", "_"))
            fig.savefig(output_path)
        plt.show()
        input("Press Enter to continue...")

    # bin the data and create several plots for the decay time in each bin, each plot should be the decay time of the data past a certain delay.

    # figure out binning params needed to aim for a certain data shape
    data_shape = all_datas["long_delays"][0].sweep_raw[0].shape
    target_shape = (24, 53)
    # bin_params_0 = data_shape[0] // target_shape[0]
    # bin_params_1 = data_shape[1] // target_shape[1]
    # datas = bin_sweep_datas(
    #     [all_datas["long_delays"]],
    #     {Axes.ENERGY: bin_params_1, Axes.THETA_X: bin_params_0},
    # )
    datas = all_datas["long_delays"]
    datas = bin_sweep_datas([datas], bin_params={Axes.ENERGY: 5, Axes.THETA_X: 2})
    data = datas[0][0]
    data = subtract_noise_single(data)

    data.energy_offset = data.metadata["BiasVoltage"] - 1.5
    decay_rate_starting_points = [700]

    fig, axes = plt.subplots(1, len(decay_rate_starting_points))
    fig_name = "decay_times"
    energy_limit_low = 0
    energy_limit_high = 1
    fig_name += f"_{energy_limit_low}eV_{energy_limit_high}eV"
    if len(decay_rate_starting_points) == 1:
        axes = [axes]
    for ax, starting_point in zip(axes, decay_rate_starting_points):
        fig_name += f"_{starting_point}fs"
        cax = ax.inset_axes([0, 1, 1, 0.03])
        display_pixel_resolved_decay_times(
            data,
            starting_point,
            ax=ax,
            color_min=300,
            color_max=3000,
            energy_limit_low=energy_limit_low,
            energy_limit_high=energy_limit_high,
            momentum_limit_low=-0.75,
            momentum_limit_high=0.75,
            target_shape=(40, 70),
            band_structure=Bi2Se3_BandStructure_Gamma_M,
        )
        ax.set_ylabel(r"$E-E_{F}\left[eV\right]$", fontsize=8)
        ax.set_xlabel(r"$k_{x}\left[\AA^{-1}\right]$", fontsize=8)
        cb = fig.colorbar(ax.collections[0], cax=cax, orientation="horizontal")
        cb.ax.xaxis.set_ticks_position("top")
        cb.locator = MaxNLocator(nbins=6)
        cb.update_ticks()
        cb.set_label("Decay Time [fs]")
        cb.ax.xaxis.set_label_position("top")
    plt.tight_layout()
    fig.savefig(os.path.join(dir, f"{fig_name}.png").replace(" ", "_"))
    plt.show()
    input("Press Enter to continue...")


basic_analysis_template(verbose=True)
