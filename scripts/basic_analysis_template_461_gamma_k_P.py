import os
import sys

path = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from SweepData import SweepData
from typing import Callable

os.chdir(os.path.join(path, "..", "plots"))


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
    ):
        self.plot_type = plot_type
        self.keywords = keywords
        self.axis_choices = axis_choices
        self.summing_ranges = summing_ranges
        self.plot_name = plot_name if plot_name else keywords[0]
        self.legend = legend
        self.data_operations = data_operations if data_operations else []
        self.data_preparations = data_preparations if data_preparations else []


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
    datas_keywords = [
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "31.95_sum"],
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "32.6_sum"],
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "33.25_sum"],
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "33.9_sum"],
    ]
    # list to hold all the generated plots
    # each plot can be a 2D array of datasets to be plotted together
    # e.g. plots = [ [[data1, data2], [data3, data4]], [[data5, data6]] ]
    plots = ["override-me"]
    times = [
        # (-1010, -990),
        # (-160, -140),
        # (-110, -90),
        # (-80, -60),
        (-55, -45),
        (-10, 10),
        (45, 55),
        # (60, 80),
        # (130, 160),
        # (240, 260),
        # (690, 710),
        # (1990, 2010),
    ]
    plots = []
    for time_1, time_2 in times:
        plots.append(
            Plot(
                subplots=[
                    [
                        SubPlot(
                            plot_type=SubPlotType.CD,
                            keywords=[
                                # Gamma-K probe P
                                "31.95_sum",
                                "32.6_sum",
                                "33.25_sum",
                                "33.9_sum",
                            ],
                            axis_choices=[Axes.THETA_X, Axes.ENERGY],
                            summing_ranges={
                                Axes.TIME: [[time_1, time_2]],
                            },
                            data_operations=[],
                            data_preparations=[
                                cd_data_prep,
                                concat_energy_datas,
                                remove_bias,
                                remove_e_fermi,
                                pad_energy,
                            ],
                            plot_name="461nm CD",
                        ),
                        SubPlot(
                            plot_type=SubPlotType.SUM,
                            keywords=[
                                # Gamma-K probe P
                                "31.95_sum",
                                "32.6_sum",
                                "33.25_sum",
                                "33.9_sum",
                            ],
                            axis_choices=[Axes.THETA_X, Axes.ENERGY],
                            summing_ranges={
                                Axes.TIME: [[time_1, time_2]],
                            },
                            data_operations=[],
                            data_preparations=[
                                remove_bias,
                                concat_energy_datas_smooth_stitching,
                                sum_data_prep,
                                remove_e_fermi,
                                pad_energy,
                                # subtract_noise,
                            ],
                            plot_name="461nm SUM",
                        ),
                    ],
                ],
                plot_name=f"Gamma-K Probe P Pol Delay {int((time_1 + time_2) / 2)}fs",
            )
        )
    # load all data items by keyword
    all_datas = (
        {}
    )  # dictionary to hold the datasets, dict keys are the last keyword in each keywords list
    for data_keywords in datas_keywords:
        keyword_str = data_keywords[-1]
        if verbose:
            print(f"loading data for keywords: {data_keywords}")
        datas = read_all_sweep_data(data_root_dir, data_keywords)
        if verbose:
            data_ex = datas[0]
            print(get_all_axes_limits(data_ex))
            print(
                f"times: [{', '.join([str(t) for t in sorted(data_ex.sweep_raw.keys())])}]"
            )
        all_datas[keyword_str] = datas
        if verbose:
            print(f"loaded {len(datas)} datasets for keywords: {data_keywords}")
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
                    plot_data = preparation(plot_data)
                plot_datas_row.append(plot_data[0])
            plot_datas.append(plot_datas_row)

        plot_shape = (len(plot_datas), len(plot_datas[0]) if plot_datas else 1)
        fig, axes = plt.subplots(*plot_shape)
        if plot_shape[0] > 1:
            for i in range(plot_shape[0]):
                for j in range(plot_shape[1]):
                    if plot_shape[1] == 1:
                        ax = axes[i]
                    else:
                        ax = axes[i, j]
                    plot = plot_plan.subplots[i][j]
                    plot: SubPlot
                    if len(plot.axis_choices) > 1:
                        datas = plot_datas[i][j]
                    else:
                        datas = plot_datas[i][j]
                    color_map = "bwr" if plot.plot_type == SubPlotType.CD else "viridis"
                    for preparation in plot.data_preparations:
                        datas = preparation(datas)
                    for data in datas:
                        display_sweep_data(
                            data,
                            axes_choice=plot.axis_choices,
                            summing_ranges=plot.summing_ranges,
                            plot_axis=ax,
                            color_map=color_map,
                            percentile_for_colorscale=(
                                50 if plot.plot_type == SubPlotType.CD else None
                            ),
                            data_operations=plot.data_operations,
                        )
                    ax.set_title(f"{plot.plot_name}")
        else:
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
                        percentile_for_colorscale=(99.2),
                        data_operations=plot.data_operations,
                        angle_to_momentum=True,
                        momentum_cutoff=0.75,
                    )
                if plot.legend is not None:
                    ax.legend(plot.legend)
                title = plot.plot_name
                for operation in plot.data_operations:
                    title += f" + {operation.__name__}"
                file_name += f"_{title}"
                ax.set_title(title)
        fig.suptitle(f"{plot_plan.plot_name}")
        fig.tight_layout()
        fig.savefig(f"{file_name}.png".replace(" ", "_"))
    plt.show()
    input("Press Enter to continue...")


basic_analysis_template(verbose=True)
