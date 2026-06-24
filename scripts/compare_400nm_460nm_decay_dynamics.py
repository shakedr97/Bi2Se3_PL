import copy
import os
import sys
from typing import Callable

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
        if pump_wavelength is not None and band_structure is not None:
            under_fermi_offset = 1240 / pump_wavelength
            above_fermi_offest = 1240 / pump_wavelength - probe_energy
            offsets = {
                # under_fermi_offset: band_structure.bands,
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


def basic_analysis_template(verbose: bool = False):
    # directory which contains all the data for the relevant analysis
    plt.ion()
    data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"
    # list of lists of keywords to search for in the filenames
    # each list of keywords will be treated as a separate dataset
    # e.g. datas_keywords = [["pump_on", "delay_0"], ["pump_off", "delay_0"]]
    data_items = [
        DataItem(data_root_dir, ["Sep", "837"], tag="400nm_gamma_k_probe_p_31.95eV"),
        DataItem(data_root_dir, ["Sep", "840"], tag="400nm_gamma_k_probe_p_32.6eV"),
        DataItem(data_root_dir, ["Sep", "841"], tag="400nm_gamma_k_probe_p_33.25eV"),
        DataItem(data_root_dir, ["Sep", "842"], tag="400nm_gamma_k_probe_p_33.9eV"),
        DataItem(data_root_dir, ["Sep", "843"], tag="400nm_gamma_k_probe_s_33.9eV"),
        DataItem(data_root_dir, ["Sep", "844"], tag="400nm_gamma_k_probe_s_33.25eV"),
        DataItem(data_root_dir, ["Sep", "845"], tag="400nm_gamma_k_probe_s_32.6eV"),
        DataItem(data_root_dir, ["Sep", "846"], tag="400nm_gamma_k_probe_s_31.95eV"),
        DataItem(data_root_dir, ["Sep", "848"], tag="400nm_gamma_m_probe_p_31.95eV"),
        DataItem(data_root_dir, ["Sep", "855"], tag="400nm_gamma_m_probe_p_33.9eV"),
        DataItem(data_root_dir, ["Sep", "856"], tag="400nm_gamma_m_probe_p_33.25eV"),
        DataItem(data_root_dir, ["Sep", "857"], tag="400nm_gamma_m_probe_p_32.6eV"),
        DataItem(data_root_dir, ["Sep", "851"], tag="400nm_gamma_m_probe_s_31.95eV"),
        DataItem(data_root_dir, ["Sep", "852"], tag="400nm_gamma_m_probe_s_32.6eV"),
        DataItem(data_root_dir, ["Sep", "853"], tag="400nm_gamma_m_probe_s_33.25eV"),
        DataItem(data_root_dir, ["Sep", "854"], tag="400nm_gamma_m_probe_s_33.9eV"),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.25_sum"],
            tag="460nm_gamma_k_probe_p_33.25eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_82", "31.95_sum"],
            tag="460nm_gamma_k_probe_p_31.95eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.9_sum"],
            tag="460nm_gamma_k_probe_p_33.9eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_82", "32.6_sum"],
            tag="460nm_gamma_k_probe_p_32.6eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_37", "33.25_sum"],
            tag="460nm_gamma_k_probe_s_33.25eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_37", "33.9_sum"],
            tag="460nm_gamma_k_probe_s_33.9eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_37", "32.6_sum"],
            tag="460nm_gamma_k_probe_s_32.6eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "G_Gamma_K", "ProbeRotator_37", "31.95_sum"],
            tag="460nm_gamma_k_probe_s_31.95eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_82", "33.81eV_0"],
            tag="460nm_gamma_m_probe_p_33.9eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_82", "32.57eV_0"],
            tag="460nm_gamma_m_probe_p_32.6eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_82", "31.95eV_0"],
            tag="460nm_gamma_m_probe_p_31.95eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_82", "33.19eV_0"],
            tag="460nm_gamma_m_probe_p_33.25eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_37", "33.19eV_0"],
            tag="460nm_gamma_m_probe_s_33.25eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_37", "33.81eV_0"],
            tag="460nm_gamma_m_probe_s_33.9eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_37", "32.57eV_0"],
            tag="460nm_gamma_m_probe_s_32.6eV",
        ),
        DataItem(
            data_root_dir,
            ["Dec", "Gamma_M", "ProbeRotator_37", "31.95eV_0"],
            tag="460nm_gamma_m_probe_s_31.95eV",
        ),
    ]
    datas_keywords = [
        # ["Sep", "851"],  # 400nm Gamma-M probe S 31.95eV
        # ["Sep", "852"],  # 400nm Gamma-M probe S 32.6eV
        # ["Sep", "853"],  # 400nm Gamma-M probe S 33.25eV
        # ["Sep", "854"],  # 400nm Gamma-M probe S 33.9eV
        # ["Sep", "848"],  # 400nm Gamma-M probe P 31.95eV
        # ["Sep", "855"],  # 400nm Gamma-M probe P 33.9eV
        # ["Sep", "856"],  # 400nm Gamma-M probe P 33.25eV
        # ["Sep", "857"],  # 400nm Gamma-M probe P 32.6eV
        ["Sep", "837"],  # 400nm Gamma-K probe P 31.95eV
        ["Sep", "840"],  # 400nm Gamma-K probe P 32.6eV
        ["Sep", "841"],  # 400nm Gamma-K probe P 33.25eV
        ["Sep", "842"],  # 400nm Gamma-K probe P 33.9eV
        ["Sep", "843"],  # 400nm Gamma-K probe S 33.9eV
        ["Sep", "844"],  # 400nm Gamma-K probe S 33.25eV
        ["Sep", "845"],  # 400nm Gamma-K probe S 32.6eV
        ["Sep", "846"],  # 400nm Gamma-K probe S 31.95eV
        ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.25_sum"],  # 460
        ["Dec", "G_Gamma_K", "ProbeRotator_82", "31.95_sum"],  # 460
        ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.9_sum"],  # 460
        ["Dec", "G_Gamma_K", "ProbeRotator_82", "32.6_sum"],  # 460
        # ["Sep", "893"],  # 450nm Gamma-M probe P 31.95eV
        # ["Sep", "894"],  # 450nm Gamma-M probe S 31.95eV
        # ["Sep", "896"],  # 450nm Gamma-M probe S 32.6eV
        # ["Sep", "897"],  # 450nm Gamma-M probe S 33.25eV
        # ["Sep", "898"],  # 450nm Gamma-M probe S 33.9eV
        # ["Sep", "899"],  # 450nm Gamma-M probe P 33.9eV
        # ["Sep", "900"],  # 450nm Gamma-M probe P 33.25eV
        # ["Sep", "901"],  # 450nm Gamma-M probe P 32.6eV
        # ["Sep", "902"],  # 450nm Gamma-K probe S 32.6eV
        # ["Sep", "903"],  # 450nm Gamma-K probe S 33.25eV
        # ["Sep", "904"],  # 450nm Gamma-K probe S 33.9eV
        # ["Sep", "905"],  # 450nm Gamma-K probe P 33.9eV
        # ["Sep", "906"],  # 450nm Gamma-K probe P 33.25eV
        # ["Sep", "907"],  # 450nm Gamma-K probe P 32.6eV
        # ["Sep", "908"],  # 450nm Gamma-K probe P 31.95eV
        # ["Sep", "909"],  # 450nm Gamma-K probe S 31.95eV
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
        (49, 71),
        (130, 170),
        (240, 260),
        (395, 405),
        (690, 710),
        # (1990, 2010),
    ]
    color_scale_percentiles = [90, 92, 94, 96, 98]
    # gamma-K probe P
    subplots = [[], []]
    for time_1, time_2 in times:
        subplots[0].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-K probe P
                    "400nm_gamma_k_probe_p_31.95eV",
                    "400nm_gamma_k_probe_p_32.6eV",
                    "400nm_gamma_k_probe_p_33.25eV",
                    "400nm_gamma_k_probe_p_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas_smooth_stitching,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"400nm {50 * int((time_1 + time_2) / 100)}fs",
                color_scale_min=5,
                color_scale_max=1000,
            )
        )
        subplots[1].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-K probe P
                    "460nm_gamma_k_probe_p_31.95eV",
                    "460nm_gamma_k_probe_p_32.6eV",
                    "460nm_gamma_k_probe_p_33.25eV",
                    "460nm_gamma_k_probe_p_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"460nm {int((time_1 + time_2) / 2)}fs",
                color_scale_min=5,
                color_scale_max=500,
            )
        )
    plots.append(
        Plot(
            subplots=subplots,
            plot_name="Excitation Dynamics Gamma-K probe P 400nm & 460nm",
        )
    )

    # gamma-K probe S
    subplots = [[], []]
    for time_1, time_2 in times:
        subplots[0].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-K probe P
                    "400nm_gamma_k_probe_s_31.95eV",
                    "400nm_gamma_k_probe_s_32.6eV",
                    "400nm_gamma_k_probe_s_33.25eV",
                    "400nm_gamma_k_probe_s_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas_smooth_stitching,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"400nm {50 * int((time_1 + time_2) / 100)}fs",
                color_scale_min=5,
                color_scale_max=1000,
            )
        )
        subplots[1].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-K probe P
                    "460nm_gamma_k_probe_s_31.95eV",
                    "460nm_gamma_k_probe_s_32.6eV",
                    "460nm_gamma_k_probe_s_33.25eV",
                    "460nm_gamma_k_probe_s_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"460nm {int((time_1 + time_2) / 2)}fs",
                color_scale_min=5,
                color_scale_max=500,
            )
        )
    plots.append(
        Plot(
            subplots=subplots,
            plot_name="Excitation Dynamics Gamma-K probe S 400nm & 460nm",
        )
    )

    # gamma-M probe P
    subplots = [[], []]
    for time_1, time_2 in times:
        subplots[0].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-M probe P
                    "400nm_gamma_m_probe_p_31.95eV",
                    "400nm_gamma_m_probe_p_32.6eV",
                    "400nm_gamma_m_probe_p_33.25eV",
                    "400nm_gamma_m_probe_p_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas_smooth_stitching,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"400nm {50 * int((time_1 + time_2) / 100)}fs",
                color_scale_min=5,
                color_scale_max=700,
            )
        )
        subplots[1].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-K probe P
                    "460nm_gamma_m_probe_p_31.95eV",
                    "460nm_gamma_m_probe_p_32.6eV",
                    "460nm_gamma_m_probe_p_33.25eV",
                    "460nm_gamma_m_probe_p_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"460nm {int((time_1 + time_2) / 2)}fs",
                color_scale_min=5,
                color_scale_max=700,
            )
        )
    plots.append(
        Plot(
            subplots=subplots,
            plot_name="Excitation Dynamics Gamma-M probe P 400nm & 460nm",
        )
    )

    # gamma-M probe S
    subplots = [[], []]
    for time_1, time_2 in times:
        subplots[0].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-M probe S
                    "400nm_gamma_m_probe_s_31.95eV",
                    "400nm_gamma_m_probe_s_32.6eV",
                    "400nm_gamma_m_probe_s_33.25eV",
                    "400nm_gamma_m_probe_s_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas_smooth_stitching,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"400nm {50 * int((time_1 + time_2) / 100)}fs",
                color_scale_min=5,
                color_scale_max=600,
            )
        )
        subplots[1].append(
            SubPlot(
                plot_type=SubPlotType.SUM,
                keywords=[
                    # Gamma-M probe S
                    "460nm_gamma_m_probe_s_31.95eV",
                    "460nm_gamma_m_probe_s_32.6eV",
                    "460nm_gamma_m_probe_s_33.25eV",
                    "460nm_gamma_m_probe_s_33.9eV",
                ],
                axis_choices=[Axes.THETA_X, Axes.ENERGY],
                summing_ranges={
                    Axes.TIME: [[time_1, time_2]],
                },
                data_operations=[],
                data_preparations=[
                    concat_energy_datas,
                    bin_sweep_datas,
                    subtract_noise,
                    remove_bias,
                    remove_e_fermi,
                    pad_energy,
                    sum_data_prep,
                ],
                plot_name=f"460nm {int((time_1 + time_2) / 2)}fs",
                color_scale_min=5,
                color_scale_max=800,
            )
        )
    plots.append(
        Plot(
            subplots=subplots,
            plot_name="Excitation Dynamics Gamma-M probe S 400nm & 460nm",
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
                    color_map = "bwr" if plot.plot_type == SubPlotType.CD else "terrain"
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
                        percentile_for_colorscale=(98),
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
        # fig.suptitle(f"{plot_plan.plot_name}")
        fig.subplots_adjust(hspace=0)
        # fig.tight_layout()
        output_path = os.path.join(dir, f"{file_name}.png".replace(" ", "_"))
        fig.savefig(output_path)
    plt.show()
    input("Press Enter to continue...")


def tau_comparison(
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
                concat_energy_datas_smooth_stitching,
                bin_sweep_datas,
                subtract_noise,
                remove_bias,
                remove_e_fermi,
                pad_energy,
                sum_data_prep,
            ],
            "title": "400nm Tau",
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
                concat_energy_datas,
                bin_sweep_datas,
                subtract_noise,
                remove_bias,
                remove_e_fermi,
                pad_energy,
                sum_data_prep,
            ],
            "title": "460nm Tau",
        },
    ]

    # Compute taus for both datasets
    scatter_results = []
    for config in tau_configs:
        plot_data = []
        for data_item in config["data_items"]:
            if verbose:
                print(f"Loading {data_item.search_keywords}")
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
        for preparation in config["preparations"]:
            plot_data = preparation(plot_data)
        sweep_data = plot_data[0][0]
        result = compute_taus_data(
            sweep_data,
            export_name=config["title"],
            energy_low_limit=energy_low,
            energy_high_limit=energy_high,
            taus_percentiles=taus_percentile,
        )
        scatter_results.append(result)

    # Build a shared color scale from the union of both ranges
    all_low = min(cs[3][0] for cs in scatter_results)
    all_high = max(cs[3][-1] for cs in scatter_results)
    shared_color_scale = np.linspace(all_low, all_high, 70)

    # Plot both with the shared color scale
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    fig.suptitle("Tau Comparison: 400nm vs 460nm (Gamma-K probe P)", fontsize=14)
    for ax, config, (x, y, c_tau, _, extent, data), pump_wavelength in zip(
        axes, tau_configs, scatter_results, [400, 460]
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
            pump_wavelength=pump_wavelength,
        )

    output_path = os.path.join(
        dir, f"tau_comparison_400nm_vs_460nm_{energy_low}eV_{energy_high}eV.png"
    )
    fig.savefig(output_path)
    if verbose:
        print(f"Saved tau comparison to {output_path}")
    plt.show()
    # input("Press Enter to continue...")


if False:
    basic_analysis_template(verbose=True)
tau_comparison(verbose=True, energy_low=0.0, energy_high=1.0, taus_percentile=(40, 95))
tau_comparison(verbose=True, energy_low=1.0, energy_high=3.0, taus_percentile=(5, 50))
