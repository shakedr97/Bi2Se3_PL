import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

dir = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from analysis.Bi2Se3_BandStructure import (
    Bi2Se3_BandStructure_Gamma_K,
    Bi2Se3_BandStructure_Gamma_M,
)

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

# BC2 display window
E_LOW, E_HIGH = 1.9, 2.6
K_CUTOFF = 0.25

# Angular offset per (combination, wavelength) in degrees.
# Tune to center the signal; positive = shift momentum axis to higher values.
ANGLE_OFFSET_GK_P_461 = -0.5
ANGLE_OFFSET_GK_P_423 = 2.5
ANGLE_OFFSET_GK_P_400 = 0.0

ANGLE_OFFSET_GK_S_461 = -0.5
ANGLE_OFFSET_GK_S_423 = 2.5
ANGLE_OFFSET_GK_S_400 = 0.0

ANGLE_OFFSET_GM_P_461 = 0.0
ANGLE_OFFSET_GM_P_423 = 0.0
ANGLE_OFFSET_GM_P_400 = 0.0

ANGLE_OFFSET_GM_S_461 = 0.0
ANGLE_OFFSET_GM_S_423 = 0.0
ANGLE_OFFSET_GM_S_400 = 0.0

time_delays = [
    (-80, -30),  # captures key -50
    (-10, 10),  # captures key 0
    (30, 80),  # captures key 50
    (90, 110),  # captures key 100
    (140, 160),  # captures key 150
]
time_labels = ["-50 fs", "0 fs", "+50 fs", "+100 fs", "+150 fs"]

# ── DataItem definitions ───────────────────────────────────────────────────────

# 461nm Gamma-K P: SUM uses the pre-summed G_Gamma_K directory; CD uses the
# individual-channel Bi2Se3_G_CD_Gamma_K directory (only 3 time points).
_di_461_gk_p_sum = [
    DataItem(
        data_root_dir,
        ["Dec", "G_Gamma_K", "ProbeRotator_82", "31.95_sum"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "32.6_sum"], tag="32.6eV"
    ),
    DataItem(
        data_root_dir,
        ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.25_sum"],
        tag="33.25eV",
    ),
    DataItem(
        data_root_dir, ["Dec", "G_Gamma_K", "ProbeRotator_82", "33.9_sum"], tag="33.9eV"
    ),
]
_di_461_gk_p_cd = [
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "31.95_sum"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "32.6_sum"],
        tag="32.6eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "33.25_sum"],
        tag="33.25eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_82", "33.9_sum"],
        tag="33.9eV",
    ),
]

# 461nm Gamma-K S: both SUM and CD from the individual-channel directory
_di_461_gk_s = [
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_37", "31.95_sum"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_37", "32.6_sum"],
        tag="32.6eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_37", "33.25_sum"],
        tag="33.25eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_G_CD_Gamma_K", "ProbeRotator_37", "33.9_sum"],
        tag="33.9eV",
    ),
]

# 461nm Gamma-M P
_di_461_gm_p = [
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_82", "31.95eV_0"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_82", "32.57_sum"],
        tag="32.57eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_82", "33.19_sum"],
        tag="33.19eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_82", "33.81eV_0"],
        tag="33.81eV",
    ),
]

# 461nm Gamma-M S
_di_461_gm_s = [
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "31.95eV_0"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "32.57_sum"],
        tag="32.57eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "33.19_sum"],
        tag="33.19eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "33.81eV_0"],
        tag="33.81eV",
    ),
]

# 400nm Gamma-K P
_di_400_gk_p = [
    DataItem(data_root_dir, ["Sep", "837"], tag="31.95eV"),
    DataItem(data_root_dir, ["Sep", "840"], tag="32.6eV"),
    DataItem(data_root_dir, ["Sep", "841"], tag="33.25eV"),
    DataItem(data_root_dir, ["Sep", "842"], tag="33.9eV"),
]

# 400nm Gamma-K S
_di_400_gk_s = [
    DataItem(data_root_dir, ["Sep", "846"], tag="31.95eV"),
    DataItem(data_root_dir, ["Sep", "845"], tag="32.6eV"),
    DataItem(data_root_dir, ["Sep", "844"], tag="33.25eV"),
    DataItem(data_root_dir, ["Sep", "843"], tag="33.9eV"),
]

# 400nm Gamma-M P
_di_400_gm_p = [
    DataItem(data_root_dir, ["Sep", "848"], tag="31.95eV"),
    DataItem(data_root_dir, ["Sep", "857"], tag="32.6eV"),
    DataItem(data_root_dir, ["Sep", "856"], tag="33.25eV"),
    DataItem(data_root_dir, ["Sep", "855"], tag="33.9eV"),
]

# 400nm Gamma-M S
_di_400_gm_s = [
    DataItem(data_root_dir, ["Sep", "851"], tag="31.95eV"),
    DataItem(data_root_dir, ["Sep", "852"], tag="32.6eV"),
    DataItem(data_root_dir, ["Sep", "853"], tag="33.25eV"),
    DataItem(data_root_dir, ["Sep", "854"], tag="33.9eV"),
]

# 461nm Gamma-K S backup: Bi2Se3_Gamma_K has all 15 time points for S probe;
# used to fill +100 fs and +150 fs panels missing from the primary _summed dataset.
_di_461_gk_s_backup = [
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_K", "ProbeRotator_37", "31.95eV_0"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_K", "ProbeRotator_37", "32.57eV_0"],
        tag="32.57eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_K", "ProbeRotator_37", "33.19eV_0"],
        tag="33.19eV",
    ),
    DataItem(
        data_root_dir,
        ["Dec", "Bi2Se3_Gamma_K", "ProbeRotator_37", "33.81eV_0"],
        tag="33.81eV",
    ),
]

# 423nm Gamma-K S (loaded via DataItem; ProbeRotator-37 selects S probe)
_di_423_gk_s = [
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_K", "ProbeRotator-37", "eV_0", "31.95"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_K", "ProbeRotator-37", "eV_0", "32.6"],
        tag="32.6eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_K", "ProbeRotator-37", "eV_1", "33.25"],
        tag="33.25eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_K", "ProbeRotator-37", "eV_0", "33.9"],
        tag="33.9eV",
    ),
]

# 423nm Gamma-M P
_di_423_gm_p = [
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_82", "eV_0", "31.95"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_82", "eV_0", "32.6"],
        tag="32.6eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_82", "eV_0", "33.25"],
        tag="33.25eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_82", "eV_0", "33.9"],
        tag="33.9eV",
    ),
]

# 423nm Gamma-M S
_di_423_gm_s = [
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_37", "eV_0", "31.95"],
        tag="31.95eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_37", "eV_0", "32.6"],
        tag="32.6eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_37", "eV_0", "33.25"],
        tag="33.25eV",
    ),
    DataItem(
        data_root_dir,
        ["Nov", "423nm", "Gamma_M", "ProbeRotator_37", "eV_0", "33.9"],
        tag="33.9eV",
    ),
]

# ── Loading and preparation helpers ───────────────────────────────────────────

_SUM_SMOOTH = [
    concat_energy_datas_smooth_stitching,
    remove_bias,
    remove_e_fermi,
    pad_energy,
    subtract_noise,
    sum_data_prep,
]
_CD_SMOOTH = [
    cd_data_prep,
    concat_energy_datas_smooth_stitching_no_filling,
    remove_bias,
    remove_e_fermi,
    pad_energy,
]
_SUM_461 = [
    concat_energy_datas,
    remove_bias,
    remove_e_fermi,
    pad_energy,
    sum_data_prep,
]
_CD_461 = [
    cd_data_prep,
    concat_energy_datas,
    remove_bias,
    remove_e_fermi,
    pad_energy,
]


def load_raw(data_items):
    plot_data = []
    for di in data_items:
        print(f"  Loading {di.search_keywords}")
        try:
            datas = read_sweep_data(di.base_directory, di.search_keywords)
            if len(datas) <= 1:
                raise Exception()
        except Exception:
            datas = read_all_sweep_data(di.base_directory, di.search_keywords)
        plot_data.append(datas)
    return plot_data


def _prep(plot_data, pipeline):
    for fn in pipeline:
        plot_data = fn(plot_data)
    return plot_data[0][0]


def _load_423nm_gk_p():
    """Load 423nm Gamma-K P data as [[c1, c2], ...] per energy window."""
    d_2318 = read_all_sweep_data(data_root_dir, ["Nov", "423nm", "2318"])
    windows = [
        d_2318,
        [
            read_sweep_data(
                data_root_dir,
                ["Nov", "423nm", "Gamma_K", "eV_0", "32.6", "PumpRotator-160"],
            )[0],
            read_sweep_data(
                data_root_dir,
                ["Nov", "423nm", "Gamma_K", "eV_0", "32.6", "PumpRotator-70"],
            )[0],
        ],
        [
            read_sweep_data(
                data_root_dir,
                ["Nov", "423nm", "Gamma_K", "eV_1", "33.25", "PumpRotator-160"],
            )[0],
            read_sweep_data(
                data_root_dir,
                ["Nov", "423nm", "Gamma_K", "eV_1", "33.25", "PumpRotator-70"],
            )[0],
        ],
        [
            read_sweep_data(
                data_root_dir,
                ["Nov", "423nm", "Gamma_K", "eV_0", "33.9", "PumpRotator-160"],
            )[0],
            read_sweep_data(
                data_root_dir,
                ["Nov", "423nm", "Gamma_K", "eV_0", "33.9", "PumpRotator-70"],
            )[0],
        ],
    ]
    return windows


# ── Figure generation ──────────────────────────────────────────────────────────


def make_closeup_figure(
    row_specs,
    row_labels,
    color_map,
    percentile,
    fig_title,
    output_name,
    band_structure=Bi2Se3_BandStructure_Gamma_K,
    angle_offsets_deg=None,
):
    """
    row_specs: list of rows. Each row is either:
      - a single SweepData  (same dataset for all time-delay columns), or
      - a list[SweepData]   (one dataset per column, length == len(time_delays))
    band_structure: BandStructure object to overlay (Gamma-K or Gamma-M).
    angle_offsets_deg: optional list of angular offsets in degrees, one per row.
    """
    n_rows = len(row_specs)
    n_cols = len(time_delays)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        sharey=True,
        sharex=True,
    )
    fig.subplots_adjust(
        left=0.08, right=0.98, top=0.88, bottom=0.1, wspace=0.03, hspace=0.15
    )
    if n_rows == 1:
        axes = [axes]

    offsets = angle_offsets_deg if angle_offsets_deg is not None else [0.0] * n_rows

    for i, (row_spec, label) in enumerate(zip(row_specs, row_labels)):
        col_data = row_spec if isinstance(row_spec, list) else [row_spec] * n_cols
        angle_offset = offsets[i]

        for j, (t1, t2) in enumerate(time_delays):
            ax = axes[i][j]
            data = col_data[j]
            time_keys = sorted(data.sweep_raw.keys())
            available = [t for t in time_keys if t1 <= t <= t2]

            if not available:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="gray",
                )
                ax.set_facecolor("#f4f4f4")
            else:
                with data.angle_offset(angle_offset):
                    display_sweep_data(
                        data,
                        axes_choice=[Axes.THETA_X, Axes.ENERGY],
                        summing_ranges={Axes.TIME: [[t1, t2]]},
                        plot_axis=ax,
                        color_map=color_map,
                        percentile_for_colorscale=percentile,
                        angle_to_momentum=True,
                        momentum_cutoff=K_CUTOFF,
                        display_plot_title=False,
                        display_x_label=(i == n_rows - 1),
                        display_y_label=False,
                    )
                band_structure.plot(
                    ax,
                    momentum_low_limit=-K_CUTOFF,
                    momentum_high_limit=K_CUTOFF,
                    energy_low_limit=E_LOW,
                    energy_high_limit=E_HIGH,
                    plot_band_names=False,
                )

            ax.set_ylim(E_LOW, E_HIGH)
            ax.set_xlim(-K_CUTOFF, K_CUTOFF)
            if i == 0:
                ax.set_title(time_labels[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{label}\n" + r"$E-E_F$ [eV]", fontsize=10)
            if i == n_rows - 1:
                ax.set_xlabel(r"$k_x\ [\AA^{-1}]$", fontsize=10)

    for row in axes:
        for ax in row:
            ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3, prune="both"))

    fig.suptitle(fig_title, fontsize=13)
    output_path = os.path.join(dir, output_name)
    fig.savefig(output_path, dpi=150, pil_kwargs={"quality": 85})
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Gamma-K, probe P ──────────────────────────────────────────────────────────

print("\n=== Gamma-K, probe P ===")
print("  Loading 461 nm SUM...")
sum_461_gk_p = _prep(load_raw(_di_461_gk_p_sum), _SUM_461)
print("  Loading 461 nm CD...")
cd_461_gk_p = _prep(load_raw(_di_461_gk_p_cd), _CD_461)
print("  Loading 423 nm SUM...")
sum_423_gk_p = _prep(_load_423nm_gk_p(), _SUM_SMOOTH)
print("  Loading 423 nm CD...")
cd_423_gk_p = _prep(_load_423nm_gk_p(), _CD_SMOOTH)
print("  Loading 400 nm SUM...")
sum_400_gk_p = _prep(load_raw(_di_400_gk_p), _SUM_SMOOTH)
print("  Loading 400 nm CD...")
cd_400_gk_p = _prep(load_raw(_di_400_gk_p), _CD_SMOOTH)

# 461nm K P CD only has time points -50, 0, +50; fall back to SUM for later
cd_461_gk_p_row = [cd_461_gk_p, cd_461_gk_p, cd_461_gk_p, sum_461_gk_p, sum_461_gk_p]

make_closeup_figure(
    row_specs=[sum_461_gk_p, sum_423_gk_p, sum_400_gk_p],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="terrain",
    percentile=95,
    fig_title=r"BC2 Close-Up ARPES Snapshots — SUM ($\Gamma$-K, probe P)",
    output_name="bc2_closeup_SUM_Gamma_K_P.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_K,
    angle_offsets_deg=[
        ANGLE_OFFSET_GK_P_461,
        ANGLE_OFFSET_GK_P_423,
        ANGLE_OFFSET_GK_P_400,
    ],
)
make_closeup_figure(
    row_specs=[cd_461_gk_p_row, cd_423_gk_p, cd_400_gk_p],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="bwr",
    percentile=99,
    fig_title=r"BC2 Close-Up ARPES Snapshots — CD ($\Gamma$-K, probe P)",
    output_name="bc2_closeup_CD_Gamma_K_P.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_K,
    angle_offsets_deg=[
        ANGLE_OFFSET_GK_P_461,
        ANGLE_OFFSET_GK_P_423,
        ANGLE_OFFSET_GK_P_400,
    ],
)

# ── Gamma-K, probe S ──────────────────────────────────────────────────────────

print("\n=== Gamma-K, probe S ===")
print("  Loading 461 nm SUM...")
sum_461_gk_s = _prep(load_raw(_di_461_gk_s), _SUM_461)
print("  Loading 461 nm CD...")
cd_461_gk_s = _prep(load_raw(_di_461_gk_s), _CD_461)
print("  Loading 461 nm SUM backup (all time points)...")
backup_sum_461_gk_s = _prep(load_raw(_di_461_gk_s_backup), _SUM_461)
print("  Loading 423 nm SUM...")
sum_423_gk_s = _prep(load_raw(_di_423_gk_s), _SUM_SMOOTH)
print("  Loading 423 nm CD...")
cd_423_gk_s = _prep(load_raw(_di_423_gk_s), _CD_SMOOTH)
print("  Loading 400 nm SUM...")
sum_400_gk_s = _prep(load_raw(_di_400_gk_s), _SUM_SMOOTH)
print("  Loading 400 nm CD...")
cd_400_gk_s = _prep(load_raw(_di_400_gk_s), _CD_SMOOTH)

# Primary K S dataset only has -50, 0, +50; back-fill +100 and +150 from backup
sum_461_gk_s_row = [
    sum_461_gk_s,
    sum_461_gk_s,
    sum_461_gk_s,
    backup_sum_461_gk_s,
    backup_sum_461_gk_s,
]
cd_461_gk_s_row = [
    cd_461_gk_s,
    cd_461_gk_s,
    cd_461_gk_s,
    backup_sum_461_gk_s,
    backup_sum_461_gk_s,
]

make_closeup_figure(
    row_specs=[sum_461_gk_s_row, sum_423_gk_s, sum_400_gk_s],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="terrain",
    percentile=95,
    fig_title=r"BC2 Close-Up ARPES Snapshots — SUM ($\Gamma$-K, probe S)",
    output_name="bc2_closeup_SUM_Gamma_K_S.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_K,
    angle_offsets_deg=[
        ANGLE_OFFSET_GK_S_461,
        ANGLE_OFFSET_GK_S_423,
        ANGLE_OFFSET_GK_S_400,
    ],
)
make_closeup_figure(
    row_specs=[cd_461_gk_s_row, cd_423_gk_s, cd_400_gk_s],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="bwr",
    percentile=99,
    fig_title=r"BC2 Close-Up ARPES Snapshots — CD ($\Gamma$-K, probe S)",
    output_name="bc2_closeup_CD_Gamma_K_S.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_K,
    angle_offsets_deg=[
        ANGLE_OFFSET_GK_S_461,
        ANGLE_OFFSET_GK_S_423,
        ANGLE_OFFSET_GK_S_400,
    ],
)

# ── Gamma-M, probe P ──────────────────────────────────────────────────────────

print("\n=== Gamma-M, probe P ===")
print("  Loading 461 nm SUM...")
sum_461_gm_p = _prep(load_raw(_di_461_gm_p), _SUM_461)
print("  Loading 461 nm CD...")
cd_461_gm_p = _prep(load_raw(_di_461_gm_p), _CD_461)
print("  Loading 423 nm SUM...")
sum_423_gm_p = _prep(load_raw(_di_423_gm_p), _SUM_SMOOTH)
print("  Loading 423 nm CD...")
cd_423_gm_p = _prep(load_raw(_di_423_gm_p), _CD_SMOOTH)
print("  Loading 400 nm SUM...")
sum_400_gm_p = _prep(load_raw(_di_400_gm_p), _SUM_SMOOTH)
print("  Loading 400 nm CD...")
cd_400_gm_p = _prep(load_raw(_di_400_gm_p), _CD_SMOOTH)

make_closeup_figure(
    row_specs=[sum_461_gm_p, sum_423_gm_p, sum_400_gm_p],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="terrain",
    percentile=95,
    fig_title=r"BC2 Close-Up ARPES Snapshots — SUM ($\Gamma$-M, probe P)",
    output_name="bc2_closeup_SUM_Gamma_M_P.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_M,
    angle_offsets_deg=[
        ANGLE_OFFSET_GM_P_461,
        ANGLE_OFFSET_GM_P_423,
        ANGLE_OFFSET_GM_P_400,
    ],
)
make_closeup_figure(
    row_specs=[cd_461_gm_p, cd_423_gm_p, cd_400_gm_p],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="bwr",
    percentile=99,
    fig_title=r"BC2 Close-Up ARPES Snapshots — CD ($\Gamma$-M, probe P)",
    output_name="bc2_closeup_CD_Gamma_M_P.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_M,
    angle_offsets_deg=[
        ANGLE_OFFSET_GM_P_461,
        ANGLE_OFFSET_GM_P_423,
        ANGLE_OFFSET_GM_P_400,
    ],
)

# ── Gamma-M, probe S ──────────────────────────────────────────────────────────

print("\n=== Gamma-M, probe S ===")
print("  Loading 461 nm SUM...")
sum_461_gm_s = _prep(load_raw(_di_461_gm_s), _SUM_461)
print("  Loading 461 nm CD...")
cd_461_gm_s = _prep(load_raw(_di_461_gm_s), _CD_461)
print("  Loading 423 nm SUM...")
sum_423_gm_s = _prep(load_raw(_di_423_gm_s), _SUM_SMOOTH)
print("  Loading 423 nm CD...")
cd_423_gm_s = _prep(load_raw(_di_423_gm_s), _CD_SMOOTH)
print("  Loading 400 nm SUM...")
sum_400_gm_s = _prep(load_raw(_di_400_gm_s), _SUM_SMOOTH)
print("  Loading 400 nm CD...")
cd_400_gm_s = _prep(load_raw(_di_400_gm_s), _CD_SMOOTH)

make_closeup_figure(
    row_specs=[sum_461_gm_s, sum_423_gm_s, sum_400_gm_s],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="terrain",
    percentile=95,
    fig_title=r"BC2 Close-Up ARPES Snapshots — SUM ($\Gamma$-M, probe S)",
    output_name="bc2_closeup_SUM_Gamma_M_S.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_M,
    angle_offsets_deg=[
        ANGLE_OFFSET_GM_S_461,
        ANGLE_OFFSET_GM_S_423,
        ANGLE_OFFSET_GM_S_400,
    ],
)
make_closeup_figure(
    row_specs=[cd_461_gm_s, cd_423_gm_s, cd_400_gm_s],
    row_labels=["461 nm", "423 nm", "400 nm"],
    color_map="bwr",
    percentile=99,
    fig_title=r"BC2 Close-Up ARPES Snapshots — CD ($\Gamma$-M, probe S)",
    output_name="bc2_closeup_CD_Gamma_M_S.jpg",
    band_structure=Bi2Se3_BandStructure_Gamma_M,
    angle_offsets_deg=[
        ANGLE_OFFSET_GM_S_461,
        ANGLE_OFFSET_GM_S_423,
        ANGLE_OFFSET_GM_S_400,
    ],
)
