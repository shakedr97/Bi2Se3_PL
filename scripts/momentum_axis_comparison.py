import os
import sys
import copy

path = os.path.dirname(__file__)
os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from SweepData import SweepData

output_dir = os.path.join(path, "..", "plots")
os.makedirs(output_dir, exist_ok=True)

data_root_dir = "/home/shakedr/Downloads/Bi2Se3/"

# 400nm Gamma-K probe P, four energy windows
datas_keywords = [
    ["Sep", "837"],
    ["Sep", "840"],
    ["Sep", "841"],
    ["Sep", "842"],
]

print("Loading data...")
all_datas = {}
for kws in datas_keywords:
    key = kws[-1]
    all_datas[key] = read_sweep_data(data_root_dir, kws)
    print(f"  loaded {len(all_datas[key])} sweeps for {kws}")

# Assemble the four windows into one list-of-lists as expected by prep functions
raw = [all_datas["837"], all_datas["840"], all_datas["841"], all_datas["842"]]
raw = [copy.deepcopy(r) for r in raw]

# Standard SUM pipeline (mirrors basic_analysis_template_400_gamma_k_P.py)
data = concat_energy_datas_smooth_stitching(raw)
data = remove_bias(data)
data = remove_e_fermi(data)
data = pad_energy(data)
data = subtract_noise(data)
data = sum_data_prep(data)

# data is List[List[SweepData]]; first group, first (and only) sweep
sweep = data[0][0]

summing_ranges = {Axes.TIME: [[40, 60]]}

plt.ioff()
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, cutoff, label in [
    (axes[0], 1.0, r"$k$ limits: $\pm1.0\ \mathrm{\AA}^{-1}$"),
    (axes[1], 0.7, r"$k$ limits: $\pm0.7\ \mathrm{\AA}^{-1}$"),
]:
    display_sweep_data(
        sweep,
        axes_choice=[Axes.THETA_X, Axes.ENERGY],
        summing_ranges=summing_ranges,
        plot_axis=ax,
        color_map="terrain",
        percentile_for_colorscale=98,
        angle_to_momentum=True,
        momentum_cutoff=cutoff,
    )
    ax.set_title(label)

fig.suptitle("400 nm Γ-K, probe P, delay ≈ +50 fs — momentum axis comparison")
fig.tight_layout()
out_path = os.path.join(output_dir, "momentum_axis_comparison.png")
fig.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
