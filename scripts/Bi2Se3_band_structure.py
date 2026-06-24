import os
import sys
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

dir = os.path.dirname(__file__)

os.chdir("/home/shakedr/git/taudaqcode")
sys.path.append("/home/shakedr/git/taudaqcode")
from analysis.analysis_tools import *
from analysis.analysis_tools import BandStructure, plot_transitions
from SweepData import SweepData
from analysis.Bi2Se3_BandStructure import (
    Bi2Se3_BandStructure_Gamma_K,
    Bi2Se3_BandStructure_Gamma_M,
    Bi2Se3_Conduction_2_3eV_Gamma_M,
    Bi2Se3_Conduction_2_3eV_Gamma_K,
    Bi2Se3_DiracOne_ArmOne_Gamma_K,
    Bi2Se3_DiracOne_ArmTwo_Gamma_K,
    Bi2Se3_DiracOne_ArmOne_Gamma_M,
    Bi2Se3_DiracOne_ArmTwo_Gamma_M,
    Bi2Se3_DiracZero_ArmOne_Gamma_K,
    Bi2Se3_DiracZero_ArmTwo_Gamma_K,
    Bi2Se3_DiracZero_ArmOne_Gamma_M,
    Bi2Se3_DiracZero_ArmTwo_Gamma_M,
    Bi2Se3_Valence_RSS_Gamma_K,
    Bi2Se3_Valence_RSS_Gamma_M,
)

dir = os.path.join(dir, "..", "plots")
os.chdir(dir)
momentum_low_limit = -0.6
momentum_high_limit = 0.6

Bi2Se3_BandStructure_Gamma_K.plot(
    momentum_low_limit=momentum_low_limit, momentum_high_limit=momentum_high_limit
)


pump_wave_length = 460
plot_transitions(
    -2.26,
    Bi2Se3_BandStructure_Gamma_K,
    color="green",
    momentum_low=momentum_low_limit,
    momentum_high=momentum_high_limit,
    to_band=Bi2Se3_Valence_RSS_Gamma_K,
    from_band=Bi2Se3_DiracOne_ArmTwo_Gamma_K,
)
plot_transitions(
    -2.26,
    Bi2Se3_BandStructure_Gamma_K,
    color="green",
    momentum_low=momentum_low_limit,
    momentum_high=momentum_high_limit,
    to_band=Bi2Se3_Valence_RSS_Gamma_K,
    from_band=Bi2Se3_DiracOne_ArmOne_Gamma_K,
)
plot_transitions(
    1240 / pump_wave_length,
    Bi2Se3_BandStructure_Gamma_K,
    color="blue",
    momentum_low=momentum_low_limit,
    momentum_high=momentum_high_limit,
    from_band=Bi2Se3_Valence_RSS_Gamma_K,
)
plt.savefig(f"transitions_{pump_wave_length}_rss_gamma_k.png")

input("Enter to continue...")
