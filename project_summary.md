# Bi₂Se₃ PL Project Summary

## Running scripts

All scripts require the `taudaqcode` virtual environment. Activate it before running any script:

```bash
source ~/git/taudaqcode/bin/activate        # bash/zsh
source ~/git/taudaqcode/bin/activate.fish   # fish
```

To understand how a dataset is loaded and plotted, read an existing `basic_analysis_template_<nm>_<dir>_<pol>.py` script — they are the canonical reference for keyword patterns, prep function ordering, and `display_sweep_data` call signatures.

---

## Core dependency: `taudaqcode`

All scripts import from `/home/shakedr/git/taudaqcode` — a custom library that is the real engine of this project. It provides:
- `SweepData` — the main data container, with axes THETA_X (detector angle), ENERGY (kinetic), and TIME (pump-probe delay)
- `analysis.analysis_tools` — the full toolkit: data loading, stitching, preprocessing, and visualization
- `analysis.Bi2Se3_BandStructure` — band structure objects for Γ-K and Γ-M directions, including Dirac cone arms, surface states (RSS, SS2), and conduction bands

---

## Data pipeline (common to all scripts)

1. **Load** — `read_sweep_data(root, keywords)` finds raw scan files by keyword list; returns two `SweepData` objects (one per circular polarization channel, c1/c2)
2. **Combine channels** — `sum_data_sweep(c1, c2)` or `cd_data_sweep(c1, c2)` for SUM vs CD signal
3. **Stitch energy windows** — each dataset spans ~0.65 eV; four windows are concatenated with `concat_energy_datas_smooth_stitching` to cover 0–3 eV above E_F
4. **Calibrate** — `remove_bias` (kinetic energy offset from metadata), `remove_e_fermi` (set zero to Fermi level)
5. **Clean** — `subtract_noise`, `remove_outliers_single`, `bin_sweep_datas`, `pad_energy`
6. **Visualize** — `display_sweep_data` (snapshot ARPES images) or `display_pixel_resolved_time_dynamics` (maps of fitted decay times τ across the (k, E) plane)

**Data nesting convention:** all prep functions (`concat_energy_datas_smooth_stitching`, `remove_bias`, `sum_data_prep`, etc.) operate on `List[List[SweepData]]` — outer list = energy windows (or dataset groups), inner list = polarization channels (c1, c2). After the full pipeline, the result is `data[0][0]` to get a single `SweepData` for plotting.

**CD prep ordering:** `cd_data_prep` must be called *before* any stitching function (e.g. `concat_energy_datas_smooth_stitching_no_filling`). The smooth stitching variants normalise each energy window's overlap region against its neighbour to match scales — a step that requires positive-definite data. CD = (c1 − c2) is a difference signal that can be zero or negative, making that normalisation meaningless or numerically explosive. Running `cd_data_prep` first computes CD per-window while each window is still individually positive; stitching is then done with `concat_energy_datas_smooth_stitching_no_filling`, which skips the overlap normalisation for exactly this reason. Reversing the order produces silent data corruption.

**461 nm Γ-K SUM: use `Bi2Se3_G_Gamma_K_*_summed`, NOT `Bi2Se3_Gamma_K`.**  The `Bi2Se3_Gamma_K` directory (keywords `["Dec", "Bi2Se3_Gamma_K", ...]`) is a single-pass measurement with ~8× fewer accumulated counts than the summed dataset. Using it gives an artificially noisy ARPES image and unreliable τ fits (e.g. 334 fs vs the correct 454 fs at the blob ROI). The correct dataset for 461 nm Γ-K SUM analysis is `Bi2Se3_G_Gamma_K_*_summed` (keywords `["Dec", "G_Gamma_K", "31.95_summed", "ProbeRotator_37"]` etc.), which has 4 summed runs, `PumpRotator_167/212` channels, and all 15 time points (-140 to 2000 fs). Energy windows: 31.95, 32.6, 33.25, 33.9 eV.

**Γ-M keyword loading silently hits duplicate directories.** Keywords `["Dec", "Bi2Se3_Gamma_M", "ProbeRotator_37", "31.95eV_0"]` match files in both `Bi2Se3_Dec2025/Bi2Se3_Gamma_M_31.95eV_0/` and `Bi2Se3_Dec2025/Gamma_M/Bi2Se3_Gamma_M_31.95eV_0/`, producing 4 items for windows 1 and 4 instead of 2 (non-uniform warning from `concat_energy_datas`). The pipeline still produces correct time-series τ values (the extra channels cancel in sum_data_prep), but the non-uniform warning is expected and benign here.

**At the blob ROI (−0.75 to −0.5 Å⁻¹, 0.49–0.61 eV), τ_Γ-K ≈ 454 fs vs τ_Γ-M ≈ 358 fs for 461 nm.** The same (k, E) region decays ~25% slower in the Γ-K direction than in Γ-M. This supports the interpretation that the blob feature is Γ-M-specific: in Γ-M it stands out as anomalously long-lived relative to nearby states, while in Γ-K the same momentum/energy window captures a different part of the band structure with a naturally longer decay. Scripts: `scripts/gamma_m_vs_k_blob_461nm.py`.

**`display_pixel_resolved_decay_times` requires a raw kinetic-energy axis.** This function internally computes `energy = x_scatter + data.energy_offset` and uses the x_axis values as kinetic energy for the angle→momentum conversion. It therefore expects `data.x_axis_minimum` to be in raw KE units (~31.5 eV) with `energy_offset` set to the negative Fermi-level KE (~−31.5 eV), so that `KE + offset ≈ E−EF`. If you run the standard `remove_bias` + `remove_e_fermi` pipeline first, `x_axis_minimum` is shifted to E−EF units (~0 eV). Calling the function on such data causes two bugs: (1) the energy window filter matches nothing (all computed energies are ~−31.5 eV), giving a uniform zero-decay map; and (2) the momentum conversion uses ~0.5 eV instead of ~32 eV as KE, compressing the momentum axis by ~8×. **Fix:** immediately before calling the function, restore the raw KE axis with `data.x_axis_minimum -= data.energy_offset` — this un-does the E-EF shift while leaving `energy_offset` intact for the function to use. See `scripts/gamma_m_blob_tau_map.py` for the pattern.

**`get_1D_data` axis_map update bug — fixed.** When `summing_ranges` contains more than one axis (e.g. `THETA_X` and `TIME`), the original axis_map update loop compared values from two incompatible index spaces (current data positions vs. original dimension indices), causing it to null out the wrong axis (TIME instead of THETA_X) after the first sum. This silently broke every `display_sweep_data` call that extracted a 1D EDC with both `THETA_X` and `TIME` in `summing_ranges` — including `edc_comparison.py`. Fixed in `taudaqcode/analysis/analysis_tools.py` (`get_1D_data`): the loop is replaced with `axis_map[axis] = None; axis_indices.pop(axis_index)`, which marks the correct axis as removed and preserves the position→original-dim mapping without re-sorting.

**`bin_sweep_datas` 1-off in `x_axis_count` — fixed.** `bin_sweep_datas` used `int(count / bin_param)` (floor) to update `x_axis_count`, but `bin_data` internally uses `np.arange(0, count, bin_param)` to create bins, whose length is `ceil(count / bin_param)`. For a non-integer ratio (e.g. 1255 / 1.5 = 836.67) this left `x_axis_count` one short of the actual array size, propagating through every subsequent pipeline step. Fixed by changing both `x_axis_count` and `y_axis_count` updates to `int(np.ceil(count / bin_param))`.

**Time grids are not guaranteed to match across datasets.** Different datasets may have been acquired with different time-point sets — e.g. finer steps near t₀ to improve time resolution, or a reduced set focused on a few key delays. Always inspect `sorted(sweep_data.sweep_raw.keys())` before assuming a time point exists. In `display_sweep_data`, if no time key falls within a `summing_ranges` interval, `get_2D_data` now emits a `warnings.warn` explaining which axis and ranges matched nothing, then leaves the axis unsummed (the downstream `AssertionError` still fires, but the cause is now visible).

**423nm Gamma-K P requires explicit per-polarization loading.** Unlike other datasets where `read_sweep_data(root, keywords)` returns both circular-pump channels (c1, c2) in one call, the 423nm Γ-K P dataset has its pump polarization channels in separate `PumpRotator-160` and `PumpRotator-70` subdirectories that the generic search does not descend into correctly. The 31.95 eV window is further identified by scan number (`2318`) rather than an energy directory. Each channel must be loaded with an explicit `read_sweep_data(..., ["...", "PumpRotator-160"])[0]` call and assembled manually as `[c1, c2]` before passing to the prep pipeline. The other 423nm datasets (Γ-K S, Γ-M P/S) use standard `DataItem` + `load_raw` loading. See `_load_423nm_gk_p()` in `scripts/bc2_closeup_snapshots.py` for the reference implementation.

**`SweepData.angle_offset(degrees)` — context manager for momentum re-centering.** Every dataset has a small angular offset caused by the sample or detector not being perfectly aligned. To re-center the momentum axis, use the `angle_offset` context manager on the `SweepData` object before calling `display_sweep_data` with `angle_to_momentum=True`. It temporarily shifts `y_axis_minimum` by the given degrees, which feeds into the angle→momentum conversion inside `display_sweep_data`, then unconditionally restores the original value on exit. Example: `with sweep.angle_offset(2.5): display_sweep_data(sweep, ..., angle_to_momentum=True)`. Per-dataset offset constants are defined at the top of `scripts/bc2_closeup_snapshots.py`.

---

## What each script category does

| Script(s) | Purpose |
|---|---|
| `analysis_script_<nm>.py` | Top-level orchestrators — loads all geometries for one pump wavelength and produces pixel-resolved τ maps |
| `basic_analysis_template_<nm>_<dir>_<pol>.py` | Declarative snapshot system — generates CD + SUM ARPES images at specific time delays (e.g. −50 fs, 0 fs, +50 fs) using a `SubPlot`/`Plot` object model |
| `compare_all_wavelengths_decay_dynamics.py` | Loads all 5 pump wavelengths, computes τ on a shared color scale, overlays band structure — the main cross-wavelength comparison figure |
| `compare_400nm_460nm_decay_dynamics.py` | Same, restricted to 400 vs 460 nm |
| `polarization_decay_analysis.py` | Constructs a pixel-wise polarization metric (c1−c2)/(norm+c1+c2), computes its spatial correlation across neighbors, then fits τ to that correlation signal |
| `long_delays_analysis.py` | Focuses on 452 nm Γ-M data at delays up to 5000 fs; fits single-exponential decay pixel-by-pixel using `display_pixel_resolved_decay_times` |
| `roi_time_series_comparison.py` | Integrates counts in user-defined (energy, momentum) ROI boxes vs time for multiple datasets — directly compares dynamics at specific k-E points |
| `Bi2Se3_band_structure.py` | Standalone — plots the band structure and draws optical transition arrows for a given pump wavelength using `1240/λ` energy |

---

## Open to-do list (`data_manipulation_tasks.txt`)

priorty - 1 (lowest), 2 or 3 (highest)
items with no priority description should be considered lowest priority.

### Text Sections
- Introduction
    * PRIORITY 1
    * explanation on excitons + references
        -  where in the band structure the exciton population is expected and why?
        

### Corrections to Existing Items
- time dynamics of the polarization measure - clarify whether gamma-m or gamma-k
    * PRIORITY - 1
- τ comparison - 3 energy zones instead of 2
    * PRIORITY - 2
    * the color scale is too saturated at the bottom of the plot with higher energies (around 1eV-1.2eV). dividing 
    those plots to 3 zones instead of 2 might reveal more info.
    * talking about the Tau comparison in the `Comparing Decay Times` section inside the proj.tex.
- ~~Verify angle-to-momentum conversion~~ — math confirmed correct; fixed a plotting bug in `display_sweep_data` (`analysis_tools.py:1041`) where `extent` used the full unclipped momentum range instead of `grid_points_y` bounds, causing ~16% momentum scale inflation when `momentum_cutoff` < detector range

### New Items
- gauss exp fit + time series plot for the gamma-m blob
    * at energy range 0.49-0.61eV, momentum -0.75 to -0.5 (L) and +0.5 to +0.75 (R) Å⁻¹; probe S; 400nm and 461nm.
    * ~~closeup plots (SUM + CD) at times 0, +250, +400, +700 fs~~ — `scripts/gamma_m_blob_closeup_snapshots.py`
    * ~~ROI time series + Gauss-exp fit~~ — `scripts/gamma_m_blob_time_series.py`; τ ≈ 330–380 fs for both blobs/wavelengths
    * ~~pixel-resolved tau map~~ — `scripts/gamma_m_blob_tau_map.py`; fit from t > 150 fs; see project_summary.md for `display_pixel_resolved_decay_times` energy-axis gotcha
    * ~~added to proj.tex as `\subsubsection{Anomalous Life Time Localized Momenta State}`~~
    * ~~try comparing blobs gauss-exp fit to the fit of a different band at similar energy~~ — `scripts/gamma_m_blob_vs_center_time_series.py` (Γ-M blob vs Γ-point center) and `scripts/gamma_m_vs_k_blob_461nm.py` (same ROI in Γ-K vs Γ-M, 461nm); added to proj.tex
    * ~~expand time axis limits of the onset-aligned plot to longer delays~~ — COMPARISON_TIME_AXIS_MAX=2000 fs in all comparison scripts
- explaining blob analysis in the report
    * PRIORITY - 2
    * the report section currently describes the observations; it should be expanded with physical interpretation of the blobs — what state they may correspond to, why the lifetime is long, and how they relate to the photoluminescence hypothesis.
    * complete the interpretation of the different time series.
- 515 NM data - add to gamma-k/gamma-m comparisons over time
    * PRIORITY - 2
- make sure there are no transitions outside of the current momentum range
    * PRIORITY - 2
    * the current leading belief is that the light is emitted when electrons decay from band `BC2` (bulk conduction band 2, at around 2.3eV above fermi level) to the part of the first dirac cone that is slightly above the fermi level.
    * this conclusion relies on plots of possible transitions based on the band structure extracted from ARPES measurements. however those plots were for momenta between -0.75 to 0.75 (A^-1), better look at a broaded range of momenta if possible.
- focus on band `BC2` (called Bi2Se3_Conduction_2_3eV_Gamma_(M/K))
    * MAIN ITEM
    * PRIORITY - 3
    * track time dynamic of suspected photoemission band
    * close up of suspected photo emission band in a few time frames
    * in general - do a summary of the data on the top state
    * ~~BC2 ARPES close-up snapshots — 5 time delays (-50/0/+60/+250/+500 fs), energy zoom 1.9–2.6 eV, ±0.2 Å⁻¹, SUM + CD panels, 400 nm + 461 nm Γ-K; script: `bc2_closeup_snapshots.py`~~
        - DONE
    * ~~BC2 ROI time series + Gauss-exp fit — ROI (2.23–2.37 eV, ±0.132 Å⁻¹), compare 400 nm vs 461 nm Γ-K, fit to extract τ_decay and t_peak; script: `bc2_time_series.py`~~
        - DONE
    * ~~BC2-zoomed τ map — pixel-resolved decay times restricted to 2.0–2.6 eV energy range, BC2 band overlay, 400 nm + 461 nm Γ-K; script: `bc2_tau_map.py`~~
        - DONE
    * ~~close up snapshots - replicate to include all combinations - Gamma-M S/P, Gamma-K S/P.~~
        - DONE
    * 460nm data at 100fs and 150fs is very different from -50, 0, +50 fs.
    * 423nm data might need some normalization to look symmetrical in CD plots.
    * `BC2` closeup figures, at 461nm for times 100fs and 150fs might need different color scale to look better.
- examine EDCs in various times to check for excitonic population 2.25eV above the first dirac point.
    * PRIORITY - 2
    * the photoluminescence article is suggesting the light might be emitted from decaying excitons binding an electron and a hole between the two dirac points. if that would be the case we would be expecting to see in ARPES a signal of populated states one photon energy above the energy level of the theorized hole state.
    * to check for such a signal we want to look at EDCs around momentum=0, at various times (to see an increase at some point after t_0), and at various wavelengths.

### Extras
- print one graph of "before" and "after" not taking resonances into account
    * PRIORITY - 1
    * the graph type relevant for this is the display of decay time (tau) per pixel (energy-momentum). when fitting the time dependent data to decaying exponent starting from t_0, the results show areas with long taus (relative to their environment), looking like "hot spots" in the taus plot. and when cutting out the time points closest to t_0, starting the fit from ~50fs and onward, those "hot spots" are no longer there.
    * my belief is that those spots are there because of instant 2 photon processes making states from lower energies show in the measurement high above fermi level. and are thus not describing the dynamics of real excited electrons.
    * I want to be able to show how significant the difference is between the two kinds of plots (fitting including t_0 and not including t_0)
- for 400nm, 423nm, 460nm Gamma-M S/P and Gamma-K S/P do a CD closeup for the area of the dirac cone close to fermi level up to 0.75eV, at times -50, 0, 50, 100, 150, 250, 400 (fs).
    * PRIORITY - 2
    * it is possible that this CD signal is the one that actually matters, because it might have a significantly longer life time than the CD of the second dirac cone.
    * maybe add a graph of the polarization measure life time, similarly to the ambiguous attempt with the second dirac cone.
    * another interesting possibility - the significant CD signal seems to be at Gamma-M for this energy range.

### Extra Data
- 515nm Gamma-M probe S - blobs at ~0.5eV
- close up of band `BC2` at 515nm
- close up of band `BC2` at 423nm, 460nm, 400nm where better SNR or resolution is needed.
- band `BC2` deflection at 460nm, 400nm, 515nm if seems to produce good results.