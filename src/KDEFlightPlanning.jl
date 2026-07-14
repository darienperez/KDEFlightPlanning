"""
    KDEFlightPlanning

A Julia package for KDE-guided adaptive UAV flight planning, developed for:

> "KDE-Guided Offline Variable-Speed Flight Planning for UAV LiDAR in Forested Terrain"
> Darien D. Perez Martin, Adam G. Hunsaker, Jennifer M. Jacobs, May-Win Thein
> *Remote Sensing* (MDPI) — in preparation

## Pipeline overview (end-to-end)

    Orthomosaic image (GeoTIFF or raw array)
        │
        ▼  load_rgb_georef() or rgb_from_array()
    Matrix{RGB{Float32}}
        │
        ▼  rgb_to_lab()
    CIELAB channels (L, a, b) — Float32 matrices H×W
        │
        ▼  stack_features() → standardize_features!() → [pca_fit/pca_transform]
    Pixel feature matrix (n × d), optionally PCA-reduced
        │
        ▼  kmedoids_fit() / build_mask_from_image() / build_mask_autok()
    Cluster labels → labels_to_mask()
        │
        ▼  [optional: remove_small_components!, fill_small_holes!, binary_open_close!]
    Binary canopy mask (BitMatrix / RasterGrid)
        │
        ▼  build_density_surface()  →  kde_from_mask()
    KDE density surface (RasterGrid, normalised [0,1])
        │
        ▼  plan_mission() / generate_waypoints()
    Waypoints (Vector{Waypoint}) with speed assignments
        │
        ▼  coverage_ratio() / density_bin_summary()
    Coverage metrics  →  figures / tables

## Public naming

| Manuscript label                    | Strategy type               |
|-------------------------------------|------------------------------|
| Constant 2 m/s                      | `ConstantSpeed(v=2.0)`       |
| Constant 8 m/s                      | `ConstantSpeed(v=8.0)`       |
| KDE-guided variable speed (Gaussian)| `KDEGuidedSpeed(...)`        |
| KDE-guided Epanechnikov             | `KDEGuidedSpeed(...)`        |
| Curvature-spaced KDE-guided         | `CurvatureGuidedSpeed(...)` |

"Speed-aware" and "density-aware" are internal legacy terms; they are not
exposed in the public API.

## What is ported from CanopyDensity.jl

- `colorspace.jl`: `rgb_to_lab`, `is_grayscale`, `rgb_from_array`,
  `rgb_to_lab_array`, `lab_to_array`
- `features.jl`:   `stack_features`, `standardize_features!`, `local_var`,
  `grad_mag`, `FeatureProjector`, `feature_projector`, `apply_projector`,
  `build_projector`
- `pca.jl`:        `pca_fit`, `pca_transform`, `pca_explained`
- `clustering.jl`: `kmedoids_fit`, `choose_sample_indices`,
  `sample_distance_matrix`, `sweep_k_quality`, `choose_k`, `ClusterMetrics`
- `masking.jl`:    `labels_to_mask`, `mask_stats`, morphology stubs
- `kde.jl`:        `gaussian_kernel`, `epanechnikov_kernel`, `scotts_sigma`,
  `scott_sigma_indices`, `kde_from_mask`
- `pipeline.jl`:   `build_mask_from_image`, `build_mask_autok`,
  `build_density_surface`, `run_pipeline`, `run_full_pipeline`,
  `axes_from_geotransform`, `pixel_axes`, `load_rgb_georef`

## Quick start (synthetic data — no GeoTIFF needed)

```julia
using KDEFlightPlanning

# 1. Synthetic RGB image array (H=40, W=40, 3 channels, channel-last)
arr = rand(UInt8, 40, 40, 3)
L, a, b = rgb_to_lab_array(arr)

# 2. Feature matrix and k-medoids mask
X = Float64.(stack_features(L, a, b))
standardize_features!(X)
labels, _, _ = kmedoids_fit(X; k=2, seed=42)
mask_bm = labels_to_mask(labels, 40, 40; tree_labels=[1])
xs, ys  = pixel_axes(40, 40)
mask    = RasterGrid(Float64.(mask_bm), xs, ys)

# 3. KDE density surface
cfg   = PipelineConfig(kde_kernel=:epanechnikov)
dens, _ = build_density_surface(mask, cfg)

# 4. Plan mission
strategy = KDEGuidedSpeed(dens; vmin=2.0, vmax=8.0)
config   = FlightConfig(strategy, 80.0, "KDE-guided Epanechnikov";
                        kernel=:epanechnikov, line_spacing=20.0)
wps = plan_mission(dens, config; seconds_per_wp=1.0, spacing_min=2.0)
println("Generated \$(length(wps)) waypoints")

# 5. Coverage metric
cr = coverage_ratio(ones(Int, 40, 40))

# 6. One-call pipeline (image array → waypoints)
mask2, dens2, wps2, info = run_full_pipeline(arr;
    k=2, tree_labels=[1], flight_cfg=config)
```
"""
module KDEFlightPlanning

# ===========================================================================
# Centralised `using` statements (idiomatic: all in the top-level module)
# ===========================================================================

# --- Standard library ---
using Statistics: mean, std, median, median!, quantile
using LinearAlgebra: norm
using Random
using Printf: @printf, @sprintf
using Dates

# --- Registered packages (all listed in [deps] of Project.toml) ---
using CSV
using DataFrames
using FFTW
using JSON
using JLD2
using CairoMakie
using ColorSchemes
using Colors: Lab, RGB, convert
using ColorTypes
using Clustering
using Distances
using MultivariateStats
using ArchGDAL
using LASDatasets

# ===========================================================================
# Source files — load order matters
# ===========================================================================
include("inputs.jl")      # RunInputs, load_inputs (TOML), validate,
                           # inputs_template, MissingInputError
include("types.jl")       # RasterGrid, Waypoint, SpeedStrategy hierarchy,
                           # LawnmowerSpec, FlightConfig, PipelineConfig, Kernel2D
include("raster.jl")      # make_grid, rescale_density!, bilinear_interp,
                           # sample_density, gradient_*, sample_line, synthetic grids
include("kde.jl")          # gaussian_kernel, epanechnikov_kernel,
                           # scotts_sigma, scott_sigma_indices, kde_from_mask
include("colorspace.jl")   # rgb_to_lab, is_grayscale, rgb_from_array,
                           # rgb_to_lab_array, lab_to_array
include("features.jl")     # stack_features, standardize_features!,
                           # local_var, grad_mag, FeatureProjector, build_projector
include("pca.jl")          # pca_fit, pca_transform, pca_explained
include("clustering.jl")   # kmedoids_fit, sweep_k_quality, choose_k, ClusterMetrics
include("masking.jl")      # labels_to_mask, mask_stats, morphology stubs
include("speedmap.jl")     # assign_speed, speed_bounds, is_monotone_decreasing
include("path.jl")         # lawnmower_from_extents, annotate_line_ids!
include("waypoints.jl")    # generate_waypoints, plan_mission, smooth_speeds!
include("metrics.jl")      # coverage_ratio, density_bin_summary, coverage_profile
include("count_statistics.jl") # CountKey, CountSummary, summary_statistics,
                           # gini_coefficient, morans_i, load_counts_json,
                           # build_cover_masks, summarize_counts_json,
                           # write_summary_csv, DURHAM_COVER_N
include("bootstrap.jl")    # bootstrap_cr, bootstrap_cr_difference,
                           # bootstrap_cr_table, bootstrap_cr_difference_table,
                           # BootstrapCI, Comparison, default_narrative_comparisons,
                           # manuscript_mission_label
include("io.jl")           # write/read_waypoints_csv, write_ugcs_csv
include("pipeline.jl")     # build_density_surface, build_mask_from_image,
                           # build_mask_autok, run_pipeline, run_full_pipeline
include("trajectory.jl")   # derive_planned_lines, clean_survey_segments,
                           # ground_speed, assign_to_planned_lines,
                           # contiguous_segments, mission_speed_bounds,
                           # tracking_summary, tracking_line_summary,
                           # tracking_segment_summary, mission_time_metrics
include("linescan.jl")     # decode_count_grid, load_count_grids,
                           # actual_line_extents, common_x_overlap,
                           # line_scan_cover_metrics, single_line_cover_metrics
include("metadata.jl")    # export_planning_metadata, save_cluster_overlays,
                           # require_tree_labels, interactive_tree_labels,
                           # waypoint_spacing_note
include("kde_strata.jl")  # kde_cr_by_quantile_bins, kde_strata_table,
                           # export_kde_strata_csv, kde_strata_narrative
include("kde_surface_io.jl")  # GeoTransform, KDESurface (loaded before geotiff_io)
                               # NearestNeighbor, Bilinear, GT_NATIVE, GT_COUNTGRID,
                               # CRS_DURHAM, validate_kde_range,
                               # save_kde_surface, load_kde_surface,
                               # save_kde_surface_csv, load_kde_surface_csv,
                               # resample_to_count_grid, resample_to_image_grid
include("kde_density_classes.jl")  # ThresholdMethod, OtsuThreshold, QuantileThreshold,
                                    # ManualThreshold, DensityClassResult,
                                    # KDE_CLASS_*, compute_kde_thresholds,
                                    # assign_kde_density_classes, kde_class_support,
                                    # kde_class_cr, kde_class_summary,
                                    # export_kde_class_summary_csv, kde_class_narrative
include("figures.jl")              # ALL publication figures (single figure module):
                                    # trajectory/bootstrap figures (FIG_BG, MISSION_COLORS,
                                    # MISSION_ORDER, _theme_axis!, fig_forest_line_scan_differences,
                                    # fig_bootstrap_cr_main, …) PLUS the KDE-surface figures
                                    # merged from the former figures_kde.jl:
                                    # fig_kde_surface_heatmap (KDESurface | matrix),
                                    # fig_kde_class_map (DensityClassResult),
                                    # fig_kde_class_cr (DensityClassResult+dict | DataFrame),
                                    # fig_kde_strata_within_cover_cr (DataFrame)
include("along_track.jl") # along_track_cr, split_at_gaps,
                           # multi_mission_along_track, export_along_track_csv
include("geotiff_io.jl")  # GeoRasterStack, load_rgb_geotiff, read_band,
                           # raster_extents, write_single_band_geotiff
include("lidar_counts.jl") # LASMission, crop_las, bin_to_count_grid,
                            # build_count_grids, percent_density_table,
                            # cover_stats_table, write_count_grids_json,
                            # STANDARD_MISSION_PLAN
include("reports.jl")     # stage-by-stage validation visuals + run manifest

# ===========================================================================
# Exports
# ===========================================================================

# --- Run configuration (inputs.jl) ---
export RunInputs, load_inputs, inputs_template, MissingInputError

# --- GeoTIFF I/O (geotiff_io.jl) ---
export GeoRasterStack, load_rgb_geotiff, read_band, raster_extents
export write_single_band_geotiff, axes_from_georasterstack

# --- LiDAR counts (lidar_counts.jl) ---
export LASMission, STANDARD_MISSION_PLAN, DEFAULT_PD_BINS
export read_classes_geotiff, crop_las, ground_points
export bin_to_count_grid, build_count_grids
export percent_density_table, cover_stats_table
export write_count_grids_json
export is_kernel_agnostic

# --- Reports / visuals (reports.jl) ---
export report_ingest_preview, report_geotiff_metadata
export report_lab_channels, report_pca_explained
export report_cluster_metrics_sweep, report_cluster_overlays
export report_cluster_lab_summary, report_tree_label_decision
export report_kde_density, report_speed_map
export report_waypoints_overlay
export report_run_manifest, report_provenance, report_run_md
export report_proposed_artifacts_checklist

# --- Types ---
export RasterGrid, Waypoint, LawnmowerSpec, FlightConfig, PipelineConfig
export SpeedStrategy, ConstantSpeed, KDEGuidedSpeed, CurvatureGuidedSpeed
export FullKernel, Kernel2D
export FeatureProjector
export ClusterMetrics

# --- Colorspace (CanopyDensity port) ---
export rgb_to_lab, is_grayscale
export rgb_from_array, rgb_to_lab_array, lab_to_array

# --- Features (CanopyDensity port) ---
export stack_features, standardize_features!
export local_var, grad_mag
export feature_projector, apply_projector, build_projector

# --- PCA (CanopyDensity port) ---
export pca_fit, pca_transform, pca_explained

# --- Clustering (CanopyDensity port) ---
export kmedoids_fit
export choose_sample_indices, sample_distance_matrix
export sweep_k_quality, choose_k
export export_cluster_metrics_csv

# --- Masking (CanopyDensity port) ---
export labels_to_mask, mask_stats
export remove_small_components!, fill_small_holes!, binary_open_close!

# --- Speed mapping ---
export assign_speed, speed_bounds, is_monotone_decreasing

# --- Raster / grid utilities ---
export make_grid, rescale_density!, density_minmax
export bilinear_interp, sample_density
export gradient_at, gradient_magnitude, directional_curvature
export sample_line
export synthetic_gaussian_grid, uniform_grid, synthetic_mask_grid

# --- KDE ---
export gaussian_kernel, epanechnikov_kernel
export scotts_sigma, scott_sigma_indices
export kde_from_mask

# --- Path generation ---
export lawnmower_from_extents, annotate_line_ids!

# --- Waypoint generation ---
export generate_waypoints, plan_mission, smooth_speeds!, filter_bbox

# --- Count statistics (count_statistics.jl) ---
export CountKey, CountRecord, CountSummary
export return_label, mission_label, kernel_label, cover_label
export gini_coefficient, morans_i
export summary_statistics
export load_counts_json, build_cover_masks
export summarize_counts_json, write_summary_csv
export DURHAM_COVER_N

# --- Block bootstrap (bootstrap.jl) ---
export BootstrapCI, Comparison
export bootstrap_cr, bootstrap_cr_difference
export bootstrap_cr_table, bootstrap_cr_difference_table
export default_narrative_comparisons
export manuscript_mission_label

# --- Metrics ---
export coverage_ratio, percent_in_bin, density_bin_summary
export per_cover_summary, coverage_profile, flight_line_profile

# --- I/O ---
export write_waypoints_csv, read_waypoints_csv
export write_coverage_csv, read_coverage_csv
export write_ugcs_csv, write_xy_speed_csv

# --- Pipeline (CanopyDensity + FlightPlanning combined) ---
export build_density_surface, run_pipeline
export build_mask_from_image, build_mask_autok
export build_mask_from_image_strided
export run_full_pipeline
export axes_from_geotransform, geotransform_resolution, pixel_axes
export load_rgb_georef

# --- Trajectory analysis ---
export derive_planned_lines, ground_speed, assign_to_planned_lines
export contiguous_segments, mission_speed_bounds
export clean_survey_segments
export fitted_line_centers
export tracking_summary, tracking_line_summary, tracking_segment_summary
export mission_time_metrics

# --- Line-scan ---
export SCAN_NROWS, SCAN_NCOLS, COVER_KEYS, COVER_LABELS
export decode_count_grid, load_count_grids
export actual_line_extents, common_x_overlap
export line_scan_cover_metrics, single_line_cover_metrics

# --- Planning metadata ---
export export_planning_metadata, waypoint_spacing_note
export save_cluster_overlays, require_tree_labels, interactive_tree_labels

# --- KDE-density-stratified coverage (secondary mechanism) ---
export kde_cr_by_quantile_bins, kde_strata_table
export kde_strata_within_cover      # fixed within-cover bins (Bug-B2/B3/B4 fix)
export export_kde_strata_csv, kde_strata_narrative

# --- True KDE surface I/O: types + JLD2 + CSV ---
# Types (GeoTransform, KDESurface, ResampleMethod tokens)
export GeoTransform, KDESurface
export ResampleMethod, NearestNeighbor, Bilinear
# Study-site defaults
export GT_NATIVE, GT_COUNTGRID, CRS_DURHAM
# Validation
export validate_kde_range
# JLD2 I/O
export save_kde_surface, load_kde_surface
# CSV+JSON I/O
export save_kde_surface_csv, load_kde_surface_csv
# Resampling (dispatched on KDESurface or bare matrix + GeoTransform)
export resample_to_count_grid
# Image-grid resampling (screenshot-derived KDE path — no geotransform)
export resample_to_image_grid

# --- KDE-derived density classes (mechanism diagnostics; distinct from GLI ground truth) ---
# Threshold method types
export ThresholdMethod, MultiOtsuThreshold, OtsuThreshold, QuantileThreshold, ManualThreshold
# Result type
export DensityClassResult
# Class labels
export KDE_CLASS_FIELD, KDE_CLASS_DECIDUOUS, KDE_CLASS_CONIFEROUS, KDE_CLASS_LABELS
# Computation
export compute_kde_thresholds
export assign_kde_density_classes
# Result inspection (dispatched on DensityClassResult or bare matrix)
export kde_class_support, kde_class_cr
# Multi-mission table
export kde_class_summary, export_kde_class_summary_csv
# Narrative (dispatched on DensityClassResult or DataFrame)
export kde_class_narrative

# --- KDE figures (merged into figures.jl) — dispatched on typed structs ---
export fig_kde_surface_heatmap      # KDESurface or bare matrix overload
export fig_kde_class_map            # DensityClassResult
export fig_kde_class_cr             # DensityClassResult+dict  or  DataFrame
export fig_kde_strata_within_cover_cr  # DataFrame from kde_strata_within_cover

# --- Along-track coverage analysis ---
export along_track_cr, split_at_gaps
export multi_mission_along_track, export_along_track_csv
export along_track_weakness_note

end # module KDEFlightPlanning
