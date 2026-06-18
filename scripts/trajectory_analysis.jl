#!/usr/bin/env julia
"""
    scripts/trajectory_analysis.jl

Reproducible script that regenerates all trajectory-analysis CSV outputs
from the Space data files. Running this script reproduces:

  tracking_summary_metrics_refined.csv
  tracking_line_metrics_refined.csv
  tracking_segment_metrics_refined.csv
  planned_lines_from_waypoints.csv
  actual_trajectory_line_extents.csv
  actual_trajectory_line_scan_summary.csv
  actual_trajectory_line_cover_metrics.csv
  actual_trajectory_line_cover_metrics_wide.csv
  actual_trajectory_line_column_profiles.csv
  line2_actual_overlap_cover_metrics.csv
  line2_actual_overlap_column_profile.csv
  line2_actual_overlap_extents.csv

Usage
-----
  julia --project=. scripts/trajectory_analysis.jl [CONFIG.toml] [OUTPUT_DIR]

  CONFIG.toml RunInputs TOML config. Data discovery is driven by it:
                [paths].counts_json        — count grids
                [paths].gli_class_raster   — optional 263×324 GLI class raster
                [paths.waypoints].Ed       — E density-aware waypoints CSV
                [paths.trajectory].{const2,missionE,const8} — flown-track CSVs
  OUTPUT_DIR  (optional) where to write CSVs; default = <config outdir>/trajectory

Data files required:
  counts.json + E_density_aware__waypoints_xy.csv are bundled under
  data/ground_truth/. The three large flown-track trajectory CSVs are NOT
  bundled (tens of MB each) and must be supplied via [paths.trajectory] keys
  (const2 / missionE / const8). Without them this stage cannot run; the unified
  entry point (scripts/run_pipeline.jl) skips it with a clear message.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CSV, DataFrames, Statistics, Printf
using FileIO, ImageIO, ColorTypes

include(joinpath(@__DIR__, "ground_truth.jl"))

# ---------------------------------------------------------------------------
# Paths (config-driven; no hardcoded space_files)
# ---------------------------------------------------------------------------

const _GT      = resolve_ground_truth(get(ARGS, 1, ""))
const OUT_DIR  = get(ARGS, 2, _GT.trajectory_dir)
const GLI_CLASS_PATH = something(find_ground_truth(_GT, :gli_class_raster), "")
mkpath(OUT_DIR)

# Resolve the three flown-track trajectory CSVs from [paths.trajectory].
# Accepts the canonical keys used in config/run_durham.toml.
function _trajectory_path(gt::GroundTruth, keys::Vector{Symbol}, human::AbstractString)
    if gt.inputs !== nothing
        for k in keys
            if haskey(gt.inputs.trajectory, k) && isfile(gt.inputs.trajectory[k])
                return gt.inputs.trajectory[k]
            end
        end
    end
    error("Trajectory CSV '$human' not configured/found.\n" *
          "  Set [paths.trajectory] keys in your config (one of: $(keys)).\n" *
          "  These large flown-track CSVs are intentionally NOT bundled.")
end

function _class_value(px)
    if px isa Colorant
        return round(Int, Float64(red(px)) * 255)
    elseif px isa Number
        x = Float64(px)
        return round(Int, x <= 1 ? x * 255 : x)
    else
        error("Unsupported GLI pixel type: $(typeof(px))")
    end
end

function load_gli_cover_masks(path::AbstractString)
    isempty(path) && return nothing
    isfile(path) || error("GLI_CLASS_PATH does not exist: $path")
    img = FileIO.load(path)
    expected_size = (SCAN_NROWS, SCAN_NCOLS)
    size(img) == expected_size ||
        error("GLI class raster has size $(size(img)); expected $(expected_size)")

    classes = Matrix{Int}(undef, size(img)...)
    for I in CartesianIndices(img)
        classes[I] = _class_value(img[I])
    end

    # The attached GLI image is in display/image row order; decoded count grids
    # use row 1 as geographic north. Flip rows so masks align with count grids.
    classes = reverse(classes; dims = 1)

    return Dict(
        "decid" => classes .== 0,
        "conif" => classes .== 1,
        "field" => classes .== 2,
    )
end

# ---------------------------------------------------------------------------
# Mission definitions
# (label, counts.json mission key, counts.json kernel key)
# ---------------------------------------------------------------------------
const MISSIONS = [
    ("Const. 2 m/s",              "const2",  "NA"),
    ("KDE-guided (Epanechnikov)", "density", "E"),
    ("Const. 8 m/s",              "const8",  "NA"),
]
const MISSION_LABELS = [m[1] for m in MISSIONS]

# ---------------------------------------------------------------------------
# 1. Derive planned lines from waypoints
# ---------------------------------------------------------------------------
println("=== Step 1: Planned lines from waypoints ===")
waypoint_csv  = require_ground_truth(_GT, :waypoints_xy, "E_density_aware__waypoints_xy.csv")
planned_lines = derive_planned_lines(waypoint_csv)
println("  Found $(nrow(planned_lines)) planned lines")
CSV.write(joinpath(OUT_DIR, "planned_lines_from_waypoints.csv"), planned_lines)
println("  Wrote planned_lines_from_waypoints.csv")

# ---------------------------------------------------------------------------
# 2. Load and clean trajectories
# ---------------------------------------------------------------------------
println("\n=== Step 2: Clean survey-leg trajectories ===")
trajectory_files = Dict(
    "Const. 2 m/s"              => _trajectory_path(_GT, [:const2, :mission2mps], "constant_speed_2mps__trajectory.csv"),
    "KDE-guided (Epanechnikov)" => _trajectory_path(_GT, [:missionE, :Ed, :E], "E_density_aware__trajectory.csv"),
    "Const. 8 m/s"              => _trajectory_path(_GT, [:const8, :mission8mps], "constant_speed_8mps__trajectory.csv"),
)

processed = Dict{String,DataFrame}()
for (mission, path) in trajectory_files
    print("  Cleaning $mission … ")
    flush(stdout)
    raw = CSV.read(path, DataFrame)
    survey, _ = clean_survey_segments(raw, mission, planned_lines)
    processed[mission] = survey
    println("$(nrow(survey)) survey samples, $(length(unique(survey.segment_id))) segments")
end

# ---------------------------------------------------------------------------
# 3. Tracking metrics
# ---------------------------------------------------------------------------
println("\n=== Step 3: Tracking metrics ===")

summary_df  = tracking_summary(processed)
line_df     = tracking_line_summary(processed)
segment_df  = tracking_segment_summary(processed)

CSV.write(joinpath(OUT_DIR, "tracking_summary_metrics_refined.csv"),  summary_df)
CSV.write(joinpath(OUT_DIR, "tracking_line_metrics_refined.csv"),     line_df)
CSV.write(joinpath(OUT_DIR, "tracking_segment_metrics_refined.csv"),  segment_df)
println("  Wrote tracking_summary_metrics_refined.csv ($(nrow(summary_df)) rows)")
println("  Wrote tracking_line_metrics_refined.csv ($(nrow(line_df)) rows)")
println("  Wrote tracking_segment_metrics_refined.csv ($(nrow(segment_df)) rows)")

# Pretty-print summary
println("\n  Mission summary:")
for row in eachrow(summary_df)
    println("    $(row.mission): $(row.survey_samples) samples, speed_median=$(round(row.speed_median_mps; digits=2)) m/s, track_rms=$(round(row.tracking_error_rms_m; digits=3)) m")
end

# ---------------------------------------------------------------------------
# 4. Actual line extents
# ---------------------------------------------------------------------------
println("\n=== Step 4: Actual trajectory line extents ===")
extents_df = actual_line_extents(processed, planned_lines)
CSV.write(joinpath(OUT_DIR, "actual_trajectory_line_extents.csv"), extents_df)
println("  Wrote actual_trajectory_line_extents.csv ($(nrow(extents_df)) rows)")

# ---------------------------------------------------------------------------
# 5. All-line scan: cover metrics and column profiles
# ---------------------------------------------------------------------------
println("\n=== Step 5: All-line scan (cover metrics and profiles) ===")
counts_json = require_ground_truth(_GT, :counts_json, "counts.json")
println("  Loading counts.json …")
count_grids = load_count_grids(counts_json)
println("  Loaded $(length(count_grids)) count-grid entries")

cover_masks = load_gli_cover_masks(GLI_CLASS_PATH)
if isnothing(cover_masks)
    @warn "No GLI_CLASS_PATH supplied; line-scan CR denominators will use return-inferred support."
else
    println("  Loaded GLI class masks for fixed line-scan CR denominators:")
    for cover in COVER_KEYS
        println("    $(COVER_LABELS[cover]): $(count(cover_masks[cover])) cells")
    end
end

scan_df, cover_df, wide_df, profile_df = line_scan_cover_metrics(
    count_grids, processed, planned_lines, MISSIONS;
    half_band_rows = 20,
    cover_masks = cover_masks,
)

# Merge best_forest ranking into scan_df
forest = wide_df[map(c -> c ∈ ["Deciduous", "Coniferous"], wide_df.cover), :]
if !isempty(forest) && hasproperty(forest, :kde_minus_const2) && hasproperty(forest, :kde_minus_const8)
    forest = copy(forest)
    forest[!, :abs_min_gain] = min.(forest.kde_minus_const2, forest.kde_minus_const8)
    best = combine(groupby(forest, :line_number)) do g
        idx = argmax(g.abs_min_gain)
        DataFrame(
            best_forest_cover    = [g.cover[idx]],
            best_forest_min_gain = [g.abs_min_gain[idx]],
        )
    end
    scan_df = leftjoin(scan_df, best; on = :line_number)
end

CSV.write(joinpath(OUT_DIR, "actual_trajectory_line_scan_summary.csv"),         scan_df)
CSV.write(joinpath(OUT_DIR, "actual_trajectory_line_cover_metrics.csv"),         cover_df)
CSV.write(joinpath(OUT_DIR, "actual_trajectory_line_cover_metrics_wide.csv"),    wide_df)
CSV.write(joinpath(OUT_DIR, "actual_trajectory_line_column_profiles.csv"),       profile_df)
println("  Wrote actual_trajectory_line_scan_summary.csv ($(nrow(scan_df)) rows)")
println("  Wrote actual_trajectory_line_cover_metrics.csv ($(nrow(cover_df)) rows)")
println("  Wrote actual_trajectory_line_cover_metrics_wide.csv ($(nrow(wide_df)) rows)")
println("  Wrote actual_trajectory_line_column_profiles.csv ($(nrow(profile_df)) rows)")

# ---------------------------------------------------------------------------
# 6. Line 2 detail
# ---------------------------------------------------------------------------
println("\n=== Step 6: Line 2 actual-overlap detail ===")
ext2_df, cover2_df, profile2_df = single_line_cover_metrics(
    count_grids, processed, planned_lines, MISSIONS, 2;
    half_band_rows = 20,
    cover_masks = cover_masks,
)

CSV.write(joinpath(OUT_DIR, "line2_actual_overlap_extents.csv"),        ext2_df)
CSV.write(joinpath(OUT_DIR, "line2_actual_overlap_cover_metrics.csv"),  cover2_df)
CSV.write(joinpath(OUT_DIR, "line2_actual_overlap_column_profile.csv"), profile2_df)
println("  Wrote line2_actual_overlap_extents.csv")
println("  Wrote line2_actual_overlap_cover_metrics.csv")
println("  Wrote line2_actual_overlap_column_profile.csv")

# Print Line 2 cover metrics table
println("\n  Line 2 cover metrics:")
for row in eachrow(cover2_df)
    println("    $(row.cover) | $(row.mission): CR=$(round(row.coverage_ratio*100; digits=2))%")
end

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
println("\n=== Done. All CSVs written to: $OUT_DIR ===")

# ---------------------------------------------------------------------------
# 7. Mission time metrics
# ---------------------------------------------------------------------------
# NOTE: only raw trajectory CSVs and the `processed` Dict (from clean_survey_segments)
# are used here.  No *_with_line_id.csv files are read.
println("\n=== Step 7: Mission time metrics ===")

raw_trajectory_files = trajectory_files

mission_time_df = mission_time_metrics(raw_trajectory_files, processed)

CSV.write(joinpath(OUT_DIR, "mission_time_summary.csv"), mission_time_df)
println("  Wrote mission_time_summary.csv ($(nrow(mission_time_df)) rows)")

# --- Primary reporting: cleaned-segment duration ---
println("\n  Mission time summary (cleaned-segment duration = sum of retained leg durations):")
for row in eachrow(mission_time_df)
    @printf("    %-30s  raw=%.1f min  cleaned_seg=%.1f min  wall_span=%.1f min  frac_raw=%.1f%%\n",
        row.mission,
        row.raw_duration_min,
        row.cleaned_segment_duration_min,
        row.survey_wall_clock_span_min,
        row.cleaned_segment_fraction_raw * 100,
    )
end

# --- Time savings vs. Const. 2 m/s based on cleaned_segment_duration ---
const2_row = filter(r -> r.mission == "Const. 2 m/s", mission_time_df)
if !isempty(const2_row)
    t_const2_seg = only(const2_row).cleaned_segment_duration_min
    t_const2_raw = only(const2_row).raw_duration_min
    println("\n  Cleaned-segment time savings vs. Const. 2 m/s:")
    for row in eachrow(mission_time_df)
        row.mission == "Const. 2 m/s" && continue
        seg_savings = t_const2_seg - row.cleaned_segment_duration_min
        raw_savings = t_const2_raw - row.raw_duration_min
        @printf("    %-30s  cleaned-seg saves %.1f min (%.0f%%)  |  raw saves %.1f min (%.0f%%)\n",
            row.mission,
            seg_savings, seg_savings / t_const2_seg * 100,
            raw_savings, raw_savings / t_const2_raw * 100)
    end
end
