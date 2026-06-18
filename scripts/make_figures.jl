#!/usr/bin/env julia
"""
    scripts/make_figures.jl

Regenerate all publication-ready figures from trajectory CSV outputs.
Run this AFTER scripts/trajectory_analysis.jl has been executed.

Usage
-----
  julia --project=. scripts/make_figures.jl [CONFIG.toml] [OUTPUT_DIR] [TRAJ_DIR]

  CONFIG.toml RunInputs TOML config. Drives data discovery
              ([paths].counts_json, [paths.trajectory] keys). Default: bundled
              data/ground_truth/ files.
  OUTPUT_DIR  where to write figure files; default = <config outdir>/figures
  TRAJ_DIR    where trajectory CSVs live;   default = <config outdir>/trajectory

Figures generated
-----------------
  actual_trajectory_line_scan_forest_differences.png/.pdf
  line2_actual_overlap_detail.png/.pdf
  tracking_metrics_refined.png/.pdf
  bootstrap_cr_summary.png           (if bootstrap CSV is present)

All figures also output as high-resolution PNG (2× px_per_unit) and PDF.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CairoMakie
using CSV, DataFrames, Statistics

include(joinpath(@__DIR__, "ground_truth.jl"))

# ─────────────────────────────────────────────────────────────
# Paths (config-driven; no hardcoded space_files)
# ─────────────────────────────────────────────────────────────
const _GT       = resolve_ground_truth(get(ARGS, 1, ""))
const OUT_DIR   = get(ARGS, 2, _GT.figures_dir)
const TRAJ_DIR  = get(ARGS, 3, _GT.trajectory_dir)

mkpath(OUT_DIR)

# Resolve the three flown-track trajectory CSVs from [paths.trajectory].
function _traj_csv(keys::Vector{Symbol}, human::AbstractString)
    if _GT.inputs !== nothing
        for k in keys
            if haskey(_GT.inputs.trajectory, k) && isfile(_GT.inputs.trajectory[k])
                return _GT.inputs.trajectory[k]
            end
        end
    end
    error("Trajectory CSV '$human' not configured/found.\n" *
          "  Set [paths.trajectory] keys in your config (one of: $(keys)).")
end

function _traj(name)
    p = joinpath(TRAJ_DIR, name)
    isfile(p) && return p
    error("Expected trajectory CSV not found: $p\n  → Run scripts/trajectory_analysis.jl first.")
end

# ─────────────────────────────────────────────────────────────
# Load trajectory CSVs
# ─────────────────────────────────────────────────────────────
println("Loading trajectory CSVs from: $TRAJ_DIR")

cover_df   = CSV.read(_traj("actual_trajectory_line_cover_metrics.csv"),  DataFrame)
profile_df = CSV.read(_traj("actual_trajectory_line_column_profiles.csv"), DataFrame)
cover2_df  = CSV.read(_traj("line2_actual_overlap_cover_metrics.csv"),     DataFrame)
profile2_df= CSV.read(_traj("line2_actual_overlap_column_profile.csv"),    DataFrame)
ext2_df    = CSV.read(_traj("line2_actual_overlap_extents.csv"),           DataFrame)
summary_df = CSV.read(_traj("tracking_summary_metrics_refined.csv"),       DataFrame)
line_df    = CSV.read(_traj("tracking_line_metrics_refined.csv"),          DataFrame)
planned_lines = CSV.read(_traj("planned_lines_from_waypoints.csv"),        DataFrame)

# ─────────────────────────────────────────────────────────────
# Load and clean trajectories (needed for map + boxplot)
# ─────────────────────────────────────────────────────────────
println("Loading and cleaning trajectories …")

trajectory_files = Dict(
    "Const. 2 m/s"              => _traj_csv([:const2, :mission2mps], "constant_speed_2mps__trajectory.csv"),
    "KDE-guided (Epanechnikov)" => _traj_csv([:missionE, :Ed, :E], "E_density_aware__trajectory.csv"),
    "Const. 8 m/s"              => _traj_csv([:const8, :mission8mps], "constant_speed_8mps__trajectory.csv"),
)

processed = Dict{String,DataFrame}()
for (mission, path) in trajectory_files
    print("  Cleaning $mission … ")
    flush(stdout)
    raw = CSV.read(path, DataFrame)
    survey, _ = clean_survey_segments(raw, mission, planned_lines)
    processed[mission] = survey
    println("$(nrow(survey)) samples, $(length(unique(survey.segment_id))) segs")
end

# ─────────────────────────────────────────────────────────────
# Load count grids (needed for heatmap figure)
# ─────────────────────────────────────────────────────────────
println("Loading counts.json …")
counts_json = require_ground_truth(_GT, :counts_json, "counts.json")
count_grids = load_count_grids(counts_json)
println("  Loaded $(length(count_grids)) count-grid entries")

# Figure functions live inside the package (figures.jl is included by the
# module). They are not exported, so reference them via the module namespace
# rather than re-including the source (which would redefine methods).
const KFP = KDEFlightPlanning

FORMATS = ["png", "pdf"]

# ─────────────────────────────────────────────────────────────
# Figure 1: Forest line scan differences
# ─────────────────────────────────────────────────────────────
println("\n=== Figure 1: Forest line scan differences ===")
paths1 = KFP.fig_forest_line_scan_differences(cover_df;
    out_dir = OUT_DIR,
    formats = FORMATS,
)
for p in paths1
    println("  → $p")
end

# ─────────────────────────────────────────────────────────────
# Figure 2: Line 2 actual-overlap detail
# ─────────────────────────────────────────────────────────────
println("\n=== Figure 2: Line 2 actual-overlap detail ===")
paths2 = KFP.fig_line2_overlap_detail(
    cover2_df, profile2_df, ext2_df,
    count_grids, processed;
    out_dir = OUT_DIR,
    formats = FORMATS,
)
for p in paths2
    println("  → $p")
end

# ─────────────────────────────────────────────────────────────
# Figure 3: Tracking metrics
# ─────────────────────────────────────────────────────────────
println("\n=== Figure 3: Tracking metrics ===")
paths3 = KFP.fig_tracking_metrics(
    summary_df, line_df, processed;
    out_dir = OUT_DIR,
    formats = FORMATS,
)
for p in paths3
    println("  → $p")
end

# ─────────────────────────────────────────────────────────────
# Figure 4: Bootstrap CI summary — simplified (if legacy CSV exists)
# ─────────────────────────────────────────────────────────────
bootstrap_csv = joinpath(TRAJ_DIR, "bootstrap_cr_summary.csv")
paths4 = String[]
if isfile(bootstrap_csv)
    println("\n=== Figure 4: Bootstrap CI summary (legacy) ===")
    paths4 = KFP.fig_bootstrap_ci_summary(bootstrap_csv; out_dir=OUT_DIR)
    for p in paths4; println("  → $p"); end
else
    println("\n  (Skipping legacy bootstrap figure — no bootstrap_cr_summary.csv found)")
end

# ─────────────────────────────────────────────────────────────
# Figure 5: Main manuscript bootstrap CI figure
#   Reads from output/bootstrap/ (generated by run_bootstrap.jl)
# ─────────────────────────────────────────────────────────────
paths5 = String[]
bootstrap_dir = _GT.bootstrap_dir
cr_csv_main   = joinpath(bootstrap_dir, "bootstrap_cr_cis_all36.csv")
diff_csv_main = joinpath(bootstrap_dir, "bootstrap_cr_difference_cis.csv")

if isfile(cr_csv_main) && isfile(diff_csv_main)
    println("\n=== Figure 5: Main manuscript bootstrap CI figure ===")
    paths5 = KFP.fig_bootstrap_cr_main(
        cr_csv_main, diff_csv_main;
        out_dir = OUT_DIR,
        formats = FORMATS,
    )
    for p in paths5; println("  → $p"); end
else
    println("\n  (Skipping main bootstrap figure — run scripts/run_bootstrap.jl first)")
    println("  Expected: $cr_csv_main")
    println("  Expected: $diff_csv_main")
end

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
println("\n=== All figures written to: $OUT_DIR ===")
all_paths = vcat(paths1, paths2, paths3, paths4, paths5)
println("Files:")
for p in all_paths
    sz = round(stat(p).size / 1024; digits=1)
    println("  $(basename(p))  ($(sz) kB)")
end
