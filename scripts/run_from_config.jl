"""
    scripts/run_from_config.jl

Thin orchestrator: loads a TOML config (`RunInputs`) and drives the full
all-Julia pipeline, writing per-stage validation artefacts under the
configured `outdir`.

Usage:
    julia --project=. scripts/run_from_config.jl <path-to-config.toml>

Inputs (from TOML):
    [paths].rgb           — required GeoTIFF orthomosaic
    [paths].gli           — optional GLI cover-class GeoTIFF
    [paths.lidar]         — optional dict of LAS missions
    [paths.trajectory]    — optional dict of flown-track CSVs
    [paths.waypoints]     — optional dict of planned-waypoint CSVs
    …                     — see config/run_durham.toml for the schema

Outputs (under inputs.outdir):
    ingest/         — RGB preview, GeoTIFF metadata
    features/       — PCA scree
    cluster/        — k-medoids quality sweep, overlays, lab summary,
                      tree_label_decision.md
    kde/            — density heatmap, speed map (+ histogram), sidecar GT
    waypoints/      — overlay + spacing histogram
    lidar/          — counts.json, stats_and_coverage.csv, percent_densities.csv
                      (only if [paths.lidar] is populated and LAS files exist)
    manifest.csv    — full artefact manifest
    provenance.json — Julia/package versions, run inputs digest
    run_report.md   — auto-generated narrative
    run_log.txt     — captured stdout (write your own redirect)

This script never invokes python3.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using ArchGDAL
using CSV
using DataFrames
using Statistics
using Random
using Dates
using JSON
using Colors: red, green, blue
import KDEFlightPlanning: pixel_axes, _resize_nearest

length(ARGS) >= 1 || error("Usage: julia run_from_config.jl <config.toml>")

const CONFIG_PATH = abspath(ARGS[1])
println("[config] ", CONFIG_PATH)
inputs = load_inputs(CONFIG_PATH)
println(inputs)
println()

mkpath(inputs.outdir)
println("[outdir] ", inputs.outdir)

# Validate (rgb required; everything else only if listed)
KDEFlightPlanning.validate(inputs)

# ---------------------------------------------------------------------------
# 1. Ingest
# ---------------------------------------------------------------------------
println("\n[1/6] Ingest GeoTIFF orthomosaic …")
rs = load_rgb_geotiff(inputs.rgb)
H, W = size(rs)
println("  shape = $(H) × $(W)  px = $(rs.gt.dx) × $(rs.gt.dy)")
println("  CRS   = $(isempty(rs.crs) ? "<unknown>" : "<set>")")
report_ingest_preview(rs, inputs.outdir;
    max_preview_px = inputs.max_preview_px,
    formats        = inputs.report_formats)
report_geotiff_metadata(rs, inputs.outdir)

# ---------------------------------------------------------------------------
# 2. Optional GLI
# ---------------------------------------------------------------------------
gli_classes  = nothing
gli_gt       = nothing
if !isnothing(inputs.gli) && isfile(inputs.gli)
    println("\n[2/6] Read GLI cover-class raster …")
    gli_classes, gli_gt = read_classes_geotiff(inputs.gli)
    println("  shape = $(size(gli_classes))")
end

# ---------------------------------------------------------------------------
# 3. Mask + clustering + tree-label diagnostics
# ---------------------------------------------------------------------------
println("\n[3/6] Vegetation mask via k-medoids …")
arr_hwc = let
    Z = rs.Z
    a = Array{UInt8}(undef, H, W, 3)
    @inbounds for j in 1:H, i in 1:W
        p = Z[j, i]
        a[j, i, 1] = round(UInt8, clamp(Float64(red(p))   * 255, 0, 255))
        a[j, i, 2] = round(UInt8, clamp(Float64(green(p)) * 255, 0, 255))
        a[j, i, 3] = round(UInt8, clamp(Float64(blue(p))  * 255, 0, 255))
    end
    a
end

ks = inputs.kmedoids_k_range[1]:inputs.kmedoids_k_range[2]
# v0.5.2 hotfix: stride-decimate the orthomosaic before clustering when the
# user requests cluster_stride > 1. This is the only way to make a full-res
# (~9000x11000) Durham GeoTIFF finish clustering in minutes.
println("  cluster_stride = $(inputs.cluster_stride)  (effective image ≈ ",
        "$(cld(H, inputs.cluster_stride))×$(cld(W, inputs.cluster_stride)))")
mask_grid_px, mask_info = build_mask_from_image_strided(arr_hwc;
    stride = inputs.cluster_stride,
    k = first(ks), tree_labels = inputs.tree_labels,
    seed = inputs.seed, nsample = inputs.nsample,
    ks = ks, k_strategy = :vote,
    use_pca = false, do_cleanup = false)
println("  chosen k = $(mask_info.k)  (sample image $(get(mask_info, :H_sample, H))×",
        "$(get(mask_info, :W_sample, W)))")

# Build a RasterGrid carrying the GeoTIFF axes (METRES). This is critical:
# downstream waypoint planning interprets `spacing` in axis units, so using
# pixel_axes here would mean "spacing=40 → 40 pixels ≈ 1.13 m" — exactly the
# bug the user observed. axes_from_geotransform produces UTM coords.
xs_geo, ys_geo = axes_from_geotransform(KDEFlightPlanning._gt_as_vector(rs.gt), W, H)
label_img = permutedims(reshape(mask_info.labels_full, W, H), (2, 1))
tree_mask = in.(label_img, Ref(inputs.tree_labels))

mask_rg = RasterGrid(Float64.(tree_mask), xs_geo, ys_geo)

# Determine pixel size in metres from the geotransform (used for the
# m → px conversion in [waypoints] below).
px_m = abs(rs.gt.dx)
@assert px_m > 0 "GeoTIFF pixel size is zero — cannot convert metres → pixels."
println("  geo pixel size: $(round(px_m; digits=6)) m/px")

if mask_info.metrics !== nothing
    report_cluster_metrics_sweep(mask_info.metrics, inputs.outdir;
        formats = inputs.report_formats)
    # Also write the underlying metric values as a CSV so reviewers can
    # cross-check the sweep figure without parsing the PNG.
    try
        mkpath(joinpath(inputs.outdir, "cluster"))
        export_cluster_metrics_csv(mask_info.metrics,
            joinpath(inputs.outdir, "cluster", "cluster_quality_metrics.csv"))
    catch e
        @warn "Failed to write cluster_quality_metrics.csv" exception=e
    end
end
report_cluster_overlays(rs, mask_info.labels_full, mask_info.k, inputs.outdir;
    max_preview_px = inputs.max_preview_px,
    formats        = inputs.report_formats)

# Recompute LAB for the cluster-LAB summary so the diagnostic isn't redundant.
# When cluster_stride > 1 we run this on the decimated image only to save
# time; statistics are unaffected (clusters are the same population).
arr_for_lab = inputs.cluster_stride > 1 ?
    @view(arr_hwc[1:inputs.cluster_stride:end, 1:inputs.cluster_stride:end, :]) :
    arr_hwc
L_chan, a_chan, b_chan = rgb_to_lab_array(Array(arr_for_lab))
# Use the decimated labels (length H_sample * W_sample) for the LAB summary.
labels_for_summary = if inputs.cluster_stride > 1
    H_s = get(mask_info, :H_sample, H)
    W_s = get(mask_info, :W_sample, W)
    full = mask_info.labels_full
    # Reconstruct the decimated labels by sampling matching positions.
    out = Vector{Int}(undef, H_s * W_s)
    @inbounds for j in 1:H_s, i in 1:W_s
        out[(j - 1) * W_s + i] = full[(j - 1) * inputs.cluster_stride * W + (i - 1) * inputs.cluster_stride + 1]
    end
    out
else
    mask_info.labels_full
end

# Try to give the cluster-LAB summary access to the GLI raster too — but
# only when the GLI shape matches the (possibly downsampled) feature shape.
gli_for_summary = nothing
if !isnothing(gli_classes) && size(gli_classes) == size(L_chan)
    gli_for_summary = gli_classes
end

df_lab, _ = report_cluster_lab_summary(L_chan, a_chan, b_chan,
                                        labels_for_summary, mask_info.k,
                                        inputs.outdir;
                                        gli_classes = gli_for_summary,
                                        gli_class_codes = inputs.gli_class_codes)
report_tree_label_decision(inputs.outdir;
    tree_labels    = inputs.tree_labels,
    lab_summary_df = df_lab,
    source         = "manual (RunInputs.tree_labels)",
    notes          = "Recorded from run_from_config.jl. Re-edit `tree_labels` in $(basename(CONFIG_PATH)) to change.")

# PCA scree diagnostic on the same feature matrix used for clustering
try
    Xfeat = Float64.(stack_features(L_chan, a_chan, b_chan))
    standardize_features!(Xfeat)
    pca_model = pca_fit(Xfeat; variance_ratio = 0.95)
    pca_info  = pca_explained(pca_model)
    report_pca_explained(pca_info.explained, inputs.outdir;
        formats = inputs.report_formats)
catch e
    @warn "PCA scree report failed; continuing." exception=e
end

# ---------------------------------------------------------------------------
# 4. KDE + speed map + waypoints
# ---------------------------------------------------------------------------
println("\n[4/6] KDE density surface + speed map …")
kde_cfg = PipelineConfig(
    src_epsg      = 6348,
    kmed_k        = mask_info.k,
    seed          = inputs.seed,
    tree_labels   = inputs.tree_labels,
    kde_bandwidth = :auto,
    kde_kernel    = inputs.kde_kernel,
    kde_scaling   = :none,
)
dens_grid, _kde_info = build_density_surface(mask_rg, kde_cfg)
report_kde_density(dens_grid, inputs.outdir;
    gt = rs.gt, crs = rs.crs,
    max_preview_px = inputs.max_preview_px,
    formats        = inputs.report_formats)

# Write the density as a co-registered GeoTIFF too (preserves CRS).
kde_tif = joinpath(inputs.outdir, "kde", "kde_density.tif")
mkpath(dirname(kde_tif))
try
    write_single_band_geotiff(kde_tif, dens_grid.Z, rs.gt, rs.crs)
    println("  → kde_density.tif (co-registered)")
catch e
    @warn "Failed to write co-registered kde_density.tif" exception=e
end

strategy_kde = CurvatureGuidedSpeed(dens_grid;
    vmin = inputs.speed_bounds_mps[1], vmax = inputs.speed_bounds_mps[2])
report_speed_map(dens_grid, strategy_kde, inputs.outdir;
    gt = rs.gt, crs = rs.crs,
    max_preview_px = inputs.max_preview_px,
    formats        = inputs.report_formats)

println("\n[5/6] Mission waypoints …")
# v0.5.2 hotfix: track_spacing_m and waypoint spacing bounds are in METRES.
# Since RasterGrid axes are now in UTM metres (xs_geo, ys_geo above), the
# spacing values feed straight through without conversion.
configs = [
    FlightConfig(ConstantSpeed(inputs.speed_bounds_mps[1]), 80.0,
                 "Constant $(inputs.speed_bounds_mps[1]) m/s";
                 kernel = inputs.kde_kernel,
                 line_spacing = inputs.flightlines_spacing_m),
    FlightConfig(ConstantSpeed(inputs.speed_bounds_mps[2]), 80.0,
                 "Constant $(inputs.speed_bounds_mps[2]) m/s";
                 kernel = inputs.kde_kernel,
                 line_spacing = inputs.flightlines_spacing_m),
    FlightConfig(strategy_kde, 80.0, "KDE-guided ($(inputs.kde_kernel))";
                 kernel = inputs.kde_kernel,
                 line_spacing = inputs.flightlines_spacing_m),
]

wps_dict = Dict{String, Vector{Waypoint}}()
mkpath(joinpath(inputs.outdir, "waypoints"))
for cfg in configs
    wps = plan_mission(dens_grid, cfg;
        seconds_per_wp = 1.0,
        spacing_min    = inputs.min_waypoint_spacing_m,
        spacing_max    = inputs.max_waypoint_spacing_m)
    fname = "waypoints_" * replace(lowercase(cfg.label), " " => "_", "/" => "per", "(" => "", ")" => "") * ".csv"
    write_waypoints_csv(joinpath(inputs.outdir, "waypoints", fname), wps)
    wps_dict[cfg.label] = wps
    println("  $(cfg.label) → $(length(wps)) wps  v∈[",
            "$(round(minimum(w.speed for w in wps); digits=2)), ",
            "$(round(maximum(w.speed for w in wps); digits=2))]")
end
report_waypoints_overlay(rs, dens_grid, wps_dict, inputs.outdir;
    max_preview_px = inputs.max_preview_px,
    formats        = inputs.report_formats)

# ---------------------------------------------------------------------------
# 5. Optional LiDAR stage
# ---------------------------------------------------------------------------
if !isempty(inputs.lidar) && all(isfile(p) for p in values(inputs.lidar))
    println("\n[6/6] LiDAR count grids (LAS) …")
    if isnothing(gli_classes) || isnothing(gli_gt)
        @warn "[paths].gli not configured — skipping LiDAR cover-stratified counts."
    else
        try
            mission_order = STANDARD_MISSION_PLAN
            las_keys = (:Gd, :Gs, :Ed, :Es, :const2, :const8)
            missions = LASMission[]
            for (k, plan) in zip(las_keys, mission_order)
                haskey(inputs.lidar, k) || (
                    @warn "Missing LAS for mission key $k; skipping LiDAR stage";
                    missions = LASMission[];
                    break)
                push!(missions, LASMission(plan.mission, plan.kernel, inputs.lidar[k]))
            end
            if !isempty(missions)
                counts, _datasets = build_count_grids(missions, gli_classes, gli_gt;
                    class_codes = inputs.gli_class_codes)
                lidar_dir = joinpath(inputs.outdir, "lidar")
                mkpath(lidar_dir)
                write_count_grids_json(counts,
                    joinpath(lidar_dir, "counts.json");
                    nrows = size(gli_classes, 1), ncols = size(gli_classes, 2))
                stats_df = cover_stats_table(counts, gli_classes;
                    class_codes = inputs.gli_class_codes)
                CSV.write(joinpath(lidar_dir, "stats_and_coverage.csv"), stats_df)
                pd_df = percent_density_table(counts, gli_classes;
                    class_codes = inputs.gli_class_codes)
                CSV.write(joinpath(lidar_dir, "percent_densities.csv"), pd_df)
                println("  → counts.json, stats_and_coverage.csv, percent_densities.csv")
            end
        catch e
            @error "LiDAR stage failed; pipeline continues." exception=(e, catch_backtrace())
        end
    end
else
    println("\n[6/6] LiDAR stage skipped (no LAS paths configured / files absent).")
end

# ---------------------------------------------------------------------------
# Manifest + provenance + run report
# ---------------------------------------------------------------------------
println("\n[manifest] writing run summary …")
report_run_manifest(inputs.outdir)
report_provenance(inputs.outdir;
    package_version = "0.5.3", run_inputs = inputs)
report_run_md(inputs.outdir;
    title    = "KDEFlightPlanning run",
    sections = Dict(
        "Inputs"       => "RGB: `$(inputs.rgb)`\n\nGLI: $(isnothing(inputs.gli) ? "(none)" : "`$(inputs.gli)`")\n\nLiDAR keys: $(collect(keys(inputs.lidar)))",
        "Output stages" => "ingest, features, cluster, kde, waypoints" * (isempty(inputs.lidar) ? "" : ", lidar"),
        "Tree labels"  => "`$(inputs.tree_labels)` — see `cluster/tree_label_decision.md`",
        "Performance knobs" => "cluster_stride=$(inputs.cluster_stride), max_preview_px=$(inputs.max_preview_px), report_formats=$(inputs.report_formats)",
        "Spacing (m)"  => "track=$(inputs.flightlines_spacing_m), waypoint=$(inputs.min_waypoint_spacing_m)–$(inputs.max_waypoint_spacing_m)",
    ))
# Generate the checklist LAST so manifest.csv / provenance.json / run_report.md
# are detected as 'produced'.
report_proposed_artifacts_checklist(inputs.outdir)
println("\nDone. Artefacts in: $(inputs.outdir)")
