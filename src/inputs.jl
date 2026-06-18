"""
    inputs.jl — Typed run configuration loaded from TOML

Replaces hard-coded absolute paths scattered across scripts with a single
immutable `RunInputs` struct loaded from a TOML config file.

Public API
----------
- `RunInputs`            — immutable struct of all run paths + planning params
- `load_inputs(path)`    — parse TOML → `RunInputs`
- `validate(inputs)`     — assert all required files exist; raise informative error
- `inputs_template(io)`  — emit a fully-commented TOML scaffold

`tree_labels` is intentionally exposed as a `Vector{Int}` so callers can specify
multiple clusters belonging to the vegetation class. Diagnostic heuristic
scoring is recorded in reports but never overrides this manual choice.
"""

using TOML

# ---------------------------------------------------------------------------
# RunInputs
# ---------------------------------------------------------------------------

"""
    MissingInputError(key, path)

Raised by [`validate`](@ref) when a required input file is missing.
"""
struct MissingInputError <: Exception
    key ::String
    path::String
end

Base.showerror(io::IO, e::MissingInputError) = print(io,
    "MissingInputError: '$(e.key)' refers to '$(e.path)' which does not exist or is not readable.")

"""
    RunInputs

Immutable bundle of all run paths and planning parameters.

Required:
- `rgb`: path to the orthomosaic GeoTIFF.

Optional (default `nothing` / empty):
- `gli`:        cover-class GeoTIFF (integer codes; class scheme below).
- `lidar`:      `Dict{Symbol,String}` keyed by `:Gd, :Gs, :Ed, :Es, :const2, :const8`.
- `trajectory`: `Dict{Symbol,String}` of flown-track CSVs per mission.
- `waypoints`:  `Dict{Symbol,String}` of planned-waypoint CSVs per mission.
- `counts_json`: pre-computed count grid JSON.
- `tree_labels`: cluster ids (1-based) that count as vegetation in `labels_to_mask`. Defaults to `[1]` — the package's historical default. **Manual setting is encouraged**: see `tree_label_decision.md` written by the reports module.
- `gli_class_codes`: integer codes for the GLI cover classes (must match the GLI raster).

Planning parameters mirror `PipelineConfig` / `FlightConfig` defaults so a single TOML can drive the whole pipeline.
"""
Base.@kwdef struct RunInputs
    # Required input
    rgb        ::String

    # Optional rasters / point clouds
    gli            ::Union{String, Nothing} = nothing
    lidar          ::Dict{Symbol, String}   = Dict{Symbol, String}()
    trajectory     ::Dict{Symbol, String}   = Dict{Symbol, String}()
    waypoints      ::Dict{Symbol, String}   = Dict{Symbol, String}()

    # Manuscript ground-truth summary files (bundled under data/ground_truth/).
    # These drive the bootstrap and figure stages without raw LiDAR/rasters.
    counts_json          ::Union{String, Nothing} = nothing
    stats_csv            ::Union{String, Nothing} = nothing
    percent_densities_csv::Union{String, Nothing} = nothing
    # Optional 263×324 integer-coded GLI class raster used as fixed cover-class
    # denominators in the trajectory line-scan / characterization figures.
    gli_class_raster     ::Union{String, Nothing} = nothing

    # Output directory
    outdir         ::String                 = "output"

    # ---------- Planning parameters ----------
    seed                  ::Int             = 6213
    # Inter-flight-line ("track") spacing in METRES. For GeoTIFF inputs this
    # is converted to pixel units internally using the geotransform's pixel
    # size. The historical field name remains for backward compatibility but
    # the canonical name is `track_spacing_m`.
    flightlines_spacing_m ::Float64         = 40.0
    # Curvature-adaptive waypoint spacing in METRES (min/max). The
    # waypoint generator emits steps in [min, max] depending on local
    # curvature/gradient. Values are also converted m → px for GeoTIFF runs.
    min_waypoint_spacing_m::Float64         = 10.0
    max_waypoint_spacing_m::Float64         = 30.0
    speed_bounds_mps      ::NTuple{2, Float64} = (2.0, 8.0)
    kde_kernel            ::Symbol          = :epanechnikov     # :gaussian | :epanechnikov
    kmedoids_k_range      ::Tuple{Int, Int} = (2, 12)
    nsample               ::Int             = 2000

    # ---------- Performance knobs (added in v0.5.2 hotfix) ----------
    # Stride used to decimate the orthomosaic before clustering. Setting
    # `cluster_stride=1` runs k-medoids over every pixel (the historical
    # behaviour); set ≥ 4 for full-resolution orthomosaics so the pipeline
    # completes in minutes instead of hours.
    cluster_stride        ::Int             = 1
    # Maximum side length (pixels) of any preview/heatmap figure. Larger
    # rasters are stride-decimated for plotting only — pixel values are
    # unchanged.
    max_preview_px        ::Int             = 2000
    # Output formats for every Makie-rendered diagnostic figure. PDF is
    # disabled by default in v0.5.2.1 because Makie's CairoMakie PDF save
    # has been observed to time out on `SystemError(close, "Operation
    # timed out")` for large heatmaps on synced filesystems (e.g. macOS +
    # iCloud / NFS / Dropbox). Set `report_formats = ["png", "pdf"]` in
    # the TOML to opt back in to PDFs; failures are logged with `@warn`
    # and the pipeline continues.
    report_formats        ::Vector{String}  = ["png"]

    # Manual vegetation cluster assignment. May contain multiple cluster ids
    # if the tree class spans more than one cluster in the chosen k-medoids
    # output. Reports flag the manual choice and record candidate heuristic
    # scores; nothing here is auto-overridden.
    tree_labels           ::Vector{Int}     = [1]

    # GLI class codes (used by lidar_counts to build cover masks).
    # Defaults match `lidar.jl` conventions for the Durham NH site.
    gli_class_codes       ::Dict{Symbol, Int} = Dict(:field => 2, :decid => 0, :conif => 1)
end

# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

function Base.show(io::IO, ::MIME"text/plain", r::RunInputs)
    println(io, "RunInputs")
    println(io, "  rgb           = ", r.rgb)
    println(io, "  gli           = ", isnothing(r.gli) ? "(none)" : r.gli)
    println(io, "  lidar         = ", isempty(r.lidar) ? "(none)" : keys(r.lidar))
    println(io, "  trajectory    = ", isempty(r.trajectory) ? "(none)" : keys(r.trajectory))
    println(io, "  waypoints     = ", isempty(r.waypoints) ? "(none)" : keys(r.waypoints))
    println(io, "  counts_json   = ", isnothing(r.counts_json) ? "(none)" : r.counts_json)
    println(io, "  stats_csv     = ", isnothing(r.stats_csv) ? "(none)" : r.stats_csv)
    println(io, "  pct_densities = ", isnothing(r.percent_densities_csv) ? "(none)" : r.percent_densities_csv)
    println(io, "  gli_class_raster = ", isnothing(r.gli_class_raster) ? "(none)" : r.gli_class_raster)
    println(io, "  outdir        = ", r.outdir)
    println(io, "  tree_labels   = ", r.tree_labels)
    println(io, "  seed          = ", r.seed)
    println(io, "  line_spacing  = ", r.flightlines_spacing_m, " m")
    println(io, "  speed_bounds  = ", r.speed_bounds_mps, " m/s")
    println(io, "  kde_kernel    = :", r.kde_kernel)
    println(io, "  k range       = ", r.kmedoids_k_range[1], ":", r.kmedoids_k_range[2])
    println(io, "  nsample       = ", r.nsample)
    print(io,   "  gli_class_codes = ", r.gli_class_codes)
end

# ---------------------------------------------------------------------------
# load_inputs
# ---------------------------------------------------------------------------

_to_dict_sym(d::AbstractDict) = Dict{Symbol, String}(Symbol(k) => String(v) for (k, v) in d)

"""
    load_inputs(path::AbstractString) -> RunInputs

Parse a TOML file. Schema:

```toml
[paths]
rgb         = "data/ortho.tif"      # required
gli         = "data/gli.tif"
counts_json = "output/counts.json"
outdir      = "output/run1"

[paths.lidar]
Gd = "data/Gd.las"
…

[paths.trajectory]
Gd = "data/traj_Gd.csv"
…

[paths.waypoints]
Ed = "data/waypoints_Ed.csv"

[planning]
seed                   = 6213
flightlines_spacing_m  = 40.0
speed_bounds_mps       = [2.0, 8.0]
kde_kernel             = "epanechnikov"
kmedoids_k_range       = [2, 12]
nsample                = 2000
tree_labels            = [1]

[gli_class_codes]
field = 2
decid = 0
conif = 1
```

Unknown keys are silently ignored to allow forward-compatible TOMLs.
Relative paths are interpreted relative to the TOML file's directory.
"""
function load_inputs(path::AbstractString)::RunInputs
    isfile(path) || throw(ArgumentError("TOML config not found: $path"))
    raw      = TOML.parsefile(path)
    base_dir = dirname(abspath(path))

    _resolve(p::AbstractString)::String =
        isabspath(p) ? String(p) : abspath(joinpath(base_dir, p))

    paths     = get(raw, "paths", Dict{String, Any}())
    planning  = get(raw, "planning", Dict{String, Any}())
    gli_codes = get(raw, "gli_class_codes", Dict{String, Any}())

    # Required
    haskey(paths, "rgb") || throw(ArgumentError("[paths].rgb is required in $path"))
    rgb_path = _resolve(paths["rgb"])

    # Optional file/dir paths
    gli         = haskey(paths, "gli")         ? _resolve(paths["gli"])         : nothing
    counts_json = haskey(paths, "counts_json") ? _resolve(paths["counts_json"]) : nothing
    stats_csv   = haskey(paths, "stats_csv")   ? _resolve(paths["stats_csv"])   : nothing
    percent_densities_csv = haskey(paths, "percent_densities_csv") ?
                              _resolve(paths["percent_densities_csv"]) : nothing
    gli_class_raster = haskey(paths, "gli_class_raster") ?
                              _resolve(paths["gli_class_raster"]) : nothing
    outdir      = haskey(paths, "outdir")      ? _resolve(paths["outdir"])      : abspath(joinpath(base_dir, "output"))

    # Dicts of dataset paths (resolve each value relative to base_dir)
    _resolve_dict(d) = Dict{Symbol, String}(
        Symbol(k) => _resolve(String(v)) for (k, v) in d
    )
    lidar      = _resolve_dict(get(paths, "lidar",      Dict{String, Any}()))
    trajectory = _resolve_dict(get(paths, "trajectory", Dict{String, Any}()))
    waypoints  = _resolve_dict(get(paths, "waypoints",  Dict{String, Any}()))

    # Planning
    seed           = Int(get(planning, "seed", 6213))
    # Track spacing accepts either the new name `track_spacing_m` or the
    # legacy `flightlines_spacing_m`; both encode METRES.
    line_spacing   = Float64(get(planning, "track_spacing_m",
                          get(planning, "flightlines_spacing_m", 40.0)))
    min_wp_m       = Float64(get(planning, "min_waypoint_spacing_m", 10.0))
    max_wp_m       = Float64(get(planning, "max_waypoint_spacing_m", 30.0))
    speed_b_v      = get(planning, "speed_bounds_mps", [2.0, 8.0])
    speed_bounds   = (Float64(speed_b_v[1]), Float64(speed_b_v[2]))
    kde_kernel     = Symbol(get(planning, "kde_kernel", "epanechnikov"))
    k_range_v      = get(planning, "kmedoids_k_range", [2, 12])
    kmedoids_range = (Int(k_range_v[1]), Int(k_range_v[2]))
    nsample        = Int(get(planning, "nsample", 2000))
    tree_labels    = Int.(get(planning, "tree_labels", [1]))
    cluster_stride = Int(get(planning, "cluster_stride", 1))
    max_preview_px = Int(get(planning, "max_preview_px", 2000))
    report_formats = String.(get(planning, "report_formats", ["png"]))

    gli_class_codes = if isempty(gli_codes)
        Dict(:field => 2, :decid => 0, :conif => 1)
    else
        Dict{Symbol, Int}(Symbol(k) => Int(v) for (k, v) in gli_codes)
    end

    return RunInputs(;
        rgb                    = rgb_path,
        gli                    = gli,
        lidar                  = lidar,
        trajectory             = trajectory,
        waypoints              = waypoints,
        counts_json            = counts_json,
        stats_csv              = stats_csv,
        percent_densities_csv  = percent_densities_csv,
        gli_class_raster       = gli_class_raster,
        outdir                 = outdir,
        seed                   = seed,
        flightlines_spacing_m  = line_spacing,
        min_waypoint_spacing_m = min_wp_m,
        max_waypoint_spacing_m = max_wp_m,
        speed_bounds_mps       = speed_bounds,
        kde_kernel             = kde_kernel,
        kmedoids_k_range       = kmedoids_range,
        nsample                = nsample,
        tree_labels            = tree_labels,
        cluster_stride         = cluster_stride,
        max_preview_px         = max_preview_px,
        report_formats         = report_formats,
        gli_class_codes        = gli_class_codes,
    )
end

# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

"""
    validate(r::RunInputs; require_lidar=false, require_gli=false)

Throw [`MissingInputError`](@ref) on any missing file. The default mode only
verifies `rgb` (the single mandatory input) plus any optional file paths that
*are* set (i.e. set-but-missing is an error, unset is fine). Pass
`require_lidar=true` / `require_gli=true` to also error when those are unset.
"""
function validate(r::RunInputs;
                  require_lidar::Bool = false,
                  require_gli  ::Bool = false)
    isfile(r.rgb) || throw(MissingInputError("rgb", r.rgb))

    if !isnothing(r.gli)
        isfile(r.gli) || throw(MissingInputError("gli", r.gli))
    elseif require_gli
        throw(MissingInputError("gli", "(unset)"))
    end

    if !isnothing(r.counts_json)
        isfile(r.counts_json) || throw(MissingInputError("counts_json", r.counts_json))
    end
    if !isnothing(r.stats_csv)
        isfile(r.stats_csv) || throw(MissingInputError("stats_csv", r.stats_csv))
    end
    if !isnothing(r.percent_densities_csv)
        isfile(r.percent_densities_csv) ||
            throw(MissingInputError("percent_densities_csv", r.percent_densities_csv))
    end
    if !isnothing(r.gli_class_raster)
        isfile(r.gli_class_raster) ||
            throw(MissingInputError("gli_class_raster", r.gli_class_raster))
    end

    if !isempty(r.lidar)
        for (k, v) in r.lidar
            isfile(v) || throw(MissingInputError("lidar[$k]", v))
        end
    elseif require_lidar
        throw(MissingInputError("lidar", "(empty)"))
    end

    for (k, v) in r.trajectory
        isfile(v) || throw(MissingInputError("trajectory[$k]", v))
    end
    for (k, v) in r.waypoints
        isfile(v) || throw(MissingInputError("waypoints[$k]", v))
    end

    return r
end

# ---------------------------------------------------------------------------
# inputs_template
# ---------------------------------------------------------------------------

const _TEMPLATE_TOML = """
# KDEFlightPlanning run config — template
#
# Required:
#   [paths].rgb — orthomosaic GeoTIFF (any band layout supported by GDAL)
#
# All other paths are optional; missing-but-unreferenced is fine, missing-
# but-listed errors at validate() time so typos surface early.
#
# Relative paths are resolved against this TOML file's directory.

[paths]
rgb         = "data/ortho.tif"
gli         = "data/gli.tif"                 # cover-class raster (integer codes)
outdir      = "output/run1"

# Manuscript ground-truth summary files. The reviewer package bundles these
# under data/ground_truth/ so the bootstrap and figure stages reproduce the
# manuscript statistics without raw LiDAR/rasters.
counts_json           = "data/ground_truth/counts.json"
stats_csv             = "data/ground_truth/stats_and_coverage.csv"
percent_densities_csv = "data/ground_truth/percent_densities.csv"
# Optional 263×324 integer-coded GLI class raster (fixed cover-class
# denominators for the line-scan / characterization figures).
# gli_class_raster    = "data/ground_truth/gli_class_263x324.tif"

[paths.lidar]
# Mission keys: Gd, Gs, Ed, Es, const2, const8
Gd     = "data/lidar/Gd.las"
Gs     = "data/lidar/Gs.las"
Ed     = "data/lidar/Ed.las"
Es     = "data/lidar/Es.las"
const2 = "data/lidar/const2.las"
const8 = "data/lidar/const8.las"

[paths.trajectory]
# Flown-track CSVs per mission (UTM Easting/Northing + time)
# Gd = "data/trajectory/Gd.csv"

[paths.waypoints]
# Planned-waypoint CSVs per mission
# Ed = "data/waypoints/Ed.csv"

[planning]
seed                  = 6213
# Lawnmower track spacing, in METRES. For GeoTIFF inputs the script
# converts this to pixel units using the file's pixel size. The legacy
# `flightlines_spacing_m` name is still accepted as a synonym.
track_spacing_m       = 40.0
# Curvature-adaptive waypoint spacing (m). The generator emits steps in
# [min, max] depending on local curvature; high-density / high-curvature
# regions get tighter spacing.
min_waypoint_spacing_m = 10.0
max_waypoint_spacing_m = 30.0
speed_bounds_mps      = [2.0, 8.0]            # [vmin, vmax]
kde_kernel            = "epanechnikov"        # "epanechnikov" or "gaussian"
kmedoids_k_range      = [2, 12]
nsample               = 2000

# Performance knobs added in the v0.5.2 hotfix:
#   cluster_stride : decimate the orthomosaic by this stride before
#                    clustering, then nearest-neighbour upsample labels back
#                    to full resolution. For a full-res Durham GeoTIFF
#                    (~9000 x 11000) use 8 or 16. Default 1 = unchanged.
#   max_preview_px : per-side cap for any preview/heatmap figure; larger
#                    rasters are stride-decimated for plotting only.
cluster_stride         = 1
max_preview_px         = 2000

# Output formats for diagnostic visuals. PNG is the default — fast and
# fail-safe. Add "pdf" if you want vector copies, but be aware that on
# macOS with iCloud/Dropbox-synced output directories CairoMakie's PDF
# writer occasionally times out at file close. A failure here is logged
# with @warn and the pipeline continues.
#   report_formats = ["png", "pdf"]
report_formats         = ["png"]

# Manual tree-cluster assignment. May contain multiple cluster ids if
# vegetation spans multiple k-medoids clusters. Reports record diagnostic
# heuristic scores (mean L*, a*, b*, GLI overlap if present) but never
# auto-override this choice.
tree_labels           = [1]

[gli_class_codes]
# Integer codes used in the GLI cover-class raster
field = 2
decid = 0
conif = 1
"""

"""
    inputs_template(io::IO=stdout) -> Nothing

Print a fully-commented TOML scaffold. Use:

    open("config/template.toml", "w") do io
        inputs_template(io)
    end
"""
function inputs_template(io::IO = stdout)
    print(io, _TEMPLATE_TOML)
    return nothing
end
