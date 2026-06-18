# ground_truth.jl — shared, config-driven discovery of manuscript ground-truth
# files for the figure / bootstrap / trajectory stages.
#
# There are NO hardcoded absolute paths here. Resolution order for each file:
#   1. The RunInputs field from the supplied TOML config (if a config is given
#      and the field is set), else
#   2. the bundled copy under <repo>/data/ground_truth/.
#
# `include` this from a script that already has `using KDEFlightPlanning`.
#
# Include-guard: the unified entry point (run_pipeline.jl) includes several
# stage scripts into the same Main module, each of which includes this file.
# Re-including would redefine the GroundTruth struct (an error in Julia), so
# we no-op on the second and later includes.
if !isdefined(@__MODULE__, :_GROUND_TRUTH_JL_LOADED)

const _GROUND_TRUTH_JL_LOADED = true

const _REPO_ROOT = abspath(joinpath(@__DIR__, ".."))
const _GT_DIR    = joinpath(_REPO_ROOT, "data", "ground_truth")

struct GroundTruth
    inputs       ::Union{RunInputs, Nothing}
    outdir       ::String
    bootstrap_dir::String
    figures_dir  ::String
    trajectory_dir::String
end

"""
    resolve_ground_truth(config_path) -> GroundTruth

`config_path` may be an empty string (use bundled defaults + project-root
`output/`) or a path to a RunInputs TOML. Output directories are derived from
the config's `outdir` when a config is supplied, otherwise from project-root
`output/`.
"""
function resolve_ground_truth(config_path::AbstractString)
    inputs = nothing
    outdir = joinpath(_REPO_ROOT, "output")
    if !isempty(config_path)
        isfile(config_path) || error("Config not found: $config_path")
        inputs = load_inputs(config_path)
        outdir = inputs.outdir
    end
    return GroundTruth(
        inputs,
        outdir,
        joinpath(outdir, "bootstrap"),
        joinpath(outdir, "figures"),
        joinpath(outdir, "trajectory"),
    )
end

# Map a logical ground-truth key to (RunInputs field, bundled filename).
const _GT_KEYS = Dict(
    :counts_json           => (:counts_json,           "counts.json"),
    :stats_csv             => (:stats_csv,             "stats_and_coverage.csv"),
    :percent_densities_csv => (:percent_densities_csv, "percent_densities.csv"),
    :waypoints_xy          => (nothing,                "E_density_aware__waypoints_xy.csv"),
    :gli_class_raster      => (:gli_class_raster,      nothing),
)

"""
    find_ground_truth(gt, key) -> Union{String, Nothing}

Return the resolved path for a ground-truth `key`, or `nothing` if it is
neither configured nor bundled. `key` ∈ keys(`_GT_KEYS`).
"""
function find_ground_truth(gt::GroundTruth, key::Symbol)
    field, bundled = _GT_KEYS[key]
    # 1. Config-supplied path (if the field exists and is set).
    if gt.inputs !== nothing && field !== nothing
        v = getfield(gt.inputs, field)
        if v !== nothing && isfile(v)
            return v
        end
    end
    # 1b. The E density-aware waypoints CSV may be supplied via
    #     [paths.waypoints].Ed (a Dict field, not a scalar RunInputs field).
    if key === :waypoints_xy && gt.inputs !== nothing
        for k in (:Ed, :E, :missionE)
            if haskey(gt.inputs.waypoints, k) && isfile(gt.inputs.waypoints[k])
                return gt.inputs.waypoints[k]
            end
        end
    end
    # 2. Bundled default under data/ground_truth/.
    if bundled !== nothing
        p = joinpath(_GT_DIR, bundled)
        isfile(p) && return p
    end
    return nothing
end

"""
    require_ground_truth(gt, key, human_name) -> String

Like [`find_ground_truth`](@ref) but throws a clear error when the file is
absent from both the config and the bundle.
"""
function require_ground_truth(gt::GroundTruth, key::Symbol, human_name::AbstractString)
    p = find_ground_truth(gt, key)
    p === nothing && error(
        "Required ground-truth file '$human_name' not found.\n" *
        "  Set [paths].$(string(key)) in your config, or place the file at\n" *
        "  $(joinpath(_GT_DIR, string(human_name))).")
    return p
end

end  # include-guard (_GROUND_TRUTH_JL_LOADED)
