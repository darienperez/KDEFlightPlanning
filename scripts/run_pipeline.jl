#!/usr/bin/env julia
"""
    scripts/run_pipeline.jl  —  SINGLE ENTRY POINT for the whole pipeline

One command to run the KDE-guided flight-planning analysis for:

  "KDE-Guided Offline Variable-Speed Flight Planning for UAV LiDAR
   in Forested Terrain"
  Perez Martin, Hunsaker, Jacobs, Thein — Remote Sensing (MDPI), in prep.

It orchestrates every stage in order, delegating to the package functions in
`src/` and the stage scripts (no analysis logic is re-implemented here):

  1. Ingest + mask + KDE + speed map + waypoints   (run_from_config.jl)
  2. Block-bootstrap CR / ΔCR confidence intervals  (run_bootstrap.jl)
  3a. Characterization figures                       (gen_characterization_figures.jl)
  3b. Trajectory analysis + publication figures      (trajectory_analysis.jl,
      make_figures.jl) — OPTIONAL: needs the large flown-track CSVs declared
      under [paths.trajectory]. Skipped with a clear message if absent.

All data discovery is config-driven (RunInputs). The manuscript summary tables
(counts.json, stats_and_coverage.csv, percent_densities.csv) are bundled under
`data/ground_truth/` and resolved automatically when not overridden by config.

Usage
-----
    julia --project=. scripts/run_pipeline.jl [CONFIG] [options]

Arguments
    CONFIG            Path to a RunInputs TOML config.
                      Default: config/run_durham.toml

Options
    --smoke           Build a small synthetic GeoTIFF fixture and run the REAL
                      config-driven pipeline against it (planning) plus the
                      bootstrap and characterization-figure stages against the
                      bundled ground-truth tables. No large data required;
                      always succeeds offline.
    --skip-bootstrap  Skip stage 2 (bootstrap CIs).
    --skip-figures    Skip stage 3 (figure generation).
    -h, --help        Print this message and exit.

Data requirements (FULL mode)
    Planning (stage 1) needs [paths].rgb — an orthomosaic GeoTIFF. A
    lower-resolution ortho is fine; set cluster_stride for full-res files.
    Bootstrap + characterization figures (stages 2, 3a) run from the bundled
    data/ground_truth/ tables with no extra inputs.
    Trajectory figures (stage 3b) additionally need the large flown-track CSVs
    under [paths.trajectory]; these are NOT bundled. If they are absent the
    stage is skipped (not failed).

Examples
    # Offline smoke run (synthetic ortho fixture + bundled ground truth):
    julia --project=. scripts/run_pipeline.jl --smoke

    # Full reproduction against your local ortho (paths set in the TOML):
    julia --project=. scripts/run_pipeline.jl config/run_durham.toml
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const HERE = @__DIR__
const ROOT = abspath(joinpath(HERE, ".."))

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
args            = copy(ARGS)
want_help       = any(a -> a in ("-h", "--help"), args)
smoke           = "--smoke"          in args
skip_bootstrap  = "--skip-bootstrap" in args
skip_figures    = "--skip-figures"   in args
positional      = filter(a -> !startswith(a, "-"), args)

if want_help
    src = read(@__FILE__, String)
    stop = findfirst("\nusing Pkg", src)
    println(strip(replace(src[1:(stop === nothing ? length(src) : first(stop))],
                          "\"\"\"" => "")))
    exit(0)
end

using KDEFlightPlanning

# ---------------------------------------------------------------------------
# Determine the config to drive the run.
#   --smoke: generate a synthetic GeoTIFF fixture + its TOML, augmented with
#            the bundled ground-truth paths so stages 2/3a have inputs.
#   FULL   : the supplied (or default) config.
# ---------------------------------------------------------------------------
const GT_DIR = joinpath(ROOT, "data", "ground_truth")

function _augment_config_with_ground_truth(toml_path::AbstractString)
    # Inject the bundled ground-truth keys into the [paths] table of a generated
    # fixture TOML so the bootstrap / figure stages can resolve counts.json etc.
    # These MUST live under [paths]; appending at EOF would attach them to
    # whatever table header comes last (e.g. [gli_class_codes]) and break parsing.
    txt = read(toml_path, String)
    occursin("counts_json", txt) && return toml_path
    block = """
    # Ground-truth summary tables (bundled) — injected by run_pipeline.jl --smoke
    counts_json           = "$(joinpath(GT_DIR, "counts.json"))"
    stats_csv             = "$(joinpath(GT_DIR, "stats_and_coverage.csv"))"
    percent_densities_csv = "$(joinpath(GT_DIR, "percent_densities.csv"))"
    """
    # Insert immediately after the [paths] header line.
    new_txt = replace(txt, r"(?m)^\[paths\][ \t]*\n" => "[paths]\n" * block; count = 1)
    new_txt == txt && error("Could not find a [paths] table in $toml_path to augment.")
    write(toml_path, new_txt)
    return toml_path
end

if smoke
    println("=" ^ 72)
    println("KDEFlightPlanning — UNIFIED PIPELINE ENTRY POINT")
    println("Mode      : SMOKE (synthetic ortho fixture + bundled ground truth)")
    println("=" ^ 72)
    fixture_dir = joinpath(ROOT, "output", "smoke_fixture")
    mkpath(fixture_dir)
    println("\n[smoke] Generating synthetic GeoTIFF fixture in $fixture_dir …")
    push!(empty!(ARGS), fixture_dir)
    include(joinpath(HERE, "make_synthetic_geotiff_fixture.jl"))
    config_path = _augment_config_with_ground_truth(
        joinpath(fixture_dir, "synthetic_run.toml"))
else
    config_path = isempty(positional) ?
                    joinpath(ROOT, "config", "run_durham.toml") :
                    abspath(positional[1])
    println("=" ^ 72)
    println("KDEFlightPlanning — UNIFIED PIPELINE ENTRY POINT")
    println("Mode      : FULL (data-driven)")
    println("Config    : ", config_path)
    println("Stages    : ingest/KDE/waypoints",
            skip_bootstrap ? "" : " + bootstrap",
            skip_figures   ? "" : " + figures")
    println("=" ^ 72)

    isfile(config_path) || error(
        "Config not found: $config_path\n" *
        "Pass a TOML path or use config/run_durham.toml. " *
        "Run with --help for usage, or --smoke for an offline synthetic run.")
end

inputs = load_inputs(config_path)

# ---------------------------------------------------------------------------
# Stage runner: include each stage script into its OWN fresh module.
#
# The stage scripts were written to run standalone (`julia stage.jl …`) and
# declare top-level globals (e.g. `H, W = size(rs)`, `const _GT`, `OUT_DIR`).
# Including them all into `Main` makes those globals collide — and under
# Julia ≥ 1.12 a re-assignment of an implicitly-const top-level binding is a
# hard error. Giving each stage its own module isolates its globals while
# still sharing the loaded `KDEFlightPlanning` package and the global `ARGS`.
# ---------------------------------------------------------------------------
let _stage_counter = Ref(0)
    global function _run_stage(script::AbstractString, cfg::AbstractString)
        push!(empty!(ARGS), cfg)
        modname = Symbol("Stage_", _stage_counter[] += 1)
        mod = Core.eval(Main, :(module $modname end))
        Base.include(mod, joinpath(HERE, script))
        return nothing
    end
end

# ---------------------------------------------------------------------------
# Stage 1: data-driven planning pipeline (ingest → mask → KDE → waypoints)
# ---------------------------------------------------------------------------
println("\n[1/3] Data-driven planning pipeline (ingest → KDE → waypoints) …")
try
    _run_stage("run_from_config.jl", config_path)
catch e
    # Base.include wraps a thrown error in a LoadError; unwrap to inspect it.
    err = e isa LoadError ? e.error : e
    if err isa KDEFlightPlanning.MissingInputError
        println("\n" * "!" ^ 72)
        println("MISSING INPUT — planning stage skipped:")
        showerror(stdout, err); println()
        println("Provide [paths].rgb (a lower-res orthomosaic is fine) in")
        println("$config_path, or run with --smoke for an offline demo.")
        println("Stages 2/3 below still run from the bundled ground-truth tables.")
        println("!" ^ 72)
    else
        rethrow()
    end
end

# ---------------------------------------------------------------------------
# Stage 2: block-bootstrap confidence intervals (bundled counts.json)
# ---------------------------------------------------------------------------
if !skip_bootstrap
    println("\n[2/3] Block-bootstrap CR / ΔCR confidence intervals …")
    try
        _run_stage("run_bootstrap.jl", config_path)
    catch e
        println("  ⚠ Bootstrap stage could not run: ", sprint(showerror, e))
        println("    (Ensure data/ground_truth/counts.json exists or set")
        println("     [paths].counts_json in $config_path, or use --skip-bootstrap.)")
    end
else
    println("\n[2/3] Bootstrap stage skipped (--skip-bootstrap).")
end

# ---------------------------------------------------------------------------
# Stage 3: figures
#   3a. Characterization figures — bundled ground truth only.
#   3b. Trajectory analysis + publication figures — OPTIONAL, needs the large
#       flown-track CSVs declared under [paths.trajectory].
# ---------------------------------------------------------------------------
function _have_trajectory_inputs(inp)::Bool
    isempty(inp.trajectory) && return false
    return any(isfile, values(inp.trajectory))
end

if !skip_figures
    println("\n[3/3] Figures …")

    println("\n  [3a] Characterization figures (bundled ground truth) …")
    try
        _run_stage("gen_characterization_figures.jl", config_path)
    catch e
        println("  ⚠ gen_characterization_figures.jl could not run: ",
                sprint(showerror, e))
    end

    if _have_trajectory_inputs(inputs)
        println("\n  [3b] Trajectory analysis + publication figures …")
        try
            _run_stage("trajectory_analysis.jl", config_path)
            _run_stage("make_figures.jl", config_path)
        catch e
            println("  ⚠ Trajectory figure stage could not run: ",
                    sprint(showerror, e))
        end
    else
        println("\n  [3b] Trajectory figures SKIPPED — no [paths.trajectory] CSVs found.")
        println("       The large flown-track CSVs are not bundled. To generate")
        println("       Figures 1–3 + the main bootstrap CI figure, set")
        println("       [paths.trajectory] (const2 / missionE / const8) in")
        println("       $config_path to your local trajectory CSVs.")
    end
else
    println("\n[3/3] Figure stage skipped (--skip-figures).")
end

println("\n" * "=" ^ 72)
println("Pipeline complete. Review the output directory for all artefacts:")
println("  ", inputs.outdir)
println("=" ^ 72)
