#!/usr/bin/env julia
"""
    scripts/reproduce_all.jl  —  DEPRECATED

This legacy script has been superseded by the single, config-driven entry
point:

    julia --project=. scripts/run_pipeline.jl [CONFIG] [--smoke]

The old `reproduce_all.jl` relied on hardcoded absolute paths to
`space_files/` and on screenshot-image clustering. Both are gone:

  • Manuscript data discovery is now entirely config-driven (RunInputs TOML);
    see config/run_durham.toml.
  • The smoke path builds a *synthetic GeoTIFF fixture* and runs the REAL
    config-driven pipeline against it (run_pipeline.jl --smoke), rather than
    clustering an arbitrary screenshot.

This file is kept only as a redirect so existing instructions don't silently
break. It forwards its arguments to run_pipeline.jl.
"""

const _HERE = @__DIR__
println("scripts/reproduce_all.jl is DEPRECATED — forwarding to scripts/run_pipeline.jl")
println("Use:  julia --project=. scripts/run_pipeline.jl [CONFIG] [--smoke]\n")
include(joinpath(_HERE, "run_pipeline.jl"))
