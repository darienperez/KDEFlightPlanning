#!/usr/bin/env julia
"""
    scripts/gen_kde_diagnostic_figs.jl  —  DEPRECATED / DISABLED

This script regenerated KDE-diagnostic figures from a *screenshot-derived* KDE
surface CSV (`output/orthomosaic_kde/kmedoids_kde_density.csv`). That surface
was produced by the old screenshot-clustering workflow, which has been removed
in favour of the config-driven pipeline. No stage produces its input any more,
so the script can no longer run as written and is intentionally disabled.

The KDE surface is now produced data-drivenly by the planning stage
(`scripts/run_from_config.jl`, written as `kde_density.tif` under the run's
`outdir`). If you want diagnostic plots of that surface, build them from the
GeoTIFF the planning stage emits rather than from the obsolete screenshot CSV.

Run the supported pipeline instead:
    julia --project=. scripts/run_pipeline.jl [CONFIG] [--smoke]
"""

error("""
    gen_kde_diagnostic_figs.jl is DEPRECATED and disabled.
    Its input (a screenshot-derived KDE CSV) is no longer produced by any stage.
    Use:  julia --project=. scripts/run_pipeline.jl [CONFIG] [--smoke]
    """)
