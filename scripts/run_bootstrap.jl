#!/usr/bin/env julia
"""
    scripts/run_bootstrap.jl

Reproducible bootstrap CI generation for all coverage-ratio statistics.
Outputs are written to `output/bootstrap/`.

This script produces two CSV files matching the spatial block-bootstrap
parameters approved for the manuscript:
  - nboot     = 5000
  - block_frac = 0.03  (block_side = 9 cells for the Durham 324×263 grid)
  - seed      = 42
  - level     = 0.95 (two-sided percentile CI)

Outputs
-------
  output/bootstrap/bootstrap_cr_cis_all36.csv
      Block-bootstrap CIs for all 36 (cover × return × mission × kernel)
      combinations present in counts.json.

  output/bootstrap/bootstrap_cr_difference_cis.csv
      Paired block-bootstrap CIs for the six primary narrative comparisons:
      KDE-guided Epanechnikov (density-aware) vs. Constant 2 m/s and
      KDE-guided Epanechnikov (density-aware) vs. Constant 8 m/s,
      for each cover (Field, Deciduous, Coniferous) × return (Ground, All).

Usage
-----
  julia --project=. scripts/run_bootstrap.jl [CONFIG.toml] [OUTPUT_DIR]

  CONFIG.toml  RunInputs TOML config. Its [paths].counts_json points at the
               count grids. Default: config/run_durham.toml, which references
               the bundled data/ground_truth/counts.json.
  OUTPUT_DIR   where to write CSVs; default = <config outdir>/bootstrap, or
               project-root/output/bootstrap when no config is given.

Data discovery is driven entirely by the config (RunInputs.counts_json). When
no config is supplied, the bundled data/ground_truth/counts.json is used.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CSV, DataFrames, Dates, Printf

include(joinpath(@__DIR__, "ground_truth.jl"))

# ─────────────────────────────────────────────────────────────
# Paths (config-driven; no hardcoded space_files)
# ─────────────────────────────────────────────────────────────
const _GT       = resolve_ground_truth(get(ARGS, 1, ""))
const counts_json = require_ground_truth(_GT, :counts_json, "counts.json")
const OUT_DIR   = get(ARGS, 2, _GT.bootstrap_dir)
mkpath(OUT_DIR)

# ─────────────────────────────────────────────────────────────
# Bootstrap defaults (manuscript-approved)
# ─────────────────────────────────────────────────────────────
const NBOOT      = 5000
const BLOCK_FRAC = 0.03
const SEED       = 42
const LEVEL      = 0.95

println("Bootstrap CI generation")
println("  nboot      = $NBOOT")
println("  block_frac = $BLOCK_FRAC  (block_side = 9 for Durham 324×263 grid)")
println("  seed       = $SEED")
println("  level      = $(round(Int, LEVEL*100))%")
println()

println("Loading counts.json from: $counts_json")

# ─────────────────────────────────────────────────────────────
# Table 1: Per-mission bootstrap CIs for all 36 rows
# ─────────────────────────────────────────────────────────────
println("\n=== Table 1: bootstrap_cr_cis_all36.csv ===")
println("  Running bootstrap for all (cover × return × mission × kernel) combinations …")
t0 = time()

cr_df = bootstrap_cr_table(
    counts_json;
    nboot      = NBOOT,
    block_frac = BLOCK_FRAC,
    seed       = SEED,
    level      = LEVEL,
    covers     = [:field, :decid, :conif],
    returns    = [:all, :ground],
    missions   = [:density, :speed, :const2, :const8],
    kernels    = [:G, :E],
)

elapsed = round(time() - t0; digits=1)
println("  Done in $(elapsed)s — $(nrow(cr_df)) rows")

out1 = joinpath(OUT_DIR, "bootstrap_cr_cis_all36.csv")
CSV.write(out1, cr_df)
println("  Wrote: $out1")

# Pretty-print key rows (Ground, KDE Epanechnikov)
println("\n  Key results (Ground returns, KDE-guided Epanechnikov density-aware):")
key_rows = filter(r ->
    r.Return == "Ground" &&
    r.Mission == "Density-aware" &&
    r.Kernel  == "E",
    cr_df
)
for row in eachrow(key_rows)
    @printf("    %-12s  CR=%.4f  95%% CI=[%.4f, %.4f]\n",
        row.Cover, row.CR, row.CI_lower, row.CI_upper)
end

# ─────────────────────────────────────────────────────────────
# Table 2: Paired difference CIs for narrative comparisons
# ─────────────────────────────────────────────────────────────
println("\n=== Table 2: bootstrap_cr_difference_cis.csv ===")
println("  Running paired bootstrap for narrative KDE-guided vs. Const comparisons …")

comparisons = default_narrative_comparisons(
    covers  = [:field, :decid, :conif],
    returns = [:ground, :all],
)
println("  $(length(comparisons)) comparisons defined")

t0 = time()
diff_df = bootstrap_cr_difference_table(
    counts_json,
    comparisons;
    nboot      = NBOOT,
    block_frac = BLOCK_FRAC,
    seed       = SEED,
    level      = LEVEL,
)
elapsed = round(time() - t0; digits=1)
println("  Done in $(elapsed)s — $(nrow(diff_df)) rows")

out2 = joinpath(OUT_DIR, "bootstrap_cr_difference_cis.csv")
CSV.write(out2, diff_df)
println("  Wrote: $out2")

# Pretty-print ground-return differences
println("\n  Ground-return differences (KDE-guided Epanechnikov vs. baselines):")
ground_diffs = filter(r -> r.Return == "Ground", diff_df)
for row in eachrow(ground_diffs)
    sig = (row.CI_lower > 0 || row.CI_upper < 0) ? " *" : ""
    @printf("    %-55s  Δ=%.4f  95%% CI=[%.4f, %.4f]%s\n",
        row.Comparison, row.CR_diff, row.CI_lower, row.CI_upper, sig)
end

# ─────────────────────────────────────────────────────────────
# Summary note
# ─────────────────────────────────────────────────────────────
println("""

=== Summary ===
  bootstrap_cr_cis_all36.csv    → $(nrow(cr_df)) rows
  bootstrap_cr_difference_cis.csv → $(nrow(diff_df)) rows
  Output directory: $OUT_DIR

Manuscript narrative (ground returns, KDE-guided Epanechnikov vs. baselines):
  Deciduous: positive difference, CI excludes zero → statistically significant
  Coniferous: positive difference, wide CI overlaps zero → directional but uncertain
  Field: negative difference, CI excludes zero → saturation effect confirmed
""")
println("=== Done ===")
