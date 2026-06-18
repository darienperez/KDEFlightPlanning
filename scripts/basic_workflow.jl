"""
    scripts/basic_workflow.jl

End-to-end demonstration of KDEFlightPlanning.jl using a synthetic density
surface. This script can be run directly:

    julia --project=. scripts/basic_workflow.jl

It exercises all five mission variants described in the manuscript and exports
waypoints to CSV.
"""

using KDEFlightPlanning

# ---------------------------------------------------------------------------
# 1. Build a synthetic 200×200 m density grid
# ---------------------------------------------------------------------------
println("Building synthetic density grid...")

grid = synthetic_gaussian_grid(
    nx=80, ny=80,
    xmin=0.0, xmax=200.0,
    ymin=0.0, ymax=200.0,
    # Two Gaussian peaks mimicking deciduous (broad) + coniferous (narrow)
    centers=[(80.0, 100.0), (150.0, 100.0)],
    sigmas =[(25.0, 25.0),  (10.0, 10.0)],
    weights=[1.0, 0.9]
)

println("  Grid: $(size(grid)) (Ny × Nx), " *
        "density range = $(round.(extrema(grid.Z); digits=3))")

# ---------------------------------------------------------------------------
# 2. Define mission configurations (all five manuscript variants)
# ---------------------------------------------------------------------------

# Paper workflow: 40 m line spacing (the 20 m value was used in an earlier
# speed-troubleshooting step that has been removed).
configs = [
    FlightConfig(ConstantSpeed(2.0),            80.0, "Constant 2 m/s";
                 line_spacing=40.0),
    FlightConfig(ConstantSpeed(8.0),            80.0, "Constant 8 m/s";
                 line_spacing=40.0),
    FlightConfig(KDEGuidedSpeed(grid; vmin=2.0, vmax=8.0),
                                                80.0, "KDE-guided Gaussian";
                 kernel=:gaussian,     line_spacing=40.0),
    FlightConfig(KDEGuidedSpeed(grid; vmin=2.0, vmax=8.0),
                                                80.0, "KDE-guided Epanechnikov";
                 kernel=:epanechnikov, line_spacing=40.0),
    FlightConfig(CurvatureGuidedSpeed(grid; vmin=2.0, vmax=8.0,
                                      alpha=1.0, lambda=1.5, eta=2.0,
                                      spacing_min=2.0, spacing_max=40.0),
                                                80.0, "Curvature-spaced KDE-guided";
                 kernel=:gaussian,     line_spacing=40.0),
]

# ---------------------------------------------------------------------------
# 3. Generate waypoints and print summary
# ---------------------------------------------------------------------------

results = Dict{String, Vector{Waypoint}}()

println("\nGenerating waypoints for each mission:")
println("-" ^ 60)
println(rpad("Label", 35) * rpad("N_wps", 8) * rpad("v_min", 8) * "v_max")
println("-" ^ 60)

for cfg in configs
    wps = plan_mission(grid, cfg;
                       seconds_per_wp=1.0,
                       spacing_min=2.0,
                       spacing_max=20.0,
                       include_vertices=true)
    results[cfg.label] = wps

    vmin_obs = minimum(w.speed for w in wps)
    vmax_obs = maximum(w.speed for w in wps)
    println(rpad(cfg.label, 35) *
            rpad(length(wps), 8) *
            rpad(round(vmin_obs; digits=2), 8) *
            round(vmax_obs; digits=2))
end

println("-" ^ 60)

# ---------------------------------------------------------------------------
# 4. Coverage ratio on synthetic "count" grid (all ones)
# ---------------------------------------------------------------------------
println("\nCoverage ratio (synthetic all-ones count grid):")
count_grid = ones(Int, 80, 80)
cr = coverage_ratio(count_grid)
println("  CR = $cr (expected 1.0 since every cell is hit)")

# ---------------------------------------------------------------------------
# 5. Export waypoints to CSV
# ---------------------------------------------------------------------------
println("\nExporting waypoints to /tmp/...")
for (label, wps) in results
    safe_name = replace(label, " " => "_", "/" => "-")
    path = "/tmp/$(safe_name)_waypoints.csv"
    write_waypoints_csv(path, wps)
    println("  → $(path) ($(length(wps)) rows)")
end

println("\nDone.")
