"""
    metrics.jl — Coverage and density-bin analysis helpers

Public API
----------
- `coverage_ratio(count_grid; min_returns=1)` — fraction of cells with ≥ min_returns
- `percent_in_bin(count_grid, lo, hi)` — fraction of cells in a return-count bin
- `density_bin_summary(count_grid; bins)` — named-tuple summary of multiple bins
- `per_cover_summary(counts_dict; covers, bins)` — per-cover table as NamedTuple vector
- `coverage_profile(count_grid, waypoints; axis=:y)` — 1-D coverage vs. position
- `flight_line_profile(count_grid, grid, line_wps)` — single-line heatmap helper

Background
----------
The primary outcome metric in the manuscript is the **coverage ratio (CR)**:
the fraction of 1 m² grid cells that contain at least one ground return.
High CR reflects uniform spatial sampling; low CR implies gaps in coverage
(common at high speed or in dense canopy where returns are scarce).
"""

# ---------------------------------------------------------------------------
# Coverage ratio
# ---------------------------------------------------------------------------

"""
    coverage_ratio(count_grid::AbstractMatrix{<:Real}; min_returns::Int=1) -> Float64

Compute the coverage ratio: the fraction of cells in `count_grid` that
contain at least `min_returns` LiDAR ground returns.

    CR = N_covered / N_total

where `N_covered = |{i,j : count_grid[i,j] ≥ min_returns}|`.

A value of 1.0 means every 1 m² cell was hit at least once;
a value near 0 means near-complete gaps.
"""
function coverage_ratio(count_grid::AbstractMatrix{<:Real};
                         min_returns::Int=1)
    n_total   = length(count_grid)
    n_total == 0 && return 0.0
    n_covered = count(c -> c >= min_returns, count_grid)
    return n_covered / n_total
end

# ---------------------------------------------------------------------------
# Percent-in-bin helpers
# ---------------------------------------------------------------------------

"""
    percent_in_bin(count_grid, lo, hi; include_zeros=false) -> Float64

Return the percentage (0–100) of cells whose return count falls in `[lo, hi]`.

If `include_zeros=false` (default) the denominator is the number of non-zero
cells (matches the manuscript's "percent density" metric). If `true`, the
denominator is all cells.
"""
function percent_in_bin(count_grid::AbstractMatrix{<:Real},
                         lo::Real, hi::Real;
                         include_zeros::Bool=false)
    cells = vec(count_grid)
    denom = include_zeros ? length(cells) : count(!iszero, cells)
    denom == 0 && return 0.0
    n_in  = count(c -> lo <= c <= hi, cells)
    return 100.0 * n_in / denom
end

"""
    density_bin_summary(count_grid; bins=[(1,5), (6,10), (11,Inf)],
                         include_zeros=false) -> NamedTuple

Compute the percentage of cells in each return-count bin.

Arguments
---------
- `bins`: vector of `(lo, hi)` tuples. Each pair defines a half-open interval.
- `include_zeros`: passed to `percent_in_bin`.

Returns a `NamedTuple` with keys `bin_<lo>_<hi>` for each bin plus
`coverage_ratio`, `mean_returns`, `median_returns`.
"""
function density_bin_summary(count_grid::AbstractMatrix{<:Real};
                              bins::AbstractVector=[(1,5),(6,10),(11,Inf)],
                              include_zeros::Bool=false)
    cells = Float64.(vec(count_grid))
    cr    = coverage_ratio(count_grid)

    bin_vals = Dict{String,Float64}()
    for (lo, hi) in bins
        key = "bin_$(Int(lo))_$(isinf(hi) ? "inf" : Int(hi))"
        bin_vals[key] = percent_in_bin(count_grid, lo, hi; include_zeros=include_zeros)
    end

    nz = cells[cells .> 0]
    mr  = isempty(nz) ? 0.0 : mean(nz)
    mdr = isempty(nz) ? 0.0 : median(nz)

    return (coverage_ratio=cr, mean_returns=mr, median_returns=mdr,
            bin_percents=bin_vals)
end

# ---------------------------------------------------------------------------
# Per-cover summary
# ---------------------------------------------------------------------------

"""
    per_cover_summary(counts_dict;
                       covers = ["Field", "Deciduous", "Coniferous"],
                       missions = nothing,
                       bins = [(1,5),(6,10),(11,Inf)]) -> Vector{NamedTuple}

Compute coverage statistics for each cover type and mission variant.

Arguments
---------
- `counts_dict`: nested dict `counts[cover][mission] = count_matrix`
- `covers`:      list of cover-type keys
- `missions`:    list of mission keys to include (nothing = all)
- `bins`:        return-count bins for `density_bin_summary`

Returns a `Vector` of `NamedTuple`s with fields:
  `cover`, `mission`, `coverage_ratio`, `mean_returns`, `median_returns`,
  `bin_percents`
"""
function per_cover_summary(counts_dict::AbstractDict;
                            covers  ::AbstractVector=["Field","Deciduous","Coniferous"],
                            missions::Union{Nothing,AbstractVector}=nothing,
                            bins    ::AbstractVector=[(1,5),(6,10),(11,Inf)])
    rows = NamedTuple[]
    for cover in covers
        haskey(counts_dict, cover) || continue
        mission_dict = counts_dict[cover]
        ks = isnothing(missions) ? collect(keys(mission_dict)) : missions
        for mission in ks
            haskey(mission_dict, mission) || continue
            cg   = mission_dict[mission]
            summ = density_bin_summary(cg; bins=bins)
            push!(rows, merge((cover=cover, mission=mission), summ))
        end
    end
    return rows
end

# ---------------------------------------------------------------------------
# Single-flight-line coverage profile
# ---------------------------------------------------------------------------

"""
    coverage_profile(count_grid::RasterGrid, waypoints::Vector{Waypoint};
                     axis=:y, bin_size=5.0) -> (positions, coverages)

Compute a 1-D coverage profile: fraction of covered cells in strip bins
along `axis` (`:x` or `:y`).

Arguments
---------
- `count_grid`:  A `RasterGrid` whose `Z` holds per-cell return counts
- `waypoints`:   Vector of `Waypoint`s (used to determine mission extent)
- `axis`:        `:y` (along-track) or `:x` (cross-track)
- `bin_size`:    Strip width (m)

Returns
-------
- `positions`:  Bin centres (m)
- `coverages`:  Coverage ratio per strip bin
"""
function coverage_profile(count_grid::RasterGrid,
                           waypoints::Vector{Waypoint};
                           axis::Symbol=:y,
                           bin_size::Real=5.0)
    axis in (:x, :y) || throw(ArgumentError("axis must be :x or :y"))
    bin_size > 0 || throw(ArgumentError("bin_size must be > 0"))

    coords = axis === :y ? count_grid.ys : count_grid.xs
    lo, hi = first(coords), last(coords)
    edges  = collect(range(lo, hi; step=Float64(bin_size)))
    isempty(edges) && push!(edges, hi)
    push!(edges, hi + eps())

    positions  = Float64[]
    coverages  = Float64[]

    xs, ys, Z = count_grid.xs, count_grid.ys, count_grid.Z

    for k in 1:length(edges)-1
        e1, e2 = edges[k], edges[k+1]
        push!(positions, (e1 + e2) / 2)

        if axis === :y
            iy_lo = searchsortedfirst(ys, e1)
            iy_hi = searchsortedlast(ys,  e2)
            sub   = iy_lo <= iy_hi ? Z[iy_lo:iy_hi, :] : Matrix{Float64}(undef,0,0)
        else
            ix_lo = searchsortedfirst(xs, e1)
            ix_hi = searchsortedlast(xs,  e2)
            sub   = ix_lo <= ix_hi ? Z[:, ix_lo:ix_hi] : Matrix{Float64}(undef,0,0)
        end

        push!(coverages, isempty(sub) ? 0.0 : coverage_ratio(sub))
    end

    return positions, coverages
end

# ---------------------------------------------------------------------------
# Flight-line heatmap helper
# ---------------------------------------------------------------------------

"""
    flight_line_profile(count_grid::RasterGrid, line_wps::Vector{Waypoint})
        -> (along_track_m, densities, speeds)

Extract a 1-D along-track profile for a single flight line.

Arguments
---------
- `count_grid`:  `RasterGrid` with return-count data
- `line_wps`:    Waypoints from a single flight line (same `line_id`)

Returns
-------
- `along_track_m`:  Cumulative along-track distance (m) at each waypoint
- `densities`:      Sampled `count_grid.Z` at each waypoint location
- `speeds`:         Waypoint speed assignments (m/s)
"""
function flight_line_profile(count_grid::RasterGrid,
                              line_wps::Vector{Waypoint})
    isempty(line_wps) && return (Float64[], Float64[], Float64[])
    n = length(line_wps)
    along = zeros(Float64, n)
    for i in 2:n
        Δ = hypot(line_wps[i].x - line_wps[i-1].x,
                  line_wps[i].y - line_wps[i-1].y)
        along[i] = along[i-1] + Δ
    end
    densities = [sample_density(count_grid, w.x, w.y) for w in line_wps]
    speeds    = [w.speed for w in line_wps]
    return along, densities, speeds
end
