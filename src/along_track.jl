"""
    along_track.jl — Coverage-vs-position analysis using stable along-track bins

## Purpose

The line-scan coverage analysis requires a stable along-track coordinate to
compare coverage across missions.  A naive column-profile approach is sensitive
to flight-line alignment: if two missions fly slightly different x-ranges, the
comparison is distorted.

This module implements a cleaner approach:

1. Only use the **intersection** of the actual x-ranges flown by all compared
   missions on each line (true common-coverage overlap).
2. Bin the intersection into fixed-width along-track bins (5–10 m default).
3. Apply a **minimum support** filter: bins with fewer than `min_cells` cells
   are excluded from CR computation.
4. Apply **gap-breaking**: contiguous segments of valid bins are separated
   wherever a gap larger than `max_gap_m` metres exists between adjacent
   bin centres.

If the current figure remains weak after these improvements, the module
generates a textual note recommending heatmaps and band-level CR instead.
"""

# ---------------------------------------------------------------------------
# Along-track binned CR
# ---------------------------------------------------------------------------

"""
    along_track_cr(count_grid::AbstractMatrix,
                   xs::AbstractVector{<:Real},
                   x_lo::Real, x_hi::Real;
                   bin_size_m::Real = 5.0,
                   min_cells::Int   = 3,
                   min_returns::Int = 1)
        -> (bin_centres, coverages, n_cells_per_bin)

Compute coverage ratio in fixed-width along-track (x-axis) bins within the
intersection window [x_lo, x_hi].

Arguments
---------
- `count_grid`:  (Ny × Nx) count matrix.
- `xs`:          x-coordinate vector, length Nx (ascending).
- `x_lo`, `x_hi`: x-range of the common-overlap window.
- `bin_size_m`:  Bin width in metres (default 5).
- `min_cells`:   Minimum cells per bin for inclusion (default 3).
- `min_returns`: Minimum returns to call a cell covered (default 1).

Returns
-------
- `bin_centres`: x-coordinate of each valid bin centre (m).
- `coverages`:   Coverage ratio per bin.
- `n_cells_per_bin`: Number of cells in each bin (for support assessment).
"""
function along_track_cr(count_grid::AbstractMatrix,
                         xs::AbstractVector{<:Real},
                         x_lo::Real, x_hi::Real;
                         bin_size_m::Real = 5.0,
                         min_cells::Int   = 3,
                         min_returns::Int = 1)
    bin_size_m > 0 || throw(ArgumentError("bin_size_m must be > 0"))
    x_lo < x_hi  || throw(ArgumentError("x_lo must be < x_hi"))

    edges = collect(range(Float64(x_lo), Float64(x_hi); step=Float64(bin_size_m)))
    push!(edges, Float64(x_hi) + eps())   # close the last bin

    centres  = Float64[]
    crs      = Float64[]
    n_cells  = Int[]

    for b in 1:length(edges)-1
        e1, e2 = edges[b], edges[b+1]
        cx_lo = searchsortedfirst(xs, e1)
        cx_hi = searchsortedlast(xs, e2 - eps())
        cx_lo > cx_hi && continue

        sub = count_grid[:, cx_lo:cx_hi]
        nc  = length(sub)
        nc < min_cells && continue   # minimum support filter

        ncov = count(c -> c >= min_returns, sub)
        push!(centres, (e1 + e2) / 2)
        push!(crs,     ncov / nc)
        push!(n_cells, nc)
    end

    return centres, crs, n_cells
end

# ---------------------------------------------------------------------------
# Gap-breaking: split at large gaps
# ---------------------------------------------------------------------------

"""
    split_at_gaps(centres, crs, n_cells; max_gap_m::Real = 15.0)
        -> Vector{NamedTuple}

Split a 1-D along-track profile at positions where adjacent bin centres are
more than `max_gap_m` metres apart.  Returns a vector of named tuples, each
representing one contiguous segment:
`(centres, coverages, n_cells)`.

Gaps larger than `max_gap_m` are typical at line edges or missing-data zones
and should not be smoothed over.
"""
function split_at_gaps(centres::AbstractVector, crs::AbstractVector,
                        n_cells::AbstractVector;
                        max_gap_m::Real = 15.0)
    n = length(centres)
    n == 0 && return NamedTuple[]

    segments = NamedTuple[]
    seg_start = 1

    for i in 2:n
        gap = centres[i] - centres[i-1]
        if gap > max_gap_m
            push!(segments, (
                centres  = centres[seg_start:i-1],
                coverages = crs[seg_start:i-1],
                n_cells  = n_cells[seg_start:i-1],
            ))
            seg_start = i
        end
    end
    push!(segments, (
        centres   = centres[seg_start:end],
        coverages = crs[seg_start:end],
        n_cells   = n_cells[seg_start:end],
    ))
    return segments
end

# ---------------------------------------------------------------------------
# Multi-mission along-track comparison
# ---------------------------------------------------------------------------

"""
    multi_mission_along_track(count_grids::AbstractDict,
                               xs::AbstractVector{<:Real};
                               x_lo::Real        = minimum(xs),
                               x_hi::Real        = maximum(xs),
                               bin_size_m::Real  = 5.0,
                               min_cells::Int    = 3,
                               max_gap_m::Real   = 15.0,
                               missions::AbstractVector = collect(keys(count_grids)))
        -> DataFrame

Compute per-bin along-track coverage ratios for multiple missions over a
common x-window.

Returns a `DataFrame` with columns:
`mission, bin_centre_m, cr, n_cells`.
Only bins with `n_cells ≥ min_cells` are included.

Use `split_at_gaps` on individual mission profiles if gap-breaking is needed
for figure generation (this function returns all valid bins without gap marks).
"""
function multi_mission_along_track(count_grids::AbstractDict,
                                    xs::AbstractVector{<:Real};
                                    x_lo::Real       = minimum(xs),
                                    x_hi::Real       = maximum(xs),
                                    bin_size_m::Real = 5.0,
                                    min_cells::Int   = 3,
                                    max_gap_m::Real  = 15.0,
                                    missions::AbstractVector = collect(keys(count_grids)))
    rows = NamedTuple[]
    for m in missions
        haskey(count_grids, m) || continue
        cg = count_grids[m]
        centres, crs, nc = along_track_cr(cg, xs, x_lo, x_hi;
                                           bin_size_m=bin_size_m,
                                           min_cells=min_cells)
        for (c, cr, n) in zip(centres, crs, nc)
            push!(rows, (; mission=m, bin_centre_m=c, cr=cr, n_cells=n))
        end
    end
    isempty(rows) && return DataFrame()
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Weakness note generator
# ---------------------------------------------------------------------------

"""
    along_track_weakness_note(df::DataFrame;
                               min_valid_bins::Int = 10) -> String

If the along-track profile has fewer than `min_valid_bins` valid bins for
any mission, or if the standard deviation of CR across bins is very low
(< 0.01), generate a textual note recommending alternative visualisations.

This is called automatically by `export_along_track_csv` if weakness is detected.
"""
function along_track_weakness_note(df::DataFrame;
                                    min_valid_bins::Int = 10)
    lines = String[]
    is_weak = false

    for m in unique(df[!, :mission])
        sub = filter(r -> r.mission == m, df)
        n_bins = nrow(sub)
        if n_bins < min_valid_bins
            push!(lines, "  Mission '$m': only $n_bins valid bins (< $min_valid_bins threshold).")
            is_weak = true
        end
        if n_bins > 1
            cr_std = std(sub[!, :cr])
            if cr_std < 0.01
                push!(lines, "  Mission '$m': CR std = $(round(cr_std; digits=4)) — little along-track variation.")
                is_weak = true
            end
        end
    end

    is_weak || return ""
    return join(vcat(
        ["Along-track figure weakness detected:"],
        lines,
        ["",
         "RECOMMENDATION: If this figure is weak, consider replacing or supplementing with:",
         "  1. Spatial heatmaps of per-cell return counts for each mission (side-by-side).",
         "  2. Band-level CR: compute CR within fixed latitudinal bands of the count grid.",
         "     This preserves spatial structure without the sensitivity to along-track",
         "     alignment that affects column-profile comparisons.",
         "  3. Scatter plot of KDE density vs. coverage ratio per cell.",
         "See: KDEFlightPlanning.jl/output/notes/pipeline_generalization_note.md",
        ]
    ), "\n")
end

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

"""
    export_along_track_csv(df::DataFrame, path::AbstractString;
                            weakness_threshold::Int = 10) -> path

Write the multi-mission along-track CR table to CSV.  If weakness is
detected, appends a note file alongside the CSV.
"""
function export_along_track_csv(df::DataFrame, path::AbstractString;
                                 weakness_threshold::Int = 10)
    mkpath(dirname(abspath(path)))
    CSV.write(path, df)

    note = along_track_weakness_note(df; min_valid_bins=weakness_threshold)
    if !isempty(note)
        note_path = path * "_weakness_note.txt"
        open(note_path, "w") do io; println(io, note); end
        @info "Along-track weakness note written to: $note_path"
    end
    return path
end
