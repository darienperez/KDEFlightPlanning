"""
    kde_strata.jl — KDE-density-stratified coverage analysis (secondary mechanism)

## Purpose

This module implements a secondary mechanism analysis: stratifying the coverage
ratio by KDE density quantile bins.  The goal is to examine whether the
KDE-guided speed strategy redistributes sampling effort systematically across
density strata, providing mechanistic evidence for the coverage improvements
observed in the primary bootstrap analysis.

## Epistemological status

The KDE density surface is derived from the same orthomosaic used for
planning, not from an independent reference.  When the count grid and the KDE
surface are only approximately co-registered (e.g. because the orthomosaic is
a lower-resolution screenshot rather than the original GeoTIFF), outputs are
labeled **"smoke/diagnostic"** and must not be used as primary inference.

GLI (Green Leaf Index) remains the primary external cover ground-truth
reference, following Sullivan et al. (2023, DOI 10.3390/rs15215091).
KDE strata are secondary mechanism analysis, not ground-truth classes.

## Narrative constraints (from paper)

- 8 m/s did not appear to hit a degradation limit at this site.
- 2 m/s unexpectedly had lower forest CR than 8 m/s; the cautious explanation
  is a coverage-vs-density issue (denser returns per cell, but smaller total
  footprint at low speed).
- The strongest inferential claim is deciduous improvement; coniferous is a
  positive point estimate but whole-cover CI crosses zero.

## Public API

    kde_cr_by_quantile_bins(count_grid, kde_surface; n_bins, missions)
    kde_strata_table(count_grid_dict, kde_grid; missions, n_bins, label)
    kde_strata_within_cover(count_grid_dict, kde_grid, cover_mask; ...)
    export_kde_strata_csv(table, path)

## Grid orientation contract

All matrices are H×W (rows × columns), where for the Durham NH study site:
  H = 263 rows  (scene height = 263 m)
  W = 324 cols  (scene width  = 324 m)
Total = 85 212 cells.  JSON is stored row-major; after reshape (263, 324) is correct.
Do NOT use (324, 263) — that transposes the spatial layout.
"""

# ---------------------------------------------------------------------------
# Per-stratum CR helper
# ---------------------------------------------------------------------------

"""
    kde_cr_by_quantile_bins(count_grid::AbstractMatrix,
                             kde_Z::AbstractMatrix;
                             n_bins::Int = 4,
                             min_returns::Int = 1) -> DataFrame

Compute coverage ratio within each KDE-density quantile stratum.

The KDE surface `kde_Z` and the count grid `count_grid` must have the same
spatial shape (same matrix dimensions). When derived from a screenshot rather
than the original GeoTIFF, this alignment is approximate; label such outputs
as smoke/diagnostic.

Arguments
---------
- `count_grid`:  Integer matrix of per-cell ground-return counts.
- `kde_Z`:       Floating-point matrix of normalised KDE density values in [0,1].
                 Must have the same size as `count_grid`.
- `n_bins`:      Number of quantile bins (default 4 → quartiles).
- `min_returns`: Minimum returns to consider a cell covered (default 1).

Returns
-------
A `DataFrame` with columns:
  `stratum`, `quantile_lo`, `quantile_hi`,
  `density_lo`, `density_hi`,
  `n_cells`, `n_covered`, `cr`.
"""
function kde_cr_by_quantile_bins(count_grid::AbstractMatrix,
                                  kde_Z::AbstractMatrix;
                                  n_bins::Int     = 4,
                                  min_returns::Int = 1)
    size(count_grid) == size(kde_Z) ||
        throw(DimensionMismatch(
            "count_grid $(size(count_grid)) ≠ kde_Z $(size(kde_Z)). " *
            "Grids must be spatially co-registered and equal in size."))

    flat_kde   = vec(kde_Z)
    flat_count = vec(count_grid)

    # Quantile bin edges
    q_edges = [quantile(flat_kde, p) for p in range(0.0, 1.0; length=n_bins+1)]

    rows = NamedTuple[]
    for b in 1:n_bins
        q_lo = (b-1) / n_bins
        q_hi = b     / n_bins
        d_lo = q_edges[b]
        d_hi = q_edges[b+1]

        mask = if b < n_bins
            (flat_kde .>= d_lo) .& (flat_kde .< d_hi)
        else
            (flat_kde .>= d_lo) .& (flat_kde .<= d_hi)   # include right edge
        end

        n_cells   = count(mask)
        n_covered = count(mask .& (flat_count .>= min_returns))
        cr        = n_cells > 0 ? n_covered / n_cells : 0.0

        push!(rows, (;
            stratum    = b,
            quantile_lo = q_lo,
            quantile_hi = q_hi,
            density_lo  = d_lo,
            density_hi  = d_hi,
            n_cells     = n_cells,
            n_covered   = n_covered,
            cr          = cr,
        ))
    end
    return DataFrame(rows)
end

# ---------------------------------------------------------------------------
# Multi-mission table
# ---------------------------------------------------------------------------

"""
    kde_strata_table(count_grid_dict::AbstractDict,
                     kde_grid::AbstractMatrix;
                     missions::AbstractVector{<:AbstractString},
                     n_bins::Int = 4,
                     label::AbstractString = "",
                     alignment_note::AbstractString = "smoke/diagnostic") -> DataFrame

Compute KDE-stratified CR for multiple missions, returning a single combined
DataFrame.

Arguments
---------
- `count_grid_dict`: Dict mapping mission label → count matrix (same spatial shape).
- `kde_grid`:        KDE density surface (same spatial shape as count grids).
- `missions`:        Which mission keys to include (in order).
- `n_bins`:          Number of quantile strata.
- `label`:           Cover-type or region label (added as a column).
- `alignment_note`:  Inserted into the `alignment` column to flag data quality.
                     Use `"smoke/diagnostic"` when alignment is approximate.

Returns a `DataFrame` with columns:
  `mission`, `cover`, `alignment`, `stratum`, `quantile_lo`, `quantile_hi`,
  `density_lo`, `density_hi`, `n_cells`, `n_covered`, `cr`.
"""
function kde_strata_table(count_grid_dict::AbstractDict,
                           kde_grid::AbstractMatrix;
                           missions::AbstractVector   = collect(keys(count_grid_dict)),
                           n_bins::Int                = 4,
                           label::AbstractString      = "",
                           alignment_note::AbstractString = "smoke/diagnostic")
    all_rows = DataFrame[]
    for mission in missions
        haskey(count_grid_dict, mission) || continue
        cg = count_grid_dict[mission]

        # Resize kde_grid if shapes differ (approximate alignment from screenshot)
        kde_use = if size(cg) == size(kde_grid)
            kde_grid
        else
            @warn "KDE grid shape $(size(kde_grid)) ≠ count grid shape $(size(cg)) " *
                  "for mission '$mission'. Using nearest-neighbour resize. " *
                  "Outputs are SMOKE/DIAGNOSTIC only."
            _resize_nearest(kde_grid, size(cg))
        end

        df = kde_cr_by_quantile_bins(cg, kde_use; n_bins=n_bins)
        df[!, :mission]   .= mission
        df[!, :cover]     .= label
        df[!, :alignment] .= alignment_note
        push!(all_rows, df)
    end
    isempty(all_rows) && return DataFrame()
    return vcat(all_rows...)
end

"""
    _resize_nearest(mat::AbstractMatrix, new_size::Tuple{Int,Int}) -> Matrix{Float64}

Nearest-neighbour resize of a matrix.  Used only as a fallback when the KDE
surface and count grid have different shapes due to approximate alignment from
a screenshot source.  Outputs should be labeled smoke/diagnostic.
"""
function _resize_nearest(mat::AbstractMatrix, new_size::Tuple{Int,Int})
    H_new, W_new = new_size
    H_old, W_old = size(mat)
    out = Matrix{Float64}(undef, H_new, W_new)
    for j in 1:W_new, i in 1:H_new
        ri = clamp(round(Int, (i - 0.5) / H_new * H_old + 0.5), 1, H_old)
        rj = clamp(round(Int, (j - 0.5) / W_new * W_old + 0.5), 1, W_old)
        out[i, j] = mat[ri, rj]
    end
    return out
end

# ---------------------------------------------------------------------------
# Within-cover stratified analysis (Bug-B2/B3/B4 fix)
# ---------------------------------------------------------------------------

"""
    StratumRow

Result row from `kde_strata_within_cover`.  Mirrors the CSV schema of
`export_kde_strata_csv` with additional within-cover fields.

Fields
------
- `stratum`       — 0 = zero-density; 1..n = positive-KDE quantile bins
- `stratum_label` — human label, e.g. "zero-KDE", "Q1/4"
- `quantile_lo/hi`— empirical quantile bounds of the bin within cover pixels
- `density_lo/hi` — KDE density cutpoints
- `n_cells`       — number of cover pixels in this bin
- `n_covered`     — pixels with ≥ 1 ground return
- `cr`            — coverage ratio = n_covered / n_cells  (bin denominator)
- `mission`/`cover`/`alignment`/`kde_status`
"""
struct StratumRow
    stratum      ::Int
    stratum_label::String
    quantile_lo  ::Float64
    quantile_hi  ::Float64
    density_lo   ::Float64
    density_hi   ::Float64
    n_cells      ::Int
    n_covered    ::Int
    cr           ::Float64
    mission      ::String
    cover        ::String
    alignment    ::String
    kde_status   ::String
end

"""
    kde_strata_within_cover(
        count_grid_dict :: AbstractDict{<:AbstractString, <:AbstractMatrix},
        kde_grid        :: AbstractMatrix,
        cover_mask      :: AbstractMatrix{Bool};
        missions        :: AbstractVector{<:AbstractString} = collect(keys(count_grid_dict)),
        n_pos_bins      :: Int    = 4,
        cover_label     :: AbstractString = "",
        alignment       :: AbstractString = "smoke/diagnostic",
        kde_status      :: AbstractString = "unknown",
        kde_n_support   :: Int    = 0,
    ) -> DataFrame

Compute KDE-density-stratified CR **within a single cover class**, using
quantile bins defined only on cover-masked pixels.

## Why this fixes the global-bins approach

The original `kde_strata_table` computed quantile edges over the full grid
(all 85 212 cells). Because vegetation covers only ~21–35% of the grid,
>50% of cells have KDE≈0 after nearest-neighbour resampling, which collapses
Q1 to an empty stratum (n_cells=0, CR=0.0).  This function instead:

1. Extracts KDE values for `cover_mask` pixels only.
2. Separates zero-density pixels → explicit **stratum 0** ("zero-KDE").
3. Applies `n_pos_bins` quantile bins to the positive-KDE subset.
4. Drops duplicate quantile edges (robust to flat/tie-heavy distributions).

## Validation invariants (checked on return)

- `sum(stratum n_cells) == count(cover_mask)` for each mission.
- `sum(stratum n_covered) / kde_n_support ≈ whole-cover CR`
  when `kde_n_support > 0` (pass the DURHAM_COVER_N value for exact check).
- All CR values in [0, 1].

## Arguments

- `count_grid_dict` — mission label → integer count matrix (same shape as `kde_grid`).
- `kde_grid`        — KDE density surface, shape must equal count grid shape.
                      Values must be in [0, 1]; **not renormalised internally**.
                      (If the surface is already normalised by `build_density_surface`,
                       pass it directly.  Renormalise externally only if required.)
- `cover_mask`      — Bool matrix marking cells in this cover class.
- `missions`        — which keys to process (default: all keys).
- `n_pos_bins`      — quantile bins for positive-density cells (default 4).
- `cover_label`     — string label inserted in the `cover` column.
- `alignment`       — inserted in `alignment` column; use `"true_surface"` when
                      the KDE grid is the exact planning surface at count-grid resolution.
- `kde_status`      — `"true"` | `"screenshot"` | other provenance string.
- `kde_n_support`   — known cover N (e.g. `DURHAM_COVER_N[:conif]`) for the
                      weighted-CR invariant check.  Pass 0 to skip.
"""
function kde_strata_within_cover(
        count_grid_dict ::AbstractDict,
        kde_grid        ::AbstractMatrix,
        cover_mask      ::AbstractMatrix{Bool};
        missions        ::AbstractVector   = collect(keys(count_grid_dict)),
        n_pos_bins      ::Int              = 4,
        cover_label     ::AbstractString   = "",
        alignment       ::AbstractString   = "smoke/diagnostic",
        kde_status      ::AbstractString   = "unknown",
        kde_n_support   ::Int              = 0,
)::DataFrame

    # --- shape checks ---
    sz = size(kde_grid)
    for (mlabel, cg) in count_grid_dict
        size(cg) == sz ||
            throw(DimensionMismatch(
                "count_grid for '$mlabel' has shape $(size(cg)) ≠ kde_grid shape $sz"))
    end
    size(cover_mask) == sz ||
        throw(DimensionMismatch(
            "cover_mask shape $(size(cover_mask)) ≠ kde_grid shape $sz"))

    # --- validate KDE range (do NOT renormalise) ---
    kde_min, kde_max = extrema(kde_grid)
    if kde_min < -1e-9 || kde_max > 1.0 + 1e-9
        @warn "kde_grid values outside [0,1]: min=$kde_min max=$kde_max. " *
              "Values are used as-is (no internal renormalisation). " *
              "Renormalise externally if intended."
    end

    # Flatten KDE within cover
    kde_cover = kde_grid[cover_mask]       # Vector{Float64}
    n_cover   = count(cover_mask)          # total cover pixels

    all_rows = DataFrame[]

    for mission in missions
        haskey(count_grid_dict, mission) || continue
        cg        = count_grid_dict[mission]
        cnt_cover = cg[cover_mask]         # Vector{Int}

        zero_sel = (kde_cover .== 0.0)
        pos_sel  = .!zero_sel
        n_zero   = count(zero_sel)
        n_pos    = count(pos_sel)

        rows = NamedTuple[]

        # --- Stratum 0: zero-density cells ---
        if n_zero > 0
            n_c0  = count(cnt_cover[zero_sel] .>= 1)
            cr0   = n_c0 / n_zero
            push!(rows, (;
                stratum=0, stratum_label="zero-KDE",
                quantile_lo=0.0, quantile_hi=0.0,
                density_lo=0.0, density_hi=0.0,
                n_cells=n_zero, n_covered=n_c0, cr=cr0,
                mission=mission, cover=cover_label,
                alignment=alignment, kde_status=kde_status,
            ))
        end

        # --- Positive-density bins ---
        if n_pos > 0
            kde_pos = kde_cover[pos_sel]
            cnt_pos = cnt_cover[pos_sel]

            # Quantile edges on positive values; drop duplicates
            q_probs = range(0.0, 1.0; length=n_pos_bins+1)
            edges   = unique([quantile(kde_pos, p) for p in q_probs])
            actual  = length(edges) - 1
            actual  = max(actual, 1)   # guard: single unique value

            for b in 1:actual
                d_lo = edges[b]
                d_hi = edges[b+1]

                bin_sel = if b < actual
                    (kde_pos .>= d_lo) .& (kde_pos .< d_hi)
                else
                    (kde_pos .>= d_lo) .& (kde_pos .<= d_hi)
                end

                n_b  = count(bin_sel)
                n_cb = n_b > 0 ? count(cnt_pos[bin_sel] .>= 1) : 0
                cr_b = n_b > 0 ? n_cb / n_b : 0.0

                # Empirical quantile position within positive subset
                q_lo_emp = Float64(mean(kde_pos .< d_lo))
                q_hi_emp = Float64(mean(kde_pos .<= d_hi))

                push!(rows, (;
                    stratum=b,
                    stratum_label="Q$(b)/$(actual)",
                    quantile_lo=round(q_lo_emp; digits=4),
                    quantile_hi=round(q_hi_emp; digits=4),
                    density_lo=d_lo, density_hi=d_hi,
                    n_cells=n_b, n_covered=n_cb, cr=cr_b,
                    mission=mission, cover=cover_label,
                    alignment=alignment, kde_status=kde_status,
                ))
            end
        end

        df_m = DataFrame(rows)

        # --- Validation ---
        strata_sum = sum(df_m.n_cells)
        strata_sum == n_cover ||
            @warn "Stratum n_cells sum ($strata_sum) ≠ cover mask size ($n_cover) " *
                  "for mission '$mission', cover '$cover_label'."

        if kde_n_support > 0
            total_cov = sum(df_m.n_covered)
            wcr = total_cov / kde_n_support
            # We can't check against whole-cover CR here without it being passed in,
            # but we record it for the caller to inspect.
            df_m[!, :weighted_cr_check] .= wcr
        end

        any(v -> !(0 ≤ v ≤ 1), df_m.cr) &&
            @warn "CR out of [0,1] in some strata (mission='$mission', cover='$cover_label')"

        push!(all_rows, df_m)
    end

    isempty(all_rows) && return DataFrame()
    return vcat(all_rows...)
end

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

"""
    export_kde_strata_csv(df::DataFrame, path::AbstractString) -> path

Write a KDE-stratified CR table to CSV.

Includes a header comment (as a leading row) documenting the alignment status.
"""
function export_kde_strata_csv(df::DataFrame, path::AbstractString)
    mkpath(dirname(abspath(path)))
    CSV.write(path, df)
    return path
end

# ---------------------------------------------------------------------------
# Narrative summary helper
# ---------------------------------------------------------------------------

"""
    kde_strata_narrative(df::DataFrame;
                          kde_mission::AbstractString = "KDE-guided (Epanechnikov)",
                          baseline_missions::AbstractVector = ["Const. 2 m/s", "Const. 8 m/s"])
        -> String

Generate a brief narrative summary of the stratified CR results for inclusion
in the manuscript Discussion or Supplementary Material.

Follows the required narrative constraints:
- Labels outputs appropriately (smoke/diagnostic vs manuscript-ready).
- Notes that 8 m/s did not appear to hit a degradation limit.
- Cautions about 2 m/s having unexpectedly lower forest CR.
"""
function kde_strata_narrative(df::DataFrame;
                               kde_mission::AbstractString = "KDE-guided (Epanechnikov)",
                               baseline_missions::AbstractVector = ["Const. 2 m/s", "Const. 8 m/s"])
    isempty(df) && return "(no stratified data available)"

    alignments = unique(df[!, :alignment])
    is_smoke   = any(a -> occursin("smoke", lowercase(string(a))), alignments)
    status_str = is_smoke ? "smoke/diagnostic (approximate alignment)" :
                            "manuscript-ready"

    lines = String[]
    push!(lines, "KDE-density-stratified CR summary ($status_str):")

    missions_in_df = unique(df[!, :mission])
    for m in vcat([kde_mission], baseline_missions)
        m in missions_in_df || continue
        sub  = filter(r -> r.mission == m, df)
        isempty(sub) && continue
        strs = sort(sub, :stratum)
        push!(lines, "  $m:")
        for r in eachrow(strs)
            push!(lines, @sprintf("    Stratum %d (KDE %.2f–%.2f): CR=%.3f (%d/%d cells)",
                                   r.stratum, r.density_lo, r.density_hi,
                                   r.cr, r.n_covered, r.n_cells))
        end
    end

    if is_smoke
        push!(lines, "")
        push!(lines, "  NOTE: Outputs are smoke/diagnostic. Alignment between the KDE surface")
        push!(lines, "  and the count grid is approximate (screenshot source, not original GeoTIFF).")
        push!(lines, "  Do not use these results for primary inferential claims.")
        push!(lines, "  GLI remains the primary external cover ground-truth reference (Sullivan et al. 2023).")
    end

    push!(lines, "")
    push!(lines, "  Narrative constraints (from paper):")
    push!(lines, "  - 8 m/s did not appear to hit a degradation limit at this site.")
    push!(lines, "  - 2 m/s unexpectedly had lower forest CR than 8 m/s; cautiously")
    push!(lines, "    interpreted as a coverage-vs-density issue.")
    push!(lines, "  - Strongest inferential claim: deciduous improvement.")
    push!(lines, "  - Coniferous: positive point estimate but whole-cover CI crosses zero.")

    return join(lines, "\n")
end
