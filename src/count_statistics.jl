"""
    count_statistics.jl — Summary statistics for LiDAR ground-return count grids

This module replicates the manuscript's table statistics from `counts.json` count
matrices, matching `stats_and_coverage.csv` exactly.

## Semantics (aligned with `lidar.jl`)

| Term     | Definition                                                             |
|----------|------------------------------------------------------------------------|
| `mask`   | `BitMatrix` — cells belonging to a cover-type polygon (from raster)   |
| `valid`  | `mask .& (counts .> 0)` — hit cells within the cover zone             |
| `ncells` | `count(valid)` — number of positive cells                              |
| `N`      | `count(mask)` — total cells in the cover polygon (support denominator) |
| `CR`     | `ncells / N` — coverage ratio                                          |
| `Q1/Q2/Q3/IQR/Mean` | computed on `counts[valid]` (positive cells only)        |
| `CV`     | `std(v; corrected=true) / mean(v)` (sample std, ddof=1)               |
| `Gini`   | Gini coefficient on `counts[valid]`; see `gini_coefficient`            |
| `MoranI` | Moran's I with **queen (8-neighbor) adjacency**, restricted to `valid` |

## Key assumption: the `mask` parameter

`N` is the number of 1 m² raster cells inside the cover-type polygon — a fixed
spatial quantity that comes from the classified raster (`classes .== CLASS_CODE`),
**not** from the count data itself.  `counts.json` stores the full-grid count
vectors (length = 85 212 = 324 × 263); within each record the values outside the
cover zone are zero.

When calling `summary_statistics(grid, mask)` without the original raster:

- If you have the cover-polygon mask (BitMatrix), pass it directly.
- If you only have `counts.json`, reconstruct a per-cover mask by taking the
  **union of all nonzero positions** across every record for that cover only
  for an approximation. Cells with zero returns in every mission are absent
  from that union. `summarize_counts_json` can use known cover areas for scalar
  CR summaries, but these areas cannot restore missing spatial support for
  bootstrap or line-scan inference. Supply exact masks for those analyses.

## Neighbor convention for Moran's I

Queen adjacency (8 directions: N, S, E, W, NE, NW, SE, SW).  Only
`valid`-to-`valid` cell pairs contribute to the spatial autocovariance numerator
and the total weight `W`.  The formula is:

    I = (n / W) · (Σᵢⱼ wᵢⱼ (xᵢ − x̄)(xⱼ − x̄)) / Σᵢ (xᵢ − x̄)²

where the sums run over `valid` cells, `n = count(valid)`, and `wᵢⱼ ∈ {0,1}`
indicates a queen-adjacent pair both in `valid`.

This exactly matches the `morans_I` implementation in `lidar.jl`.

## Grid layout

Decoded matrices use `(rows, columns)` with north in the first row. Legacy
`counts.json` array records are decoded as
`reshape(values, 263, 324)` (column-major, no row flip), verified against the
original GLI class raster. The current writer's object schema
declares `nrows` and `ncols` and stores row-major values, decoded without a row
flip. All consumers use the same decoder. Scalar statistics are invariant to
orientation, but spatial adjacency and block placement require this layout.
"""

# ---------------------------------------------------------------------------
# CountKey and CountRecord — mirror lidar.jl's types
# ---------------------------------------------------------------------------

"""
    CountKey

Composite key identifying one record in a count-grid collection.

Fields
------
- `ret::Symbol`     — `:all` or `:ground`
- `cover::Symbol`   — `:field`, `:decid`, or `:conif`
- `mission::Symbol` — `:density`, `:speed`, `:const2`, or `:const8`
- `kernel::Symbol`  — `:G`, `:E`, or `:NA` (kernel-agnostic missions use `:NA`)
"""
Base.@kwdef struct CountKey
    ret    ::Symbol   # :all | :ground
    cover  ::Symbol   # :field | :decid | :conif
    mission::Symbol   # :density | :speed | :const2 | :const8
    kernel ::Symbol   # :G | :E | :NA
end

# ---------------------------------------------------------------------------
# Label helpers (match lidar.jl verbatim)
# ---------------------------------------------------------------------------

"""Return the display label for a return-type symbol."""
return_label(r::Symbol) =
    r === :all    ? "All"    :
    r === :ground ? "Ground" : String(r)

"""Return the display label for a mission symbol."""
mission_label(m::Symbol) =
    m === :density ? "Density-aware" :
    m === :speed   ? "Speed-aware"   :
    m === :const2  ? "Const 2 m/s"  :
    m === :const8  ? "Const 8 m/s"  : String(m)

"""Return the display label for a kernel symbol."""
kernel_label(k::Symbol) = k === :NA ? "——" : String(k)

"""Return the display label for a cover symbol."""
cover_label(c::Symbol) =
    c === :field ? "Field"      :
    c === :decid ? "Deciduous"  :
    c === :conif ? "Coniferous" : String(c)

# Primary missions in the characterization figures. CSV labels remain those
# produced by mission_label/kernel_label; display labels are presentation only.
const CHARACTERIZATION_MISSIONS = (
    (mission=:const2,  kernel=:NA, display_label="Const. 2 m/s"),
    (mission=:density, kernel=:E,  display_label="KDE-guided (Epanechnikov)"),
    (mission=:const8,  kernel=:NA, display_label="Const. 8 m/s"),
)

"""
    characterization_stats(stats::AbstractDataFrame) -> DataFrame

Select the three primary characterization missions from a summary CSV and add
canonical `mission_key`, `kernel_key`, and `display_label` columns. Gaussian and
speed-aware variants are excluded. `cond_mean` copies `Mean`, which is already
computed over covered cells only; it must not be divided by coverage ratio.
The input table is not modified.
"""
function characterization_stats(stats::AbstractDataFrame)
    missions = Dict((mission_label(m.mission), kernel_label(m.kernel)) => m
                    for m in CHARACTERIZATION_MISSIONS)
    primary = filter(row -> haskey(missions, (row.Mission, row.Kernel)), stats)
    metadata = [missions[(row.Mission, row.Kernel)] for row in eachrow(primary)]
    primary[!, :mission_key] = [m.mission for m in metadata]
    primary[!, :kernel_key] = [m.kernel for m in metadata]
    primary[!, :display_label] = [m.display_label for m in metadata]
    primary[!, :cond_mean] = copy(primary.Mean)
    return primary
end

# ---------------------------------------------------------------------------
# Gini coefficient (matches lidar.jl formula exactly)
# ---------------------------------------------------------------------------

"""
    gini_coefficient(v::AbstractVector{<:Real}) -> Float64

Gini coefficient of a non-negative vector.

Uses the formula from `lidar.jl`:

    G = 2·Σᵢ i·cᵢ / (n·s) − (n+1)/n

where `c = sort(v)`, `s = sum(c)`, and i is 1-based.

This is equivalent to the standard area-under-Lorenz-curve definition.
The result is in [0, 1) for non-negative inputs; 0 = perfect equality.
"""
function gini_coefficient(v::AbstractVector{<:Real})::Float64
    isempty(v) && return 0.0
    c = sort(Float64.(v))
    n = length(c)
    s = sum(c)
    s ≈ 0 && return 0.0
    num = 0.0
    @inbounds for i in 1:n
        num += i * c[i]
    end
    return 2.0 * num / (n * s) - (n + 1) / n
end

# ---------------------------------------------------------------------------
# Moran's I (queen adjacency, valid-cell restricted)
# ---------------------------------------------------------------------------

"""
    morans_i(counts::AbstractMatrix{<:Real}, mask::AbstractMatrix{Bool};
             neighbor::Symbol = :queen) -> Union{Float64, Missing}

Moran's I spatial autocorrelation statistic for a 2-D count grid.

Arguments
---------
- `counts` — 2-D count matrix (any orientation, consistent with `mask`)
- `mask`   — `BitMatrix` of the same size; `true` = cell belongs to the cover zone
- `neighbor` — `:queen` (8-direction, default) or `:rook` (4-direction)

Returns `missing` if there are no valid neighbor pairs.

### Algorithm (matches `lidar.jl` `morans_I` exactly)

Only cells where `mask[r,c] == true` **and** `counts[r,c] > 0` are `valid`.
The mean `x̄` and the deviation sums are computed over `valid` cells only.
The adjacency weight `wᵢⱼ = 1` iff both cells are valid and queen/rook-adjacent.

    I = (n / W) · (Σᵢⱼ wᵢⱼ (xᵢ − x̄)(xⱼ − x̄)) / Σᵢ (xᵢ − x̄)²

where `n = count(valid)` and `W = Σᵢⱼ wᵢⱼ`.

### Neighbor convention
- **Queen (default)**: 8 directions — (±1, 0), (0, ±1), (±1, ±1)
- **Rook**: 4 directions — (±1, 0), (0, ±1)

The manuscript uses queen adjacency to match `lidar.jl`.
"""
function morans_i(counts::AbstractMatrix{<:Real},
                  mask  ::AbstractMatrix{Bool};
                  neighbor::Symbol=:queen)::Union{Float64,Missing}
    size(counts) == size(mask) ||
        throw(DimensionMismatch("counts and mask must have the same size"))
    neighbor in (:queen, :rook) ||
        throw(ArgumentError("neighbor must be :queen or :rook"))

    ny, nx = size(counts)

    # valid = mask AND count > 0
    valid = mask .& (counts .> 0)

    n = count(valid)
    n == 0 && return missing

    # mean over valid cells
    s = 0.0
    @inbounds for i in eachindex(counts)
        valid[i] && (s += counts[i])
    end
    x̄ = s / n

    # denominator: Σᵢ (xᵢ − x̄)²
    denom = 0.0
    @inbounds for r in 1:ny, c in 1:nx
        if valid[r, c]
            d = counts[r, c] - x̄
            denom += d * d
        end
    end
    denom ≈ 0 && return missing

    # neighbor offsets
    offs = if neighbor === :rook
        ((-1,0),(1,0),(0,-1),(0,1))
    else  # :queen
        ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))
    end

    # numerator and total weight
    num = 0.0
    W   = 0.0
    @inbounds for r in 1:ny, c in 1:nx
        valid[r, c] || continue
        xi = counts[r, c] - x̄
        for (dr, dc) in offs
            rr = r + dr
            cc = c + dc
            if 1 ≤ rr ≤ ny && 1 ≤ cc ≤ nx && valid[rr, cc]
                xj = counts[rr, cc] - x̄
                num += xi * xj
                W   += 1.0
            end
        end
    end

    W == 0.0 && return missing
    return (n / W) * (num / denom)
end

# ---------------------------------------------------------------------------
# CountSummary — result struct
# ---------------------------------------------------------------------------

"""
    CountSummary

Summary statistics for one count-grid record.

Fields match `stats_and_coverage.csv` columns exactly:

| Field    | Description                                              |
|----------|----------------------------------------------------------|
| `ncells` | Positive cells: `count(mask .& counts .> 0)`             |
| `Q1`     | 25th percentile of positive-cell counts                  |
| `Q2`     | Median of positive-cell counts                           |
| `Q3`     | 75th percentile of positive-cell counts                  |
| `IQR`    | `Q3 − Q1`                                               |
| `Mean`   | Arithmetic mean of positive-cell counts                  |
| `N`      | Total cells in cover polygon: `count(mask)` (denominator for CR) |
| `CR`     | Coverage ratio: `ncells / N`                             |
| `CV`     | Coefficient of variation: `std(v; corrected=true) / mean(v)` |
| `Gini`   | Gini coefficient (see `gini_coefficient`)                |
| `MoranI` | Moran's I (queen adjacency, valid cells only)            |
"""
struct CountSummary
    ncells ::Int
    Q1     ::Float64
    Q2     ::Float64
    Q3     ::Float64
    IQR    ::Float64
    Mean   ::Float64
    N      ::Int
    CR     ::Float64
    CV     ::Float64
    Gini   ::Float64
    MoranI ::Union{Float64,Missing}
end

# Pretty printing
function Base.show(io::IO, s::CountSummary)
    mi = ismissing(s.MoranI) ? "missing" : @sprintf("%.6f", s.MoranI)
    print(io,
        "CountSummary(ncells=$(s.ncells), N=$(s.N), CR=$(@sprintf("%.4f", s.CR)), ",
        "Q1=$(s.Q1), Q2=$(s.Q2), Q3=$(s.Q3), IQR=$(s.IQR), ",
        "Mean=$(@sprintf("%.4f", s.Mean)), CV=$(@sprintf("%.4f", s.CV)), ",
        "Gini=$(@sprintf("%.4f", s.Gini)), MoranI=$(mi))")
end

# ---------------------------------------------------------------------------
# summary_statistics — primary API
# ---------------------------------------------------------------------------

"""
    summary_statistics(counts::AbstractMatrix{<:Real},
                       mask  ::AbstractMatrix{Bool};
                       neighbor::Symbol = :queen) -> CountSummary

Compute all manuscript summary statistics for a count grid.

Arguments
---------
- `counts`   — 2-D matrix of per-cell LiDAR return counts (full raster grid)
- `mask`     — `BitMatrix` of the same size; cells belonging to the cover zone
               (`true`) vs. outside (`false`).  `N = count(mask)`.
- `neighbor` — adjacency for Moran's I: `:queen` (default, matches `lidar.jl`)
               or `:rook`

Returns
-------
A [`CountSummary`](@ref) with fields matching the `stats_and_coverage.csv` columns.

Notes
-----
- All scalar statistics (Q1/Q2/Q3/IQR/Mean/CV/Gini) are computed on the subset
  `v = counts[mask .& counts .> 0]` — i.e. positive counts within the cover zone.
- `N = count(mask)` is the cover-polygon area (support denominator for CR).
- `CR = ncells / N` where `ncells = length(v)`.
- `CV` uses sample standard deviation (`std(v; corrected=true)`, ddof=1).
- Moran's I uses queen adjacency over valid cells only; see [`morans_i`](@ref).

Example
-------
```julia
counts = [0 0 3; 0 5 2; 1 0 4]
mask   = BitMatrix([true true true; true true true; true true true])
s = summary_statistics(counts, mask)
s.CR    # coverage ratio
s.Gini  # Gini coefficient
```
"""
function summary_statistics(counts  ::AbstractMatrix{<:Real},
                             mask    ::AbstractMatrix{Bool};
                             neighbor::Symbol=:queen)::CountSummary
    size(counts) == size(mask) ||
        throw(DimensionMismatch("counts and mask must have the same size"))

    # valid values: positive counts inside the cover zone
    v = Float64[counts[i] for i in eachindex(counts)
                if mask[i] && counts[i] > 0]

    N      = count(mask)
    ncells = length(v)

    if ncells == 0
        return CountSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, N, 0.0, 0.0, 0.0, missing)
    end

    Q1, Q2, Q3 = quantile(v, (0.25, 0.50, 0.75))
    IQR_val    = Q3 - Q1
    μ          = mean(v)
    CR_val     = ncells / N
    CV_val     = std(v; corrected=true) / μ
    G          = gini_coefficient(v)
    MI         = morans_i(counts, mask; neighbor=neighbor)

    return CountSummary(ncells, Q1, Q2, Q3, IQR_val, μ, N, CR_val, CV_val, G, MI)
end

"""
    summary_statistics(counts::AbstractMatrix{<:Real};
                       support::Union{Nothing,Int}=nothing,
                       neighbor::Symbol=:queen) -> CountSummary

Convenience method without an explicit mask.

When `support` is `nothing` (default), `N` is estimated as the number of
nonzero cells in `counts`.  When `support` is given, it overrides `N`
(useful when the cover-polygon area is known from external metadata).

!!! note
    This form cannot compute Moran's I correctly for a sub-region of a larger
    raster, because no mask is available to restrict adjacency to the cover zone.
    Moran's I is computed on the full `counts` matrix with a mask of all `true`.
    Use `summary_statistics(counts, mask)` for manuscript-accurate results.
"""
function summary_statistics(counts  ::AbstractMatrix{<:Real};
                             support ::Union{Nothing,Int}=nothing,
                             neighbor::Symbol=:queen)::CountSummary
    mask = counts .> 0   # treat all nonzero cells as the zone
    N_est = isnothing(support) ? count(mask) : support
    # Create a full-true mask for Moran's I (see docstring caveat)
    full_mask = trues(size(counts))
    s = summary_statistics(counts, full_mask; neighbor=neighbor)
    # Override N and CR with support-based values
    ncells = s.ncells
    CR_val = N_est > 0 ? ncells / N_est : 0.0
    return CountSummary(ncells, s.Q1, s.Q2, s.Q3, s.IQR, s.Mean,
                        N_est, CR_val, s.CV, s.Gini, s.MoranI)
end

# ---------------------------------------------------------------------------
# CountRecord — parsed counts.json entry
# ---------------------------------------------------------------------------

"""
    CountRecord

One record from `counts.json`, after parsing and reshaping.

Fields
------
- `key`    — [`CountKey`](@ref) with `(ret, cover, mission, kernel)`
- `matrix` — `Matrix{Int32}` of shape `(nrows, ncols)` (full raster grid)
"""
struct CountRecord
    key   ::CountKey
    matrix::Matrix{Int32}
end

# ---------------------------------------------------------------------------
# load_counts_json — parse counts.json
# ---------------------------------------------------------------------------

const _LEGACY_COUNT_NROWS = 263
const _LEGACY_COUNT_NCOLS = 324

function _count_grid_dimension(value, name)
    value isa Integer && !(value isa Bool) && value > 0 ||
        throw(ArgumentError("$name must be a positive integer"))
    return Int(value)
end

function _decode_count_values(values::AbstractVector, nrows::Integer, ncols::Integer;
                              layout::Symbol, T::Type=Int32)
    nr = _count_grid_dimension(nrows, "nrows")
    nc = _count_grid_dimension(ncols, "ncols")
    length(values) == nr * nc || throw(DimensionMismatch(
        "Count grid has $(length(values)) elements; expected $(nr * nc) ($nr×$nc)"))
    if layout === :legacy
        return reshape(T.(values), nr, nc)
    elseif layout === :rowmajor
        return permutedims(reshape(T.(values), nc, nr))
    end
    throw(ArgumentError("Unknown count-grid layout: $layout"))
end

"""
    load_counts_json(path::AbstractString; nrows=nothing, ncols=nothing)
        -> Vector{CountRecord}

Load either supported `counts.json` schema into `(rows, columns)` matrices:

- Legacy array: each record has a `key` object and flat `matrix` vector. The
  historical encoding is column-major and is decoded with
  `reshape(values, nrows, ncols)`, defaulting to the Durham grid of 263 rows ×
  324 columns. Its rows already match the original north-up GLI raster.
- Current writer object: `nrows`, `ncols`, and `records`, whose records contain
  `ret`, `cover`, `mission`, `kernel`, and row-major `values`. Dimensions come
  from metadata and decoding preserves the written row order.

Explicit dimensions must be positive and must agree with object metadata and
record lengths. Returned counts are `Int32`; row 1 is the northernmost row for
the bundled legacy data and the current north-up writer output.
"""
function load_counts_json(path::AbstractString;
                          nrows::Union{Nothing,Integer}=nothing,
                          ncols::Union{Nothing,Integer}=nothing)::Vector{CountRecord}
    raw = JSON.parsefile(path)
    if raw isa AbstractVector
        objects = raw
        nr = _count_grid_dimension(something(nrows, _LEGACY_COUNT_NROWS), "nrows")
        nc = _count_grid_dimension(something(ncols, _LEGACY_COUNT_NCOLS), "ncols")
        layout = :legacy
    elseif raw isa AbstractDict
        all(key -> haskey(raw, key), ("nrows", "ncols", "records")) ||
            throw(ArgumentError("Count-grid object must contain nrows, ncols, and records"))
        nr = _count_grid_dimension(raw["nrows"], "metadata nrows")
        nc = _count_grid_dimension(raw["ncols"], "metadata ncols")
        for (name, supplied, stored) in (("nrows", nrows, nr), ("ncols", ncols, nc))
            if supplied !== nothing
                requested = _count_grid_dimension(supplied, name)
                requested == stored || throw(DimensionMismatch(
                    "Requested $name=$requested disagrees with count-grid metadata $name=$stored"))
            end
        end
        objects = raw["records"]
        objects isa AbstractVector || throw(ArgumentError("records must be an array"))
        layout = :rowmajor
    else
        throw(ArgumentError("counts.json must contain a legacy array or a count-grid object"))
    end

    records = Vector{CountRecord}(undef, length(objects))
    for (i, obj) in enumerate(objects)
        kd = layout === :legacy ? obj["key"] : obj
        key = CountKey(
            ret     = Symbol(kd["ret"]),
            cover   = Symbol(kd["cover"]),
            mission = Symbol(kd["mission"]),
            kernel  = Symbol(kd["kernel"]),
        )
        vec_raw = obj[layout === :legacy ? "matrix" : "values"]
        vec_raw isa AbstractVector || throw(ArgumentError("Record $i values must be an array"))
        mat = _decode_count_values(vec_raw, nr, nc; layout=layout)
        records[i] = CountRecord(key, mat)
    end
    return records
end

# ---------------------------------------------------------------------------
# build_cover_masks — reconstruct per-cover masks from count records
# ---------------------------------------------------------------------------

"""
    build_cover_masks(records::Vector{CountRecord};
                      covers::Vector{Symbol} = [:field, :decid, :conif])
        -> Dict{Symbol, BitMatrix}

Reconstruct per-cover zone masks from count records by taking the union of
all nonzero positions across every record for each cover.

This is an approximation of the true polygon mask (which comes from the
classified raster in `lidar.jl`). It excludes cells that received zero returns
in every mission and is not a substitute for the spatial support used in
manuscript inference.

For the Durham NH study site the known discrepancies are:
- `:conif` — 1 cell (union gives 5412; true N = 5413)
- `:decid` — 61 cells (union gives 24019; true N = 24080)
- `:field` — 638 cells (union gives 55081; true N = 55719)

The true `N` values are stored in the companion constant
[`DURHAM_COVER_N`](@ref) and are used by [`summarize_counts_json`](@ref).
"""
function build_cover_masks(records::Vector{CountRecord};
                            covers::Vector{Symbol}=[:field,:decid,:conif]
                           )::Dict{Symbol,BitMatrix}
    # infer grid size from first record
    isempty(records) && return Dict{Symbol,BitMatrix}()
    sz = size(first(records).matrix)

    masks = Dict{Symbol,BitMatrix}()
    for cov in covers
        union_mask = falses(sz)
        for rec in records
            if rec.key.cover === cov
                union_mask .|= (rec.matrix .> 0)
            end
        end
        masks[cov] = BitMatrix(union_mask)
    end
    return masks
end

# ---------------------------------------------------------------------------
# Durham NH cover zone sizes (fixed, from the classified raster)
# ---------------------------------------------------------------------------

"""
    DURHAM_COVER_N :: Dict{Symbol, Int}

True cover-polygon cell counts for the Durham NH study site (263 × 324 grid).
These are derived from the classified raster (`classes .== CLASS_CODE`) and
match the `N` column in `stats_and_coverage.csv`:

| Cover     | N      |
|-----------|--------|
| `:field`  | 55 719 |
| `:decid`  | 24 080 |
| `:conif`  |  5 413 |

Sum: 55 719 + 24 080 + 5 413 = **85 212** = 324 × 263.

Use these when the classified raster is unavailable (e.g., working from
`counts.json` alone) to ensure exact reproduction of manuscript CR values.
"""
const DURHAM_COVER_N = Dict{Symbol,Int}(
    :field  => 55_719,
    :decid  => 24_080,
    :conif  =>  5_413,
)

# ---------------------------------------------------------------------------
# summarize_counts_json — convenience one-call API
# ---------------------------------------------------------------------------

"""
    SummaryRow

One row in the output of [`summarize_counts_json`](@ref), corresponding to a
single `(cover, return, mission, kernel)` combination.

Fields mirror `stats_and_coverage.csv` columns:
`Cover`, `Return`, `Mission`, `Kernel`, plus all fields of [`CountSummary`](@ref).
"""
struct SummaryRow
    Cover   ::String
    Return  ::String
    Mission ::String
    Kernel  ::String
    stats   ::CountSummary
end

"""
    summarize_counts_json(path::AbstractString;
                          cover_n    ::Dict{Symbol,Int} = DURHAM_COVER_N,
                          nrows = nothing,
                          ncols = nothing,
                          neighbor   ::Symbol = :queen,
                          covers     ::Vector{Symbol} = [:field, :decid, :conif],
                          returns    ::Vector{Symbol} = [:all, :ground],
                          missions   ::Vector{Symbol} = [:density, :speed, :const2, :const8],
                          kernels    ::Vector{Symbol} = [:G, :E],
    ) -> DataFrame

One-call convenience: load `counts.json`, build cover masks, compute all
manuscript statistics, and return a tidy `DataFrame` matching
`stats_and_coverage.csv`.

Arguments
---------
- `path`      — path to `counts.json`
- `cover_n`   — dict mapping cover symbol → polygon cell count (N denominator).
                Defaults to [`DURHAM_COVER_N`](@ref) (Durham NH site).
                Pass `nothing` to estimate N from the data union (slightly
                underestimates N for cells that are always zero).
- `nrows/ncols` — optional dimensions; metadata or legacy default 263 × 324
- `neighbor`  — adjacency for Moran's I (`:queen` matches `lidar.jl`)
- `covers / returns / missions / kernels` — subset to compute

Returns
-------
`DataFrame` with columns: `Cover`, `Return`, `Mission`, `Kernel`, `ncells`,
`Q1`, `Q2`, `Q3`, `IQR`, `Mean`, `N`, `CR`, `CV`, `Gini`, `MoranI`.

Example
-------
```julia
df = summarize_counts_json("path/to/counts.json")
CSV.write("regenerated_stats.csv", df)
```
"""
function summarize_counts_json(path     ::AbstractString;
                                cover_n  ::Union{Dict{Symbol,Int},Nothing}=DURHAM_COVER_N,
                                nrows    ::Union{Nothing,Integer}=nothing,
                                ncols    ::Union{Nothing,Integer}=nothing,
                                neighbor ::Symbol=:queen,
                                covers   ::Vector{Symbol}=[:field,:decid,:conif],
                                returns  ::Vector{Symbol}=[:all,:ground],
                                missions ::Vector{Symbol}=[:density,:speed,:const2,:const8],
                                kernels  ::Vector{Symbol}=[:G,:E],
                               )::DataFrame

    records = load_counts_json(path; nrows=nrows, ncols=ncols)

    # Build cover masks
    approx_masks = build_cover_masks(records; covers=covers)

    # Build per-cover masks with correct N
    masks = Dict{Symbol,BitMatrix}()
    for cov in covers
        if !isnothing(cover_n) && haskey(cover_n, cov)
            # Use the data-derived mask (best spatial layout we can get from counts.json);
            # N override is applied at stats time via summary_statistics(counts, mask)
            # where count(mask) == cover_n[cov].
            # Since union mask may be 1-638 cells short, we cannot reconstruct the exact
            # polygon from counts.json alone.  We use the union mask for spatial structure
            # (Moran's I adjacency) but override N when computing CR.
            masks[cov] = approx_masks[cov]
        else
            masks[cov] = approx_masks[cov]
        end
    end

    # Index records by key for fast lookup
    rec_index = Dict{CountKey,Matrix{Int32}}()
    for rec in records
        rec_index[rec.key] = rec.matrix
    end

    is_kernel_agnostic(m::Symbol) = (m === :const2 || m === :const8)

    rows = NamedTuple[]
    for cov in covers, ret in returns, m in missions
        ks = is_kernel_agnostic(m) ? [:NA] : kernels
        for k in ks
            key = CountKey(ret=ret, cover=cov, mission=m, kernel=k)
            haskey(rec_index, key) || continue

            mat  = rec_index[key]
            mask = masks[cov]

            # Compute stats with the union mask for spatial structure
            s    = summary_statistics(mat, mask; neighbor=neighbor)

            # Override N and CR if cover_n provided (exact polygon size)
            N_eff  = (!isnothing(cover_n) && haskey(cover_n, cov)) ? cover_n[cov] : s.N
            CR_eff = N_eff > 0 ? s.ncells / N_eff : 0.0

            s_final = CountSummary(s.ncells, s.Q1, s.Q2, s.Q3, s.IQR,
                                   s.Mean, N_eff, CR_eff, s.CV, s.Gini, s.MoranI)

            push!(rows, (
                Cover   = cover_label(cov),
                Return  = return_label(ret),
                Mission = mission_label(m),
                Kernel  = kernel_label(k),
                ncells  = s_final.ncells,
                Q1      = s_final.Q1,
                Q2      = s_final.Q2,
                Q3      = s_final.Q3,
                IQR     = s_final.IQR,
                Mean    = s_final.Mean,
                N       = s_final.N,
                CR      = s_final.CR,
                CV      = s_final.CV,
                Gini    = s_final.Gini,
                MoranI  = s_final.MoranI,
            ))
        end
    end

    df = DataFrame(rows)
    sort!(df, [:Return, :Cover, :Mission, :Kernel])
    return df
end

# ---------------------------------------------------------------------------
# write_summary_csv — CSV output
# ---------------------------------------------------------------------------

"""
    write_summary_csv(path::AbstractString, df::DataFrame; kwargs...) -> path

Write a summary statistics DataFrame (from [`summarize_counts_json`](@ref)
or any `DataFrame` with matching columns) to CSV.

The output format matches `stats_and_coverage.csv` exactly.

Keyword arguments are forwarded to `CSV.write`.
"""
function write_summary_csv(path::AbstractString, df::DataFrame; kwargs...)
    CSV.write(path, df; kwargs...)
    return path
end

"""
    write_summary_csv(path::AbstractString, records::Vector{CountRecord},
                      masks::Dict{Symbol,BitMatrix};
                      cover_n::Union{Dict{Symbol,Int},Nothing} = DURHAM_COVER_N,
                      neighbor::Symbol = :queen, kwargs...) -> path

Compute summary statistics from pre-loaded records and masks, then write to CSV.
"""
function write_summary_csv(path   ::AbstractString,
                            records::Vector{CountRecord},
                            masks  ::Dict{Symbol,BitMatrix};
                            cover_n::Union{Dict{Symbol,Int},Nothing}=DURHAM_COVER_N,
                            neighbor::Symbol=:queen,
                            kwargs...)

    rec_index = Dict{CountKey,Matrix{Int32}}(r.key => r.matrix for r in records)
    is_ka(m) = (m === :const2 || m === :const8)

    rows = NamedTuple[]
    for (cov, mask) in masks
        for ret in (:all, :ground), m in (:density, :speed, :const2, :const8)
            ks = is_ka(m) ? [:NA] : [:G, :E]
            for k in ks
                key = CountKey(ret=ret, cover=cov, mission=m, kernel=k)
                haskey(rec_index, key) || continue
                mat = rec_index[key]
                s   = summary_statistics(mat, mask; neighbor=neighbor)
                N_eff  = (!isnothing(cover_n) && haskey(cover_n, cov)) ?
                          cover_n[cov] : s.N
                CR_eff = N_eff > 0 ? s.ncells / N_eff : 0.0
                push!(rows, (
                    Cover   = cover_label(cov),
                    Return  = return_label(ret),
                    Mission = mission_label(m),
                    Kernel  = kernel_label(k),
                    ncells  = s.ncells,
                    Q1      = s.Q1,
                    Q2      = s.Q2,
                    Q3      = s.Q3,
                    IQR     = s.IQR,
                    Mean    = s.Mean,
                    N       = N_eff,
                    CR      = CR_eff,
                    CV      = s.CV,
                    Gini    = s.Gini,
                    MoranI  = s.MoranI,
                ))
            end
        end
    end

    df = DataFrame(rows)
    sort!(df, [:Return, :Cover, :Mission, :Kernel])
    CSV.write(path, df; kwargs...)
    return path
end

