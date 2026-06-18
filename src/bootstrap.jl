"""
    bootstrap.jl — Spatial block bootstrap for UAV LiDAR coverage-ratio statistics

This module implements spatial block-bootstrap confidence intervals for the
coverage ratio (CR) statistic and its pairwise differences, as used in:

> "KDE-Guided Adaptive Speed Control for UAV LiDAR: Improving Ground-Return
>  Spatial Coverage in Forested Terrain"
> Darien D. Perez Martin, Adam G. Hunsaker, Jennifer M. Jacobs
> *Remote Sensing* (MDPI) — in preparation

## Manuscript defaults

| Parameter    | Value | Description                                          |
|--------------|-------|------------------------------------------------------|
| `nboot`      | 5000  | Number of bootstrap replicates                       |
| `block_frac` | 0.03  | Block **side-length** as a fraction of √N_cells      |
| `seed`       | 42    | RNG seed for reproducibility                         |
| `level`      | 0.95  | Confidence level (two-sided percentile interval)     |

## Block construction (methodological decision — flagged for author review)

**Definition:**  `block_frac` is interpreted as the **block side-length fraction**
of the geometric mean of the grid dimensions:

    block_side = round(Int, block_frac × √(nrows × ncols))

For the Durham NH 324 × 263 grid: `√(324 × 263) ≈ 291.8`, so
`block_side = round(Int, 0.03 × 291.8) = 9` cells.

**Rationale:**  Square blocks of side `s` cover `s²` cells, giving a block area
of ≈ 0.09% of the raster — approximately 3% linear scale of the domain, which
matches the spirit of "3% block_frac" as a spatial-scale parameter.  This is the
most natural interpretation for a 2-D raster; if the intent was area fraction the
block side would be `round(Int, √(block_frac × N))`.

> ⚠️ **METHODOLOGICAL DECISION — NEEDS AUTHOR APPROVAL BEFORE MANUSCRIPT INTEGRATION**
>
> `block_frac = 0.03` is interpreted as the block **side-length fraction** of
> `√(nrows × ncols)`.  For the Durham 324 × 263 grid this gives `block_side = 9`.
> If Darien prefers the area-fraction interpretation, change `block_side` to
> `round(Int, √(block_frac * nrows * ncols))`, which gives `block_side = 51`.
> The current choice (side fraction) produces smaller, more numerous blocks
> and is common in the spatial statistics literature for moderate autocorrelation
> ranges.

## Resampling procedure

1. Enumerate all non-overlapping `block_side × block_side` starting positions
   in the grid (upper-left corners on a regular grid with stride = `block_side`).
2. Draw `ceil(N_needed / block_cells)` blocks with replacement, where `N_needed`
   is determined so that the resampled data covers at least `count(mask)` cells.
3. For each replicate, compute `CR* = ncells* / N*` where:
   - `ncells*` = number of mask cells in drawn blocks that have counts > 0
   - `N*`      = number of mask cells in drawn blocks
4. The 95% CI is the `[2.5%, 97.5%]` percentile of the `nboot` replicate CRs.

For paired differences (`bootstrap_cr_difference`), the **same** sequence of
block indices is used for both missions, so the difference CR_A − CR_B inherits
the spatial dependence structure.

## API

    bootstrap_cr(counts, mask; nboot, block_frac, seed, level)
    bootstrap_cr_difference(counts_a, counts_b, mask; nboot, block_frac, seed, level)
    bootstrap_cr_table(path_json; ...)
    bootstrap_cr_difference_table(path_json, comparisons; ...)
"""

# ---------------------------------------------------------------------------
# BootstrapCI — result struct
# ---------------------------------------------------------------------------

"""
    BootstrapCI

Block-bootstrap confidence interval for a coverage ratio (or CR difference).

Fields
------
- `estimate`  — Point estimate (CR or CR_A − CR_B)
- `lower`     — Lower confidence bound
- `upper`     — Upper confidence bound
- `level`     — Confidence level (e.g. 0.95)
- `nboot`     — Number of bootstrap replicates used
- `block_side`— Side length of spatial blocks (cells)
- `replicates`— `Vector{Float64}` of bootstrap replicate statistics (length `nboot`)
"""
struct BootstrapCI
    estimate  ::Float64
    lower     ::Float64
    upper     ::Float64
    level     ::Float64
    nboot     ::Int
    block_side::Int
    replicates::Vector{Float64}
end

function Base.show(io::IO, ci::BootstrapCI)
    @printf(io,
        "BootstrapCI(estimate=%.6f, %d%% CI=[%.6f, %.6f], nboot=%d, block_side=%d)",
        ci.estimate,
        round(Int, ci.level * 100),
        ci.lower, ci.upper,
        ci.nboot, ci.block_side,
    )
end

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""
    _compute_block_side(nrows, ncols, block_frac) -> Int

Compute the block side length as `round(block_frac × √(nrows × ncols))`,
clamped to [1, min(nrows, ncols)].

This interprets `block_frac` as the block side-length fraction of the
geometric mean of grid dimensions — see module docstring for rationale.
"""
function _compute_block_side(nrows::Int, ncols::Int, block_frac::Float64)::Int
    s = round(Int, block_frac * sqrt(nrows * ncols))
    return clamp(s, 1, min(nrows, ncols))
end

"""
    _block_origins(nrows, ncols, block_side) -> Vector{Tuple{Int,Int}}

Enumerate all valid non-overlapping block upper-left corner positions
(r, c) on the grid.  Blocks may extend beyond the grid boundary — such
cells are simply ignored during resampling.
"""
function _block_origins(nrows::Int, ncols::Int, block_side::Int)::Vector{Tuple{Int,Int}}
    origins = Tuple{Int,Int}[]
    for r in 1:block_side:nrows
        for c in 1:block_side:ncols
            push!(origins, (r, c))
        end
    end
    return origins
end

"""
    _cr_from_blocks(counts, mask, origins, block_side, idx_sample)
        -> Float64

Compute the bootstrap replicate CR from a given sample of block indices.

- Iterates over `idx_sample` (indices into `origins`)
- Collects all (r, c) cells in each block that are within grid bounds
- CR* = (mask ∩ counts>0 cells) / (mask cells) across sampled blocks
- Returns NaN if no mask cells were sampled.
"""
function _cr_from_blocks(
    counts    ::AbstractMatrix{<:Real},
    mask      ::AbstractMatrix{Bool},
    origins   ::Vector{Tuple{Int,Int}},
    block_side::Int,
    idx_sample::AbstractVector{Int},
)::Float64
    nrows, ncols = size(counts)
    n_hit  = 0   # mask cells with counts > 0
    n_mask = 0   # mask cells total

    @inbounds for bi in idx_sample
        r0, c0 = origins[bi]
        for dr in 0:(block_side - 1)
            r = r0 + dr
            r > nrows && continue
            for dc in 0:(block_side - 1)
                c = c0 + dc
                c > ncols && continue
                if mask[r, c]
                    n_mask += 1
                    counts[r, c] > 0 && (n_hit += 1)
                end
            end
        end
    end

    n_mask == 0 && return NaN
    return n_hit / n_mask
end

"""
    _cr_diff_from_blocks(counts_a, counts_b, mask, origins, block_side, idx_sample)
        -> Float64

Compute the paired bootstrap replicate CR difference CR_A* − CR_B* using
the **same** block sample for both missions, preserving spatial dependence.

Returns NaN if no mask cells were sampled.
"""
function _cr_diff_from_blocks(
    counts_a  ::AbstractMatrix{<:Real},
    counts_b  ::AbstractMatrix{<:Real},
    mask      ::AbstractMatrix{Bool},
    origins   ::Vector{Tuple{Int,Int}},
    block_side::Int,
    idx_sample::AbstractVector{Int},
)::Float64
    nrows, ncols = size(counts_a)
    n_hit_a = 0
    n_hit_b = 0
    n_mask  = 0

    @inbounds for bi in idx_sample
        r0, c0 = origins[bi]
        for dr in 0:(block_side - 1)
            r = r0 + dr
            r > nrows && continue
            for dc in 0:(block_side - 1)
                c = c0 + dc
                c > ncols && continue
                if mask[r, c]
                    n_mask += 1
                    counts_a[r, c] > 0 && (n_hit_a += 1)
                    counts_b[r, c] > 0 && (n_hit_b += 1)
                end
            end
        end
    end

    n_mask == 0 && return NaN
    return (n_hit_a - n_hit_b) / n_mask
end

# ---------------------------------------------------------------------------
# Primary API: bootstrap_cr
# ---------------------------------------------------------------------------

"""
    bootstrap_cr(counts::AbstractMatrix{<:Real},
                 mask  ::AbstractMatrix{Bool};
                 nboot      ::Int    = 5000,
                 block_frac ::Float64 = 0.03,
                 seed       ::Int    = 42,
                 level      ::Float64 = 0.95) -> BootstrapCI

Spatial block-bootstrap confidence interval for the coverage ratio (CR).

Arguments
---------
- `counts`     — 2-D matrix of per-cell LiDAR return counts (full raster grid)
- `mask`       — `BitMatrix` of the same size; `true` = cell belongs to the cover zone
- `nboot`      — Number of bootstrap replicates (manuscript default: 5000)
- `block_frac` — Block side-length fraction of `√(nrows × ncols)` (default: 0.03)
- `seed`       — RNG seed for reproducibility (manuscript default: 42)
- `level`      — Confidence level for the percentile interval (default: 0.95)

Returns
-------
A [`BootstrapCI`](@ref) with the point estimate, lower/upper bounds, and
replicate distribution.

## Block definition

Block side length is computed as:

    block_side = clamp(round(Int, block_frac × √(nrows × ncols)), 1, min(nrows,ncols))

For the Durham 324 × 263 grid: `block_side = 9`.

## Resampling

Each replicate draws `ceil(count(mask) / (block_side²))` block origins with
replacement, computes `CR* = n_hit* / n_mask*` over the sampled cells (where
n_mask* = number of mask cells in drawn blocks, n_hit* = mask cells with count > 0),
and the CI is the `[(1-level)/2, (1+level)/2]` percentile range.

## Example

```julia
counts = rand(0:5, 324, 263)
mask   = trues(324, 263)
ci = bootstrap_cr(counts, mask)
println(ci)
```
"""
function bootstrap_cr(
    counts    ::AbstractMatrix{<:Real},
    mask      ::AbstractMatrix{Bool};
    nboot     ::Int    = 5000,
    block_frac::Float64 = 0.03,
    seed      ::Int    = 42,
    level     ::Float64 = 0.95,
)::BootstrapCI
    size(counts) == size(mask) ||
        throw(DimensionMismatch("counts and mask must have the same size"))
    nboot  >= 1  || throw(ArgumentError("nboot must be ≥ 1"))
    0.0 < level < 1.0 || throw(ArgumentError("level must be in (0, 1)"))

    nrows, ncols = size(counts)
    N_mask = count(mask)

    # Point estimate (same formula as summary_statistics)
    n_hit_point = count(i -> mask[i] && counts[i] > 0, eachindex(counts, mask))
    estimate    = N_mask == 0 ? NaN : n_hit_point / N_mask

    # Edge case: empty mask
    if N_mask == 0
        return BootstrapCI(estimate, NaN, NaN, level, nboot, 0, fill(NaN, nboot))
    end

    # Block setup
    block_side = _compute_block_side(nrows, ncols, block_frac)
    origins    = _block_origins(nrows, ncols, block_side)
    n_origins  = length(origins)
    n_draw     = max(1, ceil(Int, N_mask / block_side^2))

    rng  = MersenneTwister(seed)
    reps = Vector{Float64}(undef, nboot)

    for b in 1:nboot
        idx = rand(rng, 1:n_origins, n_draw)
        reps[b] = _cr_from_blocks(counts, mask, origins, block_side, idx)
    end

    # Remove NaN replicates (shouldn't happen unless mask is pathological)
    valid_reps = filter(!isnan, reps)
    if isempty(valid_reps)
        return BootstrapCI(estimate, NaN, NaN, level, nboot, block_side, reps)
    end

    α     = (1.0 - level) / 2.0
    lower = quantile(valid_reps, α)
    upper = quantile(valid_reps, 1.0 - α)

    return BootstrapCI(estimate, lower, upper, level, nboot, block_side, reps)
end

# ---------------------------------------------------------------------------
# Primary API: bootstrap_cr_difference
# ---------------------------------------------------------------------------

"""
    bootstrap_cr_difference(counts_a ::AbstractMatrix{<:Real},
                            counts_b ::AbstractMatrix{<:Real},
                            mask     ::AbstractMatrix{Bool};
                            nboot      ::Int    = 5000,
                            block_frac ::Float64 = 0.03,
                            seed       ::Int    = 42,
                            level      ::Float64 = 0.95) -> BootstrapCI

Paired spatial block-bootstrap confidence interval for the CR difference
CR_A − CR_B.

Both missions are resampled using **identical block indices** in each
replicate, so the difference inherits the spatial dependence structure of
the study area.

Arguments
---------
- `counts_a` — Count matrix for mission A (same grid as `mask`)
- `counts_b` — Count matrix for mission B (same grid as `mask`)
- `mask`     — `BitMatrix`; `true` = cell belongs to the cover zone

All other keyword arguments are the same as [`bootstrap_cr`](@ref).

Returns
-------
[`BootstrapCI`](@ref) for the paired difference `CR_A − CR_B`.

A positive `estimate` means mission A has higher coverage.

## Example

```julia
# KDE-guided Epanechnikov vs. Constant 2 m/s, coniferous ground returns
ci_diff = bootstrap_cr_difference(counts_kde_e, counts_const2, mask_conif)
println(ci_diff)
```
"""
function bootstrap_cr_difference(
    counts_a  ::AbstractMatrix{<:Real},
    counts_b  ::AbstractMatrix{<:Real},
    mask      ::AbstractMatrix{Bool};
    nboot     ::Int    = 5000,
    block_frac::Float64 = 0.03,
    seed      ::Int    = 42,
    level     ::Float64 = 0.95,
)::BootstrapCI
    size(counts_a) == size(mask) ||
        throw(DimensionMismatch("counts_a and mask must have the same size"))
    size(counts_b) == size(mask) ||
        throw(DimensionMismatch("counts_b and mask must have the same size"))
    nboot >= 1 || throw(ArgumentError("nboot must be ≥ 1"))
    0.0 < level < 1.0 || throw(ArgumentError("level must be in (0, 1)"))

    nrows, ncols = size(counts_a)
    N_mask = count(mask)

    # Point estimate
    n_hit_a  = count(i -> mask[i] && counts_a[i] > 0, eachindex(counts_a, mask))
    n_hit_b  = count(i -> mask[i] && counts_b[i] > 0, eachindex(counts_b, mask))
    estimate = N_mask == 0 ? NaN : (n_hit_a - n_hit_b) / N_mask

    if N_mask == 0
        return BootstrapCI(estimate, NaN, NaN, level, nboot, 0, fill(NaN, nboot))
    end

    block_side = _compute_block_side(nrows, ncols, block_frac)
    origins    = _block_origins(nrows, ncols, block_side)
    n_origins  = length(origins)
    n_draw     = max(1, ceil(Int, N_mask / block_side^2))

    rng  = MersenneTwister(seed)
    reps = Vector{Float64}(undef, nboot)

    for b in 1:nboot
        idx = rand(rng, 1:n_origins, n_draw)
        reps[b] = _cr_diff_from_blocks(counts_a, counts_b, mask, origins,
                                        block_side, idx)
    end

    valid_reps = filter(!isnan, reps)
    if isempty(valid_reps)
        return BootstrapCI(estimate, NaN, NaN, level, nboot, block_side, reps)
    end

    α     = (1.0 - level) / 2.0
    lower = quantile(valid_reps, α)
    upper = quantile(valid_reps, 1.0 - α)

    return BootstrapCI(estimate, lower, upper, level, nboot, block_side, reps)
end

# ---------------------------------------------------------------------------
# Manuscript label helpers (KDE-guided naming, no "aware" in output)
# ---------------------------------------------------------------------------

"""
    manuscript_mission_label(mission::Symbol, kernel::Symbol) -> String

Return the manuscript-facing label for a (mission, kernel) combination.

This replaces legacy internal labels ("density-aware", "speed-aware") with
the public names used in the paper:

| Internal             | Manuscript label              |
|----------------------|-------------------------------|
| `(:density, :E)`     | `"KDE-guided Epanechnikov"`   |
| `(:density, :G)`     | `"KDE-guided Gaussian"`       |
| `(:speed, :E)`       | `"KDE-guided Epanechnikov"`   |
| `(:speed, :G)`       | `"KDE-guided Gaussian"`       |
| `(:const2, :NA)`     | `"Constant 2 m/s"`            |
| `(:const8, :NA)`     | `"Constant 8 m/s"`            |

Note: the manuscript treats density-aware and speed-aware as two variants
of the same KDE-guided strategy; this label function combines them for display.
Use the `Mission` column (with `Density-aware` / `Speed-aware`) to distinguish
them in full tables.
"""
function manuscript_mission_label(mission::Symbol, kernel::Symbol)::String
    kernel === :E && return "KDE-guided Epanechnikov"
    kernel === :G && return "KDE-guided Gaussian"
    mission === :const2 && return "Constant 2 m/s"
    mission === :const8 && return "Constant 8 m/s"
    return "$(mission_label(mission)) / $(kernel_label(kernel))"
end

# ---------------------------------------------------------------------------
# End-to-end table API: bootstrap_cr_table
# ---------------------------------------------------------------------------

"""
    bootstrap_cr_table(path_json ::AbstractString;
                       cover_n   ::Dict{Symbol,Int} = DURHAM_COVER_N,
                       nrows     ::Int = 324,
                       ncols     ::Int = 263,
                       nboot     ::Int    = 5000,
                       block_frac::Float64 = 0.03,
                       seed      ::Int    = 42,
                       level     ::Float64 = 0.95,
                       covers    ::Vector{Symbol} = [:field, :decid, :conif],
                       returns   ::Vector{Symbol} = [:all, :ground],
                       missions  ::Vector{Symbol} = [:density, :speed, :const2, :const8],
                       kernels   ::Vector{Symbol} = [:G, :E],
    ) -> DataFrame

Compute block-bootstrap CIs for CR for all rows in `counts.json`.

Returns a `DataFrame` with columns:
`Cover`, `Return`, `Mission`, `Kernel`, `CR`, `CI_lower`, `CI_upper`,
`level`, `nboot`, `block_side`.

## Example

```julia
df = bootstrap_cr_table("path/to/counts.json")
CSV.write("bootstrap_cr_cis.csv", df)
```
"""
function bootstrap_cr_table(
    path_json  ::AbstractString;
    cover_n    ::Dict{Symbol,Int} = DURHAM_COVER_N,
    nrows      ::Int = 324,
    ncols      ::Int = 263,
    nboot      ::Int    = 5000,
    block_frac ::Float64 = 0.03,
    seed       ::Int    = 42,
    level      ::Float64 = 0.95,
    covers     ::Vector{Symbol} = [:field, :decid, :conif],
    returns    ::Vector{Symbol} = [:all, :ground],
    missions   ::Vector{Symbol} = [:density, :speed, :const2, :const8],
    kernels    ::Vector{Symbol} = [:G, :E],
)::DataFrame

    records   = load_counts_json(path_json; nrows=nrows, ncols=ncols)
    rec_index = Dict{CountKey,Matrix{Int32}}(r.key => r.matrix for r in records)

    # Build approximate cover masks (spatial layout from union of nonzeros)
    approx_masks = build_cover_masks(records; covers=covers)

    # For CR computation we use the exact N from cover_n (matching manuscript)
    # Spatial blocks are laid out over the full grid; only mask cells count
    # We use the same fixed mask that summary_statistics uses.

    is_ka(m) = (m === :const2 || m === :const8)

    rows = NamedTuple[]
    for cov in covers, ret in returns, m in missions
        ks = is_ka(m) ? [:NA] : kernels
        for k in ks
            key = CountKey(ret=ret, cover=cov, mission=m, kernel=k)
            haskey(rec_index, key) || continue

            mat  = rec_index[key]
            mask = approx_masks[cov]

            # Adjust N to exact polygon size if known
            N_eff = get(cover_n, cov, count(mask))

            # Compute point estimate with exact N
            n_hit = count(i -> mask[i] && mat[i] > 0, eachindex(mat, mask))
            cr_pt = N_eff > 0 ? n_hit / N_eff : 0.0

            # Run bootstrap (uses N from mask cells drawn, not N_eff)
            ci = bootstrap_cr(mat, mask;
                              nboot=nboot, block_frac=block_frac,
                              seed=seed, level=level)

            push!(rows, (
                Cover        = cover_label(cov),
                Return       = return_label(ret),
                Mission      = mission_label(m),
                Kernel       = kernel_label(k),
                CR           = cr_pt,         # manuscript CR (uses DURHAM_COVER_N)
                CR_bootstrap = ci.estimate,   # bootstrap-internal CR (uses union mask N)
                CI_lower     = ci.lower,
                CI_upper     = ci.upper,
                level        = level,
                nboot        = nboot,
                block_side   = ci.block_side,
            ))
        end
    end

    df = DataFrame(rows)
    sort!(df, [:Return, :Cover, :Mission, :Kernel])
    return df
end

# ---------------------------------------------------------------------------
# End-to-end table API: bootstrap_cr_difference_table
# ---------------------------------------------------------------------------

"""
    Comparison

Specifies a pairwise CR comparison for use with [`bootstrap_cr_difference_table`](@ref).

Fields
------
- `key_a` — [`CountKey`](@ref) for mission A (numerator)
- `key_b` — [`CountKey`](@ref) for mission B (denominator/reference)
- `label` — Short description for the output table (e.g. `"KDE-guided E vs Const 2 m/s"`)
"""
struct Comparison
    key_a ::CountKey
    key_b ::CountKey
    label ::String
end

"""
    bootstrap_cr_difference_table(path_json   ::AbstractString,
                                  comparisons ::Vector{Comparison};
                                  cover_n     ::Dict{Symbol,Int} = DURHAM_COVER_N,
                                  nrows       ::Int = 324,
                                  ncols       ::Int = 263,
                                  nboot       ::Int    = 5000,
                                  block_frac  ::Float64 = 0.03,
                                  seed        ::Int    = 42,
                                  level       ::Float64 = 0.95,
    ) -> DataFrame

Compute paired block-bootstrap CIs for CR differences for a list of comparisons.

Each comparison specifies two [`CountKey`](@ref) objects; the mask is taken
from the cover field of `key_a` (both keys must share the same cover type).

Returns a `DataFrame` with columns:
`Comparison`, `Cover`, `Return`, `CR_A`, `CR_B`, `CR_diff`,
`CI_lower`, `CI_upper`, `level`, `nboot`, `block_side`.

## Narrative comparisons (Darien's two key contrasts)

```julia
comparisons = [
    Comparison(
        CountKey(ret=:ground, cover=:field,  mission=:density, kernel=:E),
        CountKey(ret=:ground, cover=:field,  mission=:const2,  kernel=:NA),
        "KDE-guided Epanechnikov vs Constant 2 m/s"
    ),
    Comparison(
        CountKey(ret=:ground, cover=:conif,  mission=:density, kernel=:E),
        CountKey(ret=:ground, cover=:conif,  mission=:const8,  kernel=:NA),
        "KDE-guided Epanechnikov vs Constant 8 m/s"
    ),
]
df = bootstrap_cr_difference_table("counts.json", comparisons)
```
"""
function bootstrap_cr_difference_table(
    path_json  ::AbstractString,
    comparisons::Vector{Comparison};
    cover_n    ::Dict{Symbol,Int} = DURHAM_COVER_N,
    nrows      ::Int = 324,
    ncols      ::Int = 263,
    nboot      ::Int    = 5000,
    block_frac ::Float64 = 0.03,
    seed       ::Int    = 42,
    level      ::Float64 = 0.95,
)::DataFrame

    records   = load_counts_json(path_json; nrows=nrows, ncols=ncols)
    rec_index = Dict{CountKey,Matrix{Int32}}(r.key => r.matrix for r in records)

    all_covers = unique([c.key_a.cover for c in comparisons] ∪
                        [c.key_b.cover for c in comparisons])
    approx_masks = build_cover_masks(records;
                                     covers=collect(all_covers))

    rows = NamedTuple[]
    for comp in comparisons
        ka, kb = comp.key_a, comp.key_b
        ka.cover === kb.cover ||
            @warn "Comparison '$(comp.label)': key_a.cover ($(ka.cover)) ≠ key_b.cover ($(kb.cover)); using key_a.cover mask"

        haskey(rec_index, ka) ||
            (@warn "Record for key_a not found: $ka"; continue)
        haskey(rec_index, kb) ||
            (@warn "Record for key_b not found: $kb"; continue)

        mat_a = rec_index[ka]
        mat_b = rec_index[kb]
        mask  = approx_masks[ka.cover]
        N_eff = get(cover_n, ka.cover, count(mask))

        n_hit_a = count(i -> mask[i] && mat_a[i] > 0, eachindex(mat_a, mask))
        n_hit_b = count(i -> mask[i] && mat_b[i] > 0, eachindex(mat_b, mask))
        cr_a    = N_eff > 0 ? n_hit_a / N_eff : 0.0
        cr_b    = N_eff > 0 ? n_hit_b / N_eff : 0.0

        ci = bootstrap_cr_difference(mat_a, mat_b, mask;
                                     nboot=nboot, block_frac=block_frac,
                                     seed=seed, level=level)

        push!(rows, (
            Comparison       = comp.label,
            Cover            = cover_label(ka.cover),
            Return           = return_label(ka.ret),
            Mission_A        = mission_label(ka.mission),
            Kernel_A         = kernel_label(ka.kernel),
            Mission_B        = mission_label(kb.mission),
            Kernel_B         = kernel_label(kb.kernel),
            CR_A             = cr_a,          # manuscript CR_A (uses DURHAM_COVER_N)
            CR_B             = cr_b,          # manuscript CR_B
            CR_diff          = cr_a - cr_b,   # manuscript CR difference
            CR_diff_bootstrap= ci.estimate,   # bootstrap-internal CR difference
            CI_lower         = ci.lower,
            CI_upper         = ci.upper,
            level            = level,
            nboot            = nboot,
            block_side       = ci.block_side,
        ))
    end

    return DataFrame(rows)
end

"""
    default_narrative_comparisons(; returns=[:ground]) -> Vector{Comparison}

Return the two primary narrative comparisons from the manuscript for all
cover types and specified return types:

1. KDE-guided Epanechnikov (density-aware) vs. Constant 2 m/s
2. KDE-guided Epanechnikov (density-aware) vs. Constant 8 m/s

These use `mission=:density, kernel=:E` as the KDE-guided strategy.

Pass `returns=[:all, :ground]` for both return types.
"""
function default_narrative_comparisons(;
    covers  ::Vector{Symbol} = [:field, :decid, :conif],
    returns ::Vector{Symbol} = [:ground],
)::Vector{Comparison}
    comps = Comparison[]
    for cov in covers, ret in returns
        push!(comps, Comparison(
            CountKey(ret=ret, cover=cov, mission=:density, kernel=:E),
            CountKey(ret=ret, cover=cov, mission=:const2,  kernel=:NA),
            "KDE-guided Epanechnikov vs Constant 2 m/s — $(cover_label(cov)) $(return_label(ret))",
        ))
        push!(comps, Comparison(
            CountKey(ret=ret, cover=cov, mission=:density, kernel=:E),
            CountKey(ret=ret, cover=cov, mission=:const8,  kernel=:NA),
            "KDE-guided Epanechnikov vs Constant 8 m/s — $(cover_label(cov)) $(return_label(ret))",
        ))
    end
    return comps
end
