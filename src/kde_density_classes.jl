"""
    kde_density_classes.jl — KDE-derived density class / strata assignment

## Design

This module is written to idiomatic Julia style:

- **`ThresholdMethod`** is an abstract type with four concrete subtypes —
  `MultiOtsuThreshold` (default), `OtsuThreshold`, `QuantileThreshold`, and
  `ManualThreshold` — enabling multiple dispatch on the thresholding strategy.
  No `method::Symbol` flag is needed; callers construct a typed token and the
  compiler routes to the correct `_compute_thresholds` method.

- **`MultiOtsuThreshold`** is the default and recommended method. It runs the
  3-class Otsu criterion (Otsu 1979; Liao et al. 2001) over the **full**
  normalised KDE surface including zeros. The zero-density mass anchors the
  first split, so field-like pixels (zero and near-zero densities) are cleanly
  separated from deciduous-like and coniferous-like ranges.

- **`DensityClassResult`** is an immutable struct that bundles the class
  assignment matrix with all threshold metadata. It is the canonical return
  value of `assign_kde_density_classes` and the sole argument accepted by
  `kde_class_support`, `kde_class_cr`, and `kde_class_narrative`.
  Functions downstream never need to reconstruct threshold parameters from
  a NamedTuple; the struct carries everything.

- **`DensityClass`** enum-like constants (`KDE_CLASS_FIELD`, etc.) are
  plain `Int8` values with a lookup `KDE_CLASS_LABELS` dict, avoiding a
  full `@enum` which would conflict with integer arithmetic on count grids.

- `kde_class_cr` is overloaded: one method takes `(DensityClassResult, M)`,
  another takes `(M_class, M_count)` for callers who already have the
  class matrix.

## Epistemological note

KDE-derived classes reflect the KDE planning surface, **not** independent
field observations. They are mechanism-diagnostic only.  GLI (Sullivan et al.
2023, DOI 10.3390/rs15215091) remains the primary external ground-truth
reference for cover type.

## Separation from GLI ground truth

`kde_strata_within_cover` (kde_strata.jl) accepts an external GLI-derived
`cover_mask`. This module derives classes solely from the KDE surface itself.
The two APIs serve different purposes and must not be conflated.

## Public API

    # Threshold method tokens (default: MultiOtsuThreshold)
    MultiOtsuThreshold()             # 3-class Otsu on full surface incl. zeros
    OtsuThreshold()                  # legacy: 1-threshold Otsu on positive pixels
    QuantileThreshold(p = 0.5)       # quantile of positive pixels
    ManualThreshold(lower, upper)    # user-supplied both thresholds
    ManualThreshold(upper)           # compat: sets lower via eps

    # Threshold computation (dispatched on method type)
    compute_kde_thresholds(Z, method; eps) -> NamedTuple

    # Class assignment
    assign_kde_density_classes(Z; method, eps) -> DensityClassResult

    # Result inspection
    kde_class_support(r::DensityClassResult) -> NamedTuple
    kde_class_cr(r::DensityClassResult, count_grid) -> NamedTuple
    kde_class_cr(class_matrix, count_grid)            # bare-matrix overload

    # Multi-mission summary
    kde_class_summary(Z, count_grid_dict; method, missions, ...) -> DataFrame
    export_kde_class_summary_csv(df, path) -> path

    # Narrative
    kde_class_narrative(r::DensityClassResult) -> String
    kde_class_narrative(df::DataFrame; ...)   -> String
"""

# ===========================================================================
# DensityClass labels and integer constants
# ===========================================================================

"""
    KDE_CLASS_FIELD      = Int8(1)  — zero / near-zero density (field-like)
    KDE_CLASS_DECIDUOUS  = Int8(2)  — low-to-medium density (deciduous-like)
    KDE_CLASS_CONIFEROUS = Int8(3)  — high density (coniferous-like)

Labels for density classes produced by `assign_kde_density_classes`.
Integer values are `Int8` so the class matrix is memory-compact.
"""
const KDE_CLASS_FIELD      = Int8(1)
const KDE_CLASS_DECIDUOUS  = Int8(2)
const KDE_CLASS_CONIFEROUS = Int8(3)

"""
    KDE_CLASS_LABELS :: Dict{Int8, String}

Lookup table from class integer to human label.
"""
const KDE_CLASS_LABELS = Dict{Int8,String}(
    KDE_CLASS_FIELD      => "field-like",
    KDE_CLASS_DECIDUOUS  => "deciduous-like",
    KDE_CLASS_CONIFEROUS => "coniferous-like",
)

# ===========================================================================
# ThresholdMethod type hierarchy — dispatch on strategy, not on a Symbol
# ===========================================================================

"""
    ThresholdMethod

Abstract supertype for KDE density class thresholding strategies.

Concrete subtypes:
- `MultiOtsuThreshold`   — 3-class Otsu on the full surface including zeros (default)
- `OtsuThreshold`        — legacy 1-threshold Otsu on positive-density pixels only
- `QuantileThreshold(p)` — quantile split at probability `p` of positive pixels
- `ManualThreshold(lower, upper)` — user-supplied both threshold values

Use the concrete type directly as an argument:
    assign_kde_density_classes(Z)                                    # MultiOtsu (default)
    assign_kde_density_classes(Z; method = MultiOtsuThreshold())
    assign_kde_density_classes(Z; method = OtsuThreshold())          # legacy
    assign_kde_density_classes(Z; method = QuantileThreshold(0.6))
    assign_kde_density_classes(Z; method = ManualThreshold(0.15, 0.55))
"""
abstract type ThresholdMethod end

"""
    MultiOtsuThreshold <: ThresholdMethod

Token for the 3-class multi-Otsu criterion (Otsu 1979; Liao et al. 2001).

Finds two thresholds (t1, t2) with t1 < t2 that maximise total between-class
variance over the **full** normalised KDE surface, including zero-density
pixels. Because the zero-density mass is included, the lower split naturally
isolates field-like pixels (open-area zeros and near-zero spillover) from
vegetation-dominated ranges.

This is the **default and recommended** method.

## Algorithm

Exhaustive search over a 256-bin histogram:
    σ²_B(t1,t2) = ω₁(μ₁−μ_T)² + ω₂(μ₂−μ_T)² + ω₃(μ₃−μ_T)²

where ωₖ, μₖ are the frequency and mean of class k = {field, deciduous, coniferous}.
"""
struct MultiOtsuThreshold <: ThresholdMethod end

"""
    OtsuThreshold <: ThresholdMethod

Legacy token for Otsu's binary criterion on **positive-density pixels only**.
Operates as a single-threshold (upper only) method; the field/deciduous split
falls at `eps` as before.

Preserved for sensitivity analysis and backward compatibility.
For new work, prefer `MultiOtsuThreshold()`.
"""
struct OtsuThreshold <: ThresholdMethod end

"""
    QuantileThreshold <: ThresholdMethod

Token for quantile-based thresholding.

## Field
- `p :: Float64` — probability in (0, 1); threshold = `quantile(positive_vals, p)`.
  Default `p = 0.5` (median of positive-density pixels).

## Example
    assign_kde_density_classes(Z; method = QuantileThreshold(0.6))
"""
struct QuantileThreshold <: ThresholdMethod
    p :: Float64
    function QuantileThreshold(p::Real = 0.5)
        0.0 < p < 1.0 ||
            throw(ArgumentError("QuantileThreshold: p must be in (0,1), got p=$p"))
        new(Float64(p))
    end
end

"""
    ManualThreshold <: ThresholdMethod

Token for user-specified density class boundaries.

## Fields
- `lower :: Float64` — field-like / deciduous-like boundary in (0, 1].
  Pixels with density ≤ `lower` are field-like.
- `upper :: Float64` — deciduous-like / coniferous-like boundary in (0, 1],
  must satisfy `upper > lower`.
  Pixels with `lower < density ≤ upper` are deciduous-like;
  pixels with density > `upper` are coniferous-like.

## Constructors
    ManualThreshold(lower, upper)   # both boundaries explicit
    ManualThreshold(upper)          # compat: lower = eps (field ≡ zero/near-zero)

## Example
    assign_kde_density_classes(Z; method = ManualThreshold(0.15, 0.55))
    assign_kde_density_classes(Z; method = ManualThreshold(0.35))  # compat form
"""
struct ManualThreshold <: ThresholdMethod
    lower :: Float64
    upper :: Float64
    function ManualThreshold(lower::Real, upper::Real)
        lower >= 0.0 ||
            throw(ArgumentError("ManualThreshold: lower must be ≥ 0, got lower=$lower"))
        upper > lower ||
            throw(ArgumentError("ManualThreshold: upper must be > lower (lower=$lower, upper=$upper)"))
        upper <= 1.0 ||
            @warn "ManualThreshold: upper=$upper > 1.0; ensure the surface is normalised."
        new(Float64(lower), Float64(upper))
    end
    # Backward-compatible single-argument form: lower = 0 (field ≡ ≤ eps)
    function ManualThreshold(upper::Real)
        upper > 0.0 ||
            throw(ArgumentError("ManualThreshold: upper must be > 0, got upper=$upper"))
        upper <= 1.0 ||
            @warn "ManualThreshold: upper=$upper > 1.0; ensure the surface is normalised."
        new(0.0, Float64(upper))
    end
end

Base.show(io::IO, ::MultiOtsuThreshold)  = print(io, "MultiOtsuThreshold()")
Base.show(io::IO, ::OtsuThreshold)       = print(io, "OtsuThreshold()")
Base.show(io::IO, m::QuantileThreshold)  = @printf(io, "QuantileThreshold(p=%.3f)", m.p)
Base.show(io::IO, m::ManualThreshold)    = @printf(io, "ManualThreshold(lower=%.4f, upper=%.4f)", m.lower, m.upper)

# ===========================================================================
# DensityClassResult — typed return value of assign_kde_density_classes
# ===========================================================================

"""
    DensityClassResult

Immutable struct returned by `assign_kde_density_classes`.

## Fields

| Field              | Type            | Description                                     |
|--------------------|-----------------|-------------------------------------------------|
| `class_matrix`     | Matrix{Int8}    | Per-pixel class label (1/2/3)                   |
| `eps_threshold`    | Float64         | Absolute zero guard (≤ eps → field-like)        |
| `lower_threshold`  | Float64         | Field-like / deciduous-like boundary            |
| `upper_threshold`  | Float64         | Deciduous-like / coniferous-like boundary       |
| `method`           | ThresholdMethod | Method used to compute the thresholds           |
| `n_positive`       | Int             | # positive-density pixels in input              |
| `n_field`          | Int             | # zero/near-zero pixels (≤ eps or ≤ lower)      |
| `pos_density_stats`| NamedTuple      | (min, median, mean, max) of positives           |

For `MultiOtsuThreshold`, `lower_threshold` is the data-driven t1 (field/deciduous
boundary). For legacy methods (`OtsuThreshold`, `QuantileThreshold`),
`lower_threshold == eps_threshold` so the existing three-region logic is unchanged.

## Usage

```julia
r = assign_kde_density_classes(Z)   # MultiOtsuThreshold() default

# Support counts
s = kde_class_support(r)
println(s.field, s.deciduous, s.coniferous)

# Per-class CR for one count grid
cr = kde_class_cr(r, count_grid)
println(cr.coniferous.cr)

# Narrative summary
println(kde_class_narrative(r))
```
"""
struct DensityClassResult
    class_matrix     :: Matrix{Int8}
    eps_threshold    :: Float64
    lower_threshold  :: Float64   # field-like / deciduous-like split
    upper_threshold  :: Float64   # deciduous-like / coniferous-like split
    method           :: ThresholdMethod
    n_positive       :: Int
    n_field          :: Int
    pos_density_stats:: @NamedTuple{min::Float64, median::Float64,
                                    mean::Float64, max::Float64}
end

function Base.show(io::IO, r::DensityClassResult)
    H, W = size(r.class_matrix)
    nf = count(==(KDE_CLASS_FIELD),      r.class_matrix)
    nd = count(==(KDE_CLASS_DECIDUOUS),  r.class_matrix)
    nc = count(==(KDE_CLASS_CONIFEROUS), r.class_matrix)
    print(io, @sprintf("DensityClassResult(%d\u00d7%d  method=%s  lower=%.4f  upper=%.4f  field=%d  decid=%d  conif=%d)",
                        H, W, r.method, r.lower_threshold, r.upper_threshold, nf, nd, nc))
end

# ===========================================================================
# _compute_thresholds — dispatched on ThresholdMethod; returns (lower, upper)
# ===========================================================================

"""
    _compute_thresholds(method::ThresholdMethod,
                         all_vals::Vector{Float64},
                         pos_vals::Vector{Float64},
                         eps::Float64) -> Tuple{Float64, Float64}

Return `(lower_threshold, upper_threshold)` for three-class density assignment:

    field-like      :  Z ≤ lower_threshold
    deciduous-like  :  lower_threshold < Z ≤ upper_threshold
    coniferous-like :  Z > upper_threshold

Dispatched on the concrete `ThresholdMethod` type.

- `all_vals`  — full flattened KDE surface (includes zeros)
- `pos_vals`  — subset where Z > eps (for legacy methods)
- `eps`       — absolute zero guard
"""
function _compute_thresholds(::MultiOtsuThreshold,
                               all_vals::Vector{Float64},
                               ::Vector{Float64},
                               eps::Float64)::Tuple{Float64,Float64}
    isempty(all_vals) && return (eps, eps)
    t1, t2 = _otsu_threshold_3class(all_vals; n_bins=256)
    # Ensure lower bound is at least eps so truly-zero pixels stay field-like
    t1 = max(t1, eps)
    t2 = max(t2, t1 + eps)
    return (t1, t2)
end

function _compute_thresholds(::OtsuThreshold,
                               ::Vector{Float64},
                               pos_vals::Vector{Float64},
                               eps::Float64)::Tuple{Float64,Float64}
    t_hi = if length(pos_vals) < 2
        length(pos_vals) == 1 ? pos_vals[1] : 0.0
    else
        _otsu_threshold(pos_vals; n_bins=256)
    end
    return (eps, t_hi)
end

function _compute_thresholds(m::QuantileThreshold,
                               ::Vector{Float64},
                               pos_vals::Vector{Float64},
                               eps::Float64)::Tuple{Float64,Float64}
    t_hi = isempty(pos_vals) ? 1e-6 : quantile(pos_vals, m.p)
    return (eps, t_hi)
end

function _compute_thresholds(m::ManualThreshold,
                               ::Vector{Float64},
                               ::Vector{Float64},
                               eps::Float64)::Tuple{Float64,Float64}
    # If lower == 0.0 (compat form), treat it as eps so zeros stay field-like
    t_lo = m.lower > 0.0 ? m.lower : eps
    return (t_lo, m.upper)
end

# ---------------------------------------------------------------------------
# Otsu implementations (pure Julia, no extra deps)
# ---------------------------------------------------------------------------

"""
    _otsu_threshold(vals; n_bins=256) -> Float64

Otsu's optimal binary threshold for a 1-D vector. Maximises:
    σ²_B(t) = ω₀ ω₁ (μ₀ − μ₁)²
where ω₀, ω₁ are class frequencies and μ₀, μ₁ are class means.

Used by the legacy `OtsuThreshold` path (positive pixels only).
"""
function _otsu_threshold(vals::Vector{Float64}; n_bins::Int = 256)::Float64
    isempty(vals) && return 0.0
    vmin, vmax = extrema(vals)
    vmax ≈ vmin && return vmin

    bin_width = (vmax - vmin) / n_bins
    hist      = zeros(Float64, n_bins)

    for v in vals
        b = clamp(floor(Int, (v - vmin) / bin_width) + 1, 1, n_bins)
        hist[b] += 1.0
    end
    hist ./= sum(hist)

    total_mean = sum((i - 0.5) * hist[i] for i in 1:n_bins)
    best_var   = -Inf
    best_t_idx = 1
    ω0 = 0.0
    μ0_sum = 0.0

    for t in 1:n_bins-1
        ω0    += hist[t]
        μ0_sum += (t - 0.5) * hist[t]
        ω1 = 1.0 - ω0
        ω1 <= 0.0 && continue
        m0  = ω0 > 0.0 ? μ0_sum / ω0 : 0.0
        m1  = (total_mean - μ0_sum) / ω1
        σ²B = ω0 * ω1 * (m0 - m1)^2
        if σ²B > best_var
            best_var   = σ²B
            best_t_idx = t
        end
    end

    return vmin + best_t_idx * bin_width
end

"""
    _otsu_threshold_3class(vals; n_bins=256) -> Tuple{Float64, Float64}

Multi-Otsu 3-class threshold for a 1-D vector (Otsu 1979; Liao et al. 2001).

Finds (t1, t2) with t1 < t2 that maximise total between-class variance:
    σ²_B(t1,t2) = Σₖ ωₖ (μₖ − μ_T)²
where ωₖ, μₖ are frequency and mean of class k ∈ {1,2,3}.

The search is over all valid (t1, t2) bin-index pairs in an exhaustive
O(B²/2) sweep. Cumulative sum prefix arrays make the inner loop O(1).

Runs on the **full** surface including zeros.
"""
function _otsu_threshold_3class(vals::Vector{Float64}; n_bins::Int = 256)::Tuple{Float64,Float64}
    isempty(vals) && return (0.0, 0.0)
    vmin, vmax = extrema(vals)
    if vmax ≈ vmin
        # Degenerate — all values identical; both thresholds at the single value
        return (vmin, vmin)
    end

    bin_width = (vmax - vmin) / n_bins
    hist = zeros(Float64, n_bins)
    for v in vals
        b = clamp(floor(Int, (v - vmin) / bin_width) + 1, 1, n_bins)
        hist[b] += 1.0
    end
    hist ./= sum(hist)   # normalise to probabilities

    # Precompute prefix sums: P[i] = Σ hist[1..i], S[i] = Σ (k-0.5)*hist[k] for k=1..i
    P = cumsum(hist)
    S = cumsum((i - 0.5) * hist[i] for i in 1:n_bins)
    μ_T = S[end]   # total mean

    best_var   = -Inf
    best_t1    = 1
    best_t2    = 2

    for t1 in 1:n_bins-2
        ω1 = P[t1]
        ω1 <= 0.0 && continue
        μ1 = S[t1] / ω1

        for t2 in t1+1:n_bins-1
            ω2_raw = P[t2] - P[t1]
            ω2_raw <= 0.0 && continue
            μ2 = (S[t2] - S[t1]) / ω2_raw

            ω3 = 1.0 - P[t2]
            ω3 <= 0.0 && continue
            μ3 = (μ_T - S[t2]) / ω3

            σ²B = ω1*(μ1 - μ_T)^2 + ω2_raw*(μ2 - μ_T)^2 + ω3*(μ3 - μ_T)^2
            if σ²B > best_var
                best_var = σ²B
                best_t1  = t1
                best_t2  = t2
            end
        end
    end

    t1_val = vmin + best_t1 * bin_width
    t2_val = vmin + best_t2 * bin_width
    return (t1_val, t2_val)
end

# ===========================================================================
# compute_kde_thresholds — public threshold-only API
# ===========================================================================

"""
    compute_kde_thresholds(Z::AbstractMatrix,
                            method::ThresholdMethod = MultiOtsuThreshold();
                            eps::Float64 = 1e-9) -> NamedTuple

Compute class thresholds from a normalised KDE surface, without performing
the full class assignment.

Useful for inspecting thresholds before committing to a particular method.

## Returns

```
(eps_threshold, lower_threshold, upper_threshold, method,
 n_positive, n_field, pos_density_stats)
```

`lower_threshold` is the field-like / deciduous-like split.  For
`MultiOtsuThreshold` this is the data-driven t1 from the 3-class Otsu
criterion. For legacy methods it equals `eps`.

## Example

```julia
t = compute_kde_thresholds(Z)                        # MultiOtsu (default)
t = compute_kde_thresholds(Z, OtsuThreshold())       # legacy 1-threshold
t = compute_kde_thresholds(Z, QuantileThreshold(0.6))
t = compute_kde_thresholds(Z, ManualThreshold(0.15, 0.55))
println("lower_threshold = ", t.lower_threshold)
println("upper_threshold = ", t.upper_threshold)
```
"""
function compute_kde_thresholds(Z::AbstractMatrix,
                                 method::ThresholdMethod = MultiOtsuThreshold();
                                 eps::Float64 = 1e-9)

    flat     = vec(Float64.(Z))
    pos_vals = flat[flat .> eps]
    n_pos    = length(pos_vals)

    t_lower, t_upper = _compute_thresholds(method, flat, pos_vals, eps)

    # n_field: pixels at or below the effective lower split
    n_field = count(v -> v <= t_lower, flat)

    pos_stats = if n_pos > 0
        (min    = minimum(pos_vals),
         median = quantile(pos_vals, 0.5),
         mean   = mean(pos_vals),
         max    = maximum(pos_vals))
    else
        (min=0.0, median=0.0, mean=0.0, max=0.0)
    end

    return (
        eps_threshold    = eps,
        lower_threshold  = t_lower,
        upper_threshold  = t_upper,
        method           = method,
        n_positive       = n_pos,
        n_field          = n_field,
        pos_density_stats = pos_stats,
    )
end

# ===========================================================================
# assign_kde_density_classes — primary classification API
# ===========================================================================

"""
    assign_kde_density_classes(Z::AbstractMatrix;
                                method :: ThresholdMethod = MultiOtsuThreshold(),
                                eps    :: Float64         = 1e-9)
        -> DensityClassResult

Assign each pixel to a KDE-derived density class.

## Class definitions

| Label           | Condition                               | Int8 value |
|-----------------|-----------------------------------------|------------|
| field-like      | `Z[i,j] ≤ lower_threshold`              | `1`        |
| deciduous-like  | `lower_threshold < Z[i,j] ≤ upper_threshold` | `2`   |
| coniferous-like | `Z[i,j] > upper_threshold`              | `3`        |

For `MultiOtsuThreshold` (default), `lower_threshold` is the data-driven t1
from the 3-class Otsu criterion over the full surface including zeros.
Zero-density pixels are **always** classified as field-like and are never
dropped, because zero-density regions are meaningful (open-area pixels).

## Method dispatch

Pass a `ThresholdMethod` token to select the strategy:

```julia
r = assign_kde_density_classes(Z)                          # MultiOtsu (default)
r = assign_kde_density_classes(Z; method=OtsuThreshold())  # legacy 1-threshold
r = assign_kde_density_classes(Z; method=QuantileThreshold(0.6))
r = assign_kde_density_classes(Z; method=ManualThreshold(0.15, 0.55))
```

## Returns

`DensityClassResult` — carries `class_matrix`, both thresholds, and statistics.

## Example

```julia
r = assign_kde_density_classes(Z_out)
println(r)
# → DensityClassResult(263×324  method=MultiOtsuThreshold()  lower=0.0214  upper=0.4087
#                       field=40384  decid=26632  conif=18196)
```
"""
function assign_kde_density_classes(Z::AbstractMatrix;
                                     method :: ThresholdMethod = MultiOtsuThreshold(),
                                     eps    :: Float64         = 1e-9)::DensityClassResult

    t = compute_kde_thresholds(Z, method; eps=eps)

    t_lo = t.lower_threshold
    t_hi = t.upper_threshold
    H, W = size(Z)

    class_mat = Matrix{Int8}(undef, H, W)
    @inbounds for idx in eachindex(Z)
        v = Float64(Z[idx])
        class_mat[idx] = if v <= t_lo
            KDE_CLASS_FIELD
        elseif v <= t_hi
            KDE_CLASS_DECIDUOUS
        else
            KDE_CLASS_CONIFEROUS
        end
    end

    return DensityClassResult(
        class_mat,
        t.eps_threshold,
        t.lower_threshold,
        t.upper_threshold,
        t.method,
        t.n_positive,
        t.n_field,
        t.pos_density_stats,
    )
end

# ===========================================================================
# kde_class_support — dispatched on DensityClassResult
# ===========================================================================

"""
    kde_class_support(r::DensityClassResult)
    kde_class_support(class_matrix::AbstractMatrix)

Return per-class pixel counts.

## Returns

`NamedTuple` — `(field, deciduous, coniferous, total)`.
Zero-density pixels contribute to `field`; none are dropped.

## Example
```julia
s = kde_class_support(r)
println("field=\$(s.field)  decid=\$(s.deciduous)  conif=\$(s.coniferous)")
```
"""
function kde_class_support(r::DensityClassResult)
    kde_class_support(r.class_matrix)
end

function kde_class_support(class_matrix::AbstractMatrix)
    (field      = count(==(KDE_CLASS_FIELD),      class_matrix),
     deciduous  = count(==(KDE_CLASS_DECIDUOUS),  class_matrix),
     coniferous = count(==(KDE_CLASS_CONIFEROUS), class_matrix),
     total      = length(class_matrix))
end

# ===========================================================================
# kde_class_cr — dispatched on DensityClassResult or bare matrices
# ===========================================================================

"""
    kde_class_cr(r::DensityClassResult, count_grid::AbstractMatrix;
                  min_returns=1) -> NamedTuple

    kde_class_cr(class_matrix::AbstractMatrix, count_grid::AbstractMatrix;
                  min_returns=1) -> NamedTuple

Compute coverage ratio within each KDE-derived density class.

Dispatched on `DensityClassResult` (preferred) or directly on a bare
`class_matrix` (for callers who already have the matrix).

## Returns

NamedTuple:
```
(field      = (cr, n_cells, n_covered),
 deciduous  = (cr, n_cells, n_covered),
 coniferous = (cr, n_cells, n_covered))
```
where each inner NamedTuple contains the coverage ratio and raw counts.

## Example
```julia
r   = assign_kde_density_classes(Z_out)
cr  = kde_class_cr(r, count_grid_kde)
println("coniferous CR = ", round(cr.coniferous.cr; digits=3))
```
"""
function kde_class_cr(r::DensityClassResult, count_grid::AbstractMatrix;
                       min_returns::Int = 1)
    kde_class_cr(r.class_matrix, count_grid; min_returns=min_returns)
end

function kde_class_cr(class_matrix::AbstractMatrix, count_grid::AbstractMatrix;
                       min_returns::Int = 1)
    size(class_matrix) == size(count_grid) ||
        throw(DimensionMismatch(
            "class_matrix $(size(class_matrix)) ≠ count_grid $(size(count_grid))"))

    _cr(cls) = begin
        mask = class_matrix .== cls
        n    = count(mask)
        nc   = n > 0 ? count(mask .& (count_grid .>= min_returns)) : 0
        (cr = n > 0 ? nc / n : 0.0, n_cells = n, n_covered = nc)
    end

    (field      = _cr(KDE_CLASS_FIELD),
     deciduous  = _cr(KDE_CLASS_DECIDUOUS),
     coniferous = _cr(KDE_CLASS_CONIFEROUS))
end

# ===========================================================================
# kde_class_summary — multi-mission DataFrame table
# ===========================================================================

"""
    kde_class_summary(Z::AbstractMatrix,
                       count_grid_dict::AbstractDict{<:AbstractString, <:AbstractMatrix};
                       method       :: ThresholdMethod = MultiOtsuThreshold(),
                       eps          :: Float64 = 1e-9,
                       missions     :: AbstractVector = collect(keys(count_grid_dict)),
                       min_returns  :: Int = 1,
                       kde_status   :: AbstractString = "unknown",
                       alignment    :: AbstractString = "diagnostic") -> DataFrame

Compute KDE-density-class CR for each mission in `count_grid_dict`.

Thresholds are computed **once** from `Z` and applied uniformly across all
missions. This ensures per-class comparisons between missions are valid —
each mission's pixels are classified by the same boundary.

Accepts a `ThresholdMethod` token so the choice is fully explicit and
type-checked at the call site.

## Example

```julia
Z_out  = resample_to_count_grid(surf)
cg_dict = Dict("Const. 2 m/s"              => cg_2mps,
               "KDE-guided (Epanechnikov)" => cg_kde,
               "Const. 8 m/s"              => cg_8mps)

df = kde_class_summary(Z_out, cg_dict;
         method    = OtsuThreshold(),
         kde_status = "true",
         alignment  = "true_surface")
```
"""
function kde_class_summary(Z::AbstractMatrix,
                             count_grid_dict::AbstractDict{<:AbstractString, <:AbstractMatrix};
                             method      ::ThresholdMethod  = MultiOtsuThreshold(),
                             eps         ::Float64          = 1e-9,
                             missions    ::AbstractVector   = collect(keys(count_grid_dict)),
                             min_returns ::Int              = 1,
                             kde_status  ::AbstractString   = "unknown",
                             alignment   ::AbstractString   = "diagnostic")::DataFrame

    result = assign_kde_density_classes(Z; method=method, eps=eps)

    rows = NamedTuple[]
    for mission in missions
        haskey(count_grid_dict, mission) || continue
        cg = count_grid_dict[mission]

        size(cg) == size(Z) ||
            throw(DimensionMismatch(
                "count_grid for '$mission' has shape $(size(cg)) ≠ Z shape $(size(Z))"))

        cr_info = kde_class_cr(result, cg; min_returns=min_returns)

        for (cls, label, info) in (
                (KDE_CLASS_FIELD,      "field-like",      cr_info.field),
                (KDE_CLASS_DECIDUOUS,  "deciduous-like",  cr_info.deciduous),
                (KDE_CLASS_CONIFEROUS, "coniferous-like", cr_info.coniferous),
            )
            push!(rows, (;
                mission           = string(mission),
                kde_status        = string(kde_status),
                alignment         = string(alignment),
                kde_class         = Int(cls),
                class_label       = label,
                eps_threshold     = result.eps_threshold,
                lower_threshold   = result.lower_threshold,
                upper_threshold   = result.upper_threshold,
                threshold_method  = string(result.method),
                n_cells           = info.n_cells,
                n_covered         = info.n_covered,
                cr                = info.cr,
            ))
        end
    end
    return DataFrame(rows)
end

# ===========================================================================
# export_kde_class_summary_csv
# ===========================================================================

"""
    export_kde_class_summary_csv(df::DataFrame, path::AbstractString) -> String

Write a KDE-density-class CR summary table to CSV. Returns the path.
"""
function export_kde_class_summary_csv(df::DataFrame, path::AbstractString)::String
    mkpath(dirname(abspath(path)))
    CSV.write(path, df)
    @info "export_kde_class_summary_csv → $path  ($(nrow(df)) rows)"
    return path
end

# ===========================================================================
# kde_class_narrative — dispatched on DensityClassResult or DataFrame
# ===========================================================================

"""
    kde_class_narrative(r::DensityClassResult) -> String

Generate a concise text summary from a `DensityClassResult`, labelled with
epistemological status. Suitable for embedding in a manuscript Methods or
Supplementary section.

```julia
r = assign_kde_density_classes(Z_out)
println(kde_class_narrative(r))
```
"""
function kde_class_narrative(r::DensityClassResult)::String
    s = kde_class_support(r)
    lines = String[
        "KDE-derived density class thresholds:",
        @sprintf("  Method: %s", r.method),
        @sprintf("  eps (absolute zero guard):              %.2e", r.eps_threshold),
        @sprintf("  lower (field-like ≤):                  %.4f", r.lower_threshold),
        @sprintf("  upper (deciduous-like ≤ / conif >):    %.4f", r.upper_threshold),
        "",
        "Class support (pixels):",
        @sprintf("  field-like:       %6d  (%.1f%%)", s.field,      100s.field/s.total),
        @sprintf("  deciduous-like:   %6d  (%.1f%%)", s.deciduous,  100s.deciduous/s.total),
        @sprintf("  coniferous-like:  %6d  (%.1f%%)", s.coniferous, 100s.coniferous/s.total),
        "",
        "Positive-density stats:",
        @sprintf("  min=%.4f  median=%.4f  mean=%.4f  max=%.4f",
                 r.pos_density_stats.min, r.pos_density_stats.median,
                 r.pos_density_stats.mean, r.pos_density_stats.max),
        "",
        "NOTE: KDE-derived classes are mechanism diagnostics only.",
        "They are NOT a replacement for GLI ground-truth cover types",
        "(Sullivan et al. 2023, DOI 10.3390/rs15215091).",
        "Zero/near-zero density pixels → field-like (retained, not dropped).",
    ]
    return join(lines, "\n")
end

"""
    kde_class_narrative(df::DataFrame;
                         kde_mission       = "KDE-guided (Epanechnikov)",
                         baseline_missions = ["Const. 2 m/s", "Const. 8 m/s"]) -> String

Generate a multi-mission narrative from a summary DataFrame produced by
`kde_class_summary`. Labels outputs with the alignment / status column.
"""
function kde_class_narrative(df::DataFrame;
                              kde_mission      ::AbstractString   = "KDE-guided (Epanechnikov)",
                              baseline_missions::AbstractVector   = ["Const. 2 m/s", "Const. 8 m/s"])::String

    isempty(df) && return "(no KDE class data available)"

    alignments = unique(string.(df.alignment))
    is_true    = any(a -> occursin("true", lowercase(a)), alignments)
    status_str = is_true ? "true-surface diagnostic" : "screenshot-derived diagnostic"

    meth  = isempty(df) ? "?" : string(first(df.threshold_method))
    t_eps = isempty(df) ? 0.0 : first(df.eps_threshold)
    t_lo  = hasproperty(df, :lower_threshold) && !isempty(df) ? first(df.lower_threshold) : t_eps
    t_hi  = isempty(df) ? 0.0 : first(df.upper_threshold)

    lines = String[
        "KDE-derived density class CR summary ($status_str):",
        @sprintf("  Method: %s   eps=%.2e  lower=%.4f  upper=%.4f", meth, t_eps, t_lo, t_hi),
        "  1=field-like (≤lower)   2=deciduous-like (lower..upper]   3=coniferous-like (>upper)",
        "",
    ]

    for m in vcat([kde_mission], collect(baseline_missions))
        sub = filter(r -> r.mission == m, df)
        isempty(sub) && continue
        push!(lines, "  $m:")
        for r in sort(sub, :kde_class) |> eachrow
            push!(lines, @sprintf("    %-20s  CR=%.3f  (%d/%d cells)",
                                   r.class_label, r.cr, r.n_covered, r.n_cells))
        end
    end

    push!(lines, "")
    push!(lines, "  NOTE: KDE-derived classes are mechanism diagnostics only.")
    push!(lines, "  GLI (Sullivan et al. 2023) remains the primary external ground truth.")
    push!(lines, "  Zero-density pixels → field-like (retained, not dropped).")

    return join(lines, "\n")
end
