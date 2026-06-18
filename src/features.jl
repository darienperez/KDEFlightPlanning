# features.jl — Feature extraction from raster imagery
#
# Ported from CanopyDensity/features.jl and CanopyDensity/projector.jl.
# Provides the per-pixel feature matrix used by k-medoids clustering.
#
# All `using` statements are centralised in KDEFlightPlanning.jl.
# Statistics (mean, std) are assumed in scope.

# ---------------------------------------------------------------------------
# Feature stacking
# ---------------------------------------------------------------------------

"""
    stack_features(L, a, b; extra=()) -> Matrix{Float32}

Return an `(H*W × d)` feature matrix from CIELAB channels `L`, `a`, `b`
(each `H×W`). Additional channels (e.g. local texture) can be passed via
`extra` as a tuple of `H×W` arrays.

Column order: `[L, a, b, extra...]`.

Usage in the pipeline:
    L, a, b = rgb_to_lab(img)
    X = stack_features(L, a, b)
    μ, σ = standardize_features!(X)
"""
function stack_features(L::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix;
                         extra::Tuple=())
    H, W = size(L)
    @assert size(a) == (H, W) && size(b) == (H, W) "L, a, b must have the same size"
    for C in extra
        @assert size(C) == (H, W) "extra channel has wrong size ($(size(C)) ≠ ($H, $W))"
    end

    chans = (L, a, b, extra...)
    d     = length(chans)
    X     = Array{Float32}(undef, H*W, d)

    @inbounds for k in 1:d
        X[:, k] = vec(Float32.(chans[k]))
    end
    return X
end

# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

"""
    standardize_features!(X; center=true, scale=true, corrected=true) -> (μ, σ)

Column-wise z-score normalisation of `X::AbstractMatrix{<:AbstractFloat}`.

- `center=true`     → subtract column mean
- `scale=true`      → divide by column std (floored at `eps(eltype(X))`)
- `corrected=true`  → sample std (n-1 denominator)

Modifies `X` in-place. Returns `(μ, σ)` vectors of the same element type.
These can be stored in a `FeatureProjector` for later consistent projection
of new pixels.
"""
function standardize_features!(X::AbstractMatrix{<:AbstractFloat};
                                center::Bool=true, scale::Bool=true,
                                corrected::Bool=true)
    T   = eltype(X)
    n, d = size(X)
    μ = zeros(T, d)
    σ = ones(T, d)
    ε = eps(T)

    @inbounds @views for j in 1:d
        col = X[:, j]

        if center
            m    = mean(col)
            μ[j] = T(m)
            col .-= μ[j]
        end

        if scale
            s = center ?
                sqrt(max(sum(col .* col) / max(n - 1, 1), 0.0)) :
                std(col; corrected=corrected)
            s    = max(T(s), ε)
            σ[j] = s
            col ./= s
        end
    end
    return μ, σ
end

# ---------------------------------------------------------------------------
# Optional texture features
# ---------------------------------------------------------------------------

"""
    local_var(img; r=2) -> Matrix{Float32}

Local variance with square window radius `r` (window size (2r+1)²).
Boundary pixels use partial windows (clamped to valid range).
Useful as an additional texture feature alongside CIELAB channels.
"""
function local_var(img::AbstractMatrix{<:Real}; r::Int=2)
    H, W  = size(img)
    out   = Array{Float32}(undef, H, W)
    @inbounds for i in 1:H, j in 1:W
        i1 = max(1, i-r); i2 = min(H, i+r)
        j1 = max(1, j-r); j2 = min(W, j+r)
        blk = @view img[i1:i2, j1:j2]
        m   = mean(blk)
        s2  = mean((x - m)^2 for x in blk)
        out[i, j] = Float32(s2)
    end
    return out
end

"""
    grad_mag(img) -> Matrix{Float32}

Gradient magnitude using central differences (clamped at borders).
Provides an edge/texture feature complementary to CIELAB channels.
"""
function grad_mag(img::AbstractMatrix{<:Real})
    H, W = size(img)
    out  = Array{Float32}(undef, H, W)
    @inbounds for i in 1:H, j in 1:W
        im1 = max(1, i-1); ip1 = min(H, i+1)
        jm1 = max(1, j-1); jp1 = min(W, j+1)
        gx  = (img[i, jp1] - img[i, jm1]) * 0.5
        gy  = (img[ip1, j] - img[im1, j]) * 0.5
        out[i, j] = Float32(hypot(gx, gy))
    end
    return out
end

# ---------------------------------------------------------------------------
# FeatureProjector  (from CanopyDensity/projector.jl)
# ---------------------------------------------------------------------------

"""
    FeatureProjector{T, P}

Bundle for consistent feature projection on new data:
1. Column-wise standardisation with stored `mu`, `sigma`.
2. Optional PCA projection (stored in `pca`; `nothing` = skip PCA).

Construct via `feature_projector(μ, σ; pca=nothing)`.

To apply:
    Xproj = apply_projector(fp, X_new)
"""
struct FeatureProjector{T<:AbstractFloat, P}
    mu   ::Vector{T}
    sigma::Vector{T}
    pca  ::P   # MultivariateStats.PCA or Nothing
end

"""
    feature_projector(μ, σ; pca=nothing) -> FeatureProjector

Build a `FeatureProjector` from stored standardisation parameters `μ` and `σ`.
Pass `pca=pca_model` to also apply PCA after standardisation.
"""
feature_projector(μ::AbstractVector, σ::AbstractVector; pca=nothing) =
    FeatureProjector(Vector{Float64}(float.(μ)), Vector{Float64}(float.(σ)), pca)

"""
    apply_projector(fp::FeatureProjector, X) -> Matrix

Apply standardisation and (if present) PCA to rows of `X` (n×d).
Returns `(n × k)` projected feature matrix.
Requires MultivariateStats.jl to be loaded when `fp.pca` is not `nothing`.
"""
function apply_projector(fp::FeatureProjector, X::AbstractMatrix)
    Xstd = (X .- fp.mu') ./ fp.sigma'
    fp.pca === nothing && return Xstd
    # PCA path — MultivariateStats must be loaded; pca_transform is in pca.jl
    return pca_transform(fp.pca, Xstd)
end

"""
    build_projector(X; variance_ratio=0.95, maxoutdim=nothing, use_pca=false) -> (fp, Xproj)

Convenience: standardise `X` in-place, optionally fit PCA, and return a
`FeatureProjector` plus the projected matrix.

Set `use_pca=true` to fit PCA (requires MultivariateStats.jl).
"""
function build_projector(X::AbstractMatrix{<:AbstractFloat};
                          variance_ratio::Union{Nothing,Float64}=0.95,
                          maxoutdim::Union{Nothing,Int}=nothing,
                          use_pca::Bool=false)
    μ, σ = standardize_features!(X)
    if use_pca
        pca  = pca_fit(X; variance_ratio=variance_ratio, maxoutdim=maxoutdim)
        Xprj = pca_transform(pca, X)
        fp   = feature_projector(μ, σ; pca=pca)
    else
        fp   = feature_projector(μ, σ)
        Xprj = X
    end
    return fp, Xprj
end
