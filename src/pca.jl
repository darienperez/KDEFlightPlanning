# pca.jl — Principal Component Analysis helpers
#
# Ported from CanopyDensity/pca.jl.
# Depends on MultivariateStats.jl (listed in [deps]).
# All `using` statements are centralised in KDEFlightPlanning.jl;
# this file assumes MultivariateStats is already in scope.

"""
    pca_fit(X; variance_ratio=0.95, maxoutdim=nothing) -> PCA

Fit PCA on rows of `X` (n_samples × n_features).

Arguments
---------
- `variance_ratio` :: Union{Nothing,Float64}
    Target cumulative explained variance (e.g. 0.95).  `nothing` = no target.
- `maxoutdim` :: Union{Nothing,Int}
    Hard cap on output components.  `nothing` = no cap.

Returns a `MultivariateStats.PCA` model.
`X` should already be centred/scaled before calling this.
"""
function pca_fit(X::AbstractMatrix;
                 variance_ratio::Union{Nothing,Float64}=0.95,
                 maxoutdim::Union{Nothing,Int}=nothing)
    n, d = size(X)
    T    = float(eltype(X))

    if variance_ratio !== nothing
        0.0 < variance_ratio ≤ 1.0 ||
            throw(ArgumentError("variance_ratio must be in (0, 1]"))
    end
    if maxoutdim !== nothing
        1 ≤ maxoutdim ≤ d ||
            throw(ArgumentError("maxoutdim must be between 1 and $d"))
    end

    kwargs = Dict{Symbol,Any}(:method => :auto, :mean => zeros(T, d))
    variance_ratio !== nothing && (kwargs[:pratio]    = variance_ratio)
    maxoutdim       !== nothing && (kwargs[:maxoutdim] = maxoutdim)

    return MultivariateStats.fit(MultivariateStats.PCA, X'; kwargs...)
end

"""
    pca_transform(pca, X) -> Matrix

Project rows of `X` (n × d) into the PCA space.
Returns scores as `(n × k)` where k = number of retained components.
"""
function pca_transform(pca::MultivariateStats.PCA, X::AbstractMatrix)
    # `predict` is the current API; `transform` is deprecated in MultivariateStats ≥ 0.10
    Yt = MultivariateStats.predict(pca, X')   # k × n
    return permutedims(Yt)                     # n × k
end

"""
    pca_explained(pca) -> NamedTuple

Convenience: return `(eigvals, explained, cumulative)` from a fitted PCA model.
"""
function pca_explained(pca::MultivariateStats.PCA)
    λ = isdefined(MultivariateStats, :principalvars) ?
        MultivariateStats.principalvars(pca) :
        MultivariateStats.prinvars(pca)
    tot = sum(λ)
    ex  = tot == 0 ? zero.(λ) : λ ./ tot
    cum = cumsum(ex)
    return (eigvals=λ, explained=ex, cumulative=cum)
end
