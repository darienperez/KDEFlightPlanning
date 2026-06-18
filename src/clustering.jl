# clustering.jl — k-medoids clustering for canopy segmentation
#
# Ported and extended from CanopyDensity.
# Depends on Clustering.jl and Distances.jl (listed in [deps]).
# All `using` statements are centralised in KDEFlightPlanning.jl.

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

"""
    choose_sample_indices(n; nsample, rng=nothing) -> Vector{Int}

Draw `nsample` unique row indices from `1:n` without replacement.
Uses `Random.shuffle!` with optional seeded RNG for reproducibility.
"""
function choose_sample_indices(n::Int; nsample::Int, rng=nothing)
    nsample = min(nsample, n)
    idxs = collect(1:n)
    if rng !== nothing
        Random.shuffle!(rng, idxs)
    else
        Random.shuffle!(idxs)
    end
    return sort!(idxs[1:nsample])
end

"""
    sample_distance_matrix(X, idxs; metric=SqEuclidean()) -> Matrix{Float64}

Compute the pairwise distance matrix for the subset of `X` (rows indexed by
`idxs`) using `metric`. Returns a symmetric `(ns × ns)` matrix where
`ns = length(idxs)`.
"""
function sample_distance_matrix(X::AbstractMatrix, idxs::AbstractVector{Int};
                                  metric=Distances.SqEuclidean())
    Xs = X[idxs, :]
    return Distances.pairwise(metric, Xs; dims=1)
end

# ---------------------------------------------------------------------------
# Core k-medoids fit
# ---------------------------------------------------------------------------

"""
    kmedoids_fit(X; k, idxs_sample=nothing, D=nothing,
                  metric=SqEuclidean(), rng=nothing, seed=nothing)
    -> (labels_full, result_sample, info)

Fit k-medoids on the feature matrix `X` (n × d) and return cluster labels
for **all** rows.

Strategy
--------
1. If `idxs_sample` is provided, clustering is run on that sub-sample using
   pre-computed distance matrix `D` (or `D` is computed from the sample).
2. Each full-data row is then assigned to the nearest sample medoid using
   Euclidean distance in feature space.
3. If `idxs_sample` is `nothing`, the full `X` is used (expensive for large n).

Arguments
---------
- `k`:            Number of clusters.
- `idxs_sample`:  Indices of the sample subset (optional).
- `D`:            Pre-computed `(ns × ns)` distance matrix (optional).
- `metric`:       A `Distances.SemiMetric` for building `D` when absent.
- `rng`:          An initialised `AbstractRNG` (e.g. `MersenneTwister(seed)`).
- `seed`:         Integer seed — ignored if `rng` is provided.

Returns
-------
- `labels_full`:    `Vector{Int}` of length n with cluster assignments 1..k.
- `result_sample`:  The `Clustering.KmedoidsResult` on the sample.
- `info`:           NamedTuple `(D=D, idxs=idxs_sample, medoid_coords=…)`.

Requires Clustering.jl and Distances.jl.
"""
function kmedoids_fit(X::AbstractMatrix;
                       k::Int=2,
                       idxs_sample::Union{Nothing,AbstractVector{Int}}=nothing,
                       D::Union{Nothing,AbstractMatrix}=nothing,
                       metric=Distances.SqEuclidean(),
                       rng=nothing,
                       seed::Union{Nothing,Int}=nothing)
    n, d = size(X)

    # Resolve RNG
    if rng === nothing && seed !== nothing
        rng = Random.MersenneTwister(seed)
    end

    if idxs_sample === nothing
        # Full-data path: cluster all n rows
        idxs = collect(1:n)
        Dm   = D !== nothing ? D :
               Distances.pairwise(metric, X; dims=1)
    else
        idxs = collect(idxs_sample)
        Dm   = D !== nothing ? D :
               sample_distance_matrix(X, idxs; metric=metric)
    end

    # Run k-medoids on the sample distance matrix
    result = Clustering.kmedoids(Dm, k)
    labels_sample = Clustering.assignments(result)

    # Medoid coordinates in feature space (for assignment step)
    sample_X  = X[idxs, :]
    medoid_rows = [idxs[m] for m in result.medoids]   # indices in full X
    medoid_coords = X[medoid_rows, :]                  # k × d

    # Assign ALL rows to nearest medoid (Euclidean in feature space)
    labels_full = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        best_k   = 1
        best_d2  = Inf
        xi = @view X[i, :]
        for c in 1:k
            mc = @view medoid_coords[c, :]
            d2 = sum((xi[j] - mc[j])^2 for j in 1:d)
            if d2 < best_d2
                best_d2 = d2
                best_k  = c
            end
        end
        labels_full[i] = best_k
    end

    info = (D=Dm, idxs=idxs, medoid_coords=medoid_coords,
            medoid_rows=medoid_rows)
    return labels_full, result, info
end

# ---------------------------------------------------------------------------
# Cluster quality metrics for auto-k selection
# ---------------------------------------------------------------------------

"""
    _silhouette_score(D, labels) -> Float64

Mean silhouette coefficient from a (ns × ns) distance matrix and label vector.
Returns a value in [-1, 1]; higher is better.
"""
function _silhouette_score(D::AbstractMatrix, labels::AbstractVector{Int})
    n  = size(D, 1)
    ks = sort(unique(labels))
    length(ks) < 2 && return 0.0

    scores = Float64[]
    for i in 1:n
        ki = labels[i]
        # Mean intra-cluster distance
        same  = findall(j -> labels[j] == ki && j != i, 1:n)
        a_i   = isempty(same) ? 0.0 : mean(D[i, j] for j in same)

        # Mean distance to nearest other cluster
        b_i = Inf
        for kc in ks
            kc == ki && continue
            other = findall(j -> labels[j] == kc, 1:n)
            isempty(other) && continue
            b_kc = mean(D[i, j] for j in other)
            b_i  = min(b_i, b_kc)
        end
        b_i = isinf(b_i) ? 0.0 : b_i

        denom = max(a_i, b_i)
        push!(scores, denom == 0 ? 0.0 : (b_i - a_i) / denom)
    end
    return mean(scores)
end

"""
    _dunn_index(D, labels) -> Float64

Dunn index: min inter-cluster distance / max intra-cluster diameter.
Higher is better.
"""
function _dunn_index(D::AbstractMatrix, labels::AbstractVector{Int})
    ks = sort(unique(labels))
    length(ks) < 2 && return 0.0

    # Max intra-cluster diameter
    max_intra = 0.0
    for kc in ks
        idx = findall(==(kc), labels)
        for i in idx, j in idx
            i < j || continue
            max_intra = max(max_intra, D[i, j])
        end
    end
    max_intra == 0 && return Inf

    # Min inter-cluster distance
    min_inter = Inf
    for a in 1:length(ks), b in (a+1):length(ks)
        ia = findall(==(ks[a]), labels)
        ib = findall(==(ks[b]), labels)
        for i in ia, j in ib
            min_inter = min(min_inter, D[i, j])
        end
    end

    return min_inter / max_intra
end

"""
    _davies_bouldin(X_sample, labels) -> Float64

Davies-Bouldin index computed in feature space.
Lower is better.
"""
function _davies_bouldin(X_sample::AbstractMatrix, labels::AbstractVector{Int})
    ks    = sort(unique(labels))
    nk    = length(ks)
    nk < 2 && return Inf

    # Cluster centroids
    cents = [vec(mean(X_sample[findall(==(c), labels), :]; dims=1)) for c in ks]

    # Mean intra-cluster scatter
    s = [mean(norm(X_sample[j,:] .- cents[ci])
              for j in findall(==(ks[ci]), labels))
         for ci in 1:nk]

    # DB index
    db = 0.0
    for i in 1:nk
        worst = 0.0
        for j in 1:nk
            i == j && continue
            dij = norm(cents[i] .- cents[j])
            dij == 0 && continue
            worst = max(worst, (s[i] + s[j]) / dij)
        end
        db += worst
    end
    return db / nk
end

"""
    _calinski_harabasz(X_sample, labels) -> Float64

Calinski-Harabasz index (Variance Ratio Criterion).
Higher is better.
"""
function _calinski_harabasz(X_sample::AbstractMatrix, labels::AbstractVector{Int})
    n, d = size(X_sample)
    ks   = sort(unique(labels))
    K    = length(ks)
    K < 2 && return 0.0

    grand = vec(mean(X_sample; dims=1))

    # Between-cluster sum of squares
    SS_B = 0.0
    for c in ks
        idx = findall(==(c), labels)
        nc  = length(idx)
        cent = vec(mean(X_sample[idx, :]; dims=1))
        SS_B += nc * sum((cent .- grand).^2)
    end

    # Within-cluster sum of squares
    SS_W = 0.0
    for c in ks
        idx = findall(==(c), labels)
        cent = vec(mean(X_sample[idx, :]; dims=1))
        for i in idx
            SS_W += sum((X_sample[i, :] .- cent).^2)
        end
    end

    SS_W == 0 && return Inf
    return (SS_B / (K - 1)) / (SS_W / (n - K))
end

# ---------------------------------------------------------------------------
# k-sweep
# ---------------------------------------------------------------------------

"""
    ClusterMetrics

Holds quality metrics for a single k value from a sweep.

Fields
------
- `k`:          Number of clusters.
- `silhouette`: Mean silhouette coefficient — higher is better.
- `dunn`:       Dunn index — higher is better.
- `db`:         Davies-Bouldin index — lower is better (`nothing` if failed).
- `cal`:        Calinski-Harabasz index — higher is better (`nothing` if failed).
- `xb`:         Xie-Beni index — TODO, always `nothing` (not included in votes).
- `votes`:      Number of internal-validity metrics that nominated this k.
- `chosen`:     Whether this k was ultimately selected.
- `seed`:       RNG seed used (filled by `sweep_k_quality`).
- `nsample`:    Sample size used (filled by `sweep_k_quality`).
- `strategy`:   Selection strategy used (filled by `choose_k`).
"""
mutable struct ClusterMetrics
    k          ::Int
    silhouette ::Float64
    dunn       ::Float64
    db         ::Union{Float64,Nothing}
    cal        ::Union{Float64,Nothing}
    xb         ::Union{Float64,Nothing}   # TODO: Xie-Beni; never included in votes
    votes      ::Int
    chosen     ::Bool
    seed       ::Union{Int,Nothing}
    nsample    ::Union{Int,Nothing}
    strategy   ::Union{Symbol,Nothing}
end

# Backward-compatible constructors
ClusterMetrics(k, sil, dunn, db, cal, xb) =
    ClusterMetrics(k, sil, dunn, db, cal, xb, 0, false, nothing, nothing, nothing)

"""
    sweep_k_quality(X, idxs, D; ks=2:12, seed=nothing, nsample=nothing)
        -> Vector{ClusterMetrics}

For each k in `ks` (default `2:12` for paper workflows), run k-medoids on
the sample (`X[idxs,:]` with pre-computed distance matrix `D`) and compute
the four internal cluster-quality metrics:

| Metric             | Direction  |
|--------------------|------------|
| Silhouette         | higher ↑   |
| Dunn               | higher ↑   |
| Davies-Bouldin     | lower  ↓   |
| Calinski-Harabasz  | higher ↑   |

Xie-Beni is not implemented and is stored as `nothing`; it is never included
in voting.

Returns a `Vector{ClusterMetrics}` in order of `ks`. The `votes` field of
each entry is filled but `chosen` / `strategy` are set only after calling
`choose_k`.
"""
function sweep_k_quality(X::AbstractMatrix,
                          idxs::AbstractVector{Int},
                          D::AbstractMatrix;
                          ks::AbstractRange=2:12,   # paper default
                          seed::Union{Nothing,Int}=nothing,
                          nsample::Union{Nothing,Int}=nothing)
    Xs = X[idxs, :]
    results = ClusterMetrics[]
    for k in ks
        k > length(idxs) && continue   # can't have more clusters than samples

        res    = Clustering.kmedoids(D, k)
        labs   = Clustering.assignments(res)

        sil  = _silhouette_score(D, labs)
        dunn = _dunn_index(D, labs)
        db   = try _davies_bouldin(Xs, labs) catch; nothing end
        cal  = try _calinski_harabasz(Xs, labs) catch; nothing end

        push!(results, ClusterMetrics(k, sil, dunn, db, cal, nothing,
                                      0, false, seed, nsample, nothing))
    end

    # Compute votes for each k (which metrics nominated it as best)
    _fill_votes!(results)
    return results
end

"""
    _fill_votes!(metrics::Vector{ClusterMetrics})

For each entry in `metrics`, count how many of the four implemented
internal-validity metrics nominated it as the best k.

Criteria:
- silhouette: highest wins
- dunn:       highest wins
- db:         lowest wins
- cal:        highest wins

Xie-Beni is skipped (not implemented).
"""
function _fill_votes!(metrics::Vector{ClusterMetrics})
    isempty(metrics) && return

    _best_k(arr; maximize) = begin
        valid = [(m.k, v) for (m, v) in zip(metrics, arr)
                 if v !== nothing && isfinite(v)]
        isempty(valid) && return nothing
        maximize ? argmax(x->x[2], valid)[1] : argmin(x->x[2], valid)[1]
    end

    winners = Int[]
    let k = _best_k([m.silhouette for m in metrics]; maximize=true)
        k !== nothing && push!(winners, k)
    end
    let k = _best_k([m.dunn for m in metrics]; maximize=true)
        k !== nothing && push!(winners, k)
    end
    let k = _best_k([m.db for m in metrics]; maximize=false)
        k !== nothing && push!(winners, k)
    end
    let k = _best_k([m.cal for m in metrics]; maximize=true)
        k !== nothing && push!(winners, k)
    end
    # xb: TODO — not included

    for m in metrics
        m.votes = count(==(m.k), winners)
    end
end

"""
    choose_k(metrics; strategy=:vote) -> Int

Select the best k from a `Vector{ClusterMetrics}` using one of:

- `:vote`  (default, paper workflow) — majority vote across all four
  implemented internal-validity metrics (silhouette ↑, Dunn ↑,
  Davies-Bouldin ↓, Calinski-Harabasz ↑). Xie-Beni is excluded (not
  implemented). Ties are broken deterministically by preferring the
  **lower k** (parsimony principle), which is the project convention.
- `:silhouette`  — maximise silhouette only.
- `:dunn`        — maximise Dunn index only.
- `:db`          — minimise Davies-Bouldin only.
- `:cal`         — maximise Calinski-Harabasz only.
- `:mode`        — alias for `:vote` (backward compatible).

After selection the `chosen` and `strategy` fields of the winning
`ClusterMetrics` entry are updated in-place.

Tie-breaking (`:vote` / `:mode` only)
--------------------------------------
When two or more k values receive the same number of votes, the smallest k
is returned. This reflects a parsimony preference documented in the paper:
"we prefer fewer clusters on ties to minimise over-segmentation of the
orthomosaic."
"""
function choose_k(metrics::Vector{ClusterMetrics}; strategy::Symbol=:vote)
    isempty(metrics) && throw(ArgumentError("metrics vector is empty"))

    _best(arr; maximize=true) = begin
        valid = [(m.k, v) for (m, v) in zip(metrics, arr)
                 if v !== nothing && isfinite(v)]
        isempty(valid) && return metrics[1].k
        # Deterministic: among tied values pick the smallest k
        if maximize
            best_val = maximum(x->x[2], valid)
            tied = sort([x[1] for x in valid if x[2] == best_val])
            return first(tied)
        else
            best_val = minimum(x->x[2], valid)
            tied = sort([x[1] for x in valid if x[2] == best_val])
            return first(tied)
        end
    end

    chosen_k = if strategy === :silhouette
        _best([m.silhouette for m in metrics]; maximize=true)
    elseif strategy === :dunn
        _best([m.dunn       for m in metrics]; maximize=true)
    elseif strategy === :db
        _best([m.db         for m in metrics]; maximize=false)
    elseif strategy === :cal
        _best([m.cal        for m in metrics]; maximize=true)
    elseif strategy in (:vote, :mode)
        # Plurality vote; ties broken by lowest k (parsimony)
        # Recompute votes in case caller modified metrics
        _fill_votes!(metrics)
        max_votes = maximum(m.votes for m in metrics)
        tied_ks   = sort([m.k for m in metrics if m.votes == max_votes])
        first(tied_ks)
    else
        throw(ArgumentError(
            "Unknown k strategy: :$strategy. " *
            "Use :vote (default), :silhouette, :dunn, :db, :cal, or :mode"))
    end

    # Mark chosen and strategy in-place
    for m in metrics
        if m.k == chosen_k
            m.chosen   = true
            m.strategy = strategy
        end
    end
    return chosen_k
end

"""
    export_cluster_metrics_csv(metrics::Vector{ClusterMetrics}, path::AbstractString)

Write cluster quality metrics to a CSV file for reproducible records.

Columns: `k, silhouette, dunn, davies_bouldin, calinski_harabasz, votes,
chosen, seed, nsample, strategy`.

The file captures the full sweep result including the winning k, enabling
reproduction of the k-selection step from saved artefacts.
"""
function export_cluster_metrics_csv(metrics::Vector{ClusterMetrics},
                                    path::AbstractString)
    mkpath(dirname(abspath(path)))
    rows = [(;
        k                  = m.k,
        silhouette         = m.silhouette,
        dunn               = m.dunn,
        davies_bouldin     = m.db === nothing ? missing : m.db,
        calinski_harabasz  = m.cal === nothing ? missing : m.cal,
        votes              = m.votes,
        chosen             = m.chosen,
        seed               = m.seed === nothing ? missing : m.seed,
        nsample            = m.nsample === nothing ? missing : m.nsample,
        strategy           = m.strategy === nothing ? missing : string(m.strategy),
    ) for m in metrics]
    CSV.write(path, rows)
    return path
end
