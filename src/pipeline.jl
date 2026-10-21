# pipeline.jl — End-to-end pipeline API
#
# Connects the full CanopyDensity → FlightPlanning pipeline:
#   orthomosaic / image array
#     → RGB-to-LAB → feature stacking → standardisation → optional PCA
#     → k-medoids clustering (auto-k or fixed k)
#     → binary canopy mask → optional morphological cleanup
#     → KDE density surface (RasterGrid, [0,1])
#     → speed map → lawnmower path → curvature-spaced waypoints
#
# All `using` statements are centralised in KDEFlightPlanning.jl.
#
# Fully implemented (no external packages beyond deps):
#   axes_from_geotransform, build_density_surface, run_pipeline
#   build_mask_from_image, build_mask_autok_from_image (image-array API)
#
# Require ArchGDAL (not in deps; user must load it):
#   load_rgb_georef
#
# Require Clustering.jl + Distances.jl (both in deps):
#   kmedoids_fit, build_mask_autok

# ---------------------------------------------------------------------------
# Geotransform → grid axes
# ---------------------------------------------------------------------------

"""
    axes_from_geotransform(gt, W::Integer, H::Integer) -> (xs, ys)

Convert a GDAL 6-element geotransform vector to world-coordinate axis vectors.

GDAL convention:
    gt[1] = x_origin (upper-left corner x)
    gt[2] = x_pixel_size
    gt[3] = x_rotation (0 for north-up)
    gt[4] = y_origin (upper-left corner y)
    gt[5] = y_rotation (0 for north-up)
    gt[6] = y_pixel_size (negative for north-up, e.g. -1.0)

Returns ascending `(xs, ys)` vectors (south-up flipped if needed to ensure
the ys axis is ascending, matching `RasterGrid` convention).
"""
function axes_from_geotransform(gt::AbstractVector, W::Integer, H::Integer)
    length(gt) >= 6 || throw(ArgumentError("gt must have ≥ 6 elements (GDAL geotransform)"))
    x_origin = Float64(gt[1])
    x_res    = Float64(gt[2])
    y_origin = Float64(gt[4])
    y_res    = Float64(gt[6])

    xs = [x_origin + (i - 0.5) * x_res for i in 1:W]
    ys = [y_origin + (j - 0.5) * y_res for j in 1:H]

    issorted(xs) || reverse!(xs)
    issorted(ys) || reverse!(ys)
    return xs, ys
end

"""
    geotransform_resolution(gt) -> (xres, yres)

Extract the ground sampling distance (metres per pixel) along each pixel axis
from a 6-term GDAL affine geotransform, robust to rotation/skew.

The GDAL geotransform maps pixel `(col, row)` (0-based) to world `(X, Y)`:

    X = gt[1] + col·gt[2] + row·gt[3]
    Y = gt[4] + col·gt[5] + row·gt[6]

(Here `gt` is the package's 1-based `GeoTransform`/vector: `gt[1]=x_origin`,
`gt[2]=dx`, `gt[3]=x_rot`, `gt[4]=y_origin`, `gt[5]=y_rot`, `gt[6]=dy`, which
is GDAL 0-based `GT[0..5]` shifted by one.)

Stepping one column advances world position by `(gt[2], gt[5])`; stepping one
row advances by `(gt[3], gt[6])`. The physical pixel size along each axis is
therefore the Euclidean length of these column/row axis vectors:

    xres = hypot(gt[2], gt[5])   # length of the per-column axis vector
    yres = hypot(gt[3], gt[6])   # length of the per-row axis vector

Using `hypot` (rather than `abs(gt[2])` / `abs(gt[6])`) means rotated or
skewed geotransforms return the true on-ground spacing, not just the axis-
aligned component. For a north-up transform (`gt[3]=gt[5]=0`) this reduces to
`(abs(gt[2]), abs(gt[6]))`.

Accepts a `GeoTransform`, or any indexable length-≥6 vector.
"""
function geotransform_resolution(gt)
    length(gt) >= 6 || throw(ArgumentError("gt must have ≥ 6 elements (GDAL geotransform)"))
    xres = hypot(Float64(gt[2]), Float64(gt[5]))
    yres = hypot(Float64(gt[3]), Float64(gt[6]))
    return xres, yres
end

# ---------------------------------------------------------------------------
# Uniform pixel-space axes (for image-array workflows)
# ---------------------------------------------------------------------------

"""
    pixel_axes(H::Integer, W::Integer; xmin=0.0, xmax=Float64(W),
               ymin=0.0, ymax=Float64(H)) -> (xs, ys)

Generate uniform pixel-space axis vectors for an image of size H×W.
Useful when no GeoTIFF geotransform is available (array-based workflow).
"""
function pixel_axes(H::Integer, W::Integer;
                     xmin::Real=0.0, xmax::Real=Float64(W),
                     ymin::Real=0.0, ymax::Real=Float64(H))
    xs = collect(range(Float64(xmin), Float64(xmax); length=W))
    ys = collect(range(Float64(ymin), Float64(ymax); length=H))
    return xs, ys
end

# ---------------------------------------------------------------------------
# GeoTIFF loader (requires ArchGDAL in caller's session)
# ---------------------------------------------------------------------------

"""
    load_rgb_georef(path::AbstractString) -> (img, gt, W, H)

Load a 3-band GeoTIFF as a normalised `Matrix{RGB{Float32}}` in [0,1].
Returns `(img, gt::Vector{Float64}, W::Int, H::Int)`.

**Requires ArchGDAL.jl** to be installed and loaded in the caller's session:
    using Pkg; Pkg.add("ArchGDAL")
    using ArchGDAL

This function throws a helpful error when ArchGDAL is absent.
For array-based testing without GeoTIFFs, use `rgb_from_array` or the
`synthetic_*` grid constructors instead.
"""
function load_rgb_georef(path::AbstractString)
    if !isdefined(Main, :ArchGDAL)
        error("""
load_rgb_georef requires ArchGDAL.jl.
Install and load it first:
    using Pkg; Pkg.add("ArchGDAL")
    using ArchGDAL

For testing without real GeoTIFFs, use `synthetic_gaussian_grid` or
`synthetic_mask_grid` to generate synthetic data, or `rgb_from_array`
to wrap a raw numeric array.
""")
    end
    return Main.ArchGDAL.read(path) do ds
        r    = Main.ArchGDAL.read(ds, 1)
        g    = Main.ArchGDAL.read(ds, 2)
        b    = Main.ArchGDAL.read(ds, 3)
        rmax = Float32(typemax(eltype(r)))
        gmax = Float32(typemax(eltype(g)))
        bmax = Float32(typemax(eltype(b)))
        img  = RGB{Float32}.(Float32.(r)./rmax, Float32.(g)./gmax, Float32.(b)./bmax)
        (img, Main.ArchGDAL.getgeotransform(ds),
         Main.ArchGDAL.width(ds), Main.ArchGDAL.height(ds))
    end
end

# ---------------------------------------------------------------------------
# Image-array → mask  (full pipeline, no GeoTIFF needed)
# ---------------------------------------------------------------------------

"""
    build_mask_from_image(img_or_arr;
        k=2, tree_labels=[1], seed=6213,
        nsample=2000, metric=SqEuclidean(),
        ks=nothing, k_strategy=:silhouette,
        variance_ratio=0.95, use_pca=false,
        xmin=0.0, xmax=nothing, ymin=0.0, ymax=nothing,
        do_cleanup=false)
    -> (mask_grid::RasterGrid, info::NamedTuple)

End-to-end image-array to canopy mask pipeline (no GeoTIFF required):

    image/array → RGB{Float32} → CIELAB → feature stacking
    → standardisation → (optional PCA) → k-medoids → mask → RasterGrid

Arguments
---------
- `img_or_arr`:  Either a `Matrix{<:Colorant}` (already RGB) or a 3-D
                 numeric array accepted by `rgb_from_array` (shapes (3,H,W)
                 or (H,W,3)).
- `k`:           Number of k-medoids clusters (used when `ks=nothing`).
- `tree_labels`: Cluster label(s) that correspond to vegetation/canopy.
- `seed`:        RNG seed for reproducible sampling.
- `nsample`:     Number of pixels to sample for k-medoids (speeds up large images).
- `metric`:      A `Distances` metric for the distance matrix.
- `ks`:          Range of k values for auto-k sweep (e.g. `2:8`). When
                 provided, overrides `k` and uses `sweep_k_quality` + `choose_k`.
- `k_strategy`:  Strategy for auto-k selection (`:silhouette`, `:dunn`,
                 `:db`, `:cal`, `:mode`).
- `variance_ratio`, `use_pca`: PCA settings.
- `xmin`, `xmax`, `ymin`, `ymax`: World-coordinate extents for the output
                 `RasterGrid`. Defaults to pixel-space (0..W, 0..H).
- `do_cleanup`:  Apply morphological cleanup (requires ImageMorphology.jl).

Returns
-------
- `mask_grid`: `RasterGrid` with Z ∈ {0.0, 1.0} (vegetation=1)
- `info`:      NamedTuple with `(k, labels_full, H, W, xs, ys, mu, sigma,
                 metrics, medoids)`
"""
function build_mask_from_image(img_or_arr;
                                 k::Int=2,
                                 tree_labels::AbstractVector{<:Integer}=[1],
                                 seed::Int=6213,
                                 nsample::Int=2000,
                                 metric=Distances.SqEuclidean(),
                                 ks::Union{Nothing,AbstractRange}=nothing,
                                 k_strategy::Symbol=:vote,  # paper default
                                 variance_ratio::Union{Nothing,Float64}=0.95,
                                 use_pca::Bool=false,
                                 xmin::Real=0.0, xmax::Union{Nothing,Real}=nothing,
                                 ymin::Real=0.0, ymax::Union{Nothing,Real}=nothing,
                                 do_cleanup::Bool=false)
    rng = Random.MersenneTwister(seed)

    # 1) Normalise to Matrix{RGB{Float32}}
    img = if img_or_arr isa AbstractMatrix{<:Colorant}
        # Already a colour matrix; ensure RGB{Float32}
        convert(Matrix{RGB{Float32}}, img_or_arr)
    else
        rgb_from_array(img_or_arr)
    end

    H, W = size(img)

    # 2) CIELAB channels → feature matrix
    L, a, b = rgb_to_lab(img)
    X = Float64.(stack_features(L, a, b))   # H*W × 3, Float64

    # 3) Standardise in-place
    μ, σ = standardize_features!(X)

    # 4) Optional PCA
    Xuse = if use_pca
        pca_model = pca_fit(X; variance_ratio=variance_ratio)
        pca_transform(pca_model, X)
    else
        X
    end

    # 5) Sample + distance matrix
    n  = size(Xuse, 1)
    ns = min(nsample, n)
    idxs = choose_sample_indices(n; nsample=ns, rng=rng)
    D    = sample_distance_matrix(Xuse, idxs; metric=metric)

    # 6) Select k: auto-sweep or fixed
    # Paper default sweep: 2:12. Falls back to fixed k when ks=nothing.
    kstar, metrics = if ks !== nothing
        mets = sweep_k_quality(Xuse, idxs, D; ks=ks, seed=seed, nsample=ns)
        choose_k(mets; strategy=k_strategy), mets
    else
        k, nothing
    end

    # 7) Fit k-medoids, assign all rows
    labels_full, res_sample, kinfo = kmedoids_fit(Xuse;
        k=kstar, idxs_sample=idxs, D=D, metric=metric, rng=rng)

    # 8) Build mask
    mask_bm = labels_to_mask(labels_full, H, W; tree_labels=tree_labels)

    # 9) Optional morphological cleanup
    if do_cleanup
        remove_small_components!(mask_bm;
            min_pixels=max(50, round(Int, 0.0005*H*W)), connectivity=8)
        fill_small_holes!(mask_bm; max_hole_pixels=50, connectivity=8)
        binary_open_close!(mask_bm; open_radius=1, close_radius=1)
    end

    # 10) Build world-coordinate axes
    _xmax = xmax !== nothing ? Float64(xmax) : Float64(W)
    _ymax = ymax !== nothing ? Float64(ymax) : Float64(H)
    xs, ys = pixel_axes(H, W; xmin=xmin, xmax=_xmax, ymin=ymin, ymax=_ymax)

    mask_grid = RasterGrid(Float64.(mask_bm), xs, ys)

    info = (k=kstar, labels_full=labels_full,
            H=H, W=W, xs=xs, ys=ys,
            mu=μ, sigma=σ,
            metrics=metrics,
            medoids=kinfo.medoid_rows)
    return mask_grid, info
end

# ---------------------------------------------------------------------------
# build_mask_from_image_strided — performance shim for very large rasters
# ---------------------------------------------------------------------------

"""
    build_mask_from_image_strided(img_or_arr; stride::Int=1, kwargs...)

Performance-oriented wrapper around `build_mask_from_image`:

1. Stride-decimate the input image to size `(H/stride, W/stride)`.
2. Fit clustering and compute the mask on the decimated array.
3. Nearest-neighbour upsample the labels back to full resolution.

This is essential for full-resolution orthomosaics (e.g. 9279×11421) where
running the full feature → cluster pipeline on ~100 million pixels would
take many minutes. With `stride = 8` the effective image is ~1.6 MP, and
the produced mask covers every pixel of the original raster.

`kwargs...` are forwarded to `build_mask_from_image`. The returned info
NamedTuple gains a `:stride` field and `:H_full`, `:W_full`, `:labels_full`
remains at full resolution. The `labels_full` length always equals
`H_full * W_full`. The chosen k is unchanged from the decimated fit.

`stride = 1` is the default (no decimation) for backward compatibility.
"""
function build_mask_from_image_strided(img_or_arr;
                                         stride::Int = 1,
                                         xmin::Real = 0.0,
                                         xmax::Union{Nothing, Real} = nothing,
                                         ymin::Real = 0.0,
                                         ymax::Union{Nothing, Real} = nothing,
                                         kwargs...)
    stride < 1 && throw(ArgumentError("stride must be ≥ 1"))

    # Normalise input to a 3-D numeric array (H, W, C) so we can decimate
    # uniformly across both colour-matrix and raw-array inputs.
    img = if img_or_arr isa AbstractMatrix{<:Colorant}
        img_or_arr
    else
        rgb_from_array(img_or_arr)
    end

    H_full, W_full = size(img)

    if stride == 1
        # Forward to the existing pipeline unchanged.
        mask_grid, info = build_mask_from_image(img;
            xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, kwargs...)
        return mask_grid, merge(info, (stride = 1,
                                        H_full = H_full, W_full = W_full,
                                        H_sample = H_full, W_sample = W_full))
    end

    # Stride-decimate
    img_ds = @view img[1:stride:end, 1:stride:end]
    H_s, W_s = size(img_ds)

    mask_ds, info_ds = build_mask_from_image(img_ds;
        xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, kwargs...)

    # labels_ds (length H_s * W_s) → labels_full (length H_full * W_full)
    # by nearest-neighbour upsampling of the (H_s, W_s) label image.
    labels_ds_img = reshape(info_ds.labels_full, H_s, W_s)
    labels_full = Vector{Int}(undef, H_full * W_full)
    @inbounds for j in 1:H_full
        js = min(H_s, fld(j - 1, stride) + 1)
        for i in 1:W_full
            is = min(W_s, fld(i - 1, stride) + 1)
            labels_full[(j - 1) * W_full + i] = labels_ds_img[js, is]
        end
    end

    # Rebuild a full-resolution mask grid using axes at full-resolution scale
    _xmax = xmax !== nothing ? Float64(xmax) : Float64(W_full)
    _ymax = ymax !== nothing ? Float64(ymax) : Float64(H_full)
    xs, ys = pixel_axes(H_full, W_full; xmin = xmin, xmax = _xmax,
                                          ymin = ymin, ymax = _ymax)
    tree_labels = get(Dict(kwargs), :tree_labels, [1])
    mask_bm     = labels_to_mask(labels_full, H_full, W_full;
                                  tree_labels = tree_labels)
    mask_grid   = RasterGrid(Float64.(mask_bm), xs, ys)

    info = merge(info_ds,
                  (labels_full = labels_full,
                   H = H_full, W = W_full,
                   xs = xs, ys = ys,
                   stride = stride,
                   H_full = H_full, W_full = W_full,
                   H_sample = H_s, W_sample = W_s))
    return mask_grid, info
end

# ---------------------------------------------------------------------------
# build_mask_autok — GeoTIFF path (ArchGDAL required at call time)
# ---------------------------------------------------------------------------

"""
    build_mask_autok(rgb_path;
        cfg=PipelineConfig(),
        ks=2:12, nsample=2000,
        metric=Euclidean(), k_strategy=:vote,
        tree_labels=nothing, do_cleanup=true,
        interactive=true, isatty_fn=()->Base.isatty(stdin),
        label_selector=nothing, overlay_outdir="output/cluster_overlays")
    -> (mask_grid::RasterGrid, info::NamedTuple)

Full end-to-end from a GeoTIFF path to a canopy mask RasterGrid, with
automatic k selection via a quality-metric sweep.

Vegetation label resolution (clustering happens exactly once, first)
--------------------------------------------------------------------
There is NO silent `tree_labels` default (a missing selection is never turned
into `[1]` or the greenest cluster). After the single auto-k clustering pass,
`resolve_autok_tree_labels` decides the labels:

- **Explicit, nonempty `tree_labels`** → validated and used verbatim, with no
  prompt (deterministic; safe for CI/batch use).
- **Absent (`nothing`/empty) on an interactive TTY** → cluster overlays are
  rendered under `overlay_outdir` and the user selects the vegetation cluster(s)
  (reprompting until valid) via the canonical `interactive_tree_labels`.
- **Absent when non-interactive / non-TTY** → a descriptive error is raised
  immediately after clustering, telling the caller to pass `tree_labels`.

The confirmed labels are recorded in the returned `info` as `tree_labels` (with
`tree_labels_source ∈ (:explicit, :interactive)`); this function does not write
any config/TOML — persistence is the caller's responsibility.

Keyword seams
-------------
- `interactive`: set `false` to force the non-interactive contract (library
  callers that must never prompt).
- `isatty_fn`: predicate deciding whether stdin is interactive (injectable for
  tests; defaults to `Base.isatty(stdin)`).
- `label_selector`: optional `(img, labels_full, H, W, k) -> ids` callback used
  instead of the built-in prompt (injectable for tests); its result is validated.
- `overlay_outdir`: directory for the per-cluster overlay previews.

Requires:
- ArchGDAL.jl loaded in the caller's session (for `load_rgb_georef`)
- Clustering.jl and Distances.jl (included in deps)

See `build_mask_from_image` for the image-array equivalent (no GeoTIFF needed).
"""
function build_mask_autok(rgb_path::AbstractString;
                           cfg::PipelineConfig=PipelineConfig(),
                           ks::AbstractRange=2:12,   # paper default
                           nsample::Integer=2000,
                           metric=Distances.Euclidean(),
                           k_strategy::Symbol=:vote,  # paper default
                           tree_labels::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                           do_cleanup::Bool=true,
                           interactive::Bool=true,
                           isatty_fn=()->Base.isatty(stdin),
                           label_selector=nothing,
                           overlay_outdir::AbstractString="output/cluster_overlays")

    rng = Random.MersenneTwister(cfg.seed)

    # 1) Load + georef axes (ArchGDAL)
    img, gt, W, H = load_rgb_georef(rgb_path)
    xs, ys        = axes_from_geotransform(gt, W, H)

    # 2) CIELAB → features → standardise
    L, a, b = rgb_to_lab(img)
    X       = Float64.(stack_features(L, a, b))
    μ, σ    = standardize_features!(X)

    # 3) Optional PCA
    Xuse = if !(cfg.pca.variance_ratio === nothing && cfg.pca.maxoutdim === nothing)
        p = pca_fit(X; variance_ratio=cfg.pca.variance_ratio,
                        maxoutdim=cfg.pca.maxoutdim)
        pca_transform(p, X)
    else
        X
    end

    # 4) Sample + distance matrix (reused across all k)
    n  = size(Xuse, 1)
    ns = min(nsample, n)
    idxs = choose_sample_indices(n; nsample=ns, rng=rng)
    D    = sample_distance_matrix(Xuse, idxs; metric=metric)

    # 5) k-sweep + quality metrics (paper default ks=2:12, strategy=:vote)
    metrics = sweep_k_quality(Xuse, idxs, D; ks=ks, seed=cfg.seed, nsample=ns)
    kstar   = choose_k(metrics; strategy=k_strategy)

    println("k-sweep metrics:")
    for m in metrics
        dbs  = isnothing(m.db)  ? "—" : @sprintf("%.3f", m.db)
        cals = isnothing(m.cal) ? "—" : @sprintf("%.3f", m.cal)
        @printf("  k=%d  sil=%.3f  dunn=%.3f  db=%s  cal=%s\n",
                m.k, m.silhouette, m.dunn, dbs, cals)
    end
    println("→ chosen k = $kstar (strategy = :$k_strategy)")

    # 6) Fit k-medoids + assign all rows
    labels_full, res_sample, kinfo = kmedoids_fit(Xuse;
        k=kstar, idxs_sample=idxs, D=D, metric=metric, rng=rng)

    # 7) Tree labels — resolve AFTER the single clustering pass. No silent
    #    default: explicit labels are used verbatim; otherwise prompt on a TTY
    #    or error clearly (see resolve_autok_tree_labels).
    label_source = (tree_labels !== nothing && !isempty(tree_labels)) ?
                   :explicit : :interactive
    tl = resolve_autok_tree_labels(tree_labels, img, labels_full, kstar, H, W;
             interactive    = interactive,
             is_tty         = isatty_fn(),
             label_selector = label_selector,
             overlay_outdir = overlay_outdir)

    # 8) Build mask
    mask_bm = labels_to_mask(labels_full, H, W; tree_labels=tl)

    # 9) Optional cleanup
    if do_cleanup
        remove_small_components!(mask_bm;
            min_pixels=max(50, round(Int, 0.0005*H*W)), connectivity=8)
        fill_small_holes!(mask_bm; max_hole_pixels=50, connectivity=8)
        binary_open_close!(mask_bm; open_radius=1, close_radius=1)
    end

    mask_grid = RasterGrid(Float64.(mask_bm), xs, ys)
    info = (img=img, Xuse=Xuse,
            labels_full=labels_full,
            mask=mask_bm,
            k=kstar, metrics=metrics,
            tree_labels=tl,
            tree_labels_source=label_source,
            sample_indices=idxs,
            medoids_sample=res_sample.medoids,
            H=H, W=W, xs=xs, ys=ys, gt=gt,
            mu=μ, sigma=σ)
    return mask_grid, info
end

# ---------------------------------------------------------------------------
# Density surface builder (fully implemented)
# ---------------------------------------------------------------------------

"""
    build_density_surface(mask_grid::RasterGrid, cfg::PipelineConfig;
                           save_path=nothing, cutoff=0.0) -> (dens_grid, info)

Convert a binary canopy mask `RasterGrid` into a normalised KDE density
surface.

Arguments
---------
- `mask_grid`:  `RasterGrid` with Z values 0/1 (or floating point 0–1)
- `cfg`:        `PipelineConfig` controlling kernel type, bandwidth mode, etc.
- `save_path`:  If non-nothing, log a message (GeoTIFF writing requires ArchGDAL)
- `cutoff`:     Density values below this threshold are zeroed (default 0.0)

Returns
-------
- `dens_grid`:  `RasterGrid` with Z normalised to [0, 1]
- `info`:       NamedTuple with `(dx, dy, A, σx, σy, h)`

Notes
-----
- Uses `kde_from_mask` internally with pure-Julia FFT-based convolution (FFTW.jl).
- Both `:gaussian` and `:epanechnikov` kernels are supported.
- Ported and extended from CanopyDensity/orchestrate.jl `build_density_surface`.
"""
function build_density_surface(mask_grid::RasterGrid,
                                cfg::PipelineConfig;
                                save_path::Union{Nothing,String}=nothing,
                                cutoff::Real=0.0)
    xs, ys = mask_grid.xs, mask_grid.ys
    dx = abs(xs[2] - xs[1]); dy = abs(ys[2] - ys[1])

    kernel = cfg.kde_kernel
    bwmode = cfg.kde_bandwidth isa Symbol ? cfg.kde_bandwidth : :manual
    norm   = :sum1   # always sum-1 for speed mapping

    dens_grid, info_d = if kernel === :gaussian
        kde_from_mask(mask_grid.Z, xs, ys;
                       kernel=:gaussian,
                       bandwidth=(cfg.kde_bandwidth isa Symbol ?
                                  cfg.kde_bandwidth : :manual),
                       σx=(cfg.kde_bandwidth isa Real ? Float64(cfg.kde_bandwidth) : NaN),
                       σy=(cfg.kde_bandwidth isa Real ? Float64(cfg.kde_bandwidth) : NaN),
                       anisotropic=false, normalize=norm)
    elseif kernel === :epanechnikov
        kde_from_mask(mask_grid.Z, xs, ys;
                       kernel=:epanechnikov,
                       bandwidth=:auto_indices,
                       normalize=norm)
    else
        throw(ArgumentError("cfg.kde_kernel must be :gaussian or :epanechnikov, got $kernel"))
    end

    if cutoff > 0
        dens_grid.Z .= dens_grid.Z .* (dens_grid.Z .>= cutoff)
    end

    if !isnothing(save_path)
        @info "Skipping GeoTIFF save (ArchGDAL not a hard dep). Implement save_rastergrid_geotiff() separately." save_path
    end

    info = (dx=dx, dy=dy, info_d...)
    return dens_grid, info
end

# ---------------------------------------------------------------------------
# Full pipeline convenience
# ---------------------------------------------------------------------------

"""
    run_pipeline(mask_grid::RasterGrid, cfg::PipelineConfig,
                  flight_cfg::FlightConfig; kwargs...) -> (dens_grid, wps, info)

Full downstream pipeline from a pre-computed mask grid:
1. KDE → density surface
2. Lawnmower path from grid extents
3. Waypoint generation with the given `FlightConfig`

Returns:
- `dens_grid`:  Normalised KDE density `RasterGrid`
- `wps`:        `Vector{Waypoint}`
- `info`:       KDE diagnostics NamedTuple

`kwargs` are forwarded to `generate_waypoints`.
"""
function run_pipeline(mask_grid::RasterGrid,
                       cfg::PipelineConfig,
                       flight_cfg::FlightConfig;
                       kwargs...)
    dens_grid, info = build_density_surface(mask_grid, cfg)
    wps = plan_mission(dens_grid, flight_cfg; kwargs...)
    return dens_grid, wps, info
end

"""
    run_full_pipeline(img_or_arr_or_path;
        kde_cfg=PipelineConfig(),
        flight_cfg=nothing,
        k=2, tree_labels=[1], seed=6213,
        nsample=2000, ks=nothing, k_strategy=:silhouette,
        use_pca=false, do_cleanup=false,
        kwargs...)
    -> (mask_grid, dens_grid, wps, info)

Complete pipeline from image data to waypoints in one call:

    image/array/path → mask_grid → density surface → waypoints

Arguments
---------
- `img_or_arr_or_path`: Accepts:
    - A `Matrix{<:Colorant}` (RGB image)
    - A 3-D numeric array (shapes (3,H,W) or (H,W,3))
    - A String GeoTIFF path (requires ArchGDAL in session)
- `kde_cfg`:    `PipelineConfig` for the KDE step.
- `flight_cfg`: `FlightConfig` for waypoint generation (pass `nothing` to
                skip waypoints and return `wps=[]`).
- `tree_labels`: vegetation cluster label(s). For a GeoTIFF path input these are
                forwarded to `build_mask_autok` as explicit labels (default
                `[1]`), so this convenience API stays deterministic and never
                prompts; pass `interactive=false` to enforce that contract.
- `interactive`: forwarded to `build_mask_autok` for path inputs.
- Other kwargs: forwarded to `build_mask_from_image`.

Returns `(mask_grid, dens_grid, wps, info)`.
"""
function run_full_pipeline(img_or_arr_or_path;
                             kde_cfg::PipelineConfig=PipelineConfig(),
                             flight_cfg::Union{Nothing,FlightConfig}=nothing,
                             k::Int=2,
                             tree_labels::AbstractVector{<:Integer}=[1],
                             seed::Int=6213,
                             nsample::Int=2000,
                             ks::Union{Nothing,AbstractRange}=nothing,
                             k_strategy::Symbol=:vote,  # paper default
                             use_pca::Bool=false,
                             do_cleanup::Bool=false,
                             interactive::Bool=true,
                             kwargs...)
    # 1) Mask (paper default ks=2:12). For a GeoTIFF path, forward the explicit
    #    `tree_labels` (default `[1]`) so this convenience API stays deterministic
    #    and never silently prompts; pass `interactive=false` / explicit labels
    #    to enforce the non-interactive contract.
    mask_grid, mask_info = if img_or_arr_or_path isa AbstractString
        build_mask_autok(img_or_arr_or_path;
            cfg=kde_cfg, ks=(ks !== nothing ? ks : 2:12),
            nsample=nsample, k_strategy=k_strategy,
            tree_labels=tree_labels, interactive=interactive,
            do_cleanup=do_cleanup)
    else
        build_mask_from_image(img_or_arr_or_path;
            k=k, tree_labels=tree_labels, seed=seed,
            nsample=nsample, ks=ks, k_strategy=k_strategy,
            use_pca=use_pca, do_cleanup=do_cleanup)
    end

    # 2) KDE
    dens_grid, kde_info = build_density_surface(mask_grid, kde_cfg)

    # 3) Waypoints (optional)
    wps = if flight_cfg !== nothing
        plan_mission(dens_grid, flight_cfg; kwargs...)
    else
        Waypoint[]
    end

    info = merge(mask_info, (kde_info=kde_info,))
    return mask_grid, dens_grid, wps, info
end
