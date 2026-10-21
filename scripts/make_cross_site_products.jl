"""
    scripts/make_cross_site_products.jl

Lightweight cross-site *products* runner.

Purpose
-------
For each study site in a TOML config, this script persists ONLY the numerical /
geospatial products needed to compose the 3-column cross-site figure
(`scripts/make_cross_site_panel.jl`):

    <out_dir>/<site-slug>/
        label_clusters.tif      Int32   per-pixel k-medoids cluster id  (nodata 0)
        vegetation_mask.tif     UInt8   selected-vegetation 0/1 mask    (nodata 255)
        kde_surface.tif         Float64 min-max-normalised [0,1] KDE    (nodata NaN)
        products_metadata.json          compact reproducibility record

It DOES NOT write speed maps, waypoints, waypoint CSVs, cluster overlays,
histograms, manifests, or a run-report bundle — that is the heavy
`run_from_config.jl` pipeline's job, which this script leaves untouched.

Responsibility boundary
------------------------
  • This runner persists rasters + metadata + a reference to the source GeoTIFF.
    It NEVER writes the final panel PNGs.
  • The composer (`make_cross_site_panel.jl`) reads the source GeoTIFF + these
    persisted rasters and draws all three visual columns.
  • Label-selection previews are transient: when a site has no configured
    `tree_labels` and stdin is a TTY, the runner renders ONE temporary
    contact-sheet PNG (via `report_cluster_overlays(::LabelSelectionPreview, …)`),
    prompts for the vegetation id(s), persists them to the config, and DELETES
    the preview. Nothing about the preview is retained.

Reproducibility
---------------
Clustering runs exactly ONCE per site (no re-cluster for the k-sweep vs the
final fit — `build_mask_from_image` threads one seeded RNG through both). The
k-medoids++ seeding is made deterministic in `src/clustering.jl` by precomputing
the seeds with the configured RNG and passing them as an explicit `init` vector.
On rerun, if the source digest and clustering settings match the cached
metadata, the cluster-label raster is reloaded and clustering is skipped; only
the mask + KDE are recomputed from the (possibly changed) `tree_labels`.

KDE normalisation
-----------------
`build_density_surface` returns a sum-to-one surface (values are tiny and depend
on grid size, so they are NOT comparable across sites). This runner therefore
persists a **min-max normalised** KDE in [0,1] so the composer can use a single
fixed colour range across every site. The choice is recorded in the metadata as
`kde_normalization = "minmax"`.

Usage
-----
    julia --project=. scripts/make_cross_site_products.jl [CONFIG]

    CONFIG  Path to the products TOML (default: config/cross_site_products.toml).

Config schema (TOML)
--------------------
    out_dir = "../output/cross_site"     # products root (relative to config dir)

    [defaults]                           # optional; per-site keys override these
    seed             = 6213
    nsample          = 1500
    kmedoids_k_range = [2, 6]
    cluster_stride   = 1

    [[site]]
    name     = "Durham, NH"
    geotiff  = "../data/durham_ortho.tif" # authoritative RGB + transform/CRS
    # image  = "../data/durham.jpg"       # fallback only (image-space, no CRS)
    tree_labels = [1]                     # omit/[] → prompt (TTY) or error (non-TTY)
    # seed / nsample / kmedoids_k_range / cluster_stride may be set per site

Paths inside the TOML resolve relative to the config file's directory.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using Colors: RGB
using Statistics: mean
using TOML
using JSON
using SHA
using Printf
using Dates
using FileIO, ImageIO

include(joinpath(@__DIR__, "tree_label_selection.jl"))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

struct SiteDefaults
    seed           :: Int
    nsample        :: Int
    k_range        :: Tuple{Int,Int}
    cluster_stride :: Int
end

_get(tbl, key, default) = haskey(tbl, key) ? tbl[key] : default

function load_defaults(raw)
    d = get(raw, "defaults", Dict{String,Any}())
    kr = _get(d, "kmedoids_k_range", [2, 6])
    SiteDefaults(
        Int(_get(d, "seed", 6213)),
        Int(_get(d, "nsample", 1500)),
        (Int(kr[1]), Int(kr[2])),
        max(1, Int(_get(d, "cluster_stride", 1))),
    )
end

slug(name::AbstractString) = tree_labels_slug(name)

# ---------------------------------------------------------------------------
# Digests
# ---------------------------------------------------------------------------

"""File content digest (SHA-256 hex) — the reproducibility key for the source."""
function file_digest(path::AbstractString)
    open(path, "r") do io
        bytes2hex(sha256(io))
    end
end

"""Digest of the clustering settings that would change the label raster."""
settings_digest(seed, nsample, k_range, stride) =
    bytes2hex(sha256("seed=$seed;nsample=$nsample;ks=$(k_range[1]):$(k_range[2]);stride=$stride"))

# ---------------------------------------------------------------------------
# Product paths
# ---------------------------------------------------------------------------

struct ProductPaths
    dir      :: String
    label    :: String
    mask     :: String
    kde      :: String
    metadata :: String
end

function product_paths(out_dir::AbstractString, site_name::AbstractString)
    dir = joinpath(out_dir, slug(site_name))
    ProductPaths(dir,
        joinpath(dir, "label_clusters.tif"),
        joinpath(dir, "vegetation_mask.tif"),
        joinpath(dir, "kde_surface.tif"),
        joinpath(dir, "products_metadata.json"))
end

# ---------------------------------------------------------------------------
# Geotransform scaling for a stride-decimated grid
# ---------------------------------------------------------------------------

"""
    scaled_geotransform(gt, stride) -> GeoTransform

Cell size scales by `stride` (top-left-anchored decimation keeps the origin), so
the decimated label/mask/KDE rasters stay co-registered with the source ortho.
"""
scaled_geotransform(gt::GeoTransform, stride::Integer) =
    GeoTransform(gt.x_origin, gt.dx * stride, gt.x_rot,
                 gt.y_origin, gt.y_rot, gt.dy * stride)

# Pixel-space geotransform for the image-fallback path (no CRS available).
pixel_geotransform(H_full::Integer, stride::Integer) =
    GeoTransform(0.0, Float64(stride), 0.0, Float64(H_full), 0.0, -Float64(stride))

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

"""
    cache_valid(pp, src_digest, set_digest) -> (ok, metadata_or_nothing)

The cache is reusable when the metadata JSON + label raster exist AND both the
source-content digest and the clustering-settings digest match.
"""
function cache_valid(pp::ProductPaths, src_digest::AbstractString, set_digest::AbstractString)
    (isfile(pp.metadata) && isfile(pp.label)) || return (false, nothing)
    meta = try
        JSON.parsefile(pp.metadata)
    catch
        return (false, nothing)
    end
    ok = get(meta, "source_digest", "") == src_digest &&
         get(meta, "settings_digest", "") == set_digest
    return (ok, ok ? meta : nothing)
end

# ---------------------------------------------------------------------------
# Tree-label resolution (config → cache → TTY prompt → error)
# ---------------------------------------------------------------------------

"""
    resolve_labels(site, k, img, labels_full, cached_labels; name, config_path,
                   suggested, preview_dir) -> (labels, source::String)

Semantics (Requirement 4):
  • explicit nonempty configured `tree_labels`  → validated, used (no prompt);
    out-of-range fails BEFORE the KDE stage.
  • else a valid cached selection (matching k)   → reused (resume w/o prompt).
  • else TTY                                     → render ONE temp contact sheet,
    prompt, persist to config, DELETE preview.
  • else (non-TTY, nothing configured)           → clear error.
"""
function resolve_labels(site, k::Int, img, labels_full::AbstractVector{<:Integer},
                        cached_labels;
                        name::AbstractString, config_path::AbstractString,
                        suggested::Int, preview_dir::AbstractString)
    cfg_labels = site_configured_tree_labels(site)
    if cfg_labels !== nothing
        err = validate_configured_tree_labels(cfg_labels, k)
        err === nothing || error(
            "Configured tree_labels for site \"$name\" are invalid: $err. " *
            "Valid cluster ids are 1:$k — fix the `tree_labels` entry in the config.")
        return (sort(cfg_labels), "config")
    end

    if cached_labels !== nothing &&
       validate_configured_tree_labels(cached_labels, k) === nothing
        @info "Reusing cached tree_labels (resume; not re-prompting)" site=name labels=cached_labels
        return (sort(cached_labels), "cache")
    end

    if !stdin_is_tty()
        error(
            "No `tree_labels` configured for site \"$name\" and stdin is not a TTY.\n" *
            "Either add `tree_labels = [..]` (cluster ids in 1:$k) to the site's " *
            "[[site]] block, or run this script from an interactive terminal so the " *
            "vegetation cluster(s) can be selected after reviewing the preview.")
    end

    # Interactive: ONE transient contact sheet, prompt, persist, delete.
    greenness = nothing
    try
        _, a_chan, _ = rgb_to_lab(img)
        label_img = reshape(labels_full, size(img)...)
        greenness = [ -mean(@view a_chan[label_img .== cid]) for cid in 1:k ]
    catch
        greenness = nothing
    end
    spec = LabelSelectionPreview(; greenness = greenness)
    preview_png = joinpath(preview_dir, "_tmp_label_preview_$(slug(name)).png")
    report_cluster_overlays(spec, img, labels_full, k, preview_png)
    labels = try
        prompt_tree_labels(k, [preview_png]; name = name, suggested = suggested)
    finally
        isfile(preview_png) && rm(preview_png; force = true)   # never retained
    end
    ok, msg = persist_tree_labels(config_path, name, labels)
    ok ? (@info "Persisted tree_labels to config" site=name labels=labels) :
         (@warn "Could not persist tree_labels; used for THIS run only" site=name reason=msg)
    return (sort(labels), "interactive")
end

# ---------------------------------------------------------------------------
# Per-site processing
# ---------------------------------------------------------------------------

function process_site(site, defaults::SiteDefaults, out_dir::AbstractString,
                      config_path::AbstractString, base_dir::AbstractString)
    name = String(get(site, "name", "(unnamed site)"))
    println("\n=== Site: $name ===")

    kind, input_path = select_site_input(site, base_dir)
    kind === :none && (@warn "No readable geotiff/image for site; skipping." site=name; return nothing)

    seed    = Int(_get(site, "seed", defaults.seed))
    nsample = Int(_get(site, "nsample", defaults.nsample))
    kr      = _get(site, "kmedoids_k_range", [defaults.k_range[1], defaults.k_range[2]])
    k_range = (Int(kr[1]), Int(kr[2]))
    stride  = max(1, Int(_get(site, "cluster_stride", defaults.cluster_stride)))

    pp = product_paths(out_dir, name)
    mkpath(pp.dir)

    src_digest = file_digest(input_path)
    set_digest = settings_digest(seed, nsample, k_range, stride)

    # --- Load source RGB (+ transform/CRS when GeoTIFF) --------------------
    local img_full, crs, product_gt, H_full, W_full
    if kind === :geotiff
        rs        = load_rgb_geotiff(input_path)
        img_full  = convert(Matrix{RGB{Float32}}, rs.Z)
        H_full, W_full = size(img_full)
        crs       = rs.crs
        product_gt = scaled_geotransform(rs.gt, stride)
        println("  input = $input_path  [GeoTIFF, authoritative RGB + transform/CRS]")
        println("  dims  = $(W_full)×$(H_full) px   crs = ", isempty(crs) ? "<unknown>" : crs)
    else
        img_raw   = FileIO.load(input_path)
        img_full  = convert(Matrix{RGB{Float32}}, img_raw)
        H_full, W_full = size(img_full)
        crs       = ""
        product_gt = pixel_geotransform(H_full, stride)
        println("  input = $input_path  [image fallback — no CRS; pixel-space transform]")
        println("  dims  = $(W_full)×$(H_full) px")
    end

    # Decimated clustering grid
    img = stride > 1 ? Matrix(@view img_full[1:stride:end, 1:stride:end]) : img_full
    H, W = size(img)
    stride > 1 && println("  cluster_stride = $stride → grid $(W)×$(H) px")

    # --- Cluster once OR reload from cache --------------------------------
    ok, meta = cache_valid(pp, src_digest, set_digest)
    local labels_full::Vector{Int}, k::Int, cached_labels
    cached_labels = nothing
    if ok
        Zlab, _, _ = read_band(pp.label)
        labels_full = Int.(vec(Zlab))                       # column-major → feature order
        k = maximum(labels_full)
        cached_labels = let m = get(meta, "tree_labels", nothing)
            m === nothing ? nothing : Int.(m)
        end
        println("  [cache] reusing cluster labels (k=$k); clustering skipped")
    else
        println("  [a] k-medoids CIELAB clustering (ks=$(k_range[1]):$(k_range[2])) …")
        _, info = build_mask_from_image(img;
            tree_labels = Int[], seed = seed, nsample = nsample,
            ks = k_range[1]:k_range[2], k_strategy = :vote,
            xmin = 0.0, xmax = Float64(W), ymin = 0.0, ymax = Float64(H))
        labels_full = Int.(info.labels_full)
        k = info.k
        println("      chosen k = $k")
        # Persist the label raster now so a crash mid-selection still caches it.
        write_single_band_geotiff(pp.label, reshape(labels_full, H, W),
                                  product_gt, crs; dtype = Int32, nodata = 0)
    end

    # Greenness hint (more negative CIELAB a* ⇒ greener)
    _, a_chan, _ = rgb_to_lab(img)
    label_img = reshape(labels_full, H, W)
    greenness = [ -mean(@view a_chan[label_img .== cid]) for cid in 1:k ]
    suggested = argmax(greenness)

    # --- Resolve vegetation labels ---------------------------------------
    tree_labels, label_source = resolve_labels(site, k, img, labels_full, cached_labels;
        name = name, config_path = config_path, suggested = suggested, preview_dir = pp.dir)
    println("      tree_labels = $tree_labels  [source=$label_source]")

    # --- Vegetation mask (UInt8 0/1) -------------------------------------
    veg = labels_to_mask(labels_full, H, W; tree_labels = tree_labels)
    write_single_band_geotiff(pp.mask, UInt8.(veg), product_gt, crs;
                              dtype = UInt8, nodata = 255)

    # --- Epanechnikov KDE, min-max normalised to [0,1] -------------------
    println("  [b] Epanechnikov KDE planning surface …")
    xs, ys   = pixel_axes(H, W; xmin = 0.0, xmax = Float64(W), ymin = 0.0, ymax = Float64(H))
    mask_grid = RasterGrid(Float64.(veg), xs, ys)
    kde_cfg   = PipelineConfig(; kmed_k = k, seed = seed, tree_labels = tree_labels,
                                kde_bandwidth = :auto, kde_kernel = :epanechnikov,
                                kde_scaling = :none)
    dens_grid, _ = build_density_surface(mask_grid, kde_cfg)
    Zk   = dens_grid.Z
    lo, hi = minimum(Zk), maximum(Zk)
    Zn   = hi > lo ? (Zk .- lo) ./ (hi - lo) : zeros(size(Zk))
    write_single_band_geotiff(pp.kde, Zn, product_gt, crs;
                              dtype = Float64, nodata = NaN)

    # --- Compact reproducibility metadata --------------------------------
    metadata = Dict{String,Any}(
        "site_name"        => name,
        "source_geotiff"   => input_path,
        "source_kind"      => String(kind),
        "source_digest"    => src_digest,
        "settings_digest"  => set_digest,
        "crs"              => crs,
        "geotransform"     => [product_gt.x_origin, product_gt.dx, product_gt.x_rot,
                               product_gt.y_origin, product_gt.y_rot, product_gt.dy],
        "grid_HW"          => [H, W],
        "cluster_stride"   => stride,
        "k"                => k,
        "seed"             => seed,
        "nsample"          => nsample,
        "kmedoids_k_range" => [k_range[1], k_range[2]],
        "medoid_greenness" => greenness,
        "suggested_cluster"=> suggested,
        "tree_labels"      => tree_labels,
        "tree_labels_source" => label_source,
        "kde_normalization"  => "minmax",
        "kde_value_range_raw" => [lo, hi],
        "products" => Dict(
            "label_clusters"  => basename(pp.label),
            "vegetation_mask" => basename(pp.mask),
            "kde_surface"     => basename(pp.kde),
        ),
        "julia_version"    => string(VERSION),
        "generated_at"     => string(Dates.now()),
    )
    open(pp.metadata, "w") do io
        JSON.print(io, metadata, 2)
    end

    println("  products:")
    for p in (pp.label, pp.mask, pp.kde, pp.metadata)
        @printf("    %-22s %d bytes\n", basename(p), filesize(p))
    end
    return pp
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function main()
    config_path = abspath(get(ARGS, 1,
        joinpath(@__DIR__, "..", "config", "cross_site_products.toml")))
    isfile(config_path) || error("Config not found: $config_path")
    base_dir = dirname(config_path)
    raw = TOML.parsefile(config_path)

    out_dir = abspath(joinpath(base_dir, String(_get(raw, "out_dir", "../output/cross_site"))))
    mkpath(out_dir)
    defaults = load_defaults(raw)

    sites = get(raw, "site", Any[])
    isempty(sites) && error("Config $config_path has no [[site]] entries.")

    println("Cross-site products runner")
    println("  config  = $config_path")
    println("  out_dir = $out_dir")
    println("  sites   = $(length(sites))")

    for site in sites
        process_site(site, defaults, out_dir, config_path, base_dir)
    end
    println("\nDone. Compose the figure with:")
    println("  julia --project=. scripts/make_cross_site_panel.jl <panel-config> <out.png>")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
