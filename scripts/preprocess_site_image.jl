"""
    scripts/preprocess_site_image.jl

Run the KDEFlightPlanning planning pipeline on a single-image site (e.g. a
non-georeferenced JPEG screenshot) and emit the artefacts the cross-site
panel composer consumes:

    output/<slug>/cluster/cluster_overlay_k*.png   ← column (a) candidates
    output/<slug>/cluster/tree_label_decision.md   ← which cluster is vegetation?
    output/<slug>/kde/kde_density_heatmap.png      ← column (b)
    output/<slug>/kde/kde_density_grid.csv         ← column (b) (numeric grid)
    output/<slug>/waypoints/<slug>_waypoints_xy_speed.csv  ← column (c)
    output/<slug>/site_provenance.json             ← metric vs image-space record

This script REUSES the package algorithms (`build_mask_from_image`,
`build_density_surface`, `CurvatureGuidedSpeed`, `plan_mission`) rather than
re-implementing them. The only site-local code is the cluster-overlay
rendering (a visualisation, not an algorithm) and the metric-scale bookkeeping.

──────────────────────────────────────────────────────────────────────────
Image-space vs. georeferenced metric processing  (READ THIS)
──────────────────────────────────────────────────────────────────────────
A JPEG screenshot carries **no CRS and no ground sampling distance (GSD)**.
The clustering (a) and KDE (b) stages are scale-free — they operate on colour
and relative geometry — so they are always valid in *normalised image
coordinates*. The waypoint plan (c), however, only means something in metres:
"40 m flight-line spacing" and "2–8 m/s" require a real pixel→metre scale.

This script therefore runs in one of two explicit modes:

  • METRIC       — a `meters_per_pixel` is available (explicit, from a GeoTIFF
                   geotransform, or assumed from the Durham native ortho GSD via
                   `assume_durham_native_gsd`). Flight-line/waypoint spacing is
                   interpreted in METRES and column (c) is a genuine metric
                   plan (still pending vegetation-label review).

  • IMAGE-SPACE  — no scale is available. Columns (a)/(b) are produced in
                   normalised image coordinates. Column (c) is only produced
                   with `--preview` and is written/labelled **PROVISIONAL —
                   image-space (pixels, NOT metres)**. It must not be read as a
                   40 m-spaced, publication-ready flight plan.

Metric waypoint generation is REFUSED without a scale unless `--preview` is
passed, in which case an image-space provisional plan is emitted instead.

──────────────────────────────────────────────────────────────────────────
Scale extraction from a source geotransform
──────────────────────────────────────────────────────────────────────────
Per-axis metres/px is resolved (first match wins):

  1. `meters_per_pixel` set on the site  → used verbatim (isotropic).
  2. `geotiff` path set on the site       → read its GDAL geotransform via the
     package's `read_band` and take `geotransform_resolution` (hypot of the
     affine column/row axis vectors, so rotation/skew are handled — not merely
     abs(dx)/abs(dy)), then scale each axis by its resample factor (below).
  3. `assume_durham_native_gsd = true`    → (opt-in, default false) use the
     bundled Durham validation ortho geotransform `GT_NATIVE` (≈0.02837 m/px)
     as the source GSD × per-axis factor.
  4. otherwise                            → IMAGE-SPACE.

**Anisotropic resample factors.** The screenshots are DISPLAY-RESAMPLED views
of the source ortho, and the x/y factors generally differ (JPEG aspect ratio ≠
source aspect ratio), so a single isotropic factor is wrong. Declare the source
orthomosaic dimensions in config:
    source_width_px  = <Wsrc>    # factor_x = source_width_px  / screenshot_W
    source_height_px = <Hsrc>    # factor_y = source_height_px / screenshot_H
The screenshot GSD is then `mpp_x = xres·factor_x`, `mpp_y = yres·factor_y`.
(A legacy isotropic `source_px_per_screenshot_px` is honoured only when the
`source_*_px` pair is absent.)

Usage:
    julia --project=. scripts/preprocess_site_image.jl CONFIG.toml [options]

    CONFIG.toml   Panel/preprocess TOML with [[site]] blocks (see
                  config/cross_site_panel.toml). Each site needs an `image`
                  key to be processed; sites without one are skipped.

Options:
    --site NAME   Process only the site whose `name` matches NAME.
    --preview     Permit an IMAGE-SPACE provisional column (c) when no metric
                  scale is available. Without it, image-space sites emit (a)/(b)
                  only and column (c) is skipped with a warning.
    --outroot DIR Root output directory (default: <repo>/output).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CairoMakie
using ColorSchemes
using CSV
using DataFrames
using FileIO, ImageIO, ColorTypes
using JSON
using Statistics
using TOML

import KDEFlightPlanning: geotransform_resolution, GT_NATIVE

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_slug(name::AbstractString) =
    lowercase(replace(strip(name), r"[^A-Za-z0-9]+" => "_")) |> s -> strip(s, '_')

_get(site, key, default=nothing) = haskey(site, key) ? site[key] : default

function _resolve_path(base::AbstractString, p)
    (p === nothing || isempty(String(p))) && return ""
    ps = String(p)
    isabspath(ps) ? ps : abspath(joinpath(base, ps))
end

# ---------------------------------------------------------------------------
# Metric-scale resolution
# ---------------------------------------------------------------------------

"""
    resolve_meters_per_pixel(site, base_dir, W, H)
        -> (mpp_x, mpp_y, source, factor_x, factor_y)

`mpp_x === nothing` ⇒ image-space (no reliable scale). Otherwise `mpp_x`/`mpp_y`
are metres per screenshot-pixel along each axis and `source` documents where the
scale came from.

The screenshots are DISPLAY-RESAMPLED views of a source orthomosaic, and the
x/y resample factors generally differ (the JPEG aspect ratio ≠ the source
aspect ratio). We therefore keep the axes independent:

    factor_x = source_width_px  / W        # source px per screenshot px, x
    factor_y = source_height_px / H        # source px per screenshot px, y

The source ground resolution comes from the source GeoTIFF's affine
geotransform (`geotransform_resolution`, hypot of the axis vectors), so a
screenshot pixel spans `xres * factor_x` metres in x and `yres * factor_y` in y.
"""
function resolve_meters_per_pixel(site, base_dir, W::Int, H::Int)
    # Anisotropic source-px-per-screenshot-px factors. Prefer explicit source
    # ortho dimensions (one per axis); fall back to a legacy isotropic factor.
    sw = _get(site, "source_width_px")
    sh = _get(site, "source_height_px")
    spp = _get(site, "source_px_per_screenshot_px")
    factor_x = sw !== nothing && Float64(sw) > 0 ? Float64(sw) / W :
               (spp !== nothing && Float64(spp) > 0 ? Float64(spp) : 1.0)
    factor_y = sh !== nothing && Float64(sh) > 0 ? Float64(sh) / H :
               (spp !== nothing && Float64(spp) > 0 ? Float64(spp) : 1.0)

    # 1. Explicit screenshot GSD (isotropic; overrides everything).
    mpp_explicit = _get(site, "meters_per_pixel")
    if mpp_explicit !== nothing && Float64(mpp_explicit) > 0
        m = Float64(mpp_explicit)
        return m, m, "explicit meters_per_pixel", 1.0, 1.0
    end

    # 2. Source GeoTIFF geotransform via the package's read_band (ArchGDAL).
    gtif = _resolve_path(base_dir, _get(site, "geotiff", ""))
    if !isempty(gtif) && isfile(gtif)
        _, gt, _ = read_band(gtif)                     # (Matrix, GeoTransform, crs)
        xres, yres = geotransform_resolution(gt)       # metres / SOURCE pixel
        return xres * factor_x, yres * factor_y,
               "geotiff geotransform ($(basename(gtif))): src $(round(xres;digits=6))×$(round(yres;digits=6)) m/px × " *
               "factor $(round(factor_x;digits=3))×$(round(factor_y;digits=3))",
               factor_x, factor_y
    end

    # 3. (Disabled by default) assume Durham native ortho GSD. Only fires if a
    #    site explicitly opts in; the supplied screenshots set this false because
    #    they are anisotropically resampled and need the real geotransform.
    if _get(site, "assume_durham_native_gsd", false) == true
        xres, yres = geotransform_resolution(GT_NATIVE)
        return xres * factor_x, yres * factor_y,
               "GT_NATIVE Durham GSD ($(round(xres;digits=6))×$(round(yres;digits=6)) m/px) × " *
               "factor $(round(factor_x;digits=3))×$(round(factor_y;digits=3))  [ASSUMED]",
               factor_x, factor_y
    end

    # 4. Image-space (no scale). Metric column (c) refused unless --preview.
    return nothing, nothing, "image-space (no CRS/GSD; supply `geotiff` for metric mode)",
           factor_x, factor_y
end

# ---------------------------------------------------------------------------
# Cluster-overlay rendering (visualisation only — no algorithm duplicated)
# ---------------------------------------------------------------------------

"""
    render_cluster_overlays(img, label_img, k, dst; alpha, greenness)

Alpha-blend each cluster id over the RGB screenshot and save one PNG per
cluster to `dst/cluster_overlay_k<cid>.png`. `greenness[cid]` (higher = greener,
i.e. more negative CIELAB a*) is shown in the title as a *candidate* cue only.
"""
function render_cluster_overlays(img::AbstractMatrix{<:Colorant},
                                 label_img::AbstractMatrix{<:Integer},
                                 k::Int, dst::AbstractString;
                                 alpha::Real = 0.55,
                                 greenness = nothing)
    mkpath(dst)
    H, W = size(img)
    palette = ColorSchemes.tab10
    paths = String[]
    for cid in 1:k
        fig = Figure(size = (760, max(240, round(Int, 700 * H / W))))
        gtxt = greenness === nothing ? "" :
               "  (candidate greenness=$(round(greenness[cid]; digits=2)))"
        ax = Axis(fig[1, 1]; title = "Cluster $cid overlay$gtxt",
                  aspect = DataAspect(), yreversed = true)
        hidedecorations!(ax)
        image!(ax, permutedims(img, (2, 1)))
        col = palette[mod1(cid, length(palette))]
        overlay = fill(RGBA(col.r, col.g, col.b, 0.0), H, W)
        @inbounds for j in 1:H, i in 1:W
            label_img[j, i] == cid && (overlay[j, i] = RGBA(col.r, col.g, col.b, alpha))
        end
        image!(ax, permutedims(overlay, (2, 1)))
        p = joinpath(dst, "cluster_overlay_k$(cid).png")
        try
            CairoMakie.save(p, fig; px_per_unit = 2)
            push!(paths, p)
        catch e
            @warn "Failed to save cluster overlay" cluster=cid exception=e
        end
    end
    return paths
end

# ---------------------------------------------------------------------------
# Per-site processing
# ---------------------------------------------------------------------------

function process_site(site, base_dir::AbstractString, outroot::AbstractString;
                      preview::Bool, defaults)
    name  = String(_get(site, "name", "(unnamed)"))
    image = _resolve_path(base_dir, _get(site, "image", ""))
    if isempty(image) || !isfile(image)
        @info "Skipping site (no readable `image`)" name image
        return nothing
    end

    slug = _slug(name)
    outdir = joinpath(outroot, slug)
    mkpath(joinpath(outdir, "cluster"))
    mkpath(joinpath(outdir, "kde"))
    mkpath(joinpath(outdir, "waypoints"))

    println("\n=== $name  ($slug) ===")
    println("  image = $image")

    # --- Load image ---------------------------------------------------------
    img_raw = FileIO.load(image)                 # Matrix{<:Colorant}, row1=top
    img = convert(Matrix{RGB{Float32}}, img_raw)
    H, W = size(img)
    println("  dims  = $(W)×$(H) px (W×H)")

    # --- Resolve metric scale ----------------------------------------------
    mpp_x, mpp_y, mpp_source, factor_x, factor_y =
        resolve_meters_per_pixel(site, base_dir, W, H)
    metric = mpp_x !== nothing
    println("  scale = ", metric ?
        "$(round(mpp_x; digits=6))×$(round(mpp_y; digits=6)) m/px  [$mpp_source]" :
        "IMAGE-SPACE  [$mpp_source]")

    # World extents for the RasterGrid axes:
    #   metric      → metres (xmax = W*mpp_x, ymax = H*mpp_y) so spacing_m honoured
    #   image-space → pixels  (xmax = W,       ymax = H)       (spacing = pixels)
    xmax = metric ? W * mpp_x : Float64(W)
    ymax = metric ? H * mpp_y : Float64(H)

    # --- Params -------------------------------------------------------------
    seed        = Int(_get(site, "seed", defaults.seed))
    nsample     = Int(_get(site, "nsample", defaults.nsample))
    kr          = _get(site, "kmedoids_k_range", defaults.k_range)
    ks          = Int(kr[1]):Int(kr[2])
    tree_labels = Int.(_get(site, "tree_labels", defaults.tree_labels))
    vmin, vmax  = Float64(defaults.speed_bounds[1]), Float64(defaults.speed_bounds[2])

    # --- (a) k-medoids mask (auto-k sweep) ---------------------------------
    println("  [a] k-medoids CIELAB clustering (ks=$(ks), tree_labels=$(tree_labels)) …")
    mask_grid, info = build_mask_from_image(img;
        tree_labels = tree_labels, seed = seed, nsample = nsample,
        ks = ks, k_strategy = :vote,
        xmin = 0.0, xmax = xmax, ymin = 0.0, ymax = ymax)
    k = info.k
    println("      chosen k = $k")

    # Candidate greenness per cluster (more negative CIELAB a* ⇒ greener veg).
    _, a_chan, _ = rgb_to_lab(img)
    label_img = reshape(info.labels_full, H, W)
    greenness = [ -mean(@view(a_chan[label_img .== cid])) for cid in 1:k ]
    suggested = argmax(greenness)

    render_cluster_overlays(img, label_img, k,
                            joinpath(outdir, "cluster"); greenness = greenness)

    # Honest tree-label decision record (never treated as ground truth).
    open(joinpath(outdir, "cluster", "tree_label_decision.md"), "w") do io
        println(io, "# Tree-label decision — $name")
        println(io)
        println(io, "Chosen k = **$k**. `tree_labels` currently set to ",
                    "`$(tree_labels)` (from config / default).")
        println(io)
        println(io, "> ⚠️  These labels are **NOT reviewed ground truth**. Snow and ",
                    "strong illumination in these winter screenshots make automatic ",
                    "vegetation-cluster identification unreliable. Inspect ",
                    "`cluster_overlay_k*.png` and set the correct cluster id(s) in the ",
                    "site's `tree_labels` before using this site for anything metric.")
        println(io)
        println(io, "Per-cluster candidate greenness (−mean CIELAB a*, higher = greener):")
        println(io)
        println(io, "| cluster | greenness | note |")
        println(io, "|---|---|---|")
        for cid in 1:k
            note = cid == suggested ? "← greenest (suggestion only)" :
                   (cid in tree_labels ? "← currently in tree_labels" : "")
            println(io, "| $cid | $(round(greenness[cid]; digits=3)) | $note |")
        end
    end
    println("      suggested vegetation cluster (review!): k=$suggested")

    # --- (b) Epanechnikov KDE planning surface -----------------------------
    println("  [b] Epanechnikov KDE planning surface …")
    kde_cfg = PipelineConfig(; kmed_k = k, seed = seed, tree_labels = tree_labels,
                               kde_bandwidth = :auto, kde_kernel = :epanechnikov,
                               kde_scaling = :none)
    dens_grid, _ = build_density_surface(mask_grid, kde_cfg)

    # Numeric grid (composer can render this without ArchGDAL) + heatmap PNG.
    CSV.write(joinpath(outdir, "kde", "kde_density_grid.csv"),
              DataFrame(dens_grid.Z, :auto); writeheader = false)
    let fig = Figure(size = (760, max(240, round(Int, 700 * H / W))))
        ax = Axis(fig[1, 1]; title = "Epanechnikov KDE (normalised)",
                  aspect = DataAspect(), yreversed = true)
        hidedecorations!(ax)
        heatmap!(ax, permutedims(dens_grid.Z); colormap = :viridis, colorrange = (0.0, 1.0))
        Colorbar(fig[1, 2]; colormap = :viridis, colorrange = (0.0, 1.0),
                 label = "normalised density")
        try CairoMakie.save(joinpath(outdir, "kde", "kde_density_heatmap.png"), fig; px_per_unit = 2)
        catch e; @warn "Failed to save KDE heatmap" exception=e end
    end

    # --- (c) waypoints coloured by 2–8 m/s ---------------------------------
    wp_csv = joinpath(outdir, "waypoints", "$(slug)_waypoints_xy_speed.csv")
    provisional = !metric
    wrote_c = false
    if metric || preview
        spacing_units = metric ? "m" : "px"
        line_spacing = metric ? Float64(defaults.track_spacing_m) :
                                Float64(_get(site, "preview_line_spacing_px",
                                             max(8.0, W / 12)))
        spc_min = metric ? Float64(defaults.min_wp_m) :
                           Float64(_get(site, "preview_min_wp_px", line_spacing / 4))
        spc_max = metric ? Float64(defaults.max_wp_m) :
                           Float64(_get(site, "preview_max_wp_px", line_spacing * 0.75))
        mode = metric ? "METRIC" : "IMAGE-SPACE PROVISIONAL"
        println("  [c] boustrophedon waypoints [$mode]  line_spacing=$(line_spacing) $spacing_units …")

        strat = CurvatureGuidedSpeed(dens_grid; vmin = vmin, vmax = vmax)
        fcfg  = FlightConfig(strat, 80.0, "KDE-guided (epanechnikov)";
                             kernel = :epanechnikov, line_spacing = line_spacing)
        wps = plan_mission(dens_grid, fcfg;
                           seconds_per_wp = 1.0,
                           spacing_min = spc_min, spacing_max = spc_max)

        df = DataFrame(x = [w.x for w in wps], y = [w.y for w in wps],
                       speed = [w.speed for w in wps],
                       line_id = [w.line_id for w in wps])
        CSV.write(wp_csv, df)
        wrote_c = true
        println("      → $(length(wps)) waypoints  v∈[$(round(minimum(df.speed);digits=2)), ",
                "$(round(maximum(df.speed);digits=2))] m/s")
        if provisional
            @warn "Column (c) is IMAGE-SPACE PROVISIONAL: spacing is in PIXELS, not metres. Do NOT read as 40 m-spaced metric plan." site=name
        end
    else
        @warn "No metric scale and --preview not set: skipping column (c). " *
              "Provide meters_per_pixel / geotiff / assume_durham_native_gsd, or rerun with --preview." site=name
    end

    # --- provenance ---------------------------------------------------------
    prov = Dict(
        "site"                 => name,
        "slug"                 => slug,
        "image"                => image,
        "image_width_px"       => W,
        "image_height_px"      => H,
        "mode"                 => metric ? "metric" : "image-space",
        "meters_per_pixel_x"   => metric ? mpp_x : nothing,
        "meters_per_pixel_y"   => metric ? mpp_y : nothing,
        "scale_source"         => mpp_source,
        "source_screenshot_factor_x" => factor_x,
        "source_screenshot_factor_y" => factor_y,
        "chosen_k"             => k,
        "tree_labels_used"     => tree_labels,
        "tree_labels_reviewed" => false,
        "suggested_veg_cluster"=> suggested,
        "speed_bounds_mps"     => [vmin, vmax],
        "column_c"             => wrote_c ? (metric ? "metric" : "image-space-provisional") : "skipped",
        "column_c_spacing_units" => metric ? "m" : (wrote_c ? "px" : nothing),
        "waypoints_csv"        => wrote_c ? wp_csv : nothing,
        "warning"              => metric ?
            "Vegetation cluster labels are unreviewed; verify tree_labels before publication." :
            "IMAGE-SPACE: no CRS/GSD. Columns (a)/(b) normalised; column (c) provisional (pixels, not metres). Not publication-ready.",
    )
    open(joinpath(outdir, "site_provenance.json"), "w") do io
        JSON.print(io, prov, 2)
    end
    println("  provenance → $(joinpath(outdir, "site_provenance.json"))")
    return prov
end

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

function main()
    args = copy(ARGS)
    preview = false
    site_filter = nothing
    outroot = abspath(joinpath(@__DIR__, "..", "output"))
    positional = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--preview"
            preview = true
        elseif a == "--site"
            site_filter = args[i += 1]
        elseif a == "--outroot"
            outroot = abspath(args[i += 1])
        else
            push!(positional, a)
        end
        i += 1
    end
    isempty(positional) &&
        error("Usage: julia preprocess_site_image.jl CONFIG.toml [--site NAME] [--preview] [--outroot DIR]")
    config_path = abspath(positional[1])
    isfile(config_path) || error("Config not found: $config_path")

    raw  = TOML.parsefile(config_path)
    base = dirname(config_path)

    # Planning defaults shared across sites (override per-site).
    defaults = (
        seed          = Int(get(raw, "seed", 6213)),
        nsample       = Int(get(raw, "nsample", 2000)),
        k_range       = get(raw, "kmedoids_k_range", [2, 8]),
        tree_labels   = Int.(get(raw, "tree_labels", [1])),
        speed_bounds  = (Float64(get(raw, "speed_min", 2.0)), Float64(get(raw, "speed_max", 8.0))),
        track_spacing_m = Float64(get(raw, "track_spacing_m", 40.0)),
        min_wp_m      = Float64(get(raw, "min_waypoint_spacing_m", 10.0)),
        max_wp_m      = Float64(get(raw, "max_waypoint_spacing_m", 30.0)),
    )

    sites = get(raw, "site", Any[])
    isempty(sites) && error("Config $config_path has no [[site]] entries.")

    mkpath(outroot)
    println("[preprocess] config = $config_path")
    println("[preprocess] outroot = $outroot   preview=$preview")

    processed = 0
    for s in sites
        site_filter !== nothing && String(get(s, "name", "")) != site_filter && continue
        haskey(s, "image") || continue
        process_site(s, base, outroot; preview = preview, defaults = defaults)
        processed += 1
    end
    println("\n[preprocess] done — processed $processed site(s).")
    processed == 0 && @warn "No sites had an `image` key (or --site filter matched none)."
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
