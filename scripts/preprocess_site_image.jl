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
Input selection + scale extraction (GeoTIFF is authoritative)
──────────────────────────────────────────────────────────────────────────
Each site is processed from ONE raster, chosen GeoTIFF-first:

  • `geotiff` resolves to a readable file → it is the AUTHORITATIVE source for
    BOTH the RGB pixels AND the geospatial transform/CRS. RGB is read with
    `load_rgb_geotiff`; native metres/px come straight from
    `geotransform_resolution(gt)` (hypot of the affine axis vectors, so
    rotation/skew are handled). NO screenshot resample factor is applied and
    `source_width_px`/`source_height_px` are ignored in this mode.
  • otherwise, `image` (JPEG/PNG) is a FALLBACK only — image-space, or a scale
    resolved by `resolve_meters_per_pixel` (see below).

For the image FALLBACK, per-axis metres/px is resolved (first match wins):

  1. `meters_per_pixel` set on the site  → used verbatim (isotropic).
  2. `assume_durham_native_gsd = true`   → (opt-in, default false) use the
     bundled Durham validation ortho geotransform `GT_NATIVE` (≈0.02837 m/px)
     × per-axis resample factor.
  3. otherwise                           → IMAGE-SPACE.

**Anisotropic resample factors (image fallback only).** A display-resampled
screenshot's x/y factors generally differ (JPEG aspect ratio ≠ source aspect
ratio). Declare the source orthomosaic dimensions in config:
    source_width_px  = <Wsrc>    # factor_x = source_width_px  / screenshot_W
    source_height_px = <Hsrc>    # factor_y = source_height_px / screenshot_H
These factors scale the `assume_durham_native_gsd` GSD and are IGNORED whenever
a `geotiff` is present. (A legacy isotropic `source_px_per_screenshot_px` is
honoured only when the `source_*_px` pair is absent.)

**Large rasters.** Set `cluster_stride = N` (per-site or top-level) to cluster
on an N×-decimated grid; the world extent still spans the full ground footprint,
so metric spacing stays correct.

Usage:
    julia --project=. scripts/preprocess_site_image.jl CONFIG.toml [options]

    CONFIG.toml   Panel/preprocess TOML with [[site]] blocks (see
                  config/cross_site_panel.toml). Each site needs a readable
                  `geotiff` (preferred, authoritative) or `image` (fallback) to
                  be processed; sites with neither are skipped.

Options:
    --site NAME   Process only the site whose `name` matches NAME.
    --preview     Permit an IMAGE-SPACE provisional column (c) when no metric
                  scale is available. Without it, image-space sites emit (a)/(b)
                  only and column (c) is skipped with a warning.
    --outroot DIR Root output directory (default: <repo>/output).

──────────────────────────────────────────────────────────────────────────
Vegetation cluster labels (`tree_labels`) — interactive first run, then reproducible
──────────────────────────────────────────────────────────────────────────
There is NO silent `tree_labels = [1]` default. The vegetation cluster id(s)
are resolved per-site, AFTER clustering + overlay rendering, as follows:

  • `tree_labels` present & valid in the site's [[site]] block → used verbatim,
    no prompt (fully reproducible; CI/batch safe).
  • absent/empty AND stdin is an interactive TTY → the per-cluster overlays are
    previewed and you are prompted to enter the vegetation cluster id(s) in 1:k
    (reprompting until valid). The confirmed ids are PERSISTED back into the
    matching [[site]] block of the SAME config (line-preserving, atomic; comments
    and ordering are kept), so the next run of that site is non-interactive.
  • absent/empty AND stdin is NOT a TTY → hard error with actionable steps (add
    `tree_labels` or rerun interactively). It will NOT guess `[1]`/greenest.

So the intended workflow is: run once interactively to review + lock in labels
(written to the config), then all subsequent runs are reproducible.
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

# Pure helpers for the interactive vegetation-label workflow (parse / validate /
# prompt / line-preserving TOML persistence). Kept in a separate, dependency-light
# file so they can be unit-tested without loading CairoMakie / KDEFlightPlanning.
include(joinpath(@__DIR__, "tree_label_selection.jl"))

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
    resolve_meters_per_pixel(site, W, H)
        -> (mpp_x, mpp_y, source, factor_x, factor_y)

Scale resolution for the **image (JPEG/PNG) fallback path only**. When a site
supplies a `geotiff`, the GeoTIFF is loaded upstream (`select_site_input` →
`load_rgb_geotiff`) as the authoritative RGB + geotransform source and this
function is NOT called — native m/px comes straight from
`geotransform_resolution(gt)` with no screenshot resample factor.

`mpp_x === nothing` ⇒ image-space (no reliable scale). Otherwise `mpp_x`/`mpp_y`
are metres per screenshot-pixel along each axis and `source` documents where the
scale came from:

  1. explicit `meters_per_pixel` (isotropic; factors = 1); else
  2. opt-in `assume_durham_native_gsd` (`GT_NATIVE` × per-axis factor); else
  3. image-space (no scale).

The anisotropic factors `factor_x = source_width_px/W`,
`factor_y = source_height_px/H` describe a screenshot that was display-resampled
from a source ortho; they are only meaningful for the `assume_durham_native_gsd`
fallback and are ignored entirely in GeoTIFF mode.
"""
function resolve_meters_per_pixel(site, W::Int, H::Int)
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

    # 2. (Disabled by default) assume Durham native ortho GSD. Only fires if a
    #    site explicitly opts in. Prefer supplying a `geotiff` (authoritative)
    #    over this assumed scale.
    if _get(site, "assume_durham_native_gsd", false) == true
        xres, yres = geotransform_resolution(GT_NATIVE)
        return xres * factor_x, yres * factor_y,
               "GT_NATIVE Durham GSD ($(round(xres;digits=6))×$(round(yres;digits=6)) m/px) × " *
               "factor $(round(factor_x;digits=3))×$(round(factor_y;digits=3))  [ASSUMED]",
               factor_x, factor_y
    end

    # 3. Image-space (no scale). Metric column (c) refused unless --preview.
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
# Vegetation cluster-label resolution (NO silent default)
# ---------------------------------------------------------------------------

"""
    resolve_site_tree_labels(cfg_labels, k, overlay_paths; name, suggested,
                             config_path) -> (labels::Vector{Int}, source::String, persisted::Bool)

Decide the vegetation cluster label set for a site AFTER clustering + overlay
rendering, with NO silent `[1]`/greenest fallback:

  • `cfg_labels` valid & nonempty  → use as-is (validated against `1:k`);
    source = "config", persisted = false (already in the config).
  • absent/empty + interactive TTY → preview overlays and prompt (reprompting on
    invalid input), then persist the choice into the site's config block;
    source = "interactive".
  • absent/empty + non-TTY         → error with actionable instructions.
"""
function resolve_site_tree_labels(cfg_labels, k::Int,
                                  overlay_paths::AbstractVector{<:AbstractString};
                                  name::AbstractString, suggested::Int,
                                  config_path::AbstractString)
    if cfg_labels !== nothing
        err = validate_configured_tree_labels(cfg_labels, k)
        err === nothing || error(
            "Configured tree_labels for site \"$name\" are invalid: $err.\n" *
            "Fix the `tree_labels` entry in the config (valid cluster ids are 1:$k), " *
            "or remove it to select interactively.")
        return sort(unique(Int.(cfg_labels))), "config", false
    end

    if !stdin_is_tty()
        error(
            "No `tree_labels` configured for site \"$name\" and stdin is not a TTY.\n" *
            "This cross-site preprocessing path will NOT guess a vegetation cluster.\n" *
            "Do one of:\n" *
            "  1. Add `tree_labels = [..]` (cluster ids in 1:$k) to the site's [[site]] " *
            "block in the config, then rerun; or\n" *
            "  2. Rerun in an interactive terminal to review the cluster overlays and " *
            "select the vegetation cluster(s). Overlays written to:\n" *
            join(("       " * p for p in overlay_paths), "\n"))
    end

    labels = prompt_tree_labels(k, overlay_paths; name = name, suggested = suggested)
    ok, msg = persist_tree_labels(config_path, name, labels)
    if ok
        @info "Persisted selected tree_labels into config" site=name labels=labels config=config_path
    else
        @warn "Could not persist tree_labels to config; using them for THIS run only. " *
              "Add them manually to avoid re-prompting." site=name labels=labels reason=msg
    end
    return labels, "interactive", ok
end

# ---------------------------------------------------------------------------
# Per-site processing
# ---------------------------------------------------------------------------

function process_site(site, base_dir::AbstractString, outroot::AbstractString;
                      preview::Bool, defaults, config_path::AbstractString)
    name  = String(_get(site, "name", "(unnamed)"))

    # GeoTIFF-first input selection: a resolvable `geotiff` is authoritative for
    # BOTH the RGB pixels and the geospatial transform/CRS; the `image` JPEG/PNG
    # is a fallback used only when no usable GeoTIFF is supplied.
    input_kind, input_path = select_site_input(site, base_dir)
    if input_kind == :none
        @info "Skipping site (neither `geotiff` nor `image` resolves to a readable file)" name geotiff=_get(site, "geotiff", "") image=_get(site, "image", "")
        return nothing
    end

    slug = _slug(name)
    outdir = joinpath(outroot, slug)
    mkpath(joinpath(outdir, "cluster"))
    mkpath(joinpath(outdir, "kde"))
    mkpath(joinpath(outdir, "waypoints"))

    println("\n=== $name  ($slug) ===")

    # --- Load RGB + resolve metric scale -----------------------------------
    # GeoTIFF mode: native m/px straight from the geotransform, NO screenshot
    # resample factor. Image mode: JPEG/PNG fallback + resolve_meters_per_pixel.
    local img_full, mpp_x, mpp_y, mpp_source, factor_x, factor_y, crs
    if input_kind == :geotiff
        rs       = load_rgb_geotiff(input_path)          # authoritative RGB + gt + crs
        img_full = convert(Matrix{RGB{Float32}}, rs.Z)
        H_full, W_full = size(img_full)
        xres, yres = geotransform_resolution(rs.gt)      # native metres / GeoTIFF pixel
        mpp_x, mpp_y = xres, yres
        factor_x = factor_y = 1.0                         # no display resample in GeoTIFF mode
        crs = rs.crs
        mpp_source = "geotiff geotransform ($(basename(input_path))): native " *
                     "$(round(xres; digits=6))×$(round(yres; digits=6)) m/px (no screenshot resample)"
        println("  input = $input_path   [GeoTIFF — authoritative RGB + transform/CRS]")
        println("  dims  = $(W_full)×$(H_full) px (W×H)")
        println("  crs   = ", isempty(crs) ? "<unknown>" : crs)
        println("  scale = $(round(mpp_x; digits=6))×$(round(mpp_y; digits=6)) m/px  [native geotransform]")
    else # :image
        img_raw  = FileIO.load(input_path)               # Matrix{<:Colorant}, row1=top
        img_full = convert(Matrix{RGB{Float32}}, img_raw)
        H_full, W_full = size(img_full)
        mpp_x, mpp_y, mpp_source, factor_x, factor_y =
            resolve_meters_per_pixel(site, W_full, H_full)
        crs = ""
        println("  input = $input_path   [image fallback — JPEG/PNG, no CRS]")
        println("  dims  = $(W_full)×$(H_full) px (W×H)")
        println("  scale = ", (mpp_x !== nothing) ?
            "$(round(mpp_x; digits=6))×$(round(mpp_y; digits=6)) m/px  [$mpp_source]" :
            "IMAGE-SPACE  [$mpp_source]")
    end
    metric = mpp_x !== nothing

    # --- Optional decimation for tractable clustering on large rasters ------
    # Full-resolution orthomosaics (e.g. 11k×9k) are far too large to cluster at
    # native resolution, so cluster on a stride-decimated grid. The world extent
    # below still spans the FULL ground footprint, so metric spacing is honoured.
    cluster_stride = max(1, Int(_get(site, "cluster_stride", defaults.cluster_stride)))
    if cluster_stride > 1
        img = Matrix(@view img_full[1:cluster_stride:end, 1:cluster_stride:end])
        img_full = nothing                                # release full-res RGB for GC
        H, W = size(img)
        println("  cluster_stride = $cluster_stride → clustering grid $(W)×$(H) px " *
                "(full-res ground extent preserved)")
    else
        img = img_full
        H, W = H_full, W_full
    end

    # World extents for the RasterGrid axes span the FULL ground footprint
    # regardless of clustering decimation:
    #   metric      → metres (xmax = W_full*mpp_x, ymax = H_full*mpp_y)
    #   image-space → full pixels (xmax = W_full,   ymax = H_full)
    xmax = metric ? W_full * mpp_x : Float64(W_full)
    ymax = metric ? H_full * mpp_y : Float64(H_full)

    # --- Params -------------------------------------------------------------
    seed        = Int(_get(site, "seed", defaults.seed))
    nsample     = Int(_get(site, "nsample", defaults.nsample))
    kr          = _get(site, "kmedoids_k_range", defaults.k_range)
    ks          = Int(kr[1]):Int(kr[2])
    vmin, vmax  = Float64(defaults.speed_bounds[1]), Float64(defaults.speed_bounds[2])

    # Vegetation labels come ONLY from this site's block — no silent [1] default.
    # `nothing` here means absent/empty → resolved interactively (TTY) or errors.
    cfg_labels  = site_configured_tree_labels(site)

    # --- (a) k-medoids clustering (labels are independent of tree_labels) ---
    # Cluster first with an empty selection to obtain labels_full + chosen k and
    # render the per-cluster overlays; the vegetation label set is decided AFTER
    # the overlays exist so the user can review them.
    println("  [a] k-medoids CIELAB clustering (ks=$(ks)) …")
    _, info = build_mask_from_image(img;
        tree_labels = Int[], seed = seed, nsample = nsample,
        ks = ks, k_strategy = :vote,
        xmin = 0.0, xmax = xmax, ymin = 0.0, ymax = ymax)
    k = info.k
    println("      chosen k = $k")

    # Candidate greenness per cluster (more negative CIELAB a* ⇒ greener veg).
    _, a_chan, _ = rgb_to_lab(img)
    label_img = reshape(info.labels_full, H, W)
    greenness = [ -mean(@view(a_chan[label_img .== cid])) for cid in 1:k ]
    suggested = argmax(greenness)

    overlay_paths = render_cluster_overlays(img, label_img, k,
                            joinpath(outdir, "cluster"); greenness = greenness)
    println("      suggested vegetation cluster (hint only): k=$suggested")

    # --- Resolve vegetation labels (config → interactive prompt → error) ----
    tree_labels, label_source, label_persisted = resolve_site_tree_labels(
        cfg_labels, k, overlay_paths;
        name = name, suggested = suggested, config_path = config_path)
    tree_labels_reviewed = label_source in ("config", "interactive")
    println("      tree_labels = $tree_labels  [source=$label_source",
            label_source == "interactive" ? (label_persisted ? ", persisted" : ", NOT persisted") : "", "]")

    # Rebuild the vegetation mask from the confirmed labels (reusing the same
    # clustering + world axes) so KDE/speed/waypoints reflect the final choice.
    veg = labels_to_mask(info.labels_full, H, W; tree_labels = tree_labels)
    mask_grid = RasterGrid(Float64.(veg), info.xs, info.ys)

    # Tree-label decision record (now reflects the confirmed selection).
    open(joinpath(outdir, "cluster", "tree_label_decision.md"), "w") do io
        println(io, "# Tree-label decision — $name")
        println(io)
        println(io, "Chosen k = **$k**. `tree_labels` = `$(tree_labels)` ",
                    "(source: **$label_source**",
                    label_source == "interactive" ?
                        (label_persisted ? ", persisted to config)." : ", NOT persisted — add manually).") :
                        ").")
        println(io)
        if label_source == "config"
            println(io, "> These labels were supplied explicitly in the site's config block ",
                        "and used without prompting.")
        else
            println(io, "> These labels were confirmed interactively after reviewing the ",
                        "`cluster_overlay_k*.png` overlays.")
        end
        println(io)
        println(io, "Per-cluster candidate greenness (−mean CIELAB a*, higher = greener):")
        println(io)
        println(io, "| cluster | greenness | note |")
        println(io, "|---|---|---|")
        for cid in 1:k
            note = cid in tree_labels ? "← selected (vegetation)" :
                   (cid == suggested ? "← greenest (hint only)" : "")
            println(io, "| $cid | $(round(greenness[cid]; digits=3)) | $note |")
        end
    end

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
        "input_kind"           => String(input_kind),
        "input_path"           => input_path,
        "crs"                  => crs,
        "image_width_px"       => W_full,
        "image_height_px"      => H_full,
        "cluster_stride"       => cluster_stride,
        "cluster_grid_w"       => W,
        "cluster_grid_h"       => H,
        "mode"                 => metric ? "metric" : "image-space",
        "meters_per_pixel_x"   => metric ? mpp_x : nothing,
        "meters_per_pixel_y"   => metric ? mpp_y : nothing,
        "scale_source"         => mpp_source,
        "source_screenshot_factor_x" => factor_x,
        "source_screenshot_factor_y" => factor_y,
        "chosen_k"             => k,
        "tree_labels_used"     => tree_labels,
        "tree_labels_reviewed" => tree_labels_reviewed,
        "tree_labels_source"   => label_source,
        "tree_labels_persisted"=> label_persisted,
        "suggested_veg_cluster"=> suggested,
        "speed_bounds_mps"     => [vmin, vmax],
        "column_c"             => wrote_c ? (metric ? "metric" : "image-space-provisional") : "skipped",
        "column_c_spacing_units" => metric ? "m" : (wrote_c ? "px" : nothing),
        "waypoints_csv"        => wrote_c ? wp_csv : nothing,
        "warning"              => metric ?
            "Vegetation cluster labels were $(label_source == "config" ? "supplied in config" : "confirmed interactively"); verify tree_labels before publication." :
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
        speed_bounds  = (Float64(get(raw, "speed_min", 2.0)), Float64(get(raw, "speed_max", 8.0))),
        track_spacing_m = Float64(get(raw, "track_spacing_m", 40.0)),
        min_wp_m      = Float64(get(raw, "min_waypoint_spacing_m", 10.0)),
        max_wp_m      = Float64(get(raw, "max_waypoint_spacing_m", 30.0)),
        cluster_stride = Int(get(raw, "cluster_stride", 1)),
    )

    sites = get(raw, "site", Any[])
    isempty(sites) && error("Config $config_path has no [[site]] entries.")

    mkpath(outroot)
    println("[preprocess] config = $config_path")
    println("[preprocess] outroot = $outroot   preview=$preview")

    processed = 0
    for s in sites
        site_filter !== nothing && String(get(s, "name", "")) != site_filter && continue
        (haskey(s, "geotiff") || haskey(s, "image")) || continue
        process_site(s, base, outroot; preview = preview, defaults = defaults,
                     config_path = config_path)
        processed += 1
    end
    println("\n[preprocess] done — processed $processed site(s).")
    processed == 0 && @warn "No sites had a `geotiff` or `image` key (or --site filter matched none)."
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
