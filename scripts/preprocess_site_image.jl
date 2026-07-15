"""
    scripts/preprocess_site_image.jl

Per-site producer for the cross-site planning panel. This is a **thin
orchestrator** over the KDEFlightPlanning package: it drives the exact same
canonical pipeline as `scripts/run_from_config.jl` (the proven Durham driver),
once per `[[site]]` block, and writes the canonical stage artefacts that
`scripts/make_cross_site_panel.jl` composes into the figure:

    output/<slug>/cluster/cluster_overlay_k*.png   ← column (a)  report_cluster_overlays
    output/<slug>/cluster/tree_label_decision.md   ← report_tree_label_decision
    output/<slug>/kde/kde_density_heatmap.png      ← column (b)  report_kde_density
    output/<slug>/kde/speed_map.png                ← report_speed_map
    output/<slug>/waypoints/<slug>_waypoints_xy_speed.csv  ← column (c)  write_waypoints_csv
    output/<slug>/waypoints/waypoints_overlay.png  ← report_waypoints_overlay
    output/<slug>/site_provenance.json             ← run record

There is NO bespoke visualisation code here: every figure is produced by the
package's `report_*` functions, so the panel columns are byte-identical in
style to the single-site Durham run.

────────────────────────────────────────────────────────────────────────────
Input selection (GeoTIFF is authoritative)
────────────────────────────────────────────────────────────────────────────
Each site is processed from ONE raster, chosen GeoTIFF-first:

  • `geotiff` resolves to a readable file → AUTHORITATIVE source for BOTH the
    RGB pixels AND the geospatial transform/CRS (`load_rgb_geotiff`). Axes are
    UTM metres, so flight-line/waypoint spacing (`track_spacing_m`, etc.) is
    honoured in METRES.
  • otherwise `image` (JPEG/PNG) is a FALLBACK: it is wrapped in an identity
    pixel-space `GeoRasterStack` (no CRS). Columns (a)/(b) are scale-free;
    column (c) is emitted in PIXEL units and the row is marked
    `mode = "image-space"` in provenance. No metre scale is invented.

Sites where neither input resolves to a readable file are skipped.

────────────────────────────────────────────────────────────────────────────
Vegetation cluster labels (`tree_labels`) — interactive first run, then reproducible
────────────────────────────────────────────────────────────────────────────
There is NO silent `tree_labels` default. Resolution is per-site, AFTER
clustering + overlay rendering:

  • `tree_labels` present & valid in the site's [[site]] block → used verbatim,
    no prompt (reproducible; CI/batch safe).
  • absent/empty AND stdin is a TTY → the per-cluster overlays are previewed and
    you are prompted for the vegetation cluster id(s) in 1:k (reprompting until
    valid). The confirmed ids are PERSISTED back into the matching [[site]]
    block of the SAME config (line-preserving, atomic), so the next run is
    non-interactive.
  • absent/empty AND stdin is NOT a TTY → hard error (never guesses).

Usage:
    julia --project=. scripts/preprocess_site_image.jl CONFIG.toml [options]

Options:
    --site NAME    Process only the site whose `name` matches NAME.
    --outroot DIR  Root output directory (default: <repo>/output).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CSV
using DataFrames
using FileIO, ImageIO
using Colors: RGB, red, green, blue
using ColorTypes: N0f8
using JSON
using Statistics
using TOML

import KDEFlightPlanning: _gt_as_vector

# Pure, stdlib-only helpers for the interactive vegetation-label workflow
# (input selection, parse/validate/prompt, line-preserving TOML persistence).
# Kept separate so they can be unit-tested without loading the package.
include(joinpath(@__DIR__, "tree_label_selection.jl"))

_get(site, key, default = nothing) = haskey(site, key) ? site[key] : default

# ---------------------------------------------------------------------------
# Input → uniform GeoRasterStack
# ---------------------------------------------------------------------------

"""
    load_site_raster(kind, path) -> GeoRasterStack{RGB{N0f8}}

Load a site's RGB raster as a `GeoRasterStack`, uniform across both input kinds:

  • `:geotiff` → `load_rgb_geotiff` (real geotransform + CRS).
  • `:image`   → decode the JPEG/PNG and wrap it in an identity pixel-space
                 geotransform (`dx = 1`, `dy = -1`, origin at the top-left) with
                 an empty CRS, so every downstream `report_*` call is identical.
"""
function load_site_raster(kind::Symbol, path::AbstractString)
    if kind == :geotiff
        return load_rgb_geotiff(path)
    end
    img = convert(Matrix{RGB{N0f8}}, FileIO.load(path))   # row 1 = top
    H, W = size(img)
    gt = GeoTransform([0.0, 1.0, 0.0, Float64(H), 0.0, -1.0])   # north-up pixel grid
    return GeoRasterStack{RGB{N0f8}}(img, gt, "", String(path))
end

# ---------------------------------------------------------------------------
# Vegetation cluster-label resolution (NO silent default)
# ---------------------------------------------------------------------------

"""
    resolve_site_tree_labels(cfg_labels, k, overlay_paths; name, suggested,
                             config_path) -> (labels, source, persisted)

Config → interactive prompt → error, with NO silent fallback. See the module
docstring for the full contract.
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
# Per-site processing — mirrors scripts/run_from_config.jl exactly
# ---------------------------------------------------------------------------

function process_site(site, base_dir::AbstractString, outroot::AbstractString;
                      defaults, config_path::AbstractString)
    name = String(_get(site, "name", "(unnamed)"))

    kind, input_path = select_site_input(site, base_dir)
    if kind == :none
        @info "Skipping site (neither `geotiff` nor `image` resolves to a readable file)" name geotiff=_get(site, "geotiff", "") image=_get(site, "image", "")
        return nothing
    end

    slug   = tree_labels_slug(name)
    outdir = joinpath(outroot, slug)
    mkpath(outdir)
    println("\n=== $name  ($slug) ===")

    # --- Load RGB (GeoTIFF authoritative; image = identity pixel grid) ------
    rs = load_site_raster(kind, input_path)
    H, W = size(rs)
    metric = (kind == :geotiff)
    gt_report = metric ? rs.gt : nothing
    georef    = metric
    println("  input = $input_path   [", metric ? "GeoTIFF — authoritative RGB + transform/CRS" : "image fallback — pixel-space, no CRS", "]")
    println("  dims  = $(W)×$(H) px (W×H)")
    if metric
        xr, yr = geotransform_resolution(rs.gt)
        println("  scale = $(round(xr; digits=6))×$(round(yr; digits=6)) m/px  [native geotransform]")
    else
        println("  scale = IMAGE-SPACE (pixel units; column (c) spacing is in pixels)")
    end

    # --- Params -------------------------------------------------------------
    seed        = Int(_get(site, "seed", defaults.seed))
    nsample     = Int(_get(site, "nsample", defaults.nsample))
    kr          = _get(site, "kmedoids_k_range", defaults.k_range)
    ks          = Int(kr[1]):Int(kr[2])
    vmin, vmax  = Float64(defaults.speed_bounds[1]), Float64(defaults.speed_bounds[2])
    cluster_stride = max(1, Int(_get(site, "cluster_stride", defaults.cluster_stride)))

    # --- k-medoids clustering (labels are independent of tree_labels) -------
    # Cluster on the strided RGB (from the packed HWC array) to obtain the full-
    # resolution labels + chosen k. `tree_labels` is a placeholder here; the
    # vegetation mask is rebuilt below from the confirmed selection.
    arr_hwc = let Z = rs.Z
        a = Array{UInt8}(undef, H, W, 3)
        @inbounds for j in 1:H, i in 1:W
            p = Z[j, i]
            a[j, i, 1] = round(UInt8, clamp(Float64(red(p))   * 255, 0, 255))
            a[j, i, 2] = round(UInt8, clamp(Float64(green(p)) * 255, 0, 255))
            a[j, i, 3] = round(UInt8, clamp(Float64(blue(p))  * 255, 0, 255))
        end
        a
    end
    println("  [a] k-medoids CIELAB clustering (ks=$(ks), stride=$cluster_stride) …")
    _, mask_info = build_mask_from_image_strided(arr_hwc;
        stride = cluster_stride,
        k = first(ks), tree_labels = [1],
        seed = seed, nsample = nsample,
        ks = ks, k_strategy = :vote,
        use_pca = false, do_cleanup = false)
    k = mask_info.k
    println("      chosen k = $k")

    # World axes span the FULL ground footprint (metres for GeoTIFF, pixels
    # otherwise). Downstream waypoint spacing is interpreted in these units.
    xs, ys = axes_from_geotransform(_gt_as_vector(rs.gt), W, H)

    # --- Column (a): canonical per-cluster overlays -------------------------
    report_cluster_overlays(rs, mask_info.labels_full, k, outdir;
        georeference = georef)
    overlay_paths = [joinpath(outdir, "cluster", "cluster_overlay_k$(cid).png")
                     for cid in 1:k]

    # Per-cluster LAB summary (canonical greenness diagnostic + candidate hint).
    arr_for_lab = cluster_stride > 1 ?
        Array(@view arr_hwc[1:cluster_stride:end, 1:cluster_stride:end, :]) : arr_hwc
    L_chan, a_chan, b_chan = rgb_to_lab_array(arr_for_lab)
    labels_for_summary = if cluster_stride > 1
        H_s = get(mask_info, :H_sample, H)
        W_s = get(mask_info, :W_sample, W)
        full = mask_info.labels_full
        out = Vector{Int}(undef, H_s * W_s)
        @inbounds for j in 1:H_s, i in 1:W_s
            out[(j - 1) * W_s + i] =
                full[(j - 1) * cluster_stride * W + (i - 1) * cluster_stride + 1]
        end
        out
    else
        mask_info.labels_full
    end
    df_lab, _ = report_cluster_lab_summary(L_chan, a_chan, b_chan,
                                           labels_for_summary, k, outdir)
    suggested = df_lab.cluster_id[argmax(df_lab.greenness_neg_a)]
    println("      suggested vegetation cluster (hint only): k=$suggested")

    # --- Resolve vegetation labels (config → interactive → error) -----------
    cfg_labels = site_configured_tree_labels(site)
    tree_labels, label_source, label_persisted = resolve_site_tree_labels(
        cfg_labels, k, overlay_paths;
        name = name, suggested = suggested, config_path = config_path)
    println("      tree_labels = $tree_labels  [source=$label_source",
            label_source == "interactive" ? (label_persisted ? ", persisted" : ", NOT persisted") : "", "]")

    report_tree_label_decision(outdir;
        tree_labels    = tree_labels,
        lab_summary_df = df_lab,
        source         = label_source == "config" ?
            "config ([[site]].tree_labels)" : "interactive (reviewed cluster overlays)",
        notes          = "Recorded by preprocess_site_image.jl for site \"$name\". " *
                         "Edit `tree_labels` in $(basename(config_path)) to change.")

    # Rebuild the vegetation mask from the confirmed labels, using the SAME
    # row-major label→image convention as run_from_config.jl.
    label_img = permutedims(reshape(mask_info.labels_full, W, H), (2, 1))
    tree_mask = in.(label_img, Ref(tree_labels))
    mask_rg   = RasterGrid(Float64.(tree_mask), xs, ys)

    # --- Column (b): Epanechnikov KDE planning surface + speed map ----------
    println("  [b] Epanechnikov KDE planning surface …")
    kde_cfg = PipelineConfig(; kmed_k = k, seed = seed, tree_labels = tree_labels,
                               kde_bandwidth = :auto, kde_kernel = :epanechnikov,
                               kde_scaling = :none)
    dens_grid, _ = build_density_surface(mask_rg, kde_cfg)
    report_kde_density(dens_grid, outdir; gt = gt_report, crs = rs.crs)
    if metric
        try
            write_single_band_geotiff(joinpath(outdir, "kde", "kde_density.tif"),
                                      dens_grid.Z, rs.gt, rs.crs)
        catch e
            @warn "Failed to write co-registered kde_density.tif" exception=e
        end
    end

    strat = CurvatureGuidedSpeed(dens_grid; vmin = vmin, vmax = vmax)
    report_speed_map(dens_grid, strat, outdir; gt = gt_report, crs = rs.crs)

    # --- Column (c): boustrophedon waypoints coloured by speed --------------
    line_spacing = Float64(defaults.track_spacing_m)
    println("  [c] boustrophedon waypoints  line_spacing=$(line_spacing) ",
            metric ? "m" : "px", " …")
    fcfg = FlightConfig(strat, 80.0, "KDE-guided (epanechnikov)";
                        kernel = :epanechnikov, line_spacing = line_spacing)
    wps = plan_mission(dens_grid, fcfg;
                       seconds_per_wp = 1.0,
                       spacing_min = Float64(defaults.min_wp),
                       spacing_max = Float64(defaults.max_wp))
    wp_csv = joinpath(outdir, "waypoints", "$(slug)_waypoints_xy_speed.csv")
    mkpath(dirname(wp_csv))
    write_waypoints_csv(wp_csv, wps)
    report_waypoints_overlay(rs, dens_grid,
        Dict("KDE-guided (epanechnikov)" => wps), outdir; georeference = georef)
    println("      → $(length(wps)) waypoints  v∈[",
            "$(round(minimum(w.speed for w in wps); digits=2)), ",
            "$(round(maximum(w.speed for w in wps); digits=2))] m/s")
    if !metric
        @warn "Column (c) is IMAGE-SPACE: waypoint spacing is in PIXELS, not metres. " *
              "Supply a `geotiff` for a surveyed metric plan." site=name
    end

    # --- provenance ---------------------------------------------------------
    prov = Dict(
        "site"                 => name,
        "slug"                 => slug,
        "input_kind"           => String(kind),
        "input_path"           => input_path,
        "crs"                  => rs.crs,
        "image_width_px"       => W,
        "image_height_px"      => H,
        "cluster_stride"       => cluster_stride,
        "mode"                 => metric ? "metric" : "image-space",
        "chosen_k"             => k,
        "tree_labels_used"     => tree_labels,
        "tree_labels_source"   => label_source,
        "tree_labels_persisted"=> label_persisted,
        "suggested_veg_cluster"=> suggested,
        "speed_bounds_mps"     => [vmin, vmax],
        "waypoints_csv"        => wp_csv,
        "column_c_spacing_units" => metric ? "m" : "px",
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
    site_filter = nothing
    outroot = abspath(joinpath(@__DIR__, "..", "output"))
    positional = String[]
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--site"
            site_filter = args[i += 1]
        elseif a == "--outroot"
            outroot = abspath(args[i += 1])
        else
            push!(positional, a)
        end
        i += 1
    end
    isempty(positional) &&
        error("Usage: julia preprocess_site_image.jl CONFIG.toml [--site NAME] [--outroot DIR]")
    config_path = abspath(positional[1])
    isfile(config_path) || error("Config not found: $config_path")

    assert_no_conflict_markers(config_path)   # clear error on committed merge/stash markers
    raw  = TOML.parsefile(config_path)
    base = dirname(config_path)

    defaults = (
        seed          = Int(get(raw, "seed", 6213)),
        nsample       = Int(get(raw, "nsample", 2000)),
        k_range       = get(raw, "kmedoids_k_range", [2, 8]),
        speed_bounds  = (Float64(get(raw, "speed_min", 2.0)), Float64(get(raw, "speed_max", 8.0))),
        track_spacing_m = Float64(get(raw, "track_spacing_m", 40.0)),
        min_wp        = Float64(get(raw, "min_waypoint_spacing_m", 10.0)),
        max_wp        = Float64(get(raw, "max_waypoint_spacing_m", 30.0)),
        cluster_stride = Int(get(raw, "cluster_stride", 1)),
    )

    sites = get(raw, "site", Any[])
    isempty(sites) && error("Config $config_path has no [[site]] entries.")

    mkpath(outroot)
    println("[preprocess] config = $config_path")
    println("[preprocess] outroot = $outroot")

    processed = 0
    for s in sites
        site_filter !== nothing && String(get(s, "name", "")) != site_filter && continue
        (haskey(s, "geotiff") || haskey(s, "image")) || continue
        process_site(s, base, outroot; defaults = defaults, config_path = config_path)
        processed += 1
    end
    println("\n[preprocess] done — processed $processed site(s).")
    processed == 0 && @warn "No sites had a `geotiff` or `image` key (or --site filter matched none)."
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
