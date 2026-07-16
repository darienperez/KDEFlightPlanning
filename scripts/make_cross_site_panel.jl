"""
    scripts/make_cross_site_panel.jl

Compose the cross-site 3-column panel figure from the products persisted by
`scripts/make_cross_site_products.jl`. This composer OWNS all figure drawing;
the runner never writes panel PNGs.

Layout (rows = study sites, columns = fixed visual stages):

    ┌────────┬──────────────┬──────────────┬──────────────┐
    │        │ (a) Site     │ (b) Vegeta-  │ (c) KDE      │
    │        │ (RGB ortho)  │ tion label   │ planning     │
    │        │              │ (RGB+mask)   │ surface      │
    ├────────┼──────────────┼──────────────┼──────────────┤
    │ Site A │   cell       │   cell       │   cell       │  + shared
    │ Site B │   cell       │   cell       │   cell       │    [0,1] KDE
    │ Site C │   cell       │   cell       │   cell       │    colorbar
    └────────┴──────────────┴──────────────┴──────────────┘

Each site row is resolved from ONE `products_metadata.json` (written by the
runner) which references:

  • `source_geotiff`  — authoritative RGB ortho (column a, and the base for b),
  • `products.vegetation_mask`  — UInt8 0/1 selected-vegetation mask (column b overlay),
  • `products.kde_surface`      — Float64 min-max-normalised [0,1] KDE (column c).

All three rasters are read here with the package's ArchGDAL-backed I/O
(`load_rgb_geotiff`, `read_band`); the KDE column uses a single fixed [0,1]
colour range so every site is directly comparable.

Usage:
    julia --project=. scripts/make_cross_site_panel.jl [CONFIG] [OUTPUT.png]

    CONFIG   Path to the panel TOML (default: config/cross_site_panel.toml).
    OUTPUT   Output PNG path (default: cross_site_planning_panel.png in CWD).

Config schema (TOML):

    products_dir = "products"     # root holding <slug>/products_metadata.json
                                  #   (relative to the config file's directory)
    kde_colormap = "viridis"      # column (c) heatmap colormap
    mask_color   = "#FF6D00"      # column (b) vegetation overlay colour
    mask_alpha   = 0.45           # column (b) overlay opacity

    # Optional: explicit site order / subset. When omitted, every immediate
    # subdirectory of `products_dir` that contains a products_metadata.json is
    # discovered and sorted by site name.
    [[site]]
    dir  = "site-a"               # subdirectory under products_dir
    name = "Site A"               # optional label (defaults to metadata site_name)

Paths inside the TOML resolve relative to the config file's directory.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CairoMakie
using ColorTypes
using Colors: RGB, RGBA, N0f8
using JSON
using TOML

# ---------------------------------------------------------------------------
# Style (matches src/figures.jl palette)
# ---------------------------------------------------------------------------

const FIG_BG          = "#F7F4EF"   # warm off-white
const COL_AXIS        = "#444444"
const COL_PLACEHOLDER = "#E9E4DC"   # muted placeholder fill
const COL_L1          = parse(Makie.Colors.Colorant, "#1B474D")  # dark teal text

const COLUMN_HEADERS = [
    "(a) Site orthomosaic (RGB)",
    "(b) Vegetation label (RGB + selected mask)",
    "(c) Epanechnikov KDE planning surface",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

struct SiteProducts
    name   :: String
    source :: String   # RGB ortho GeoTIFF (may be "")
    mask   :: String   # vegetation_mask.tif (may be "")
    kde    :: String   # kde_surface.tif    (may be "")
end

struct PanelConfig
    sites        :: Vector{SiteProducts}
    kde_colormap :: Symbol
    mask_color   :: String
    mask_alpha   :: Float64
end

"""Read one site's products_metadata.json into a SiteProducts (paths resolved)."""
function site_from_metadata(site_dir::AbstractString; name_override = nothing)
    meta_path = joinpath(site_dir, "products_metadata.json")
    isfile(meta_path) || return nothing
    meta = try
        JSON.parsefile(meta_path)
    catch
        return nothing
    end
    name = name_override !== nothing ? String(name_override) :
           String(get(meta, "site_name", basename(site_dir)))
    products = get(meta, "products", Dict{String,Any}())
    prod(key) = let b = get(products, key, nothing)
        b === nothing ? "" : joinpath(site_dir, String(b))
    end
    source = String(get(meta, "source_geotiff", ""))
    return SiteProducts(name, source, prod("vegetation_mask"), prod("kde_surface"))
end

function load_panel_config(path::AbstractString)::PanelConfig
    raw  = TOML.parsefile(path)
    base = dirname(abspath(path))

    products_dir = abspath(joinpath(base, String(get(raw, "products_dir", "products"))))
    isdir(products_dir) || error("products_dir not found: $products_dir")

    entries = get(raw, "site", Any[])
    sites = SiteProducts[]
    if isempty(entries)
        # Auto-discover every subdir carrying a products_metadata.json.
        for d in sort(readdir(products_dir))
            sd = joinpath(products_dir, d)
            isdir(sd) || continue
            sp = site_from_metadata(sd)
            sp === nothing || push!(sites, sp)
        end
        sort!(sites; by = s -> s.name)
    else
        for s in entries
            dir = String(get(s, "dir", ""))
            isempty(dir) && error("Each [[site]] needs a `dir` (subdir under products_dir).")
            sd = joinpath(products_dir, dir)
            sp = site_from_metadata(sd; name_override = get(s, "name", nothing))
            sp === nothing && error("No products_metadata.json under $sd")
            push!(sites, sp)
        end
    end
    isempty(sites) && error("No sites with products_metadata.json under $products_dir")

    return PanelConfig(
        sites,
        Symbol(get(raw, "kde_colormap", "viridis")),
        String(get(raw, "mask_color", "#FF6D00")),
        Float64(get(raw, "mask_alpha", 0.45)),
    )
end

# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

function _blank_axis!(ax)
    ax.aspect = DataAspect()
    ax.yreversed = true
    hidedecorations!(ax)
    for f in (:leftspinecolor, :bottomspinecolor, :rightspinecolor, :topspinecolor)
        setproperty!(ax, f, parse(Makie.Colors.Colorant, COL_AXIS))
    end
    ax.spinewidth = 0.8
    return ax
end

function _placeholder!(fig, cell, label::AbstractString)
    ax = Axis(fig[cell...]; backgroundcolor = parse(Makie.Colors.Colorant, COL_PLACEHOLDER))
    _blank_axis!(ax)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    text!(ax, 0.5, 0.5; text = label, align = (:center, :center),
          fontsize = 11, color = COL_L1)
    return ax
end

# image!(ax, xr, yr, M): M is indexed [x, y]; our rasters are (H, W)=(y, x), so
# permute to (W, H). Combined with yreversed the picture reads north-up.
_for_image(M::AbstractMatrix) = permutedims(M, (2, 1))

# ---------------------------------------------------------------------------
# Cell renderers
# ---------------------------------------------------------------------------

"""Column (a): RGB orthomosaic from the source GeoTIFF."""
function _render_site!(fig, cell, site::SiteProducts)
    (isempty(site.source) || !isfile(site.source)) &&
        return _placeholder!(fig, cell, "$(site.name)\nsource GeoTIFF missing")
    rs  = load_rgb_geotiff(site.source)
    ax  = Axis(fig[cell...]); _blank_axis!(ax)
    H, W = size(rs.Z)
    image!(ax, (0, W), (0, H), _for_image(rs.Z); interpolate = false)
    return ax
end

"""Column (b): RGB base with the selected-vegetation mask overlaid."""
function _render_label!(fig, cell, site::SiteProducts, cfg::PanelConfig)
    (isempty(site.source) || !isfile(site.source)) &&
        return _placeholder!(fig, cell, "$(site.name)\nsource GeoTIFF missing")
    rs   = load_rgb_geotiff(site.source)
    ax   = Axis(fig[cell...]); _blank_axis!(ax)
    H, W = size(rs.Z)
    image!(ax, (0, W), (0, H), _for_image(rs.Z); interpolate = false)

    if !isempty(site.mask) && isfile(site.mask)
        Zm, _, _ = read_band(site.mask)                 # (Hm, Wm) UInt8 0/1 (255 nodata)
        rgba = parse(RGBA{Float64}, cfg.mask_color)
        fill = RGBA{Float64}(rgba.r, rgba.g, rgba.b, cfg.mask_alpha)
        clear = RGBA{Float64}(0, 0, 0, 0)
        overlay = [ (v == 1) ? fill : clear for v in Zm ] # matches (Hm, Wm)
        image!(ax, (0, W), (0, H), _for_image(overlay); interpolate = false)
    end
    return ax
end

"""Column (c): KDE surface heatmap with the shared fixed [0,1] colour range."""
function _render_kde!(fig, cell, site::SiteProducts, cfg::PanelConfig)
    (isempty(site.kde) || !isfile(site.kde)) &&
        return (nothing, _placeholder!(fig, cell, "$(site.name)\nKDE surface missing"))
    Zk, _, _ = read_band(site.kde)                       # (H, W) Float64, NaN nodata
    ax = Axis(fig[cell...]); _blank_axis!(ax)
    hm = heatmap!(ax, _for_image(Float64.(Zk));
                  colormap = cfg.kde_colormap, colorrange = (0.0, 1.0))
    return (hm, ax)
end

# ---------------------------------------------------------------------------
# Panel composition
# ---------------------------------------------------------------------------

const LAB_COL_W = 46   # width (pt) of the rotated site-label gutter

"""
    make_cross_site_panel(cfg; out_path, dpi=300) -> out_path

Compose and save the N-site × 3-column products panel.
"""
function make_cross_site_panel(cfg::PanelConfig;
                               out_path::AbstractString = "cross_site_planning_panel.png",
                               dpi::Real = 300)
    nsite = length(cfg.sites)

    LAB_COL, COL_A, COL_B, COL_C, CB_COL = 1, 2, 3, 4, 5
    HEADER_ROW = 1

    cell_w, cell_h = 340, 300
    fig = Figure(
        size = (LAB_COL_W + 3cell_w + 90, 60 + nsite * cell_h),
        backgroundcolor = parse(Makie.Colors.Colorant, FIG_BG),
    )

    for (j, header) in zip((COL_A, COL_B, COL_C), COLUMN_HEADERS)
        Label(fig[HEADER_ROW, j], header;
              fontsize = 13, font = :bold, halign = :center,
              color = COL_L1, padding = (2, 2, 6, 6))
    end

    kde_heatmap = nothing
    for (r, site) in enumerate(cfg.sites)
        row = HEADER_ROW + r
        Label(fig[row, LAB_COL], site.name;
              fontsize = 13, font = :bold, rotation = pi/2,
              halign = :center, valign = :center,
              color = COL_L1, tellheight = false)

        _render_site!(fig,  (row, COL_A), site)
        _render_label!(fig, (row, COL_B), site, cfg)
        hm, _ = _render_kde!(fig, (row, COL_C), site, cfg)
        hm === nothing || (kde_heatmap = hm)
    end

    cb_rows = (HEADER_ROW + 1):(HEADER_ROW + nsite)
    if kde_heatmap !== nothing
        Colorbar(fig[cb_rows, CB_COL], kde_heatmap;
                 label = "Normalised KDE density (min–max, [0,1])",
                 labelsize = 12, ticklabelsize = 10, width = 16)
    else
        Colorbar(fig[cb_rows, CB_COL];
                 colormap = cfg.kde_colormap, colorrange = (0.0, 1.0),
                 label = "Normalised KDE density (min–max, [0,1])",
                 labelsize = 12, ticklabelsize = 10, width = 16)
    end

    colsize!(fig.layout, LAB_COL, Fixed(LAB_COL_W))
    colsize!(fig.layout, CB_COL,  Fixed(70))
    for c in (COL_A, COL_B, COL_C)
        colsize!(fig.layout, c, Relative(1.0 / 3))
    end
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 8)

    mkpath(dirname(abspath(out_path)))
    ppu = max(2.0, dpi / 72)
    CairoMakie.save(out_path, fig; px_per_unit = ppu)
    @info "Saved cross-site panel" out_path dpi=round(Int, ppu*72) sites=nsite
    return out_path
end

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

function main()
    config_path = abspath(get(ARGS, 1,
        joinpath(@__DIR__, "..", "config", "cross_site_panel.toml")))
    out_path    = get(ARGS, 2, "cross_site_planning_panel.png")
    isfile(config_path) || error("Config not found: $config_path")
    cfg = load_panel_config(config_path)
    make_cross_site_panel(cfg; out_path = out_path)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
