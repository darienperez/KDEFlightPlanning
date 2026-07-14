"""
    scripts/make_cross_site_panel.jl

Compose the cross-site planning-stage panel figure `cross_site_planning_panel.png`.

Layout (rows = study sites, columns = pipeline stages):

    ┌──────────────┬──────────────┬──────────────┬───────────┐
    │              │ (a) RGB +    │ (b) KDE      │ (c) Way-  │
    │              │ k-medoids    │ Epanechnikov │ points by │
    │              │ overlay      │ surface      │ speed     │
    ├──────────────┼──────────────┼──────────────┼───────────┤
    │ Durham, NH   │   cell       │   cell       │   cell    │  + shared
    │ Site B       │   cell       │   cell       │   cell    │    2–8 m/s
    │ Site C       │   cell       │   cell       │   cell    │    colorbar
    │ Site D       │   cell       │   cell       │   cell    │
    └──────────────┴──────────────┴──────────────┴───────────┘

Each cell is resolved from a per-site TOML config. A cell can be driven by:

  • Column (a)  — an image PNG/JPG (a screenshot, or a pre-rendered
                  `cluster/cluster_overlay_k*.png` from `run_from_config.jl`).
  • Column (b)  — an image PNG/JPG (e.g. `kde/kde_density_heatmap.png`), OR a
                  CSV grid of normalised [0,1] density (rendered viridis), OR a
                  co-registered `.tif` KDE surface (needs ArchGDAL in session).
  • Column (c)  — a waypoint CSV carrying x/y/speed columns (rendered as a
                  boustrophedon track coloured by 2–8 m/s target speed, sharing
                  one colorbar across all sites), OR a fallback image.

Any cell whose path is empty or missing is drawn as a clearly-marked
placeholder so the figure still composes while you gather Site B/C/D imagery.

Usage:
    julia --project=. scripts/make_cross_site_panel.jl [CONFIG] [OUTPUT.png]

    CONFIG   Path to the panel TOML (default: config/cross_site_panel.toml).
    OUTPUT   Output PNG path (default: cross_site_planning_panel.png in CWD).

The figure is saved at publication quality (≥ 300 dpi).

Config schema (TOML):

    speed_min = 2.0            # shared colorbar lower bound (m/s)
    speed_max = 8.0            # shared colorbar upper bound (m/s)
    speed_colormap = "cividis" # any ColorSchemes name (matches report_speed_map)
    kde_colormap   = "viridis" # column (b) grid/tif heatmaps

    [[site]]
    name  = "Durham, NH"
    col_a = "../output/durham/cluster/cluster_overlay_k1.png"
    col_b = "../output/durham/kde/kde_density_heatmap.png"
    col_c = "../data/ground_truth/E_density_aware__waypoints_xy.csv"
    # column (c) CSV column names (defaults shown):
    col_c_xcol     = "GridX"
    col_c_ycol     = "GridY"
    col_c_speedcol = "Speed"

    [[site]]
    name  = "Site B"
    col_a = ""   # ← point at your uploaded screenshot when available
    col_b = ""
    col_c = ""

Paths inside the TOML are resolved relative to the config file's directory
(same convention as `config/run_durham.toml`).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CairoMakie
using ColorSchemes
using CSV
using DataFrames
using FileIO, ImageIO, ColorTypes
using TOML

# ---------------------------------------------------------------------------
# Style (matches src/figures.jl palette)
# ---------------------------------------------------------------------------

const FIG_BG          = "#F7F4EF"   # warm off-white
const COL_AXIS        = "#444444"
const COL_PLACEHOLDER = "#E9E4DC"   # muted placeholder fill
const COL_L1          = parse(Makie.Colors.Colorant, "#1B474D")  # dark teal text

const COLUMN_HEADERS = [
    "(a) RGB + k-medoids vegetation clusters",
    "(b) Epanechnikov KDE planning surface",
    "(c) Boustrophedon waypoints by speed",
]

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

struct SiteSpec
    name         :: String
    col_a        :: String
    col_a_overlay:: String
    col_b        :: String
    col_c        :: String
    c_xcol       :: String
    c_ycol       :: String
    c_speedcol   :: String
    provisional  :: Bool
end

struct PanelConfig
    sites          :: Vector{SiteSpec}
    speed_min      :: Float64
    speed_max      :: Float64
    speed_colormap :: Symbol
    kde_colormap   :: Symbol
end

_resolve(base::AbstractString, p) =
    (p === nothing || isempty(String(p))) ? "" : abspath(joinpath(base, String(p)))

function load_panel_config(path::AbstractString)::PanelConfig
    raw  = TOML.parsefile(path)
    base = dirname(abspath(path))

    site_entries = get(raw, "site", Any[])
    isempty(site_entries) && error("Config $path has no [[site]] entries.")

    sites = SiteSpec[]
    for s in site_entries
        push!(sites, SiteSpec(
            String(get(s, "name", "(unnamed site)")),
            _resolve(base, get(s, "col_a", "")),
            _resolve(base, get(s, "col_a_overlay", "")),
            _resolve(base, get(s, "col_b", "")),
            _resolve(base, get(s, "col_c", "")),
            String(get(s, "col_c_xcol",     "GridX")),
            String(get(s, "col_c_ycol",     "GridY")),
            String(get(s, "col_c_speedcol", "Speed")),
            Bool(get(s, "provisional", false)),
        ))
    end

    return PanelConfig(
        sites,
        Float64(get(raw, "speed_min", 2.0)),
        Float64(get(raw, "speed_max", 8.0)),
        Symbol(get(raw, "speed_colormap", "cividis")),
        Symbol(get(raw, "kde_colormap",   "viridis")),
    )
end

# ---------------------------------------------------------------------------
# Cell renderers
# ---------------------------------------------------------------------------

_is_image(path)  = occursin(r"\.(png|jpg|jpeg|tif|tiff)$"i, path)
_is_csv(path)    = occursin(r"\.csv$"i, path)
_is_geotiff(path)= occursin(r"\.(tif|tiff)$"i, path)

"""
    _blank_axis!(ax) — strip an axis to a clean image frame.
"""
function _blank_axis!(ax)
    ax.aspect = DataAspect()
    ax.yreversed = true
    hidedecorations!(ax)
    ax.leftspinecolor   = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.bottomspinecolor = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.rightspinecolor  = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.topspinecolor    = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.spinewidth = 0.8
    return ax
end

"""
    _placeholder!(fig, cell, label) — draw a labelled "pending" cell.
"""
function _placeholder!(fig, cell, label::AbstractString)
    ax = Axis(fig[cell...]; backgroundcolor = parse(Makie.Colors.Colorant, COL_PLACEHOLDER))
    _blank_axis!(ax)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    text!(ax, 0.5, 0.5; text = label, align = (:center, :center),
          fontsize = 11, color = COL_L1)
    return ax
end

"""
    _provisional_stamp!(fig, cell) — overlay a "PROVISIONAL" ribbon on a cell
    whose waypoint plan is NOT a surveyed metric plan (image-space or assumed
    GSD from a non-georeferenced screenshot). Keeps the figure honest about
    scale so no reader mistakes it for a 40 m-spaced, publication-ready plan.
"""
function _provisional_stamp!(fig, cell)
    ax = Axis(fig[cell...]; backgroundcolor = :transparent)
    hidedecorations!(ax); hidespines!(ax)
    xlims!(ax, 0, 1); ylims!(ax, 0, 1)
    text!(ax, 0.5, 0.06;
          text = "PROVISIONAL — image-space / assumed GSD (not surveyed metric)",
          align = (:center, :center), fontsize = 9,
          color = (:red, 0.9), font = :bold)
    return ax
end

"""
    _render_image!(fig, cell, path) — display a raw image (screenshot or
    pre-rendered pipeline PNG) in the given layout cell.
"""
function _render_image!(fig, cell, path::AbstractString)
    img = FileIO.load(path)                     # Matrix{<:Colorant}, row 1 = top
    ax  = Axis(fig[cell...])
    _blank_axis!(ax)
    # Makie's image! expects (x, y, z) with z indexed [x, y]; permute so the
    # picture appears upright with row 1 at the top (yreversed handles flip).
    image!(ax, permutedims(img, (2, 1)))
    return ax
end

"""
    _render_geotiff_kde!(fig, cell, path, cmap) — heatmap a single-band KDE
    GeoTIFF surface. Requires ArchGDAL in the session; returns `nothing` (so
    the caller can fall back) if ArchGDAL is unavailable or the read fails.
"""
function _render_geotiff_kde!(fig, cell, path::AbstractString, cmap::Symbol)
    isdefined(Main, :ArchGDAL) || return nothing
    Z = try
        Main.ArchGDAL.read(path) do ds
            Float64.(Main.ArchGDAL.read(ds, 1))'   # → (H, W)
        end
    catch
        return nothing
    end
    ax = Axis(fig[cell...]); _blank_axis!(ax)
    heatmap!(ax, permutedims(Z); colormap = cmap, colorrange = (0.0, 1.0))
    return ax
end

"""
    _render_grid_csv!(fig, cell, path, cmap) — heatmap a CSV grid of
    normalised [0,1] KDE density (rows = grid rows, no header row of names
    required; numeric matrix).
"""
function _render_grid_csv!(fig, cell, path::AbstractString, cmap::Symbol)
    M = Matrix(CSV.read(path, DataFrame; header = false))
    Z = Float64.(M)
    ax = Axis(fig[cell...]); _blank_axis!(ax)
    heatmap!(ax, permutedims(Z); colormap = cmap, colorrange = (0.0, 1.0))
    return ax
end

"""
    _render_waypoints!(fig, cell, site, cfg) — boustrophedon waypoint track
    coloured by assigned target speed, using the SHARED [speed_min, speed_max]
    colorrange so every site's column (c) is directly comparable.

Returns the scatter plot object (for wiring the shared colorbar) or `nothing`
on failure.
"""
function _render_waypoints!(fig, cell, site::SiteSpec, cfg::PanelConfig)
    df = CSV.read(site.col_c, DataFrame)
    nm = names(df)
    xcol = site.c_xcol in nm ? site.c_xcol : first(nm)
    ycol = site.c_ycol in nm ? site.c_ycol : nm[2]
    scol = site.c_speedcol in nm ? site.c_speedcol :
           (length(nm) >= 4 ? nm[4] : last(nm))

    xs = Float64.(df[!, xcol])
    ys = Float64.(df[!, ycol])
    vs = Float64.(df[!, scol])

    ax = Axis(fig[cell...]; aspect = DataAspect())
    hidedecorations!(ax)
    ax.leftspinecolor   = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.bottomspinecolor = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.rightspinecolor  = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.topspinecolor    = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.spinewidth = 0.8

    # Faint connecting track, then speed-coloured waypoint markers on top.
    lines!(ax, xs, ys; color = (:gray50, 0.35), linewidth = 0.6)
    sc = scatter!(ax, xs, ys;
        color       = vs,
        colormap    = cfg.speed_colormap,
        colorrange  = (cfg.speed_min, cfg.speed_max),
        markersize  = 5,
        strokewidth = 0,
    )
    return sc
end

"""
    _render_cell!(fig, cell, col, site, cfg) — dispatch a single cell to the
    right renderer with graceful fallback to a placeholder. Returns the
    scatter object when a speed-coloured waypoint plot was drawn (column c),
    else `nothing`.
"""
function _render_cell!(fig, cell, col::Int, site::SiteSpec, cfg::PanelConfig)
    # Column (a) prefers a rendered k-medoids overlay when it exists, else the
    # raw screenshot / pipeline PNG.
    path = if col == 1
        (!isempty(site.col_a_overlay) && isfile(site.col_a_overlay)) ?
            site.col_a_overlay : site.col_a
    elseif col == 2
        site.col_b
    else
        site.col_c
    end
    tag  = ("a", "b", "c")[col]

    if isempty(path) || !isfile(path)
        _placeholder!(fig, cell,
            "$(site.name)\n($tag) pending — add path in config")
        col == 3 && site.provisional && _provisional_stamp!(fig, cell)
        return nothing
    end

    try
        if col == 3
            sc = nothing
            if _is_csv(path)
                sc = _render_waypoints!(fig, cell, site, cfg)
            else                      # fallback image for column (c)
                _render_image!(fig, cell, path)
            end
            site.provisional && _provisional_stamp!(fig, cell)
            return sc
        elseif col == 2
            if _is_geotiff(path)
                ax = _render_geotiff_kde!(fig, cell, path, cfg.kde_colormap)
                ax === nothing || return nothing
                # ArchGDAL absent → try image, else placeholder
                _render_image!(fig, cell, path); return nothing
            elseif _is_csv(path)
                _render_grid_csv!(fig, cell, path, cfg.kde_colormap); return nothing
            else
                _render_image!(fig, cell, path); return nothing
            end
        else                          # column (a): always an image
            _render_image!(fig, cell, path); return nothing
        end
    catch e
        @warn "Failed to render cell; drawing placeholder." site=site.name column=tag path=path exception=e
        _placeholder!(fig, cell, "$(site.name)\n($tag) render failed")
        return nothing
    end
end

# ---------------------------------------------------------------------------
# Panel composition
# ---------------------------------------------------------------------------

"""
    make_cross_site_panel(cfg; out_path, dpi=300) -> out_path

Compose and save the 4×N-site × 3-column planning panel.
"""
function make_cross_site_panel(cfg::PanelConfig;
                                out_path::AbstractString = "cross_site_planning_panel.png",
                                dpi::Real = 300)
    nsite = length(cfg.sites)

    # Layout columns: 1 = row-label gutter, 2..4 = (a)(b)(c), 5 = colorbar.
    # Layout rows:    1 = column headers, 2..(nsite+1) = site rows.
    LAB_COL, COL_A, COL_B, COL_C, CB_COL = 1, 2, 3, 4, 5
    HEADER_ROW = 1

    # Per-cell canvas ~360×300 pt → whole figure scales with site count.
    cell_w, cell_h = 360, 300
    fig = Figure(
        size = (LAB_COL_W() + 3cell_w + 90, 60 + nsite * cell_h),
        backgroundcolor = parse(Makie.Colors.Colorant, FIG_BG),
    )

    # Column headers (row 1, over columns a/b/c)
    for (j, header) in zip((COL_A, COL_B, COL_C), COLUMN_HEADERS)
        Label(fig[HEADER_ROW, j], header;
              fontsize = 13, font = :bold, halign = :center,
              color = COL_L1, padding = (2, 2, 6, 6))
    end

    speed_scatter = nothing   # captured from any column-(c) waypoint render

    for (r, site) in enumerate(cfg.sites)
        row = HEADER_ROW + r

        # Row label gutter (rotated site name)
        Label(fig[row, LAB_COL], site.name;
              fontsize = 13, font = :bold, rotation = pi/2,
              halign = :center, valign = :center,
              color = COL_L1, tellheight = false)

        for (col, cellcol) in ((1, COL_A), (2, COL_B), (3, COL_C))
            sc = _render_cell!(fig, (row, cellcol), col, site, cfg)
            sc === nothing || (speed_scatter = sc)
        end
    end

    # Shared speed colorbar spanning all site rows in the trailing column.
    # If no waypoint plot rendered, synthesise a colorbar from the config
    # range so the legend is always present and publication-complete.
    cb_rows = (HEADER_ROW + 1):(HEADER_ROW + nsite)
    if speed_scatter !== nothing
        Colorbar(fig[cb_rows, CB_COL], speed_scatter;
                 label = "Target ground speed (m/s)",
                 labelsize = 12, ticklabelsize = 10, width = 16)
    else
        Colorbar(fig[cb_rows, CB_COL];
                 colormap = cfg.speed_colormap,
                 colorrange = (cfg.speed_min, cfg.speed_max),
                 label = "Target ground speed (m/s)",
                 labelsize = 12, ticklabelsize = 10, width = 16)
    end

    # Layout proportions
    colsize!(fig.layout, LAB_COL, Fixed(LAB_COL_W()))
    colsize!(fig.layout, CB_COL,  Fixed(70))
    for c in (COL_A, COL_B, COL_C)
        colsize!(fig.layout, c, Relative((1.0) / 3))
    end
    rowgap!(fig.layout, 8)
    colgap!(fig.layout, 8)

    mkpath(dirname(abspath(out_path)))
    # px_per_unit converts Makie points (72/inch) to pixels: dpi/72 → ≥300 dpi.
    ppu = max(2.0, dpi / 72)
    CairoMakie.save(out_path, fig; px_per_unit = ppu)
    @info "Saved cross-site planning panel" out_path dpi=round(Int, ppu*72) sites=nsite
    return out_path
end

LAB_COL_W() = 46   # width (pt) of the rotated site-label gutter

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

function main()
    config_path = get(ARGS, 1, joinpath(@__DIR__, "..", "config", "cross_site_panel.toml"))
    out_path    = get(ARGS, 2, "cross_site_planning_panel.png")
    isfile(config_path) || error("Config not found: $config_path")
    cfg = load_panel_config(config_path)
    make_cross_site_panel(cfg; out_path = out_path)
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
