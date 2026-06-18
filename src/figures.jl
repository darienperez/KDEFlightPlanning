"""
    figures.jl — Publication-ready figure generation using CairoMakie

All functions accept CSV DataFrames (pre-loaded) and optional count-grid
dictionaries (for heatmap figures). They return the output file path(s).

## Palette (from project design spec)
  Const 2 m/s  : rust   #A84B2F
  KDE-guided   : teal   #20808D
  Const 8 m/s  : dark   #1B474D

## Typical usage
  See `scripts/make_figures.jl`.
"""

# ============================================================
# Palette + style helpers
# ============================================================

const FIG_BG        = "#F7F4EF"   # warm off-white
const COL_CONST2    = "#A84B2F"   # Const 2 m/s rust
const COL_KDE       = "#20808D"   # KDE-guided teal
const COL_CONST8    = "#1B474D"   # Const 8 m/s dark teal
const COL_AXIS      = "#444444"
const COL_GRID      = "#DDDAD3"

# Mission display labels (user-facing)
const MISSION_COLORS = Dict(
    "Const. 2 m/s"              => COL_CONST2,
    "KDE-guided (Epanechnikov)" => COL_KDE,
    "Const. 8 m/s"              => COL_CONST8,
)

const MISSION_SHORT = Dict(
    "Const. 2 m/s"              => "Const. 2 m/s",
    "KDE-guided (Epanechnikov)" => "KDE-guided",
    "Const. 8 m/s"              => "Const. 8 m/s",
)

# Canonical mission order for consistent axis labels
const MISSION_ORDER = ["Const. 2 m/s", "KDE-guided (Epanechnikov)", "Const. 8 m/s"]

function _theme_axis!(ax; xlabel="", ylabel="", title="")
    ax.backgroundcolor = :transparent
    ax.xgridcolor      = parse(Makie.Colors.Colorant, COL_GRID)
    ax.ygridcolor      = parse(Makie.Colors.Colorant, COL_GRID)
    ax.xgridwidth      = 0.8
    ax.ygridwidth      = 0.8
    ax.spinewidth      = 0.7
    ax.leftspinecolor  = parse(Makie.Colors.Colorant, COL_AXIS)
    ax.bottomspinecolor= parse(Makie.Colors.Colorant, COL_AXIS)
    ax.rightspinevisible = false
    ax.topspinevisible   = false
    ax.xlabelsize = 11
    ax.ylabelsize = 11
    ax.xticklabelsize = 9
    ax.yticklabelsize = 9
    ax.titlesize = 12
    isempty(xlabel) || (ax.xlabel = xlabel)
    isempty(ylabel) || (ax.ylabel = ylabel)
    isempty(title)  || (ax.title  = title)
end

# ============================================================
# Figure 1: Forest line scan differences (bar chart)
# ============================================================

"""
    fig_forest_line_scan_differences(cover_df; out_dir, formats) → paths

Horizontal diverging bar chart of KDE-guided CR minus each constant-speed
baseline, for forest cover classes (Coniferous, Deciduous) across all lines.
Sorted by KDE–2 difference descending.

`cover_df` is the DataFrame from `actual_trajectory_line_cover_metrics.csv`.
"""
function fig_forest_line_scan_differences(
    cover_df::DataFrame;
    out_dir::AbstractString = ".",
    formats::Vector{String} = ["png", "pdf"],
)
    mkpath(out_dir)

    # Build wide-form differences
    missions = Dict(row.mission => row for row in eachrow(cover_df) if false)  # placeholder
    # Pivot: for each (line, cover) grab CR by mission
    forest = filter(r -> r.cover ∈ ["Coniferous", "Deciduous"], cover_df)
    
    # Group by line_number, cover → get CR per mission
    rows_out = NamedTuple[]
    for (key, g) in pairs(groupby(forest, [:line_number, :cover]))
        cr = Dict(r.mission => r.coverage_ratio for r in eachrow(g))
        haskey(cr, "KDE-guided (Epanechnikov)") || continue
        haskey(cr, "Const. 2 m/s") || continue
        haskey(cr, "Const. 8 m/s") || continue
        push!(rows_out, (
            line_number   = key.line_number,
            cover         = key.cover,
            kde_vs_c2     = (cr["KDE-guided (Epanechnikov)"] - cr["Const. 2 m/s"]) * 100,
            kde_vs_c8     = (cr["KDE-guided (Epanechnikov)"] - cr["Const. 8 m/s"]) * 100,
        ))
    end
    wide = DataFrame(rows_out)
    sort!(wide, :kde_vs_c2; rev=true)

    # Y labels
    ylabels = ["Line $(r.line_number) — $(r.cover)" for r in eachrow(wide)]
    n = nrow(wide)
    ys = collect(1:n)

    # color by cover: Coniferous = rust, Deciduous = teal
    bar_colors = [r.cover == "Coniferous" ? COL_CONST2 : COL_KDE for r in eachrow(wide)]

    fig = Figure(
        size = (1100, 540),
        backgroundcolor = FIG_BG,
    )

    suptitle = Label(fig[0, 1:2],
        "Forest-cover line scan using actual trajectory overlap\nEach line uses the intersection of actual flown x-ranges across the three missions; no arbitrary edge trim is applied.",
        fontsize = 13, font = :bold,
        halign = :left,
        padding = (8, 0, 4, 4),
    )

    ax1 = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    _theme_axis!(ax1;
        xlabel = "CR difference (percentage points)",
        title = "KDE-guided minus Const. 2 m/s",
    )
    _theme_axis!(ax2;
        xlabel = "CR difference (percentage points)",
        title = "KDE-guided minus Const. 8 m/s",
    )

    for (ax, vals) in [(ax1, wide.kde_vs_c2), (ax2, wide.kde_vs_c8)]
        barplot!(ax, ys, vals;
            direction = :x,
            color = bar_colors,
            bar_labels = nothing,
            strokewidth = 0,
        )
        vlines!(ax, [0.0]; color = "#222222", linewidth = 1.2)
        ax.yticks = (ys, ylabels)
        ax.yreversed = true
        ax.ylabelvisible = false
        ax.ygridvisible = false
        ax.yticklabelsize = 10
        ax.xticklabelsize = 9
    end

    # Shared x-scale: expand to max of both panels
    xmax = max(maximum(abs.(wide.kde_vs_c2)), maximum(abs.(wide.kde_vs_c8))) * 1.1
    xlims!(ax1, -xmax, xmax)
    xlims!(ax2, -xmax, xmax)

    # Remove ytick labels from ax2 (keep only ax1 labels)
    ax2.yticklabelsvisible = false
    ax2.yticksvisible = false

    # Legend patches
    elem_c = [PolyElement(color=COL_CONST2, strokecolor=:transparent),
              PolyElement(color=COL_KDE,    strokecolor=:transparent)]
    Legend(fig[1, 3], elem_c, ["Coniferous", "Deciduous"];
        framevisible = false,
        labelsize = 10,
        patchsize = (14, 10),
    )

    colgap!(fig.layout, 1, 12)
    colgap!(fig.layout, 2, 6)

    colsize!(fig.layout, 1, Relative(0.45))
    colsize!(fig.layout, 2, Relative(0.45))
    colsize!(fig.layout, 3, Relative(0.10))

    paths = String[]
    for fmt in formats
        p = joinpath(out_dir, "actual_trajectory_line_scan_forest_differences.$fmt")
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    return paths
end


# ============================================================
# Figure 2: Line 2 actual-overlap detail
# ============================================================

"""
    fig_line2_overlap_detail(
        cover2_df, profile2_df, ext2_df,
        count_grids, processed;
        out_dir, formats
    ) → paths

Figure for Line 2:
  Three count-grid heatmaps (one per mission) over the shared actual
  trajectory overlap window. This intentionally omits the former
  right-hand diagnostic panels so the map can be combined with the
  line-level return-distribution/CR figure in the manuscript.

`count_grids` is the Dict from `load_count_grids`.
`processed` is Dict{String, DataFrame} of cleaned survey trajectories.
"""
function fig_line2_overlap_detail(
    cover2_df::DataFrame,
    profile2_df::DataFrame,
    ext2_df::DataFrame,
    count_grids::Dict,
    processed::Dict{String,DataFrame};
    out_dir::AbstractString = ".",
    formats::Vector{String} = ["png", "pdf"],
    half_band_rows::Int = 20,
)
    mkpath(out_dir)
    using_cm = @isdefined(CairoMakie)

    # ---- geographic reference from extents ----
    # count-grid origin (Durham site)
    XMIN = 341301.0
    YMIN = 4774619.0
    DX   = 1.0
    DY   = 1.0

    # shared actual overlap x-range for line 2
    x0_shared = maximum(ext2_df.x_min_actual)
    x1_shared = minimum(ext2_df.x_max_actual)

    # row center for line 2 (from extents)
    y_med = median(ext2_df.y_median_actual)
    row_center = round(Int, (y_med - YMIN) / DY)

    r0 = max(1, row_center - half_band_rows)
    r1 = min(SCAN_NROWS, row_center + half_band_rows)

    # column range for the band
    col0 = max(1, round(Int, (x0_shared - XMIN) / DX))
    col1 = min(SCAN_NCOLS, round(Int, (x1_shared - XMIN) / DX))
    col0_full = 1; col1_full = SCAN_NCOLS   # full extent for heatmaps

    mission_order = ["Const. 2 m/s", "KDE-guided (Epanechnikov)", "Const. 8 m/s"]
    mission_keys  = [("const2","NA"), ("density","E"), ("const8","NA")]

    # Retrieve count grids for line 2 (ground returns, all covers)
    # We need the full 2-D grid summed across cover classes
    function get_ground_grid(mk)
        m, k = mk
        covers = ["field", "decid", "conif"]
        mats = [get(count_grids, ("ground", c, m, k), nothing) for c in covers]
        mats = filter(!isnothing, mats)
        isempty(mats) && return zeros(SCAN_NROWS, SCAN_NCOLS)
        return sum(mats)
    end

    grids = [get_ground_grid(mk) for mk in mission_keys]

    # geographic axes for heatmap
    xs_geo = XMIN .+ (0:SCAN_NCOLS-1) .* DX    # length SCAN_NCOLS
    ys_geo = YMIN .+ (0:SCAN_NROWS-1) .* DY    # row 0 = south → row NROWS-1 = north

    # slice to band rows
    xs_band = xs_geo
    ys_band = ys_geo[r0:r1]

    # ---- Build figure ----
    fig = Figure(size=(1200, 760), backgroundcolor=FIG_BG)

    title_label = Label(fig[0, 1:3],
        "Line 2 actual-trajectory overlap comparison\n" *
        "Coverage envelope = ±$(half_band_rows) count-grid rows; " *
        "x-domain is the common sustained-survey trajectory range " *
        "($(round(x0_shared, digits=1))–$(round(x1_shared, digits=1)) m).",
        fontsize = 12, font = :bold,
        halign = :left,
        padding = (8, 0, 4, 4),
    )

    caption = Label(fig[4, 1:3],
        "Dashed red rectangles mark the common overlap window. " *
        "Pale trajectory traces with dark halos show cleaned sustained-survey samples assigned to Line 2.",
        fontsize = 8,
        halign = :left,
        padding = (4, 4, 2, 2),
    )

    hm_axes = Axis[]
    for (i, (mission, mk)) in enumerate(zip(mission_order, mission_keys))
        row_fig = i  # rows 1,2,3
        ax = Axis(fig[row_fig, 1:2];
            title = mission,
            titlesize = 10,
            aspect = DataAspect(),
            ylabel = "northing (m)",
            xlabel = i == 3 ? "easting (m)" : "",
            xlabelsize = 9, ylabelsize = 9,
            xticklabelsize = 8, yticklabelsize = 8,
        )
        _theme_axis!(ax)

        # heatmap: log-scale counts
        mat = grids[i][r0:r1, :]  # (band_rows × NCOLS)
        mat_plot = copy(mat)
        mat_plot[mat_plot .== 0] .= NaN
        mat_log = log10.(max.(mat_plot, 1e-3))
        mat_log[isnan.(mat_plot)] .= NaN
        # CairoMakie heatmap: pass xs (length NCOLS), ys (length band_rows), matrix (NCOLS × band_rows)
        # The matrix is currently (band_rows × NCOLS); need to transpose for heatmap(x, y, z) convention
        hm = heatmap!(ax, xs_geo, ys_band, mat_log';
            colormap = :deep,
            colorrange = (0, 3),
            nan_color = RGBAf(0, 0, 0, 0),
        )
        if i == 2
            Colorbar(fig[1:3, 3], hm;
                label = "log₁₀(ground-return count per 1 m² cell)",
                labelsize = 9,
                ticklabelsize = 8,
                width = 14,
                ticks = ([0, 1, 2, 3], ["1", "10", "100", "1000"]),
            )
        end

        # Overlay actual overlap box
        xmin_box, xmax_box = x0_shared, x1_shared
        ymin_box = ys_band[1]
        ymax_box = ys_band[end]
        lines!(ax,
            [xmin_box, xmax_box, xmax_box, xmin_box, xmin_box],
            [ymin_box, ymin_box, ymax_box, ymax_box, ymin_box];
            color = "#DD2222", linewidth = 1.2, linestyle = :dash,
        )

        # Trajectory overlay.  A dark halo plus pale foreground line keeps
        # the flightline legible over both low- and high-return cells.
        if haskey(processed, mission)
            tdf = processed[mission]
            line2 = filter(r -> r.nearest_line == 1, tdf)  # line_index=1 → line 2
            if nrow(line2) > 0
                sort!(line2, :GridX)
                lines!(ax, line2.GridX, line2.GridY;
                    color = (:black, 0.72), linewidth = 2.6,
                )
                lines!(ax, line2.GridX, line2.GridY;
                    color = "#FFF2B8", linewidth = 1.45,
                )
            end
        end

        push!(hm_axes, ax)
    end

    # Layout tweaks
    rowgap!(fig.layout, 4, 2)
    colgap!(fig.layout, 2, 4)
    rowsize!(fig.layout, 1, Relative(0.28))
    rowsize!(fig.layout, 2, Relative(0.28))
    rowsize!(fig.layout, 3, Relative(0.28))

    paths = String[]
    for fmt in formats
        p = joinpath(out_dir, "line2_actual_overlap_detail.$fmt")
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    return paths
end


# ============================================================
# Figure 3: Tracking metrics summary
# ============================================================

"""
    fig_tracking_metrics(
        summary_df, line_df, processed;
        out_dir, formats
    ) → paths

Three-panel tracking figure:
  Left:        cleaned survey-leg trajectory map (all 3 missions)
  Top-right:   speed distribution boxplot by mission
  Bottom-right: line-following stability (p95 and RMS horizontal bars)

`summary_df`  ← `tracking_summary_metrics_refined.csv`
`line_df`     ← `tracking_line_metrics_refined.csv`
`processed`   ← Dict{String,DataFrame} of cleaned trajectories
"""
function fig_tracking_metrics(
    summary_df::DataFrame,
    line_df::DataFrame,
    processed::Dict{String,DataFrame};
    out_dir::AbstractString = ".",
    formats::Vector{String} = ["png", "pdf"],
)
    mkpath(out_dir)

    mission_order = MISSION_ORDER

    fig = Figure(size=(1200, 680), backgroundcolor=FIG_BG)

    suptitle = Label(fig[0, 1:2],
        "Tracking performance after survey-leg cleaning\n" *
        "Calibration and turns removed using planned waypoint geometry: " *
        "near horizontal leg center, within leg x-range, predominantly east-west motion, " *
        "and sustained segment duration/span.",
        fontsize = 11, font = :bold,
        halign = :left,
        padding = (8, 0, 4, 4),
    )

    footnote = Label(fig[4, 1:2],
        "Tracking error is measured around each mission's fitted horizontal leg center; " *
        "planned-line offset is saved separately. For a final paper figure, " *
        "thresholds should be frozen and reported or replaced with a deterministic waypoint-segment matcher.",
        fontsize = 8, halign = :left, padding = (4, 4, 2, 2),
    )

    # ---- Left panel: trajectory map ----
    ax_map = Axis(fig[1:3, 1];
        title = "Cleaned survey-leg trajectories",
        titlesize = 11,
        xlabel = "easting (m)",
        ylabel = "northing (m)",
        xlabelsize = 9, ylabelsize = 9,
        xticklabelsize = 8, yticklabelsize = 8,
    )
    _theme_axis!(ax_map)

    legend_elems_map = []
    for mission in mission_order
        haskey(processed, mission) || continue
        df = processed[mission]
        mc = parse(Makie.Colors.Colorant, MISSION_COLORS[mission])
        # Plot each segment separately to avoid connecting turns
        first_seg = true
        for seg in sort(unique(df.segment_id))
            seg_df = df[df.segment_id .== seg, :]
            lines!(ax_map, seg_df.GridX, seg_df.GridY;
                color = (mc, 0.55), linewidth = 0.55,
            )
        end
        push!(legend_elems_map, LineElement(color=mc, linewidth=1.8))
    end

    # Line index labels: pick one mission to annotate
    first_m = "Const. 2 m/s"
    if haskey(processed, first_m)
        df_first = processed[first_m]
        for lid in sort(unique(df_first.nearest_line))
            g = df_first[df_first.nearest_line .== lid, :]
            x_label = minimum(g.GridX) - 8
            y_label = median(g.GridY)
            text!(ax_map, "$(lid)"; position=(x_label, y_label),
                  fontsize=8, align=(:right, :center), color=:gray40)
        end
    end

    axislegend(ax_map,
        legend_elems_map,
        [MISSION_SHORT[m] for m in mission_order if haskey(processed, m)];
        position = :lt, labelsize=9, framevisible=true,
        framecolor = "#BBBBBB",
        patchsize=(16, 8),
    )

    # ---- Top-right: speed boxplot ----
    ax_spd = Axis(fig[1:2, 2];
        title = "Ground-speed distribution after cleaning",
        titlesize = 11,
        ylabel = "ground speed (m/s)",
        xlabelsize = 9, ylabelsize = 9,
        xticklabelsize = 9, yticklabelsize = 8,
    )
    _theme_axis!(ax_spd)
    ax_spd.xgridvisible = false

    xtick_pos   = Float64[]
    xtick_label = String[]
    for (xi, mission) in enumerate(mission_order)
        haskey(processed, mission) || continue
        df = processed[mission]
        speeds = df.ground_speed_mps
        speeds = filter(isfinite, speeds)
        mc = parse(Makie.Colors.Colorant, MISSION_COLORS[mission])
        push!(xtick_pos, Float64(xi))
        push!(xtick_label, MISSION_SHORT[mission])

        q1, med, q3 = quantile(speeds, [0.25, 0.5, 0.75])
        lo = max(minimum(speeds), q1 - 1.5*(q3-q1))
        hi = min(maximum(speeds), q3 + 1.5*(q3-q1))

        # box
        poly!(ax_spd, Rect(xi - 0.3, q1, 0.6, q3 - q1); color=(mc, 0.6), strokecolor=:black, strokewidth=0.8)
        # median line
        lines!(ax_spd, [xi-0.3, xi+0.3], [med, med]; color=:black, linewidth=1.5)
        # whiskers
        lines!(ax_spd, [xi, xi], [lo, q1]; color=:black, linewidth=0.9)
        lines!(ax_spd, [xi, xi], [q3, hi]; color=:black, linewidth=0.9)
        lines!(ax_spd, [xi-0.15, xi+0.15], [lo, lo]; color=:black, linewidth=0.9)
        lines!(ax_spd, [xi-0.15, xi+0.15], [hi, hi]; color=:black, linewidth=0.9)
    end

    ax_spd.xticks = (xtick_pos, xtick_label)

    # Dashed reference lines for 2 and 8 m/s
    hlines!(ax_spd, [2.0, 8.0]; color=(parse(Makie.Colors.Colorant, "#888888"), 0.7),
            linewidth=0.9, linestyle=:dash)

    # ---- Bottom-right: tracking error bars ----
    ax_trk = Axis(fig[3, 2];
        title = "Line-following stability after cleaning",
        titlesize = 11,
        xlabel = "tracking error about fitted leg center (m)",
        xlabelsize = 9, ylabelsize = 9,
        xticklabelsize = 8, yticklabelsize = 9,
    )
    _theme_axis!(ax_trk)
    ax_trk.ygridvisible = false

    bar_h = 0.18
    missions_rev = reverse(mission_order)
    for (mi, mission) in enumerate(missions_rev)
        row = filter(r -> r.mission == mission, summary_df)
        isempty(row) && continue
        r = row[1, :]
        mc = parse(Makie.Colors.Colorant, MISSION_COLORS[mission])
        y_center = Float64(mi)

        # RMS bar (lighter)
        rms = r.tracking_error_rms_m
        barplot!(ax_trk, [y_center + bar_h/2 + 0.01], [rms];
            direction = :x, color = (mc, 0.55),
            width = bar_h, strokewidth = 0,
        )
        text!(ax_trk, "$(round(rms, digits=2))";
            position = (rms + 0.04, y_center + bar_h/2 + 0.01),
            fontsize=9, align=(:left, :center),
        )

        # P95 bar (full opacity)
        p95 = r.tracking_error_p95_m
        barplot!(ax_trk, [y_center - bar_h/2 - 0.01], [p95];
            direction = :x, color = mc,
            width = bar_h, strokewidth = 0,
        )
        text!(ax_trk, "$(round(p95, digits=2))";
            position = (p95 + 0.04, y_center - bar_h/2 - 0.01),
            fontsize=9, align=(:left, :center),
        )
    end

    ax_trk.yticks = (collect(1.0:length(missions_rev)), [MISSION_SHORT[m] for m in missions_rev])
    ax_trk.yticklabelsize = 10
    ylims!(ax_trk, 0.3, length(missions_rev) + 0.7)

    # Legend for bar types
    legend_bar = [
        PolyElement(color=(:black, 0.7), strokecolor=:transparent),
        PolyElement(color=(:black, 0.4), strokecolor=:transparent),
    ]
    Legend(fig[3, 2], legend_bar, ["95th percentile", "RMS"];
        tellwidth=false, tellheight=false,
        halign=:right, valign=:top,
        framevisible=true, framecolor="#CCCCCC",
        labelsize=9, patchsize=(12,8),
        margin=(4,4,4,4),
    )

    # Layout
    rowgap!(fig.layout, 3, 4)
    colgap!(fig.layout, 1, 12)
    colsize!(fig.layout, 1, Relative(0.52))
    colsize!(fig.layout, 2, Relative(0.48))

    paths = String[]
    for fmt in formats
        p = joinpath(out_dir, "tracking_metrics_refined.$fmt")
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    return paths
end


# ============================================================
# Figure 4 (optional): Whole-cover bootstrap CI summary
# ============================================================

"""
    fig_bootstrap_ci_summary(bootstrap_csv; out_dir, formats) → paths

Summary bar chart with CIs from bootstrap output CSV, if available.
`bootstrap_csv` should have columns: cover, mission, cr_mean, ci_lo, ci_hi.
"""
function fig_bootstrap_ci_summary(
    bootstrap_csv::AbstractString;
    out_dir::AbstractString = ".",
    formats::Vector{String} = ["png"],
)
    mkpath(out_dir)
    isfile(bootstrap_csv) || return String[]

    df = CSV.read(bootstrap_csv, DataFrame)
    needed = [:cover, :mission, :cr_mean, :ci_lo, :ci_hi]
    all(c -> hasproperty(df, c), needed) || return String[]

    covers_order = ["Field", "Deciduous", "Coniferous"]
    n_covers  = length(covers_order)
    n_missions = length(MISSION_ORDER)
    bar_h = 0.22

    fig = Figure(size=(900, 400), backgroundcolor=FIG_BG)
    ax = Axis(fig[1, 1];
        title = "Whole-cover CR with block-bootstrap 95% CI (n=5000, block_frac=3%, seed=42)",
        titlesize = 11,
        xlabel = "Coverage Ratio",
        xlabelsize = 9, ylabelsize = 9,
        xticklabelsize = 8, yticklabelsize = 9,
    )
    _theme_axis!(ax)
    ax.ygridvisible = false
    ax.xtickformat = vs -> ["$(round(Int, v*100))%" for v in vs]

    ytick_positions = Float64[]
    ytick_labels    = String[]
    for (ci, cover) in enumerate(covers_order)
        y_center = Float64(ci) * (n_missions * (bar_h + 0.05) + 0.3)
        for (mi, mission) in enumerate(MISSION_ORDER)
            row = filter(r -> r.cover == cover && r.mission == mission, df)
            isempty(row) && continue
            r = row[1, :]
            mc = parse(Makie.Colors.Colorant, MISSION_COLORS[mission])
            y_pos = y_center + (mi - 2) * (bar_h + 0.04)
            barplot!(ax, [y_pos], [r.cr_mean]; direction=:x, color=mc, width=bar_h, strokewidth=0)
            errorbars!(ax, [r.cr_mean], [y_pos], [r.cr_mean - r.ci_lo], [r.ci_hi - r.cr_mean];
                direction=:x, whiskerwidth=4, color=:black, linewidth=1.0)
        end
        push!(ytick_positions, y_center)
        push!(ytick_labels, cover)
    end

    ax.yticks = (ytick_positions, ytick_labels)

    legend_elems = [PolyElement(color=parse(Makie.Colors.Colorant, MISSION_COLORS[m]), strokecolor=:transparent) for m in MISSION_ORDER]
    axislegend(ax, legend_elems, [MISSION_SHORT[m] for m in MISSION_ORDER];
        position=:rb, labelsize=9, framevisible=false, patchsize=(12,8))

    paths = String[]
    for fmt in formats
        p = joinpath(out_dir, "bootstrap_cr_summary.$fmt")
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    return paths
end


# ============================================================
# Figure 5: Whole-cover CR + bootstrap CI (manuscript main figure)
# ============================================================

"""
    fig_bootstrap_cr_main(
        cr_csv        :: AbstractString,  # bootstrap_cr_cis_all36.csv
        diff_csv      :: AbstractString;  # bootstrap_cr_difference_cis.csv
        out_dir       :: AbstractString = ".",
        formats       :: Vector{String}  = ["png", "pdf"],
        return_filter :: String          = "Ground",
    ) → paths

Publication-ready figure: ground-return CR by cover for KDE-guided
Epanechnikov, Const 2 m/s, and Const 8 m/s, with 95% block-bootstrap CI bars.
Below the main panel, a second panel shows paired CR differences with CIs
(deciduous significant positive; coniferous wide CI; field saturation).

`cr_csv`   — path to bootstrap_cr_cis_all36.csv
`diff_csv` — path to bootstrap_cr_difference_cis.csv
"""
function fig_bootstrap_cr_main(
    cr_csv   ::AbstractString,
    diff_csv ::AbstractString;
    out_dir       ::AbstractString = ".",
    formats       ::Vector{String} = ["png", "pdf"],
    return_filter ::String         = "Ground",
)
    mkpath(out_dir)
    (isfile(cr_csv) && isfile(diff_csv)) || begin
        @warn "Bootstrap CSVs not found — skipping fig_bootstrap_cr_main"
        return String[]
    end

    cr_df   = CSV.read(cr_csv,   DataFrame)
    diff_df = CSV.read(diff_csv, DataFrame)

    # ── filter to selected return type ──────────────────────────────────────
    sub_cr = filter(r ->
        r.Return  == return_filter &&
        r.Mission == "Density-aware" && r.Kernel == "E" ||
        r.Mission == "Const 2 m/s"  ||
        r.Mission == "Const 8 m/s",
        cr_df
    )
    sub_cr = filter(r -> r.Return == return_filter, sub_cr)

    sub_diff = filter(r -> r.Return == return_filter, diff_df)

    covers_order   = ["Field", "Deciduous", "Coniferous"]
    mission_groups = [
        ("Const 2 m/s",    "——",  "Const. 2 m/s",              COL_CONST2),
        ("Density-aware",  "E",   "KDE-guided (Epanechnikov)",  COL_KDE),
        ("Const 8 m/s",    "——",  "Const. 8 m/s",              COL_CONST8),
    ]

    n_cov    = length(covers_order)
    n_miss   = length(mission_groups)
    bar_w    = 0.22
    gap      = 0.08
    grp_gap  = 0.18

    # ── figure layout ───────────────────────────────────────────────────────
    fig = Figure(
        size = (900, 640),
        backgroundcolor = FIG_BG,
    )

    ax_cr = Axis(fig[1, 1];
        ylabel     = "$return_filter-return Coverage Ratio",
        xlabelsize = 10, ylabelsize = 11,
        xticklabelsize = 10, yticklabelsize = 9,
        title = "Whole-cover CR with block-bootstrap 95% CI\n(n=5000, block_frac=3%, seed=42)",
        titlesize = 11,
    )
    _theme_axis!(ax_cr)
    ax_cr.xgridvisible = false
    ax_cr.ygridvisible = true
    ax_cr.ytickformat  = vs -> ["$(round(Int, v*100))%" for v in vs]

    ax_diff = Axis(fig[2, 1];
        ylabel     = "CR difference (KDE-guided − baseline)",
        xlabel     = "Cover class",
        xlabelsize = 10, ylabelsize = 11,
        xticklabelsize = 10, yticklabelsize = 9,
        titlesize = 10,
        title = "Paired CR difference with 95% CI",
    )
    _theme_axis!(ax_diff)
    ax_diff.xgridvisible = false
    ax_diff.ygridvisible = true
    ax_diff.ytickformat  = vs -> ["$(round(v*100, digits=1)) pp" for v in vs]

    # ── draw CR bars (top panel) ─────────────────────────────────────────
    xtick_positions = Float64[]
    xtick_labels    = String[]

    for (ci, cover) in enumerate(covers_order)
        x_center = Float64(ci) * (n_miss * (bar_w + gap) + grp_gap)
        push!(xtick_positions, x_center)
        push!(xtick_labels, cover)

        for (mi, (mission, kernel, display_label, color_hex)) in enumerate(mission_groups)
            x_pos = x_center + (mi - (n_miss + 1) / 2) * (bar_w + gap)
            row   = filter(r -> r.Cover == cover && r.Mission == mission &&
                               (kernel == "——" ? r.Kernel ∈ ["——", "NA"] : r.Kernel == kernel),
                           sub_cr)
            isempty(row) && continue
            r   = row[1, :]
            col = parse(Makie.Colors.Colorant, color_hex)

            barplot!(ax_cr, [x_pos], [r.CR];
                width      = bar_w,
                color      = (col, 0.85),
                strokewidth = 0.6,
                strokecolor = col,
            )
            # CI error bar
            ci_lo = r.CI_lower
            ci_hi = r.CI_upper
            # clamp CI to [0, 1] for display
            ci_lo = max(0.0, ci_lo)
            ci_hi = min(1.0, ci_hi)
            errorbars!(ax_cr, [x_pos], [r.CR],
                [r.CR - ci_lo], [ci_hi - r.CR];
                whiskerwidth = 5, color = :black, linewidth = 1.1,
            )
        end
    end

    ax_cr.xticks = (xtick_positions, xtick_labels)
    ylims!(ax_cr, 0.70, 1.02)

    # ── draw difference bars (bottom panel) ────────────────────────────────
    diff_miss = [
        ("Const 2 m/s", "——", "vs. Const. 2 m/s", COL_CONST2),
        ("Const 8 m/s", "——", "vs. Const. 8 m/s",  COL_CONST8),
    ]
    n_diff  = length(diff_miss)
    dbar_w  = 0.28
    d_gap   = 0.10

    diff_xtick_pos  = Float64[]
    diff_xtick_labs = String[]

    for (ci, cover) in enumerate(covers_order)
        x_center = Float64(ci) * (n_diff * (dbar_w + d_gap) + grp_gap + 0.1)
        push!(diff_xtick_pos, x_center)
        push!(diff_xtick_labs, cover)

        for (di, (baseline_mission, baseline_kernel, dlabel, color_hex)) in enumerate(diff_miss)
            x_pos = x_center + (di - (n_diff + 1) / 2) * (dbar_w + d_gap)

            # Find the matching diff row: KDE-guided vs baseline, for this cover
            row = filter(r ->
                r.Cover    == cover &&
                r.Mission_B == baseline_mission &&
                (baseline_kernel == "——" ? r.Kernel_B ∈ ["——", "NA"] : r.Kernel_B == baseline_kernel),
                sub_diff
            )
            isempty(row) && continue
            r   = row[1, :]
            col = parse(Makie.Colors.Colorant, color_hex)

            barplot!(ax_diff, [x_pos], [r.CR_diff];
                width       = dbar_w,
                color       = (col, 0.70),
                strokewidth = 0.6,
                strokecolor = col,
            )
            errorbars!(ax_diff, [x_pos], [r.CR_diff],
                [r.CR_diff - r.CI_lower], [r.CI_upper - r.CR_diff];
                whiskerwidth = 6, color = :black, linewidth = 1.1,
            )
            # Significance annotation
            sig = (r.CI_lower > 0 || r.CI_upper < 0)
            if sig
                y_ann = r.CR_diff >= 0 ? r.CI_upper + 0.003 : r.CI_lower - 0.008
                text!(ax_diff, x_pos, y_ann; text="*", fontsize=14,
                      align=(:center, :bottom), color=:black)
            end
        end
    end

    ax_diff.xticks = (diff_xtick_pos, diff_xtick_labs)
    hlines!(ax_diff, [0.0]; color="#444444", linewidth=1.2, linestyle=:solid)

    # ── legends ────────────────────────────────────────────────────────────
    legend_elems_cr = [
        PolyElement(color=(parse(Makie.Colors.Colorant, c), 0.85), strokecolor=:transparent)
        for (_, _, _, c) in mission_groups
    ]
    legend_labels_cr = [d for (_, _, d, _) in mission_groups]
    Legend(fig[1, 2], legend_elems_cr, legend_labels_cr;
        labelsize=9, patchsize=(14,10), framevisible=false,
    )

    legend_elems_diff = [
        PolyElement(color=(parse(Makie.Colors.Colorant, c), 0.70), strokecolor=:transparent)
        for (_, _, _, c) in diff_miss
    ]
    legend_labels_diff = [d for (_, _, d, _) in diff_miss]
    Legend(fig[2, 2], legend_elems_diff, legend_labels_diff;
        labelsize=9, patchsize=(14,10), framevisible=false,
    )

    # Significance note
    Label(fig[3, 1:2],
        "* 95% CI excludes zero (statistically significant at α=0.05). " *
        "Block-bootstrap: n=5000, block_frac=3% (block_side=9 cells), seed=42.",
        fontsize = 8,
        halign   = :left,
        padding  = (8, 4, 2, 2),
    )

    colsize!(fig.layout, 1, Relative(0.82))
    colsize!(fig.layout, 2, Relative(0.18))
    rowgap!(fig.layout, 1, 10)
    rowgap!(fig.layout, 2, 4)
    rowsize!(fig.layout, 1, Relative(0.52))
    rowsize!(fig.layout, 2, Relative(0.44))
    rowsize!(fig.layout, 3, Relative(0.04))

    paths = String[]
    for fmt in formats
        p = joinpath(out_dir, "bootstrap_cr_main.$fmt")
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    return paths
end

# ===========================================================================
# KDE surface and density-class figures (merged from former figures_kde.jl)
# ===========================================================================
#
# Printf is provided by the KDEFlightPlanning module when this file is loaded
# as part of the package. The explicit import below makes the KDE-figure
# section self-sufficient when figures.jl is `include`d standalone (e.g. in the
# test suite's syntax-check testset), where `@sprintf` would otherwise be
# undefined. Re-importing inside the module is harmless.
using Printf: @printf, @sprintf

# These functions extend the publication-figure API with typed-dispatch
# helpers for KDE surfaces and density-class results:
#   - fig_kde_surface_heatmap(surf::KDESurface; ...)  /  (Z::AbstractMatrix; ...)
#   - fig_kde_class_map(r::DensityClassResult; ...)
#   - fig_kde_class_cr(r::DensityClassResult, dict; ...)  /  (df::DataFrame; ...)
#   - fig_kde_strata_within_cover_cr(df::DataFrame; status)
# All return Vector{String} of saved paths. Diagnostic figures carry a
# `[DIAGNOSTIC]` tag; true-surface figures carry `[true surface]`.

# ===========================================================================
# Figure: KDE density surface heatmap
# ===========================================================================

"""
    fig_kde_surface_heatmap(surf::KDESurface;
                             out_dir  = ".",
                             formats  = ["png","pdf"],
                             colormap = :viridis,
                             status   = "diagnostic") -> Vector{String}

Heatmap of a `KDESurface` resampled to the count grid. The struct carries
its own shape and CRS so no separate arguments are needed.

Row 1 is plotted at the top (north-up: `ax.yreversed = true`).

`status` controls the figure annotation:
  - `"diagnostic"` → `[DIAGNOSTIC — not manuscript-ready]`
  - `"true_surface"` → `[true surface — KDE from native GeoTIFF]`
"""
function fig_kde_surface_heatmap(surf::KDESurface;
                                  out_dir  ::AbstractString = ".",
                                  formats  ::Vector{String} = ["png","pdf"],
                                  colormap ::Symbol         = :viridis,
                                  status   ::AbstractString = "diagnostic")::Vector{String}

    mkpath(out_dir)
    H, W   = size(surf)
    is_true = occursin("true", lowercase(status)) && !occursin("diag", lowercase(status))
    tag     = is_true ? " [true surface]" : " [DIAGNOSTIC]"
    z_note  = is_true ? "KDE from native-resolution GeoTIFF." : "KDE from screenshot orthomosaic (approximate alignment)."

    fig = Figure(size=(700, 500), backgroundcolor=parse(Makie.Colors.Colorant, FIG_BG))
    ax  = Axis(fig[1, 1])
    _theme_axis!(ax;
        xlabel = "Column  (west → east,  $W cells)",
        ylabel = "Row  (north → south,  $H cells)",
        title  = "KDE density surface$tag")
    ax.yreversed = true   # row 1 at top = north-up

    hm = heatmap!(ax, 1:W, 1:H, surf.Z'; colormap=colormap, colorrange=(0.0,1.0))
    Colorbar(fig[1,2], hm;
        label="Normalised KDE density [0,1]",
        labelsize=10, ticklabelsize=9, width=14)

    Label(fig[2,1:2],
        "Shape: $(H)×$(W) cells.  CRS: $(surf.crs).  " * z_note *
        "  Values in [0,1]; not renormalised on import.",
        fontsize=7, halign=:left, padding=(8,4,2,2))

    rowsize!(fig.layout, 1, Relative(0.92))
    rowsize!(fig.layout, 2, Relative(0.08))

    _save_fig(fig, joinpath(out_dir, "kde_surface_heatmap"), formats)
end

# ---------------------------------------------------------------------------
# Convenience overload: resampled bare matrix (for scripts with pre-resampled Z)
# ---------------------------------------------------------------------------

"""
    fig_kde_surface_heatmap(Z::AbstractMatrix; kwargs...) -> Vector{String}

Overload for a bare matrix (already at count-grid resolution).
Infers shape from `Z`; CRS annotation is omitted.
"""
function fig_kde_surface_heatmap(Z::AbstractMatrix;
                                  out_dir  ::AbstractString = ".",
                                  formats  ::Vector{String} = ["png","pdf"],
                                  colormap ::Symbol         = :viridis,
                                  status   ::AbstractString = "diagnostic")::Vector{String}

    surf_anon = KDESurface(Matrix{Float64}(Z), GT_COUNTGRID, CRS_DURHAM)
    fig_kde_surface_heatmap(surf_anon; out_dir=out_dir, formats=formats,
                            colormap=colormap, status=status)
end

# ===========================================================================
# Figure: KDE density class map (spatial)
# ===========================================================================

"""
    fig_kde_class_map(r::DensityClassResult;
                       out_dir = ".",
                       formats = ["png","pdf"],
                       status  = "diagnostic") -> Vector{String}

Spatial map of KDE density class assignments drawn from a `DensityClassResult`.
Support counts are pulled from `kde_class_support(r)` so no recomputation is needed.

Colour scheme (distinct, colourblind-safer than red/green):
  - field-like:      sand `#D4B896`
  - deciduous-like:  teal `#20808D`
  - coniferous-like: dark teal `#1B474D`

A discrete swatch legend replaces the continuous colorbar so class labels are
always readable against a light background. The caption is split into two lines.
"""
function fig_kde_class_map(r::DensityClassResult;
                            out_dir ::AbstractString = ".",
                            formats ::Vector{String} = ["png","pdf"],
                            status  ::AbstractString = "diagnostic")::Vector{String}

    mkpath(out_dir)
    H, W   = size(r.class_matrix)
    s      = kde_class_support(r)
    is_true = occursin("true", lowercase(status)) && !occursin("diag", lowercase(status))
    tag    = is_true ? " [true surface]" : " [DIAGNOSTIC]"

    # Discrete colour constants (matches KDE class integer values 1/2/3)
    col_field  = parse(Makie.Colors.Colorant, "#D4B896")  # sand
    col_decid  = parse(Makie.Colors.Colorant, "#20808D")  # teal
    col_conif  = parse(Makie.Colors.Colorant, "#1B474D")  # dark teal

    # Build a 3-stop categorical colormap spanning [0.5, 3.5]
    cmap = Makie.cgrad([col_field, col_decid, col_conif]; categorical=true)

    # Figure layout: heatmap [1,1], discrete legend [1,2], two caption rows [2,1:2] [3,1:2]
    fig = Figure(size=(820, 540), backgroundcolor=parse(Makie.Colors.Colorant, FIG_BG))
    ax  = Axis(fig[1,1])
    _theme_axis!(ax;
        xlabel = "Column  (west → east,  $W cells)",
        ylabel = "Row  (north → south,  $H cells)",
        title  = "KDE density class map$tag")
    ax.yreversed = true
    # Ensure tick labels are explicit black
    ax.xticklabelcolor = :black
    ax.yticklabelcolor = :black
    ax.xlabelcolor     = :black
    ax.ylabelcolor     = :black
    ax.titlecolor      = :black

    heatmap!(ax, 1:W, 1:H, Float64.(r.class_matrix)'; colormap=cmap, colorrange=(0.5,3.5))

    # Discrete swatch legend on a light background — always readable
    legend_elems = [
        PolyElement(color=col_field,  strokecolor=:gray, strokewidth=0.5),
        PolyElement(color=col_decid,  strokecolor=:gray, strokewidth=0.5),
        PolyElement(color=col_conif,  strokecolor=:gray, strokewidth=0.5),
    ]
    legend_labels = [
        @sprintf("field-like  (%d, %.1f%%)",      s.field,      100.0*s.field/s.total),
        @sprintf("deciduous-like  (%d, %.1f%%)",   s.deciduous,  100.0*s.deciduous/s.total),
        @sprintf("coniferous-like  (%d, %.1f%%)",  s.coniferous, 100.0*s.coniferous/s.total),
    ]
    Legend(fig[1,2], legend_elems, legend_labels;
        title           = "KDE density class",
        titlesize       = 10,
        labelsize       = 9,
        framecolor      = parse(Makie.Colors.Colorant, "#BBBBBB"),
        framewidth      = 0.8,
        backgroundcolor = parse(Makie.Colors.Colorant, "#FAFAF8"),
        patchsize       = (14f0, 10f0),
        rowgap          = 4,
        padding         = (8f0, 8f0, 6f0, 6f0),
        margin          = (4f0, 4f0, 4f0, 4f0),
        tellheight      = false,
        tellwidth       = true,
        halign          = :left,
        valign          = :top,
    )

    # Caption line 1: support counts
    cap1 = @sprintf(
        "Support: field-like %d (%.1f%%), deciduous-like %d (%.1f%%), coniferous-like %d (%.1f%%);  total %d cells.",
        s.field, 100.0*s.field/s.total,
        s.deciduous, 100.0*s.deciduous/s.total,
        s.coniferous, 100.0*s.coniferous/s.total,
        s.total)
    # Caption line 2: method + thresholds + epistemological note
    cap2 = @sprintf(
        "Method: %s  │  lower=%.4f  upper=%.4f  eps=%.2e  │  Diagnostic only — NOT GLI ground truth.",
        r.method, r.lower_threshold, r.upper_threshold, r.eps_threshold)

    Label(fig[2,1:2], cap1;
        fontsize=8, halign=:left, color=:black,
        padding=(10f0, 6f0, 4f0, 0f0))
    Label(fig[3,1:2], cap2;
        fontsize=8, halign=:left, color=:black,
        padding=(10f0, 6f0, 0f0, 4f0))

    rowsize!(fig.layout, 1, Relative(0.84))
    rowsize!(fig.layout, 2, Fixed(20))
    rowsize!(fig.layout, 3, Fixed(20))
    colsize!(fig.layout, 1, Relative(0.72))  # heatmap gets ~72% of width
    colsize!(fig.layout, 2, Fixed(190))       # legend column fixed width

    _save_fig(fig, joinpath(out_dir, "kde_class_map"), formats)
end

# ===========================================================================
# Figure: per-class CR bar chart — two overloads
# ===========================================================================

"""
    fig_kde_class_cr(r::DensityClassResult,
                      count_grid_dict::AbstractDict{<:AbstractString, <:AbstractMatrix};
                      missions     = collect(keys(count_grid_dict)),
                      out_dir      = ".",
                      formats      = ["png","pdf"],
                      title_suffix = "",
                      status       = "diagnostic") -> Vector{String}

Bar chart of per-class CR drawn directly from a `DensityClassResult` and a
dict of count grids. Thresholds are read from `r`; no intermediate DataFrame
is required.

Dispatched on `DensityClassResult` — the primary typed path for new workflows.
"""
function fig_kde_class_cr(r::DensityClassResult,
                           count_grid_dict::AbstractDict{<:AbstractString, <:AbstractMatrix};
                           missions     ::AbstractVector   = collect(keys(count_grid_dict)),
                           out_dir      ::AbstractString   = ".",
                           formats      ::Vector{String}   = ["png","pdf"],
                           title_suffix ::AbstractString   = "",
                           status       ::AbstractString   = "diagnostic")::Vector{String}

    # Build a minimal DataFrame from the typed result
    rows = NamedTuple[]
    for mission in missions
        haskey(count_grid_dict, mission) || continue
        cr_info = kde_class_cr(r, count_grid_dict[mission])
        for (cls, label, info) in (
                (KDE_CLASS_FIELD,      "field-like",      cr_info.field),
                (KDE_CLASS_DECIDUOUS,  "deciduous-like",  cr_info.deciduous),
                (KDE_CLASS_CONIFEROUS, "coniferous-like", cr_info.coniferous))
            push!(rows, (;
                mission          = string(mission),
                class_label      = label,
                kde_class        = Int(cls),
                cr               = info.cr,
                n_cells          = info.n_cells,
                n_covered        = info.n_covered,
                upper_threshold  = r.upper_threshold,
                eps_threshold    = r.eps_threshold,
                threshold_method = string(r.method),
                alignment        = status,
            ))
        end
    end
    df = DataFrame(rows)
    isempty(df) && return String[]
    _fig_kde_class_cr_impl(df; out_dir=out_dir, formats=formats,
                            title_suffix=title_suffix, status=status)
end

"""
    fig_kde_class_cr(df::DataFrame;
                      out_dir      = ".",
                      formats      = ["png","pdf"],
                      title_suffix = "",
                      status       = "diagnostic") -> Vector{String}

Overload accepting a pre-computed summary DataFrame (from `kde_class_summary`
or loaded from CSV). Useful for post-hoc plotting after a pipeline run that
serialised results.
"""
function fig_kde_class_cr(df::DataFrame;
                           out_dir      ::AbstractString = ".",
                           formats      ::Vector{String} = ["png","pdf"],
                           title_suffix ::AbstractString = "",
                           status       ::AbstractString = "diagnostic")::Vector{String}
    isempty(df) && return String[]
    _fig_kde_class_cr_impl(df; out_dir=out_dir, formats=formats,
                            title_suffix=title_suffix, status=status)
end

# Shared rendering implementation (private)
function _fig_kde_class_cr_impl(df::DataFrame;
                                  out_dir      ::AbstractString,
                                  formats      ::Vector{String},
                                  title_suffix ::AbstractString,
                                  status       ::AbstractString)::Vector{String}

    mkpath(out_dir)

    is_true  = occursin("true", lowercase(status)) && !occursin("diag", lowercase(status))
    tag      = is_true ? " [true surface]" : " [DIAGNOSTIC]"

    missions_in = unique(df.mission)
    mission_ord = filter(m -> m ∈ missions_in, MISSION_ORDER)
    isempty(mission_ord) && (mission_ord = missions_in)

    class_order = ["field-like", "deciduous-like", "coniferous-like"]
    n_miss      = length(mission_ord)
    bar_w, gap, grp_gap = 0.22, 0.06, 0.30

    # Threshold metadata
    t_lo   = haskey(names(df) |> Set, "eps_threshold") ? first(df.eps_threshold) : NaN
    t_hi   = haskey(names(df) |> Set, "upper_threshold") ? first(df.upper_threshold) : NaN
    meth   = haskey(names(df) |> Set, "threshold_method") ? string(first(df.threshold_method)) : "?"

    fig = Figure(size=(700, 400), backgroundcolor=parse(Makie.Colors.Colorant, FIG_BG))
    ax  = Axis(fig[1,1])
    _theme_axis!(ax;
        xlabel="KDE density class",
        ylabel="Coverage ratio (CR)",
        title="KDE density class CR by mission$title_suffix$tag")
    ax.titlesize = 10
    ylims!(ax, 0.0, 1.08)

    xtick_pos, xtick_labs = Float64[], String[]

    for (ci, cls_label) in enumerate(class_order)
        x_center = Float64(ci) * (n_miss * (bar_w + gap) + grp_gap)
        push!(xtick_pos, x_center); push!(xtick_labs, cls_label)

        for (mi, mission) in enumerate(mission_ord)
            x_pos = x_center + (mi - (n_miss+1)/2) * (bar_w+gap)
            row   = filter(r -> r.mission == mission && r.class_label == cls_label, df)
            isempty(row) && continue
            r   = row[1,:]
            col = parse(Makie.Colors.Colorant, get(MISSION_COLORS, mission, "#888888"))

            barplot!(ax, [x_pos], [r.cr];
                width=bar_w, color=(col,0.82), strokewidth=0.6, strokecolor=col)
            text!(ax, x_pos, r.cr + 0.016;
                text=@sprintf("n=%d", r.n_cells), fontsize=7,
                align=(:center,:bottom), color="#444444")
        end
    end

    ax.xticks = (xtick_pos, xtick_labs)

    Label(fig[2,1:2],
        @sprintf("Thresholds: field≤%.2e, deciduous≤%.4f  (method: %s). ",
                  t_lo, t_hi, meth) *
        "Diagnostic only — KDE-derived classes are NOT GLI ground truth. " *
        "GLI ref: Sullivan et al. 2023 (DOI 10.3390/rs15215091).",
        fontsize=7, halign=:left, padding=(8,4,2,2))

    legend_elems = [
        PolyElement(color=(parse(Makie.Colors.Colorant, get(MISSION_COLORS, m, "#888888")), 0.82),
                    strokecolor=:transparent)
        for m in mission_ord
    ]
    Legend(fig[1,2], legend_elems, mission_ord;
        labelsize=9, patchsize=(14,10), framevisible=false)

    rowsize!(fig.layout, 1, Relative(0.87))
    rowsize!(fig.layout, 2, Relative(0.13))
    colsize!(fig.layout, 1, Relative(0.78))
    colsize!(fig.layout, 2, Relative(0.22))

    _save_fig(fig, joinpath(out_dir, "kde_class_cr"), formats)
end

# ===========================================================================
# Figure: within-cover KDE strata CR
# ===========================================================================

"""
    fig_kde_strata_within_cover_cr(df::DataFrame;
                                    out_dir      = ".",
                                    formats      = ["png","pdf"],
                                    title_suffix = "",
                                    status       = "diagnostic") -> Vector{String}

Bar chart of CR by KDE quantile stratum within a cover class.
Accepts the DataFrame from `kde_strata_within_cover`.

`status` should be `"screenshot_diagnostic"` or `"true_surface"`.
The `alignment` column in `df` is checked as a fallback.
"""
function fig_kde_strata_within_cover_cr(df::DataFrame;
                                         out_dir      ::AbstractString = ".",
                                         formats      ::Vector{String} = ["png","pdf"],
                                         title_suffix ::AbstractString = "",
                                         status       ::AbstractString = "diagnostic")::Vector{String}

    isempty(df) && return String[]
    mkpath(out_dir)

    # Determine annotation from status arg or alignment column
    eff_status = if !isempty(status) && status != "diagnostic"
        status
    elseif hasproperty(df, :alignment)
        string(first(df.alignment))
    else
        "diagnostic"
    end

    is_true = occursin("true", lowercase(eff_status)) && !occursin("diag", lowercase(eff_status))
    tag     = is_true ? " [true surface]" : " [DIAGNOSTIC]"

    missions_in = unique(df.mission)
    mission_ord = filter(m -> m ∈ missions_in, MISSION_ORDER)
    isempty(mission_ord) && (mission_ord = missions_in)

    strata_all = sort(unique(df.stratum))
    n_miss     = length(mission_ord)
    bar_w, gap, grp_gap = 0.20, 0.05, 0.22

    fig = Figure(size=(720, 400), backgroundcolor=parse(Makie.Colors.Colorant, FIG_BG))
    ax  = Axis(fig[1,1])
    _theme_axis!(ax;
        xlabel="KDE density stratum",
        ylabel="Coverage ratio (CR)",
        title="Within-cover KDE strata CR$title_suffix$tag")
    ax.titlesize = 10
    ylims!(ax, 0.0, 1.08)
    hlines!(ax, [0.0]; color="#444444", linewidth=0.8)

    xtick_pos, xtick_labs = Float64[], String[]

    for (si, stratum) in enumerate(strata_all)
        x_center = Float64(si) * (n_miss * (bar_w+gap) + grp_gap)
        sub_s    = filter(r -> r.stratum == stratum, df)
        s_label  = isempty(sub_s) ? "S$stratum" : string(first(sub_s.stratum_label))
        push!(xtick_pos, x_center); push!(xtick_labs, s_label)

        for (mi, mission) in enumerate(mission_ord)
            x_pos = x_center + (mi - (n_miss+1)/2) * (bar_w+gap)
            row   = filter(r -> r.mission == mission && r.stratum == stratum, df)
            isempty(row) && continue
            r   = row[1,:]
            col = parse(Makie.Colors.Colorant, get(MISSION_COLORS, mission, "#888888"))
            barplot!(ax, [x_pos], [r.cr];
                width=bar_w, color=(col,0.82), strokewidth=0.6, strokecolor=col)
        end
    end

    ax.xticks = (xtick_pos, xtick_labs)

    note_str = is_true ?
        "True-surface KDE: alignment with count grid is exact." :
        "Screenshot KDE: alignment is approximate — outputs are diagnostic only."

    Label(fig[2,1:2],
        "Stratum 0 = zero-density pixels inside cover mask (field-like within cover). " * note_str,
        fontsize=7, halign=:left, padding=(8,4,2,2))

    legend_elems = [
        PolyElement(color=(parse(Makie.Colors.Colorant, get(MISSION_COLORS, m, "#888888")), 0.82),
                    strokecolor=:transparent)
        for m in mission_ord
    ]
    Legend(fig[1,2], legend_elems, mission_ord;
        labelsize=9, patchsize=(14,10), framevisible=false)

    rowsize!(fig.layout, 1, Relative(0.87))
    rowsize!(fig.layout, 2, Relative(0.13))
    colsize!(fig.layout, 1, Relative(0.78))
    colsize!(fig.layout, 2, Relative(0.22))

    _save_fig(fig, joinpath(out_dir, "kde_strata_within_cover_cr"), formats)
end

# ===========================================================================
# Internal helper: save figure to multiple formats
# ===========================================================================

"""
    _save_fig(fig, base_path, formats) -> Vector{String}

Save a Makie figure to all requested formats. Returns saved paths.
`base_path` must NOT include an extension — one will be appended per format.
"""
function _save_fig(fig, base_path::AbstractString, formats::Vector{String})::Vector{String}
    paths = String[]
    for fmt in formats
        p = "$base_path.$fmt"
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    return paths
end
