#!/usr/bin/env julia
"""
    scripts/gen_characterization_figures.jl

Generates characterization figures for the manuscript from counts.json
and stats_and_coverage.csv (ground-truth summary files).

Figures generated (written to OUTPUT_DIR):
  char_return_comparison.{png,pdf}   — ground vs all-return CR by mission/cover
  char_saturation_scatter.{png,pdf}  — CR vs conditional mean (field saturation)
  char_line2_return_dist.{png,pdf}   — Line 2 return distribution + CR by cover
  char_line4_return_dist.{png,pdf}   — Line 4 return distribution + CR by cover

Usage
-----
  julia --project=. scripts/gen_characterization_figures.jl [CONFIG.toml] [OUTPUT_DIR]

  CONFIG.toml RunInputs TOML config. Its [paths].counts_json / .stats_csv /
              .gli_class_raster drive data discovery. Default: bundled
              data/ground_truth/ files.
  OUTPUT_DIR  where to write figures; default = <config outdir>/figures.

Data required (bundled under data/ground_truth/, or set via config):
  counts.json              — per-cell return count grids (36 entries)
  stats_and_coverage.csv   — precomputed stats per (cover, return, mission, kernel)
  gli_class_raster         — 263×324 integer-coded GLI raster required for C3/C4
                             (0=deciduous, 1=coniferous, 2=field) for fixed
                             class-specific line-band CR denominators.

These figures are SUPPLEMENTAL (char_ prefix) and do NOT require raw LiDAR data.
All data is derived from counts.json and stats_and_coverage.csv.

Notes
-----
- Line 2 center: row 40 in count grid (SCAN_NROWS=263, SCAN_NCOLS=324)
- Line 4 center: row 120 in count grid
- Band: ±20 rows (40-m band) around each planned line center
- Primary missions: Const. 2 m/s (const2/NA), KDE-guided Epa (density/E), Const. 8 m/s (const8/NA)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using CSV, DataFrames, JSON, Statistics
using CairoMakie

include(joinpath(@__DIR__, "ground_truth.jl"))

# ─────────────────────────────────────────────────────────────
# Paths (config-driven; no hardcoded space_files)
# ─────────────────────────────────────────────────────────────
const _GT        = resolve_ground_truth(get(ARGS, 1, ""))
const OUTPUT_DIR = get(ARGS, 2, _GT.figures_dir)
const GLI_CLASS_PATH = something(find_ground_truth(_GT, :gli_class_raster), "")

mkpath(OUTPUT_DIR)

const FORMATS = ["png", "pdf"]
const NROWS = SCAN_NROWS   # 263
const NCOLS = SCAN_NCOLS   # 324

# Mission and cover display constants (matching figures.jl palette)
const CHAR_MISSIONS = [
    (String(m.mission), String(m.kernel), m.display_label, color)
    for (m, color) in zip(KDEFlightPlanning.CHARACTERIZATION_MISSIONS,
                          ("#A84B2F", "#20808D", "#1B474D"))
]
const CHAR_COVERS = ["Field", "Deciduous", "Coniferous"]
const COVER_KEYS  = ["field", "decid", "conif"]
const FIG_BG      = "#F7F4EF"
const COL_GRID    = "#DDDAD3"

# ─────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────
println("Loading counts.json …")
counts_json = require_ground_truth(_GT, :counts_json, "counts.json")
count_grids = load_count_grids(counts_json)
println("  Loaded $(length(count_grids)) count-grid entries")

println("Loading stats_and_coverage.csv …")
stats_path = require_ground_truth(_GT, :stats_csv, "stats_and_coverage.csv")
stats_df = CSV.read(stats_path, DataFrame)
println("  $(nrow(stats_df)) rows")

class_codes = isnothing(_GT.inputs) ? Dict(:field=>2, :decid=>0, :conif=>1) :
                                     _GT.inputs.gli_class_codes
flip_y = !isnothing(_GT.inputs) && _GT.inputs.cover_mask_flip_y
cover_masks = isempty(GLI_CLASS_PATH) ? nothing : Dict(
    String(key) => mask for (key, mask) in KDEFlightPlanning.load_cover_masks(
        GLI_CLASS_PATH; nrows=NROWS, ncols=NCOLS, class_codes=class_codes, flip_y=flip_y)
)
if isnothing(cover_masks)
    @warn "No GLI class raster supplied; skipping C3/C4, which require exact cover masks."
else
    cover_masks, _support_mode, _grid_size = KDEFlightPlanning._line_scan_cover_support(
        count_grids; cover_masks=cover_masks)
    println("Loaded GLI class masks for line-band CR denominators:")
    for (cover_key, cover_label) in zip(COVER_KEYS, CHAR_COVERS)
        println("  $cover_label: $(count(cover_masks[cover_key])) cells")
    end
end

# ─────────────────────────────────────────────────────────────
# Helper: extract band CR for a planned line
# ─────────────────────────────────────────────────────────────
"""
    band_cr(count_grids, row_center, half_band; ret, mission_key, kernel)

Returns (n_covered, n_total, cr) for the ±half_band row band around row_center,
for all cover masks combined (or per cover if per_cover=true).
"""
function band_cr_per_cover(count_grids, row_center, half_band=20;
                            ret="ground", mission_key="density", kernel="E",
                            cover_masks=nothing)
    isnothing(cover_masks) && error("Line-band coverage requires exact cover masks")
    r0 = max(1, row_center - half_band)
    r1 = min(NROWS, row_center + half_band)
    rows = NamedTuple[]
    for (cover_key, cover_label) in zip(COVER_KEYS, CHAR_COVERS)
        k = (ret, cover_key, mission_key, kernel)
        mat = get(count_grids, k, nothing)
        isnothing(mat) && continue
        band = mat[r0:r1, :]
        mask = cover_masks[cover_key][r0:r1, :]
        n_total   = count(mask)
        n_covered = count((band .> 0) .& mask)
        push!(rows, (
            cover       = cover_label,
            n_total     = n_total,
            n_covered   = n_covered,
            cr          = n_total > 0 ? n_covered / n_total : 0.0,
        ))
    end
    return DataFrame(rows)
end

function band_counts_nonzero(count_grids, row_center, half_band=20;
                              ret="ground", mission_key="density", kernel="E",
                              cover_masks=nothing)
    isnothing(cover_masks) && error("Line-band return distributions require exact cover masks")
    r0 = max(1, row_center - half_band)
    r1 = min(NROWS, row_center + half_band)
    all_vals = Int[]
    for cover_key in COVER_KEYS
        k = (ret, cover_key, mission_key, kernel)
        mat = get(count_grids, k, nothing)
        isnothing(mat) && continue
        band = mat[r0:r1, :]
        mask = cover_masks[cover_key][r0:r1, :]
        append!(all_vals, filter(x -> x > 0, vec(band[mask])))
    end
    return all_vals
end

# ─────────────────────────────────────────────────────────────
# Figure C1: char_return_comparison — ground vs all CR
# ─────────────────────────────────────────────────────────────
println("\n=== Figure C1: char_return_comparison ===")

# Map source CSV labels to the primary mission IDs before selecting plot data.
primary_stats = KDEFlightPlanning.characterization_stats(stats_df)

fig_c1 = Figure(size=(1200, 480), backgroundcolor=FIG_BG)
Label(fig_c1[0, 1:3],
    "Ground-return vs. All-return Coverage Ratio by Mission and Cover Class",
    fontsize=13, font=:bold, halign=:left, padding=(8, 0, 4, 4))

for (ci, cover) in enumerate(CHAR_COVERS)
    ax = Axis(fig_c1[1, ci];
        title = cover, titlesize=12,
        ylabel = ci == 1 ? "Coverage Ratio (CR)" : "",
        ylabelsize=10, xticklabelsize=9, yticklabelsize=9,
        backgroundcolor=:transparent,
    )
    ax.rightspinevisible = false
    ax.topspinevisible   = false
    ax.ytickformat = vs -> ["$(round(Int, v*100))%" for v in vs]
    ylims!(ax, 0.82, 1.02)

    for (mi, (mission_key, kernel, display_label, color_hex)) in enumerate(CHAR_MISSIONS)
        col = parse(Makie.Colors.Colorant, color_hex)
        for (ri, ret) in enumerate(["Ground", "All"])
            sub = filter(r -> r.Cover == cover && r.mission_key == Symbol(mission_key) &&
                              r.kernel_key == Symbol(kernel) && r.Return == ret,
                         primary_stats)
            isempty(sub) && continue
            cr = sub[1, :CR]
            x_pos = Float64(ri) + (mi - 2) * 0.28
            barplot!(ax, [x_pos], [cr]; width=0.24, color=(col, 0.85), strokewidth=0)
        end
    end

    ax.xticks = ([1.0, 2.0], ["Ground\nreturn", "All\nreturn"])
end

legend_elems = [PolyElement(color=parse(Makie.Colors.Colorant, c), strokecolor=:transparent)
                for (_, _, _, c) in CHAR_MISSIONS]
legend_labels = [d for (_, _, d, _) in CHAR_MISSIONS]
Legend(fig_c1[1, 4], legend_elems, legend_labels; framevisible=false, labelsize=9, patchsize=(14,10))

paths_c1 = String[]
for fmt in FORMATS
    p = joinpath(OUTPUT_DIR, "char_return_comparison.$fmt")
    save(p, fig_c1; px_per_unit = fmt == "png" ? 2 : 1)
    push!(paths_c1, p)
end
println("  Written: ", join(basename.(paths_c1), ", "))

# ─────────────────────────────────────────────────────────────
# Figure C2: char_saturation_scatter — CR vs conditional mean
# ─────────────────────────────────────────────────────────────
println("\n=== Figure C2: char_saturation_scatter ===")

scatter_df = filter(r -> r.Return == "Ground", primary_stats)

fig_c2 = Figure(size=(1100, 420), backgroundcolor=FIG_BG)
Label(fig_c2[0, 1:3],
    "Coverage Ratio vs. Conditional Mean Ground Returns Per Cell (saturation diagnostic)\n" *
    "Conditional mean = mean returns per covered cell (cells with ≥1 return)",
    fontsize=12, font=:bold, halign=:left, padding=(8, 0, 4, 4))

for (ci, cover) in enumerate(CHAR_COVERS)
    ax = Axis(fig_c2[1, ci];
        title = cover, titlesize=12,
        xlabel = "Mean ground returns / covered cell",
        ylabel = ci == 1 ? "Coverage Ratio" : "",
        xlabelsize=9, ylabelsize=9,
        xticklabelsize=8, yticklabelsize=8,
        backgroundcolor=:transparent,
    )
    ax.rightspinevisible = false
    ax.topspinevisible   = false
    ax.ytickformat = vs -> ["$(round(Int, v*100))%" for v in vs]

    sub = filter(r -> r.Cover == cover, scatter_df)
    for row in eachrow(sub)
        col_idx = findfirst(x -> Symbol(x[1]) == row.mission_key &&
                                Symbol(x[2]) == row.kernel_key, CHAR_MISSIONS)
        isnothing(col_idx) && continue
        col = parse(Makie.Colors.Colorant, CHAR_MISSIONS[col_idx][4])
        scatter!(ax, [row.cond_mean], [row.CR]; color=col, markersize=14, strokewidth=0.5, strokecolor=:white)
        text!(ax, "$(first(row.display_label, 9))";
              position=(row.cond_mean, row.CR),
              offset=(4, 2), fontsize=7, color=col)
    end

    if !isempty(sub)
        ylims!(ax, max(0.78, minimum(sub.CR) - 0.03), 1.01)
    end
end

paths_c2 = String[]
for fmt in FORMATS
    p = joinpath(OUTPUT_DIR, "char_saturation_scatter.$fmt")
    save(p, fig_c2; px_per_unit = fmt == "png" ? 2 : 1)
    push!(paths_c2, p)
end
println("  Written: ", join(basename.(paths_c2), ", "))

# ─────────────────────────────────────────────────────────────
# Helper: make line distribution figure
# ─────────────────────────────────────────────────────────────
function make_line_dist_fig(line_num, row_center, out_name_base;
                             omit_coniferous=false)
    println("\n  Line $line_num (row_center=$row_center) …")

    # Compute band CR per cover per mission
    cr_rows = NamedTuple[]
    for (mission_key, kernel, display_label, color_hex) in CHAR_MISSIONS
        cr_df_m = band_cr_per_cover(count_grids, row_center; ret="ground",
                                     mission_key=mission_key, kernel=kernel,
                                     cover_masks=cover_masks)
        for row in eachrow(cr_df_m)
            omit_coniferous && row.cover == "Coniferous" && row.n_covered == 0 && continue
            push!(cr_rows, (
                mission    = display_label,
                cover      = row.cover,
                cr         = row.cr,
                n_covered  = row.n_covered,
                n_total    = row.n_total,
                color_hex  = color_hex,
            ))
        end
    end
    cr_table = DataFrame(cr_rows)

    # Get return distributions (non-zero counts)
    dists = Dict{String, Vector{Int}}()
    for (mission_key, kernel, display_label, _) in CHAR_MISSIONS
        vals = band_counts_nonzero(count_grids, row_center; ret="ground",
                                   mission_key=mission_key, kernel=kernel,
                                   cover_masks=cover_masks)
        dists[display_label] = vals
    end

    # Histogram bins
    all_pos = reduce(vcat, values(dists); init=Int[])
    p95 = isempty(all_pos) ? 500.0 : quantile(Float64.(all_pos), 0.95)
    nbins = 40
    bin_edges = range(1.0, max(p95 * 1.1, 10.0), length=nbins+1)

    # Covers to show in CR panel
    covers_show = filter(c -> begin
        omit_coniferous && c == "Coniferous" && return false
        any(r -> r.cover == c && r.n_covered > 0, eachrow(cr_table))
    end, CHAR_COVERS)

    # Build figure
    fig = Figure(size=(900, 750), backgroundcolor=FIG_BG)
    Label(fig[0, 1:2],
        "Line $line_num Ground Return Distribution and Coverage Ratio\n" *
        "±20-row band around planned line center (row $row_center)" *
        (omit_coniferous ? "\n[Note: Coniferous omitted — zero covered cells in band]" : ""),
        fontsize=12, font=:bold, halign=:left, padding=(8, 0, 4, 4))

    # Top panel: histogram
    ax_hist = Axis(fig[1, 1:2];
        title = "Return count distribution (covered cells only)",
        titlesize=11,
        xlabel = "Ground returns per 1 m² cell",
        ylabel = "Proportion of covered cells",
        xlabelsize=9, ylabelsize=9, xticklabelsize=8, yticklabelsize=8,
        backgroundcolor=:transparent,
    )
    ax_hist.rightspinevisible = false
    ax_hist.topspinevisible   = false

    legend_elems_h = []
    legend_labels_h = String[]

    for (mission_key, kernel, display_label, color_hex) in CHAR_MISSIONS
        vals = dists[display_label]
        isempty(vals) && continue
        col = parse(Makie.Colors.Colorant, color_hex)
        hist_counts = zeros(nbins)
        for v in vals
            bin_idx = searchsortedlast(collect(bin_edges), Float64(v))
            if 1 <= bin_idx <= nbins
                hist_counts[bin_idx] += 1
            end
        end
        norm_counts = hist_counts ./ max(sum(hist_counts), 1)
        bar_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in 1:nbins]
        bar_widths  = [bin_edges[i+1] - bin_edges[i] for i in 1:nbins]
        barplot!(ax_hist, bar_centers, norm_counts;
                 width=bar_widths, color=(col, 0.55), strokewidth=0)
        push!(legend_elems_h, PolyElement(color=(col, 0.70), strokecolor=:transparent))
        push!(legend_labels_h, display_label)
    end

    Legend(fig[1, 3], legend_elems_h, legend_labels_h;
           framevisible=false, labelsize=9, patchsize=(14, 10))

    # Bottom panel: CR bars by cover
    ax_bar = Axis(fig[2, 1:2];
        title = "Band CR by cover class",
        titlesize=11,
        xlabel = "Band CR (fraction of 1 m² cells with ≥1 ground return)",
        xlabelsize=9, ylabelsize=9, xticklabelsize=8, yticklabelsize=9,
        backgroundcolor=:transparent,
    )
    ax_bar.rightspinevisible = false
    ax_bar.topspinevisible   = false
    ax_bar.xtickformat = vs -> ["$(round(Int, v*100))%" for v in vs]
    ax_bar.ygridvisible = false

    bar_h = 0.22
    ytick_pos  = Float64[]
    ytick_labs = String[]
    y_idx = 0.0

    for (ci, cover) in enumerate(covers_show)
        y_idx += 1.0
        push!(ytick_pos, y_idx)
        push!(ytick_labs, cover)
        for (mi, (mk, kk, dl, ch)) in enumerate(CHAR_MISSIONS)
            rows = filter(r -> r.cover == cover && r.mission == dl, cr_table)
            isempty(rows) && continue
            cr = rows[1, :cr]
            col = parse(Makie.Colors.Colorant, ch)
            y_pos = y_idx + (mi - 2) * (bar_h + 0.03)
            barplot!(ax_bar, [y_pos], [cr];
                     direction=:x, color=col, width=bar_h, strokewidth=0)
            text!(ax_bar, "$(round(cr*100, digits=1))%";
                  position=(cr + 0.002, y_pos),
                  fontsize=8, align=(:left, :center))
        end
    end

    ax_bar.yticks = (ytick_pos, ytick_labs)
    xlims!(ax_bar, max(minimum(cr_table.cr) - 0.05, 0.0), 1.02)

    legend_elems_b = [PolyElement(color=parse(Makie.Colors.Colorant, c), strokecolor=:transparent)
                      for (_, _, _, c) in CHAR_MISSIONS]
    legend_labels_b = [d for (_, _, d, _) in CHAR_MISSIONS]
    Legend(fig[2, 3], legend_elems_b, legend_labels_b;
           framevisible=false, labelsize=9, patchsize=(14, 10))

    # Save
    paths = String[]
    for fmt in FORMATS
        p = joinpath(OUTPUT_DIR, "$(out_name_base).$fmt")
        save(p, fig; px_per_unit = fmt == "png" ? 2 : 1)
        push!(paths, p)
    end
    println("  Written: ", join(basename.(paths), ", "))
    return paths
end

# ─────────────────────────────────────────────────────────────
# Figure C3: char_line2_return_dist (Line 2, row 40)
# ─────────────────────────────────────────────────────────────
paths_c3 = String[]
paths_c4 = String[]
if !isnothing(cover_masks)
    waypoint_path = require_ground_truth(_GT, :waypoints_xy, "E_density_aware__waypoints_xy.csv")
    planned_lines = derive_planned_lines(waypoint_path)
    cover_geotransform = read_band(GLI_CLASS_PATH)[2]
    grid_geotransform = KDEFlightPlanning._line_scan_geotransform(
        (NROWS, NCOLS); geotransform=cover_geotransform)
    function row_for_line(number)
        rows = filter(row -> row.line_number == number, planned_lines)
        nrow(rows) == 1 || error("Expected exactly one planned line numbered $number")
        return KDEFlightPlanning._line_scan_row(only(rows.y_center), grid_geotransform)
    end
    println("\n=== Figure C3: char_line2_return_dist ===")
    append!(paths_c3, make_line_dist_fig(2, row_for_line(2), "char_line2_return_dist"; omit_coniferous=false))

    println("\n=== Figure C4: char_line4_return_dist ===")
    append!(paths_c4, make_line_dist_fig(4, row_for_line(4), "char_line4_return_dist"; omit_coniferous=false))
end

# ─────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────
println("\n=== All characterization figures written to: $OUTPUT_DIR ===")
all_paths = vcat(paths_c1, paths_c2, paths_c3, paths_c4)
for p in all_paths
    sz = round(stat(p).size / 1024; digits=1)
    println("  $(basename(p))  ($(sz) kB)")
end

