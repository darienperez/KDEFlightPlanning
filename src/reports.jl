"""
    reports.jl — Stage-by-stage validation visuals and reports

Each function takes a stage-specific result struct (or DataFrame) plus a
`RunInputs` (or output dir) and writes one or more files. All return the
list of saved paths so callers can assemble a `manifest.csv`.

Tier convention: M = manuscript candidate, D = diagnostic only, B = both.

Functions exported:

  Ingest / georeference (§3.1):
    - `report_ingest_preview(rs, outdir)`              [M+D]
    - `report_geotiff_metadata(rs, outdir)`            [D]
    - `report_gli_overlay(rs, gli_rs, outdir)`         [M]

  Features / PCA (§3.2):
    - `report_lab_channels(L, a, b, outdir)`           [D]
    - `report_pca_explained(pca, outdir)`              [M]
    - `report_feature_distributions(X, outdir)`        [D]

  Clustering / tree labels (§3.3):
    - `report_cluster_metrics_sweep(metrics, outdir)`  [M]
    - `report_cluster_overlays(rs, labels, k, outdir)` [M]
    - `report_cluster_lab_summary(L,a,b,labels,k, …)`  [D] — feeds the diagnostic
    - `report_tree_label_decision(meta, outdir)`       [M] — auto-rendered MD

  KDE / speed (§3.4–3.5):
    - `report_kde_density(dens, outdir; gt, crs)`      [M]
    - `report_speed_map(dens, strategy, outdir)`       [M]
    - `report_waypoints_overlay(rs, dens, wps_dict, outdir)` [M]

  Run summary (§3.8):
    - `report_run_manifest(outdir)`                    [M]
    - `report_provenance(outdir; extras...)`           [M]
    - `report_run_md(outdir; sections...)`             [M]

Tree-label policy
-----------------
This module never auto-overrides a manual `tree_labels` choice. The
`report_cluster_lab_summary` function records candidate heuristic scores
(mean L*, a*, b*, cluster size, optional GLI overlap) and
`report_tree_label_decision` writes a Markdown audit trail explaining that
labels came from `RunInputs.tree_labels`.
"""

using CairoMakie
using ColorSchemes
using DataFrames
using Statistics
using Printf: @sprintf
using Dates
using JSON
import Colors: RGBA

# ---------------------------------------------------------------------------
# Downsampling helpers (used to avoid 100M-pixel Makie renders on real
# GeoTIFF orthomosaics; bounded by `max_preview_px` per side).
# ---------------------------------------------------------------------------

const DEFAULT_MAX_PREVIEW_PX = 2000   # per side; ~4 MP figures max

"""
    _plot_stride(H, W; max_preview_px = DEFAULT_MAX_PREVIEW_PX) -> Int

Compute an integer stride so that the resulting thumbnail has
`max(H,W) ≤ max_preview_px`. Always ≥ 1.
"""
function _plot_stride(H::Integer, W::Integer;
                      max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX)
    mx = max(H, W)
    mx <= max_preview_px && return 1
    return cld(mx, max_preview_px)
end

"""
    _downsample(M::AbstractMatrix, stride::Integer) -> AbstractMatrix

Stride-decimate a 2-D matrix for plotting only. Does not modify pixel values.
"""
@inline _downsample(M::AbstractMatrix, stride::Integer) =
    stride <= 1 ? M : @view M[1:stride:end, 1:stride:end]

# ---------------------------------------------------------------------------
# Small file helpers
# ---------------------------------------------------------------------------

_ensure_dir(d) = (mkpath(d); d)

# Open a JSON file with a non-finite-float sanitizer (NaN/Inf → null)
_sanitise_json(x::AbstractFloat) = isfinite(x) ? x : nothing
_sanitise_json(x::AbstractArray) = map(_sanitise_json, x)
_sanitise_json(x::AbstractDict)  = Dict(k => _sanitise_json(v) for (k, v) in x)
_sanitise_json(x::NamedTuple)    = Dict(string(k) => _sanitise_json(v) for (k, v) in pairs(x))
_sanitise_json(x)                = x

function _save_json(path::AbstractString, obj)
    open(path, "w") do io
        JSON.print(io, _sanitise_json(obj), 2)
    end
    return path
end

# Default raster-export formats. PDF is no longer in the default set because
# the user observed Makie's `CairoMakie.save("...pdf", fig)` timing out on
# `SystemError(close, ...)` for large heatmap figures on slow / synced
# filesystems. PNG is enough for diagnostic visuals; manuscript-candidate
# vector figures can be requested explicitly per-call or via the `report_formats`
# config knob.
const DEFAULT_REPORT_FORMATS = ["png"]

"""
    _save_fig(fig, base_path; formats = DEFAULT_REPORT_FORMATS) -> Vector{String}

Save `fig` to one or more files with extensions taken from `formats`.

Each format is attempted in its own `try/catch`; if any single save fails
(commonly: `SystemError(close, "Operation timed out")` when CairoMakie
flushes a large PDF on a synced filesystem) the failure is logged with
`@warn` and the loop continues with the next format. The returned vector
contains only the paths that were successfully written.

Recognised formats: `"png"`, `"pdf"`. Anything else falls through to
`CairoMakie.save` unchanged.
"""
function _save_fig(fig, base_path::AbstractString;
                    formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    paths = String[]
    for fmt in formats
        p = base_path * "." * fmt
        try
            if fmt == "png"
                CairoMakie.save(p, fig; px_per_unit = 2)
            else
                CairoMakie.save(p, fig)
            end
            push!(paths, p)
        catch e
            @warn "Failed to save figure (format=$fmt); skipping this format." path=p exception=(e, catch_backtrace())
        end
    end
    return paths
end

# ---------------------------------------------------------------------------
# 1. Ingest / georeference
# ---------------------------------------------------------------------------

"""
    report_ingest_preview(rs::GeoRasterStack, outdir;
                          max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                          georeference::Bool = true) -> [paths]

Save an RGB preview PNG of the input orthomosaic at `outdir/ingest/rgb_preview.png`.

Large rasters (e.g. 9279 × 11421 typical for a full Durham GeoTIFF) are
stride-decimated to ≤ `max_preview_px` per side before plotting so Makie
does not allocate a multi-hundred-megapixel image. When the
`GeoRasterStack` carries a geotransform and `georeference=true`, axes are
labelled in projected units (Easting/Northing).
"""
function report_ingest_preview(rs::GeoRasterStack, outdir::AbstractString;
                                max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                                georeference::Bool = true,
                                formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    # `formats` is accepted for API symmetry; the preview PNG is always
    # written. Any extra format is written via `_save_fig` (best-effort).
    extra_formats = filter(f -> f != "png", formats)
    dst = _ensure_dir(joinpath(outdir, "ingest"))
    H, W = size(rs)
    stride = _plot_stride(H, W; max_preview_px = max_preview_px)
    Zds    = _downsample(rs.Z, stride)
    Hd, Wd = size(Zds)

    fig = Figure(size = (700, max(120, 700 * Hd ÷ Wd)))
    use_geo = georeference && (rs.gt.dx != 0 && rs.gt.dy != 0)
    title = "Input orthomosaic ($(H)×$(W))" *
            (stride > 1 ? "  [thumbnail stride=$stride]" : "")

    if use_geo
        ex = raster_extents(rs)
        ax = Axis(fig[1, 1]; title = title,
                  xlabel = "Easting (m)", ylabel = "Northing (m)",
                  aspect = DataAspect(),
                  limits = (ex.xmin, ex.xmax, ex.ymin, ex.ymax))
        # Makie image! does not accept colorant matrices with explicit (xs, ys)
        # ranges; for RGB we plot in pixel-coords inside the axis and rely on
        # the axis limits + DataAspect for georeferenced display.
        image!(ax,
               (ex.xmin, ex.xmax),
               (ex.ymax, ex.ymin),
               permutedims(Zds, (2, 1)))
    else
        ax = Axis(fig[1, 1]; title = title,
                  xlabel = "col", ylabel = "row",
                  aspect = DataAspect(), yreversed = true)
        image!(ax, permutedims(Zds, (2, 1)))
        hidedecorations!(ax)
    end
    p = joinpath(dst, "rgb_preview.png")
    paths = String[]
    try
        CairoMakie.save(p, fig; px_per_unit = 2)
        push!(paths, p)
    catch e
        @warn "Failed to save rgb_preview.png; continuing." path=p exception=(e, catch_backtrace())
    end
    if !isempty(extra_formats)
        append!(paths, _save_fig(fig,
                                  joinpath(dst, "rgb_preview");
                                  formats = extra_formats))
    end
    return paths
end

"""
    report_geotiff_metadata(rs, outdir) -> [path]
"""
function report_geotiff_metadata(rs::GeoRasterStack, outdir::AbstractString)
    dst = _ensure_dir(joinpath(outdir, "ingest"))
    p = joinpath(dst, "geotiff_metadata.json")
    H, W = size(rs)
    ex = raster_extents(rs)
    _save_json(p, Dict(
        "source"      => rs.source,
        "shape_HW"    => [H, W],
        "geotransform" => Dict(
            "x_origin" => rs.gt.x_origin,
            "dx"       => rs.gt.dx,
            "x_rot"    => rs.gt.x_rot,
            "y_origin" => rs.gt.y_origin,
            "y_rot"    => rs.gt.y_rot,
            "dy"       => rs.gt.dy,
        ),
        "crs_wkt_set" => !isempty(rs.crs),
        "crs_wkt"     => rs.crs,
        "extents_utm" => Dict(
            "xmin" => ex.xmin, "xmax" => ex.xmax,
            "ymin" => ex.ymin, "ymax" => ex.ymax,
            "dx"   => ex.dx,   "dy"   => ex.dy,
        ),
    ))
    return [p]
end

# ---------------------------------------------------------------------------
# 2. Features / PCA
# ---------------------------------------------------------------------------

"""
    report_lab_channels(L, a, b, outdir) -> [paths]

Three-panel small-multiples of L*, a*, b* channel matrices.
"""
function report_lab_channels(L::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix,
                              outdir::AbstractString;
                              formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    dst = _ensure_dir(joinpath(outdir, "features"))
    H, W = size(L)
    fig = Figure(size = (1200, max(220, 380 * H ÷ W)))
    for (i, (mat, lbl)) in enumerate([(L, "L*"), (a, "a*"), (b, "b*")])
        ax = Axis(fig[1, i]; title = lbl, aspect = DataAspect(), yreversed = true)
        hm = heatmap!(ax, permutedims(mat); colormap = :viridis)
        Colorbar(fig[2, i], hm; vertical = false, height = 8)
        hidedecorations!(ax)
    end
    return _save_fig(fig, joinpath(dst, "lab_channels"); formats = formats)
end

"""
    report_pca_explained(variances::AbstractVector{<:Real}, outdir; threshold=0.95) -> [paths]

Scree plot with cumulative-variance line and threshold marker.
"""
function report_pca_explained(variances::AbstractVector{<:Real},
                                outdir::AbstractString;
                                threshold::Real = 0.95,
                                formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    dst = _ensure_dir(joinpath(outdir, "features"))
    n = length(variances)
    cum = cumsum(variances) ./ sum(variances)
    fig = Figure(size = (700, 460))
    ax  = Axis(fig[1, 1];
        title  = "PCA explained variance",
        xlabel = "Component",
        ylabel = "Variance ratio",
    )
    barplot!(ax, 1:n, variances ./ sum(variances); color = (:steelblue, 0.7))
    lines!(ax, 1:n, cum; color = :firebrick, linewidth = 2)
    scatter!(ax, 1:n, cum; color = :firebrick, markersize = 8)
    hlines!(ax, [threshold]; color = :gray40, linestyle = :dash)
    text!(ax, n, threshold; text = "  $(round(Int, threshold*100))%",
          align = (:right, :bottom), color = :gray40)
    return _save_fig(fig, joinpath(dst, "pca_explained_variance"); formats = formats)
end

# ---------------------------------------------------------------------------
# 3. Clustering and tree-label diagnostics
# ---------------------------------------------------------------------------

"""
    report_cluster_metrics_sweep(metrics::Vector{ClusterMetrics}, outdir) -> [paths]

4-panel sweep of silhouette / Dunn / DB / CH against k. Marks the chosen k
(`metrics[i].chosen == true`) with a circle marker on each panel.
"""
function report_cluster_metrics_sweep(metrics::Vector{ClusterMetrics},
                                       outdir::AbstractString;
                                       formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    dst = _ensure_dir(joinpath(outdir, "cluster"))
    ks = [m.k for m in metrics]
    sil = [m.silhouette for m in metrics]
    dn  = [m.dunn for m in metrics]
    db  = [m.db for m in metrics]
    ch  = [m.cal for m in metrics]
    chosen_idx = findfirst(m -> m.chosen, metrics)

    fig = Figure(size = (1100, 700))
    titles = ["Silhouette ↑", "Dunn ↑", "Davies–Bouldin ↓", "Calinski–Harabasz ↑"]
    series = [sil, dn, db, ch]
    coords = [(1,1), (1,2), (2,1), (2,2)]
    for (i, ((r, c), title, ys)) in enumerate(zip(coords, titles, series))
        ax = Axis(fig[r, c]; title = title, xlabel = "k", ylabel = "")
        scatterlines!(ax, ks, ys; color = :steelblue, markersize = 8)
        if !isnothing(chosen_idx)
            scatter!(ax, [ks[chosen_idx]], [ys[chosen_idx]];
                     color = :firebrick, markersize = 14, marker = :circle)
        end
    end
    Label(fig[0, :],
          "k-medoids quality sweep — chosen k highlighted in red";
          fontsize = 13, halign = :center)
    return _save_fig(fig, joinpath(dst, "cluster_quality_sweep"); formats = formats)
end

"""
    LabelSelectionPreview

Reporting spec that requests a *single* temporary contact-sheet image showing
every candidate k-medoids cluster as a coloured overlay on the RGB, so a TTY
operator can pick the vegetation cluster id(s). It is dispatched by
`report_cluster_overlays` to a method distinct from the legacy per-cluster
overlay writer. The contact sheet is meant to be transient (deleted by the
lightweight cross-site runner after the operator selects labels), so it is
composed as one PNG rather than `k` separate files.

# Fields
 - `alpha::Float64`: overlay opacity for the highlighted cluster (default 0.55).
 - `max_preview_px::Int`: per-side thumbnail cap so large orthomosaics do not
   blow up the render (default `DEFAULT_MAX_PREVIEW_PX`).
 - `greenness::Union{Nothing,Vector{Float64}}`: optional per-cluster greenness
   score (−mean CIELAB a*); when supplied the greenest cluster is annotated as a
   *hint only* in the panel titles.
"""
struct LabelSelectionPreview
    alpha          :: Float64
    max_preview_px :: Int
    greenness      :: Union{Nothing, Vector{Float64}}
end
LabelSelectionPreview(; alpha::Real = 0.55,
                        max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                        greenness = nothing) =
    LabelSelectionPreview(Float64(alpha), Int(max_preview_px),
                          greenness === nothing ? nothing : Float64.(greenness))

"""
    report_cluster_overlays(spec::LabelSelectionPreview,
                            img::AbstractMatrix{<:Colorant},
                            labels::AbstractVector{<:Integer},
                            k::Integer, out_png::AbstractString) -> String

Compose ONE contact-sheet PNG (a `ceil(√k) × ceil(k/…)` grid of panels) where
panel `cid` shows the RGB image with cluster `cid` highlighted. Returns the
saved path.

Label convention: `labels` is the flat per-pixel vector in the SAME order as
`stack_features`/`labels_to_mask`, i.e. `reshape(labels, H, W)` (column-major;
row index advances first). This matches the mask actually fed to the KDE, so the
operator reviews exactly the pixels that will be used — unlike the legacy
per-cluster `report_cluster_overlays(::GeoRasterStack, …)` method, which reshapes
row-major for a different (screenshot-derived) code path and is left unchanged.
"""
function report_cluster_overlays(spec::LabelSelectionPreview,
                                 img::AbstractMatrix{<:Colorant},
                                 labels::AbstractVector{<:Integer},
                                 k::Integer,
                                 out_png::AbstractString)
    H, W = size(img)
    length(labels) == H * W ||
        throw(DimensionMismatch("labels length $(length(labels)) ≠ H*W=$(H*W)"))
    label_img = reshape(labels, H, W)                 # column-major (correct)
    stride    = _plot_stride(H, W; max_preview_px = spec.max_preview_px)
    rgb_ds    = _downsample(img, stride)
    lab_ds    = _downsample(label_img, stride)
    Hd, Wd    = size(rgb_ds)

    ncol = ceil(Int, sqrt(k))
    nrow = ceil(Int, k / ncol)
    palette = ColorSchemes.tab10

    fig = Figure(size = (max(320, 300 * ncol), max(260, 260 * nrow + 40)),
                 backgroundcolor = :white)
    Label(fig[0, 1:ncol],
          "Cluster candidates (k=$k) — pick vegetation id(s)";
          fontsize = 15, font = :bold)
    for cid in 1:k
        r = fld(cid - 1, ncol) + 1
        c = mod(cid - 1, ncol) + 1
        hint = if spec.greenness !== nothing && length(spec.greenness) == k
            cid == argmax(spec.greenness) ? "  [greenest — hint]" : ""
        else
            ""
        end
        ax = Axis(fig[r, c]; title = "cluster $cid$hint",
                  aspect = DataAspect(), yreversed = true)
        hidedecorations!(ax)
        image!(ax, permutedims(rgb_ds, (2, 1)))
        col = palette[mod1(cid, length(palette))]
        overlay = fill(RGBA(col.r, col.g, col.b, 0.0), Hd, Wd)
        @inbounds for j in 1:Hd, i in 1:Wd
            lab_ds[j, i] == cid && (overlay[j, i] = RGBA(col.r, col.g, col.b, spec.alpha))
        end
        image!(ax, permutedims(overlay, (2, 1)))
    end
    mkpath(dirname(abspath(out_png)))
    CairoMakie.save(out_png, fig; px_per_unit = 2)
    return out_png
end

"""
    report_cluster_overlays(rs::GeoRasterStack, labels::AbstractVector{<:Integer},
                            k::Integer, outdir; alpha=0.55) -> [paths]

For each cluster id in 1..k, alpha-blend the cluster mask over the RGB
orthomosaic and save as `outdir/cluster/cluster_overlay_k{i}.png`.
"""
function report_cluster_overlays(rs::GeoRasterStack,
                                  labels::AbstractVector{<:Integer},
                                  k::Integer,
                                  outdir::AbstractString;
                                  alpha::Real = 0.55,
                                  max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                                  georeference::Bool = true,
                                  formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    # `formats` is accepted for API symmetry with the other reports.
    # Cluster overlays are large composited images and only write PNG
    # natively. If "pdf" is requested we additionally export the figure as
    # a PDF via `_save_fig` (best-effort, will warn on timeout and continue).
    extra_formats = filter(f -> f != "png", formats)
    dst = _ensure_dir(joinpath(outdir, "cluster"))
    H, W = size(rs)
    @assert length(labels) == H * W "labels length must equal H*W"
    # `labels` is produced in row-major image order, while Julia `reshape`
    # fills arrays column-major. Reconstruct the H×W label image by first
    # reshaping to W×H, then transposing back to row/column layout.
    label_img = permutedims(reshape(labels, W, H), (2, 1))
    stride    = _plot_stride(H, W; max_preview_px = max_preview_px)

    paths = String[]
    base_palette = ColorSchemes.tab10
    use_geo = georeference && (rs.gt.dx != 0 && rs.gt.dy != 0)
    ex = use_geo ? raster_extents(rs) : nothing

    for cid in 1:k
        mask    = label_img .== cid
        Zds     = _downsample(rs.Z, stride)
        mask_ds = _downsample(mask, stride)
        Hd, Wd  = size(Zds)
        fig = Figure(size = (800, max(220, 700 * Hd ÷ Wd)))
        title = "Cluster $cid overlay (alpha=$(alpha))" *
                (stride > 1 ? "  [thumbnail stride=$stride]" : "")
        ax = if use_geo
            Axis(fig[1, 1]; title = title,
                 xlabel = "Easting (m)", ylabel = "Northing (m)",
                 aspect = DataAspect(),
                 limits = (ex.xmin, ex.xmax, ex.ymin, ex.ymax))
        else
            Axis(fig[1, 1]; title = title,
                 aspect = DataAspect(), yreversed = true)
        end
        if use_geo
            image!(ax,
                   (ex.xmin, ex.xmax),
                   (ex.ymax, ex.ymin),
                   permutedims(Zds, (2, 1)))
        else
            image!(ax, permutedims(Zds, (2, 1)))
        end
        col = base_palette[mod1(cid, length(base_palette))]
        overlay = fill(RGBA(col.r, col.g, col.b, 0.0), Hd, Wd)
        @inbounds for j in 1:Hd, i in 1:Wd
            mask_ds[j, i] && (overlay[j, i] = RGBA(col.r, col.g, col.b, alpha))
        end
        if use_geo
            image!(ax,
                   (ex.xmin, ex.xmax),
                   (ex.ymax, ex.ymin),
                   permutedims(overlay, (2, 1)))
        else
            image!(ax, permutedims(overlay, (2, 1)))
            hidedecorations!(ax)
        end
        p = joinpath(dst, "cluster_overlay_k$(cid).png")
        try
            CairoMakie.save(p, fig; px_per_unit = 2)
            push!(paths, p)
        catch e
            @warn "Failed to save cluster overlay; continuing." cluster=cid path=p exception=(e, catch_backtrace())
        end
        # Optional extra formats (e.g. PDF) — best-effort, never aborts.
        if !isempty(extra_formats)
            append!(paths, _save_fig(fig,
                                      joinpath(dst, "cluster_overlay_k$(cid)");
                                      formats = extra_formats))
        end
    end
    return paths
end

"""
    report_cluster_lab_summary(L, a, b, labels, k, outdir;
                                gli_classes = nothing,
                                gli_class_codes = nothing) -> (df, [paths])

Diagnostic per-cluster summary of mean L*, a*, b*, cluster size, and
optional GLI overlap. Returns the summary `DataFrame` and the saved CSV path.

This is the structured input to `report_tree_label_decision`. It records
*candidate scores* for the user's audit; **the function never picks tree
labels itself**.
"""
function report_cluster_lab_summary(L::AbstractMatrix,
                                    a::AbstractMatrix,
                                    b::AbstractMatrix,
                                    labels::AbstractVector{<:Integer},
                                    k::Integer,
                                    outdir::AbstractString;
                                    gli_classes::Union{Nothing, AbstractMatrix} = nothing,
                                    gli_class_codes::Union{Nothing, AbstractDict} = nothing)
    dst = _ensure_dir(joinpath(outdir, "cluster"))
    H, W = size(L)
    @assert length(labels) == H * W
    label_img = reshape(labels, H, W)

    rows = NamedTuple[]
    Lflat = vec(L); aflat = vec(a); bflat = vec(b)
    for cid in 1:k
        idx = labels .== cid
        nidx = count(idx)
        nidx == 0 && continue
        meanL = mean(Lflat[idx])
        meana = mean(aflat[idx])
        meanb = mean(bflat[idx])
        # Candidate "greenness": negative a* in CIELAB (more vegetation = more negative a*)
        greenness = -meana

        gli_overlap = NamedTuple()
        if !isnothing(gli_classes) && !isnothing(gli_class_codes)
            mask = label_img .== cid
            if size(mask) == size(gli_classes)
                m = (count(mask) == 0) ? Dict() : Dict(
                    string(name) => count((mask .& (gli_classes .== code))) / count(mask)
                    for (name, code) in gli_class_codes
                )
                gli_overlap = (gli_overlap = m,)
            end
        end

        push!(rows, (;
            cluster_id  = cid,
            n_pixels    = nidx,
            frac_pixels = nidx / length(labels),
            mean_L      = meanL,
            mean_a      = meana,
            mean_b      = meanb,
            greenness_neg_a = greenness,
            gli_overlap...,
        ))
    end
    df = DataFrame(rows)

    p = joinpath(dst, "cluster_lab_summary.csv")
    # CSV via DataFrames-friendly flat write (skip nested gli_overlap dict)
    flat = select(df, Not(filter(c -> startswith(string(c), "gli_overlap"), names(df))))
    open(p, "w") do io
        # Manual CSV write to handle flat columns
        cols = names(flat)
        println(io, join(cols, ","))
        for row in eachrow(flat)
            println(io, join((string(row[c]) for c in cols), ","))
        end
    end

    # If GLI overlap was present, also emit a wide JSON
    if any(haskey(r, :gli_overlap) for r in rows)
        gp = joinpath(dst, "cluster_gli_overlap.json")
        _save_json(gp, Dict(
            string(r.cluster_id) => get(r, :gli_overlap, Dict()) for r in rows
        ))
        return df, [p, gp]
    end
    return df, [p]
end

"""
    report_tree_label_decision(outdir; tree_labels, lab_summary_df, source="manual",
                               heuristic_scores=nothing, notes="") -> [path]

Auto-render `outdir/cluster/tree_label_decision.md`. Records that
`tree_labels` came from manual configuration (`RunInputs.tree_labels`) and
appends candidate heuristic scores (mean L*, a*, b*, etc.) for audit.
"""
function report_tree_label_decision(outdir::AbstractString;
                                    tree_labels::AbstractVector{<:Integer},
                                    lab_summary_df::DataFrame,
                                    source::AbstractString = "manual (RunInputs.tree_labels)",
                                    notes::AbstractString = "")
    dst = _ensure_dir(joinpath(outdir, "cluster"))
    p = joinpath(dst, "tree_label_decision.md")
    open(p, "w") do io
        println(io, "# Tree-label decision audit")
        println(io)
        println(io, "**Source:** ", source)
        println(io, "**Selected cluster ids (`tree_labels`):** ", tree_labels)
        println(io, "**Generated:** ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)
        println(io, "## Per-cluster diagnostic scores")
        println(io)
        println(io, "These scores are recorded for **audit only** — none of them is used to override the manual `tree_labels`. If you re-run with a different `tree_labels`, this document updates accordingly.")
        println(io)
        # Render summary as a Markdown table
        cols = [:cluster_id, :n_pixels, :frac_pixels, :mean_L, :mean_a, :mean_b, :greenness_neg_a]
        present = [c for c in cols if c in propertynames(lab_summary_df)]
        # Header
        println(io, "| " * join(string.(present), " | ") * " |")
        println(io, "|" * join(fill("---", length(present)), "|") * "|")
        for row in eachrow(lab_summary_df)
            cells = String[]
            for c in present
                v = row[c]
                push!(cells, v isa AbstractFloat ? @sprintf("%.4f", v) : string(v))
            end
            println(io, "| " * join(cells, " | ") * " |")
        end
        println(io)
        println(io, "## Candidate heuristics (informational, not authoritative)")
        println(io)
        println(io, "- **lowest mean L\\***: typically darker → vegetation. The cluster minimising `mean_L` is *a candidate*, not a decision.")
        println(io, "- **highest greenness (-mean_a\\*)**: more negative a\\* implies green vegetation in CIELAB.")
        println(io, "- **largest n_pixels**: the dominant cluster, useful sanity check.")
        if any(startswith.(names(lab_summary_df), "gli_overlap"))
            println(io, "- **GLI overlap**: per-cover-class fraction of cluster pixels that fall in each GLI class. Look for high `decid` + `conif` overlap.")
        end
        if !isempty(notes)
            println(io)
            println(io, "## Notes")
            println(io)
            println(io, notes)
        end
    end
    return [p]
end

# ---------------------------------------------------------------------------
# 4. KDE / speed map
# ---------------------------------------------------------------------------

"""
    report_kde_density(dens::RasterGrid, outdir;
                       gt::Union{Nothing,GeoTransform} = nothing,
                       crs::AbstractString = "") -> [paths]

Heatmap PNG/PDF of the KDE density surface. If `gt` is supplied the figure
gains UTM axis labels; the function also writes a sidecar JSON
(`kde_density.gt.json`) with the geotransform and CRS so downstream
consumers can reconstruct UTM extents.
"""
function report_kde_density(dens::RasterGrid,
                              outdir::AbstractString;
                              gt::Union{Nothing, GeoTransform} = nothing,
                              crs::AbstractString = "",
                              max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                              formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    dst = _ensure_dir(joinpath(outdir, "kde"))
    Z = dens.Z
    H, W = size(Z)
    stride = _plot_stride(H, W; max_preview_px = max_preview_px)
    Zds    = _downsample(Z, stride)
    Hd, Wd = size(Zds)

    fig = Figure(size = (900, 720))
    title = isnothing(gt) ? "KDE density" : "KDE density (UTM)"
    if stride > 1
        title *= "  [thumbnail stride=$stride]"
    end
    # `yreversed` flips the y axis for pixel-space plots (row 1 at top). For
    # georeferenced plots the heatmap maps (xs, ys) to coords directly; flip
    # only when we are NOT in geo mode.
    ax = if isnothing(gt)
        Axis(fig[1, 1]; title = title, xlabel = "col", ylabel = "row",
             aspect = DataAspect(), yreversed = true)
    else
        Axis(fig[1, 1]; title = title,
             xlabel = "Easting (m)", ylabel = "Northing (m)",
             aspect = DataAspect())
    end
    if isnothing(gt)
        hm = heatmap!(ax, Zds; colormap = :viridis)
    else
        ex = raster_extents(Z, gt)
        hm = heatmap!(ax, range(ex.xmin, ex.xmax; length = Wd),
                          range(ex.ymin, ex.ymax; length = Hd),
                          permutedims(reverse(Zds; dims=1));
                          colormap = :viridis)
    end
    Colorbar(fig[1, 2], hm; label = "Density (0 = open, 1 = max canopy)")
    paths = _save_fig(fig, joinpath(dst, "kde_density_heatmap"); formats = formats)

    # Sidecar geotransform metadata
    sidecar = joinpath(dst, "kde_density.gt.json")
    _save_json(sidecar, Dict(
        "shape_HW"    => [H, W],
        "value_range" => [minimum(Z), maximum(Z)],
        "geotransform" => isnothing(gt) ? nothing : Dict(
            "x_origin" => gt.x_origin, "dx" => gt.dx, "x_rot" => gt.x_rot,
            "y_origin" => gt.y_origin, "y_rot" => gt.y_rot, "dy" => gt.dy,
        ),
        "crs_wkt"     => crs,
    ))
    push!(paths, sidecar)
    return paths
end

"""
    report_speed_map(dens::RasterGrid, strategy::SpeedStrategy, outdir;
                     gt=nothing, crs="") -> [paths]

Visualise the speed surface produced by `assign_speed.(strategy, dens.Z)`.
Useful for confirming "high density → low speed".
"""
function report_speed_map(dens::RasterGrid,
                            strategy::SpeedStrategy,
                            outdir::AbstractString;
                            gt::Union{Nothing, GeoTransform} = nothing,
                            crs::AbstractString = "",
                            max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                            formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    dst = _ensure_dir(joinpath(outdir, "kde"))
    H, W = size(dens.Z)
    stride = _plot_stride(H, W; max_preview_px = max_preview_px)
    Zds = _downsample(dens.Z, stride)
    Hd, Wd = size(Zds)
    speeds = [assign_speed(strategy, d) for d in Zds]

    title = "Assigned speed surface (m/s)" *
            (stride > 1 ? "  [thumbnail stride=$stride]" : "")
    fig = Figure(size = (900, 720))
    ax = if isnothing(gt)
        Axis(fig[1, 1]; title = title,
             aspect = DataAspect(), yreversed = true,
             xlabel = "col", ylabel = "row")
    else
        Axis(fig[1, 1]; title = title,
             xlabel = "Easting (m)", ylabel = "Northing (m)",
             aspect = DataAspect())
    end
    hm = if isnothing(gt)
        heatmap!(ax, permutedims(speeds);
                 colormap = ColorSchemes.cividis,
                 colorrange = speed_bounds(strategy))
    else
        ex = raster_extents(dens.Z, gt)
        heatmap!(ax,
                 range(ex.xmin, ex.xmax; length = Wd),
                 range(ex.ymin, ex.ymax; length = Hd),
                 permutedims(speeds);
                 colormap = ColorSchemes.cividis,
                 colorrange = speed_bounds(strategy))
    end
    Colorbar(fig[1, 2], hm; label = "Speed (m/s)  [high density → low speed]")
    paths = _save_fig(fig, joinpath(dst, "speed_map"); formats = formats)

    # Histogram of speeds
    fig2 = Figure(size = (700, 420))
    ax2 = Axis(fig2[1, 1]; title = "Speed distribution across the surface",
               xlabel = "Speed (m/s)", ylabel = "Pixel count")
    hist!(ax2, vec(speeds); bins = 32, color = :steelblue)
    append!(paths, _save_fig(fig2, joinpath(dst, "speed_map_histogram"); formats = formats))
    return paths
end

# ---------------------------------------------------------------------------
# 5. Waypoint visuals
# ---------------------------------------------------------------------------

"""
    report_waypoints_overlay(rs::GeoRasterStack, dens::RasterGrid,
                              wps_dict::AbstractDict, outdir) -> [paths]

Draw all mission waypoint tracks over the orthomosaic + density surface.
`wps_dict` maps mission label → `Vector{Waypoint}`.
"""
function report_waypoints_overlay(rs::GeoRasterStack,
                                    dens::RasterGrid,
                                    wps_dict::AbstractDict,
                                    outdir::AbstractString;
                                    max_preview_px::Integer = DEFAULT_MAX_PREVIEW_PX,
                                    georeference::Bool = true,
                                    formats::AbstractVector{<:AbstractString} = DEFAULT_REPORT_FORMATS)
    dst = _ensure_dir(joinpath(outdir, "waypoints"))
    H, W = size(rs)
    stride = _plot_stride(H, W; max_preview_px = max_preview_px)
    Zds = _downsample(rs.Z, stride)
    Hd, Wd = size(Zds)

    use_geo = georeference && (rs.gt.dx != 0 && rs.gt.dy != 0)
    title = "Mission waypoints over orthomosaic" *
            (stride > 1 ? "  [thumbnail stride=$stride]" : "")

    fig = Figure(size = (1100, max(280, 800 * Hd ÷ Wd)))
    ax = if use_geo
        ex0 = raster_extents(rs)
        Axis(fig[1, 1]; title = title,
             xlabel = "Easting (m)", ylabel = "Northing (m)",
             aspect = DataAspect(),
             limits = (ex0.xmin, ex0.xmax, ex0.ymin, ex0.ymax))
    else
        Axis(fig[1, 1]; title = title,
             aspect = DataAspect(), yreversed = true,
             xlabel = "col", ylabel = "row")
    end
    if use_geo
        ex = raster_extents(rs)
        image!(ax,
               (ex.xmin, ex.xmax),
               (ex.ymin, ex.ymax),
               permutedims(Zds, (2, 1)))
    else
        image!(ax, permutedims(Zds, (2, 1)))
    end

    palette = ColorSchemes.Set1_4
    for (i, (label, wps)) in enumerate(collect(wps_dict))
        xs = [w.x for w in wps]
        ys = [w.y for w in wps]
        col = palette[mod1(i, length(palette))]
        lines!(ax, xs, ys; color = col, linewidth = 1.2, label = String(label))
    end
    axislegend(ax; position = :rt, framevisible = false)
    paths = _save_fig(fig, joinpath(dst, "waypoints_overlay"); formats = formats)

    # Inter-waypoint spacing histogram
    fig2 = Figure(size = (900, 480))
    ax2 = Axis(fig2[1, 1]; title = "Inter-waypoint spacing per mission",
               xlabel = "Spacing (m or px)", ylabel = "Count")
    for (i, (label, wps)) in enumerate(collect(wps_dict))
        if length(wps) < 2; continue; end
        sp = [hypot(wps[j].x - wps[j-1].x, wps[j].y - wps[j-1].y)
              for j in 2:length(wps)]
        col = palette[mod1(i, length(palette))]
        hist!(ax2, sp; bins = 32, color = (col, 0.5), label = String(label))
    end
    axislegend(ax2; position = :rt, framevisible = false)
    append!(paths, _save_fig(fig2, joinpath(dst, "waypoint_spacing_hist"); formats = formats))
    return paths
end

# ---------------------------------------------------------------------------
# 6. Run summary: manifest, provenance, run report
# ---------------------------------------------------------------------------

"""
    report_run_manifest(outdir; descriptions::AbstractDict = Dict()) -> [path]

Walk `outdir` recursively and build `manifest.csv` listing every artefact:
relative path, type, size, stage (parent dir), tier (M/D/B if encoded by
filename suffix; default D), description.
"""
function report_run_manifest(outdir::AbstractString;
                              descriptions::AbstractDict = Dict{String, String}())
    rows = String[]
    push!(rows, "file,type,size_bytes,stage,description")
    for (root, _, files) in walkdir(outdir)
        for f in files
            full = joinpath(root, f)
            f == "manifest.csv" && continue
            rel = relpath(full, outdir)
            sz = filesize(full)
            typ = endswith(f, ".csv") ? "csv" :
                  endswith(f, ".json") ? "json" :
                  endswith(f, ".png") ? "png" :
                  endswith(f, ".pdf") ? "pdf" :
                  endswith(f, ".md") ? "md" :
                  endswith(f, ".tif") || endswith(f, ".tiff") ? "geotiff" :
                  endswith(f, ".txt") ? "txt" : "other"
            stage = splitpath(rel) |> first
            desc = get(descriptions, rel, "")
            push!(rows, "$rel,$typ,$sz,$stage,\"$desc\"")
        end
    end
    p = joinpath(outdir, "manifest.csv")
    open(p, "w") do io
        for ln in rows
            println(io, ln)
        end
    end
    return [p]
end

"""
    report_provenance(outdir; package_version="", run_inputs=nothing, extras...) -> [path]
"""
function report_provenance(outdir::AbstractString;
                            package_version::AbstractString = "",
                            run_inputs = nothing,
                            extras...)
    p = joinpath(outdir, "provenance.json")
    obj = Dict{String, Any}(
        "julia_version"   => string(VERSION),
        "package_version" => package_version,
        "run_started_at"  => string(Dates.now()),
        "uses_python"     => false,
    )
    if run_inputs !== nothing
        obj["run_inputs"] = Dict(
            "rgb"          => run_inputs.rgb,
            "gli"          => run_inputs.gli,
            "lidar_keys"   => collect(string.(keys(run_inputs.lidar))),
            "outdir"       => run_inputs.outdir,
            "tree_labels"  => run_inputs.tree_labels,
            "kde_kernel"   => string(run_inputs.kde_kernel),
            "speed_bounds" => collect(run_inputs.speed_bounds_mps),
            "line_spacing" => run_inputs.flightlines_spacing_m,
        )
    end
    for (k, v) in extras
        obj[string(k)] = v
    end
    return [_save_json(p, obj)]
end

"""
    report_run_md(outdir; sections::AbstractDict, title="Run report") -> [path]

Render `outdir/run_report.md` from a Dict of Markdown section name → content.
"""
function report_run_md(outdir::AbstractString;
                        sections::AbstractDict = Dict{String, String}(),
                        title::AbstractString = "Run report")
    p = joinpath(outdir, "run_report.md")
    open(p, "w") do io
        println(io, "# ", title)
        println(io)
        println(io, "**Generated:** ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        println(io)
        for (heading, body) in sections
            println(io, "## ", heading)
            println(io)
            println(io, body)
            println(io)
        end
    end
    return [p]
end

# ---------------------------------------------------------------------------
# Proposed-artefacts checklist (PR3/PR4/PR5 design doc traceability)
# ---------------------------------------------------------------------------

# Each entry: (relative-path-pattern, tier M|D|B, one-line description)
# tier M = manuscript candidate, D = diagnostic, B = both depending on data.
const PROPOSED_ARTIFACTS = [
    # Ingest
    ("ingest/rgb_preview.png",                        "D", "RGB preview (downsampled for large rasters)"),
    ("ingest/geotiff_metadata.json",                  "D", "Geotransform + CRS + extents"),
    ("ingest/rgb_with_grid_overlay.png",              "M", "UTM grid + GLI overlay on RGB (not yet implemented)"),
    ("ingest/gli_classes_overlay.png",                "M", "GLI cover-class overlay (not yet implemented)"),
    ("ingest/gli_class_histogram.csv",                "D", "Pixel counts per GLI class (not yet implemented)"),
    # Features / PCA
    ("features/lab_channels.png",                     "D", "L*, a*, b* channel small-multiples (not yet wired into run_from_config)"),
    ("features/feature_distributions.png",            "D", "Standardised feature histograms (not yet implemented)"),
    ("features/feature_matrix.csv",                   "D", "Subsampled feature matrix (not yet implemented)"),
    ("features/pca_explained_variance.png",           "M", "PCA scree + cumulative-variance curve"),
    ("features/pca_loadings.csv",                     "D", "PCA component loadings (not yet implemented)"),
    ("features/pca_scatter_pc1_pc2.png",              "M", "PC1-vs-PC2 scatter coloured by cluster (not yet implemented)"),
    ("features/pca_metadata.json",                    "D", "PCA component metadata (not yet implemented)"),
    # Clustering
    ("cluster/cluster_quality_metrics.csv",           "M", "k-medoids quality sweep CSV (auto-saved by build_mask path)"),
    ("cluster/cluster_quality_sweep.png",             "M", "4-panel sweep figure"),
    ("cluster/cluster_overlay_k*.png",                "M", "Per-cluster alpha overlays"),
    ("cluster/cluster_lab_summary.csv",               "D", "Per-cluster mean L*/a*/b* + GLI overlap"),
    ("cluster/cluster_gli_overlap.json",              "D", "Per-cluster GLI overlap (when GLI shape matches)"),
    ("cluster/tree_label_decision.md",                "M", "Audit trail of manual tree_labels choice"),
    ("cluster/labels_full.bin",                       "D", "Per-pixel label dump (not yet implemented)"),
    ("cluster/cluster_map.png",                       "M", "Full-res cluster-id pseudocolour (not yet implemented)"),
    # KDE / speed
    ("kde/kde_density_heatmap.png",                   "M", "KDE density heatmap (UTM axes if GeoTIFF)"),
    ("kde/kde_density.tif",                           "M", "Co-registered KDE GeoTIFF (preserves CRS)"),
    ("kde/kde_density.gt.json",                       "D", "GT+CRS sidecar"),
    ("kde/kde_density_metadata.json",                 "D", "Kernel + bandwidth + range (not yet wired in v0.5.2 hotfix)"),
    ("kde/kde_kernel_profile.png",                    "D", "1-D kernel cross-section (not yet implemented)"),
    ("kde/kde_density_contour_overlay.png",           "M", "Density contours on RGB (not yet implemented)"),
    ("kde/speed_map.png",                             "M", "Speed surface (UTM, cividis, low-speed at high density)"),
    ("kde/speed_map_histogram.png",                   "D", "Speed-value histogram"),
    ("kde/kde_class_map.png",                         "D", "Multi-Otsu KDE-class map (not yet wired)"),
    # Waypoints
    ("waypoints/waypoints_*.csv",                     "M", "Per-mission waypoint CSV (x,y,altitude,speed,line_id)"),
    ("waypoints/waypoints_overlay.png",               "M", "All-mission waypoint tracks over RGB (UTM)"),
    ("waypoints/waypoint_spacing_hist.png",           "D", "Inter-waypoint spacing histogram"),
    ("waypoints/curvature_diagnostic.png",            "D", "Curvature-spaced step diagnostic (not yet implemented)"),
    ("waypoints/waypoints_summary.csv",               "M", "Per-mission speed/duration summary (not yet wired in v0.5.2 hotfix)"),
    # LiDAR (PR5) — only emitted when [paths].lidar populated AND GLI present
    ("lidar/counts.json",                             "M", "LAS → cover-stratified count grids (counts.json schema)"),
    ("lidar/stats_and_coverage.csv",                  "M", "Manuscript long table: Q1/Q2/Q3/IQR/Mean + CR/CV/Gini/MoranI"),
    ("lidar/percent_densities.csv",                   "M", "Manuscript wide table: % cells per count bin"),
    ("lidar/lidar_metadata.json",                     "D", "Per-LAS metadata (not yet implemented)"),
    ("lidar/lidar_extent_overlay.png",                "D", "LAS bounding-box overlay (not yet implemented)"),
    ("lidar/return_classification_overlay.png",       "D", "LAS classification heatmap (not yet implemented)"),
    ("lidar/density_per_cover_violins.png",           "M", "Violin plots per cover×mission×return (not yet implemented)"),
    # Trajectory / bootstrap (already present in src/, not yet wired into run_from_config)
    ("bootstrap/bootstrap_cr_main.png",               "M", "Bootstrap CR CI figure (legacy path)"),
    ("bootstrap/bootstrap_cr_difference_cis.csv",     "M", "Bootstrap CR-difference CIs (legacy path)"),
    ("trajectory/tracking_metrics_refined.png",       "M", "Tracking metrics figure (legacy path)"),
    ("trajectory/cleaned_segments_overlay.png",       "M", "Cleaned segments on RGB (not yet implemented)"),
    ("kde_strata/within_cover_cr.png",                "M", "Within-cover CR strata figure (legacy path)"),
    ("along_track/along_track_cr_lines.png",          "M", "Per-mission CR along survey strip (not yet implemented)"),
    # Run summary
    ("manifest.csv",                                  "M", "Full artefact manifest"),
    ("provenance.json",                               "M", "Julia/package versions + run digest"),
    ("run_report.md",                                 "M", "Auto-rendered run narrative"),
    ("proposed_artifacts.csv",                        "M", "This checklist"),
]

"""
    report_proposed_artifacts_checklist(outdir) -> [path]

Walk the proposed artefacts list (`PROPOSED_ARTIFACTS`) and emit
`outdir/proposed_artifacts.csv` recording which entries were produced in
this run and which are intentionally skipped / not-yet-implemented.

Status values:
- `produced`         — file exists at the expected path
- `produced_glob`    — at least one file matches the glob pattern
- `skipped`          — the description mentions "not yet implemented" /
                       "not yet wired" / "legacy path"
- `missing`          — expected but absent (treat as a regression)
"""
function report_proposed_artifacts_checklist(outdir::AbstractString)
    p = joinpath(outdir, "proposed_artifacts.csv")
    open(p, "w") do io
        println(io, "path_pattern,tier,status,description")
        for (pattern, tier, desc) in PROPOSED_ARTIFACTS
            full = joinpath(outdir, pattern)
            status = if occursin("not yet", lowercase(desc)) || occursin("legacy path", lowercase(desc))
                "skipped"
            elseif occursin("*", pattern)
                # Glob match
                dir = joinpath(outdir, dirname(pattern))
                base = basename(pattern)
                rx = Regex("^" * replace(base, "*" => ".*") * "\$")
                isdir(dir) && !isempty(filter(f -> occursin(rx, f), readdir(dir))) ?
                    "produced_glob" : "missing"
            else
                isfile(full) ? "produced" : "missing"
            end
            println(io, "$pattern,$tier,$status,\"$desc\"")
        end
    end
    return [p]
end
