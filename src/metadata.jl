"""
    metadata.jl — Planning metadata export and cluster-overlay workflow

## Purpose

Two responsibilities:

1. **Planning metadata JSON/TOML** (`export_planning_metadata`): writes a
   structured record of every pipeline decision that influences the output
   waypoints — selected k, vegetation labels, feature representation, PCA
   settings, kernel, bandwidth rule, speed bounds, line spacing,
   seconds_per_wp, spacing bounds, image source, mask source, and a
   created timestamp.  This file is the primary reproducibility artefact.

2. **Cluster-overlay / tree-label workflow** (`save_cluster_overlays`,
   `interactive_tree_labels`, `confirm_tree_labels`): because the pipeline
   must know which cluster labels correspond to vegetation, this module
   provides:
   - A *noninteractive* mode that writes one PNG overlay per cluster onto
     the orthomosaic thumbnail and **errors** with a clear message if
     `tree_labels` is not provided.  This is the path used in automated
     tests and CI.
   - An *interactive* mode (requires a TTY) that cycles through overlays
     and prompts the user to select vegetation clusters.

## Usage (non-interactive / CI)

    save_cluster_overlays(img, labels_full, H, W;
                          out_dir="output/cluster_overlays",
                          prefix="cluster")
    # Then pass confirmed tree_labels to build_mask_from_image:
    mask_grid, info = build_mask_from_image(img; k=..., tree_labels=[2,3])

## Usage (interactive / human in the loop)

    chosen = interactive_tree_labels(img, labels_full, H, W;
                                     out_dir="output/cluster_overlays")
    mask_grid, info = build_mask_from_image(img; k=..., tree_labels=chosen)
"""

# Dates is imported in the module-level KDEFlightPlanning.jl
# (using Dates is a stdlib and does not need re-importing here)

# ---------------------------------------------------------------------------
# Planning metadata export
# ---------------------------------------------------------------------------

"""
    export_planning_metadata(path::AbstractString;
        selected_k, tree_labels, feature_repr, pca_settings,
        kernel, bandwidth_rule, speed_bounds, line_spacing_m,
        seconds_per_wp, spacing_bounds_m,
        image_source="", mask_source="",
        notes="") -> path

Write a JSON file recording all pipeline decisions that control output waypoints.

This is the primary reproducibility artefact: a reader with this file plus
the source code can exactly re-run the pipeline.

Parameters
----------
- `selected_k`:       Integer k chosen by the sweep.
- `tree_labels`:      Vector{Int} of cluster labels mapped to vegetation.
- `feature_repr`:     String description, e.g. `"CIELAB (L, a, b)"`.
- `pca_settings`:     NamedTuple or Dict with `use_pca`, `variance_ratio`, `maxoutdim`.
- `kernel`:           String, e.g. `"epanechnikov"`.
- `bandwidth_rule`:   String, e.g. `"silverman_scott_indices"`.
- `speed_bounds`:     NamedTuple/Dict with `vmin` and `vmax` (m/s).
- `line_spacing_m`:   Flight-line spacing in metres (paper default: 40).
- `seconds_per_wp`:   Time budget per waypoint in seconds for constant/KDE
                      strategies. Note: 4 s was used in a now-removed
                      speed-troubleshooting step; the retained KDE-guided
                      planning uses the value stored here.
- `spacing_bounds_m`: NamedTuple/Dict with `min` and `max` (m).
- `image_source`:     Path or description of the input orthomosaic.
- `mask_source`:      Path or description of the binary mask (if pre-built).
- `notes`:            Free-form string for any additional context.

Returns the written path.
"""
function export_planning_metadata(path::AbstractString;
                                   selected_k       ::Int,
                                   tree_labels      ::AbstractVector{<:Integer},
                                   feature_repr     ::AbstractString,
                                   pca_settings,
                                   kernel           ::AbstractString,
                                   bandwidth_rule   ::AbstractString,
                                   speed_bounds,
                                   line_spacing_m   ::Real,
                                   seconds_per_wp   ::Real,
                                   spacing_bounds_m,
                                   image_source     ::AbstractString = "",
                                   mask_source      ::AbstractString = "",
                                   notes            ::AbstractString = "")
    mkpath(dirname(abspath(path)))

    # Normalise pca_settings, speed_bounds, spacing_bounds_m to plain Dicts
    _to_dict(x::AbstractDict) = Dict(string(k)=>v for (k,v) in x)
    _to_dict(x::NamedTuple)   = Dict(string(k)=>v for (k,v) in pairs(x))
    _to_dict(x)               = Dict("value" => x)

    meta = Dict(
        "pipeline"        => "KDEFlightPlanning.jl",
        "paper_title"     => "KDE-Guided Offline Variable-Speed Flight Planning " *
                             "for UAV LiDAR in Forested Terrain",
        "created"         => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
        "selected_k"      => selected_k,
        "tree_labels"     => collect(tree_labels),
        "feature_repr"    => feature_repr,
        "pca_settings"    => _to_dict(pca_settings),
        "kernel"          => kernel,
        "bandwidth_rule"  => bandwidth_rule,
        "speed_bounds"    => _to_dict(speed_bounds),
        "line_spacing_m"  => Float64(line_spacing_m),
        "seconds_per_wp"  => Float64(seconds_per_wp),
        "seconds_per_wp_note" => (
            "seconds_per_wp controls the time-budget spacing for ConstantSpeed " *
            "and KDEGuidedSpeed strategies. " *
            "The value 4 s belonged to a removed speed-troubleshooting workflow " *
            "and is NOT used in the retained KDE-guided planning. " *
            "The retained pipeline uses the value stored above."
        ),
        "spacing_bounds_m" => _to_dict(spacing_bounds_m),
        "image_source"    => image_source,
        "mask_source"     => mask_source,
        "notes"           => notes,
    )

    open(path, "w") do io
        JSON.print(io, meta, 2)
    end
    return path
end

# ---------------------------------------------------------------------------
# Waypoint spacing note (for paper methods documentation)
# ---------------------------------------------------------------------------

"""
    waypoint_spacing_note(waypoints_csv_path::AbstractString) -> String

Read a waypoints CSV and estimate the actual waypoint spacing statistics.
Returns a formatted note string suitable for inclusion in pipeline metadata
or manuscript methods.

This function addresses the audit question: what spacing did the retained
KDE-guided waypoint files actually use?

Outputs median, mean, and the [5th, 95th] percentile range of inter-waypoint
distances from the file.
"""
function waypoint_spacing_note(waypoints_csv_path::AbstractString)
    isfile(waypoints_csv_path) || return "(file not found: $waypoints_csv_path)"
    df = CSV.read(waypoints_csv_path, DataFrame)
    xcol = "x" in names(df) ? "x" : first(names(df))
    ycol = "y" in names(df) ? "y" : names(df)[2]
    xs   = Float64.(df[!, xcol])
    ys   = Float64.(df[!, ycol])
    n    = length(xs)
    n < 2 && return "(fewer than 2 waypoints in file)"

    dists = [hypot(xs[i+1]-xs[i], ys[i+1]-ys[i]) for i in 1:n-1]
    # Filter out long transit jumps (> 3× median) that are line-change transits
    med   = median(dists)
    survey_dists = filter(d -> d <= 3*med, dists)
    isempty(survey_dists) && (survey_dists = dists)

    p5    = quantile(survey_dists, 0.05)
    p95   = quantile(survey_dists, 0.95)
    μ     = mean(survey_dists)
    mdn   = median(survey_dists)

    return """
Waypoint spacing note (from $waypoints_csv_path):
  N waypoints         : $n
  Median spacing      : $(round(mdn; digits=2)) m
  Mean spacing        : $(round(μ; digits=2)) m
  5th–95th pctile     : $(round(p5; digits=2)) – $(round(p95; digits=2)) m
  Note: transit jumps (>3× median) excluded from survey-segment statistics.
  The value seconds_per_wp=4 belonged to a removed speed-troubleshooting
  workflow step. The retained KDE-guided planner uses the explicit
  seconds_per_wp passed to generate_waypoints (see export_planning_metadata).
"""
end

# ---------------------------------------------------------------------------
# Cluster overlay helpers
# ---------------------------------------------------------------------------

"""
    save_cluster_overlays(img, labels_full::AbstractVector{Int}, H::Int, W::Int;
                           out_dir::AbstractString = "output/cluster_overlays",
                           prefix::AbstractString  = "cluster",
                           alpha::Float64          = 0.45)

Write one PNG overlay image per unique cluster label.  Each overlay shows the
input orthomosaic thumbnail with the pixels of that cluster highlighted in a
distinct colour.

**Noninteractive path** (CI / automated tests): call this first, then inspect
the saved PNGs and pass the confirmed `tree_labels` to `build_mask_from_image`.
If `tree_labels` is not subsequently provided, call `require_tree_labels` to
raise a descriptive error.

Returned value: `Vector{String}` of written PNG paths.

Notes
-----
- Requires Images.jl and FileIO.jl to be loaded in the caller's session
  (they are not hard dependencies to keep the CI environment lean).
- If neither is available, prints a message and returns an empty vector.
"""
function save_cluster_overlays(img, labels_full::AbstractVector{Int},
                                H::Int, W::Int;
                                out_dir::AbstractString = "output/cluster_overlays",
                                prefix ::AbstractString = "cluster",
                                alpha  ::Float64        = 0.45)
    mkpath(out_dir)
    ks = sort(unique(labels_full))

    # Cluster highlight colours (colorblind-safe)
    cluster_colors_rgb = [
        (0x20/255, 0x80/255, 0x8D/255),   # teal
        (0xA8/255, 0x4B/255, 0x2F/255),   # rust/orange
        (0x1B/255, 0x47/255, 0x4D/255),   # dark teal
        (0xFF/255, 0xC5/255, 0x53/255),   # gold
        (0x94/255, 0x44/255, 0x54/255),   # mauve
        (0x84/255, 0x84/255, 0x56/255),   # olive
        (0x6E/255, 0x52/255, 0x2B/255),   # brown
        (0xBC/255, 0xE2/255, 0xE7/255),   # light cyan
    ]

    written = String[]

    # Check for Images.jl / FileIO.jl (optional heavy deps, not in [deps])
    can_write = isdefined(Main, :Images) || isdefined(Main, :FileIO)

    # Whether img is a 2D colorant matrix (loaded via FileIO) or a raw array
    is_colorant_matrix = img isa AbstractMatrix{<:ColorTypes.Colorant}

    # Reshape labels
    lab_mat = reshape(labels_full, H, W)

    for (ci, kval) in enumerate(ks)
        out_path = joinpath(out_dir, "$(prefix)_k$(kval).png")

        if !can_write || !is_colorant_matrix
            # Write a simple ASCII cluster summary instead of a PNG
            n_pix   = count(==(kval), labels_full)
            txt_path = out_path * ".txt"
            open(txt_path, "w") do io
                println(io, "Cluster overlay: k=$kval")
                println(io, "  N pixels : $n_pix ($(round(100n_pix/length(labels_full); digits=1))%)")
                println(io, "  H x W    : $(H) x $(W)")
                println(io, "")
                println(io, "To generate PNG overlays, load FileIO.jl and Images.jl in the")
                println(io, "caller's session before calling save_cluster_overlays, and pass")
                println(io, "a Matrix{<:Colorant} image (e.g. loaded via FileIO.load).")
            end
            push!(written, txt_path)
            continue
        end

        # Build overlay using Images.jl (caller must have loaded FileIO)
        col = cluster_colors_rgb[mod1(ci, length(cluster_colors_rgb))]
        cr_val, cg_val, cb_val = col

        # Composite: blend cluster pixels with the highlight colour
        out_img = copy(img)  # img is Matrix{<:Colorant}
        for row in 1:H, col_idx in 1:W
            if lab_mat[row, col_idx] == kval
                p    = img[row, col_idx]
                pr   = Float64(ColorTypes.red(p))
                pg   = Float64(ColorTypes.green(p))
                pb   = Float64(ColorTypes.blue(p))
                nr   = (1-alpha)*pr + alpha*cr_val
                ng   = (1-alpha)*pg + alpha*cg_val
                nb   = (1-alpha)*pb + alpha*cb_val
                out_img[row, col_idx] = RGB{Float32}(Float32(nr), Float32(ng), Float32(nb))
            end
        end

        Main.FileIO.save(out_path, out_img)
        push!(written, out_path)
    end

    println("Cluster overlays written to: $out_dir")
    println("Review each cluster and identify which correspond to vegetation/trees.")
    println("Then pass tree_labels=[k1, k2, ...] to build_mask_from_image.")
    return written
end

"""
    require_tree_labels(tree_labels; context="") -> nothing

If `tree_labels` is `nothing` or empty, throw an `ArgumentError` with a
descriptive message that explains the noninteractive workflow.

Call this after `save_cluster_overlays` in any automated path to ensure the
pipeline never silently uses a default label without explicit confirmation.
"""
function require_tree_labels(tree_labels; context::AbstractString="")
    if isnothing(tree_labels) || isempty(tree_labels)
        ctx = isempty(context) ? "" : " (context: $context)"
        error("""
AutomaticTreeLabelRequired$ctx:

  tree_labels must be provided explicitly. This pipeline cannot automatically
  determine which clusters correspond to vegetation/trees without human review.

  Workflow:
    1. Run save_cluster_overlays(img, labels_full, H, W; out_dir="...")
       to write one PNG overlay per cluster.
    2. Inspect the PNGs and identify which cluster numbers show trees/canopy.
    3. Re-run build_mask_from_image (or build_mask_autok) with the confirmed
       tree_labels, for example:
         mask_grid, info = build_mask_from_image(img; k=..., tree_labels=[2, 4])
    4. All selected labels are combined into a single binary vegetation mask.

  In automated tests, pass a valid tree_labels vector derived from a prior
  interactive run, or use the default k=2, tree_labels=[1] and document the
  assumption in your reproducibility notes.
""")
    end
    nothing
end

"""
    interactive_tree_labels(img, labels_full::AbstractVector{Int}, H::Int, W::Int;
                             out_dir::AbstractString = "output/cluster_overlays",
                             prefix ::AbstractString = "cluster")
        -> Vector{Int}

Interactive workflow (requires a TTY) for selecting vegetation cluster labels.

Steps:
1. Saves one PNG overlay per cluster to `out_dir`.
2. Prompts the user to enter comma-separated cluster numbers for vegetation.
3. Returns the confirmed `Vector{Int}` of tree labels.

Falls back gracefully in non-TTY environments by printing instructions and
returning an empty vector (caller should then call `require_tree_labels`).
"""
function interactive_tree_labels(img, labels_full::AbstractVector{Int},
                                  H::Int, W::Int;
                                  out_dir::AbstractString = "output/cluster_overlays",
                                  prefix ::AbstractString = "cluster")
    written = save_cluster_overlays(img, labels_full, H, W;
                                     out_dir=out_dir, prefix=prefix)

    ks = sort(unique(labels_full))
    println("\nCluster overlay PNGs written:")
    for p in written; println("  $p"); end
    println("\nAvailable clusters: $(ks)")
    println("Enter comma-separated cluster numbers for VEGETATION/TREES (e.g. 2,4): ")

    if !isatty(stdin)
        @warn "interactive_tree_labels: stdin is not a TTY. " *
              "Returning empty tree_labels. " *
              "Call require_tree_labels() to validate before proceeding."
        return Int[]
    end

    line = strip(readline())
    if isempty(line)
        @warn "No labels entered. Returning empty tree_labels."
        return Int[]
    end

    chosen = Int[]
    for tok in split(line, ',')
        tok2 = strip(tok)
        isempty(tok2) && continue
        try
            k = parse(Int, tok2)
            k in ks || @warn "Cluster $k not in available clusters $ks"
            push!(chosen, k)
        catch
            @warn "Could not parse '$tok2' as an integer — skipping."
        end
    end
    println("Selected vegetation labels: $chosen")
    return chosen
end
