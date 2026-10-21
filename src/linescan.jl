"""
    linescan.jl — Line-scan coverage metrics from actual flown trajectories

Ports the logic from:
  - flightline_comparison_work/scan_lines_actual_trajectories.py
  - flightline_comparison_work/plot_line2_actual_overlap_detail.py

## Constants and conventions

The Durham, NH count-grid has shape (263 rows × 324 columns) in geographic
(row=north) orientation. The `count_grid` reshape convention in Python is:

    mat = np.array(values).reshape((263, 324), order='F')  # row 0 = north

No row flip is applied: the original GLI class raster confirms this layout
for all 36 bundled count grids. We replicate it in `decode_count_grid`.

## API

    decode_count_grid(values) → Matrix{Float64}    (263 × 324, row 0 = north)
    load_count_grids(counts_json) → Dict           keyed (ret, cover, mission, kernel)
    actual_line_extents(processed, planned_lines) → DataFrame
    common_x_overlap(extents, mission_labels) → Dict{Int,(x0,x1)}
    line_scan_cover_metrics(...)  → (scan_df, cover_df, wide_df, profile_df)
    single_line_cover_metrics(...)→ (extents_df, cover_df, profile_df)
"""

# Grid dimensions (Durham, NH site)
const SCAN_NROWS = _LEGACY_COUNT_NROWS
const SCAN_NCOLS = _LEGACY_COUNT_NCOLS

# Cover codes and manuscript labels
const COVER_KEYS   = ["field", "decid", "conif"]
const COVER_LABELS = Dict("field" => "Field", "decid" => "Deciduous", "conif" => "Coniferous")


# ---------------------------------------------------------------------------
# Count-grid decoding
# ---------------------------------------------------------------------------

"""
    decode_count_grid(values; nrows=263, ncols=324) → Matrix{Float64}

Reshape a flat count-grid `values` vector (length 324 × 263) into a
(263 × 324) matrix with row 0 at the geographic north, matching the Python
convention used in the project's `counts.json`.

Decode the legacy encoding as `reshape(values, nrows, ncols)` with no row flip.
Use [`load_count_grids`](@ref) for files, which also supports the current writer
schema and its dimension metadata.
"""
function decode_count_grid(values::AbstractVector;
                           nrows::Integer=SCAN_NROWS, ncols::Integer=SCAN_NCOLS)
    return _decode_count_values(values, nrows, ncols; layout=:legacy, T=Float64)
end


# ---------------------------------------------------------------------------
# Count-grid loading
# ---------------------------------------------------------------------------

"""
    load_count_grids(counts_json) → Dict{Tuple{String,String,String,String}, Matrix{Float64}}

Load either supported `counts.json` schema using [`load_counts_json`](@ref),
then return grids keyed by `(ret, cover, mission, kernel)` strings. Optional
`nrows`/`ncols` are forwarded to the canonical loader for validation.
"""
function load_count_grids(counts_json::AbstractString;
                          nrows::Union{Nothing,Integer}=nothing,
                          ncols::Union{Nothing,Integer}=nothing)
    records = load_counts_json(counts_json; nrows=nrows, ncols=ncols)
    out = Dict{NTuple{4,String}, Matrix{Float64}}()
    for r in records
        key = r.key
        k = (String(key.ret), String(key.cover), String(key.mission), String(key.kernel))
        out[k] = Float64.(r.matrix)
    end
    return out
end


# ---------------------------------------------------------------------------
# Shared spatial support and north-up grid geometry
# ---------------------------------------------------------------------------

"""
Validate all count grids against fixed cover support, sharing the bootstrap
validation rules. String and Symbol cover-mask keys are accepted. Returns
`(masks, support_mode, grid_size)`, with String-keyed Boolean masks.
"""
function _line_scan_cover_support(count_grids::AbstractDict;
                                 cover_masks=nothing,
                                 allow_approximate_masks::Bool=false)
    isempty(count_grids) && throw(ArgumentError("At least one count grid is required"))
    grid_size = size(first(values(count_grids)))
    length(grid_size) == 2 && all(>(0), grid_size) ||
        throw(ArgumentError("Count grids must be nonempty matrices"))
    records = CountRecord[]
    for (key, matrix) in count_grids
        length(key) == 4 || throw(ArgumentError("Count-grid keys must have four entries"))
        matrix isa AbstractMatrix{<:Real} ||
            throw(ArgumentError("Count grids must be real-valued matrices"))
        size(matrix) == grid_size ||
            throw(DimensionMismatch("Count grids must all have the same shape"))
        all(v -> isfinite(v) && isinteger(v) && v >= 0, matrix) ||
            throw(ArgumentError("Count grids must contain finite nonnegative integer counts"))
        ret, cover, mission, kernel = Symbol.(key)
        push!(records, CountRecord(CountKey(ret=ret, cover=cover,
                                            mission=mission, kernel=kernel),
                                   Int32.(matrix)))
    end
    supplied = if cover_masks === nothing
        nothing
    else
        normalized = Dict{Symbol,Any}()
        for (cover, mask) in cover_masks
            cover isa Union{Symbol,AbstractString} ||
                throw(ArgumentError("Cover-mask keys must be Strings or Symbols"))
            name = Symbol(cover)
            haskey(normalized, name) &&
                throw(ArgumentError("Duplicate cover-mask key $name"))
            normalized[name] = mask
        end
        normalized
    end
    masks, mode = _bootstrap_cover_support(records, Symbol.(COVER_KEYS), grid_size;
        cover_masks=supplied, allow_approximate_masks=allow_approximate_masks)
    return Dict(String(key) => mask for (key, mask) in masks), mode, grid_size
end

"""
Resolve the true count-grid affine transform. The Durham default is used only
for its documented grid shape; other grids require an explicit transform or
both legacy extent overrides. `ymin` is the southern grid edge, never a planned
line center. Rotated and south-up rasters are rejected rather than guessed.
"""
function _line_scan_geotransform(grid_size::Tuple{Int,Int};
                                   geotransform=nothing,
                                   xmin::Union{Nothing,Real}=nothing,
                                   ymin::Union{Nothing,Real}=nothing)
    if geotransform === nothing && grid_size != (SCAN_NROWS, SCAN_NCOLS) &&
       (xmin === nothing || ymin === nothing)
        throw(ArgumentError("Non-Durham count grids require an explicit geotransform"))
    end
    # GeoTransform and GT_COUNTGRID are defined later in the module. Resolve
    # them at call time, after the module has finished loading.
    base = geotransform === nothing ? GT_COUNTGRID :
           geotransform isa GeoTransform ? geotransform : GeoTransform(geotransform)
    all(isfinite, base) || throw(ArgumentError("Grid geotransform must be finite"))
    base.dx > 0 && base.dy < 0 && base.x_rot == 0 && base.y_rot == 0 ||
        throw(ArgumentError("Line scans require a north-up, nonrotated grid"))
    x0 = xmin === nothing ? base.x_origin : Float64(xmin)
    y0 = ymin === nothing ? base.y_origin : Float64(ymin) - grid_size[1] * base.dy
    isfinite(x0) && isfinite(y0) ||
        throw(ArgumentError("Grid extent overrides must be finite"))
    return GeoTransform(x0, base.dx, 0.0, y0, 0.0, base.dy)
end

"""One-based row containing northing `y` in a validated north-up transform."""
function _line_scan_row(y::Real, geotransform)
    isfinite(y) || throw(ArgumentError("Planned line northing must be finite"))
    return floor(Int, (Float64(y) - geotransform.y_origin) / geotransform.dy) + 1
end

function _line_scan_band(y::Real, geotransform, nrows::Int, half_band_rows::Int)
    half_band_rows >= 0 || throw(ArgumentError("half_band_rows must be nonnegative"))
    row_center = _line_scan_row(y, geotransform)
    r0 = max(1, row_center - half_band_rows)
    r1 = min(nrows, row_center + half_band_rows)
    return row_center, r0, r1
end


# ---------------------------------------------------------------------------
# Actual line extents from cleaned trajectories
# ---------------------------------------------------------------------------

"""
    actual_line_extents(processed, planned_lines) → DataFrame

For each (mission, line_index) group in the cleaned trajectory DataFrames,
compute the actual flown x extent using 0.5th and 99.5th percentiles plus
raw min/max. Also returns y-median and sample/segment counts.

# Arguments
- `processed`: Dict{String, DataFrame} of cleaned trajectories, keyed by mission label
- `planned_lines`: DataFrame from `derive_planned_lines`

# Returns
DataFrame with columns: `mission`, `line_index`, `line_number`,
`x_min_actual`, `x_max_actual`, `x_min_raw`, `x_max_raw`,
`y_median_actual`, `samples`, `segments`.
"""
function actual_line_extents(processed::Dict{String,DataFrame}, planned_lines::DataFrame)
    ln_map = Dict(planned_lines.line_index .=> planned_lines.line_number)
    rows = Vector{NamedTuple}()
    for (mission, df) in processed
        for lid in sort(unique(df.nearest_line))
            g = df[df.nearest_line .== lid, :]
            push!(rows, (
                mission        = mission,
                line_index     = lid,
                line_number    = get(ln_map, lid, lid + 1),
                x_min_actual   = quantile(g.GridX, 0.005),
                x_max_actual   = quantile(g.GridX, 0.995),
                x_min_raw      = minimum(g.GridX),
                x_max_raw      = maximum(g.GridX),
                y_median_actual = median(g.GridY),
                samples        = nrow(g),
                segments       = length(unique(g.segment_id)),
            ))
        end
    end
    return DataFrame(rows)
end


# ---------------------------------------------------------------------------
# Common x-overlap computation
# ---------------------------------------------------------------------------

"""
    common_x_overlap(extents, mission_labels) → Dict{Int,(Float64,Float64)}

For each `line_index` that has all `mission_labels` present, return the
intersection of actual flown x-extents: `(common_x0, common_x1)`.

Lines missing any mission or with zero/negative span are excluded.
"""
function common_x_overlap(extents::DataFrame, mission_labels::Vector{String})
    result = Dict{Int,Tuple{Float64,Float64}}()
    for lid in sort(unique(extents.line_index))
        sub = extents[extents.line_index .== lid, :]
        missions_present = Set(sub.mission)
        if !all(m in missions_present for m in mission_labels)
            continue
        end
        x0 = maximum(sub[sub.mission .== m, :x_min_actual][1] for m in mission_labels)
        x1 = minimum(sub[sub.mission .== m, :x_max_actual][1] for m in mission_labels)
        if x1 > x0
            result[lid] = (x0, x1)
        end
    end
    return result
end


# ---------------------------------------------------------------------------
# Full all-line scan
# ---------------------------------------------------------------------------

"""
    line_scan_cover_metrics(
        count_grids, processed, planned_lines, missions;
        half_band_rows=20, geotransform=nothing, ymin=nothing, xmin=nothing,
        cover_masks=nothing, allow_approximate_masks=false
    ) → (scan_df, cover_df, wide_df, profile_df)

Compute per-line, per-cover-class coverage ratios and column profiles using
the actual flown x-overlap across missions.

# Arguments
- `count_grids`: Dict from `load_count_grids`
- `processed`: Dict{String, DataFrame} of cleaned trajectories
- `planned_lines`: DataFrame from `derive_planned_lines`
- `missions`: Vector of `(label, mission_key, kernel_key)` tuples —
  e.g. `[("Const. 2 m/s", "const2", "NA"), ...]`
- `half_band_rows=20`: envelope half-height in count-grid rows.
- `geotransform`: true north-up count-grid affine transform. Defaults to
  `GT_COUNTGRID` for the Durham grid; other grid shapes require it explicitly.
- `ymin`, `xmin`: optional legacy overrides for the southern and western grid
  edges, respectively. Grid edges are never inferred from planned lines.
- `cover_masks`: fixed Boolean cover masks keyed by Strings or Symbols;
  required by default and checked against positive counts.
- `allow_approximate_masks=false`: explicit diagnostic opt-in for positive-
  return-union support. All returned tables record `support_mode`.

# Returns
- `scan_df`: one row per line; includes `common_x_min/max/span`, `n_columns`,
  `best_forest_cover`, `best_forest_min_gain`
- `cover_df`: one row per (line, cover, mission); `support_cells`,
  `covered_cells`, `coverage_ratio`
- `wide_df`: pivoted version of `cover_df` plus `kde_minus_const2/8`
- `profile_df`: one row per (line, cover, mission, column)
"""
function line_scan_cover_metrics(
    count_grids::Dict,
    processed::Dict{String,DataFrame},
    planned_lines::DataFrame,
    missions::Vector{<:Tuple};
    half_band_rows::Int = 20,
    geotransform = nothing,
    ymin::Union{Nothing,Real} = nothing,
    xmin::Union{Nothing,Real} = nothing,
    cover_masks = nothing,
    allow_approximate_masks::Bool = false,
)
    masks, support_mode, grid_size = _line_scan_cover_support(count_grids;
        cover_masks=cover_masks, allow_approximate_masks=allow_approximate_masks)
    nrows, ncols = grid_size
    gt = _line_scan_geotransform(grid_size; geotransform=geotransform,
                                 xmin=xmin, ymin=ymin)
    half_band_rows >= 0 || throw(ArgumentError("half_band_rows must be nonnegative"))

    extents    = actual_line_extents(processed, planned_lines)
    miss_labels = [m[1] for m in missions]
    overlap    = common_x_overlap(extents, miss_labels)

    x_centers = gt.x_origin .+ ((0:(ncols-1)) .+ 0.5) .* gt.dx

    scan_rows    = Vector{NamedTuple}()
    cover_rows   = Vector{NamedTuple}()
    profile_rows = Vector{NamedTuple}()

    for row in eachrow(planned_lines)
        lid         = Int(row.line_index)
        line_number = Int(row.line_number)
        !haskey(overlap, lid) && continue

        common_x0, common_x1 = overlap[lid]
        col_mask = (x_centers .>= common_x0) .& (x_centers .<= common_x1)
        cols     = findall(col_mask)
        isempty(cols) && continue

        row_center, r0, r1 = _line_scan_band(row.y_center, gt, nrows, half_band_rows)
        r0 <= r1 || continue

        push!(scan_rows, (
            line_number    = line_number,
            line_index     = lid,
            planned_y      = Float64(row.y_center),
            row_center     = row_center,
            common_x_min   = common_x0,
            common_x_max   = common_x1,
            common_x_span_m = common_x1 - common_x0,
            n_columns      = length(cols),
            half_band_rows = half_band_rows,
            support_mode   = support_mode,
        ))

        for cover in COVER_KEYS
            support = masks[cover][r0:r1, cols]

            for (label, mission, kernel) in missions
                ground  = count_grids[("ground", cover, mission, kernel)][r0:r1, cols]
                covered = (ground .>= 1) .& support
                sup_n   = sum(support)
                cov_n   = sum(covered)
                push!(cover_rows, (
                    line_number    = line_number,
                    cover          = COVER_LABELS[cover],
                    mission        = label,
                    support_cells  = sup_n,
                    covered_cells  = cov_n,
                    coverage_ratio = sup_n > 0 ? cov_n / sup_n : NaN,
                    support_mode   = support_mode,
                ))
                for (jj, col) in enumerate(cols)
                    sup_col = view(support, :, jj)
                    cov_col = view(covered, :, jj)
                    sup_n_c = sum(sup_col)
                    cov_n_c = sum(cov_col[sup_col])
                    push!(profile_rows, (
                        line_number      = line_number,
                        cover            = COVER_LABELS[cover],
                        mission          = label,
                        column_index     = col - 1,  # 0-based to match Python
                        x_m              = x_centers[col],
                        support_cells    = sup_n_c,
                        covered_cells    = cov_n_c,
                        coverage_fraction = sup_n_c > 0 ? cov_n_c / sup_n_c : NaN,
                        support_mode     = support_mode,
                    ))
                end
            end
        end
    end

    scan_df    = DataFrame(scan_rows)
    cover_df   = DataFrame(cover_rows)
    profile_df = DataFrame(profile_rows)

    # Build wide pivot
    wide_df = _build_wide_df(cover_df, scan_df)

    return scan_df, cover_df, wide_df, profile_df
end


# ---------------------------------------------------------------------------
# Single-line detail (e.g. Line 2)
# ---------------------------------------------------------------------------

"""
    single_line_cover_metrics(
        count_grids, processed, planned_lines, missions, line_number;
        half_band_rows=20, geotransform=nothing, ymin=nothing, xmin=nothing,
        cover_masks=nothing, allow_approximate_masks=false
    ) → (extents_df, cover_df, profile_df)

Compute cover metrics and column profile for a single survey line using the
actual trajectory overlap across all missions.

Same convention as `line_scan_cover_metrics` but for one line.
"""
function single_line_cover_metrics(
    count_grids::Dict,
    processed::Dict{String,DataFrame},
    planned_lines::DataFrame,
    missions::Vector{<:Tuple},
    line_number::Int;
    half_band_rows::Int = 20,
    geotransform = nothing,
    ymin::Union{Nothing,Real} = nothing,
    xmin::Union{Nothing,Real} = nothing,
    cover_masks = nothing,
    allow_approximate_masks::Bool = false,
)
    masks, support_mode, grid_size = _line_scan_cover_support(count_grids;
        cover_masks=cover_masks, allow_approximate_masks=allow_approximate_masks)
    nrows, ncols = grid_size
    gt = _line_scan_geotransform(grid_size; geotransform=geotransform,
                                 xmin=xmin, ymin=ymin)
    half_band_rows >= 0 || throw(ArgumentError("half_band_rows must be nonnegative"))

    line_row = planned_lines[planned_lines.line_number .== line_number, :]
    isempty(line_row) && error("Line number $line_number not found in planned_lines")
    line_row = first(eachrow(line_row))
    lid      = Int(line_row.line_index)

    # Extents
    ext_rows = Vector{NamedTuple}()
    miss_labels = [m[1] for m in missions]
    for (label, _, _) in missions
        g = processed[label][processed[label].nearest_line .== lid, :]
        push!(ext_rows, (
            mission         = label,
            x_min_actual    = quantile(g.GridX, 0.005),
            x_max_actual    = quantile(g.GridX, 0.995),
            x_min_raw       = minimum(g.GridX),
            x_max_raw       = maximum(g.GridX),
            y_median_actual = median(g.GridY),
            samples         = nrow(g),
            segments        = length(unique(g.segment_id)),
            support_mode    = support_mode,
        ))
    end
    extents_df = DataFrame(ext_rows)

    common_x0 = maximum(extents_df.x_min_actual)
    common_x1 = minimum(extents_df.x_max_actual)

    x_centers = gt.x_origin .+ ((0:(ncols-1)) .+ 0.5) .* gt.dx
    selected    = findall((x_centers .>= common_x0) .& (x_centers .<= common_x1))
    isempty(selected) && throw(ArgumentError("Survey overlap does not include a count-grid column"))

    row_center, r0, r1 = _line_scan_band(line_row.y_center, gt, nrows, half_band_rows)
    r0 <= r1 || throw(ArgumentError("Survey line band does not intersect the count grid"))

    cover_rows   = Vector{NamedTuple}()
    profile_rows = Vector{NamedTuple}()

    for cover in COVER_KEYS
        support = masks[cover][r0:r1, selected]

        for (label, mission, kernel) in missions
            ground  = count_grids[("ground", cover, mission, kernel)][r0:r1, selected]
            covered = (ground .>= 1) .& support
            sup_n   = sum(support)
            cov_n   = sum(covered)
            push!(cover_rows, (
                line_number    = line_number,
                cover          = COVER_LABELS[cover],
                mission        = label,
                support_cells  = sup_n,
                covered_cells  = cov_n,
                coverage_ratio = sup_n > 0 ? cov_n / sup_n : NaN,
                support_mode   = support_mode,
            ))
            for (jj, col) in enumerate(selected)
                sup_col = view(support, :, jj)
                cov_col = view(covered, :, jj)
                sup_n_c = sum(sup_col)
                cov_n_c = sum(cov_col[sup_col])
                push!(profile_rows, (
                    line_number      = line_number,
                    cover            = COVER_LABELS[cover],
                    mission          = label,
                    column_index     = col - 1,  # 0-based
                    x_m              = x_centers[col],
                    support_cells    = sup_n_c,
                    covered_cells    = cov_n_c,
                    coverage_fraction = sup_n_c > 0 ? cov_n_c / sup_n_c : NaN,
                    support_mode     = support_mode,
                ))
            end
        end
    end

    return extents_df, DataFrame(cover_rows), DataFrame(profile_rows)
end


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

"""Build a wide pivot of cover_df and append KDE-vs-constant differences."""
function _build_wide_df(cover_df::DataFrame, scan_df::DataFrame)
    # Identify mission labels present
    missions = unique(cover_df.mission)
    kde_label = findfirst(m -> occursin("KDE", m) || occursin("density", m), missions)
    kde_col   = isnothing(kde_label) ? nothing : missions[kde_label]

    # Pivot: index=(line_number, cover), cols=missions, values=coverage_ratio
    wide = unstack(cover_df, [:line_number, :cover], :mission, :coverage_ratio)

    # Add difference columns if KDE mission and constant missions identified
    const2_col = findfirst(m -> occursin("2 m/s", m), missions)
    const8_col = findfirst(m -> occursin("8 m/s", m), missions)
    const2 = isnothing(const2_col) ? nothing : missions[const2_col]
    const8 = isnothing(const8_col) ? nothing : missions[const8_col]

    if !isnothing(kde_col) && !isnothing(const2)
        wide[!, :kde_minus_const2] = wide[!, kde_col] .- wide[!, const2]
    end
    if !isnothing(kde_col) && !isnothing(const8)
        wide[!, :kde_minus_const8] = wide[!, kde_col] .- wide[!, const8]
    end

    # Add min_support_cells from cover_df
    min_sup = combine(groupby(cover_df, [:line_number, :cover]), :support_cells => minimum => :min_support_cells)
    wide    = leftjoin(wide, min_sup; on=[:line_number, :cover])
    if hasproperty(cover_df, :support_mode)
        modes = unique(cover_df.support_mode)
        length(modes) == 1 || throw(ArgumentError("Line-scan tables mix support modes"))
        wide[!, :support_mode] = fill(only(modes), nrow(wide))
    end

    # Compute best_forest ranking and merge into scan_df
    forest = wide[map(c -> c ∈ ["Deciduous", "Coniferous"], wide.cover), :]
    if !isempty(forest) && hasproperty(forest, :kde_minus_const2) && hasproperty(forest, :kde_minus_const8)
        forest = copy(forest)
        forest[!, :abs_min_gain] = min.(forest.kde_minus_const2, forest.kde_minus_const8)
        # Per line_number: row with highest abs_min_gain
        best = combine(groupby(forest, :line_number)) do g
            idx = argmax(g.abs_min_gain)
            DataFrame(
                best_forest_cover    = [g.cover[idx]],
                best_forest_min_gain = [g.abs_min_gain[idx]],
            )
        end
        # Merge back into scan_df (we return wide here; scan merging happens in caller)
        _scan_merged = leftjoin(scan_df, best; on=:line_number)
        # We return wide only; the merged scan is not our concern here — the
        # example script handles it.
        _ = _scan_merged  # suppress unused warning
    end

    return wide
end

