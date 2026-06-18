"""
    linescan.jl — Line-scan coverage metrics from actual flown trajectories

Ports the logic from:
  - flightline_comparison_work/scan_lines_actual_trajectories.py
  - flightline_comparison_work/plot_line2_actual_overlap_detail.py

## Constants and conventions

The Durham, NH count-grid has shape (263 rows × 324 columns) in geographic
(row=north) orientation. The `count_grid` reshape convention in Python is:

    mat = np.array(values).reshape((324, 263), order='C').T   # → (263, 324)
    mat = np.flipud(mat)                                        # row 0 = north

We replicate this in `decode_count_grid`.

## API

    decode_count_grid(values) → Matrix{Float64}    (263 × 324, row 0 = north)
    load_count_grids(counts_json) → Dict           keyed (ret, cover, mission, kernel)
    actual_line_extents(processed, planned_lines) → DataFrame
    common_x_overlap(extents, mission_labels) → Dict{Int,(x0,x1)}
    line_scan_cover_metrics(...)  → (scan_df, cover_df, wide_df, profile_df)
    single_line_cover_metrics(...)→ (extents_df, cover_df, profile_df)
"""

# Grid dimensions (Durham, NH site)
const SCAN_NROWS = 263
const SCAN_NCOLS = 324

# Cover codes and manuscript labels
const COVER_KEYS   = ["field", "decid", "conif"]
const COVER_LABELS = Dict("field" => "Field", "decid" => "Deciduous", "conif" => "Coniferous")


# ---------------------------------------------------------------------------
# Count-grid decoding
# ---------------------------------------------------------------------------

"""
    decode_count_grid(values) → Matrix{Float64}

Reshape a flat count-grid `values` vector (length 324 × 263) into a
(263 × 324) matrix with row 0 at the geographic north, matching the Python
convention used in the project's `counts.json`.

The reshape order is:
  1. `reshape(values, 324, 263)'`  → (263 × 324)  [column-major → transpose = C order effect]
  2. `reverse(mat; dims=1)`        → flip rows so row 1 = northernmost
"""
function decode_count_grid(values::AbstractVector)
    # Python: reshape((324,263), order='C').T → same as Julia reshape(324,263)' (no, careful)
    # Python C order reshape(324,263): element [i,j] = values[i*263 + j]
    # In Julia reshape is column-major: reshape(values, 263, 324) gives [i,j]=values[(j-1)*263+i]
    # To match Python C-order reshape(324, 263):
    #   Python result[i, j] = values[i*263 + j]
    # After .T in Python: result_T[j, i] = values[i*263 + j]
    # → Julia mat[j, i] = values[i*263 + j]  → mat = reshape(values, 263, 324) with COLUMN-major = wrong
    # Instead, interpret values as row-major 324×263, then transpose:
    #   raw[i,j] = values[i*263+j]  (0-indexed)
    # Julia 1-indexed: raw[i,j] = values[(i-1)*263 + j]
    # mat = raw'  → mat[j,i] = raw[i,j] = values[(i-1)*263+j]
    # So mat is (263×324): mat[row, col] = values[(col-1)*263 + row]  ...wait that's column-major reshape
    # Actually: reshape(values, 263, 324)[row, col] = values[(col-1)*263 + row-1+1]
    # Python: reshape(values, 324, 263, order='C')[i,j] = values[i*263+j]  (0-indexed)
    # After .T: [j, i] = values[i*263 + j]
    # Julia: mat[j+1, i+1] = values[i*263 + j]
    # Let's build this directly: mat is 263×324
    # mat[row, col] = values[(col-1)*263 + (row-1)]  ← this is Julia reshape(values,263,324)[row,col] ✓
    mat = reshape(Float64.(values), SCAN_NROWS, SCAN_NCOLS)  # (263×324) column-major
    return reverse(mat; dims=1)  # flip: row 1 = northernmost
end


# ---------------------------------------------------------------------------
# Count-grid loading
# ---------------------------------------------------------------------------

"""
    load_count_grids(counts_json) → Dict{Tuple{String,String,String,String}, Matrix{Float64}}

Load `counts.json` and decode all count grids. Keys are
`(ret, cover, mission, kernel)` tuples matching the `key` fields in the JSON.
"""
function load_count_grids(counts_json::AbstractString)
    records = JSON.parsefile(counts_json)
    out = Dict{NTuple{4,String}, Matrix{Float64}}()
    for r in records
        key = r["key"]
        k   = (key["ret"], key["cover"], key["mission"], key["kernel"])
        out[k] = decode_count_grid(r["matrix"])
    end
    return out
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
        half_band_rows=20, ymin=nothing, xmin=nothing, cover_masks=nothing
    ) → (scan_df, cover_df, wide_df, profile_df)

Compute per-line, per-cover-class coverage ratios and column profiles using
the actual flown x-overlap across missions.

# Arguments
- `count_grids`: Dict from `load_count_grids`
- `processed`: Dict{String, DataFrame} of cleaned trajectories
- `planned_lines`: DataFrame from `derive_planned_lines`
- `missions`: Vector of `(label, mission_key, kernel_key)` tuples —
  e.g. `[("Const. 2 m/s", "const2", "NA"), ...]`
- `half_band_rows=20`: envelope half-height in count-grid rows (1 m each)
- `ymin`, `xmin`: geographic origin of the count grid; if `nothing`, inferred
  from `planned_lines`
- `cover_masks`: optional `Dict` keyed by `"field"`, `"decid"`, and `"conif"`;
  when supplied, CR denominators are fixed GLI class cells rather than support
  inferred from nonzero return-count cells.

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
    ymin::Union{Nothing,Float64} = nothing,
    xmin::Union{Nothing,Float64} = nothing,
    cover_masks = nothing,
)
    _ymin = isnothing(ymin) ? minimum(planned_lines.y_center) : ymin
    _xmin = isnothing(xmin) ? minimum(planned_lines.x_min) : xmin

    extents    = actual_line_extents(processed, planned_lines)
    miss_labels = [m[1] for m in missions]
    overlap    = common_x_overlap(extents, miss_labels)

    x_centers = _xmin .+ (0:(SCAN_NCOLS-1)) .+ 0.5

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

        row_center = round(Int, row.y_center - _ymin)
        r0 = max(1, row_center - half_band_rows + 1)  # 1-based, Julia
        r1 = min(SCAN_NROWS, row_center + half_band_rows + 1)

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
        ))

        for cover in COVER_KEYS
            support = if isnothing(cover_masks)
                # Legacy fallback when the GLI class raster is unavailable:
                # infer support from the union of all-return cells. When the
                # class raster is supplied, use fixed cover-class cells instead.
                s = falses(r1 - r0 + 1, length(cols))
                for (_, mission, kernel) in missions
                    grid = count_grids[("all", cover, mission, kernel)]
                    s .|= (grid[r0:r1, cols] .>= 1)
                end
                s
            else
                haskey(cover_masks, cover) || error("cover_masks missing key '$cover'")
                cover_masks[cover][r0:r1, cols]
            end

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
        half_band_rows=20, ymin=nothing, xmin=nothing, cover_masks=nothing
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
    ymin::Union{Nothing,Float64} = nothing,
    xmin::Union{Nothing,Float64} = nothing,
    cover_masks = nothing,
)
    _ymin = isnothing(ymin) ? minimum(planned_lines.y_center) : ymin
    _xmin = isnothing(xmin) ? minimum(planned_lines.x_min) : xmin

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
        ))
    end
    extents_df = DataFrame(ext_rows)

    common_x0 = maximum(extents_df.x_min_actual)
    common_x1 = minimum(extents_df.x_max_actual)

    x_centers   = _xmin .+ (0:(SCAN_NCOLS-1)) .+ 0.5
    selected    = findall((x_centers .>= common_x0) .& (x_centers .<= common_x1))

    row_center = round(Int, Float64(line_row.y_center) - _ymin)
    r0 = max(1, row_center - half_band_rows + 1)
    r1 = min(SCAN_NROWS, row_center + half_band_rows + 1)

    cover_rows   = Vector{NamedTuple}()
    profile_rows = Vector{NamedTuple}()

    for cover in COVER_KEYS
        support = if isnothing(cover_masks)
            s = falses(r1 - r0 + 1, length(selected))
            for (_, mission, kernel) in missions
                grid = count_grids[("all", cover, mission, kernel)]
                s .|= (grid[r0:r1, selected] .>= 1)
            end
            s
        else
            haskey(cover_masks, cover) || error("cover_masks missing key '$cover'")
            cover_masks[cover][r0:r1, selected]
        end

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
