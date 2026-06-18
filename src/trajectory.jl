"""
    trajectory.jl — Survey-leg segmentation and tracking metrics

Ports the logic from:
  - flightline_comparison_work/trajectory_segmentation.py
  - flightline_comparison_work/plot_tracking_metrics_refined.py

Public naming: "KDE-guided" everywhere (legacy "density-aware" appears only in
filename/key parsing).

## API overview

    derive_planned_lines(waypoint_csv)  → DataFrame
    ground_speed(VEast, VNorth)         → Vector{Float64}
    assign_to_planned_lines(y, line_centers) → (nearest_line, cross_track)
    contiguous_segments(mask, time, line)    → Vector{Int}
    clean_survey_segments(df, mission, planned_lines) → (clean_df, full_df)
    mission_speed_bounds(mission)       → (lo, hi)

Tracking-metrics summary builders (operate on the cleaned DataFrames):

    tracking_summary(processed)                         → DataFrame
    tracking_line_summary(processed)                    → DataFrame
    tracking_segment_summary(processed)                 → DataFrame
"""

# ---------------------------------------------------------------------------
# Planned-line derivation from waypoint CSV
# ---------------------------------------------------------------------------

"""
    derive_planned_lines(waypoint_csv) → DataFrame

Read a waypoint CSV with columns `GridX`, `GridY` and infer horizontal
survey-leg centers. Waypoints are grouped by rounded `GridY`; groups with
≥ 10 waypoints and an x-span ≥ 250 m are treated as survey legs.

Returns a `DataFrame` with columns:
  `line_index`, `line_number`, `y_center`, `x_min`, `x_max`, `n`
sorted by ascending `y_center`.
"""
function derive_planned_lines(waypoint_csv::AbstractString)
    wp = CSV.read(waypoint_csv, DataFrame)
    return _derive_planned_lines_df(wp)
end

function _derive_planned_lines_df(wp::DataFrame)
    wp_copy = copy(wp)
    wp_copy[!, :y_round] = round.(wp_copy[!, :GridY])

    # Group by rounded y, compute aggregates
    result_rows = DataFrame(
        y_round = Float64[],
        n       = Int[],
        y_center = Float64[],
        x_min    = Float64[],
        x_max    = Float64[],
    )
    for (y_r, g) in pairs(groupby(wp_copy, :y_round))
        push!(result_rows, (
            y_round  = y_r[:y_round],
            n        = nrow(g),
            y_center = median(g.GridY),
            x_min    = minimum(g.GridX),
            x_max    = maximum(g.GridX),
        ))
    end

    # Filter: ≥ 10 waypoints, x-span ≥ 250 m
    mask = (result_rows.n .>= 10) .& ((result_rows.x_max .- result_rows.x_min) .>= 250.0)
    lines = sort(result_rows[mask, :], :y_center)
    lines[!, :line_index]  = 0:(nrow(lines)-1)
    lines[!, :line_number] = 1:nrow(lines)

    return select(lines, :line_index, :line_number, :y_center, :x_min, :x_max, :n)
end


# ---------------------------------------------------------------------------
# Speed utilities
# ---------------------------------------------------------------------------

"""
    ground_speed(veast, vnorth) → Vector{Float64}

Compute ground speed from east and north velocity components.
"""
function ground_speed(veast::AbstractVector, vnorth::AbstractVector)
    return sqrt.(veast .^ 2 .+ vnorth .^ 2)
end

"""
    mission_speed_bounds(mission) → (lo::Float64, hi::Float64)

Return the mission-specific speed filter bounds used in survey-leg cleaning.

| Mission pattern  | lo   | hi   |
|------------------|------|------|
| contains "2 m/s" | 1.0  | 3.2  |
| contains "8 m/s" | 6.0  | 10.5 |
| otherwise (KDE)  | 1.0  | 9.5  |
"""
function mission_speed_bounds(mission::AbstractString)
    if occursin("2 m/s", mission)
        return 1.0, 3.2
    elseif occursin("8 m/s", mission)
        return 6.0, 10.5
    else
        return 1.0, 9.5  # KDE-guided
    end
end


# ---------------------------------------------------------------------------
# Line assignment
# ---------------------------------------------------------------------------

"""
    assign_to_planned_lines(y, line_centers) → (nearest_line, cross_track)

For each sample with northing coordinate `y`, find the nearest planned-line
center (by |Δy|) and return:
- `nearest_line`: 0-based line index (Int vector matching `line_centers` order)
- `cross_track`:  absolute cross-track distance in metres
"""
function assign_to_planned_lines(y::AbstractVector{<:Real}, line_centers::AbstractVector{<:Real})
    n = length(y)
    nearest_line = Vector{Int}(undef, n)
    cross_track  = Vector{Float64}(undef, n)
    for i in eachindex(y)
        best_idx = argmin(abs.(y[i] .- line_centers))
        nearest_line[i] = best_idx - 1  # 0-based to match Python convention
        cross_track[i]  = abs(y[i] - line_centers[best_idx])
    end
    return nearest_line, cross_track
end


# ---------------------------------------------------------------------------
# Contiguous segment labelling
# ---------------------------------------------------------------------------

"""
    contiguous_segments(mask, time, line) → Vector{Int}

Label contiguous runs of selected samples on the same planned line with a
monotonically increasing segment id (0-based). Unselected samples receive -1.

A new segment starts when:
- the previous sample was not selected, OR
- the line index changes, OR
- the time gap exceeds 0.25 s.
"""
function contiguous_segments(
    mask::AbstractVector{Bool},
    time::AbstractVector{<:Real},
    line::AbstractVector{<:Integer},
)
    n   = length(mask)
    seg = fill(-1, n)
    current       = -1
    prev_selected = false
    prev_line     = -1
    prev_time     = -Inf

    for i in 1:n
        if !mask[i]
            prev_selected = false
            prev_line     = -1
            prev_time     = -Inf
            continue
        end
        new_seg = (
            !prev_selected ||
            prev_line < 0 ||
            line[i] != prev_line ||
            (time[i] - prev_time) > 0.25
        )
        if new_seg
            current += 1
        end
        seg[i]        = current
        prev_selected = true
        prev_line     = line[i]
        prev_time     = time[i]
    end
    return seg
end


# ---------------------------------------------------------------------------
# Fitted-center computation
# ---------------------------------------------------------------------------

"""
    fitted_line_centers(df_all, base_mask, line_centers, line_indices) → Dict{Int,Float64}

Compute the median GridY of all candidate-survey samples for each planned line.
If fewer than 100 samples are available for a line, fall back to the planned
`line_center`.
"""
function fitted_line_centers(
    df_all::DataFrame,
    base_mask::AbstractVector{Bool},
    line_centers::AbstractVector{<:Real},
    line_indices::AbstractVector{<:Integer},
)
    fitted = Dict{Int,Float64}()
    for lid in line_indices
        vals = df_all[base_mask .& (df_all.nearest_line .== lid), :GridY]
        if length(vals) >= 100
            fitted[lid] = float(median(vals))
        else
            fitted[lid] = float(line_centers[lid + 1])  # 1-based indexing
        end
    end
    return fitted
end


# ---------------------------------------------------------------------------
# Survey-leg cleaning
# ---------------------------------------------------------------------------

"""
    clean_survey_segments(df, mission, planned_lines) → (clean_df, full_df)

Remove calibration, turning, and transition samples using planned waypoint
geometry and motion filters. The returned `clean_df` contains only sustained
horizontal survey legs; `full_df` is the full trajectory with diagnostic
columns added.

# Filters applied (all must pass):
- `cross_track_error_m ≤ 12` m
- within planned x-range ± 12 m
- `|VEast| / speed ≥ 0.72` (predominantly east-west motion)
- mission-specific speed bounds (see `mission_speed_bounds`)

# Segment retention (all must pass):
- segment duration ≥ 2.5 s
- x-span ≥ 80 m
- sample count ≥ 150

# Columns added to `full_df`:
- `ground_speed_mps`, `nearest_line`, `cross_track_error_m`
- `fitted_line_y`, `tracking_error_m`, `planned_offset_m`
- `candidate_survey` (Bool), `segment_id` (Int, -1 = not selected)
- `survey_leg_clean` (Bool)

# Arguments
- `df`: raw trajectory DataFrame with columns `GridX`, `GridY`, `VEast`,
  `VNorth`, `Time`
- `mission`: mission label string used to select speed bounds
- `planned_lines`: DataFrame from `derive_planned_lines`
"""
function clean_survey_segments(df::DataFrame, mission::AbstractString, planned_lines::DataFrame)
    df_work = copy(df)
    line_centers = Float64.(planned_lines.y_center)
    line_indices = Int.(planned_lines.line_index)
    line_xmin    = Dict(planned_lines.line_index .=> planned_lines.x_min)
    line_xmax    = Dict(planned_lines.line_index .=> planned_lines.x_max)

    # --- Ground speed ---
    df_work[!, :ground_speed_mps] = ground_speed(df_work.VEast, df_work.VNorth)

    # --- Line assignment ---
    nl, cte = assign_to_planned_lines(df_work.GridY, line_centers)
    df_work[!, :nearest_line]       = nl
    df_work[!, :cross_track_error_m] = cte

    speed      = df_work.ground_speed_mps
    along_frac = abs.(df_work.VEast) ./ max.(speed, 1e-6)
    x          = df_work.GridX
    line       = df_work.nearest_line

    # Within-x filter
    within_x = [
        line_xmin[line[i]] - 12.0 <= x[i] <= line_xmax[line[i]] + 12.0
        for i in 1:nrow(df_work)
    ]

    # Speed filter
    lo, hi   = mission_speed_bounds(mission)
    speed_ok = (speed .>= lo) .& (speed .<= hi)

    base_mask = (
        (df_work.cross_track_error_m .<= 12.0) .&
        within_x .&
        (along_frac .>= 0.72) .&
        speed_ok
    )

    # --- Fitted line centers ---
    fcenters = fitted_line_centers(df_work, base_mask, line_centers, line_indices)

    df_work[!, :fitted_line_y] = [fcenters[line[i]] for i in 1:nrow(df_work)]
    df_work[!, :tracking_error_m] = abs.(df_work.GridY .- df_work.fitted_line_y)
    df_work[!, :planned_offset_m] = df_work.fitted_line_y .- [line_centers[line[i]+1] for i in 1:nrow(df_work)]
    df_work[!, :candidate_survey] = base_mask

    # --- Segment labelling ---
    seg_ids = contiguous_segments(base_mask, Float64.(df_work.Time), Int.(df_work.nearest_line))
    df_work[!, :segment_id] = seg_ids

    # --- Filter: keep only sustained segments ---
    keep_segments = Set{Int}()
    for sid in unique(seg_ids[seg_ids .>= 0])
        g = df_work[df_work.segment_id .== sid, :]
        dur    = maximum(g.Time) - minimum(g.Time)
        x_span = maximum(g.GridX) - minimum(g.GridX)
        if dur >= 2.5 && x_span >= 80.0 && nrow(g) >= 150
            push!(keep_segments, sid)
        end
    end
    df_work[!, :survey_leg_clean] = [sid in keep_segments for sid in df_work.segment_id]

    clean = df_work[df_work.survey_leg_clean, :]
    return clean, df_work
end


# ---------------------------------------------------------------------------
# Tracking metrics
# ---------------------------------------------------------------------------

"""
    tracking_summary(processed) → DataFrame

Compute mission-level tracking summary metrics from a Dict of cleaned
trajectory DataFrames (`mission_label => DataFrame`).

Columns: `mission`, `survey_samples`, `clean_segments`, `survey_duration_s`,
`speed_mean_mps`, `speed_median_mps`, `speed_p10_mps`, `speed_p90_mps`,
`planned_offset_abs_median`, `planned_offset_abs_p95`,
`planned_cross_track_p95_m`, `tracking_error_mean_m`,
`tracking_error_median_m`, `tracking_error_p95_m`, `tracking_error_rms_m`.
"""
function tracking_summary(processed::Dict{String,DataFrame})
    rows = Vector{NamedTuple}()
    for (mission, survey) in processed
        spd = survey.ground_speed_mps
        cte = survey.cross_track_error_m
        trk = survey.tracking_error_m
        off = survey.planned_offset_m

        push!(rows, (
            mission                   = mission,
            survey_samples            = nrow(survey),
            clean_segments            = length(unique(survey.segment_id)),
            survey_duration_s         = isempty(survey.Time) ? NaN :
                                        maximum(survey.Time) - minimum(survey.Time),
            speed_mean_mps            = mean(spd),
            speed_median_mps          = median(spd),
            speed_p10_mps             = quantile(spd, 0.10),
            speed_p90_mps             = quantile(spd, 0.90),
            planned_offset_abs_median = median(abs.(off)),
            planned_offset_abs_p95    = quantile(abs.(off), 0.95),
            planned_cross_track_p95_m = quantile(cte, 0.95),
            tracking_error_mean_m     = mean(trk),
            tracking_error_median_m   = median(trk),
            tracking_error_p95_m      = quantile(trk, 0.95),
            tracking_error_rms_m      = sqrt(mean(trk .^ 2)),
        ))
    end
    return DataFrame(rows)
end


"""
    tracking_line_summary(processed) → DataFrame

Compute per-line tracking metrics from cleaned trajectory DataFrames.

Columns: `mission`, `line_index`, `samples`, `n_segments`,
`speed_median_mps`, `speed_iqr_mps`, `planned_offset_m`,
`tracking_error_p95_m`, `tracking_error_rms_m`.
"""
function tracking_line_summary(processed::Dict{String,DataFrame})
    rows = Vector{NamedTuple}()
    for (mission, survey) in processed
        for lid in sort(unique(survey.nearest_line))
            g = survey[survey.nearest_line .== lid, :]
            spd = g.ground_speed_mps
            push!(rows, (
                mission              = mission,
                line_index           = lid,
                samples              = nrow(g),
                n_segments           = length(unique(g.segment_id)),
                speed_median_mps     = median(spd),
                speed_iqr_mps        = quantile(spd, 0.75) - quantile(spd, 0.25),
                planned_offset_m     = median(g.planned_offset_m),
                tracking_error_p95_m = quantile(g.tracking_error_m, 0.95),
                tracking_error_rms_m = sqrt(mean(g.tracking_error_m .^ 2)),
            ))
        end
    end
    return DataFrame(rows)
end


"""
    tracking_segment_summary(processed) → DataFrame

Compute per-segment tracking metrics from cleaned trajectory DataFrames.

Columns: `mission`, `segment_id`, `line_index`, `samples`, `duration_s`,
`x_span_m`, `speed_median_mps`, `planned_offset_m`, `tracking_error_rms_m`.
"""
function tracking_segment_summary(processed::Dict{String,DataFrame})
    rows = Vector{NamedTuple}()
    for (mission, survey) in processed
        for sid in sort(unique(survey.segment_id))
            g = survey[survey.segment_id .== sid, :]
            # mode of nearest_line
            line_counts = Dict{Int,Int}()
            for l in g.nearest_line
                line_counts[l] = get(line_counts, l, 0) + 1
            end
            dominant_line = argmax(line_counts)
            push!(rows, (
                mission              = mission,
                segment_id           = sid,
                line_index           = dominant_line,
                samples              = nrow(g),
                duration_s           = maximum(g.Time) - minimum(g.Time),
                x_span_m             = maximum(g.GridX) - minimum(g.GridX),
                speed_median_mps     = median(g.ground_speed_mps),
                planned_offset_m     = median(g.planned_offset_m),
                tracking_error_rms_m = sqrt(mean(g.tracking_error_m .^ 2)),
            ))
        end
    end
    return DataFrame(rows)
end


# ---------------------------------------------------------------------------
# Mission time metrics
# ---------------------------------------------------------------------------

"""
    mission_time_metrics(raw_paths, processed) -> DataFrame

Compute mission duration metrics from raw trajectory files and cleaned survey legs.

Arguments
---------
- `raw_paths`  — Dict{String, String} mapping mission label to raw trajectory CSV path
- `processed`  — Dict{String, DataFrame} of cleaned survey-leg DataFrames
                 (must have columns `Time` and `segment_id`)

Returns a DataFrame with one row per mission and the following columns:

**Raw logged mission duration**
- `raw_duration_s` / `raw_duration_min` — elapsed time from first to last logged
  timestamp in the raw trajectory CSV.  Includes calibration runs, inter-leg turns,
  and all other logged motion.

**Cleaned-segment duration** (recommended metric for manuscript reporting)
- `cleaned_segment_duration_s` / `cleaned_segment_duration_min` — sum of
  `(max(Time) − min(Time))` over every retained sustained survey-leg segment.
  Excludes inter-leg transit gaps, turns, and calibration portions.
  This is the total time the UAV actually spent flying on retained survey legs.
- `cleaned_segment_fraction_raw` — `cleaned_segment_duration_s / raw_duration_s`.

**Survey wall-clock span** (diagnostic only; includes inter-leg gaps)
- `survey_wall_clock_span_s` / `survey_wall_clock_span_min` — time from first
  cleaned sample to last cleaned sample (a single continuous span that includes
  inter-leg transit gaps, so it *overstates* active survey time).
- `survey_wall_clock_fraction` — `survey_wall_clock_span_s / raw_duration_s`.

**Auxiliary**
- `raw_start_time_s`, `raw_end_time_s` — absolute timestamps of first/last raw sample.
- `survey_start_time_s`, `survey_end_time_s` — absolute timestamps of first/last
  cleaned survey sample.
- `n_raw_samples`, `n_survey_samples` — sample counts.

Note: only raw trajectory CSVs and the `processed` Dict (output of
`clean_survey_segments`) are used.  No `*_with_line_id.csv` files are read.
"""
function mission_time_metrics(
    raw_paths ::Dict{String,String},
    processed ::Dict{String,DataFrame},
)::DataFrame
    rows = NamedTuple[]
    for mission in sort(collect(keys(raw_paths)))
        raw_path = raw_paths[mission]
        raw      = CSV.read(raw_path, DataFrame; select=[:Time])
        t_raw    = Float64.(raw.Time)

        raw_t0  = minimum(t_raw)
        raw_t1  = maximum(t_raw)
        raw_dur = raw_t1 - raw_t0

        survey = get(processed, mission, DataFrame())
        if isempty(survey) || !hasproperty(survey, :Time) || !hasproperty(survey, :segment_id)
            wall_span = NaN
            seg_dur   = NaN
            surv_t0   = NaN
            surv_t1   = NaN
            n_survey  = 0
        else
            # Wall-clock span: first-to-last cleaned timestamp (includes inter-leg gaps)
            surv_t0   = minimum(survey.Time)
            surv_t1   = maximum(survey.Time)
            wall_span = surv_t1 - surv_t0
            n_survey  = nrow(survey)

            # Cleaned-segment duration: sum of (max-min Time) per retained segment.
            # Uses only survey.segment_id and survey.Time — no line_id files.
            seg_dur = 0.0
            for sid in unique(survey.segment_id)
                g = survey[survey.segment_id .== sid, :]
                seg_dur += maximum(g.Time) - minimum(g.Time)
            end
        end

        push!(rows, (
            mission                      = mission,
            raw_duration_s               = raw_dur,
            raw_duration_min             = raw_dur / 60.0,
            cleaned_segment_duration_s   = seg_dur,
            cleaned_segment_duration_min = seg_dur / 60.0,
            cleaned_segment_fraction_raw = isnan(seg_dur) ? NaN : seg_dur / raw_dur,
            survey_wall_clock_span_s     = wall_span,
            survey_wall_clock_span_min   = wall_span / 60.0,
            survey_wall_clock_fraction   = isnan(wall_span) ? NaN : wall_span / raw_dur,
            raw_start_time_s             = raw_t0,
            raw_end_time_s               = raw_t1,
            survey_start_time_s          = isnan(wall_span) ? NaN : surv_t0,
            survey_end_time_s            = isnan(wall_span) ? NaN : surv_t1,
            n_raw_samples                = length(t_raw),
            n_survey_samples             = n_survey,
        ))
    end
    return DataFrame(rows)
end
