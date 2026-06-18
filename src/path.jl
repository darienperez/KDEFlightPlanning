"""
    path.jl — Lawnmower (boustrophedon) path generation

Public API
----------
- `lawnmower_from_extents(spec::LawnmowerSpec) -> Vector{NTuple{2,Float64}}`
  Generate the ordered list of path vertices for a lawnmower survey.

The returned vector includes the turnaround corners so that every line and
every inter-line transit is represented. Flight-line assignment can be
recovered from the vertex index (see `line_id_for_vertex`).

Design notes
------------
The path is defined entirely in the grid's local coordinate system (Easting,
Northing in metres). No CRS transformation is applied here; that is handled
in `io.jl` if needed.

The `yaw_deg` field in `LawnmowerSpec` rotates the sweep direction. When
`yaw_deg == 0` and `primary == :x`, the UAV sweeps east–west and steps north
between lines (standard cross-track survey).
"""

# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------

"""
    _rotation_matrix(θ_deg) -> (cos_θ, sin_θ)

Return `(cos θ, sin θ)` for angle `θ` in degrees (CCW from East).
"""
function _rotation_matrix(θ_deg::Real)
    θ = deg2rad(Float64(θ_deg))
    return cos(θ), sin(θ)
end

"""
    _sweep_range(lo, hi, spacing) -> Vector{Float64}

Generate flight-line positions from `lo` to `hi` with step `spacing`.
Always includes the first position and at least one line.
"""
function _sweep_range(lo::Real, hi::Real, spacing::Real)
    lo, hi = Float64(lo), Float64(hi)
    spacing = Float64(spacing)
    spacing > 0 || throw(ArgumentError("spacing must be positive"))
    n = max(1, floor(Int, (hi - lo) / spacing) + 1)
    return [lo + (i-1)*spacing for i in 1:n]
end

# ---------------------------------------------------------------------------
# lawnmower_from_extents
# ---------------------------------------------------------------------------

"""
    lawnmower_from_extents(spec::LawnmowerSpec) -> Vector{NTuple{2,Float64}}

Generate the ordered sequence of 2-D path vertices for a lawnmower survey
defined by `spec`.

The path visits one flight line at a time, alternating sweep direction
(boustrophedon). Turnaround corners are included so that the path is
continuous. The result can be passed directly to `generate_waypoints`.

Arguments
---------
- `spec`: A `LawnmowerSpec` with fields:
  - `xmin`, `xmax`, `ymin`, `ymax`: bounding box
  - `spacing`: cross-track line separation (m)
  - `yaw_deg`: rotation of the sweep axis (degrees CCW from east)
  - `primary`: `:x` → sweep along x, step along y; `:y` → sweep along y
  - `start`:   `:low` → start at the lower bound; `:high` → upper bound

Returns
-------
`Vector{NTuple{2,Float64}}` — path vertices in order.

Example
-------
```julia
spec = LawnmowerSpec(xmin=0.0, xmax=200.0, ymin=0.0, ymax=200.0,
                     spacing=20.0, yaw_deg=0.0, primary=:x, start=:low)
path = lawnmower_from_extents(spec)
# path[1] = (0.0, 0.0), path[2] = (200.0, 0.0), path[3] = (200.0, 20.0), ...
```
"""
function lawnmower_from_extents(spec::LawnmowerSpec)
    xmin, xmax = Float64(spec.xmin), Float64(spec.xmax)
    ymin, ymax = Float64(spec.ymin), Float64(spec.ymax)
    sp         = Float64(spec.spacing)

    path = NTuple{2,Float64}[]

    if spec.primary === :x
        # Sweep along x, step along y
        ys = _sweep_range(ymin, ymax, sp)
        spec.start === :high && reverse!(ys)

        for (k, y) in enumerate(ys)
            # alternate sweep direction
            if isodd(k)
                push!(path, (xmin, y))
                push!(path, (xmax, y))
            else
                push!(path, (xmax, y))
                push!(path, (xmin, y))
            end
        end

    elseif spec.primary === :y
        # Sweep along y, step along x
        xs = _sweep_range(xmin, xmax, sp)
        spec.start === :high && reverse!(xs)

        for (k, x) in enumerate(xs)
            if isodd(k)
                push!(path, (x, ymin))
                push!(path, (x, ymax))
            else
                push!(path, (x, ymax))
                push!(path, (x, ymin))
            end
        end

    else
        throw(ArgumentError("spec.primary must be :x or :y, got $(spec.primary)"))
    end

    # If yaw_deg != 0 rotate the path about the bounding-box centre
    if spec.yaw_deg != 0
        cx = (xmin + xmax) / 2; cy = (ymin + ymax) / 2
        cosθ, sinθ = _rotation_matrix(spec.yaw_deg)
        path = map(path) do (px, py)
            rx = px - cx; ry = py - cy
            (cx + cosθ*rx - sinθ*ry, cy + sinθ*rx + cosθ*ry)
        end
    end

    return path
end

# ---------------------------------------------------------------------------
# Line-ID annotation
# ---------------------------------------------------------------------------

"""
    line_id_for_vertex(path_index::Int) -> Int

Return the 1-based flight-line index for a vertex at position `path_index`
in the path returned by `lawnmower_from_extents`. Two consecutive vertices
share the same line id; every other pair is a turnaround.

Assumes path was generated with alternating (boustrophedon) sweeps of two
vertices per line segment.
"""
line_id_for_vertex(path_index::Int) = (path_index + 1) ÷ 2

"""
    annotate_line_ids!(wps::Vector{Waypoint}, path::Vector) -> Vector{Waypoint}

Fill the `line_id` field of each `Waypoint` based on its nearest path segment.
Waypoints inherit the line id of the path segment whose midpoint is closest.

Returns the modified vector (mutation is in-place).
"""
function annotate_line_ids!(wps::Vector{Waypoint},
                            path::Vector{NTuple{2,Float64}})
    n_seg = length(path) - 1
    for (wi, wp) in enumerate(wps)
        best_seg = 1
        best_d   = Inf
        for s in 1:n_seg
            p1, p2 = path[s], path[s+1]
            mx = (p1[1] + p2[1]) / 2; my = (p1[2] + p2[2]) / 2
            d  = hypot(wp.x - mx, wp.y - my)
            if d < best_d
                best_d   = d
                best_seg = s
            end
        end
        lid = line_id_for_vertex(best_seg)
        wps[wi] = Waypoint(wp.x, wp.y, wp.altitude, wp.speed; line_id=lid)
    end
    return wps
end
