"""
    waypoints.jl — Unified waypoint generator

Public API
----------
- `generate_waypoints(path, grid, strategy; kwargs...) -> Vector{Waypoint}`
  Single entry point that dispatches on the `SpeedStrategy` type to produce
  appropriately spaced, speed-tagged waypoints.

Strategy → spacing logic mapping
---------------------------------
| Strategy              | Spacing logic                              |
|-----------------------|--------------------------------------------|
| ConstantSpeed         | Uniform spacing (seconds_per_wp × v)       |
| KDEGuidedSpeed        | Density-proportional inverse-linear spacing|
| CurvatureGuidedSpeed  | Gradient + curvature adaptive formula      |

All strategies share the same loop skeleton; only the `_compute_step` helper
differs (multiple dispatch).

Internal helpers (not exported)
--------------------------------
- `_compute_step(strategy, g, κ, speed, ...) -> Float64`
- `_reference_scales(path, grid, strategy; ...) -> (g0, k0)`
- `_smooth_speeds!(wps; window) -> wps`
- `_quantise_speed(v, q) -> Float64`
"""

# ---------------------------------------------------------------------------
# Spacing formula helpers (multiple dispatch on strategy)
# ---------------------------------------------------------------------------

"""
    _compute_step(strategy, g, κ, speed, spacing_min, spacing_max,
                  g0, k0, seconds_per_wp) -> Float64

Compute the next forward step (metres) given current gradient magnitude `g`,
directional curvature `κ`, current `speed` (m/s), and strategy-specific params.

Dispatch table
--------------
- `ConstantSpeed`        → `speed × seconds_per_wp` (uniform time budget)
- `KDEGuidedSpeed`       → `speed × seconds_per_wp` (density drives speed,
                           time budget drives spacing)
- `CurvatureGuidedSpeed` → curvature formula (see below)

Curvature-spaced KDE-guided formula
------------------------------------
    w = (|∇d|/g0)^α + λ·(|κ|/k0)^η
    u = 1 / (1 + w)
    s = spacing_min + (spacing_max − spacing_min) · u

`g0`, `k0` are reference scales (median over the path, or user-supplied).
"""
function _compute_step end

function _compute_step(::ConstantSpeed, g, κ, speed,
                        spacing_min, spacing_max, g0, k0, seconds_per_wp)
    return clamp(speed * seconds_per_wp, spacing_min, spacing_max)
end

function _compute_step(::KDEGuidedSpeed, g, κ, speed,
                        spacing_min, spacing_max, g0, k0, seconds_per_wp)
    return clamp(speed * seconds_per_wp, spacing_min, spacing_max)
end

function _compute_step(s::CurvatureGuidedSpeed, g, κ, speed,
                        spacing_min, spacing_max, g0, k0, seconds_per_wp)
    w = (g / g0)^s.alpha + s.lambda * (abs(κ) / k0)^s.eta
    u = 1.0 / (1.0 + w)
    return clamp(s.spacing_min + (s.spacing_max - s.spacing_min) * u,
                 s.spacing_min, s.spacing_max)
end

# ---------------------------------------------------------------------------
# Reference-scale estimation for curvature strategy
# ---------------------------------------------------------------------------

"""
    _reference_scales(path, grid, strategy; sampler, h_factor, n_probe_per_seg)
        -> (g0::Float64, k0::Float64)

Estimate reference gradient and curvature scales by sampling the path.
For `CurvatureGuidedSpeed`, uses the median of sampled values unless the
user already provided `grad_ref` / `curv_ref`.
For all other strategies, returns `(1.0, 1.0)` (unused).
"""
function _reference_scales(path, grid::RasterGrid, strategy::SpeedStrategy;
                             sampler::Symbol=:bilinear,
                             h_factor::Real=0.5,
                             n_probe_per_seg::Int=32)
    # For non-curvature strategies, reference scales are irrelevant
    strategy isa CurvatureGuidedSpeed || return (1.0, 1.0)

    # Use user-provided refs if both are specified
    g0_user = strategy.grad_ref
    k0_user = strategy.curv_ref
    if !isnothing(g0_user) && !isnothing(k0_user)
        return (max(Float64(g0_user), eps()), max(Float64(k0_user), eps()))
    end

    grads = Float64[]
    curvs = Float64[]
    smin  = strategy.spacing_min

    for i in 1:length(path)-1
        p1, p2 = path[i], path[i+1]
        dx = p2[1]-p1[1]; dy = p2[2]-p1[2]
        L  = hypot(dx, dy)
        L == 0 && continue
        ux, uy = dx/L, dy/L
        n_probe = min(128, max(n_probe_per_seg, ceil(Int, L / max(smin, 1.0))))
        @inbounds for k in 0:n_probe
            t = k / n_probe
            x = p1[1] + ux * (t*L)
            y = p1[2] + uy * (t*L)
            push!(grads, gradient_magnitude(grid, x, y; sampler=sampler, h_factor=h_factor))
            push!(curvs, abs(directional_curvature(grid, x, y, ux, uy;
                                                    sampler=sampler, h_factor=h_factor)))
        end
    end

    g0 = if isnothing(g0_user)
        isempty(grads) ? 1.0 : max(median(grads), eps())
    else
        max(Float64(g0_user), eps())
    end
    k0 = if isnothing(k0_user)
        isempty(curvs) ? 1.0 : max(median(curvs), eps())
    else
        max(Float64(k0_user), eps())
    end

    return g0, k0
end

# ---------------------------------------------------------------------------
# Speed quantisation
# ---------------------------------------------------------------------------

"""
    _quantise_speed(v, q) -> Float64

Round speed `v` to the nearest multiple of `q`. If `q ≤ 0`, returns `v`
unchanged. Useful for matching the discrete speed commands sent to the drone
autopilot (e.g. nearest 0.25 m/s).
"""
function _quantise_speed(v::Real, q::Real)
    q <= 0 && return Float64(v)
    return round(Float64(v) / Float64(q)) * Float64(q)
end

# ---------------------------------------------------------------------------
# Moving-average speed smoother
# ---------------------------------------------------------------------------

"""
    smooth_speeds!(wps::Vector{Waypoint}; window=5) -> Vector{Waypoint}

Apply a centred moving-average of width `window` (must be odd) to the speed
field of each waypoint. Modifies `wps` in-place and returns it.
"""
function smooth_speeds!(wps::Vector{Waypoint}; window::Int=5)
    isodd(window) || throw(ArgumentError("window must be odd"))
    n    = length(wps)
    half = (window - 1) ÷ 2
    speeds    = [w.speed for w in wps]
    smoothed  = similar(speeds)
    @inbounds for i in 1:n
        lo = max(1, i - half); hi = min(n, i + half)
        smoothed[i] = sum(@view speeds[lo:hi]) / (hi - lo + 1)
    end
    @inbounds for i in 1:n
        w = wps[i]
        wps[i] = Waypoint(w.x, w.y, w.altitude, smoothed[i]; line_id=w.line_id)
    end
    return wps
end

# ---------------------------------------------------------------------------
# Waypoint spatial filter
# ---------------------------------------------------------------------------

"""
    filter_bbox(wps::Vector{Waypoint}; xmin, xmax, ymin, ymax) -> Vector{Waypoint}

Return only waypoints inside the given bounding box.
"""
function filter_bbox(wps::Vector{Waypoint};
                     xmin::Real=-Inf, xmax::Real=Inf,
                     ymin::Real=-Inf, ymax::Real=Inf)
    return filter(w -> xmin <= w.x <= xmax && ymin <= w.y <= ymax, wps)
end

# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

"""
    generate_waypoints(path, grid, strategy;
                       altitude          = 80.0,
                       sampler           = :bilinear,
                       include_vertices  = true,
                       seconds_per_wp    = 1.0,
                       spacing_min       = 2.0,
                       spacing_max       = 20.0,
                       min_step          = 0.5,
                       speed_quantise    = 0.0,
                       smooth_window     = nothing,
                       h_factor          = 0.5,
                       time_budget       = nothing) -> Vector{Waypoint}

Generate `Waypoint`s along `path` using the given `strategy`.

Parameters
----------
- `path`:             Vector of 2-tuples `(x, y)` from `lawnmower_from_extents`
- `grid`:             `RasterGrid` density surface
- `strategy`:         A `SpeedStrategy` (dispatch selects spacing logic)
- `altitude`:         Flight altitude AGL (m)
- `sampler`:          `:bilinear` or `:nearest`
- `include_vertices`: Always include path corner vertices (recommended: `true`)
- `seconds_per_wp`:   Time-budget spacing for constant/KDE strategies (s)
- `spacing_min`:      Minimum waypoint spacing (m); overrides time budget
- `spacing_max`:      Maximum waypoint spacing (m)
- `min_step`:         Hard minimum step size (m) to avoid degenerate loops
- `speed_quantise`:   Round speeds to nearest `q` m/s (0 = no rounding)
- `smooth_window`:    If an odd Int, apply moving-average smoothing of that width
- `h_factor`:         Finite-difference step as fraction of local cell size
- `time_budget`:      Optional total flight-time cap (seconds). Generation stops
                      when elapsed estimated time exceeds this value.

Returns
-------
`Vector{Waypoint}` — ordered waypoints including speed assignments.

Notes
-----
All path vertices (`include_vertices = true`) are always emitted even if
their spacing is smaller than `spacing_min`. Between vertices, intermediate
waypoints are inserted at the strategy-computed spacing.
"""
function generate_waypoints(path::AbstractVector,
                             grid::RasterGrid,
                             strategy::SpeedStrategy;
                             altitude         ::Real    = 80.0,
                             sampler          ::Symbol  = :bilinear,
                             include_vertices ::Bool    = true,
                             seconds_per_wp   ::Real    = 1.0,
                             spacing_min      ::Real    = 2.0,
                             spacing_max      ::Real    = 20.0,
                             min_step         ::Real    = 0.5,
                             speed_quantise   ::Real    = 0.0,
                             smooth_window    ::Union{Nothing,Int} = nothing,
                             h_factor         ::Real    = 0.5,
                             time_budget      ::Union{Nothing,Real} = nothing)

    # --- validate
    length(path) >= 2 || throw(ArgumentError("path must have ≥ 2 vertices"))
    spacing_min > 0   || throw(ArgumentError("spacing_min must be > 0"))
    spacing_max >= spacing_min || throw(ArgumentError("spacing_max must be ≥ spacing_min"))
    min_step > 0      || throw(ArgumentError("min_step must be > 0"))
    altitude > 0      || throw(ArgumentError("altitude must be > 0"))

    # --- pre-compute reference scales (for CurvatureGuidedSpeed only)
    g0, k0 = _reference_scales(path, grid, strategy;
                                sampler=sampler, h_factor=h_factor)

    speed_fn = _speed_fn(strategy)

    out = Waypoint[]
    elapsed_time = 0.0    # running estimate of mission time (s)

    for i in 1:length(path)-1
        p1, p2 = path[i], path[i+1]
        x1, y1 = Float64(p1[1]), Float64(p1[2])
        x2, y2 = Float64(p2[1]), Float64(p2[2])

        dx = x2 - x1; dy = y2 - y1
        L  = hypot(dx, dy)
        L == 0 && continue
        ux, uy = dx/L, dy/L

        # tolerance for "close enough to endpoint"
        tol = max(1e-9 * max(L, 1.0), eps(Float64))

        # --- emit start vertex
        if include_vertices
            d0  = sample_density(grid, x1, y1; sampler=sampler)
            v0  = _quantise_speed(speed_fn(d0), speed_quantise)
            push!(out, Waypoint(x1, y1, altitude, v0))

            # time-budget check
            if !isnothing(time_budget) && elapsed_time >= time_budget
                break
            end
        end

        # --- initialise step from start
        d0 = sample_density(grid, x1, y1; sampler=sampler)
        g0_loc = gradient_magnitude(grid, x1, y1; sampler=sampler, h_factor=h_factor)
        κ0_loc = directional_curvature(grid, x1, y1, ux, uy; sampler=sampler, h_factor=h_factor)
        v0     = speed_fn(d0)

        s0  = _compute_step(strategy, g0_loc, κ0_loc, v0,
                             spacing_min, spacing_max, g0, k0, seconds_per_wp)
        pos = clamp(s0, min_step, L)

        # --- walk along segment
        while true
            x = x1 + ux*pos; y = y1 + uy*pos

            d   = sample_density(grid, x, y; sampler=sampler)
            spd = _quantise_speed(speed_fn(d), speed_quantise)
            g_  = gradient_magnitude(grid, x, y; sampler=sampler, h_factor=h_factor)
            κ_  = directional_curvature(grid, x, y, ux, uy;
                                         sampler=sampler, h_factor=h_factor)

            is_end = pos >= L - tol

            if include_vertices || !is_end
                push!(out, Waypoint(x, y, altitude, spd))

                # running time estimate: distance / speed
                seg_d = length(out) >= 2 ?
                    hypot(out[end].x - out[end-1].x, out[end].y - out[end-1].y) : pos
                elapsed_time += seg_d / max(spd, eps())

                if !isnothing(time_budget) && elapsed_time >= time_budget
                    @goto done
                end
            end

            rem = L - pos
            rem <= 0 && break

            step = _compute_step(strategy, g_, κ_, spd,
                                  spacing_min, spacing_max, g0, k0, seconds_per_wp)
            step = clamp(step, min_step, rem)
            pos += step
        end

        # --- emit end vertex on last segment
        if include_vertices && i == length(path) - 1
            d2  = sample_density(grid, x2, y2; sampler=sampler)
            v2  = _quantise_speed(speed_fn(d2), speed_quantise)
            if isempty(out) || !(out[end].x ≈ x2 && out[end].y ≈ y2)
                push!(out, Waypoint(x2, y2, altitude, v2))
            end
        end
    end

    @label done

    # --- optional post-processing
    if !isnothing(smooth_window)
        smooth_speeds!(out; window=smooth_window)
    end

    return out
end

# ---------------------------------------------------------------------------
# High-level convenience wrapper
# ---------------------------------------------------------------------------

"""
    plan_mission(grid, config::FlightConfig; spec=nothing, kwargs...) -> Vector{Waypoint}

Generate waypoints for an entire mission given a `FlightConfig`.

If `spec` is `nothing`, a `LawnmowerSpec` is auto-constructed from the grid
extents and `config.line_spacing`. Otherwise the provided `spec` is used.

All `kwargs` are forwarded to `generate_waypoints`.
"""
function plan_mission(grid::RasterGrid,
                      config::FlightConfig;
                      spec  ::Union{Nothing,LawnmowerSpec}=nothing,
                      kwargs...)

    if isnothing(spec)
        xmin, xmax = extrema(grid.xs)
        ymin, ymax = extrema(grid.ys)
        spec = LawnmowerSpec(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                             spacing=config.line_spacing,
                             yaw_deg=zero(Float64))
    end

    path = lawnmower_from_extents(spec)
    wps  = generate_waypoints(path, grid, config.strategy;
                               altitude=config.altitude,
                               kwargs...)
    return wps
end
