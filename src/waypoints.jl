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
    # Single source of truth for spacing bounds: the spacing_min/spacing_max
    # ARGUMENTS threaded through generate_waypoints (previously this used the
    # struct fields s.spacing_min/s.spacing_max, silently ignoring caller/config
    # bounds — see hubbard diagnosis RC1).
    w = (g / g0)^s.alpha + s.lambda * (abs(κ) / k0)^s.eta
    u = 1.0 / (1.0 + w)
    return clamp(spacing_min + (spacing_max - spacing_min) * u,
                 spacing_min, spacing_max)
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
# Event-driven two-pass placement (CurvatureGuidedSpeed)
# ---------------------------------------------------------------------------
#
# Rationale (hubbard diagnosis RC2/RC3): the legacy marcher was purely local,
# forward-only and median-normalised, so uniform regions never reached
# spacing_max and narrow transitions could be leapt over. The two-pass placer
# below decouples *where transitions are* (Pass 1: fine probe + absolute
# triggers) from *how to space* (Pass 2: max in quiet intervals, min inside
# events, with a forced anticipatory point before and trailing point after
# every event).

"""
    _event_triggers(path, grid; sampler, h_factor, probe_spacing,
                    trigger_quantile, grad_floor, curv_floor) -> (g_hi, k_hi)

Derive absolute gradient/curvature event triggers. The floor
(`grad_floor`/`curv_floor`, density-per-metre and per-metre²) is the absolute
minimum sensitivity and the primary criterion; a HIGH quantile (default 0.9) of
the probed |∇d|/|κ| can only RAISE the threshold on genuinely rough surfaces.
This is deliberately NOT median-relative — on a uniform or gently-rippled
surface the quantile falls below the floor, so quiet terrain never trips an
event and stays at spacing_max.
"""
function _event_triggers(path, grid::RasterGrid;
                          sampler::Symbol=:bilinear,
                          h_factor::Real=0.5,
                          probe_spacing::Real=2.0,
                          trigger_quantile::Real=0.9,
                          grad_floor::Real=1e-2,
                          curv_floor::Real=5e-3)
    grads = Float64[]
    curvs = Float64[]
    for i in 1:length(path)-1
        p1, p2 = path[i], path[i+1]
        dx = p2[1]-p1[1]; dy = p2[2]-p1[2]
        L  = hypot(dx, dy)
        L == 0 && continue
        ux, uy = dx/L, dy/L
        n = max(2, ceil(Int, L / max(probe_spacing, eps())))
        @inbounds for k in 0:n
            t = k / n
            x = p1[1] + ux*(t*L); y = p1[2] + uy*(t*L)
            push!(grads, gradient_magnitude(grid, x, y; sampler=sampler, h_factor=h_factor))
            push!(curvs, abs(directional_curvature(grid, x, y, ux, uy;
                                                    sampler=sampler, h_factor=h_factor)))
        end
    end
    g_hi = isempty(grads) ? grad_floor : max(grad_floor, quantile(grads, trigger_quantile))
    k_hi = isempty(curvs) ? curv_floor : max(curv_floor, quantile(curvs, trigger_quantile))
    return g_hi, k_hi
end

"""
    _detect_event_intervals(ss, ds, gs, ks; g_hi, k_hi, density_threshold)
        -> Vector{Tuple{Float64,Float64}}

Pass 1. Given fine along-track probes (`ss` arc positions, `ds` density,
`gs` |∇d|, `ks` |κ|), return the core event intervals `[a, b]` (arc length)
where any of these absolute triggers fire:

- `gs[k] ≥ g_hi`                              (gradient trigger)
- `ks[k] ≥ k_hi`                              (curvature trigger)
- `ds` crosses `density_threshold` between k-1 and k (level crossing)

Density *level crossings* — not plain exceedance — are used so a broad
high-density plateau keeps a quiet interior (only its entering/exiting edges
become events). Contiguous flagged probes are coalesced into one interval.
"""
function _detect_event_intervals(ss::Vector{Float64}, ds::Vector{Float64},
                                  gs::Vector{Float64}, ks::Vector{Float64};
                                  g_hi::Real, k_hi::Real,
                                  density_threshold::Union{Real,Nothing}=0.5)
    n = length(ss)
    flag = falses(n)
    @inbounds for k in 1:n
        if gs[k] >= g_hi || ks[k] >= k_hi
            flag[k] = true
        end
    end
    if !isnothing(density_threshold)
        thr = Float64(density_threshold)
        @inbounds for k in 2:n
            if (ds[k-1] - thr) * (ds[k] - thr) < 0.0   # sign change ⇒ crossing
                flag[k-1] = true; flag[k] = true
            end
        end
    end
    intervals = Tuple{Float64,Float64}[]
    k = 1
    while k <= n
        if flag[k]
            j = k
            while j < n && flag[j+1]
                j += 1
            end
            push!(intervals, (ss[k], ss[j]))
            k = j + 1
        else
            k += 1
        end
    end
    return intervals
end

"""
    _finalise_intervals(intervals, L; margin, merge_gap)
        -> Vector{Tuple{Float64,Float64}}

Expand each core event interval by `margin` metres on both sides (the
anticipatory/trailing band), clamp to `[0, L]`, then merge intervals whose gap
is below `merge_gap`. Guarantees the forced before/after points land at the
expanded interval edges.
"""
function _finalise_intervals(intervals::Vector{Tuple{Float64,Float64}}, L::Real;
                             margin::Real, merge_gap::Real)
    isempty(intervals) && return intervals
    exp = [(max(0.0, a - margin), min(Float64(L), b + margin)) for (a, b) in intervals]
    sort!(exp; by=first)
    merged = Tuple{Float64,Float64}[exp[1]]
    for (a, b) in exp[2:end]
        la, lb = merged[end]
        if a - lb <= merge_gap
            merged[end] = (la, max(lb, b))
        else
            push!(merged, (a, b))
        end
    end
    return merged
end

"""
    _place_positions(L, intervals; spacing_min, spacing_max, min_step)
        -> Vector{Float64}

Pass 2. Produce sorted arc positions in `[0, L]` (inclusive of both endpoints).
Quiet gaps are stepped at exactly `spacing_max`; event intervals at exactly
`spacing_min`. Interval edges are always emitted, structurally guaranteeing an
anticipatory point before and a trailing point after every event. Positions
closer than `min_step` are collapsed.
"""
function _place_positions(L::Real, intervals::Vector{Tuple{Float64,Float64}};
                          spacing_min::Real, spacing_max::Real, min_step::Real)
    L = Float64(L)
    pts = Float64[0.0]
    cursor = 0.0
    _fill!(pts, lo, hi, Δ) = begin
        p = lo
        while p + Δ < hi - 1e-9
            p += Δ
            push!(pts, p)
        end
        push!(pts, hi)
    end
    for (a, b) in intervals
        a = clamp(a, 0.0, L); b = clamp(b, 0.0, L)
        a <= cursor && (a = cursor)
        if a > cursor + 1e-9          # quiet run before the event
            _fill!(pts, cursor, a, spacing_max)
        end
        if b > a + 1e-9               # event run (fine spacing)
            _fill!(pts, a, b, spacing_min)
        end
        cursor = max(cursor, b)
    end
    if cursor < L - 1e-9              # trailing quiet run
        _fill!(pts, cursor, L, spacing_max)
    end
    push!(pts, L)
    sort!(pts)
    # collapse near-duplicates / sub-min_step steps
    out = Float64[pts[1]]
    for p in pts[2:end]
        if p - out[end] >= min_step - 1e-9
            push!(out, p)
        end
    end
    out[end] < L - 1e-9 && push!(out, L)   # never drop the endpoint
    out[end] = L
    return out
end

"""
    _segment_event_positions(grid, p1, p2, ux, uy, L; ...) -> Vector{Float64}

Convenience wrapper running Pass 1 + Pass 2 for a single segment.
"""
function _segment_event_positions(grid::RasterGrid, p1, p2, ux, uy, L::Real;
                                   spacing_min::Real, spacing_max::Real,
                                   min_step::Real, g_hi::Real, k_hi::Real,
                                   density_threshold::Union{Real,Nothing},
                                   event_margin::Real, probe_spacing::Real,
                                   sampler::Symbol, h_factor::Real)
    n = max(2, ceil(Int, L / max(probe_spacing, eps())))
    ss = Float64[]; ds = Float64[]; gs = Float64[]; ks = Float64[]
    @inbounds for k in 0:n
        t = k / n
        s = t * L
        x = p1[1] + ux*s; y = p1[2] + uy*s
        push!(ss, s)
        push!(ds, sample_density(grid, x, y; sampler=sampler))
        push!(gs, gradient_magnitude(grid, x, y; sampler=sampler, h_factor=h_factor))
        push!(ks, abs(directional_curvature(grid, x, y, ux, uy;
                                            sampler=sampler, h_factor=h_factor)))
    end
    core = _detect_event_intervals(ss, ds, gs, ks;
                                   g_hi=g_hi, k_hi=k_hi,
                                   density_threshold=density_threshold)
    merge_gap = max(spacing_min, min_step)
    intervals = _finalise_intervals(core, L; margin=event_margin, merge_gap=merge_gap)
    return _place_positions(L, intervals;
                            spacing_min=spacing_min, spacing_max=spacing_max,
                            min_step=min_step)
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
                       time_budget       = nothing,
                       # --- event-driven placement (CurvatureGuidedSpeed) ---
                       event_driven      = true,
                       grad_trigger      = :auto,
                       curv_trigger      = :auto,
                       density_threshold = 0.5,
                       event_margin      = nothing,
                       probe_spacing     = nothing,
                       trigger_quantile  = 0.9,
                       dedup             = true) -> Vector{Waypoint}

Generate `Waypoint`s along `path` using the given `strategy`.

Spacing-bounds source of truth
------------------------------
The `spacing_min` / `spacing_max` **arguments** are authoritative for every
strategy, including `CurvatureGuidedSpeed` (this fixes the historical foot-gun
where the struct fields silently overrode caller/config bounds — see the
hubbard diagnosis RC1). The `CurvatureGuidedSpeed` struct still carries
`spacing_min`/`spacing_max` fields for API compatibility, but they no longer
drive placement.

Placement algorithm
-------------------
- `ConstantSpeed` / `KDEGuidedSpeed`: uniform time-budget marcher
  (`clamp(speed·seconds_per_wp, spacing_min, spacing_max)`).
- `CurvatureGuidedSpeed` with `event_driven=true` (default): a two-pass,
  event-driven placer per segment. Pass 1 probes density/gradient/curvature on
  a fine grid and flags **event intervals** from absolute gradient/curvature
  triggers and density-level crossings. Pass 2 places `spacing_max`-spaced
  points in quiet intervals, `spacing_min`-spaced points inside events, and
  forces one anticipatory point before and one trailing point after every
  event. Set `event_driven=false` to fall back to the legacy local marcher.

Event-detection parameters (all explicit / configurable)
--------------------------------------------------------
- `grad_trigger`:      absolute |∇d| trigger, or `:auto` (high quantile + floor)
- `curv_trigger`:      absolute |κ| trigger, or `:auto`
- `density_threshold`: density level whose crossings mark an edge (`nothing`
                       disables level-crossing detection)
- `event_margin`:      metres of anticipatory/trailing band each side of an
                       event core (default `spacing_min`)
- `probe_spacing`:     fine probe step in metres (default `spacing_min/2`)
- `trigger_quantile`:  quantile used by `:auto` triggers (default 0.9; NOT the
                       median, so uniform terrain never trips an event)

Other parameters
----------------
- `dedup`:             drop consecutive duplicate coordinates (fixes RC5)
- remaining parameters as before (altitude, sampler, include_vertices, …)

`line_id` is populated per survey line from the path segment index (fixes RC4).

Returns
-------
`Vector{Waypoint}` — ordered, deduplicated, line-id-tagged waypoints.
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
                             time_budget      ::Union{Nothing,Real} = nothing,
                             event_driven     ::Bool    = true,
                             grad_trigger     ::Union{Real,Symbol}  = :auto,
                             curv_trigger     ::Union{Real,Symbol}  = :auto,
                             density_threshold::Union{Real,Nothing} = 0.5,
                             event_margin     ::Union{Real,Nothing} = nothing,
                             probe_spacing    ::Union{Real,Nothing} = nothing,
                             trigger_quantile ::Real    = 0.9,
                             dedup            ::Bool    = true)

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

    use_events = event_driven && strategy isa CurvatureGuidedSpeed
    margin  = isnothing(event_margin)  ? Float64(spacing_min)     : Float64(event_margin)
    probe_s = isnothing(probe_spacing) ? Float64(spacing_min)/2   : Float64(probe_spacing)
    probe_s = max(probe_s, min_step, 1e-3)

    # resolve absolute triggers (once, over the whole path)
    g_hi = 0.0; k_hi = 0.0
    if use_events
        ag, ak = _event_triggers(path, grid; sampler=sampler, h_factor=h_factor,
                                  probe_spacing=probe_s, trigger_quantile=trigger_quantile)
        g_hi = grad_trigger isa Symbol ? ag : Float64(grad_trigger)
        k_hi = curv_trigger isa Symbol ? ak : Float64(curv_trigger)
    end

    out = Waypoint[]
    elapsed_time = 0.0        # running estimate of mission time (s)
    budget_hit   = false
    dedup_eps    = max(1e-6, 1e-7 * max(maximum(grid.xs) - minimum(grid.xs),
                                        maximum(grid.ys) - minimum(grid.ys)))

    # unified emitter: assigns speed + line_id, dedups, tracks the time budget.
    function emit!(x, y, lid)
        if dedup && !isempty(out) &&
           hypot(x - out[end].x, y - out[end].y) <= dedup_eps
            return                                   # RC5: no consecutive dup
        end
        d   = sample_density(grid, x, y; sampler=sampler)
        spd = _quantise_speed(speed_fn(d), speed_quantise)
        if !isempty(out)
            elapsed_time += hypot(x - out[end].x, y - out[end].y) / max(spd, eps())
        end
        push!(out, Waypoint(x, y, altitude, spd; line_id=lid))
        if !isnothing(time_budget) && elapsed_time >= time_budget
            budget_hit = true
        end
        return
    end

    for i in 1:length(path)-1
        budget_hit && break
        p1, p2 = path[i], path[i+1]
        x1, y1 = Float64(p1[1]), Float64(p1[2])
        x2, y2 = Float64(p2[1]), Float64(p2[2])

        dx = x2 - x1; dy = y2 - y1
        L  = hypot(dx, dy)
        L == 0 && continue
        ux, uy = dx/L, dy/L
        tol = max(1e-9 * max(L, 1.0), eps(Float64))
        lid = line_id_for_vertex(i)          # RC4: populate line_id

        if use_events
            # ---- two-pass event-driven placement (RC2/RC3) ----
            positions = _segment_event_positions(grid, p1, p2, ux, uy, L;
                            spacing_min=spacing_min, spacing_max=spacing_max,
                            min_step=min_step, g_hi=g_hi, k_hi=k_hi,
                            density_threshold=density_threshold,
                            event_margin=margin, probe_spacing=probe_s,
                            sampler=sampler, h_factor=h_factor)
            for s in positions
                is_end = s >= L - tol
                if !include_vertices && (s <= tol || is_end)
                    continue
                end
                emit!(x1 + ux*s, y1 + uy*s, lid)
                budget_hit && break
            end
        else
            # ---- legacy local forward marcher (Constant/KDE) ----
            if include_vertices
                emit!(x1, y1, lid)
                budget_hit && break
            end
            g0_loc = gradient_magnitude(grid, x1, y1; sampler=sampler, h_factor=h_factor)
            κ0_loc = directional_curvature(grid, x1, y1, ux, uy; sampler=sampler, h_factor=h_factor)
            v0     = speed_fn(sample_density(grid, x1, y1; sampler=sampler))
            s0  = _compute_step(strategy, g0_loc, κ0_loc, v0,
                                 spacing_min, spacing_max, g0, k0, seconds_per_wp)
            pos = clamp(s0, min_step, L)
            while true
                x = x1 + ux*pos; y = y1 + uy*pos
                is_end = pos >= L - tol
                if include_vertices || !is_end
                    emit!(x, y, lid)
                    budget_hit && break
                end
                rem = L - pos
                rem <= 0 && break
                spd = _quantise_speed(speed_fn(sample_density(grid, x, y; sampler=sampler)),
                                      speed_quantise)
                g_  = gradient_magnitude(grid, x, y; sampler=sampler, h_factor=h_factor)
                κ_  = directional_curvature(grid, x, y, ux, uy; sampler=sampler, h_factor=h_factor)
                step = _compute_step(strategy, g_, κ_, spd,
                                      spacing_min, spacing_max, g0, k0, seconds_per_wp)
                step = clamp(step, min_step, rem)
                pos += step
            end
            budget_hit && break
        end

        # --- emit end vertex on the final segment (dedup handles interior corners)
        if include_vertices && i == length(path) - 1
            emit!(x2, y2, lid)
        end
    end

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
