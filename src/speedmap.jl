"""
    speedmap.jl — Speed assignment via multiple dispatch on SpeedStrategy

Public API
----------
- `assign_speed(strategy, d) -> Float64`
  Map a scalar density value `d` to a target speed (m/s).

- `speed_bounds(strategy) -> (vmin, vmax)`
  Return the speed limits of a strategy.

Internal helper (not exported)
-------------------------------
- `_inverse_linear_speed(d; dmin, dmax, vmin, vmax) -> Float64`
  The core inverse-linear formula used by KDE-guided strategies.
"""

# ---------------------------------------------------------------------------
# Core formula (used by multiple strategies)
# ---------------------------------------------------------------------------

"""
    _inverse_linear_speed(d; dmin, dmax, vmin, vmax) -> Float64

Inverse-linear density-to-speed map with clamping:

    t  = clamp((d − dmin) / (dmax − dmin), 0, 1)   # normalised density
    v  = vmin + (1 − t) * (vmax − vmin)             # speed (high d → low v)

When `dmax == dmin` (flat surface), returns the midpoint `(vmin + vmax) / 2`.

This formula is the core of the KDE-guided speed assignment described in
the manuscript (Section 2.x). Higher KDE density indicates more complex
canopy structure and drives the UAV to slow down, increasing dwell time and
therefore ground-return spatial coverage.
"""
function _inverse_linear_speed(d::Real;
                                dmin::Real, dmax::Real,
                                vmin::Real, vmax::Real)
    dmax == dmin && return (vmin + vmax) / 2.0
    t = clamp((Float64(d) - Float64(dmin)) / (Float64(dmax) - Float64(dmin)), 0.0, 1.0)
    return Float64(vmin) + (1.0 - t) * (Float64(vmax) - Float64(vmin))
end

# ---------------------------------------------------------------------------
# assign_speed — multiple dispatch
# ---------------------------------------------------------------------------

"""
    assign_speed(strategy::SpeedStrategy, d::Real) -> Float64

Map density value `d` to a target ground speed (m/s) according to `strategy`.
The result is always clamped to the strategy's `[vmin, vmax]` range.
"""
function assign_speed end

"""
    assign_speed(strategy::ConstantSpeed, d) -> Float64

Always returns `strategy.v` regardless of density.
"""
assign_speed(strategy::ConstantSpeed, ::Real) = strategy.v

"""
    assign_speed(strategy::KDEGuidedSpeed, d) -> Float64

Applies the inverse-linear density-to-speed formula.
"""
function assign_speed(strategy::KDEGuidedSpeed, d::Real)
    v = _inverse_linear_speed(d;
            dmin=strategy.dmin, dmax=strategy.dmax,
            vmin=strategy.vmin, vmax=strategy.vmax)
    return clamp(v, strategy.vmin, strategy.vmax)
end

"""
    assign_speed(strategy::CurvatureGuidedSpeed, d) -> Float64

Speed assignment is identical to `KDEGuidedSpeed`; the curvature-adaptive
spacing is handled separately in `generate_waypoints`.
"""
function assign_speed(strategy::CurvatureGuidedSpeed, d::Real)
    v = _inverse_linear_speed(d;
            dmin=strategy.dmin, dmax=strategy.dmax,
            vmin=strategy.vmin, vmax=strategy.vmax)
    return clamp(v, strategy.vmin, strategy.vmax)
end

# ---------------------------------------------------------------------------
# speed_bounds — extract [vmin, vmax] from any strategy
# ---------------------------------------------------------------------------

"""
    speed_bounds(strategy::SpeedStrategy) -> (vmin, vmax)

Return the `(vmin, vmax)` speed range for `strategy`.
For `ConstantSpeed`, both bounds equal `strategy.v`.
"""
function speed_bounds end

speed_bounds(s::ConstantSpeed)        = (s.v, s.v)
speed_bounds(s::KDEGuidedSpeed)       = (s.vmin, s.vmax)
speed_bounds(s::CurvatureGuidedSpeed) = (s.vmin, s.vmax)

# ---------------------------------------------------------------------------
# Convenience: build a speed map function (used internally in waypoints.jl)
# ---------------------------------------------------------------------------

"""
    _speed_fn(strategy) -> (d -> Float64)

Return a closure `d -> speed` for use inside tight waypoint-generation loops.
"""
_speed_fn(strategy::SpeedStrategy) = d -> assign_speed(strategy, d)

# ---------------------------------------------------------------------------
# Monotonicity check (useful for tests and diagnostics)
# ---------------------------------------------------------------------------

"""
    is_monotone_decreasing(strategy; n=100) -> Bool

Test that `assign_speed(strategy, ·)` is non-increasing on a uniform sample
of `n` density values spanning `[dmin, dmax]`.

Returns `true` for `KDEGuidedSpeed` and `CurvatureGuidedSpeed` (both use
the inverse-linear map). Always returns `true` for `ConstantSpeed`.
"""
function is_monotone_decreasing(strategy::SpeedStrategy; n::Int=100)
    vmin, vmax = speed_bounds(strategy)
    if vmin == vmax
        return true  # constant speed: trivially monotone
    end
    # infer density range for KDE-guided strategies
    dmin, dmax = _density_range(strategy)
    ds = range(dmin, dmax; length=n)
    speeds = [assign_speed(strategy, d) for d in ds]
    # allow for floating-point noise: check each step is ≥ (not strictly >)
    for i in 2:n
        speeds[i] > speeds[i-1] + sqrt(eps()) && return false
    end
    return true
end

_density_range(s::ConstantSpeed)        = (0.0, 1.0)
_density_range(s::KDEGuidedSpeed)       = (s.dmin, s.dmax)
_density_range(s::CurvatureGuidedSpeed) = (s.dmin, s.dmax)
