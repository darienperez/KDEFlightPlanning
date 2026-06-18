"""
    raster.jl — RasterGrid construction, sampling, gradient, and curvature utilities

Sources: FlightPlanning (bilinear, gradient, curvature, sample_line) and
         CanopyDensity (make_grid, rescale_density!, density_minmax).

Key API
-------
Grid construction/manipulation:
  - `make_grid(Z, xs, ys)`       — safe constructor; handles descending axes
  - `rescale_density!(grid)`     — affine rescale of Z to [0,1] in-place
  - `density_minmax(grid)`       — (dmin, dmax)
  - `synthetic_gaussian_grid(…)` — synthetic test grid from Gaussian mixture
  - `uniform_grid(value)`        — flat test grid

Point sampling:
  - `sample_density(grid, x, y; sampler)` — bilinear or nearest lookup
  - `bilinear_interp(grid, x, y)`         — bilinear interpolation

Spatial derivatives:
  - `gradient_at(grid, x, y)`             — ∇d(x,y)
  - `gradient_magnitude(grid, x, y)`      — ‖∇d‖
  - `directional_curvature(grid, x, y, ux, uy)` — d²d/ds²

Line sampling:
  - `sample_line(grid, x1, y1, x2, y2; n)` — uniform samples along segment
"""

# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

"""
    make_grid(Z, xs, ys) -> RasterGrid

Safe `RasterGrid` constructor. If `xs` or `ys` is descending, both the axis
and the corresponding dimension of `Z` are flipped so that the stored grid
is always ascending (north-up convention after flip). Ported from CanopyDensity.
"""
function make_grid(Z::AbstractMatrix, xs::AbstractVector, ys::AbstractVector)
    @assert size(Z, 2) == length(xs) "size(Z,2) must equal length(xs)"
    @assert size(Z, 1) == length(ys) "size(Z,1) must equal length(ys)"

    Zc  = Matrix{Float64}(Z)
    xsc = Vector{Float64}(xs)
    ysc = Vector{Float64}(ys)

    if first(xsc) > last(xsc)
        reverse!(xsc)
        Zc = reverse(Zc; dims=2)
    end
    if first(ysc) > last(ysc)
        reverse!(ysc)
        Zc = reverse(Zc; dims=1)
    end
    return RasterGrid(Zc, xsc, ysc)
end

"""
    density_minmax(grid::RasterGrid) -> (dmin, dmax)

Return the extrema of `grid.Z`. Ported from CanopyDensity/grids.jl.
"""
density_minmax(grid::RasterGrid) = (minimum(grid.Z), maximum(grid.Z))

"""
    rescale_density!(grid::RasterGrid; newmin=0.0, newmax=1.0) -> grid

Affine rescale `grid.Z` to `[newmin, newmax]` in-place. Anchors the min and
max cells exactly to avoid floating-point drift. No-op when the surface is
flat (dmin == dmax). Ported from CanopyDensity/grids.jl.
"""
function rescale_density!(grid::RasterGrid; newmin::Real=0.0, newmax::Real=1.0)
    Z = grid.Z
    dmin, dmax = extrema(Z)
    if dmax == dmin
        fill!(Z, (newmin + newmax) / 2)
    else
        r      = (newmax - newmin) / (dmax - dmin)
        offset = newmin - r * dmin
        imin   = argmin(Z)
        imax   = argmax(Z)
        @inbounds @simd for i in eachindex(Z)
            Z[i] = clamp(muladd(r, Z[i], offset), newmin, newmax)
        end
        Z[imin] = newmin
        Z[imax] = newmax
    end
    return grid
end

# ---------------------------------------------------------------------------
# Nearest-index helper (internal)
# ---------------------------------------------------------------------------

"""
    _nearest_index(vec, val) -> Int

Return the index of the element in sorted ascending `vec` closest to `val`.
"""
function _nearest_index(vec::AbstractVector{<:Real}, val::Real)
    idx = searchsortedfirst(vec, val)
    idx == firstindex(vec)  && return idx
    idx > lastindex(vec)    && return lastindex(vec)
    abs(val - vec[idx-1]) <= abs(vec[idx] - val) ? idx - 1 : idx
end

# ---------------------------------------------------------------------------
# Bilinear interpolation
# ---------------------------------------------------------------------------

"""
    bilinear_interp(grid::RasterGrid, x::Real, y::Real) -> Float64

Bilinear interpolation of `grid.Z` at world coordinates `(x, y)`.
Out-of-bounds coordinates are clamped to the grid extent.

Formula:
    Z ≈ (1-tx)(1-ty)·Z[j,i] + tx(1-ty)·Z[j,i+1]
      + (1-tx)ty·Z[j+1,i]  + tx·ty·Z[j+1,i+1]
"""
function bilinear_interp(grid::RasterGrid, x::Real, y::Real)
    xs, ys, Z = grid.xs, grid.ys, grid.Z

    x = clamp(Float64(x), first(xs), last(xs))
    y = clamp(Float64(y), first(ys), last(ys))

    ix1 = clamp(searchsortedlast(xs, x), firstindex(xs), lastindex(xs) - 1)
    ix2 = ix1 + 1
    iy1 = clamp(searchsortedlast(ys, y), firstindex(ys), lastindex(ys) - 1)
    iy2 = iy1 + 1

    dx = xs[ix2] - xs[ix1]; tx = dx == 0 ? 0.0 : (x - xs[ix1]) / dx
    dy = ys[iy2] - ys[iy1]; ty = dy == 0 ? 0.0 : (y - ys[iy1]) / dy

    z11 = Z[iy1, ix1]; z21 = Z[iy1, ix2]
    z12 = Z[iy2, ix1]; z22 = Z[iy2, ix2]

    return (1-tx)*(1-ty)*z11 + tx*(1-ty)*z21 + (1-tx)*ty*z12 + tx*ty*z22
end

# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------

"""
    sample_density(grid, x, y; sampler=:bilinear) -> Float64

Sample `grid.Z` at world coordinates `(x, y)`.

Sampler options:
- `:bilinear` (default) — smooth, suitable for speed mapping
- `:nearest`            — nearest cell, exact at cell centres
"""
function sample_density(grid::RasterGrid, x::Real, y::Real;
                         sampler::Symbol=:bilinear)
    if sampler === :bilinear
        return bilinear_interp(grid, x, y)
    elseif sampler === :nearest
        ix = _nearest_index(grid.xs, Float64(x))
        iy = _nearest_index(grid.ys, Float64(y))
        return Float64(grid.Z[iy, ix])
    else
        throw(ArgumentError("Unknown sampler=$sampler. Use :bilinear or :nearest."))
    end
end

# ---------------------------------------------------------------------------
# Gradient (finite differences)
# ---------------------------------------------------------------------------

"""
    _local_step(grid, x, y; h_factor=0.5) -> (hx, hy)

Return a finite-difference probe step sized as `h_factor` times the local
grid cell in each axis.
"""
function _local_step(grid::RasterGrid, x::Real, y::Real; h_factor::Real=0.5)
    xs, ys = grid.xs, grid.ys
    ix = _nearest_index(xs, Float64(x))
    iy = _nearest_index(ys, Float64(y))
    hx = (ix < lastindex(xs) ? xs[ix+1] - xs[ix] : xs[ix] - xs[ix-1])
    hy = (iy < lastindex(ys) ? ys[iy+1] - ys[iy] : ys[iy] - ys[iy-1])
    return max(eps(), h_factor * hx), max(eps(), h_factor * hy)
end

"""
    gradient_at(grid, x, y; sampler=:bilinear, h_factor=0.5) -> (gx, gy)

Estimate the density gradient ∇d(x,y) = (∂d/∂x, ∂d/∂y) using central finite
differences with a locally adaptive step size (~half a grid cell).
"""
function gradient_at(grid::RasterGrid, x::Real, y::Real;
                      sampler::Symbol=:bilinear, h_factor::Real=0.5)
    hx, hy = _local_step(grid, x, y; h_factor=h_factor)
    gx = (sample_density(grid, x+hx, y; sampler=sampler) -
          sample_density(grid, x-hx, y; sampler=sampler)) / (2hx)
    gy = (sample_density(grid, x, y+hy; sampler=sampler) -
          sample_density(grid, x, y-hy; sampler=sampler)) / (2hy)
    return gx, gy
end

"""
    gradient_magnitude(grid, x, y; sampler=:bilinear, h_factor=0.5) -> Float64

Return ‖∇d(x,y)‖ = hypot(∂d/∂x, ∂d/∂y).
"""
function gradient_magnitude(grid::RasterGrid, x::Real, y::Real;
                             sampler::Symbol=:bilinear, h_factor::Real=0.5)
    gx, gy = gradient_at(grid, x, y; sampler=sampler, h_factor=h_factor)
    return hypot(gx, gy)
end

# ---------------------------------------------------------------------------
# Directional curvature
# ---------------------------------------------------------------------------

"""
    directional_curvature(grid, x, y, ux, uy; sampler=:bilinear, h_factor=0.5) -> Float64

Directional second derivative of `grid.Z` at `(x, y)` along unit direction `(ux, uy)`:

    d²d/ds² ≈ [d(x+h·u) − 2d(x) + d(x−h·u)] / h²

Used in the curvature-spaced waypoint spacing formula:
    w = (|∇d|/g0)^α + λ·(|d²d/ds²|/k0)^η
"""
function directional_curvature(grid::RasterGrid, x::Real, y::Real,
                                ux::Real, uy::Real;
                                sampler::Symbol=:bilinear, h_factor::Real=0.5)
    hx, hy = _local_step(grid, x, y; h_factor=h_factor)
    h  = min(hx, hy)
    da = sample_density(grid, x + h*ux, y + h*uy; sampler=sampler)
    d0 = sample_density(grid, x,        y;         sampler=sampler)
    db = sample_density(grid, x - h*ux, y - h*uy; sampler=sampler)
    return (da - 2*d0 + db) / (h * h)
end

# ---------------------------------------------------------------------------
# Line sampling
# ---------------------------------------------------------------------------

"""
    sample_line(grid, x1, y1, x2, y2; n=200, sampler=:bilinear)
        -> (ts, densities)

Uniformly sample the density surface along the segment from `(x1,y1)` to
`(x2,y2)` at `n` equally-spaced parameter values t ∈ [0,1].

Returns:
- `ts`:        parameter values (length n)
- `densities`: sampled density values (length n)

Useful for single-flight-line coverage profiles.
"""
function sample_line(grid::RasterGrid,
                     x1::Real, y1::Real, x2::Real, y2::Real;
                     n::Int=200, sampler::Symbol=:bilinear)
    n >= 2 || throw(ArgumentError("n must be ≥ 2"))
    ts        = range(0.0, 1.0; length=n)
    densities = [sample_density(grid,
                                x1 + (x2-x1)*t,
                                y1 + (y2-y1)*t;
                                sampler=sampler) for t in ts]
    return collect(ts), densities
end

sample_line(grid::RasterGrid, p1, p2; kwargs...) =
    sample_line(grid, p1[1], p1[2], p2[1], p2[2]; kwargs...)

# ---------------------------------------------------------------------------
# Synthetic grid factories
# ---------------------------------------------------------------------------

"""
    synthetic_gaussian_grid(; nx, ny, xmin, xmax, ymin, ymax,
                              centers, sigmas, weights) -> RasterGrid

Create a `RasterGrid` from a mixture of 2-D Gaussians, normalised to [0,1].
Used for tests and examples when real GeoTIFFs are unavailable.

Example:
```julia
grid = synthetic_gaussian_grid(nx=50, ny=50, xmin=0.0, xmax=100.0,
                                ymin=0.0, ymax=100.0)
```
"""
function synthetic_gaussian_grid(;
    nx::Int=50, ny::Int=50,
    xmin::Real=0.0, xmax::Real=100.0,
    ymin::Real=0.0, ymax::Real=100.0,
    centers::AbstractVector=[(50.0, 50.0)],
    sigmas ::AbstractVector=[(15.0, 15.0)],
    weights::AbstractVector=[1.0])

    xs = collect(range(Float64(xmin), Float64(xmax); length=nx))
    ys = collect(range(Float64(ymin), Float64(ymax); length=ny))
    Z  = zeros(Float64, ny, nx)

    for (c, σ, w) in zip(centers, sigmas, weights)
        cx, cy = Float64(c[1]), Float64(c[2])
        sx, sy = Float64(σ[1]), Float64(σ[2])
        for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
            Z[j, i] += w * exp(-0.5 * ((x-cx)/sx)^2 - 0.5 * ((y-cy)/sy)^2)
        end
    end

    zmin, zmax = extrema(Z)
    if zmax > zmin
        Z .= (Z .- zmin) ./ (zmax - zmin)
    end

    return RasterGrid(Z, xs, ys)
end

"""
    uniform_grid(value=0.5; nx=20, ny=20, xmin=0.0, xmax=100.0,
                  ymin=0.0, ymax=100.0) -> RasterGrid

Create a flat `RasterGrid` with constant density `value`.
Useful for testing constant-speed baselines.
"""
function uniform_grid(value::Real=0.5;
                       nx::Int=20, ny::Int=20,
                       xmin::Real=0.0, xmax::Real=100.0,
                       ymin::Real=0.0, ymax::Real=100.0)
    xs = collect(range(Float64(xmin), Float64(xmax); length=nx))
    ys = collect(range(Float64(ymin), Float64(ymax); length=ny))
    Z  = fill(Float64(value), ny, nx)
    return RasterGrid(Z, xs, ys)
end

"""
    synthetic_mask_grid(; nx=50, ny=50, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0,
                         frac=0.4, seed=42) -> RasterGrid

Create a binary mask `RasterGrid` (values 0.0 or 1.0) with approximately
`frac` fraction of cells set to 1 (vegetation). Used to test KDE functions
without real imagery.
"""
function synthetic_mask_grid(;
    nx::Int=50, ny::Int=50,
    xmin::Real=0.0, xmax::Real=100.0,
    ymin::Real=0.0, ymax::Real=100.0,
    frac::Real=0.4, seed::Int=42)

    xs = collect(range(Float64(xmin), Float64(xmax); length=nx))
    ys = collect(range(Float64(ymin), Float64(ymax); length=ny))

    # Deterministic pattern: a centred elliptical blob
    Z = zeros(Float64, ny, nx)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    rx, ry = (xmax - xmin) * sqrt(frac), (ymax - ymin) * sqrt(frac)
    for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        if ((x - cx) / rx)^2 + ((y - cy) / ry)^2 <= 0.25
            Z[j, i] = 1.0
        end
    end
    return RasterGrid(Z, xs, ys)
end
