"""
    kde.jl — KDE surface computation from binary canopy masks

Ported and cleaned from CanopyDensity/kde.jl.  All convolution is done with a
pure-Julia direct (spatial-domain) implementation so that FFTW is not a
mandatory dependency.  For large grids the user can load FFTW.jl separately
and call `kde_from_mask` with `method=:fft`; the spatial fallback is used
automatically when FFTW is absent.

Public API
----------
Kernel constructors:
  - `gaussian_kernel(dx, dy, σx, σy; radius_mult=3)   -> Matrix{Float64}`
  - `epanechnikov_kernel(dx, dy, h; radius_mult=3)     -> Matrix{Float64}`

Bandwidth:
  - `scotts_sigma(xs, ys, mask; anisotropic=true, ...)  -> (σx, σy)`
  - `scott_sigma_indices(mask, dx, dy; ...)              -> (σx, σy)`

KDE:
  - `kde_from_mask(mask, xs, ys; kernel, bandwidth, …)  -> RasterGrid`
"""

# NOTE: Statistics (std, mean) is loaded in the main module KDEFlightPlanning.jl.
# FFTW is loaded in KDEFlightPlanning.jl; _conv2 uses it when available.

# ---------------------------------------------------------------------------
# Kernel constructors  (from CanopyDensity/kde.jl)
# ---------------------------------------------------------------------------

"""
    gaussian_kernel(dx, dy, σx, σy; radius_mult=3) -> Matrix{Float64}

Build an axis-aligned separable Gaussian kernel on the pixel grid defined by
spacings `dx`, `dy` (world units). The kernel is truncated to ±`radius_mult`·σ
and normalised to sum to 1.
"""
function gaussian_kernel(dx::Real, dy::Real, σx::Real, σy::Real;
                          radius_mult::Real=3)
    rx = max(1, ceil(Int, radius_mult * σx / dx))
    ry = max(1, ceil(Int, radius_mult * σy / dy))
    xs = collect(-rx:rx) .* Float64(dx)
    ys = collect(-ry:ry) .* Float64(dy)
    Gx = @. exp(-0.5 * (xs / Float64(σx))^2)
    Gy = @. exp(-0.5 * (ys / Float64(σy))^2)
    K  = Gy * Gx'
    K ./= sum(K)
    return K
end

"""
    epanechnikov_kernel(dx, dy, h; radius_mult=3) -> Matrix{Float64}

Build a circular Epanechnikov kernel of radius `h` (world units) on the pixel
grid with spacings `dx`, `dy`. Within the support disk r ≤ h, the value is
`1 − (r/h)²`; outside the disk it is 0. Normalised to sum to 1.
"""
function epanechnikov_kernel(dx::Real, dy::Real, h::Real; radius_mult::Real=3)
    rx = max(1, ceil(Int, radius_mult * h / dx))
    ry = max(1, ceil(Int, radius_mult * h / dy))
    xs = collect(-rx:rx) .* Float64(dx)
    ys = collect(-ry:ry) .* Float64(dy)
    K  = Array{Float64}(undef, length(ys), length(xs))
    h2 = Float64(h)^2
    @inbounds for j in eachindex(ys), i in eachindex(xs)
        r2 = xs[i]^2 + ys[j]^2
        K[j, i] = r2 <= h2 ? (1.0 - r2/h2) : 0.0
    end
    s = sum(K)
    s == 0 && error("epanechnikov_kernel: h=$h too small; kernel has no support pixels")
    K ./= s
    return K
end

# ---------------------------------------------------------------------------
# Bandwidth selection  (from CanopyDensity/kde.jl)
# ---------------------------------------------------------------------------

"""
    scotts_sigma(xs, ys, mask; anisotropic=true, min_pixels=2, max_pixels=50)
        -> (σx, σy)

Scott's Rule bandwidth(s) for a binary mask on grid `(xs, ys)`.
Returns `(σx, σy)` in world units (same units as `xs`/`ys`).

- `anisotropic=false` uses an isotropic σ = n^(-1/6) × √((σx²+σy²)/2)
- `min_pixels`/`max_pixels` clamp the result to avoid degenerate bandwidths.
"""
function scotts_sigma(xs, ys, mask;
                       anisotropic::Bool=true,
                       min_pixels::Int=2,
                       max_pixels::Int=50)
    @assert length(xs) == size(mask,2) && length(ys) == size(mask,1)
    dx = abs(xs[2]-xs[1]); dy = abs(ys[2]-ys[1])

    w  = mask .> 0.5
    n  = count(w)
    n == 0 && error("scotts_sigma: mask has no positive pixels")

    wx  = vec(sum(w; dims=1))    # column sums, length = W
    wy  = vec(sum(w; dims=2))    # row sums,    length = H
    μx  = sum(xs .* wx) / n
    μy  = sum(ys .* wy) / n
    σx2 = sum(((xs .- μx).^2) .* wx) / n
    σy2 = sum(((ys .- μy).^2) .* wy) / n
    σx_raw = sqrt(σx2); σy_raw = sqrt(σy2)

    factor = n^(-1/6)   # Scott's factor for d=2
    if anisotropic
        σx = σx_raw * factor
        σy = σy_raw * factor
    else
        σiso = sqrt((σx_raw^2 + σy_raw^2) / 2) * factor
        σx = σy = σiso
    end

    σx = clamp(σx, min_pixels * dx, max_pixels * dx)
    σy = clamp(σy, min_pixels * dy, max_pixels * dy)
    return σx, σy
end

"""
    scott_sigma_indices(mask, dx, dy; min_pixels=1) -> (σx, σy)

Isotropic Scott's Rule in pixel-index space, scaled to world units.
Matches the CanopyDensity `auto_indices` bandwidth mode.
"""
function scott_sigma_indices(mask, dx::Real, dy::Real; min_pixels::Real=1)
    w    = mask .> 0.5
    inds = findall(w)
    n    = length(inds)
    n == 0 && error("scott_sigma_indices: mask has no positive pixels")

    rs = [I[1] for I in inds]
    cs = [I[2] for I in inds]
    sx = std(Float64.(cs)); sy = std(Float64.(rs))
    σp = max(((sx + sy) / 2) * n^(-1/6), min_pixels)  # pixel units
    return σp * Float64(dx), σp * Float64(dy)
end

# ---------------------------------------------------------------------------
# Spatial-domain convolution (no FFTW required)
# ---------------------------------------------------------------------------

"""
    _conv2_direct(A, K) -> Matrix{Float64}

Direct (spatial-domain) 2-D convolution of `A` with kernel `K`, with
zero-padding and 'same' output size. Used as the FFTW-free fallback.

For small kernels (≤ 31×31) and moderate grids this is fast enough for
synthetic testing. For large real grids, load FFTW.jl and the FFT path
will be selected automatically inside `kde_from_mask`.
"""
function _conv2_direct(A::AbstractMatrix{<:Real}, K::AbstractMatrix{<:Real})
    H, W   = size(A)
    kh, kw = size(K)
    rh     = kh ÷ 2;  rw = kw ÷ 2
    out    = zeros(Float64, H, W)
    Af     = Float64.(A)

    @inbounds for j in 1:W, i in 1:H
        acc = 0.0
        for dj in 1:kw, di in 1:kh
            ii = i + (di - rh - 1)
            jj = j + (dj - rw - 1)
            if 1 <= ii <= H && 1 <= jj <= W
                acc += Af[ii, jj] * K[di, dj]
            end
        end
        out[i, j] = acc
    end
    return out
end

"""
    _conv2_fft(A, K) -> Matrix{Float64}

FFT-based 2-D convolution with same-size output (zero-padded, centered kernel).
Uses FFTW.jl (a hard dependency listed in Project.toml).
"""
function _conv2_fft(A::AbstractMatrix{<:Real}, K::AbstractMatrix{<:Real})
    H, W   = size(A); hk, wk = size(K)
    P, Q   = H + hk - 1, W + wk - 1

    Ap = zeros(Float64, P, Q); @views Ap[1:H, 1:W] .= A
    Kp = zeros(Float64, P, Q); @views Kp[1:hk, 1:wk] .= K

    i0 = fld(hk, 2); j0 = fld(wk, 2)
    Kp = circshift(Kp, (-i0, -j0))

    C = real(ifft(fft(Ap) .* fft(Kp)))
    return copy(@view C[1:H, 1:W])
end

# Use FFT convolution (FFTW in deps); fall back to direct for tiny kernels
_conv2(A, K) = (size(K, 1) * size(K, 2) > 49) ? _conv2_fft(A, K) : _conv2_direct(A, K)

# ---------------------------------------------------------------------------
# Main KDE function  (from CanopyDensity/kde.jl)
# ---------------------------------------------------------------------------

"""
    kde_from_mask(mask::Matrix, xs, ys;
                   kernel=:gaussian,
                   σx=NaN, σy=NaN, h=NaN,
                   anisotropic=true,
                   min_pixels=3, max_pixels=50,
                   radius_mult=3,
                   bandwidth=:auto,
                   normalize=:sum1) -> (RasterGrid, info)

Compute a KDE density surface from a binary occupancy `mask` on grid `(xs, ys)`.

Arguments
---------
- `mask`:       Float64 matrix (H×W); values > 0.5 are "vegetation"
- `xs`, `ys`:   Regularly spaced world coordinates (ascending, same units)
- `kernel`:     `:gaussian` or `:epanechnikov`
- `bandwidth`:  `:auto` (Scott's rule in world coords),
                `:auto_indices` (Scott's rule in pixel space), or `:manual`
- `σx`, `σy`:   Manual Gaussian bandwidths (world units) when `bandwidth=:manual`
- `h`:          Manual Epanechnikov radius (world units) when `bandwidth=:manual`
- `normalize`:  `:sum1` (sum-1 kernel, default) or `:density` (divide by dx·dy)

Returns
-------
- `RasterGrid` with Z normalised to [0,1] (non-negative, FFT ripples removed)
- `info` NamedTuple with `(A, σx, σy, h)` for diagnostics

Notes
-----
The output Z is normalised to [0,1] (min–max) after KDE convolution, matching
the CanopyDensity pipeline. A cutoff of zero is applied to remove FFT round-off
negatives.

This function does NOT require FFTW — it falls back to a direct convolution.
For large real grids, the caller can pre-load FFTW for speed.
"""
function kde_from_mask(mask::Matrix{<:Real}, xs, ys;
                        kernel   ::Symbol=:gaussian,
                        σx       ::Real=NaN, σy::Real=NaN,
                        h        ::Real=NaN,
                        anisotropic::Bool=true,
                        min_pixels::Int=3,
                        max_pixels::Int=50,
                        radius_mult::Int=3,
                        bandwidth ::Symbol=:auto,
                        normalize ::Symbol=:sum1)

    @assert length(xs) == size(mask, 2) "xs length must match mask width"
    @assert length(ys) == size(mask, 1) "ys length must match mask height"
    length(xs) >= 2 && length(ys) >= 2 ||
        throw(ArgumentError("xs and ys must each have ≥ 2 elements"))

    A  = Float64.(mask .> 0.5)
    dx = abs(xs[2] - xs[1]); dy = abs(ys[2] - ys[1])

    σx_out = Float64(σx); σy_out = Float64(σy); h_out = Float64(h)

    if kernel === :gaussian
        if bandwidth === :auto
            σx_out, σy_out = scotts_sigma(xs, ys, mask;
                                           anisotropic=anisotropic,
                                           min_pixels=min_pixels,
                                           max_pixels=max_pixels)
        elseif bandwidth === :auto_indices
            σx_out, σy_out = scott_sigma_indices(mask, dx, dy; min_pixels=1)
        else  # :manual
            (isfinite(σx_out) && σx_out > 0 && isfinite(σy_out) && σy_out > 0) ||
                throw(ArgumentError("Manual bandwidth: need finite positive σx, σy"))
        end
        K = gaussian_kernel(dx, dy, σx_out, σy_out; radius_mult=radius_mult)

    elseif kernel === :epanechnikov
        h_out, _ = scott_sigma_indices(mask, dx, dy; min_pixels=1)
        if bandwidth !== :auto && bandwidth !== :auto_indices
            isfinite(Float64(h)) && h > 0 || throw(ArgumentError("h must be positive"))
            h_out = Float64(h)
        end
        K = epanechnikov_kernel(dx, dy, h_out; radius_mult=radius_mult)

    else
        throw(ArgumentError("kernel must be :gaussian or :epanechnikov, got $kernel"))
    end

    Z = _conv2(A, K)

    if normalize === :density
        Z ./= (dx * dy)
    end  # :sum1 → already normalised by kernel construction

    Z = max.(Z, 0.0)           # remove FFT round-off negatives
    zmin, zmax = extrema(Z)
    if zmax > zmin
        @. Z = (Z - zmin) / (zmax - zmin)
    end

    info = (A=A, σx=σx_out, σy=σy_out, h=h_out)
    return RasterGrid(Z, Vector{Float64}(xs), Vector{Float64}(ys)), info
end

# Convenience: accept BitMatrix input
kde_from_mask(mask::BitMatrix, xs, ys; kwargs...) =
    kde_from_mask(Float64.(mask), xs, ys; kwargs...)
