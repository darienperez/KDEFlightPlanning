# colorspace.jl — RGB ↔ CIELAB conversion
#
# Ported from CanopyDensity/colorspace.jl.
# Depends on Colors.jl (listed in [deps] of Project.toml).
# All `using` statements are centralised in KDEFlightPlanning.jl;
# this file assumes Colors and ColorTypes are already in scope.

"""
    rgb_to_lab(img_rgb) -> (L, a, b)

Convert a 2-D colour image to three Float32 matrices (H×W) in CIELAB.

Supported input types:
- `Matrix{RGB{N0f8}}`       — 0–255 bytes packed (e.g. from FileIO / Images.jl)
- `Matrix{RGB{Float32}}`    — float channels in [0,1]
- `Matrix{RGB{Float64}}`    — float channels in [0,1]
- Any `AbstractMatrix{<:Colorant}` that Colors can convert to `Lab`

Returns `(L, a, b)` where each component is a `Matrix{Float32}`.

Usage in the pipeline:
```julia
img = rgb_from_array(raw_uint8_array)    # or load via FileIO
L, a, b = rgb_to_lab(img)
X = stack_features(L, a, b)
```
"""
function rgb_to_lab(img_rgb::AbstractMatrix{<:Colorant})
    H, W = size(img_rgb)
    L = Array{Float32}(undef, H, W)
    a = Array{Float32}(undef, H, W)
    b = Array{Float32}(undef, H, W)

    @inbounds for j in 1:W, i in 1:H
        c = convert(Lab, img_rgb[i, j])
        L[i, j] = Float32(float(c.l))
        a[i, j] = Float32(float(c.a))
        b[i, j] = Float32(float(c.b))
    end
    return L, a, b
end

"""
    is_grayscale(img_rgb; tol=1e-6) -> Bool

Return `true` if all pixels have R≈G≈B within `tol`.
Useful to skip colour features on panchromatic imagery.
"""
function is_grayscale(img_rgb::AbstractMatrix{<:Colorant}; tol::Real=1e-6)
    @inbounds for c in img_rgb
        rc = convert(RGB, c)
        r, g, bv = float(rc.r), float(rc.g), float(rc.b)
        if !(isapprox(r, g; atol=tol) && isapprox(g, bv; atol=tol))
            return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Array-based API (no external file I/O required)
# ---------------------------------------------------------------------------

"""
    rgb_from_array(arr::AbstractArray{<:Real,3}) -> Matrix{RGB{Float32}}

Build an `H×W` matrix of `RGB{Float32}` from a raw numeric array.

Accepted layouts:
- `(3, H, W)` — channel-first (common from PyTorch / GDAL band-interleaved)
- `(H, W, 3)` — channel-last  (common from Images.jl / PIL)

Pixel values are scaled so that:
- Integer arrays (UInt8, UInt16, …) are divided by `typemax(eltype(arr))`
  to map to [0, 1].
- Floating-point arrays are used as-is (assumed to be in [0, 1]).

Returns `Matrix{RGB{Float32}}` of size `(H, W)`.
"""
function rgb_from_array(arr::AbstractArray{<:Real,3})
    sz = size(arr)
    if sz[1] == 3
        # channel-first: (3, H, W)
        _, H, W = sz
        scale = _arr_scale(eltype(arr))
        img = Matrix{RGB{Float32}}(undef, H, W)
        @inbounds for j in 1:W, i in 1:H
            img[i, j] = RGB{Float32}(arr[1,i,j]*scale,
                                      arr[2,i,j]*scale,
                                      arr[3,i,j]*scale)
        end
        return img
    elseif sz[3] == 3
        # channel-last: (H, W, 3)
        H, W, _ = sz
        scale = _arr_scale(eltype(arr))
        img = Matrix{RGB{Float32}}(undef, H, W)
        @inbounds for j in 1:W, i in 1:H
            img[i, j] = RGB{Float32}(arr[i,j,1]*scale,
                                      arr[i,j,2]*scale,
                                      arr[i,j,3]*scale)
        end
        return img
    else
        throw(ArgumentError(
            "rgb_from_array: array must be (3,H,W) or (H,W,3), got size $sz"))
    end
end

# Separate out the UInt8 method to avoid repeated branching in the inner loop
function rgb_from_array(arr::AbstractArray{UInt8,3})
    sz = size(arr)
    if sz[1] == 3
        _, H, W = sz
        img = Matrix{RGB{Float32}}(undef, H, W)
        @inbounds for j in 1:W, i in 1:H
            img[i, j] = RGB{Float32}(arr[1,i,j]/255f0,
                                      arr[2,i,j]/255f0,
                                      arr[3,i,j]/255f0)
        end
        return img
    else
        H, W, _ = sz
        img = Matrix{RGB{Float32}}(undef, H, W)
        @inbounds for j in 1:W, i in 1:H
            img[i, j] = RGB{Float32}(arr[i,j,1]/255f0,
                                      arr[i,j,2]/255f0,
                                      arr[i,j,3]/255f0)
        end
        return img
    end
end

_arr_scale(::Type{T}) where {T<:Integer}       = Float32(1) / Float32(typemax(T))
_arr_scale(::Type{T}) where {T<:AbstractFloat} = Float32(1)

"""
    rgb_to_lab_array(arr::AbstractArray{<:Real,3}) -> (L, a, b)

Convenience: convert a raw numeric array directly to CIELAB channels.
Combines `rgb_from_array` + `rgb_to_lab` in one call.
"""
rgb_to_lab_array(arr::AbstractArray{<:Real,3}) = rgb_to_lab(rgb_from_array(arr))

"""
    lab_to_array(L, a, b) -> Array{Float32,3}  (shape H×W×3, channel-last)

Pack CIELAB channels back into a single array (useful for debugging/display).
"""
function lab_to_array(L::AbstractMatrix, a::AbstractMatrix, b::AbstractMatrix)
    H, W = size(L)
    out = Array{Float32}(undef, H, W, 3)
    @inbounds for j in 1:W, i in 1:H
        out[i, j, 1] = Float32(L[i, j])
        out[i, j, 2] = Float32(a[i, j])
        out[i, j, 3] = Float32(b[i, j])
    end
    return out
end
