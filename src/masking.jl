# masking.jl — Binary mask operations for canopy segmentation
#
# Ported from CanopyDensity/masking.jl.
# All `using` statements are centralised in KDEFlightPlanning.jl.
#
# Always-available (stdlib only):
#   labels_to_mask, mask_stats
#
# Morphology (require ImageMorphology.jl loaded externally; stubbed):
#   remove_small_components!, fill_small_holes!, binary_open_close!

# ---------------------------------------------------------------------------
# Always-available: label → mask
# ---------------------------------------------------------------------------

"""
    labels_to_mask(labels, H, W; tree_labels=[1]) -> BitMatrix

Reshape flat cluster `labels` (length H×W) into an H×W grid and mark pixels
whose label is in `tree_labels` as `true`. Returns a `BitMatrix`.

Arguments
---------
- `labels`:      Integer vector of length H*W (row-major: row index advances first)
- `H`, `W`:      Height and width of the original image
- `tree_labels`: Collection of cluster label values corresponding to vegetation

Example
-------
```julia
mask = labels_to_mask(km_labels, H, W; tree_labels=[1, 3])
```
"""
function labels_to_mask(labels::AbstractVector{<:Integer}, H::Integer, W::Integer;
                         tree_labels::Union{AbstractVector{<:Integer}, AbstractSet}=[1])
    length(labels) == H * W ||
        throw(DimensionMismatch("labels must have length H*W = $(H*W), got $(length(labels))"))
    labimg = reshape(labels, Int(H), Int(W))
    S = Set(tree_labels)
    return in.(labimg, Ref(S))   # BitMatrix H×W
end

# ---------------------------------------------------------------------------
# Always-available: mask statistics
# ---------------------------------------------------------------------------

"""
    mask_stats(mask) -> NamedTuple

Return basic statistics for a binary mask:
- `area`:  number of `true` pixels
- `frac`:  `area / length(mask)` (fraction of image covered)
- `bbox`:  `(imin, imax, jmin, jmax)` bounding box of `true` pixels (row, col)
"""
function mask_stats(mask::AbstractMatrix{Bool})
    H, W = size(mask)
    area = count(mask)
    frac = area / (H * W)
    imin = H; imax = 0; jmin = W; jmax = 0
    @inbounds for i in 1:H, j in 1:W
        if mask[i, j]
            i < imin && (imin = i); i > imax && (imax = i)
            j < jmin && (jmin = j); j > jmax && (jmax = j)
        end
    end
    bbox = area > 0 ? (imin, imax, jmin, jmax) : (0, 0, 0, 0)
    return (area=area, frac=frac, bbox=bbox)
end

# ---------------------------------------------------------------------------
# Morphology stubs (require ImageMorphology.jl loaded in the caller's session)
# ---------------------------------------------------------------------------

const _MORPHOLOGY_MSG = """
This function requires ImageMorphology.jl.
Load it in your session first:
    using Pkg; Pkg.add("ImageMorphology")
    using ImageMorphology
"""

"""
    remove_small_components!(mask::BitMatrix; min_pixels=100, connectivity=4) -> mask

Remove connected components with fewer than `min_pixels` pixels.
`connectivity` ∈ {4, 8}.

**Requires ImageMorphology.jl to be loaded in the caller's session.**
"""
function remove_small_components!(mask::BitMatrix;
                                   min_pixels::Int=100, connectivity::Int=4)
    if !isdefined(Main, :ImageMorphology)
        error("remove_small_components!: " * _MORPHOLOGY_MSG)
    end
    se  = connectivity == 4 ? Main.ImageMorphology.SEDiamond(1) :
          connectivity == 8 ? trues(3, 3) :
          error("connectivity must be 4 or 8")
    lbl  = Main.ImageMorphology.label_components(mask, se)
    lens = Main.ImageMorphology.component_lengths(lbl)
    @inbounds for i in eachindex(mask)
        lab = lbl[i]
        if lab != 0 && lens[lab] < min_pixels
            mask[i] = false
        end
    end
    return mask
end

"""
    fill_small_holes!(mask::BitMatrix; max_hole_pixels=100, connectivity=4) -> mask

Fill interior holes (false-islands surrounded by true) with area ≤ `max_hole_pixels`.

**Requires ImageMorphology.jl to be loaded in the caller's session.**
"""
function fill_small_holes!(mask::BitMatrix;
                            max_hole_pixels::Int=100, connectivity::Int=4)
    if !isdefined(Main, :ImageMorphology)
        error("fill_small_holes!: " * _MORPHOLOGY_MSG)
    end
    se  = connectivity == 4 ? Main.ImageMorphology.SEDiamond(1) :
          connectivity == 8 ? trues(3, 3) :
          error("connectivity must be 4 or 8")
    inv  = .!mask
    lbl  = Main.ImageMorphology.label_components(inv, se)
    lens = Main.ImageMorphology.component_lengths(lbl)
    H, W = size(mask)
    border_touch = falses(length(lens))
    @inbounds for j in 1:W
        l = lbl[1, j];  l != 0 && (border_touch[l] = true)
        l = lbl[H, j];  l != 0 && (border_touch[l] = true)
    end
    @inbounds for i in 1:H
        l = lbl[i, 1];  l != 0 && (border_touch[l] = true)
        l = lbl[i, W];  l != 0 && (border_touch[l] = true)
    end
    @inbounds for i in eachindex(inv)
        lab = lbl[i]
        if lab != 0 && !border_touch[lab] && lens[lab] <= max_hole_pixels
            inv[i] = false
        end
    end
    @. mask = .!inv
    return mask
end

"""
    binary_open_close!(mask::BitMatrix; open_radius=0, close_radius=0) -> mask

Apply morphological opening and/or closing with a square structuring element.

**Requires ImageMorphology.jl to be loaded in the caller's session.**
"""
function binary_open_close!(mask::BitMatrix; open_radius::Int=0, close_radius::Int=0)
    if !isdefined(Main, :ImageMorphology)
        error("binary_open_close!: " * _MORPHOLOGY_MSG)
    end
    if open_radius > 0
        se = ones(Bool, 2open_radius+1, 2open_radius+1)
        mask .= Main.ImageMorphology.opening(mask, se)
    end
    if close_radius > 0
        se = ones(Bool, 2close_radius+1, 2close_radius+1)
        mask .= Main.ImageMorphology.closing(mask, se)
    end
    return mask
end
