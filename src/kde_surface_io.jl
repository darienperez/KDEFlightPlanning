"""
    kde_surface_io.jl — True KDE surface: types, export/import, and geotransform-aware resampling

## Design

This module is written to idiomatic Julia style:

- **`GeoTransform`** is a concrete immutable struct (not a bare `Vector`) carrying
  the six GDAL coefficients with field names. It supports `getindex` for backward
  compatibility with positional access, but field names (`x_origin`, `dx`, …) are
  the primary API. Conversion from `AbstractVector` is via an explicit constructor.

- **`KDESurface`** is an immutable struct that bundles the density matrix with its
  `GeoTransform` and CRS string. All downstream functions are dispatched on
  `KDESurface`, so user code never needs to track (Z, gt, crs) as three separate
  variables.

- **`ResampleMethod`** is an abstract type with two concrete subtypes —
  `NearestNeighbor` and `Bilinear` — enabling multiple dispatch on the
  interpolation strategy rather than a `method::Symbol` flag.

- The resampling function `resample_to_count_grid` is overloaded on
  `ResampleMethod` subtypes; adding a new method (e.g. cubic) requires only
  a new subtype and a three-line dispatch method.

- JLD2 `save`/`load` are thin wrappers that serialise/deserialise `KDESurface`
  directly; field names in the file match struct fields for transparency.

## Geotransform orientation contract

GDAL 6-element convention (1-indexed Julia, matching GDAL C order):
    gt[1] = x_origin   — upper-left corner easting (metres)
    gt[2] = dx         — x pixel size (positive, metres/pixel)
    gt[3] = x_rot      — x rotation (0 for north-up)
    gt[4] = y_origin   — upper-left corner northing (metres)
    gt[5] = y_rot      — y rotation (0 for north-up)
    gt[6] = dy         — y pixel size (negative for north-up, metres/pixel)

Rows index **northing / y**; columns index **easting / x**.
Row 1 is the northernmost row (dy < 0 for north-up rasters).

## Durham NH study-site defaults

    GT_NATIVE    ≈ [341300.35, 0.028369..., 0, 4774892.895, 0, -0.028368...]
    GT_COUNTGRID =  [341300.35, 1.0, 0, 4774892.895, 0, -1.0]
    CRS_DURHAM   = "EPSG:6348"   (NAD83(2011)/UTM zone 19N)

## Public API

    # Types
    GeoTransform(x_origin, dx, x_rot, y_origin, y_rot, dy)
    GeoTransform(v::AbstractVector)          # construct from length-6 vector
    KDESurface(Z, geotransform, crs; notes)  # bundle surface + metadata

    # Resample method tokens (dispatch-friendly)
    NearestNeighbor()
    Bilinear()

    # I/O
    save_kde_surface(surf, path)             # JLD2
    load_kde_surface(path)  -> KDESurface   # JLD2
    save_kde_surface_csv(surf, path)         # CSV + JSON sidecar
    load_kde_surface_csv(path) -> KDESurface

    # Resampling
    resample_to_count_grid(surf; H, W, gt_out, method)  -> Matrix{Float64}
    resample_to_count_grid(Z, gt_in; H, W, gt_out, method)  # convenience

    # Validation
    validate_kde_range(Z)                    # throws ArgumentError if outside [0,1]

## User export snippet (run locally)

```julia
# You have:
#   true_kde_surface :: Matrix{Float64}   # shape (9279, 11421), normalised [0,1]
#   true_gt          :: Vector{Float64}   # 6-element GDAL geotransform (or GT_NATIVE)
#   true_crs = "EPSG:6348"               # NAD83(2011)/UTM zone 19N

using Pkg; Pkg.activate("path/to/KDEFlightPlanning.jl")
using KDEFlightPlanning

# 1. Wrap in a KDESurface (validates range, stores metadata)
surf = KDESurface(true_kde_surface, GeoTransform(true_gt), true_crs;
                  notes = "Epanechnikov KDE, Silverman BW, 2026 run")

# 2. Save to JLD2 (lossless)
save_kde_surface(surf, "output/orthomosaic_kde/kde_surface_true.jld2")

# 3. Resample to 263×324 count grid (nearest-neighbour, default)
Z_263 = resample_to_count_grid(surf)         # → Matrix{Float64}(263, 324)

# Or bilinear:
Z_263b = resample_to_count_grid(surf; method=Bilinear())

# 4. Assign KDE density classes
classes, tinfo = assign_kde_density_classes(Z_263)
println(kde_class_narrative(tinfo))
```
"""

using JLD2

# ===========================================================================
# Resample method tokens — dispatch-first design
# ===========================================================================

"""
    ResampleMethod

Abstract supertype for KDE surface resampling strategies.
Concrete subtypes: `NearestNeighbor`, `Bilinear`.

Use these as values, not symbols:
    resample_to_count_grid(surf; method = NearestNeighbor())   # default
    resample_to_count_grid(surf; method = Bilinear())
"""
abstract type ResampleMethod end

"""
    NearestNeighbor <: ResampleMethod

Token for nearest-neighbour resampling. Default strategy for
`resample_to_count_grid`.
"""
struct NearestNeighbor <: ResampleMethod end

"""
    Bilinear <: ResampleMethod

Token for bilinear resampling. Use when smoothing across native pixels
is acceptable (sub-pixel precision at count-grid scale).
"""
struct Bilinear <: ResampleMethod end

# ===========================================================================
# GeoTransform — typed wrapper for GDAL 6-element affine transform
# ===========================================================================

"""
    GeoTransform

Immutable struct encoding a GDAL 6-element affine geotransform.

## Fields (1-indexed, GDAL order)

| Field      | GDAL index | Description                              |
|------------|-----------|------------------------------------------|
| `x_origin` | 1         | Upper-left corner easting (m)            |
| `dx`       | 2         | X pixel size (m/pixel, positive)         |
| `x_rot`    | 3         | X rotation (0 for north-up)              |
| `y_origin` | 4         | Upper-left corner northing (m)           |
| `y_rot`    | 5         | Y rotation (0 for north-up)              |
| `dy`       | 6         | Y pixel size (m/pixel, **negative** for  |
|            |           |   north-up rasters)                      |

## Constructors

    GeoTransform(x_origin, dx, x_rot, y_origin, y_rot, dy)
    GeoTransform(v::AbstractVector)   # from any length-6 numeric vector

## Orientation contract

- Rows index the y/northing direction. Row 1 is **north** (dy < 0).
- Columns index the x/easting direction. Column 1 is **west** (dx > 0).
- North-up: `x_rot == y_rot == 0`, `dy < 0`.

## Interop

`getindex(gt, i)` is defined so a `GeoTransform` can be passed anywhere a
1-indexed 6-element numeric container is expected (e.g., `axes_from_geotransform`).

## Study-site defaults

    const GT_NATIVE = GeoTransform(341300.35, 0.028369669906313746, 0.0,
                                   4774892.895, 0.0, -0.02836868197004068)
    const GT_COUNTGRID = GeoTransform(341300.35, 1.0, 0.0, 4774892.895, 0.0, -1.0)
"""
struct GeoTransform
    x_origin :: Float64   # gt[1]
    dx       :: Float64   # gt[2]
    x_rot    :: Float64   # gt[3]
    y_origin :: Float64   # gt[4]
    y_rot    :: Float64   # gt[5]
    dy       :: Float64   # gt[6]
end

"""
    GeoTransform(v::AbstractVector) -> GeoTransform

Construct from any length-6 numeric vector (e.g. from GDAL, ArchGDAL, CSV).
Throws `ArgumentError` if `length(v) < 6`.
"""
function GeoTransform(v::AbstractVector)
    length(v) >= 6 ||
        throw(ArgumentError(
            "GeoTransform requires a vector of length ≥ 6 (GDAL convention). " *
            "Got length=$(length(v))."))
    GeoTransform(Float64(v[1]), Float64(v[2]), Float64(v[3]),
                 Float64(v[4]), Float64(v[5]), Float64(v[6]))
end

# Positional indexing so GeoTransform is a drop-in for AbstractVector in
# functions that use gt[1]..gt[6].
Base.getindex(gt::GeoTransform, i::Integer) = (gt.x_origin, gt.dx, gt.x_rot,
                                                gt.y_origin, gt.y_rot, gt.dy)[i]
Base.length(::GeoTransform) = 6
Base.iterate(gt::GeoTransform, s=1) = s > 6 ? nothing : (gt[s], s+1)

function Base.show(io::IO, gt::GeoTransform)
    @printf(io, "GeoTransform(x₀=%.2f, dx=%.6f, y₀=%.2f, dy=%.6f)",
            gt.x_origin, gt.dx, gt.y_origin, gt.dy)
end

# Study-site defaults (Durham NH, EPSG:6348)
"""Durham NH native-resolution geotransform (≈ 0.02837 m/px, NAD83(2011)/UTM19N)."""
const GT_NATIVE    = GeoTransform(341300.35, 0.028369669906313746, 0.0,
                                  4774892.895, 0.0, -0.02836868197004068)

"""Durham NH count-grid geotransform (1 m/cell, 263×324 grid)."""
const GT_COUNTGRID = GeoTransform(341300.35, 1.0, 0.0, 4774892.895, 0.0, -1.0)

"""CRS for the Durham NH study site."""
const CRS_DURHAM = "EPSG:6348"

# ===========================================================================
# KDESurface — bundled surface + metadata
# ===========================================================================

"""
    KDESurface

Immutable struct bundling a normalised KDE density matrix with its spatial
metadata (geotransform and CRS string).

## Fields

| Field         | Type          | Description                               |
|---------------|---------------|-------------------------------------------|
| `Z`           | Matrix{Float64} | Density values, validated in [0,1]      |
| `geotransform`| `GeoTransform`  | GDAL-convention spatial reference        |
| `crs`         | String          | CRS identifier, e.g. `"EPSG:6348"`      |
| `notes`       | String          | Free-form provenance string              |

## Constructor

    KDESurface(Z, geotransform, crs; notes="")

Validates that all values in `Z` are in [0, 1] (± 1e-9 tolerance).
Throws `ArgumentError` on violation — no silent renormalisation.

## Usage

```julia
surf = KDESurface(Z_native, GT_NATIVE, CRS_DURHAM;
                  notes = "Epanechnikov KDE, Silverman BW")

# Query
size(surf)          # → (9279, 11421)
surf.geotransform   # → GeoTransform(...)

# Resample to count grid
Z_out = resample_to_count_grid(surf)         # NearestNeighbor (default)
Z_out = resample_to_count_grid(surf; method=Bilinear())
```
"""
struct KDESurface
    Z            :: Matrix{Float64}
    geotransform :: GeoTransform
    crs          :: String
    notes        :: String

    function KDESurface(Z::Matrix{Float64}, gt::GeoTransform, crs::AbstractString;
                        notes::AbstractString = "")
        validate_kde_range(Z)
        new(Z, gt, String(crs), String(notes))
    end
end

# Convenience constructor accepting a raw vector geotransform
"""
    KDESurface(Z, gt::AbstractVector, crs; notes="") -> KDESurface

Convenience constructor: converts `gt` to `GeoTransform` before storing.
"""
function KDESurface(Z::Matrix{Float64}, gt::AbstractVector, crs::AbstractString;
                    notes::AbstractString = "")
    KDESurface(Z, GeoTransform(gt), crs; notes=notes)
end

Base.size(s::KDESurface)         = size(s.Z)
Base.size(s::KDESurface, d::Int) = size(s.Z, d)

function Base.show(io::IO, s::KDESurface)
    H, W = size(s)
    zmin, zmax = extrema(s.Z)
    @printf(io, "KDESurface(%d×%d  z∈[%.4f,%.4f]  crs=%s)",
            H, W, zmin, zmax, s.crs)
end

# ===========================================================================
# Validation
# ===========================================================================

"""
    validate_kde_range(Z::AbstractMatrix; context="") -> nothing

Throw `ArgumentError` if any value in `Z` is outside [-1e-9, 1+1e-9].
Call this before saving or resampling to guard against un-normalised surfaces.

To normalise explicitly:
    Z = (Z .- minimum(Z)) ./ (maximum(Z) - minimum(Z))
"""
function validate_kde_range(Z::AbstractMatrix; context::AbstractString = "")
    zmin, zmax = extrema(Z)
    ctx = isempty(context) ? "" : "[$context] "
    zmin >= -1e-9 ||
        throw(ArgumentError(
            "$(ctx)Z has negative values (min=$(round(zmin; digits=8))). " *
            "Surface must be normalised to [0,1]. " *
            "To normalise: Z = (Z .- minimum(Z)) ./ (maximum(Z) - minimum(Z))"))
    zmax <= 1.0 + 1e-9 ||
        throw(ArgumentError(
            "$(ctx)Z has values > 1 (max=$(round(zmax; digits=8))). " *
            "Surface must be normalised to [0,1]. " *
            "To normalise: Z = (Z .- minimum(Z)) ./ (maximum(Z) - minimum(Z))"))
    return nothing
end

# ===========================================================================
# JLD2 I/O — dispatched on KDESurface
# ===========================================================================

"""
    save_kde_surface(surf::KDESurface, path::AbstractString) -> String

Save a `KDESurface` to a JLD2 file. Returns the path.

## File fields

| JLD2 key       | Julia value                         |
|----------------|-------------------------------------|
| `kde_surface`  | `surf.Z`  (Matrix{Float64})         |
| `geotransform` | 6-element Vector{Float64}           |
| `crs`          | `surf.crs`  (String)                |
| `normalized`   | `true`  (Bool)                      |
| `notes`        | `surf.notes`  (String)              |
| `shape_H`      | `size(surf,1)`  (Int)               |
| `shape_W`      | `size(surf,2)`  (Int)               |
| `exported_at`  | ISO-8601 timestamp  (String)        |

## Example
```julia
save_kde_surface(surf, "output/orthomosaic_kde/kde_surface_true.jld2")
```
"""
function save_kde_surface(surf::KDESurface, path::AbstractString)::String
    mkpath(dirname(abspath(path)))
    H, W = size(surf)
    ts   = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    gt6  = [surf.geotransform.x_origin, surf.geotransform.dx, surf.geotransform.x_rot,
            surf.geotransform.y_origin, surf.geotransform.y_rot, surf.geotransform.dy]

    JLD2.jldsave(path;
        kde_surface  = surf.Z,
        geotransform = gt6,
        crs          = surf.crs,
        normalized   = true,
        notes        = surf.notes,
        shape_H      = H,
        shape_W      = W,
        exported_at  = ts,
    )

    zmin, zmax = extrema(surf.Z)
    @info "save_kde_surface → $path  shape=($H×$W)  " *
          "range=[$(round(zmin;digits=6)), $(round(zmax;digits=6))]  crs=$(surf.crs)"
    return path
end

"""
    load_kde_surface(path::AbstractString) -> KDESurface

Load a `KDESurface` from a JLD2 file produced by `save_kde_surface`.
Validates the density range on load.

## Example
```julia
surf = load_kde_surface("output/orthomosaic_kde/kde_surface_true.jld2")
println(surf)  # KDESurface(9279×11421  z∈[0.0,1.0]  crs=EPSG:6348)
```
"""
function load_kde_surface(path::AbstractString)::KDESurface
    isfile(path) || throw(ArgumentError("JLD2 file not found: $path"))
    data = JLD2.load(path)

    for k in ("kde_surface", "geotransform", "crs")
        haskey(data, k) ||
            throw(ArgumentError("JLD2 file '$path' is missing required field '$k'."))
    end

    Z   = Matrix{Float64}(data["kde_surface"])
    gt  = GeoTransform(data["geotransform"])
    crs = String(data["crs"])
    notes = get(data, "notes", "")

    return KDESurface(Z, gt, crs; notes=notes)   # validates range inside constructor
end

# ===========================================================================
# CSV + JSON I/O (always available, no extra dependencies)
# ===========================================================================

"""
    save_kde_surface_csv(surf::KDESurface, path::AbstractString) -> (csv_path, json_path)

Save a `KDESurface` as a row-major CSV with a JSON sidecar.
This backend has no optional dependencies. Comment lines start with `#` and are
skipped by `load_kde_surface_csv`.

Intended for count-grid-resolution (263×324) surfaces. For native-resolution
surfaces, prefer `save_kde_surface` (JLD2 is ~30× smaller and lossless at
full Float64 precision).

## Example
```julia
Z_263 = resample_to_count_grid(surf)
surf_small = KDESurface(Z_263, GT_COUNTGRID, CRS_DURHAM)
save_kde_surface_csv(surf_small, "output/orthomosaic_kde/kde_surface_true.csv")
```
"""
function save_kde_surface_csv(surf::KDESurface, path::AbstractString)::Tuple{String,String}
    H, W = size(surf)
    ts   = Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS")
    gt   = surf.geotransform
    gt6  = [gt.x_origin, gt.dx, gt.x_rot, gt.y_origin, gt.y_rot, gt.dy]

    mkpath(dirname(abspath(path)))

    # ── CSV ──────────────────────────────────────────────────────────────────
    open(path, "w") do io
        println(io, "# kde_surface_true")
        println(io, "# shape: H=$H W=$W")
        println(io, "# crs: $(surf.crs)")
        println(io, "# geotransform: $gt6")
        println(io, "# normalized: true")
        println(io, "# exported_at: $ts")
        isempty(surf.notes) || println(io, "# notes: $(surf.notes)")
        for i in 1:H
            println(io, join(round.(surf.Z[i, :]; digits=9), ","))
        end
    end

    # ── JSON sidecar ─────────────────────────────────────────────────────────
    json_path = replace(path, r"\.\w+$" => "_meta.json")
    meta = Dict{String,Any}(
        "kde_surface_shape" => [H, W],
        "geotransform"      => gt6,
        "crs"               => surf.crs,
        "normalized"        => true,
        "notes"             => surf.notes,
        "exported_at"       => ts,
        "csv_path"          => basename(path),
        "format"            => "row_major_csv_float64",
    )
    open(json_path, "w") do io; JSON.print(io, meta, 2); end

    zmin, zmax = extrema(surf.Z)
    @info "save_kde_surface_csv → $path  shape=($H×$W)  " *
          "range=[$(round(zmin;digits=6)), $(round(zmax;digits=6))]  crs=$(surf.crs)"
    return (path, json_path)
end

"""
    load_kde_surface_csv(path::AbstractString) -> KDESurface

Load a `KDESurface` from a CSV file produced by `save_kde_surface_csv`.

Comment lines beginning with `#` are parsed for metadata. Falls back to
`GT_COUNTGRID` and `CRS_DURHAM` when no geotransform / CRS is embedded.
"""
function load_kde_surface_csv(path::AbstractString)::KDESurface
    isfile(path) || throw(ArgumentError("CSV file not found: $path"))

    rows = Vector{Float64}[]
    meta = Dict{String,String}()

    open(path, "r") do io
        for line in eachline(io)
            s = strip(line)
            if startswith(s, "#")
                # Accept both strict word keys (e.g. # geotransform: ...)
                # and descriptive keys (e.g. # Geotransform (screenshot): ...).
                # The broader regex captures the key up to the first ':'.
                m = match(r"^#\s*([^:]+?)\s*:\s*(.+)$", s)
                if m !== nothing
                    raw_key = lowercase(strip(m[1]))
                    # Normalise common key variants
                    key = if occursin("geotransform", raw_key)
                        "geotransform"
                    elseif raw_key == "crs" || raw_key == "epsg" || raw_key == "coordinate_system"
                        "crs"
                    elseif raw_key == "notes" || raw_key == "note"
                        "notes"
                    else
                        # Store under the raw lowercase key for unknown fields
                        replace(raw_key, r"[^a-z0-9_]" => "_")
                    end
                    meta[key] = strip(m[2])
                end
            elseif !isempty(s)
                push!(rows, parse.(Float64, split(s, ",")))
            end
        end
    end

    isempty(rows) && throw(ArgumentError("No data rows found in: $path"))
    Z = Matrix{Float64}(reduce(vcat, reshape.(rows, 1, :)))

    H, W = size(Z)

    # Parse geotransform — now also found under the normalised key "geotransform"
    gt = if haskey(meta, "geotransform")
        try
            nums = parse.(Float64,
                split(replace(replace(meta["geotransform"], r"[\[\]]" => ""),
                              r"\s+" => " "),
                      r"[,\s]+"; keepempty=false))
            if length(nums) >= 6
                gt_cand = GeoTransform(nums)
                # Sanity check: if gt implies a grid much larger than Z, it is
                # probably the native-resolution geotransform applied to a
                # screenshot-sized matrix — warn and fall back to identity.
                implied_H = abs(gt_cand.dy) > 1e-9 ? round(Int, 263.0 * 1.0 / abs(gt_cand.dy)) : H
                implied_W = abs(gt_cand.dx) > 1e-9 ? round(Int, 324.0 * 1.0 / abs(gt_cand.dx)) : W
                if implied_H > H * 2 || implied_W > W * 2
                    @warn "load_kde_surface_csv: embedded geotransform implies a grid " *
                          "(≈$(implied_H)×$(implied_W)) much larger than the CSV dimensions " *
                          "($(H)×$(W)). This usually means a native-resolution geotransform " *
                          "was saved with a downsampled surface. " *
                          "Use `resample_to_image_grid` (not `resample_to_count_grid`) " *
                          "for screenshot-derived surfaces. Storing geotransform as-is; " *
                          "do NOT pass this KDESurface to `resample_to_count_grid`."
                end
                gt_cand
            else
                GT_COUNTGRID
            end
        catch
            GT_COUNTGRID
        end
    else
        GT_COUNTGRID
    end

    crs   = get(meta, "crs", CRS_DURHAM)
    notes = get(meta, "notes", "")

    return KDESurface(Z, gt, crs; notes=notes)
end

# ===========================================================================
# Geotransform-aware resampling — dispatched on ResampleMethod
# ===========================================================================

"""
    resample_to_count_grid(surf::KDESurface;
                            H_out  = 263,
                            W_out  = 324,
                            gt_out = GT_COUNTGRID,
                            method = NearestNeighbor()) -> Matrix{Float64}

Resample `surf` to the count-grid at `(H_out, W_out)` using the coordinate
mapping defined by `gt_out`.

## Dispatch

The `method` argument selects between `NearestNeighbor()` (default) and
`Bilinear()`.  This is a type-dispatched design: adding a new interpolation
strategy requires only defining a new `ResampleMethod` subtype and a
corresponding `_sample_native` method.

## Algorithm (both methods)

For each output cell (i, j):
1. Compute world coordinates from `gt_out`:
       x = gt_out.x_origin + (j - 0.5) * gt_out.dx
       y = gt_out.y_origin + (i - 0.5) * gt_out.dy   # dy < 0 for north-up
2. Map to fractional native pixel index:
       col_f = (x - gt_in.x_origin) / gt_in.dx + 0.5
       row_f = (y - gt_in.y_origin) / gt_in.dy + 0.5
3. Sample `surf.Z` at (row_f, col_f) using the chosen method.

## Orientation

Row 1 of the output is the northernmost row, matching the count grid
convention (GT_COUNTGRID.dy = -1.0 < 0).

## Example

```julia
surf = load_kde_surface("kde_surface_true.jld2")
Z_263 = resample_to_count_grid(surf)                   # nearest-neighbour
Z_263 = resample_to_count_grid(surf; method=Bilinear()) # bilinear
@assert size(Z_263) == (263, 324)
@assert all(0.0 .<= Z_263 .<= 1.0)
```
"""
function resample_to_count_grid(surf::KDESurface;
                                 H_out ::Int          = 263,
                                 W_out ::Int          = 324,
                                 gt_out::GeoTransform = GT_COUNTGRID,
                                 method::ResampleMethod = NearestNeighbor())::Matrix{Float64}

    gt_in  = surf.geotransform
    Z_in   = surf.Z
    H_in, W_in = size(Z_in)

    Z_out = Matrix{Float64}(undef, H_out, W_out)

    @inbounds for j in 1:W_out
        x = gt_out.x_origin + (j - 0.5) * gt_out.dx
        col_f = (x - gt_in.x_origin) / gt_in.dx + 0.5

        for i in 1:H_out
            y = gt_out.y_origin + (i - 0.5) * gt_out.dy  # dy < 0 north-up
            row_f = (y - gt_in.y_origin) / gt_in.dy + 0.5
            Z_out[i, j] = _sample_native(method, Z_in, row_f, col_f, H_in, W_in)
        end
    end

    # Warn if output is outside [0,1] (indicates geotransform mismatch)
    vmin, vmax = extrema(Z_out)
    (vmin < -1e-9 || vmax > 1.0 + 1e-9) &&
        @warn "resample_to_count_grid: output range [$vmin, $vmax] is outside [0,1]. " *
              "Check that gt_out covers the same region as surf.geotransform."

    return Z_out
end

"""
    resample_to_count_grid(Z::Matrix{Float64}, gt_in::GeoTransform; kwargs...)

Convenience overload: wraps `(Z, gt_in)` in a `KDESurface` with `GT_COUNTGRID`
and forwards to the primary method. `crs` is set to `CRS_DURHAM` by default.

Useful when you already have a resampled surface stored as a bare matrix.
"""
function resample_to_count_grid(Z::Matrix{Float64}, gt_in::GeoTransform;
                                 kwargs...)::Matrix{Float64}
    surf = KDESurface(Z, gt_in, CRS_DURHAM)
    return resample_to_count_grid(surf; kwargs...)
end

"""
    resample_to_count_grid(Z::Matrix{Float64}, gt_in::AbstractVector; kwargs...)

Convenience overload accepting a raw length-6 vector geotransform.
"""
function resample_to_count_grid(Z::Matrix{Float64}, gt_in::AbstractVector;
                                 kwargs...)::Matrix{Float64}
    resample_to_count_grid(Z, GeoTransform(gt_in); kwargs...)
end

# ===========================================================================
# Image-grid resampling — screenshot / arbitrary-resolution path
# ===========================================================================

"""
    resample_to_image_grid(Z_src::AbstractMatrix;
                            H_out  = 263,
                            W_out  = 324,
                            method = NearestNeighbor()) -> Matrix{Float64}

Resample `Z_src` to `(H_out, W_out)` using **proportional pixel-fraction
mapping only** — no geotransform is involved.  The mapping is:

    row_f = (i - 0.5) / H_out * H_src + 0.5
    col_f = (j - 0.5) / W_out * W_src + 0.5

This is the correct approach for screenshot-derived KDE surfaces where no
true geospatial registration is available.  It preserves the spatial layout
of the source image: blobs in the upper-left of `Z_src` will appear in the
upper-left of the output, blobs in the lower-right will appear in the
lower-right, etc.

## When to use

- **Screenshot-derived KDE surfaces**: `Z_src` is a 1080×1330 (or similar)
  density grid derived from a screenshot orthomosaic.  No UTM geotransform
  relates it to the count grid.  Using `resample_to_count_grid` with an
  invented or mismatched geotransform will produce a collapsed edge band.
- **Any case where the source grid has no known UTM registration.**

## When NOT to use

- **True native KDE surface** (9279×11421, EPSG:6348): use
  `resample_to_count_grid` with the correct `GeoTransform` — that path uses
  actual UTM coordinates to achieve sub-metre alignment with the count grid.

## Orientation

Row 1 of the output maps to row 1 of the source (north-up if the source is
north-up).  No flip is applied.  If `Z_src` has dy < 0 (north-up, row-1 =
north), the output preserves that orientation.

## Example

```julia
# Screenshot KDE: 1080×1330, no reliable geotransform
Z_screen = load_kde_surface_csv("output/orthomosaic_kde/kmedoids_kde_density.csv").Z
Z_263 = resample_to_image_grid(Z_screen)              # NN default
Z_263 = resample_to_image_grid(Z_screen; method=Bilinear())
@assert size(Z_263) == (263, 324)
```
"""
function resample_to_image_grid(Z_src::AbstractMatrix;
                                 H_out  ::Int            = 263,
                                 W_out  ::Int            = 324,
                                 method ::ResampleMethod = NearestNeighbor())::Matrix{Float64}

    H_src, W_src = size(Z_src)
    Z_in  = Matrix{Float64}(Z_src)   # ensure concrete type for @inbounds
    Z_out = Matrix{Float64}(undef, H_out, W_out)

    @inbounds for i in 1:H_out
        # Map output row i (1-indexed, 0.5-centred) to source fractional row
        row_f = (Float64(i) - 0.5) / Float64(H_out) * Float64(H_src) + 0.5
        for j in 1:W_out
            col_f = (Float64(j) - 0.5) / Float64(W_out) * Float64(W_src) + 0.5
            Z_out[i, j] = _sample_native(method, Z_in, row_f, col_f, H_src, W_src)
        end
    end

    return Z_out
end

"""
    resample_to_image_grid(surf::KDESurface; kwargs...) -> Matrix{Float64}

Dispatch overload: unwraps `surf.Z` and forwards to the matrix method.  The
geotransform stored in `surf` is intentionally ignored — this function is
for spatial-geometry-preserving downsampling of screenshot-derived surfaces
where no valid UTM registration exists.
"""
function resample_to_image_grid(surf::KDESurface; kwargs...)::Matrix{Float64}
    resample_to_image_grid(surf.Z; kwargs...)
end

# ---------------------------------------------------------------------------
# Internal sampling kernels — dispatched on ResampleMethod
# ---------------------------------------------------------------------------

"""
    _sample_native(::NearestNeighbor, Z, row_f, col_f, H, W) -> Float64

Sample `Z` at fractional index `(row_f, col_f)` by nearest-neighbour rounding.
Clamps to matrix bounds.
"""
@inline function _sample_native(::NearestNeighbor,
                                 Z::Matrix{Float64},
                                 row_f::Float64, col_f::Float64,
                                 H::Int, W::Int)::Float64
    ri = clamp(round(Int, row_f), 1, H)
    ci = clamp(round(Int, col_f), 1, W)
    return @inbounds Z[ri, ci]
end

"""
    _sample_native(::Bilinear, Z, row_f, col_f, H, W) -> Float64

Sample `Z` at fractional index `(row_f, col_f)` by bilinear interpolation.
Clamps corner indices to matrix bounds; extrapolation is flat (boundary value).
"""
@inline function _sample_native(::Bilinear,
                                 Z::Matrix{Float64},
                                 row_f::Float64, col_f::Float64,
                                 H::Int, W::Int)::Float64
    # Clamp the fractional coordinates themselves to the valid pixel-centre
    # range [1, H] / [1, W] before computing the interpolation corners.
    # This prevents extrapolation artefacts at the boundary where floor(row_f)
    # could land on a pixel index of 0 (outside the matrix).
    row_fc = clamp(row_f, 1.0, Float64(H))
    col_fc = clamp(col_f, 1.0, Float64(W))
    r0 = clamp(floor(Int, row_fc), 1, H - 1)
    r1 = r0 + 1
    c0 = clamp(floor(Int, col_fc), 1, W - 1)
    c1 = c0 + 1
    dr = row_fc - r0
    dc = col_fc - c0
    @inbounds (Z[r0, c0] * (1 - dr) * (1 - dc) +
               Z[r0, c1] * (1 - dr) * dc        +
               Z[r1, c0] * dr       * (1 - dc)  +
               Z[r1, c1] * dr       * dc)
end
