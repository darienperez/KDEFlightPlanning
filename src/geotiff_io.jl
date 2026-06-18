"""
    geotiff_io.jl — First-class GeoTIFF I/O via ArchGDAL.jl

Replaces the pre-1.9 `Main.ArchGDAL` reflection pattern with a direct
`using ArchGDAL` dependency (declared in Project.toml `[deps]`).

Public types
------------
- `GeoRasterStack`         — generic band stack + geotransform + CRS
- (extends `GeoTransform`  — already defined in kde_surface_io.jl)

Public API
----------
- `load_rgb_geotiff(path)            → GeoRasterStack{RGB{N0f8}}`
- `read_band(path, band=1)           → (Matrix, GeoTransform, crs_wkt)`
- `raster_extents(z, gt)             → (xmin, xmax, ymin, ymax, dx, dy)`
- `write_single_band_geotiff(path, Z, gt, crs_wkt)  → path`
- `axes_from_georasterstack(rs)      → (xs, ys)`

GeoTransform convention (GDAL):
    gt = [x_origin, dx, x_rot, y_origin, y_rot, dy]
    dy is typically negative for north-up rasters.
"""

using ArchGDAL
const AG = ArchGDAL

# RGB and N0f8 come from Colors / ColorTypes already loaded at the top of
# the package. Re-import them here for clarity / type signatures below.
import Colors: RGB
# N0f8 is the FixedPointNumbers type Colors ships with; resolve it via
# ColorTypes (already a dep) without declaring FixedPointNumbers explicitly.
import ColorTypes: N0f8

# ---------------------------------------------------------------------------
# GeoRasterStack
# ---------------------------------------------------------------------------

"""
    GeoRasterStack{T}

Bundles a 2-D matrix (band) with its GDAL `GeoTransform` and CRS WKT string.
For multi-band rasters, the channel dim is folded into `T` as a colorant
(e.g. `RGB{N0f8}`) for first-class image semantics.

Fields:
- `Z::Matrix{T}`        — pixel data (row=y, col=x by GDAL convention)
- `gt::GeoTransform`    — 6-coefficient GDAL geotransform
- `crs::String`         — CRS WKT string ("" if unknown)
- `source::String`      — file path of origin (informational)
"""
struct GeoRasterStack{T}
    Z      ::Matrix{T}
    gt     ::GeoTransform
    crs    ::String
    source ::String
end

Base.size(rs::GeoRasterStack) = size(rs.Z)
Base.eltype(::GeoRasterStack{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", rs::GeoRasterStack{T}) where {T}
    H, W = size(rs.Z)
    print(io, "GeoRasterStack{$T}  $(H)×$(W)  px=$(rs.gt.dx)×$(rs.gt.dy)  crs=", isempty(rs.crs) ? "<unknown>" : "<set>")
end

# ---------------------------------------------------------------------------
# read_band — single-band read
# ---------------------------------------------------------------------------

"""
    read_band(path; band=1) -> (Matrix, GeoTransform, crs_wkt::String)

Read one band of a GeoTIFF (or any GDAL-supported raster) plus its
geotransform and CRS. Returned matrix is shape (H, W) = (rows, cols).
"""
function read_band(path::AbstractString; band::Integer = 1)
    isfile(path) || throw(ArgumentError("File not found: $path"))
    return AG.read(path) do ds
        Z      = AG.read(ds, band)
        gt_vec = AG.getgeotransform(ds)
        crs    = try
            String(AG.getproj(ds))
        catch
            ""
        end
        # GDAL returns the band as a (W, H) array (col-major axis order matches
        # GDAL pixel scanline). Transpose to (H, W) = (rows, cols).
        Z2 = permutedims(Z)
        (Z2, GeoTransform(gt_vec), crs)
    end
end

# ---------------------------------------------------------------------------
# load_rgb_geotiff — 3-band RGB read
# ---------------------------------------------------------------------------

"""
    load_rgb_geotiff(path) -> GeoRasterStack{RGB{N0f8}}

Read a 3-band RGB GeoTIFF into a `Matrix{RGB{N0f8}}`. Each pixel value is
8-bit normalised; band order R=1, G=2, B=3 (GDAL convention).
"""
function load_rgb_geotiff(path::AbstractString)::GeoRasterStack{RGB{N0f8}}
    isfile(path) || throw(ArgumentError("File not found: $path"))
    return AG.read(path) do ds
        nbands = AG.nraster(ds)
        nbands >= 3 || throw(ArgumentError(
            "Expected ≥ 3 bands (R, G, B) in $path; got $nbands"))
        r_raw  = AG.read(ds, 1)
        g_raw  = AG.read(ds, 2)
        b_raw  = AG.read(ds, 3)
        # GDAL returns (W, H); transpose to (H, W) — match north-up image
        # convention used elsewhere in the package.
        r  = permutedims(r_raw)
        g  = permutedims(g_raw)
        b  = permutedims(b_raw)
        H, W = size(r)
        Z = Matrix{RGB{N0f8}}(undef, H, W)
        @inbounds for j in 1:H, i in 1:W
            Z[j, i] = RGB{N0f8}(
                reinterpret(N0f8, UInt8(r[j, i])),
                reinterpret(N0f8, UInt8(g[j, i])),
                reinterpret(N0f8, UInt8(b[j, i])),
            )
        end
        gt_vec = AG.getgeotransform(ds)
        crs    = try
            String(AG.getproj(ds))
        catch
            ""
        end
        GeoRasterStack{RGB{N0f8}}(Z, GeoTransform(gt_vec), crs, String(path))
    end
end

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

"""
    raster_extents(Z, gt) -> (xmin, xmax, ymin, ymax, dx, dy)

Compute UTM (or otherwise CRS-mapped) extents of a raster from its pixel
matrix and geotransform. `dy` is returned as a positive magnitude even when
the underlying geotransform stores a negative `dy` (north-up convention).
"""
function raster_extents(Z::AbstractMatrix, gt::GeoTransform)
    nrows, ncols = size(Z)
    xmin = gt.x_origin
    xmax = gt.x_origin + ncols * gt.dx
    ymax = gt.y_origin
    ymin = gt.y_origin + nrows * gt.dy   # dy negative → ymin < ymax
    dx   = abs(gt.dx)
    dy   = abs(gt.dy)
    return (xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, dx = dx, dy = dy)
end

raster_extents(rs::GeoRasterStack) = raster_extents(rs.Z, rs.gt)

"""
    axes_from_georasterstack(rs) -> (xs, ys)

Return cell-centre coordinate axes for the raster.
"""
function axes_from_georasterstack(rs::GeoRasterStack)
    H, W = size(rs.Z)
    return axes_from_geotransform(_gt_as_vector(rs.gt), W, H)
end

_gt_as_vector(gt::GeoTransform) = [
    gt.x_origin, gt.dx, gt.x_rot, gt.y_origin, gt.y_rot, gt.dy,
]

# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

"""
    write_single_band_geotiff(path, Z, gt, crs_wkt; nodata=nothing) -> path

Write a single-band Float64 raster as a GeoTIFF with the supplied
geotransform and CRS. Used for emitting KDE density and speed-map products
that should remain co-registered with the source orthomosaic.
"""
function write_single_band_geotiff(path::AbstractString,
                                    Z::AbstractMatrix{<:Real},
                                    gt::GeoTransform,
                                    crs_wkt::AbstractString;
                                    nodata::Union{Nothing, Real} = nothing)
    Z64 = Matrix{Float64}(Z)
    H, W = size(Z64)
    mkpath(dirname(abspath(path)))
    # GDAL expects (W, H); transpose back.
    Z_gdal = permutedims(Z64)
    AG.create(path;
        driver = AG.getdriver("GTiff"),
        width  = W, height = H, nbands = 1,
        dtype  = Float64) do ds
        AG.setgeotransform!(ds, _gt_as_vector(gt))
        if !isempty(crs_wkt)
            AG.setproj!(ds, crs_wkt)
        end
        AG.write!(ds, Z_gdal, 1)
        if !isnothing(nodata)
            try
                AG.setnodatavalue!(AG.getband(ds, 1), Float64(nodata))
            catch
                # nodata setting is best-effort
            end
        end
    end
    return path
end
