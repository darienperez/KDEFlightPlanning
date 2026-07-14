"""
    scripts/inspect_geotiff.jl

Print the six-term GDAL geotransform, raster size, per-axis ground
resolutions, and CRS/WKT of a source orthomosaic GeoTIFF — using ONLY the
package's existing GeoTIFF API (`KDEFlightPlanning.read_band`, which wraps
`ArchGDAL.getgeotransform`/`ArchGDAL.getproj`) plus `geotransform_resolution`.

This is the tool to run BEFORE enabling metric processing for any screenshot
site: the uploaded JPEGs carry NO geotransform metadata, so a co-registered
source GeoTIFF is required to obtain a real ground scale.

Geotransform indexing (note the 0-based GDAL ↔ 1-based Julia offset):

    Julia gt[1] = GDAL GT[0] = x_origin  (top-left X)
    Julia gt[2] = GDAL GT[1] = dx        (x pixel-axis, +east per column)
    Julia gt[3] = GDAL GT[2] = x_rot     (row rotation / skew)
    Julia gt[4] = GDAL GT[3] = y_origin  (top-left Y)
    Julia gt[5] = GDAL GT[4] = y_rot     (column rotation / skew)
    Julia gt[6] = GDAL GT[5] = dy        (y pixel-axis, usually −north/row)

Per-axis ground resolution (rotation/skew-safe, via hypot of each affine
axis vector — NOT abs(dx)/abs(dy)):

    xres = hypot(gt[2], gt[5])   # Julia 1-based  == hypot(GDAL GT[1], GT[4])
    yres = hypot(gt[3], gt[6])   # Julia 1-based  == hypot(GDAL GT[2], GT[5])

`geotransform_resolution(gt)` (src/pipeline.jl) returns exactly this pair.

Usage:
    julia --project=. scripts/inspect_geotiff.jl PATH/TO/source_ortho.tif
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using KDEFlightPlanning
using Printf

import KDEFlightPlanning: read_band, geotransform_resolution

function inspect(path::AbstractString)
    isfile(path) || error("GeoTIFF not found: $path")

    # read_band → (Matrix{H×W}, GeoTransform, crs_wkt::String). The returned
    # matrix is (rows, cols) = (H, W); the geotransform is the 6-term GDAL
    # affine; crs is the projection WKT (empty string if the raster is
    # un-projected).
    Z, gt, crs = read_band(path; band = 1)
    H, W = size(Z)
    xres, yres = geotransform_resolution(gt)   # metres / source pixel, per axis

    println("── GeoTIFF: $path ──")
    @printf("raster size            : W=%d  H=%d  (px)\n", W, H)
    println()
    println("GDAL geotransform (Julia 1-based tuple ↔ GDAL 0-based):")
    @printf("  gt[1] x_origin  (GT[0]) = %.6f\n", gt[1])
    @printf("  gt[2] dx        (GT[1]) = %.9f\n", gt[2])
    @printf("  gt[3] x_rot     (GT[2]) = %.9f\n", gt[3])
    @printf("  gt[4] y_origin  (GT[3]) = %.6f\n", gt[4])
    @printf("  gt[5] y_rot     (GT[4]) = %.9f\n", gt[5])
    @printf("  gt[6] dy        (GT[5]) = %.9f\n", gt[6])
    println()
    @printf("x pixel-axis resolution : hypot(gt[2],gt[5]) = %.9f m/px\n", xres)
    @printf("y pixel-axis resolution : hypot(gt[3],gt[6]) = %.9f m/px\n", yres)
    @printf("ground extent           : %.3f m (W) × %.3f m (H)\n", W * xres, H * yres)
    println()
    if isempty(crs)
        println("CRS / WKT               : <none — raster is not georeferenced>")
    else
        println("CRS / WKT               :")
        println(crs)
    end
    return (; W, H, gt, xres, yres, crs)
end

function main()
    isempty(ARGS) &&
        error("Usage: julia --project=. scripts/inspect_geotiff.jl PATH/TO/source_ortho.tif")
    inspect(abspath(ARGS[1]))
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
