"""
    lidar_counts.jl — LAS point cloud → cover-stratified count grids

Ports the original `lidar.jl` standalone script's pipeline into the package,
**reusing** the existing `CountKey`, `gini_coefficient`, `morans_i`, and
`summary_statistics` already defined in `src/count_statistics.jl`. New
functionality:

- `LASMission`             — typed metadata (mission/kernel) attached to a LAS
- `read_classes_geotiff`   — load the GLI cover-class raster
- `crop_las`               — clip a LAS to a (xmin, xmax, ymin, ymax) bbox
- `bin_to_count_grid`      — accumulate point counts per (row, col, cover)
- `build_count_grids`      — orchestrate the full Mission × {all, ground} sweep
- `write_count_grids_json` — persist counts in the format expected by
                             `load_counts_json`
- `cover_stats_table`      — long-table Q1/Q2/Q3/IQR/Mean + CR/CV/Gini/MoranI
- `percent_density_table`  — wide table of % cells in count bins
- `STANDARD_MISSION_PLAN`  — 6-tuple of (mission, kernel) used for the paper

The kernel-agnostic missions (`:const2`, `:const8`) use `kernel = :NA`,
mirroring `count_statistics.jl`.
"""

# LASDatasets is a [deps] entry in Project.toml. Import it eagerly here so
# the module exposes a stable API surface — the previous `@eval using` inside
# `_require_las()` caused "package LASDatasets not loaded" at world-age
# crossings on real runs. Importing eagerly is cheap because LASDatasets is
# precompiled alongside the rest of the package.

using DataFrames
using Statistics
using Printf: @sprintf
using LASDatasets
const _LAS = LASDatasets

# ---------------------------------------------------------------------------
# Constants and types
# ---------------------------------------------------------------------------

"""
    STANDARD_MISSION_PLAN

The 6-mission plan used in the manuscript:

    Density-aware × G, Speed-aware × G, Density-aware × E, Speed-aware × E,
    Constant 2 m/s (kernel = :NA), Constant 8 m/s (kernel = :NA).

Mirrors `lidar.jl::PLAN`.
"""
const STANDARD_MISSION_PLAN = (
    (mission = :density, kernel = :G),
    (mission = :speed,   kernel = :G),
    (mission = :density, kernel = :E),
    (mission = :speed,   kernel = :E),
    (mission = :const2,  kernel = :NA),
    (mission = :const8,  kernel = :NA),
)

"""
    DEFAULT_PD_BINS

Default percent-density bins used in the manuscript table:
    "0", "1", ">1–10", "10–25", ">25"
"""
const DEFAULT_PD_BINS = (
    (0,  0,                 "0"),
    (1,  1,                 "1"),
    (2,  10,                ">1–10"),
    (11, 25,                "10–25"),
    (26, typemax(Int),      ">25"),
)

const GROUND_CLASS_DEFAULT = UInt8(2)   # ASPRS ground classification

"""
    LASMission

Metadata attached to a single LAS dataset for downstream count-grid building.

Fields:
- `mission::Symbol`  — :density, :speed, :const2, :const8
- `kernel::Symbol`   — :G, :E, :NA
- `path::String`     — file path
- `ds`               — opaque LAS dataset handle (filled by `crop_las`)
"""
mutable struct LASMission
    mission::Symbol
    kernel ::Symbol
    path   ::String
    ds     ::Any
end

LASMission(mission::Symbol, kernel::Symbol, path::AbstractString) =
    LASMission(mission, kernel, String(path), nothing)

is_kernel_agnostic(m::Symbol) = (m === :const2 || m === :const8)

# ---------------------------------------------------------------------------
# GLI / classes raster
# ---------------------------------------------------------------------------

"""
    read_classes_geotiff(path; band=1) -> (classes::Matrix, gt::GeoTransform)

Read the GLI cover-class raster. Pixel values must be integer-coded as in
`RunInputs.gli_class_codes` (default Field=2, Decid=0, Conif=1).

Returns a `Matrix{<:Integer}` shaped `(H, W)` and a `GeoTransform`.
"""
function read_classes_geotiff(path::AbstractString; band::Integer = 1)
    Z, gt, _crs = read_band(path; band = band)
    return Z, gt
end

# ---------------------------------------------------------------------------
# crop_las and helpers
# ---------------------------------------------------------------------------

"""
    crop_las(path; xmin, xmax, ymin, ymax) -> NamedTuple

Open a LAS file, crop to the spatial bbox, and extract:
- `X` ∈ ℝ^{n×3}  — (x, y, z) per remaining point
- `cls`          — classification codes (Vector{UInt8})

Uses `LASDatasets.jl`. Loaded lazily.
"""
function crop_las(path::AbstractString;
                  xmin::Real, xmax::Real, ymin::Real, ymax::Real)
    isfile(path) || throw(ArgumentError("LAS file not found: $path"))
    ds  = _LAS.load_las(path)
    pc  = _LAS.get_pointcloud(ds)
    pos = pc.position
    xs  = [Float64(p[1]) for p in pos]
    ys  = [Float64(p[2]) for p in pos]
    zs  = [Float64(p[3]) for p in pos]
    keep = (xmin .<= xs .<= xmax) .& (ymin .<= ys .<= ymax)
    remove_idx = findall(.!keep)
    _LAS.remove_points!(ds, remove_idx)
    # Re-extract after removal so positions and classification stay aligned
    pc2  = _LAS.get_pointcloud(ds)
    pos2 = pc2.position
    cls  = pc2.classification
    n = length(pos2)
    X = Matrix{Float64}(undef, n, 3)
    @inbounds for i in 1:n
        p = pos2[i]
        X[i, 1] = Float64(p[1])
        X[i, 2] = Float64(p[2])
        X[i, 3] = Float64(p[3])
    end
    return (ds = ds, X = X, cls = collect(cls))
end

# (Previously held a lazy-load shim. Kept as a stable no-op accessor so any
# external caller that imported `KDEFlightPlanning._require_las` continues to
# resolve. Prefer the eagerly-bound module constant `_LAS` inside this file.)
_require_las() = _LAS

"""
    ground_points(X, cls; ground_class=GROUND_CLASS_DEFAULT) -> Matrix

Subset of `X` rows where `cls == ground_class`.
"""
function ground_points(X::AbstractMatrix, cls::AbstractVector;
                       ground_class::Integer = GROUND_CLASS_DEFAULT)
    mask = cls .== ground_class
    return X[mask, :]
end

# ---------------------------------------------------------------------------
# bin_to_count_grid — vectorised scatter into (row, col)
# ---------------------------------------------------------------------------

"""
    bin_to_count_grid(X, classes; xmin, xmax, ymin, ymax, dx, dy,
                       class_codes=Dict(:field=>2,:decid=>0,:conif=>1))
        -> (counts_field, counts_decid, counts_conif)

Bin point positions in `X[:,1:2]` to the (row, col) cells of a raster
sized (`size(classes)`) and accumulate per cover class. Cells outside the
raster, and points whose underlying cover code is not in `class_codes`, are
ignored.
"""
function bin_to_count_grid(X::AbstractMatrix, classes::AbstractMatrix;
                            xmin::Real, xmax::Real, ymin::Real, ymax::Real,
                            dx::Real, dy::Real,
                            class_codes::AbstractDict = Dict(:field => 2, :decid => 0, :conif => 1))
    nrows, ncols = size(classes)
    nx = Int(ceil((xmax - xmin) / dx))
    ny = Int(ceil((ymax - ymin) / dy))
    @assert nx == ncols "Grid columns mismatch: counts $(nx) vs classes $(ncols)"
    @assert ny == nrows "Grid rows mismatch:    counts $(ny) vs classes $(nrows)"

    cf = zeros(Int32, ny, nx)
    cd = zeros(Int32, ny, nx)
    cc = zeros(Int32, ny, nx)

    code_field = Int(class_codes[:field])
    code_decid = Int(class_codes[:decid])
    code_conif = Int(class_codes[:conif])

    @inbounds for i in axes(X, 1)
        xg = X[i, 1]
        yg = X[i, 2]
        col = Int(floor((xg - xmin) / dx)) + 1
        row = Int(floor((ymax - yg) / dy)) + 1
        (1 ≤ col ≤ nx && 1 ≤ row ≤ ny) || continue
        cls = classes[row, col]
        if cls == code_field
            cf[row, col] += 1
        elseif cls == code_decid
            cd[row, col] += 1
        elseif cls == code_conif
            cc[row, col] += 1
        end
    end
    return cf, cd, cc
end

# ---------------------------------------------------------------------------
# build_count_grids — full mission sweep
# ---------------------------------------------------------------------------

"""
    build_count_grids(missions::Vector{LASMission}, classes, gt;
                      class_codes = Dict(:field=>2, :decid=>0, :conif=>1))
        -> (counts::Dict{CountKey, Matrix{Int32}}, datasets)

For each mission in `missions` (matching the order/length of
`STANDARD_MISSION_PLAN`), open the LAS, crop to the classes raster extent,
compute per-cover counts for both all returns and ground returns, and
populate a `Dict{CountKey, Matrix{Int32}}`.
"""
function build_count_grids(missions::Vector{LASMission},
                            classes::AbstractMatrix, gt::GeoTransform;
                            class_codes::AbstractDict = Dict(:field => 2, :decid => 0, :conif => 1))
    ex = raster_extents(classes, gt)

    out = Dict{CountKey, Matrix{Int32}}()
    datasets = LASMission[]

    for m in missions
        cropped = crop_las(m.path; xmin = ex.xmin, xmax = ex.xmax,
                                    ymin = ex.ymin, ymax = ex.ymax)
        m.ds = cropped.ds
        push!(datasets, m)

        # All returns
        cf, cd, cc = bin_to_count_grid(cropped.X, classes;
            xmin = ex.xmin, xmax = ex.xmax, ymin = ex.ymin, ymax = ex.ymax,
            dx = ex.dx, dy = ex.dy, class_codes = class_codes)
        _insert_counts!(out, :all, m.mission, m.kernel, cf, cd, cc)

        # Ground returns
        Xg = ground_points(cropped.X, cropped.cls)
        cf_g, cd_g, cc_g = bin_to_count_grid(Xg, classes;
            xmin = ex.xmin, xmax = ex.xmax, ymin = ex.ymin, ymax = ex.ymax,
            dx = ex.dx, dy = ex.dy, class_codes = class_codes)
        _insert_counts!(out, :ground, m.mission, m.kernel, cf_g, cd_g, cc_g)
    end
    return out, datasets
end

function _insert_counts!(out::Dict{CountKey, Matrix{Int32}},
                          ret::Symbol, mission::Symbol, kernel::Symbol,
                          cf::Matrix{Int32}, cd::Matrix{Int32}, cc::Matrix{Int32})
    out[CountKey(ret = ret, cover = :field, mission = mission, kernel = kernel)] = cf
    out[CountKey(ret = ret, cover = :decid, mission = mission, kernel = kernel)] = cd
    out[CountKey(ret = ret, cover = :conif, mission = mission, kernel = kernel)] = cc
    return out
end

# ---------------------------------------------------------------------------
# percent_density_table — % of cells per count bin per (mission × cover × ret)
# ---------------------------------------------------------------------------

"""
    percent_density_table(counts, classes; class_codes, bins=DEFAULT_PD_BINS) -> DataFrame

Wide table with columns `Return, Mission, Kernel, Cover, Ncells, "0", "1",
">1–10", "10–25", ">25"` (or whatever bin labels are supplied).
"""
function percent_density_table(counts::AbstractDict{CountKey, <:AbstractMatrix},
                                classes::AbstractMatrix;
                                class_codes::AbstractDict = Dict(:field => 2, :decid => 0, :conif => 1),
                                bins = DEFAULT_PD_BINS)
    masks = Dict{Symbol, BitMatrix}()
    for (cov, code) in class_codes
        masks[cov] = BitMatrix(classes .== code)
    end

    rows = NamedTuple[]
    for (key, grid) in counts
        haskey(masks, key.cover) || continue
        mask = masks[key.cover]
        v = grid[mask]
        N = length(v)
        row = Dict{Symbol, Any}(
            :Return  => return_label(key.ret),
            :Mission => mission_label(key.mission),
            :Kernel  => kernel_label(key.kernel),
            :Cover   => cover_label(key.cover),
            :Ncells  => N,
        )
        for (lo, hi, label) in bins
            row[Symbol(label)] = (N == 0) ? 0.0 : 100.0 * count(x -> lo ≤ x ≤ hi, v) / N
        end
        push!(rows, (; row...))
    end
    df = DataFrame(rows)
    sort!(df, [:Return, :Cover, :Mission, :Kernel])
    return df
end

# ---------------------------------------------------------------------------
# cover_stats_table — Q1/Q2/Q3/IQR/Mean + CR/CV/Gini/MoranI
# ---------------------------------------------------------------------------

"""
    cover_stats_table(counts, classes; …) -> DataFrame

Long-format statistics table mirroring `lidar.jl::stats_table`. Reuses the
package's `summary_statistics` (single-arg form on count vectors) and
`morans_i` from `count_statistics.jl`.
"""
function cover_stats_table(counts::AbstractDict{CountKey, <:AbstractMatrix},
                            classes::AbstractMatrix;
                            class_codes::AbstractDict = Dict(:field => 2, :decid => 0, :conif => 1),
                            covers   = (:field, :decid, :conif),
                            returns  = (:all, :ground),
                            missions = (:density, :speed, :const2, :const8),
                            kernels  = (:G, :E),
                            dropzeros::Bool = true,
                            pretty::Bool = true,
                            add_uniformity::Bool = true)
    masks = Dict{Symbol, BitMatrix}()
    for cov in covers
        haskey(class_codes, cov) || continue
        masks[cov] = BitMatrix(classes .== class_codes[cov])
    end

    rows = NamedTuple[]
    for cover in covers, rtype in returns, m in missions
        haskey(masks, cover) || continue
        mask = masks[cover]
        ks = is_kernel_agnostic(m) ? (:NA,) : kernels
        for k in ks
            key = CountKey(ret = rtype, cover = cover, mission = m, kernel = k)
            haskey(counts, key) || continue
            A = counts[key]
            N = count(mask)
            v = vec(A[mask])
            v_nz = v[0 .< v]
            ncells = length(v_nz)
            stats = if ncells == 0
                (Q1 = 0.0, Q2 = 0.0, Q3 = 0.0, IQR = 0.0, Mean = 0.0)
            else
                q1, q2, q3 = quantile(v_nz, (0.25, 0.50, 0.75))
                (Q1 = q1, Q2 = q2, Q3 = q3, IQR = q3 - q1, Mean = mean(v_nz))
            end

            unif = NamedTuple()
            if add_uniformity
                CR = ncells / N
                CV = ncells == 0 ? NaN : std(v_nz; corrected = true) / mean(v_nz)
                G  = ncells == 0 ? NaN : gini_coefficient(v_nz)
                I  = morans_i(A, mask; neighbor = :queen)
                unif = (N = N, CR = CR, CV = CV, Gini = G, MoranI = I)
            end

            push!(rows, (;
                Cover   = pretty ? cover_label(cover)   : cover,
                Return  = pretty ? return_label(rtype)  : rtype,
                Mission = pretty ? mission_label(m)     : m,
                Kernel  = pretty ? kernel_label(k)      : k,
                ncells  = ncells,
                stats...,
                unif...,
            ))
        end
    end
    df = DataFrame(rows)
    sort!(df, [:Return, :Cover, :Mission, :Kernel])
    return df
end

# ---------------------------------------------------------------------------
# write_count_grids_json — emit counts.json compatible with load_counts_json
# ---------------------------------------------------------------------------

"""
    write_count_grids_json(counts, path; nrows, ncols) -> path

Serialise a `Dict{CountKey, Matrix{Int32}}` to the same JSON schema accepted
by [`load_counts_json`](@ref). Each record is a flat row-major vector of
length `nrows * ncols`.
"""
function write_count_grids_json(counts::AbstractDict{CountKey, <:AbstractMatrix},
                                  path::AbstractString;
                                  nrows::Int, ncols::Int)
    records = Vector{Dict{String, Any}}()
    for (key, grid) in counts
        @assert size(grid) == (nrows, ncols) "count grid shape != ($nrows, $ncols)"
        push!(records, Dict(
            "ret"     => string(key.ret),
            "cover"   => string(key.cover),
            "mission" => string(key.mission),
            "kernel"  => string(key.kernel),
            "values"  => collect(reshape(transpose(grid), :)),  # row-major
        ))
    end
    obj = Dict(
        "nrows"   => nrows,
        "ncols"   => ncols,
        "records" => records,
    )
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        JSON.print(io, obj, 2)
    end
    return path
end
