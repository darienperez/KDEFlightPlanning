"""
    types.jl — Core data types for KDEFlightPlanning

Defines the structs used throughout the pipeline. The layout mirrors the
CanopyDensity codebase (RasterGrid field order: Z, xs, ys) so that grids can
be passed between the two packages without conversion.

Types
-----
- `RasterGrid`:             2-D density/KDE surface on a regular axis grid
- `Waypoint`:               Single UAV waypoint with position, altitude, speed, line_id
- `LawnmowerSpec`:          Boustrophedon path geometry spec
- `FlightConfig`:           Top-level mission config
- `PipelineConfig`:         CanopyDensity-side pipeline config (used in pipeline.jl)
- Speed strategy hierarchy: `SpeedStrategy`, `ConstantSpeed`, `KDEGuidedSpeed`,
                             `CurvatureGuidedSpeed`
- Kernel types:             `Kernel2D`, `FullKernel` (used by KDE convolution)

Public naming conventions
--------------------------
Do NOT use "speed-aware" or "density-aware" as main public labels. See FlightConfig
for the manuscript-aligned labels.
"""

# ---------------------------------------------------------------------------
# RasterGrid
# ---------------------------------------------------------------------------

"""
    RasterGrid{T<:Real}

A 2-D grid storing a scalar field (typically a KDE density surface normalised
to [0, 1]).

Field layout (matches CanopyDensity convention):
- `Z`:  (Ny × Nx) data matrix; `Z[j, i]` → `(xs[i], ys[j])`
- `xs`: ascending Easting (x) coordinates, length Nx
- `ys`: ascending Northing (y) coordinates, length Ny

Row index j runs along y; column index i runs along x. This matches
north-up GeoTIFF band-interleaved layout after any needed axis flips.
"""
struct RasterGrid{T<:Real}
    Z ::Matrix{T}   # size (Ny, Nx)  ← Z first, matching CanopyDensity
    xs::Vector{T}   # length Nx, ascending
    ys::Vector{T}   # length Ny, ascending

    function RasterGrid(Z::Matrix{T}, xs::Vector{T}, ys::Vector{T}) where {T<:Real}
        Nx, Ny = length(xs), length(ys)
        size(Z) == (Ny, Nx) || throw(DimensionMismatch(
            "Z must be ($(Ny), $(Nx)) = (length(ys), length(xs)), got $(size(Z))"))
        issorted(xs) || throw(ArgumentError("xs must be ascending"))
        issorted(ys) || throw(ArgumentError("ys must be ascending"))
        new{T}(Z, xs, ys)
    end
end

Base.size(g::RasterGrid) = size(g.Z)

"""
    RasterGrid(Z, xs, ys)

Construct a `RasterGrid` with automatic type promotion to Float64.
"""
function RasterGrid(Z::AbstractMatrix, xs::AbstractVector, ys::AbstractVector)
    T = promote_type(eltype(Z), eltype(xs), eltype(ys), Float64)
    RasterGrid(convert(Matrix{T}, Z), convert(Vector{T}, xs), convert(Vector{T}, ys))
end

# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------

"""
    Waypoint

A single UAV waypoint produced by the flight planner.

Fields: `x`, `y` (Easting/Northing, m), `altitude` (AGL, m),
`speed` (m/s, clamped to strategy bounds), `line_id` (0 = unassigned).
"""
struct Waypoint
    x       ::Float64
    y       ::Float64
    altitude::Float64
    speed   ::Float64
    line_id ::Int
end

"""
    Waypoint(x, y, altitude, speed; line_id=0)
"""
Waypoint(x, y, altitude, speed; line_id::Int=0) =
    Waypoint(Float64(x), Float64(y), Float64(altitude), Float64(speed), line_id)

Base.show(io::IO, w::Waypoint) =
    print(io, "Waypoint(x=$(w.x), y=$(w.y), alt=$(w.altitude), v=$(w.speed), line=$(w.line_id))")

# ---------------------------------------------------------------------------
# Speed strategy hierarchy
# ---------------------------------------------------------------------------

"""
    SpeedStrategy

Abstract supertype for all speed-assignment strategies.
Concrete subtypes dispatch on `assign_speed`.
"""
abstract type SpeedStrategy end

"""
    ConstantSpeed(v)

Fixed ground speed `v` (m/s). Use for Constant 2 m/s and 8 m/s baselines.
"""
struct ConstantSpeed <: SpeedStrategy
    v::Float64
    function ConstantSpeed(v::Real)
        v > 0 || throw(ArgumentError("speed must be positive"))
        new(Float64(v))
    end
end

"""
    KDEGuidedSpeed(dmin, dmax, vmin, vmax)

Inverse-linear density-to-speed map: high KDE density → slow speed.

    v(d) = vmin + (1 − t) × (vmax − vmin),   t = clamp((d − dmin)/(dmax − dmin), 0, 1)

Used for "KDE-guided variable speed" (Gaussian or Epanechnikov — the kernel
choice affects the density surface, not this struct).
"""
struct KDEGuidedSpeed <: SpeedStrategy
    dmin::Float64
    dmax::Float64
    vmin::Float64
    vmax::Float64

    function KDEGuidedSpeed(dmin::Real, dmax::Real, vmin::Real, vmax::Real)
        vmin > 0 && vmax >= vmin || throw(ArgumentError("need 0 < vmin ≤ vmax"))
        new(Float64(dmin), Float64(dmax), Float64(vmin), Float64(vmax))
    end
end

"""
    KDEGuidedSpeed(grid; vmin=2.0, vmax=8.0)

Convenience constructor: reads `dmin`/`dmax` from a `RasterGrid`.
"""
KDEGuidedSpeed(grid::RasterGrid; vmin::Real=2.0, vmax::Real=8.0) =
    KDEGuidedSpeed(minimum(grid.Z), maximum(grid.Z), vmin, vmax)

"""
    CurvatureGuidedSpeed(dmin, dmax, vmin, vmax; alpha, lambda, eta,
                          spacing_min, spacing_max, grad_ref, curv_ref)

Combines KDE-guided speed with curvature/gradient-adaptive waypoint spacing
("curvature-spaced KDE-guided" variant in the manuscript).

Speed formula: identical to `KDEGuidedSpeed`.

Spacing formula (Methods section):
    w = (|∇d|/g0)^alpha + lambda × (|d''|/k0)^eta
    u = 1 / (1 + w)
    spacing = spacing_min + (spacing_max − spacing_min) × u
"""
struct CurvatureGuidedSpeed <: SpeedStrategy
    dmin       ::Float64
    dmax       ::Float64
    vmin       ::Float64
    vmax       ::Float64
    alpha      ::Float64
    lambda     ::Float64
    eta        ::Float64
    spacing_min::Float64
    spacing_max::Float64
    grad_ref   ::Union{Float64, Nothing}
    curv_ref   ::Union{Float64, Nothing}

    function CurvatureGuidedSpeed(dmin, dmax, vmin, vmax;
                                   alpha::Real=1.0, lambda::Real=1.5, eta::Real=2.0,
                                   spacing_min::Real=2.0, spacing_max::Real=20.0,
                                   grad_ref=nothing, curv_ref=nothing)
        vmin > 0 && vmax >= vmin || throw(ArgumentError("need 0 < vmin ≤ vmax"))
        alpha >= 0 && lambda >= 0 && eta >= 0 || throw(ArgumentError("exponents must be ≥ 0"))
        spacing_min > 0 && spacing_max >= spacing_min ||
            throw(ArgumentError("need 0 < spacing_min ≤ spacing_max"))
        new(Float64(dmin), Float64(dmax), Float64(vmin), Float64(vmax),
            Float64(alpha), Float64(lambda), Float64(eta),
            Float64(spacing_min), Float64(spacing_max),
            isnothing(grad_ref) ? nothing : Float64(grad_ref),
            isnothing(curv_ref) ? nothing : Float64(curv_ref))
    end
end

"""
    CurvatureGuidedSpeed(grid; vmin=2.0, vmax=8.0, kwargs...)
"""
CurvatureGuidedSpeed(grid::RasterGrid; vmin::Real=2.0, vmax::Real=8.0, kwargs...) =
    CurvatureGuidedSpeed(minimum(grid.Z), maximum(grid.Z), vmin, vmax; kwargs...)

# ---------------------------------------------------------------------------
# LawnmowerSpec
# ---------------------------------------------------------------------------

"""
    LawnmowerSpec{T<:Real}

Geometry specification for a boustrophedon (lawnmower) survey path.

Fields: `xmin`, `xmax`, `ymin`, `ymax` (m), `spacing` (m),
`yaw_deg` (degrees CCW from east), `primary` (`:x` or `:y`),
`start` (`:low` or `:high`).
"""
Base.@kwdef struct LawnmowerSpec{T<:Real}
    xmin   ::T
    xmax   ::T
    ymin   ::T
    ymax   ::T
    spacing::T
    yaw_deg::T      = zero(T)
    primary::Symbol = :x
    start  ::Symbol = :low
end

# ---------------------------------------------------------------------------
# FlightConfig
# ---------------------------------------------------------------------------

"""
    FlightConfig

Top-level configuration for a single simulated mission.

Fields
------
- `strategy`:     A `SpeedStrategy` instance
- `altitude`:     Flight altitude AGL (m)
- `label`:        Human-readable label for figures/tables
                  (e.g. "Constant 2 m/s", "KDE-guided Epanechnikov")
- `kernel`:       KDE kernel (`:gaussian` or `:epanechnikov`); informational only
- `line_spacing`: Flight-line spacing (m). Paper workflow default: **40 m**.
                  Earlier versions used 20 m; that value was associated with
                  a removed speed-troubleshooting step and should not be used
                  for paper results.
"""
struct FlightConfig
    strategy    ::SpeedStrategy
    altitude    ::Float64
    label       ::String
    kernel      ::Symbol
    line_spacing::Float64

    function FlightConfig(strategy::SpeedStrategy, altitude::Real, label::String;
                          kernel::Symbol=:gaussian, line_spacing::Real=40.0)
        altitude > 0 || throw(ArgumentError("altitude must be positive"))
        line_spacing > 0 || throw(ArgumentError("line_spacing must be positive"))
        # Paper workflow default: 40 m line spacing (not 20 m).
        # The 20 m value was used in an earlier workflow version.
        new(strategy, Float64(altitude), label, kernel, Float64(line_spacing))
    end
end

Base.show(io::IO, c::FlightConfig) =
    print(io, "FlightConfig(\"$(c.label)\", alt=$(c.altitude) m, Δ=$(c.line_spacing) m, kernel=:$(c.kernel))")

# ---------------------------------------------------------------------------
# PipelineConfig  (CanopyDensity upstream; drives build_density_surface)
# ---------------------------------------------------------------------------

"""
    PipelineConfig

Configuration for the upstream canopy-density pipeline
(orthomosaic → clustering → KDE → density surface).

Fields
------
- `src_epsg`:      EPSG code for the grid CRS (default 6348 = NAD83(2011)/UTM 18N)
- `kmed_k`:        k-medoids cluster count (default 2: vegetation vs. non-vegetation)
- `seed`:          RNG seed for reproducibility (default 6213)
- `tree_labels`:   Cluster labels corresponding to vegetation (default [1]).
                   Used by callers that build a mask/KDE config directly (e.g.
                   `run_from_config.jl`, `build_density_surface`). NOTE:
                   `build_mask_autok` does NOT consult this field as a silent
                   fallback — pass its own `tree_labels` keyword (or select
                   interactively) so a missing selection is never turned into
                   `[1]`.
- `pca`:           PCA settings NamedTuple: `(variance_ratio, maxoutdim, center)`
- `kde_bandwidth`: `:auto` (Scott's rule), `:auto_indices` (index-space Scott's), or numeric
- `kde_kernel`:    `:gaussian` or `:epanechnikov`
- `kde_scaling`:   Post-KDE normalisation (`:none` = no extra rescaling)
"""
Base.@kwdef struct PipelineConfig
    src_epsg      ::Int              = 6348
    kmed_k        ::Int              = 2
    seed          ::Int              = 6213
    tree_labels   ::Vector{Int}      = [1]
    pca           ::NamedTuple       = (variance_ratio=0.95, maxoutdim=nothing, center=true)
    kde_bandwidth ::Union{Symbol,Float64} = :auto
    kde_kernel    ::Symbol           = :gaussian
    kde_scaling   ::Symbol           = :none
end

# ---------------------------------------------------------------------------
# Kernel types (used by KDE convolution in kde.jl)
# ---------------------------------------------------------------------------

"""
    Kernel2D

Abstract type for 2-D convolution kernels used in KDE computation.
"""
abstract type Kernel2D end

"""
    FullKernel{T<:AbstractMatrix}

Wraps a real-valued 2-D kernel matrix.
"""
struct FullKernel{T<:AbstractMatrix} <: Kernel2D
    K::T
end
