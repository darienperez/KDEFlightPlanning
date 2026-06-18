"""
    io.jl — CSV import/export for waypoints and analysis data

All functions in this file depend only on the stdlib `CSV` and `DataFrames`
packages (already declared in Project.toml). CRS projection (Proj.jl) is
intentionally optional: the low-level `write_waypoints_csv` / `read_waypoints_csv`
work in the grid's native XY coordinate system. A CRS-aware export hook is
provided but will throw a helpful error if Proj.jl is not loaded.

Public API
----------
Waypoints (native XY)
  - `write_waypoints_csv(path, wps; kwargs...)`
  - `read_waypoints_csv(path) -> Vector{Waypoint}`

Analysis tables
  - `write_coverage_csv(path, rows; kwargs...)`
  - `read_coverage_csv(path) -> DataFrame`

CRS export (optional — requires Proj.jl loaded by caller)
  - `write_ugcs_csv(path, wps, src_epsg; kwargs...)`
"""

# ---------------------------------------------------------------------------
# Native XY waypoint CSV
# ---------------------------------------------------------------------------

"""
    write_waypoints_csv(path::AbstractString, wps::Vector{Waypoint};
                         header=["x","y","altitude","speed","line_id"]) -> path

Write waypoints to a CSV file in native grid coordinates (Easting/Northing).
Columns: x, y, altitude, speed, line_id.
"""
function write_waypoints_csv(path::AbstractString,
                              wps ::Vector{Waypoint};
                              header::Vector{String}=["x","y","altitude","speed","line_id"])
    df = DataFrame(
        x        = [w.x        for w in wps],
        y        = [w.y        for w in wps],
        altitude = [w.altitude for w in wps],
        speed    = [w.speed    for w in wps],
        line_id  = [w.line_id  for w in wps],
    )
    rename!(df, Symbol.(header))
    CSV.write(path, df)
    return path
end

"""
    read_waypoints_csv(path::AbstractString;
                        xcol="x", ycol="y", altcol="altitude",
                        spdcol="speed", lidcol="line_id") -> Vector{Waypoint}

Read a waypoint CSV written by `write_waypoints_csv`. Column names are
configurable. Missing `line_id` column is tolerated (defaults to 0).
"""
function read_waypoints_csv(path::AbstractString;
                             xcol  ::String="x",
                             ycol  ::String="y",
                             altcol::String="altitude",
                             spdcol::String="speed",
                             lidcol::String="line_id")
    df = CSV.read(path, DataFrame)
    has_lid = lidcol in names(df)
    wps = Vector{Waypoint}(undef, nrow(df))
    for (i, row) in enumerate(eachrow(df))
        lid = has_lid ? Int(row[Symbol(lidcol)]) : 0
        wps[i] = Waypoint(row[Symbol(xcol)],
                          row[Symbol(ycol)],
                          row[Symbol(altcol)],
                          row[Symbol(spdcol)];
                          line_id=lid)
    end
    return wps
end

# ---------------------------------------------------------------------------
# Analysis / coverage table CSV
# ---------------------------------------------------------------------------

"""
    write_coverage_csv(path::AbstractString, rows; kwargs...) -> path

Write a coverage-statistics table to CSV. `rows` can be:
- A `DataFrame`
- A `Vector{NamedTuple}` (e.g., from `per_cover_summary`)

Keyword arguments are passed through to `CSV.write`.
"""
function write_coverage_csv(path::AbstractString, rows; kwargs...)
    df = rows isa DataFrame ? rows : DataFrame(rows)
    CSV.write(path, df; kwargs...)
    return path
end

"""
    read_coverage_csv(path::AbstractString; kwargs...) -> DataFrame

Read a coverage CSV into a `DataFrame`. Keyword arguments are passed to
`CSV.read`.
"""
read_coverage_csv(path::AbstractString; kwargs...) =
    CSV.read(path, DataFrame; kwargs...)

# ---------------------------------------------------------------------------
# UGCS-format export (CRS-aware; optional Proj.jl)
# ---------------------------------------------------------------------------

"""
    write_ugcs_csv(path::AbstractString, wps::Vector{Waypoint},
                   src_epsg::Int=6348; tf=nothing) -> path

Write a UGCS-compatible mission CSV with columns:
  Latitude, Longitude, AltitudeAGL, Speed

`src_epsg` is the EPSG code of the grid coordinate system
(default 6348 = NAD83(2011) / UTM Zone 18N, matching the Durham NH site).

`tf` is an optional pre-constructed transformation object. If `nothing`,
this function will attempt to call `Proj.Transformation(...)`. If Proj.jl
is not available, a helpful error is thrown.

For purely local (XY) workflows, use `write_waypoints_csv` instead.
"""
function write_ugcs_csv(path::AbstractString,
                         wps ::Vector{Waypoint},
                         src_epsg::Int=6348;
                         tf=nothing)
    if isnothing(tf)
        # Require Proj.jl to be loaded by the caller
        if !isdefined(Main, :Proj)
            error("""
write_ugcs_csv requires Proj.jl for coordinate reprojection.
Load it with `using Proj` in your session before calling this function,
or provide a pre-constructed transformation via the `tf` keyword argument.
For local-XY workflows, use `write_waypoints_csv` instead.
""")
        end
        tf = Main.Proj.Transformation("EPSG:$src_epsg", "EPSG:4326")
    end

    open(path, "w") do io
        println(io, "Latitude,Longitude,AltitudeAGL,Speed")
        for wp in wps
            lat, lon = tf(wp.x, wp.y)
            println(io, "$(lat),$(lon),$(wp.altitude),$(wp.speed)")
        end
    end
    return path
end

# ---------------------------------------------------------------------------
# Waypoints as plain tuple CSV (backward compatibility)
# ---------------------------------------------------------------------------

"""
    write_xy_speed_csv(path::AbstractString, wps::Vector{Waypoint}) -> path

Write analysis-friendly CSV with columns GridX, GridY, Height, Speed.
Matches the column names used in the original project trajectory logs.
"""
function write_xy_speed_csv(path::AbstractString, wps::Vector{Waypoint})
    open(path, "w") do io
        println(io, "GridX,GridY,Height,Speed")
        for wp in wps
            println(io, "$(wp.x),$(wp.y),$(wp.altitude),$(wp.speed)")
        end
    end
    return path
end
