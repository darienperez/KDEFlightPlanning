using Test
using DataFrames
using KDEFlightPlanning

@testset "Line scans use true north-up raster geometry and exact support" begin
    # Deliberately unequal resolutions and asymmetric north/south data expose
    # both the former row reversal and a mistaken planned-line x origin.
    gt = GeoTransform(100.0, 2.0, 0.0, 200.0, 0.0, -3.0)
    shape = (6, 8)
    masks = Dict(cover => falses(shape) for cover in ("field", "decid", "conif"))
    masks["field"][[1, 6], 2:3] .= true
    masks["decid"][[1, 6], 4:5] .= true
    masks["conif"][[1, 6], 6:7] .= true
    field = zeros(Int32, shape)
    field[1, 2:3] .= 1
    field[6, 3] = 1
    decid = zeros(Int32, shape)
    decid[1, 4] = 1
    decid[6, 4:5] .= 1
    conif = zeros(Int32, shape)
    conif[1, 6] = 1
    conif[6, 7] = 1
    matrices = Dict("field" => field, "decid" => decid, "conif" => conif)
    grids = Dict{NTuple{4,String},Matrix{Float64}}(
        (ret, cover, "density", "E") => Float64.(matrices[cover])
        for ret in ("all", "ground"), cover in ("field", "decid", "conif")
    )
    # Line numbering follows the existing south-to-north convention, while
    # matrix rows must run north to south.
    planned = DataFrame(line_index=[0, 1], line_number=[1, 2],
        y_center=[183.5, 198.5], x_min=[102.2, 102.2],
        x_max=[113.8, 113.8], n=[20, 20])
    trajectory = DataFrame(
        GridX=repeat(collect(range(102.2, 113.8; length=20)), 2),
        GridY=vcat(fill(183.5, 20), fill(198.5, 20)),
        nearest_line=vcat(fill(0, 20), fill(1, 20)),
        segment_id=vcat(fill(0, 20), fill(1, 20)),
    )
    processed = Dict("KDE-guided" => trajectory)
    missions = [("KDE-guided", "density", "E")]
    options = (geotransform=gt, half_band_rows=0, cover_masks=masks)

    scan, cover, wide, profile = line_scan_cover_metrics(
        grids, processed, planned, missions; options...)
    @test scan.row_center == [6, 1]
    field_rows = filter(row -> row.cover == "Field", cover)
    @test field_rows.line_number == [1, 2]
    @test field_rows.support_cells == [2, 2]
    @test field_rows.covered_cells == [1, 2]
    @test field_rows.coverage_ratio == [0.5, 1.0]
    @test sort(unique(profile.x_m)) == [103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
    @test sort(unique(profile.column_index)) == collect(1:6)
    @test all(all(table.support_mode .== "exact") for table in (scan, cover, wide, profile))

    extents, south, south_profile = single_line_cover_metrics(
        grids, processed, planned, missions, 1; options...)
    @test only(filter(row -> row.cover == "Field", south).coverage_ratio) == 0.5
    @test sort(unique(south_profile.x_m)) == [103.0, 105.0, 107.0, 109.0, 111.0, 113.0]
    @test all(all(table.support_mode .== "exact") for table in (extents, south, south_profile))
    _, north, _ = single_line_cover_metrics(
        grids, processed, planned, missions, 2; options...)
    @test only(filter(row -> row.cover == "Field", north).coverage_ratio) == 1.0

    # Exact support includes zero-return cells and is required by both APIs.
    @test_throws ArgumentError line_scan_cover_metrics(
        grids, processed, planned, missions; geotransform=gt)
    @test_throws ArgumentError single_line_cover_metrics(
        grids, processed, planned, missions, 1; geotransform=gt)
    wrong_masks = deepcopy(masks)
    wrong_masks["field"][1, 2] = false
    @test_throws ArgumentError line_scan_cover_metrics(
        grids, processed, planned, missions; geotransform=gt, cover_masks=wrong_masks)
    @test_throws ArgumentError single_line_cover_metrics(
        grids, processed, planned, missions, 1; geotransform=gt, cover_masks=wrong_masks)
    overlap_masks = deepcopy(masks)
    overlap_masks["decid"][1, 2] = true
    @test_throws ArgumentError KDEFlightPlanning._line_scan_cover_support(
        grids; cover_masks=overlap_masks)
    validated, mode, grid_size = KDEFlightPlanning._line_scan_cover_support(
        grids; cover_masks=Dict(Symbol(k) => v for (k, v) in masks))
    @test validated == masks
    @test mode == "exact"
    @test grid_size == shape

    diagnostic = @test_logs (:warn, r"approximate positive-return-union") line_scan_cover_metrics(
        grids, processed, planned, missions; geotransform=gt,
        half_band_rows=0, allow_approximate_masks=true)
    @test all(all(table.support_mode .== "approximate_positive_union") for table in diagnostic)
    diagnostic_single = @test_logs (:warn, r"approximate positive-return-union") single_line_cover_metrics(
        grids, processed, planned, missions, 1; geotransform=gt,
        half_band_rows=0, allow_approximate_masks=true)
    @test all(all(table.support_mode .== "approximate_positive_union") for table in diagnostic_single)

    # Non-Durham geometry is explicit, and raster edges are never reconstructed
    # from the first or last planned line.
    @test_throws ArgumentError KDEFlightPlanning._line_scan_geotransform(shape)
    @test KDEFlightPlanning._line_scan_row(198.5, gt) == 1
    @test KDEFlightPlanning._line_scan_row(183.5, gt) == 6
    @test KDEFlightPlanning._line_scan_row(181.9, gt) == 7
    @test KDEFlightPlanning._line_scan_band(181.9, gt, 6, 1) == (7, 6, 6)
    @test_throws ArgumentError KDEFlightPlanning._line_scan_geotransform(shape;
        geotransform=GeoTransform(100.0, 2.0, 1.0, 200.0, 0.0, -3.0))
    @test_throws ArgumentError KDEFlightPlanning._line_scan_geotransform(shape;
        geotransform=GeoTransform(100.0, 2.0, 0.0, 200.0, 0.0, 3.0))
    @test_throws ArgumentError line_scan_cover_metrics(
        grids, processed, planned, missions; geotransform=gt,
        cover_masks=masks, half_band_rows=-1)
end
