@testset "Count-grid schemas preserve canonical spatial layout" begin
    # Unique values make a transpose, reshape, or row-flip mistake observable.
    expected = Int32[11 12 13; 21 22 23]
    legacy_values = [11, 21, 12, 22, 13, 23]
    key = CountKey(ret=:ground, cover=:field, mission=:density, kernel=:E)
    tuple_key = ("ground", "field", "density", "E")

    @test decode_count_grid(legacy_values; nrows=2, ncols=3) == expected
    @test_throws DimensionMismatch decode_count_grid(legacy_values; nrows=3, ncols=3)
    @test_throws ArgumentError decode_count_grid(legacy_values; nrows=0, ncols=3)

    mktempdir() do dir
        legacy_path = joinpath(dir, "legacy.json")
        legacy = [Dict(
            "key" => Dict("ret"=>"ground", "cover"=>"field",
                          "mission"=>"density", "kernel"=>"E"),
            "matrix" => legacy_values,
        )]
        open(legacy_path, "w") do io
            JSON.print(io, legacy)
        end
        records = load_counts_json(legacy_path; nrows=2, ncols=3)
        @test length(records) == 1
        @test records[1].key == key
        @test records[1].matrix == expected
        @test eltype(records[1].matrix) == Int32
        @test load_count_grids(legacy_path; nrows=2, ncols=3)[tuple_key] == expected
        @test_throws DimensionMismatch load_counts_json(legacy_path; nrows=3, ncols=3)

        current_path = joinpath(dir, "current.json")
        write_count_grids_json(Dict(key => expected), current_path; nrows=2, ncols=3)
        written = JSON.parsefile(current_path)
        @test written["records"][1]["values"] == [11, 12, 13, 21, 22, 23]
        current_records = load_counts_json(current_path)
        @test length(current_records) == 1
        @test current_records[1].key == key
        @test current_records[1].matrix == expected
        @test load_count_grids(current_path)[tuple_key] == expected
        @test load_counts_json(current_path; nrows=2, ncols=3)[1].matrix == expected
        @test_throws DimensionMismatch load_counts_json(current_path; nrows=3, ncols=2)
        @test_throws DimensionMismatch load_count_grids(current_path; ncols=2)
        @test_throws ArgumentError load_counts_json(current_path; nrows=0)

        written["records"][1]["values"] = [11, 12]
        open(current_path, "w") do io
            JSON.print(io, written)
        end
        @test_throws DimensionMismatch load_counts_json(current_path)
    end
end

@testset "Bundled count-grid loaders agree spatially" begin
    bundled = joinpath(@__DIR__, "..", "data", "ground_truth", "counts.json")
    if isfile(bundled)
        records = load_counts_json(bundled)
        grids = load_count_grids(bundled)
        raw = JSON.parsefile(bundled)
        @test length(records) == length(grids) == length(raw)
        for (record, source) in zip(records, raw)
            key = record.key
            tuple_key = (String(key.ret), String(key.cover), String(key.mission), String(key.kernel))
            @test size(record.matrix) == (263, 324)
            @test grids[tuple_key] == record.matrix
            @test record.matrix == reshape(Int32.(source["matrix"]), 263, 324)
        end
    else
        @test_skip isfile(bundled)
    end
end

@testset "Bundled grids align with original GLI cover support" begin
    # Optional private input supplies independent spatial ground truth. It is
    # intentionally not reconstructed from counts or committed to the repo.
    gli_path = get(ENV, "KDEFLIGHTPLANNING_GLI_CLASS_RASTER", "")
    if isempty(gli_path)
        @test_skip false
    else
        bundled = joinpath(@__DIR__, "..", "data", "ground_truth", "counts.json")
        masks = load_cover_masks(gli_path)
        records = load_counts_json(bundled)
        unions = build_cover_masks(records)
        expected_never_hit = Dict(:field=>638, :decid=>61, :conif=>1)
        for cover in (:field, :decid, :conif)
            @test count(masks[cover]) == DURHAM_COVER_N[cover]
            @test count(masks[cover] .& .!unions[cover]) == expected_never_hit[cover]
        end
        for record in records
            mask = masks[record.key.cover]
            @test all(i -> record.matrix[i] == 0 || mask[i], eachindex(mask))
        end
        # A reflected mask has the same areas but must fail spatial validation.
        flipped = Dict(cover=>reverse(mask; dims=1) for (cover, mask) in masks)
        @test_throws ArgumentError KDEFlightPlanning._bootstrap_cover_support(
            records, [:field, :decid, :conif], (263, 324); cover_masks=flipped)
    end
end
