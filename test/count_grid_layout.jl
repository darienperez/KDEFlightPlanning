@testset "Count-grid schemas preserve canonical spatial layout" begin
    # Unique values make a transpose, reshape, or row-flip mistake observable.
    expected = Int32[11 12 13; 21 22 23]
    legacy_values = [21, 11, 22, 12, 23, 13]
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
            @test record.matrix == reverse(reshape(Int32.(source["matrix"]), 263, 324); dims=1)
        end
    else
        @test_skip isfile(bundled)
    end
end
