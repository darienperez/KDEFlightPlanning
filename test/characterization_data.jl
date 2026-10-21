@testset "Characterization CSV mission mapping and conditional means" begin
    stats_path = joinpath(@__DIR__, "..", "data", "ground_truth", "stats_and_coverage.csv")
    stats = CSV.read(stats_path, DataFrame)
    source_names = names(stats)
    source_means = copy(stats.Mean)

    primary = KDEFlightPlanning.characterization_stats(stats)
    expected = Set([(:const2, :NA), (:density, :E), (:const8, :NA)])
    @test nrow(primary) == 18  # three covers × two return types × three missions
    @test Set(zip(primary.mission_key, primary.kernel_key)) == expected
    @test all(count(==(mission), primary.mission_key) == 6
              for mission in (:const2, :density, :const8))
    for cover in ("Field", "Deciduous", "Coniferous"), ret in ("Ground", "All")
        panel_rows = filter(row -> row.Cover == cover && row.Return == ret, primary)
        @test Set(zip(panel_rows.mission_key, panel_rows.kernel_key)) == expected
    end
    @test Set(primary.display_label) == Set([
        "Const. 2 m/s", "KDE-guided (Epanechnikov)", "Const. 8 m/s",
    ])

    # These variants exist in the input but do not belong to the primary figures.
    @test any(==("G"), stats.Kernel)
    @test any(==("Speed-aware"), stats.Mission)
    @test !any(==("G"), primary.Kernel)
    @test !any(==("Speed-aware"), primary.Mission)
    @test all(row -> row.Mission == mission_label(row.mission_key) &&
                     row.Kernel == kernel_label(row.kernel_key), eachrow(primary))

    # Mean is conditional on a cell being covered in this CSV. Coverage below
    # one must not inflate it through a second division by CR.
    @test primary.cond_mean == primary.Mean
    @test any(primary.CR .< 1)
    @test any(primary.cond_mean .!= primary.Mean ./ primary.CR)
    @test names(stats) == source_names
    @test stats.Mean == source_means
end
