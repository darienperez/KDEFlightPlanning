using Test
using Random
using JSON
using KDEFlightPlanning

# Include cells that received no returns in either mission. Legacy count arrays
# are flattened in column-major order with north-first rows. The fixture is
# deliberately nonsquare, so exact-mask tests exercise the north-first decoder.
function _write_bootstrap_mask_fixture(path, a, b)
    records = [
        Dict("key" => Dict("ret" => "ground", "cover" => "field",
                           "mission" => "density", "kernel" => "E"),
             "matrix" => vec(a)),
        Dict("key" => Dict("ret" => "ground", "cover" => "field",
                           "mission" => "const2", "kernel" => "NA"),
             "matrix" => vec(b)),
    ]
    open(path, "w") do io
        JSON.print(io, records)
    end
    return path
end

@testset "Bootstrap tables retain exact zero-return support" begin
    mktempdir() do dir
        mask = falses(4, 5)
        mask[1:3, 1:4] .= true
        a = zeros(Int32, size(mask))
        b = zeros(Int32, size(mask))
        a[1, 1] = 2
        a[2, 3] = 1
        a[3, 4] = 4
        b[1, 1] = 1
        path = _write_bootstrap_mask_fixture(joinpath(dir, "counts.json"), a, b)
        masks = Dict(:field => mask)
        ka = CountKey(ret=:ground, cover=:field, mission=:density, kernel=:E)
        kb = CountKey(ret=:ground, cover=:field, mission=:const2, kernel=:NA)
        comparisons = [Comparison(ka, kb, "KDE versus constant")]
        common = (nrows=4, ncols=5, nboot=200, block_frac=0.4, seed=19)
        selection = (covers=[:field], returns=[:ground],
                     missions=[:density, :const2], kernels=[:E])

        @test count(mask) == 12
        @test count((a .== 0) .& (b .== 0) .& mask) == 9
        @test_throws ArgumentError bootstrap_cr_table(path; common..., selection...)
        @test_throws ArgumentError bootstrap_cr_difference_table(path, comparisons; common...)

        table = bootstrap_cr_table(path; common..., selection..., cover_masks=masks,
                                   cover_n=Dict(:field => 12))
        kde = only(eachrow(table[table.Mission .== "Density-aware", :]))
        constant = only(eachrow(table[table.Mission .== "Const 2 m/s", :]))
        direct = bootstrap_cr(a, mask; nboot=200, block_frac=0.4, seed=19)
        @test kde.CR == 3 / 12
        @test constant.CR == 1 / 12
        @test all(table.N .== 12)
        @test all(table.support_N .== 12)
        @test all(table.support_mode .== "exact")
        @test kde.CR_bootstrap == direct.estimate == kde.CR
        @test kde.CI_lower == direct.lower
        @test kde.CI_upper == direct.upper

        differences = bootstrap_cr_difference_table(path, comparisons;
            common..., cover_masks=masks, cover_n=Dict(:field => 12))
        difference = only(eachrow(differences))
        paired = bootstrap_cr_difference(a, b, mask;
            nboot=200, block_frac=0.4, seed=19)
        @test difference.CR_A == 3 / 12
        @test difference.CR_B == 1 / 12
        @test difference.CR_diff == 2 / 12
        @test difference.CR_diff_bootstrap == paired.estimate == difference.CR_diff
        @test difference.CI_lower == paired.lower
        @test difference.CI_upper == paired.upper
        @test difference.N == difference.support_N == 12
        @test difference.support_mode == "exact"

        diagnostic = @test_logs (:warn, r"approximate positive-return-union") bootstrap_cr_table(
            path; common..., selection..., allow_approximate_masks=true)
        approximate_kde = only(eachrow(diagnostic[diagnostic.Mission .== "Density-aware", :]))
        @test approximate_kde.N == 3
        @test approximate_kde.CR == 1.0
        @test all(diagnostic.support_mode .== "approximate_positive_union")
        diagnostic_pair = @test_logs (:warn, r"approximate positive-return-union") bootstrap_cr_difference_table(
            path, comparisons; common..., allow_approximate_masks=true)
        @test only(diagnostic_pair.CR_diff) == 2 / 3
        @test only(diagnostic_pair.support_N) == 3
        @test only(diagnostic_pair.support_mode) == "approximate_positive_union"

        # Totals cannot be used to invent the nine missing spatial cells.
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., allow_approximate_masks=true,
            cover_n=Dict(:field => 12))
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=masks, cover_n=Dict(:field => 13))

        @test_throws DimensionMismatch bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:field => trues(5, 4)))
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:unknown => mask))
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:field => Int.(mask)))
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:conif => mask))
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:field => falses(size(mask))))
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:field => mask, :decid => copy(mask)))
        cropped = copy(mask)
        cropped[3, 4] = false
        @test_throws ArgumentError bootstrap_cr_table(path;
            common..., selection..., cover_masks=Dict(:field => cropped))
        @test_throws ArgumentError bootstrap_cr_difference_table(path, comparisons;
            common..., cover_masks=Dict(:field => cropped))

        conif = falses(size(mask))
        conif[4, 5] = true
        cross_cover = [Comparison(ka,
            CountKey(ret=:ground, cover=:conif, mission=:const2, kernel=:NA), "invalid")]
        @test_throws ArgumentError bootstrap_cr_difference_table(path, cross_cover;
            common..., cover_masks=Dict(:field => mask, :conif => conif))
    end
end

@testset "Block samples reach the retained cover-cell target" begin
    # A sparse cover spans two cells in a large raster, with many empty blocks.
    mask = falses(40, 50)
    mask[2, 3] = true
    mask[39, 49] = true
    side = 4
    origins, sizes = KDEFlightPlanning._mask_block_catalog(mask, side)
    @test length(origins) == 2
    @test sizes == [1, 1]
    @test sum(sizes) == count(mask)
    rng = MersenneTwister(42)
    for _ in 1:50
        sample = KDEFlightPlanning._sample_mask_blocks(rng, sizes, count(mask))
        retained = sum(sizes[sample])
        @test retained >= count(mask)
        @test retained - sizes[last(sample)] < count(mask)
    end

    # Unequal boundary intersections also reach the target with minimal prefix.
    irregular = falses(9, 11)
    irregular[1:5, 2:7] .= true
    irregular[9, 11] = true
    _, clipped_sizes = KDEFlightPlanning._mask_block_catalog(irregular, 4)
    @test sum(clipped_sizes) == count(irregular)
    for _ in 1:50
        sample = KDEFlightPlanning._sample_mask_blocks(rng, clipped_sizes, count(irregular))
        retained = sum(clipped_sizes[sample])
        @test retained >= count(irregular)
        @test retained - clipped_sizes[last(sample)] < count(irregular)
    end

    a = Int.(mask)
    b = zeros(Int, size(mask))
    ci = bootstrap_cr(a, mask; nboot=100, block_frac=0.1, seed=42)
    pair = bootstrap_cr_difference(a, b, mask; nboot=100, block_frac=0.1, seed=42)
    repeated = bootstrap_cr_difference(a, b, mask; nboot=100, block_frac=0.1, seed=42)
    @test all(isfinite, ci.replicates)
    @test all(isfinite, pair.replicates)
    @test all(ci.replicates .== 1.0)
    @test all(pair.replicates .== 1.0)
    @test pair.replicates == repeated.replicates
    @test ci.lower == ci.upper == pair.lower == pair.upper == 1.0
end
