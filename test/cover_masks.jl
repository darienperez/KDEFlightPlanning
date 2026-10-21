using Test
using ArchGDAL
using KDEFlightPlanning: load_cover_masks

@testset "Exact categorical cover masks" begin
    mktempdir() do dir
        function write_class_fixture(name, classes; nodata=nothing, validity=nothing)
            fixture_path = joinpath(dir, name * ".tif")
            h, w = size(classes)
            ArchGDAL.create(fixture_path; driver=ArchGDAL.getdriver("GTiff"),
                            width=w, height=h, nbands=1,
                            dtype=eltype(classes)) do ds
                ArchGDAL.setgeotransform!(ds, [100.0, 1.0, 0.0, 200.0, 0.0, -1.0])
                ArchGDAL.write!(ds, permutedims(classes), 1)
                band = ArchGDAL.getband(ds, 1)
                if nodata !== nothing
                    ArchGDAL.setnodatavalue!(band, Float64(nodata))
                end
                if validity !== nothing
                    # GMF_PER_DATASET also supports GeoTIFF's internal-mask
                    # default in recent GDAL releases.
                    ArchGDAL.createmaskband!(band, 2)
                    ArchGDAL.write!(ArchGDAL.getmaskband(band),
                                   permutedims(UInt8.(validity) .* 0xff))
                end
            end
            return fixture_path
        end

        # Asymmetry makes an accidental transpose or row reversal detectable.
        classes = UInt8[0 1 2 255;
                        2 0 1   2;
                        1 2 0   0]
        path = write_class_fixture("classes", classes; nodata=255)
        masks = load_cover_masks(path; nrows=3, ncols=4)
        @test masks isa Dict{Symbol,BitMatrix}
        @test Set(keys(masks)) == Set([:field, :decid, :conif])
        @test masks[:decid] == (classes .== 0)
        @test masks[:conif] == (classes .== 1)
        @test masks[:field] == (classes .== 2)
        @test count(masks[:decid]) == 4  # valid class zero must survive
        @test !any(mask[1, 4] for mask in values(masks))
        @test sum(count, values(masks)) == length(classes) - 1

        remapped = load_cover_masks(path; nrows=3, ncols=4,
                                    class_codes=Dict(:field=>0, :decid=>2, :conif=>1))
        @test remapped[:field] == masks[:decid]
        @test remapped[:decid] == masks[:field]
        @test remapped[:conif] == masks[:conif]

        default_classes = fill(UInt8(2), 263, 324)
        default_classes[1, 1] = 0
        default_classes[end, end] = 1
        default_path = write_class_fixture("durham_shape", default_classes)
        default_masks = load_cover_masks(default_path)
        @test size(default_masks[:field]) == (263, 324)
        @test default_masks[:decid][1, 1]
        @test default_masks[:conif][end, end]

        flipped = load_cover_masks(path; nrows=3, ncols=4, flip_y=true)
        for cover in keys(masks)
            @test size(flipped[cover]) == (3, 4)
            @test flipped[cover] == reverse(masks[cover]; dims=1)
        end

        # An explicit GDAL mask can invalidate a pixel whose raw category is
        # otherwise a valid cover code, independently of NoData metadata.
        validity = trues(size(classes))
        validity[1, 1] = false
        masked_path = write_class_fixture("validity", classes;
                                          nodata=255, validity=validity)
        masked = load_cover_masks(masked_path; nrows=3, ncols=4)
        @test !masked[:decid][1, 1]
        @test count(masked[:decid]) == 3
        @test masked[:conif] == masks[:conif]
        @test !any(mask[1, 4] for mask in values(masked))

        conflict_path = write_class_fixture("conflict", classes; nodata=0)
        @test_throws ArgumentError load_cover_masks(conflict_path; nrows=3, ncols=4)
        @test_throws DimensionMismatch load_cover_masks(path; nrows=4, ncols=3)
        @test_throws ArgumentError load_cover_masks(path; nrows=0, ncols=4)
        @test_throws ArgumentError load_cover_masks(path; nrows=3, ncols=4,
                                                    class_codes=Dict(:field=>2, :decid=>2))
        @test_throws ArgumentError load_cover_masks(path; nrows=3, ncols=4,
                                                    class_codes=Dict(:field=>2.0))

        fractional_path = write_class_fixture("fractional", Float32[0.5 1; 2 0])
        @test_throws ArgumentError load_cover_masks(fractional_path; nrows=2, ncols=2)
        @test_throws ArgumentError load_cover_masks(joinpath(dir, "missing.tif"))
    end
end
