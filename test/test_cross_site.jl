"""
    test/test_cross_site.jl — Focused tests for the cross-site products workflow

Covers EXACTLY the new lightweight cross-site runner + composer (nothing from
the legacy report pipeline):

  • pure runner helpers (geotransform scaling, digests, cache validity, paths);
  • end-to-end runner on synthetic fixtures (non-TTY, explicit tree_labels):
      – exact expected products written (label/mask/KDE tif + metadata json),
      – ABSENCE of any waypoint / speed / overlay / manifest / report files,
      – GeoTIFF CRS + geotransform preserved, mask 0/1, KDE ⊂ [0,1], k == 2;
  • cross-process determinism (two separate Julia processes → byte-identical
    product rasters by SHA-256);
  • single clustering pass + cache reuse on rerun (clustering skipped);
  • out-of-range configured tree_labels fail (before the KDE stage);
  • composer renders a non-empty 3-column PNG from the persisted products.

Run in isolation (does NOT pull in the 3000-line legacy suite):
    julia --project=. test/test_cross_site.jl
"""

using Test
using KDEFlightPlanning
using JSON
using SHA

const ROOT     = abspath(joinpath(@__DIR__, ".."))
const RUNNER   = joinpath(ROOT, "scripts", "make_cross_site_products.jl")
const FIXTURES = joinpath(ROOT, "scripts", "make_synthetic_cross_site_fixtures.jl")
const COMPOSER = joinpath(ROOT, "scripts", "make_cross_site_panel.jl")

# Same interpreter + project as this test process (cross-process, not in-proc).
juliacmd(args...) =
    Cmd(vcat(Base.julia_cmd().exec, ["--project=$ROOT"], collect(String.(args))))

sha256_hex(path) = open(path, "r") do io; bytes2hex(sha256(io)); end

# Load ONLY the runner's pure helpers into a throwaway module (main() is gated
# by PROGRAM_FILE, so including it does not execute the CLI).
module RunnerHelpers
    include(joinpath(@__DIR__, "..", "scripts", "make_cross_site_products.jl"))
end

# ===========================================================================
# 1. Pure runner helpers (in-process)
# ===========================================================================
@testset "runner helpers: geotransform scaling" begin
    gt = GeoTransform(1000.0, 0.25, 0.0, 5000.0, 0.0, -0.25)
    s  = RunnerHelpers.scaled_geotransform(gt, 4)
    @test s.x_origin == gt.x_origin           # origin unchanged (top-left anchor)
    @test s.y_origin == gt.y_origin
    @test s.dx == gt.dx * 4                    # cell size scales with stride
    @test s.dy == gt.dy * 4
    # stride 1 is the identity.
    id = RunnerHelpers.scaled_geotransform(gt, 1)
    @test (id.dx, id.dy) == (gt.dx, gt.dy)
    # pixel-space fallback transform (no CRS).
    px = RunnerHelpers.pixel_geotransform(60, 2)
    @test px.dx == 2.0 && px.dy == -2.0 && px.y_origin == 60.0
end

@testset "runner helpers: digests are deterministic + sensitive" begin
    d1 = RunnerHelpers.settings_digest(6213, 1200, (2, 4), 1)
    d2 = RunnerHelpers.settings_digest(6213, 1200, (2, 4), 1)
    d3 = RunnerHelpers.settings_digest(6213, 1200, (2, 5), 1)   # k-range changed
    @test d1 == d2
    @test d1 != d3
    mktempdir() do dir
        f = joinpath(dir, "a.bin"); write(f, "hello")
        @test RunnerHelpers.file_digest(f) == bytes2hex(sha256("hello"))
    end
end

@testset "runner helpers: cache_valid" begin
    mktempdir() do dir
        pp = RunnerHelpers.product_paths(dir, "Site A")
        @test basename(pp.dir) == "site_a"
        mkpath(pp.dir)
        # Nothing written yet → invalid.
        @test RunnerHelpers.cache_valid(pp, "src", "set")[1] == false
        # Write label + matching metadata → valid.
        write(pp.label, "x")
        open(pp.metadata, "w") do io
            JSON.print(io, Dict("source_digest" => "src", "settings_digest" => "set"))
        end
        ok, meta = RunnerHelpers.cache_valid(pp, "src", "set")
        @test ok && meta !== nothing
        # Digest mismatch → invalid.
        @test RunnerHelpers.cache_valid(pp, "src", "OTHER")[1] == false
    end
end

@testset "runner helpers: load_defaults" begin
    d = RunnerHelpers.load_defaults(Dict("defaults" => Dict(
        "seed" => 42, "nsample" => 500, "kmedoids_k_range" => [2, 3], "cluster_stride" => 2)))
    @test d.seed == 42 && d.nsample == 500 && d.k_range == (2, 3) && d.cluster_stride == 2
    # Missing defaults → documented fallbacks.
    d0 = RunnerHelpers.load_defaults(Dict{String,Any}())
    @test d0.k_range == (2, 6) && d0.cluster_stride == 1
end

@testset "runner helpers: display-ready PNG previews (non-black + categorical)" begin
    # Categorical raster whose RAW values (ids 1..3) would render near-black in a
    # naive viewer; the styled preview must be bright and per-cluster distinct.
    H, W = 6, 9
    labels = [ (i <= W ÷ 3) ? 1 : (i <= 2W ÷ 3 ? 2 : 3) for j in 1:H, i in 1:W ]
    li = RunnerHelpers.render_label_png(labels, 3)
    @test size(li) == (H, W)
    intensity(c) = (Float64(c.r) + Float64(c.g) + Float64(c.b)) / 3
    @test all(intensity(c) > 0.05 for c in li)              # nothing near-black
    @test length(unique(li)) == 3                            # one colour per cluster
    # The three cluster colours are mutually distinct.
    cols = [ li[1, 1], li[1, W ÷ 3 + 1], li[1, W - 1] ]
    @test length(unique(cols)) == 3
    # nodata id (0) falls back to the neutral background, not a cluster colour.
    l0 = RunnerHelpers.render_label_png([0 1; 2 3], 3)
    @test l0[1, 1] == RunnerHelpers._PREVIEW_BG

    # Binary mask → veg is the orange highlight, background is dark grey; the
    # background is deliberately above the near-black threshold so 0/1 rasters
    # never read as an all-black image.
    veg = [1 0; 0 1]
    mi = RunnerHelpers.render_mask_png(veg)
    @test mi[1, 1] == RunnerHelpers._MASK_VEG_COLOR
    @test mi[1, 2] == RunnerHelpers._PREVIEW_BG
    @test all(intensity(c) > 0.05 for c in mi)

    # KDE preview: fixed [0,1] viridis ramp — low ≠ high, both non-black-ish,
    # and out-of-range / non-finite inputs are clamped rather than throwing.
    ki = RunnerHelpers.render_kde_png([0.0 0.5 1.0; NaN -0.2 1.7])
    @test size(ki) == (2, 3)
    @test ki[1, 1] != ki[1, 3]                               # 0.0 and 1.0 differ
    @test all(isfinite(intensity(c)) for c in ki)
end

# Real user-supplied KDE GeoTIFF (the artifact from the black-output report):
# a valid [0,1] Float64 surface that renders perfectly once styled. Guarded by
# isfile so the suite stays portable when the artifact is absent.
@testset "real uploaded KDE GeoTIFF reads valid + renders non-black" begin
    candidates = filter(isfile, [
        get(ENV, "UPLOADED_KDE_TIF", ""),
        "/home/user/workspace/uploaded_attachments/3812b92b615448a7bc25b567ba0613f7/kde_surface.tif",
    ])
    if isempty(candidates)
        @test_skip true    # artifact not present in this environment
    else
        Z, gt, crs = read_band(first(candidates))
        finite = filter(isfinite, vec(Float64.(Z)))
        @test !isempty(finite)
        @test minimum(finite) ≥ -1e-9 && maximum(finite) ≤ 1 + 1e-9
        @test !isempty(crs)                                  # georeferenced
        @test gt.dx != 0                                     # real geotransform
        img = RunnerHelpers.render_kde_png(Float64.(Z))
        intensity(c) = (Float64(c.r) + Float64(c.g) + Float64(c.b)) / 3
        @test count(c -> intensity(c) > 0.05, img) / length(img) > 0.5
    end
end

# ===========================================================================
# 2. End-to-end runner + composer (subprocess; cross-process determinism)
# ===========================================================================
@testset "cross-site runner + composer (end-to-end)" begin
    dir1 = mktempdir()
    dir2 = mktempdir()

    # Generate deterministic synthetic fixtures ONCE, copy to a second tree so
    # both runner invocations read byte-identical inputs.
    run(juliacmd(FIXTURES, dir1, "20260716"))
    for f in ("site_a.tif", "site_b.tif", "site_c.tif", "products_config.toml")
        cp(joinpath(dir1, f), joinpath(dir2, f); force = true)
    end
    cfg1 = joinpath(dir1, "products_config.toml")
    cfg2 = joinpath(dir2, "products_config.toml")

    run(juliacmd(RUNNER, cfg1))
    run(juliacmd(RUNNER, cfg2))

    prod1 = joinpath(dir1, "products")
    prod2 = joinpath(dir2, "products")
    slugs = ["site_a", "site_b", "site_c"]

    @testset "exact expected products present" begin
        for s in slugs
            sd = joinpath(prod1, s)
            @test isdir(sd)
            @test Set(readdir(sd)) == Set([
                "label_clusters.tif", "vegetation_mask.tif", "kde_surface.tif",
                "label_clusters.png", "vegetation_mask.png", "kde_surface.png",
                "products_metadata.json"])
        end
    end

    @testset "no waypoint / speed / overlay / manifest / report files" begin
        for (root, _, files) in walkdir(prod1), f in files
            fl = lowercase(f)
            for bad in ("waypoint", "speed", "overlay", "manifest",
                        "histogram", "report", ".csv")
                @test !occursin(bad, fl)
            end
        end
    end

    @testset "raster products: CRS, transform, dtypes, ranges" begin
        for s in slugs
            sd   = joinpath(prod1, s)
            meta = JSON.parsefile(joinpath(sd, "products_metadata.json"))
            @test meta["k"] == 2
            @test meta["tree_labels"] == [1]
            @test meta["tree_labels_source"] == "config"
            @test meta["kde_normalization"] == "minmax"

            # Label raster: CRS preserved, geotransform matches metadata.
            Zl, gtl, crs = read_band(joinpath(sd, "label_clusters.tif"))
            @test !isempty(crs)                                   # EPSG:6348 WKT
            gm = meta["geotransform"]
            @test gtl.x_origin ≈ gm[1] && gtl.dx ≈ gm[2] && gtl.dy ≈ gm[6]
            @test Set(unique(vec(Int.(Zl)))) ⊆ Set([1, 2])        # k = 2 labels

            # Mask raster: strictly 0/1.
            Zm, _, _ = read_band(joinpath(sd, "vegetation_mask.tif"))
            @test Set(unique(vec(Int.(Zm)))) ⊆ Set([0, 1])
            @test any(==(1), Int.(Zm))                            # selected veg present

            # KDE raster: normalised into [0, 1].
            Zk, _, _ = read_band(joinpath(sd, "kde_surface.tif"))
            finite = filter(isfinite, vec(Float64.(Zk)))
            @test minimum(finite) ≥ -1e-9 && maximum(finite) ≤ 1 + 1e-9
            @test isapprox(maximum(finite), 1.0; atol = 1e-6)     # min-max hits 1
        end
    end

    @testset "display-ready PNG previews written, non-black, in metadata" begin
        using FileIO
        intensity(c) = (Float64(c.r) + Float64(c.g) + Float64(c.b)) / 3
        for s in slugs
            sd   = joinpath(prod1, s)
            meta = JSON.parsefile(joinpath(sd, "products_metadata.json"))
            # Previews are recorded in metadata alongside the scientific rasters.
            @test meta["previews"]["label_clusters"]  == "label_clusters.png"
            @test meta["previews"]["vegetation_mask"] == "vegetation_mask.png"
            @test meta["previews"]["kde_surface"]     == "kde_surface.png"
            for png in ("label_clusters.png", "vegetation_mask.png", "kde_surface.png")
                p = joinpath(sd, png)
                @test isfile(p)
                img = FileIO.load(p)
                # A truthful preview is NOT a near-black image (the raw
                # categorical rasters would be); require a bright majority.
                bright = count(c -> intensity(c) > 0.05, img) / length(img)
                @test bright > 0.5
            end
            # KDE preview shows real spread (not a flat single colour).
            @test length(unique(FileIO.load(joinpath(sd, "kde_surface.png")))) > 5
        end
    end

    @testset "cross-process determinism (SHA-256)" begin
        for s in slugs, f in ("label_clusters.tif", "vegetation_mask.tif", "kde_surface.tif")
            @test sha256_hex(joinpath(prod1, s, f)) == sha256_hex(joinpath(prod2, s, f))
        end
    end

    @testset "cache reuse: rerun skips clustering" begin
        out = read(juliacmd(RUNNER, cfg1), String)
        @test occursin("[cache]", out)                           # cluster labels reused
        # Products still byte-identical after a cache-hit rerun.
        for s in slugs
            @test sha256_hex(joinpath(prod1, s, "label_clusters.tif")) ==
                  sha256_hex(joinpath(prod2, s, "label_clusters.tif"))
        end
    end

    @testset "out-of-range configured tree_labels fail" begin
        dbad = mktempdir()
        for f in ("site_a.tif", "site_b.tif", "site_c.tif")
            cp(joinpath(dir1, f), joinpath(dbad, f); force = true)
        end
        badcfg = joinpath(dbad, "bad.toml")
        write(badcfg, """
        out_dir = "products"
        [defaults]
        seed = 6213
        nsample = 1200
        kmedoids_k_range = [2, 4]
        [[site]]
        name = "Site A"
        geotiff = "site_a.tif"
        tree_labels = [99]
        """)
        # Runner must exit nonzero (validation error before the KDE stage).
        @test !success(juliacmd(RUNNER, badcfg))
    end

    @testset "composer renders 3-column PNG" begin
        panelcfg = joinpath(dir1, "panel.toml")
        write(panelcfg, """
        products_dir = "products"
        kde_colormap = "viridis"
        mask_color   = "#FF6D00"
        mask_alpha   = 0.45
        """)
        out_png = joinpath(dir1, "panel.png")
        run(juliacmd(COMPOSER, panelcfg, out_png))
        @test isfile(out_png)
        @test filesize(out_png) > 5_000                          # non-trivial image
    end
end
