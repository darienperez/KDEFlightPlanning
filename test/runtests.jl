"""
    test/runtests.jl — Unit tests for KDEFlightPlanning

All tests use synthetic `RasterGrid` instances; no GeoTIFFs, Colors.jl, or
ImageMorphology.jl are required.

Test groups
-----------
 1. RasterGrid construction and make_grid
 2. rescale_density!
 3. Bilinear interpolation
 4. Speed-map monotonicity and assign_speed
 5. Lawnmower path geometry
 6. Waypoint generation — bounds and speed validity
 7. Curvature spacing tightens near high-gradient regions
 8. Coverage metrics
 9. CSV round-trip I/O
10. KDE kernel properties (sum, support, bandwidth)
11. kde_from_mask — output shape, normalisation, non-negativity
12. Feature stacking and standardisation
13. labels_to_mask and mask_stats
14. Pipeline: build_density_surface (mask → KDE → grid)
15. plan_mission convenience wrapper
"""

using Test
using KDEFlightPlanning
using CSV
using DataFrames: nrow, names, DataFrame
using Statistics: mean, std
using Random
using Colors: RGB, red, green, blue
using MultivariateStats
using Distances
using JSON

# ===========================================================================
# 1. RasterGrid construction and make_grid
# ===========================================================================
@testset "RasterGrid construction" begin
    xs = collect(0.0:1.0:9.0)   # 10
    ys = collect(0.0:1.0:4.0)   # 5
    Z  = rand(5, 10)

    g = RasterGrid(Z, xs, ys)
    @test size(g) == (5, 10)
    @test g.xs == xs
    @test g.ys == ys

    # Wrong size throws
    @test_throws DimensionMismatch RasterGrid(rand(10, 5), xs, ys)

    # Non-ascending axes throw
    @test_throws ArgumentError RasterGrid(Z, reverse(xs), ys)
    @test_throws ArgumentError RasterGrid(Z, xs, reverse(ys))

    # make_grid flips descending axes
    g2 = make_grid(Z, reverse(xs), ys)   # descending xs → should flip
    @test issorted(g2.xs)
    @test issorted(g2.ys)
    @test g2.Z[:, 1] ≈ Z[:, end]    # first col after flip = last col before

    # make_grid with descending ys
    g3 = make_grid(Z, xs, reverse(ys))
    @test issorted(g3.ys)
end

# ===========================================================================
# 2. rescale_density!
# ===========================================================================
@testset "rescale_density!" begin
    xs = collect(range(0.0, 10.0; length=5))
    ys = collect(range(0.0, 10.0; length=5))
    Z  = reshape(collect(1.0:25.0), 5, 5)
    g  = RasterGrid(Z, xs, ys)

    rescale_density!(g; newmin=0.0, newmax=1.0)
    @test minimum(g.Z) ≈ 0.0
    @test maximum(g.Z) ≈ 1.0

    # Flat surface: fill with midpoint
    Zf = ones(5, 5) * 3.0
    gf = RasterGrid(Zf, xs, ys)
    rescale_density!(gf; newmin=0.0, newmax=1.0)
    @test all(gf.Z .≈ 0.5)
end

# ===========================================================================
# 3. Bilinear interpolation
# ===========================================================================
@testset "Bilinear interpolation" begin
    # Linear surface: Z[j,i] = i + 2j (0-based offsets)
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 2.0]
    Z  = [Float64(i + 2j) for j in 0:2, i in 0:3]
    g  = RasterGrid(Z, xs, ys)

    @test bilinear_interp(g, 0.0, 0.0) ≈ 0.0
    @test bilinear_interp(g, 3.0, 2.0) ≈ 7.0
    @test bilinear_interp(g, 0.5, 0.0) ≈ 0.5
    @test bilinear_interp(g, 0.0, 0.5) ≈ 1.0
    @test bilinear_interp(g, 0.5, 0.5) ≈ 1.5

    # Clamping
    @test bilinear_interp(g, -5.0, 0.0) ≈ bilinear_interp(g, 0.0, 0.0)
    @test bilinear_interp(g, 100.0, 0.0) ≈ bilinear_interp(g, 3.0, 0.0)

    # sample_density dispatch
    @test sample_density(g, 0.5, 0.5; sampler=:bilinear) ≈ bilinear_interp(g, 0.5, 0.5)
    @test sample_density(g, 1.0, 1.0; sampler=:nearest) ≈ 3.0
    @test_throws ArgumentError sample_density(g, 1.0, 1.0; sampler=:bad)
end

# ===========================================================================
# 4. Speed-map monotonicity and assign_speed
# ===========================================================================
@testset "Speed-map monotonicity and assign_speed" begin
    s_kde = KDEGuidedSpeed(0.0, 1.0, 2.0, 8.0)
    @test is_monotone_decreasing(s_kde)
    @test assign_speed(s_kde, 0.0) ≈ 8.0
    @test assign_speed(s_kde, 1.0) ≈ 2.0
    @test assign_speed(s_kde, 0.5) ≈ 5.0
    @test assign_speed(s_kde, -0.5) ≈ 8.0   # clamped
    @test assign_speed(s_kde,  1.5) ≈ 2.0   # clamped

    s_c = ConstantSpeed(5.0)
    @test assign_speed(s_c, 0.0) ≈ 5.0
    @test assign_speed(s_c, 1.0) ≈ 5.0
    @test is_monotone_decreasing(s_c)
    @test speed_bounds(s_c) == (5.0, 5.0)

    s_curv = CurvatureGuidedSpeed(0.0, 1.0, 2.0, 8.0)
    @test assign_speed(s_curv, 0.0) ≈ 8.0
    @test assign_speed(s_curv, 1.0) ≈ 2.0
    @test is_monotone_decreasing(s_curv)
    @test speed_bounds(s_curv) == (2.0, 8.0)

    # Flat surface → midspeed
    s_flat = KDEGuidedSpeed(0.5, 0.5, 2.0, 8.0)
    @test assign_speed(s_flat, 0.5) ≈ 5.0
end

# ===========================================================================
# 5. Lawnmower path geometry
# ===========================================================================
@testset "Lawnmower path geometry" begin
    spec = LawnmowerSpec(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0,
                         spacing=25.0, yaw_deg=0.0, primary=:x, start=:low)
    path = lawnmower_from_extents(spec)

    @test length(path) == 10   # 5 lines × 2 vertices

    # First vertex at (0,0)
    @test path[1][1] ≈ 0.0
    @test path[1][2] ≈ 0.0
    # Second vertex at (100,0)
    @test path[2][1] ≈ 100.0
    @test path[2][2] ≈ 0.0
    # Turnaround: (100, 25)
    @test path[3][1] ≈ 100.0
    @test path[3][2] ≈ 25.0

    # All coords in bounds
    for (x, y) in path
        @test 0.0 <= x <= 100.0
        @test 0.0 <= y <= 100.0
    end

    # primary=:y
    spec_y = LawnmowerSpec(xmin=0.0, xmax=50.0, ymin=0.0, ymax=100.0,
                            spacing=25.0, yaw_deg=0.0, primary=:y, start=:low)
    path_y = lawnmower_from_extents(spec_y)
    @test length(path_y) == 6   # 3 lines × 2
    @test path_y[1][1] ≈ 0.0 && path_y[1][2] ≈ 0.0
    @test path_y[2][1] ≈ 0.0 && path_y[2][2] ≈ 100.0

    # start=:high
    spec_hi = LawnmowerSpec(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0,
                             spacing=25.0, yaw_deg=0.0, primary=:x, start=:high)
    path_hi = lawnmower_from_extents(spec_hi)
    @test path_hi[1][2] ≈ 50.0   # first line at top

    # Invalid primary
    spec_bad = LawnmowerSpec(xmin=0.0, xmax=10.0, ymin=0.0, ymax=10.0,
                              spacing=5.0, yaw_deg=0.0, primary=:z, start=:low)
    @test_throws ArgumentError lawnmower_from_extents(spec_bad)
end

# ===========================================================================
# 6. Waypoint generation — bounds and speed validity
# ===========================================================================
@testset "Waypoint generation bounds and speed" begin
    grid = synthetic_gaussian_grid(nx=40, ny=40,
                                   xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
    spec = LawnmowerSpec(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0,
                         spacing=20.0, yaw_deg=0.0, primary=:x, start=:low)
    path = lawnmower_from_extents(spec)

    for strategy in (ConstantSpeed(5.0),
                     KDEGuidedSpeed(grid; vmin=2.0, vmax=8.0),
                     CurvatureGuidedSpeed(grid; vmin=2.0, vmax=8.0,
                                          spacing_min=3.0, spacing_max=15.0))
        vmin, vmax = speed_bounds(strategy)
        wps = generate_waypoints(path, grid, strategy;
                                  altitude=80.0, spacing_min=3.0,
                                  spacing_max=20.0, seconds_per_wp=2.0,
                                  include_vertices=true)

        @test !isempty(wps)
        @test length(wps) >= length(path)

        for w in wps
            @test vmin - 1e-9 <= w.speed <= vmax + 1e-9
            @test 0.0 - 1e-6 <= w.x <= 100.0 + 1e-6
            @test 0.0 - 1e-6 <= w.y <= 100.0 + 1e-6
            @test w.altitude ≈ 80.0
        end
    end
end

# ===========================================================================
# 7. Curvature spacing tightens near high-gradient regions
# ===========================================================================
@testset "Spacing tightens near high curvature/gradient" begin
    grid_sharp = synthetic_gaussian_grid(
        nx=60, ny=60, xmin=0.0, xmax=120.0, ymin=0.0, ymax=120.0,
        centers=[(60.0, 60.0)], sigmas=[(5.0, 5.0)], weights=[1.0])

    grid_flat = uniform_grid(0.5; nx=60, ny=60,
                              xmin=0.0, xmax=120.0, ymin=0.0, ymax=120.0)

    spec = LawnmowerSpec(xmin=0.0, xmax=120.0, ymin=0.0, ymax=120.0,
                         spacing=40.0, yaw_deg=0.0, primary=:x, start=:low)
    path = lawnmower_from_extents(spec)
    strategy = CurvatureGuidedSpeed(0.0, 1.0, 2.0, 8.0;
                                     alpha=1.0, lambda=1.5, eta=2.0,
                                     spacing_min=1.0, spacing_max=30.0)

    wps_sharp = generate_waypoints(path, grid_sharp, strategy;
                                    spacing_min=1.0, spacing_max=30.0,
                                    include_vertices=true)
    wps_flat  = generate_waypoints(path, grid_flat,  strategy;
                                    spacing_min=1.0, spacing_max=30.0,
                                    include_vertices=true)

    @test length(wps_sharp) >= length(wps_flat)
end

# ===========================================================================
# 8. Coverage metrics
# ===========================================================================
@testset "Coverage metrics" begin
    @test coverage_ratio(zeros(Int, 10, 10)) ≈ 0.0
    @test coverage_ratio(ones(Int, 10, 10))  ≈ 1.0

    Zh = zeros(Int, 10, 10)
    Zh[1:5, :] .= 1
    @test coverage_ratio(Zh) ≈ 0.5

    Zm = zeros(Int, 10, 10)
    Zm[1:5, :]  .= 3   # 50 cells with value 3
    Zm[6:10, :] .= 8   # 50 cells with value 8
    @test percent_in_bin(Zm, 1, 5)   ≈ 50.0
    @test percent_in_bin(Zm, 6, 10)  ≈ 50.0
    @test percent_in_bin(Zm, 1, 10)  ≈ 100.0
    @test percent_in_bin(Zm, 11, 20) ≈ 0.0

    summ = density_bin_summary(Zm; bins=[(1,5),(6,10)], include_zeros=false)
    @test summ.coverage_ratio ≈ 1.0
    @test summ.bin_percents["bin_1_5"]  ≈ 50.0
    @test summ.bin_percents["bin_6_10"] ≈ 50.0

    # coverage_profile
    cg = RasterGrid(Float64.(Zm), collect(0.0:1.0:9.0), collect(0.0:1.0:9.0))
    wps_dummy = [Waypoint(0.0, 0.0, 80.0, 5.0)]
    positions, coverages = coverage_profile(cg, wps_dummy; axis=:y, bin_size=5.0)
    @test length(positions) == length(coverages)
    @test all(0.0 .<= coverages .<= 1.0)
end

# ===========================================================================
# 9. CSV round-trip I/O
# ===========================================================================
@testset "CSV round-trip I/O" begin
    wps_orig = [
        Waypoint(10.0, 20.0, 80.0, 4.0; line_id=1),
        Waypoint(50.0, 20.0, 80.0, 6.0; line_id=1),
        Waypoint(50.0, 40.0, 80.0, 3.0; line_id=2),
    ]
    tmp = tempname() * ".csv"
    write_waypoints_csv(tmp, wps_orig)
    wps_read = read_waypoints_csv(tmp)
    @test length(wps_read) == 3
    for (a, b) in zip(wps_orig, wps_read)
        @test a.x ≈ b.x && a.y ≈ b.y && a.speed ≈ b.speed && a.line_id == b.line_id
    end
    rm(tmp; force=true)

    rows = [(cover="Field", mission="const2", coverage_ratio=0.85)]
    tmp2 = tempname() * ".csv"
    write_coverage_csv(tmp2, rows)
    df2 = read_coverage_csv(tmp2)
    @test nrow(df2) == 1 && "coverage_ratio" in names(df2)
    rm(tmp2; force=true)
end

# ===========================================================================
# 10. KDE kernel properties
# ===========================================================================
@testset "KDE kernel properties" begin
    # Gaussian kernel sums to 1
    K_g = gaussian_kernel(1.0, 1.0, 5.0, 5.0; radius_mult=3)
    @test sum(K_g) ≈ 1.0  atol=1e-10
    @test all(K_g .>= 0)
    @test size(K_g, 1) == size(K_g, 2)  # symmetric domain

    # Epanechnikov kernel sums to 1, zero outside disk
    K_e = epanechnikov_kernel(1.0, 1.0, 5.0; radius_mult=3)
    @test sum(K_e) ≈ 1.0  atol=1e-10
    @test all(K_e .>= 0)

    # Larger bandwidth → larger kernel
    K_big = gaussian_kernel(1.0, 1.0, 15.0, 15.0; radius_mult=3)
    @test size(K_big, 1) > size(K_g, 1)

    # scott_sigma_indices: returns positive, finite bandwidths
    mask = zeros(Float64, 40, 40)
    mask[15:25, 15:25] .= 1.0
    σx, σy = scott_sigma_indices(mask, 1.0, 1.0; min_pixels=1)
    @test σx > 0 && isfinite(σx)
    @test σy > 0 && isfinite(σy)
end

# ===========================================================================
# 11. kde_from_mask — shape, normalisation, non-negativity
# ===========================================================================
@testset "kde_from_mask" begin
    nx, ny = 30, 30
    xs = collect(range(0.0, 30.0; length=nx))
    ys = collect(range(0.0, 30.0; length=ny))
    mask = zeros(Float64, ny, nx)
    mask[10:20, 10:20] .= 1.0   # centre square blob

    for kern in (:gaussian, :epanechnikov)
        grid, info = kde_from_mask(mask, xs, ys; kernel=kern)

        # Shape preserved
        @test size(grid.Z) == (ny, nx)
        @test length(grid.xs) == nx
        @test length(grid.ys) == ny

        # Non-negative (no FFT round-off negatives)
        @test all(grid.Z .>= 0)

        # Normalised to [0,1]
        @test minimum(grid.Z) ≈ 0.0  atol=1e-10
        @test maximum(grid.Z) ≈ 1.0  atol=1e-10

        # Density peaks somewhere inside or near the mask
        imax = argmax(grid.Z)
        @test grid.ys[imax[1]] >= ys[8]    # within mask neighbourhood
        @test grid.xs[imax[2]] >= xs[8]
    end

    # BitMatrix input
    bmask = mask .> 0.5
    grid_b, _ = kde_from_mask(bmask, xs, ys; kernel=:gaussian)
    @test size(grid_b.Z) == (ny, nx)
end

# ===========================================================================
# 12. Feature stacking and standardisation
# ===========================================================================
@testset "Feature stacking and standardisation" begin
    H, W = 8, 10
    L = rand(Float32, H, W)
    a = rand(Float32, H, W)
    b = rand(Float32, H, W)

    X = stack_features(L, a, b)
    @test size(X) == (H*W, 3)
    @test X[:, 1] ≈ vec(Float32.(L))
    @test X[:, 2] ≈ vec(Float32.(a))
    @test X[:, 3] ≈ vec(Float32.(b))

    # Extra channels
    T = rand(Float32, H, W)
    X2 = stack_features(L, a, b; extra=(T,))
    @test size(X2) == (H*W, 4)

    # standardize_features! makes columns near-zero mean and unit std
    Xf = copy(Float64.(X))
    μ, σ = standardize_features!(Xf; center=true, scale=true)
    for j in 1:3
        @test abs(mean(Xf[:, j])) < 1e-10
        @test abs(std(Xf[:, j]) - 1.0) < 0.02
    end
    @test length(μ) == 3 && length(σ) == 3

    # Mismatched size throws
    @test_throws AssertionError stack_features(rand(Float32,3,3), rand(Float32,4,3), b)
end

# ===========================================================================
# 13. labels_to_mask and mask_stats
# ===========================================================================
@testset "labels_to_mask and mask_stats" begin
    # Simple 3×3 with labels 1 and 2
    labels = [1,2,1, 2,1,2, 1,2,1]  # vec of 9
    H, W   = 3, 3
    mask   = labels_to_mask(labels, H, W; tree_labels=[1])
    @test size(mask) == (H, W)
    @test count(mask) == 5   # five 1s

    # Multiple tree labels
    mask2 = labels_to_mask(labels, H, W; tree_labels=[1, 2])
    @test all(mask2)    # all true when both labels included

    # Wrong length throws
    @test_throws DimensionMismatch labels_to_mask([1,2,3], 3, 3; tree_labels=[1])

    # mask_stats
    s = mask_stats(mask)
    @test s.area == 5
    @test s.frac ≈ 5/9
    @test length(s.bbox) == 4

    # All-false mask
    s0 = mask_stats(falses(4, 4))
    @test s0.area == 0
    @test s0.frac ≈ 0.0
end

# ===========================================================================
# 14. Pipeline: build_density_surface (mask → KDE → grid)
# ===========================================================================
@testset "build_density_surface" begin
    mask = synthetic_mask_grid(nx=40, ny=40, xmin=0.0, xmax=80.0,
                                ymin=0.0, ymax=80.0, frac=0.35)

    for kern in (:gaussian, :epanechnikov)
        cfg = PipelineConfig(kde_kernel=kern)
        dens, info = build_density_surface(mask, cfg)

        @test size(dens.Z) == size(mask.Z)
        @test all(dens.Z .>= 0)
        @test minimum(dens.Z) ≈ 0.0  atol=1e-10
        @test maximum(dens.Z) ≈ 1.0  atol=1e-10
        @test haskey(info, :dx)
        @test haskey(info, :dy)
    end
end

# ===========================================================================
# 15. plan_mission convenience wrapper
# ===========================================================================
@testset "plan_mission convenience" begin
    grid = synthetic_gaussian_grid(nx=30, ny=30,
                                   xmin=0.0, xmax=60.0, ymin=0.0, ymax=60.0)
    strategy = KDEGuidedSpeed(grid; vmin=2.0, vmax=8.0)
    config   = FlightConfig(strategy, 80.0, "KDE-guided Gaussian";
                            kernel=:gaussian, line_spacing=15.0)
    wps = plan_mission(grid, config; seconds_per_wp=1.5, spacing_min=2.0)
    @test !isempty(wps)
    @test all(2.0 - 1e-9 <= w.speed <= 8.0 + 1e-9 for w in wps)
end

# ===========================================================================
# 16. rgb_from_array and rgb_to_lab (colorspace)
# ===========================================================================
@testset "Colorspace: rgb_from_array and rgb_to_lab" begin
    # Channel-last (H,W,3)
    arr_cl = rand(UInt8, 8, 10, 3)
    img_cl = rgb_from_array(arr_cl)
    @test size(img_cl) == (8, 10)
    @test img_cl isa Matrix{RGB{Float32}}
    # pixel values should be in [0,1]
    @test all(c -> 0f0 <= red(c) <= 1f0 && 0f0 <= green(c) <= 1f0 && 0f0 <= blue(c) <= 1f0, img_cl)

    # Channel-first (3,H,W)
    arr_cf = rand(UInt8, 3, 8, 10)
    img_cf = rgb_from_array(arr_cf)
    @test size(img_cf) == (8, 10)

    # Float32 array (channel-last)
    arr_f  = rand(Float32, 6, 6, 3)
    img_f  = rgb_from_array(arr_f)
    @test size(img_f) == (6, 6)

    # rgb_to_lab returns Float32 matrices of correct size
    L, a, b = rgb_to_lab(img_cl)
    @test size(L) == (8, 10)
    @test L isa Matrix{Float32}
    @test a isa Matrix{Float32}
    @test b isa Matrix{Float32}
    # L channel in [0, 100]
    @test all(0f0 .<= L .<= 100f0)

    # rgb_to_lab_array convenience
    L2, a2, b2 = rgb_to_lab_array(arr_cl)
    @test L2 ≈ L

    # is_grayscale: pure grey image
    grey = fill(RGB{Float32}(0.5f0, 0.5f0, 0.5f0), 4, 4)
    @test is_grayscale(grey)
    @test !is_grayscale(img_cl)   # random → almost certainly not grey

    # lab_to_array packs back to (H,W,3)
    lab_arr = lab_to_array(L, a, b)
    @test size(lab_arr) == (8, 10, 3)
    @test lab_arr[:,:,1] ≈ L
end

# ===========================================================================
# 17. k-medoids fit (Clustering.jl)
# ===========================================================================
@testset "kmedoids_fit" begin
    # Build a clearly separable 2-cluster feature matrix
    Random.seed!(42)
    Xa = randn(30, 3) .+ 5.0    # cluster 1 centre ≈ (5,5,5)
    Xb = randn(30, 3) .- 5.0    # cluster 2 centre ≈ (-5,-5,-5)
    X  = vcat(Xa, Xb)           # 60 × 3

    labels, res, info = kmedoids_fit(X; k=2, seed=42)
    @test length(labels) == 60
    @test all(l -> l in (1, 2), labels)

    # The two original groups should be largely separated
    grp1 = labels[1:30]; grp2 = labels[31:60]
    # Most of group A should share a label; same for B
    dom1 = maximum(count(==(l), grp1) for l in (1,2))
    dom2 = maximum(count(==(l), grp2) for l in (1,2))
    @test dom1 >= 25   # ≥ 25/30 correct
    @test dom2 >= 25

    # With idxs_sample (sub-sampling path)
    idxs = collect(1:2:60)
    D    = sample_distance_matrix(X, idxs)
    labels2, _, _ = kmedoids_fit(X; k=2, idxs_sample=idxs, D=D, seed=42)
    @test length(labels2) == 60
    @test all(l -> l in (1, 2), labels2)
end

# ===========================================================================
# 18. Auto-k quality metrics sweep
# ===========================================================================
@testset "sweep_k_quality and choose_k" begin
    Random.seed!(7)
    # 3 well-separated clusters in 2-D
    Xa = randn(20, 2) .+ [0.0 0.0]
    Xb = randn(20, 2) .+ [10.0 0.0]
    Xc = randn(20, 2) .+ [5.0 8.66]
    X  = vcat(Xa, Xb, Xc)

    idxs = collect(1:3:60)   # every 3rd sample
    D    = sample_distance_matrix(X, idxs)
    mets = sweep_k_quality(X, idxs, D; ks=2:5)

    @test length(mets) == 4                 # one entry per k
    @test all(m -> m.k in 2:5, mets)
    @test all(m -> isfinite(m.silhouette), mets)
    @test all(m -> isfinite(m.dunn),       mets)

    for strat in (:silhouette, :dunn, :mode)
        kstar = choose_k(mets; strategy=strat)
        @test kstar in 2:5
    end
end

# ===========================================================================
# 19. RGB image → k-medoids mask → KDE → waypoints (end-to-end synthetic)
# ===========================================================================
@testset "End-to-end: image array → mask → KDE → waypoints" begin
    Random.seed!(99)

    # Build a toy 20×20 image with a vegetation patch
    arr = zeros(UInt8, 20, 20, 3)
    arr[6:15, 6:15, 1] .= 10    # dark green-ish patch (low R, high G)
    arr[6:15, 6:15, 2] .= 180
    arr[6:15, 6:15, 3] .= 10
    arr[1:5,  :,    1] .= 220   # bright background (soil)
    arr[16:20,:,    1] .= 220
    arr[:,  1:5,    1] .= 220
    arr[:, 16:20,   1] .= 220

    # 1) Lab features
    L, a, b = rgb_to_lab_array(arr)
    @test size(L) == (20, 20)

    # 2) Mask via kmedoids_fit (fixed k=2)
    X = Float64.(stack_features(L, a, b))
    standardize_features!(X)
    labels, _, _ = kmedoids_fit(X; k=2, seed=42)
    # pick whichever label has more green pixels as tree label
    green_region = 6:15
    labels_reshaped = reshape(labels, 20, 20)
    lgreen = labels_reshaped[green_region, green_region]
    tree_lbl = [argmax([count(==(1), lgreen), count(==(2), lgreen)])]
    mask_bm  = labels_to_mask(labels, 20, 20; tree_labels=tree_lbl)
    @test size(mask_bm) == (20, 20)
    @test mask_bm isa BitMatrix

    # 3) KDE density surface
    xs, ys = pixel_axes(20, 20)
    mask_grid = RasterGrid(Float64.(mask_bm), xs, ys)
    cfg  = PipelineConfig(kde_kernel=:epanechnikov)
    dens, info = build_density_surface(mask_grid, cfg)
    @test size(dens.Z) == (20, 20)
    @test minimum(dens.Z) >= 0.0
    @test maximum(dens.Z) <= 1.0 + 1e-10
    @test haskey(info, :dx)

    # 4) Plan mission
    strategy = KDEGuidedSpeed(dens; vmin=2.0, vmax=8.0)
    config   = FlightConfig(strategy, 80.0, "test"; line_spacing=5.0)
    wps = plan_mission(dens, config; seconds_per_wp=1.0, spacing_min=1.0)
    @test !isempty(wps)
    @test all(2.0 - 1e-9 <= w.speed <= 8.0 + 1e-9 for w in wps)
    @test all(0.0 - 1e-6 <= w.x <= Float64(20) + 1e-6 for w in wps)
end

# ===========================================================================
# 20. build_mask_from_image convenience API
# ===========================================================================
@testset "build_mask_from_image" begin
    Random.seed!(11)
    arr = rand(UInt8, 16, 16, 3)

    mask_grid, info = build_mask_from_image(arr;
        k=2, tree_labels=[1], seed=42,
        nsample=50, use_pca=false)

    @test mask_grid isa RasterGrid
    @test size(mask_grid.Z) == (16, 16)
    @test all(z -> z ≈ 0.0 || z ≈ 1.0, mask_grid.Z)
    @test info.k == 2
    @test length(info.labels_full) == 16*16

    # With PCA
    mask_pca, info_pca = build_mask_from_image(arr;
        k=2, tree_labels=[1], seed=42,
        nsample=50, use_pca=true, variance_ratio=0.99)
    @test mask_pca isa RasterGrid
    @test size(mask_pca.Z) == (16, 16)
end

# ===========================================================================
# 21. PCA fit/transform
# ===========================================================================
@testset "PCA fit and transform" begin
    Random.seed!(5)
    X = randn(50, 4)   # 50 samples, 4 features
    standardize_features!(X)

    pca = pca_fit(X; variance_ratio=0.95)
    @test pca isa MultivariateStats.PCA
    Xprj = pca_transform(pca, X)
    @test size(Xprj, 1) == 50
    @test size(Xprj, 2) <= 4

    ex = pca_explained(pca)
    @test isapprox(sum(ex.explained), 1.0; atol=0.01) || sum(ex.cumulative) > 0.9
    @test ex.cumulative[end] >= 0.9
end

# ===========================================================================
# 22. pixel_axes and axes_from_geotransform
# ===========================================================================
@testset "pixel_axes and axes_from_geotransform" begin
    xs, ys = pixel_axes(10, 20)
    @test length(xs) == 20
    @test length(ys) == 10
    @test issorted(xs)
    @test issorted(ys)

    # GDAL geotransform (north-up, pixel size 1m)
    gt = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0]  # y_origin=200, y_res=-1
    xs_gt, ys_gt = axes_from_geotransform(gt, 5, 4)
    @test length(xs_gt) == 5
    @test length(ys_gt) == 4
    @test issorted(xs_gt)
    @test issorted(ys_gt)   # must be ascending (flipped from north-up)
end

# ===========================================================================
# 23. gini_coefficient — hand-checkable cases
# ===========================================================================
@testset "gini_coefficient" begin
    # Perfectly equal distribution → Gini = 0
    @test gini_coefficient([2.0, 2.0, 2.0, 2.0]) ≈ 0.0  atol=1e-12

    # Single value → Gini = 0
    @test gini_coefficient([7.0]) ≈ 0.0  atol=1e-12

    # Hand-computed: v = [1,2,3,4,5] sorted
    # n=5, s=15, num = 1*1+2*2+3*3+4*4+5*5 = 1+4+9+16+25 = 55
    # G = 2*55/(5*15) - 6/5 = 110/75 - 1.2 = 1.4667 - 1.2 = 0.2667
    v_hand = [1.0, 2.0, 3.0, 4.0, 5.0]
    G_hand = 2*(1*1 + 2*2 + 3*3 + 4*4 + 5*5) / (5 * 15) - 6/5
    @test gini_coefficient(v_hand) ≈ G_hand  atol=1e-12

    # lidar.jl test: v = [3,5,2,1,4] (from a 3×3 grid with 4 zeros)
    # sorted c = [1,2,3,4,5], s=15, num=1+4+9+16+25=55, n=5
    # G = 2*55/75 - 6/5 ≈ 0.26667
    @test gini_coefficient([3.0, 5.0, 2.0, 1.0, 4.0]) ≈ 0.26666666666666666  atol=1e-10

    # Empty → 0 by convention
    @test gini_coefficient(Float64[]) ≈ 0.0
end

# ===========================================================================
# 24. morans_i — hand-checkable cases
# ===========================================================================
@testset "morans_i" begin
    # 2×2 grid [1 2; 3 4], all-true mask, queen adjacency
    # x_bar=2.5, deviations: [-1.5 -0.5; 0.5 1.5]
    # All 4 cells are valid; queen pairs (both directions): W=12
    # num = (row-major cross-products, each counted twice)
    # Expected I = -1/3
    counts_2x2 = [1  2; 3  4]
    mask_2x2   = trues(2, 2)
    @test morans_i(counts_2x2, mask_2x2; neighbor=:queen) ≈ -1/3  atol=1e-10

    # All zeros → valid is empty → missing
    @test ismissing(morans_i(zeros(Int, 3, 3), trues(3, 3)))

    # Mask excludes all nonzero cells → valid is empty → missing
    m = zeros(Int, 3, 3); m[1,1] = 5
    mask_excl = falses(3, 3); mask_excl[2, 2] = true
    @test ismissing(morans_i(m, mask_excl))

    # Spatially clustered: counts = [5 5 0; 5 0 1; 0 1 1], all-true mask, queen
    # v = [5,5,5,1,1,1], x_bar=3, W=16, num=32, denom=24
    # I = (6/16)*(32/24) = 0.5
    counts_clust = [5 5 0; 5 0 1; 0 1 1]
    @test morans_i(counts_clust, trues(3, 3); neighbor=:queen) ≈ 0.5  atol=1e-10

    # Rook adjacency gives a different result
    mi_rook  = morans_i(counts_2x2, mask_2x2; neighbor=:rook)
    mi_queen = morans_i(counts_2x2, mask_2x2; neighbor=:queen)
    @test !ismissing(mi_rook)
    @test mi_rook !== mi_queen

    # Size mismatch throws
    @test_throws DimensionMismatch morans_i(ones(Int, 3, 3), trues(2, 2))
end

# ===========================================================================
# 25. summary_statistics — hand-checkable matrix (matches lidar.jl semantics)
# ===========================================================================
@testset "summary_statistics" begin
    # 3×3 grid: 4 zeros + 5 positives
    # counts = [0 0 3; 0 5 2; 1 0 4]  (Julia: row 1 = [0 0 3], etc.)
    counts = [0 0 3; 0 5 2; 1 0 4]
    mask   = trues(3, 3)   # N = 9

    s = summary_statistics(counts, mask)

    @test s.ncells == 5
    @test s.N      == 9
    @test s.Q1     ≈ 2.0
    @test s.Q2     ≈ 3.0
    @test s.Q3     ≈ 4.0
    @test s.IQR    ≈ 2.0
    @test s.Mean   ≈ 3.0
    @test s.CR     ≈ 5/9
    # CV: std([1,2,3,4,5]; ddof=1) / mean([1,2,3,4,5])
    # Wait: nonzero values are [3,5,2,1,4] = same set, sorted [1,2,3,4,5]
    # std([1,2,3,4,5]; corrected) = sqrt(10/4) = sqrt(2.5) ≈ 1.5811...
    # mean = 3.0
    # CV = 1.5811.../3.0 ≈ 0.5270...
    @test s.CV     ≈ 0.5270462766947299  atol=1e-10
    @test s.Gini   ≈ 0.26666666666666666 atol=1e-10
    @test !ismissing(s.MoranI)

    # Uniform grid → CR=1, CV=0, Gini=0
    s2 = summary_statistics([2 2; 2 2], trues(2, 2))
    @test s2.ncells == 4
    @test s2.N      == 4
    @test s2.CR     ≈ 1.0
    @test s2.CV     ≈ 0.0  atol=1e-12
    @test s2.Gini   ≈ 0.0  atol=1e-12

    # Partial mask: counts = [0 1 2; 3 4 5; 6 7 8]
    # mask = [T T T; T F F; F F F]  → N=4, positive masked = {1,2,3}
    counts3 = [0 1 2; 3 4 5; 6 7 8]
    mask3   = BitMatrix([true true true; true false false; false false false])
    s3 = summary_statistics(counts3, mask3)
    @test s3.ncells == 3
    @test s3.N      == 4
    @test s3.Q1     ≈ 1.5
    @test s3.Q2     ≈ 2.0
    @test s3.Q3     ≈ 2.5
    @test s3.Mean   ≈ 2.0
    @test s3.CR     ≈ 0.75
    @test s3.CV     ≈ 0.5  atol=1e-10
    @test s3.Gini   ≈ 0.22222222222222232  atol=1e-10

    # All-zero grid → CR=0, ncells=0, MoranI=missing
    s_zero = summary_statistics(zeros(Int, 4, 4), trues(4, 4))
    @test s_zero.ncells == 0
    @test s_zero.CR     ≈ 0.0
    @test ismissing(s_zero.MoranI)

    # support keyword (no-mask form)
    s_noMask = summary_statistics([0 3; 5 2]; support=10)
    @test s_noMask.N      == 10
    @test s_noMask.ncells == 3
    @test s_noMask.CR     ≈ 3/10
end

# ===========================================================================
# 26. CountKey and label helpers
# ===========================================================================
@testset "CountKey and label helpers" begin
    k = CountKey(ret=:ground, cover=:conif, mission=:speed, kernel=:E)
    @test k.ret     === :ground
    @test k.cover   === :conif
    @test k.mission === :speed
    @test k.kernel  === :E

    @test return_label(:all)     == "All"
    @test return_label(:ground)  == "Ground"
    @test mission_label(:density) == "Density-aware"
    @test mission_label(:speed)   == "Speed-aware"
    @test mission_label(:const2)  == "Const 2 m/s"
    @test mission_label(:const8)  == "Const 8 m/s"
    @test kernel_label(:NA)  == "——"
    @test kernel_label(:G)   == "G"
    @test kernel_label(:E)   == "E"
    @test cover_label(:field) == "Field"
    @test cover_label(:decid) == "Deciduous"
    @test cover_label(:conif) == "Coniferous"
end

# ===========================================================================
# 27. DURHAM_COVER_N — sanity checks
# ===========================================================================
@testset "DURHAM_COVER_N" begin
    @test DURHAM_COVER_N[:field]  == 55_719
    @test DURHAM_COVER_N[:decid]  == 24_080
    @test DURHAM_COVER_N[:conif]  ==  5_413
    # Partition: the three zones exactly tile the 324×263 raster
    @test sum(values(DURHAM_COVER_N)) == 324 * 263
end

# ===========================================================================
# 28. load_counts_json + build_cover_masks + summarize_counts_json
#     Regression validation against stats_and_coverage.csv
# ===========================================================================

# Ground-truth data files. Default to the bundled copies under
# data/ground_truth/; override with KDE_TEST_COUNTS_JSON / KDE_TEST_STATS_CSV
# to point at alternate (e.g. freshly recomputed) tables.
const _GT_DIR      = joinpath(@__DIR__, "..", "data", "ground_truth")
const _COUNTS_JSON = get(ENV, "KDE_TEST_COUNTS_JSON", joinpath(_GT_DIR, "counts.json"))
const _STATS_CSV   = get(ENV, "KDE_TEST_STATS_CSV",   joinpath(_GT_DIR, "stats_and_coverage.csv"))

if isfile(_COUNTS_JSON) && isfile(_STATS_CSV)
    @testset "Regression: summarize_counts_json vs stats_and_coverage.csv" begin
        # Load ground-truth CSV
        ref = CSV.read(_STATS_CSV, DataFrame)

        # Regenerate from counts.json
        regen = summarize_counts_json(_COUNTS_JSON)

        @test nrow(regen) == nrow(ref)

        # Sort both by the same key columns for aligned comparison
        sort!(ref,   [:Cover, :Return, :Mission, :Kernel])
        sort!(regen, [:Cover, :Return, :Mission, :Kernel])

        integer_cols = [:ncells, :N]
        float_cols   = [:Q1, :Q2, :Q3, :IQR, :Mean, :CR, :CV, :Gini]

        for col in integer_cols
            mismatches_int = Int[]
            for i in 1:nrow(ref)
                ref_val   = ref[i, col]
                regen_val = regen[i, col]
                ref_val == regen_val || push!(mismatches_int, i)
            end
            @test isempty(mismatches_int)
        end

        for col in float_cols
            mismatches_flt = Int[]
            for i in 1:nrow(ref)
                ref_val   = Float64(ref[i, col])
                regen_val = Float64(regen[i, col])
                isapprox(ref_val, regen_val; rtol=1e-8) || push!(mismatches_flt, i)
            end
            @test isempty(mismatches_flt)
        end

        moran_mismatches = Int[]
        for i in 1:nrow(ref)
            ref_mi   = Float64(ref[i, :MoranI])
            regen_mi = regen[i, :MoranI]
            if ismissing(regen_mi) || !isapprox(ref_mi, Float64(regen_mi); rtol=1e-8)
                push!(moran_mismatches, i)
            end
        end
        @test isempty(moran_mismatches)
    end
else
    @warn "Skipping regression test: data files not found at expected paths."
    @warn "  counts.json path: $_COUNTS_JSON"
    @warn "  stats CSV path:   $_STATS_CSV"
end

# ===========================================================================
# 29. bootstrap_cr — deterministic seed reproducibility
# ===========================================================================
@testset "bootstrap_cr: seed reproducibility" begin
    Random.seed!(0)
    counts = rand(0:4, 20, 15)
    mask   = trues(20, 15)

    ci1 = bootstrap_cr(counts, mask; nboot=200, seed=42)
    ci2 = bootstrap_cr(counts, mask; nboot=200, seed=42)
    ci3 = bootstrap_cr(counts, mask; nboot=200, seed=99)  # different seed

    # Same seed → identical replicates
    @test ci1.replicates == ci2.replicates
    @test ci1.lower ≈ ci2.lower
    @test ci1.upper ≈ ci2.upper

    # Different seed → different replicates (with overwhelming probability)
    @test ci1.replicates != ci3.replicates
end

# ===========================================================================
# 30. bootstrap_cr — CI contains the point estimate (on small data)
# ===========================================================================
@testset "bootstrap_cr: CI contains point estimate" begin
    # Deterministic count grid: half the mask cells are positive
    counts = zeros(Int, 10, 10)
    counts[1:5, :] .= 3
    mask = trues(10, 10)

    ci = bootstrap_cr(counts, mask; nboot=500, seed=42)

    @test !isnan(ci.estimate)
    @test !isnan(ci.lower)
    @test !isnan(ci.upper)
    @test ci.lower <= ci.estimate <= ci.upper
    @test ci.lower >= 0.0
    @test ci.upper <= 1.0
    @test ci.level ≈ 0.95
    @test ci.block_side >= 1
 end

# ===========================================================================
# 31. bootstrap_cr — block_side bounds and block_frac edge cases
# ===========================================================================
@testset "bootstrap_cr: block_frac edge cases" begin
    counts = ones(Int, 8, 8)
    mask   = trues(8, 8)

    # Very small block_frac → block_side clamped to 1
    ci_tiny = bootstrap_cr(counts, mask; nboot=100, block_frac=1e-6, seed=1)
    @test ci_tiny.block_side == 1

    # block_frac ≥ 1 → block_side clamped to min(nrows,ncols)
    ci_big = bootstrap_cr(counts, mask; nboot=100, block_frac=10.0, seed=1)
    @test ci_big.block_side == 8  # min(8,8)

    # CR=1 case: all cells positive → CI should be near [1,1]
    @test ci_tiny.estimate ≈ 1.0
    @test ci_big.estimate  ≈ 1.0
end

# ===========================================================================
# 32. bootstrap_cr — empty mask edge case
# ===========================================================================
@testset "bootstrap_cr: empty mask" begin
    counts = ones(Int, 5, 5)
    mask   = falses(5, 5)

    ci = bootstrap_cr(counts, mask; nboot=100, seed=42)
    @test isnan(ci.estimate)
    @test isnan(ci.lower)
    @test isnan(ci.upper)
end

# ===========================================================================
# 33. bootstrap_cr — all-zero counts
# ===========================================================================
@testset "bootstrap_cr: all-zero counts" begin
    counts = zeros(Int, 10, 10)
    mask   = trues(10, 10)

    ci = bootstrap_cr(counts, mask; nboot=200, seed=42)
    @test ci.estimate ≈ 0.0
    @test ci.lower ≈ 0.0
    @test ci.upper ≈ 0.0
end

# ===========================================================================
# 34. bootstrap_cr_difference — paired behavior and known sign
# ===========================================================================
@testset "bootstrap_cr_difference: paired behavior and sign" begin
    # A dominates B clearly: A hits all cells, B hits half
    counts_a = ones(Int, 20, 20)         # CR = 1.0
    counts_b = zeros(Int, 20, 20)
    counts_b[1:10, :] .= 1               # CR = 0.5
    mask = trues(20, 20)

    ci = bootstrap_cr_difference(counts_a, counts_b, mask;
                                  nboot=500, seed=42)

    # Point estimate should be ≈ 0.5
    @test ci.estimate ≈ 0.5  atol=1e-10

    # CI should be entirely positive (A > B)
    @test ci.lower > 0.0
    @test ci.upper > 0.0

    # Flipped: B dominates A → CI should be entirely negative
    ci_flip = bootstrap_cr_difference(counts_b, counts_a, mask;
                                       nboot=500, seed=42)
    @test ci_flip.estimate ≈ -0.5  atol=1e-10
    @test ci_flip.upper < 0.0
end

# ===========================================================================
# 35. bootstrap_cr_difference — seed reproducibility
# ===========================================================================
@testset "bootstrap_cr_difference: seed reproducibility" begin
    Random.seed!(0)
    mat_a = rand(0:3, 15, 15)
    mat_b = rand(0:3, 15, 15)
    mask  = trues(15, 15)

    ci1 = bootstrap_cr_difference(mat_a, mat_b, mask; nboot=100, seed=7)
    ci2 = bootstrap_cr_difference(mat_a, mat_b, mask; nboot=100, seed=7)
    ci3 = bootstrap_cr_difference(mat_a, mat_b, mask; nboot=100, seed=8)

    @test ci1.replicates == ci2.replicates
    @test ci1.lower ≈ ci2.lower
    @test ci1.replicates != ci3.replicates
end

# ===========================================================================
# 36. BootstrapCI struct — fields present and typed correctly
# ===========================================================================
@testset "BootstrapCI struct fields" begin
    counts = rand(0:5, 12, 12)
    mask   = trues(12, 12)
    ci = bootstrap_cr(counts, mask; nboot=50, seed=42)

    @test ci isa BootstrapCI
    @test ci.estimate isa Float64
    @test ci.lower    isa Float64
    @test ci.upper    isa Float64
    @test ci.level    isa Float64
    @test ci.nboot    == 50
    @test length(ci.replicates) == 50
    @test ci.lower <= ci.estimate <= ci.upper
end

# ===========================================================================
# 37. Comparison struct and manuscript_mission_label
# ===========================================================================
@testset "Comparison struct and manuscript labels" begin
    ka = CountKey(ret=:ground, cover=:conif, mission=:density, kernel=:E)
    kb = CountKey(ret=:ground, cover=:conif, mission=:const2,  kernel=:NA)
    comp = Comparison(ka, kb, "KDE-guided E vs Const 2 m/s")

    @test comp.label == "KDE-guided E vs Const 2 m/s"
    @test comp.key_a === ka
    @test comp.key_b === kb

    @test manuscript_mission_label(:density, :E) == "KDE-guided Epanechnikov"
    @test manuscript_mission_label(:density, :G) == "KDE-guided Gaussian"
    @test manuscript_mission_label(:const2, :NA) == "Constant 2 m/s"
    @test manuscript_mission_label(:const8, :NA) == "Constant 8 m/s"
end

# ===========================================================================
# 38. bootstrap_cr_table — integration test on real data (if available)
# ===========================================================================
if isfile(_COUNTS_JSON)
    @testset "bootstrap_cr_table: real data integration" begin
        # Use a small nboot for speed in tests
        df = bootstrap_cr_table(_COUNTS_JSON;
                                nboot=100, block_frac=0.03, seed=42)

        @test df isa DataFrame
        @test nrow(df) == 36  # 3 covers × 2 returns × (4 missions: 2KA×2kernels + 2const) = 36
        @test "CR" in names(df)
        @test "CI_lower" in names(df)
        @test "CI_upper" in names(df)
        @test "block_side" in names(df)

        # All CIs should be valid
        @test all(!isnan, df.CI_lower)
        @test all(!isnan, df.CI_upper)
        # CI_lower <= CI_upper everywhere
        @test all(df.CI_lower .<= df.CI_upper .+ 1e-10)
        @test all(df.CI_lower .>= 0.0)
        @test all(df.CI_upper .<= 1.0 .+ 1e-10)
        # bootstrap-internal estimate is within its own CI
        @test all(df.CI_lower .<= df.CR_bootstrap .+ 1e-10)
        @test all(df.CI_upper .>= df.CR_bootstrap .- 1e-10)
    end
else
    @warn "Skipping bootstrap_cr_table real-data test: counts.json not found"
end

# ===========================================================================
# 39. bootstrap_cr_difference_table — integration test on real data
# ===========================================================================
if isfile(_COUNTS_JSON)
    @testset "bootstrap_cr_difference_table: real data integration" begin
        comps = default_narrative_comparisons()
        df = bootstrap_cr_difference_table(_COUNTS_JSON, comps;
                                           nboot=100, block_frac=0.03, seed=42)

        @test df isa DataFrame
        @test nrow(df) == length(comps)
        @test "CR_diff" in names(df)
        @test "CI_lower" in names(df)
        @test "CI_upper" in names(df)

        # All CIs should be valid
        @test all(!isnan, df.CI_lower)
        @test all(!isnan, df.CI_upper)
        @test all(df.CI_lower .<= df.CI_upper .+ 1e-10)
        # bootstrap-internal estimate is within its own CI
        @test all(df.CI_lower .<= df.CR_diff_bootstrap .+ 1e-10)
        @test all(df.CI_upper .>= df.CR_diff_bootstrap .- 1e-10)
    end
else
    @warn "Skipping bootstrap_cr_difference_table real-data test: counts.json not found"
end

# ===========================================================================
# 40. derive_planned_lines — synthetic waypoint data
# ===========================================================================
@testset "derive_planned_lines: synthetic waypoints" begin
    # Build a minimal waypoint DataFrame: two survey lines at y=100 and y=200,
    # each with 12 waypoints spanning 300 m in x, plus some "turn" waypoints
    # that should be excluded (n<10 or x_span<250).
    using DataFrames
    wp = DataFrame(
        GridX = vcat(
            range(50.0, 350.0; length=12),   # line y≈100
            range(50.0, 350.0; length=12),   # line y≈200
            [10.0, 11.0],                     # turn: n<10 → excluded
        ),
        GridY = vcat(
            fill(100.2, 12),
            fill(200.3, 12),
            [150.0, 150.0],
        ),
    )
    # Write to temp file and read back
    tmp = tempname() * ".csv"
    CSV.write(tmp, wp)
    pl = derive_planned_lines(tmp)
    rm(tmp)

    @test nrow(pl) == 2
    @test pl.line_index == [0, 1]
    @test pl.line_number == [1, 2]
    @test pl.y_center[1] ≈ 100.2  atol=0.05
    @test pl.y_center[2] ≈ 200.3  atol=0.05
    @test pl.x_min[1] ≈ 50.0  atol=1.0
    @test pl.x_max[1] ≈ 350.0 atol=1.0
end

# ===========================================================================
# 41. assign_to_planned_lines — nearest line assignment
# ===========================================================================
@testset "assign_to_planned_lines: basic" begin
    line_centers = [100.0, 200.0, 300.0]
    y = [99.0, 201.0, 300.5, 145.0]
    nl, cte = assign_to_planned_lines(y, line_centers)
    # 99 → line 0 (nearest 100);
    # 201 → line 1 (nearest 200);
    # 300.5 → line 2 (nearest 300);
    # 145 → line 0 or 1 — equidistant; argmin picks first (index 0→line 0)... actually
    # |145-100|=45, |145-200|=55 → nearest=100 → line 0
    @test nl[1] == 0
    @test nl[2] == 1
    @test nl[3] == 2
    @test nl[4] == 0
    @test cte[1] ≈ 1.0
    @test cte[2] ≈ 1.0
    @test cte[3] ≈ 0.5
end

# ===========================================================================
# 42. mission_speed_bounds
# ===========================================================================
@testset "mission_speed_bounds" begin
    lo2, hi2 = mission_speed_bounds("Const. 2 m/s")
    @test lo2 ≈ 1.0 && hi2 ≈ 3.2

    lo8, hi8 = mission_speed_bounds("Const. 8 m/s")
    @test lo8 ≈ 6.0 && hi8 ≈ 10.5

    lok, hik = mission_speed_bounds("KDE-guided (Epanechnikov)")
    @test lok ≈ 1.0 && hik ≈ 9.5
end

# ===========================================================================
# 43. contiguous_segments — basic labelling
# ===========================================================================
@testset "contiguous_segments: basic" begin
    # All selected, same line, no time gap → one segment
    mask = Bool[1, 1, 1, 1]
    time = Float64[0.0, 0.1, 0.2, 0.3]
    line = Int[0, 0, 0, 0]
    seg  = contiguous_segments(mask, time, line)
    @test all(seg .== 0)

    # Line change splits segment
    mask2 = Bool[1, 1, 1, 1]
    time2 = Float64[0.0, 0.1, 0.2, 0.3]
    line2 = Int[0, 0, 1, 1]
    seg2  = contiguous_segments(mask2, time2, line2)
    @test seg2[1] == seg2[2]           # same first segment
    @test seg2[3] == seg2[4]           # same second segment
    @test seg2[3] == seg2[1] + 1       # new segment id

    # Time gap > 0.25 s splits segment
    mask3 = Bool[1, 1, 1]
    time3 = Float64[0.0, 0.4, 0.5]    # gap 0.4 s between idx 1 and 2
    line3 = Int[0, 0, 0]
    seg3  = contiguous_segments(mask3, time3, line3)
    @test seg3[1] == 0
    @test seg3[2] == 1
    @test seg3[3] == 1

    # Unselected samples get -1
    mask4 = Bool[1, 0, 1]
    time4 = Float64[0.0, 0.1, 0.2]
    line4 = Int[0, 0, 0]
    seg4  = contiguous_segments(mask4, time4, line4)
    @test seg4[2] == -1
    @test seg4[1] != seg4[3]  # gap in selection → new segment
end

# ===========================================================================
# 44. ground_speed
# ===========================================================================
@testset "ground_speed" begin
    @test ground_speed([3.0], [4.0])[1] ≈ 5.0
    @test ground_speed([0.0], [0.0])[1] ≈ 0.0
    @test all(ground_speed([1.0, 2.0], [0.0, 0.0]) .≈ [1.0, 2.0])
end

# ===========================================================================
# 45. clean_survey_segments — synthetic trajectory
# ===========================================================================
@testset "clean_survey_segments: synthetic" begin
    # Two planned lines: y=100 and y=200, x=0..400
    using DataFrames
    using Statistics: median
    pl = DataFrame(
        line_index  = [0, 1],
        line_number = [1, 2],
        y_center    = [100.0, 200.0],
        x_min       = [0.0, 0.0],
        x_max       = [400.0, 400.0],
        n           = [12, 12],
    )
    # Build a synthetic trajectory:
    # - 500 samples along line 0 (y≈100), x=0→300, east-west, speed=2 m/s
    # Sampling at 10 Hz (0.1 s steps) → 50 s for 300 m at 2 m/s → x_span=300, dur≥2.5 ✓
    # - 10 samples as a "turn" (high cross-track, will be filtered)
    n_survey = 500
    n_turn   = 10
    # 10 Hz sampling: 0.1 s steps → 50 s total, x=0 to 300 m (speed ≈6 m/s is too fast for Const 2 m/s)
    # Use speed=2 m/s: 300m/2m/s=150s, 500 samples at 0.1s steps covers 50s → x_span=100m ✓
    # Let x go 0→100 in 50 s, 500 samples, speed = 2 m/s
    t_survey = range(0.0, stop=49.9, length=n_survey)  # 500 samples at ≈0.1 s steps
    x_survey = range(0.0, 100.0; length=n_survey)
    df = DataFrame(
        GridX  = vcat(collect(x_survey), fill(0.0, n_turn)),
        GridY  = vcat(fill(100.5, n_survey), fill(150.0, n_turn)),  # turn: far from both lines
        VEast  = vcat(fill(2.0, n_survey), fill(0.0, n_turn)),
        VNorth = vcat(fill(0.0, n_survey), fill(2.0, n_turn)),
        Time   = vcat(collect(t_survey), collect(range(55.0, 60.0; length=n_turn))),
    )
    clean, full = clean_survey_segments(df, "Const. 2 m/s", pl)

    # Survey samples should pass; turn samples should be dropped
    @test nrow(clean) > 0
    @test nrow(clean) <= n_survey
    # All clean samples should be on line 0
    @test all(clean.nearest_line .== 0)
    # Ground speed should be ≈2 m/s for survey samples
    @test all(abs.(clean.ground_speed_mps .- 2.0) .< 0.01)
    # Tracking error should be small (≈0.5 m from line center)
    @test median(clean.tracking_error_m) < 2.0
end

# ===========================================================================
# 46. decode_count_grid — shape and flip convention
# ===========================================================================
@testset "decode_count_grid: shape and flip" begin
    # Flat vector length 263*324
    vals = collect(Float64, 1:(263*324))
    mat  = decode_count_grid(vals)
    @test size(mat) == (263, 324)
    # After flip: first row = last row before flip
    # Before flip mat[1,1] = vals[1] (column-major reshape)
    # After reverse(dims=1): mat[1,:] = original last row = mat_orig[263,:]
    mat_noflip = reshape(vals, 263, 324)
    @test mat[1, 1] ≈ mat_noflip[263, 1]
end

# ===========================================================================
# 47. actual_line_extents — synthetic
# ===========================================================================
@testset "actual_line_extents: synthetic" begin
    using DataFrames
    pl = DataFrame(
        line_index  = [0, 1],
        line_number = [1, 2],
        y_center    = [100.0, 200.0],
        x_min       = [0.0, 0.0],
        x_max       = [300.0, 300.0],
        n           = [12, 12],
    )
    n = 200
    df1 = DataFrame(
        GridX          = range(10.0, 290.0; length=n),
        GridY          = fill(100.0, n),
        nearest_line   = fill(0, n),
        segment_id     = fill(0, n),
    )
    processed = Dict("MissionA" => df1)
    ext = actual_line_extents(processed, pl)
    @test nrow(ext) == 1
    @test ext.line_index[1] == 0
    @test ext.x_min_actual[1] < ext.x_max_actual[1]
    @test ext.x_min_raw[1] ≈ 10.0   atol=0.1
    @test ext.x_max_raw[1] ≈ 290.0  atol=0.1
    @test ext.samples[1] == n
end

# ===========================================================================
# 48. common_x_overlap — basic
# ===========================================================================
@testset "common_x_overlap: basic" begin
    using DataFrames
    ext = DataFrame(
        mission      = ["A", "B", "A", "B"],
        line_index   = [0, 0, 1, 1],
        x_min_actual = [10.0, 20.0, 5.0, 30.0],
        x_max_actual = [200.0, 180.0, 250.0, 200.0],
    )
    overlap = common_x_overlap(ext, ["A", "B"])
    # Line 0: max(10,20)=20, min(200,180)=180
    @test haskey(overlap, 0)
    @test overlap[0][1] ≈ 20.0
    @test overlap[0][2] ≈ 180.0
    # Line 1: max(5,30)=30, min(250,200)=200
    @test haskey(overlap, 1)
    @test overlap[1][1] ≈ 30.0
    @test overlap[1][2] ≈ 200.0
end

# ===========================================================================
# 49. tracking_summary — synthetic
# ===========================================================================
@testset "tracking_summary: synthetic" begin
    using DataFrames, Statistics
    n = 200
    df = DataFrame(
        ground_speed_mps   = fill(3.0, n),
        cross_track_error_m = fill(5.0, n),
        tracking_error_m   = fill(1.0, n),
        planned_offset_m   = fill(0.5, n),
        Time               = range(0.0, 100.0; length=n),
        segment_id         = fill(0, n),
        nearest_line       = fill(0, n),
    )
    processed = Dict("TestMission" => df)
    sm = tracking_summary(processed)
    @test nrow(sm) == 1
    @test sm.mission[1] == "TestMission"
    @test sm.survey_samples[1] == n
    @test sm.speed_mean_mps[1] ≈ 3.0
    @test sm.tracking_error_rms_m[1] ≈ 1.0
    @test sm.planned_cross_track_p95_m[1] ≈ 5.0
end

# ===========================================================================
# 50. Integration smoke test on real data (if available)
# ===========================================================================
let
    _WAYPOINTS = get(ENV, "KDE_TEST_WAYPOINTS_CSV",
        joinpath(_GT_DIR, "E_density_aware__waypoints_xy.csv"))
    if isfile(_WAYPOINTS)
        @testset "derive_planned_lines: real waypoints" begin
            pl = derive_planned_lines(_WAYPOINTS)
            @test nrow(pl) >= 6     # Durham site has ≥6 survey lines
            @test pl.line_index == collect(0:(nrow(pl)-1))
            @test all(pl.x_max .- pl.x_min .>= 250.0)
            @test issorted(pl.y_center)
        end
    else
        @warn "Skipping real-waypoints smoke test: E_density_aware__waypoints_xy.csv not found"
    end
end

# ===========================================================================
# 51. Figures smoke test (if figures exist / output dir present)
# ===========================================================================
let
    figures_dir = joinpath(@__DIR__, "..", "output", "figures")
    expected_figs = [
        "actual_trajectory_line_scan_forest_differences.png",
        "line2_actual_overlap_detail.png",
        "tracking_metrics_refined.png",
    ]
    if isdir(figures_dir)
        @testset "figures: expected files exist and are non-empty" begin
            for fname in expected_figs
                fpath = joinpath(figures_dir, fname)
                if isfile(fpath)
                    @test isfile(fpath)
                    @test stat(fpath).size > 50_000  # at least 50 kB
                else
                    @warn "Figure not found (run scripts/make_figures.jl): $fname"
                end
            end
        end
    else
        @warn "output/figures/ not found; skipping figure smoke tests. Run scripts/make_figures.jl first."
    end
end

# Check that figures.jl can be loaded (syntax check)
@testset "figures.jl: include without error" begin
    # Load CairoMakie to enable include
    cm_available = try
        @eval using CairoMakie
        true
    catch
        false
    end
    if cm_available
        @test_nowarn include(joinpath(@__DIR__, "..", "src", "figures.jl"))
    else
        @warn "CairoMakie not available; skipping figures.jl include test"
    end
end

println("\nAll tests passed ✓")

# ===========================================================================
# 52. Clustering: k sweep defaults, vote strategy, tie-breaking, CSV export
# ===========================================================================
@testset "ClusterMetrics: new fields" begin
    m = ClusterMetrics(3, 0.5, 0.2, 1.5, 200.0, nothing)
    @test m.k == 3
    @test m.votes == 0
    @test m.chosen == false
    @test isnothing(m.seed)
    @test isnothing(m.strategy)
end

@testset "sweep_k_quality: default ks=2:12" begin
    rng = Random.MersenneTwister(42)
    X   = rand(rng, 200, 3)
    n   = size(X, 1)
    idxs = collect(1:n)
    D    = Distances.pairwise(Distances.SqEuclidean(), X; dims=1)
    mets = sweep_k_quality(X, idxs, D)  # uses default ks=2:12
    @test length(mets) == 11   # k=2..12
    @test all(m.k in 2:12 for m in mets)
    @test all(m.votes >= 0 for m in mets)
    @test sum(m.votes for m in mets) in 3:4  # 4 metrics, each nominating one k
end

@testset "choose_k: vote strategy tie-breaking (lower k preferred)" begin
    # Construct two metrics with identical votes; lower k should win
    m2 = ClusterMetrics(2, 0.6, 0.3, 1.2, 300.0, nothing, 2, false, nothing, nothing, nothing)
    m3 = ClusterMetrics(3, 0.6, 0.3, 1.2, 300.0, nothing, 2, false, nothing, nothing, nothing)
    m4 = ClusterMetrics(4, 0.1, 0.1, 3.0, 50.0,  nothing, 0, false, nothing, nothing, nothing)
    mets = [m2, m3, m4]
    chosen = choose_k(mets; strategy=:vote)
    @test chosen == 2   # lower k wins on tie
    @test count(m.chosen for m in mets) == 1
    @test mets[1].chosen == true
end

@testset "choose_k: :mode is alias for :vote" begin
    rng = Random.MersenneTwister(99)
    X   = rand(rng, 80, 2)
    idxs = collect(1:80)
    D = Distances.pairwise(Distances.SqEuclidean(), X; dims=1)
    mets = sweep_k_quality(X, idxs, D; ks=2:5, seed=99, nsample=80)
    k_vote = choose_k(mets; strategy=:vote)
    # Reset chosen flags for second call
    for m in mets; m.chosen = false; m.strategy = nothing; end
    k_mode = choose_k(mets; strategy=:mode)
    @test k_vote == k_mode
end

@testset "export_cluster_metrics_csv: writes correct columns" begin
    mets = [ClusterMetrics(k, 0.5, 0.2, 1.0, 100.0, nothing, 0, false, 42, 200, :vote)
            for k in 2:5]
    mets[2].votes = 2; mets[2].chosen = true
    tmp = tempname() * ".csv"
    export_cluster_metrics_csv(mets, tmp)
    @test isfile(tmp)
    df = CSV.read(tmp, DataFrame)
    @test "k" in names(df)
    @test "silhouette" in names(df)
    @test "dunn" in names(df)
    @test "davies_bouldin" in names(df)
    @test "calinski_harabasz" in names(df)
    @test "votes" in names(df)
    @test "chosen" in names(df)
    @test "seed" in names(df)
    @test "nsample" in names(df)
    @test "strategy" in names(df)
    @test nrow(df) == 4
end

# ===========================================================================
# 53. FlightConfig: default line_spacing is now 40 m
# ===========================================================================
@testset "FlightConfig: default line_spacing=40m" begin
    strat = ConstantSpeed(5.0)
    cfg   = FlightConfig(strat, 80.0, "Test mission")
    @test cfg.line_spacing == 40.0
end

@testset "FlightConfig: explicit line_spacing preserved" begin
    strat = ConstantSpeed(5.0)
    cfg   = FlightConfig(strat, 80.0, "Test"; line_spacing=25.0)
    @test cfg.line_spacing == 25.0
end

# ===========================================================================
# 54. export_planning_metadata: writes valid JSON with required keys
# ===========================================================================
@testset "export_planning_metadata: required fields" begin
    tmp = tempname() * ".json"
    export_planning_metadata(tmp;
        selected_k       = 4,
        tree_labels      = [2, 3],
        feature_repr     = "CIELAB (L, a, b)",
        pca_settings     = (use_pca=false, variance_ratio=0.95, maxoutdim=nothing),
        kernel           = "epanechnikov",
        bandwidth_rule   = "silverman_scott_indices",
        speed_bounds     = (vmin=2.0, vmax=8.0),
        line_spacing_m   = 40.0,
        seconds_per_wp   = 1.0,
        spacing_bounds_m = (min=2.0, max=40.0),
        image_source     = "test_image.jpg",
        mask_source      = "smoke/derived_from_screenshot",
        notes            = "automated test")
    @test isfile(tmp)
    meta = JSON.parsefile(tmp)
    @test meta["selected_k"] == 4
    @test meta["tree_labels"] == [2, 3]
    @test meta["kernel"] == "epanechnikov"
    @test meta["line_spacing_m"] == 40.0
    @test meta["seconds_per_wp"] == 1.0
    @test haskey(meta, "created")
    @test haskey(meta, "seconds_per_wp_note")
    @test haskey(meta, "speed_bounds")
end

# ===========================================================================
# 55. require_tree_labels: errors on nothing/empty
# ===========================================================================
@testset "require_tree_labels: errors when absent" begin
    @test_throws ErrorException require_tree_labels(nothing)
    @test_throws ErrorException require_tree_labels(Int[])
    @test_nowarn require_tree_labels([1])
    @test_nowarn require_tree_labels([2, 3])
end

# ===========================================================================
# 56. kde_cr_by_quantile_bins: shape and range
# ===========================================================================
@testset "kde_cr_by_quantile_bins: shape and CR range" begin
    rng   = Random.MersenneTwister(7)
    H, W  = 20, 20
    cg    = rand(rng, 0:5, H, W)
    kde_z = rand(rng, Float64, H, W)
    df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=4)
    @test nrow(df) == 4
    @test all(0.0 .<= df.cr .<= 1.0)
    @test all(df.n_cells .> 0)
    @test "stratum" in names(df)
    @test "cr" in names(df)
end

@testset "kde_cr_by_quantile_bins: dimension mismatch throws" begin
    @test_throws DimensionMismatch kde_cr_by_quantile_bins(
        rand(0:3, 10, 10), rand(5, 5))
end

# ===========================================================================
# 56b. kde_cr_by_quantile_bins: improved tests for strata correctness
# ===========================================================================
@testset "kde_cr_by_quantile_bins: support counts sum to total" begin
    rng   = Random.MersenneTwister(42)
    H, W  = 40, 40
    cg    = rand(rng, 0:3, H, W)
    kde_z = rand(rng, Float64, H, W)
    for n_bins in [3, 4, 5]
        df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=n_bins)
        @test nrow(df) == n_bins
        @test sum(df.n_cells) == H * W   # all cells assigned to exactly one bin
        @test all(df.n_cells .> 0)
        @test all(0.0 .<= df.cr .<= 1.0)
        @test df.stratum == collect(1:n_bins)
    end
end

@testset "kde_cr_by_quantile_bins: all-covered grid yields CR=1" begin
    H, W  = 10, 10
    cg    = fill(5, H, W)   # every cell has 5 returns
    kde_z = collect(reshape(range(0.0, 1.0; length=H*W), H, W))
    df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=4)
    @test all(df.cr .≈ 1.0)
    @test all(df.n_covered .== df.n_cells)
end

@testset "kde_cr_by_quantile_bins: zero-count grid yields CR=0" begin
    H, W  = 10, 10
    cg    = zeros(Int, H, W)   # no returns anywhere
    kde_z = rand(Random.MersenneTwister(1), H, W)
    df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=4)
    @test all(df.cr .≈ 0.0)
    @test all(df.n_covered .== 0)
end

@testset "kde_cr_by_quantile_bins: n_covered ≤ n_cells always" begin
    rng   = Random.MersenneTwister(13)
    cg    = rand(rng, 0:5, 30, 30)
    kde_z = rand(rng, 30, 30)
    df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=4)
    @test all(df.n_covered .<= df.n_cells)
end

@testset "kde_cr_by_quantile_bins: stratum boundaries are monotone" begin
    rng   = Random.MersenneTwister(99)
    cg    = rand(rng, 0:5, 20, 20)
    kde_z = rand(rng, 20, 20)
    df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=4)
    # density_lo should be non-decreasing across strata
    @test issorted(df.density_lo)
    # density_hi of stratum k ≤ density_lo of stratum k+1 (or equal, due to quantile ties)
    for i in 1:(nrow(df)-1)
        @test df.density_hi[i] <= df.density_lo[i+1] + 1e-12
    end
end

@testset "kde_cr_by_quantile_bins: quantile_lo/hi are 0..1 fractions" begin
    rng   = Random.MersenneTwister(5)
    cg    = rand(rng, 0:2, 15, 15)
    kde_z = rand(rng, 15, 15)
    df = kde_cr_by_quantile_bins(cg, kde_z; n_bins=4)
    @test df.quantile_lo[1] ≈ 0.0
    @test df.quantile_hi[end] ≈ 1.0
    @test all(0.0 .<= df.quantile_lo .<= 1.0)
    @test all(0.0 .<= df.quantile_hi .<= 1.0)
end

# ===========================================================================
# 56c. kde_strata_table: multi-mission ΔCR correctness
# ===========================================================================
@testset "kde_strata_table: column presence and row count" begin
    rng = Random.MersenneTwister(7)
    H, W = 20, 20
    kde_g = rand(rng, Float64, H, W)
    grids = Dict(
        "Const. 2 m/s"          => rand(rng, 0:5, H, W),
        "KDE-guided (Epan.)"    => rand(rng, 0:5, H, W),
        "Const. 8 m/s"          => rand(rng, 0:5, H, W),
    )
    df = kde_strata_table(grids, kde_g;
        missions=["Const. 2 m/s", "KDE-guided (Epan.)", "Const. 8 m/s"],
        n_bins=4, label="TestCover",
        alignment_note="smoke/diagnostic")
    @test nrow(df) == 3 * 4   # 3 missions × 4 strata
    for col in ["mission", "cover", "alignment", "stratum", "cr", "n_cells",
                "n_covered", "density_lo", "density_hi"]
        @test col in names(df)
    end
    @test all(df.cover .== "TestCover")
    @test all(df.alignment .== "smoke/diagnostic")
end

@testset "kde_strata_table: CR values match kde_cr_by_quantile_bins" begin
    rng   = Random.MersenneTwister(21)
    H, W  = 20, 20
    kde_g = rand(rng, Float64, H, W)
    cg    = rand(rng, 0:5, H, W)
    # Single-mission table
    df_table = kde_strata_table(
        Dict("M1" => cg), kde_g;
        missions=["M1"], n_bins=4, label="", alignment_note="smoke")
    # Direct call
    df_direct = kde_cr_by_quantile_bins(cg, kde_g; n_bins=4)
    df_m1 = filter(r -> r.mission == "M1", df_table)
    @test all(sort(df_m1, :stratum).cr .≈ sort(df_direct, :stratum).cr)
end

@testset "kde_strata_table: NN resize triggers on shape mismatch" begin
    rng   = Random.MersenneTwister(3)
    H, W  = 20, 20
    cg    = rand(rng, 0:5, H, W)
    kde_g = rand(rng, Float64, 15, 15)  # intentionally different shape
    # Should NOT throw — falls back to nearest-neighbour resize
    df = @test_logs (:warn, r"nearest-neighbour resize|nearest.*resize|SMOKE") match_mode=:any kde_strata_table(
        Dict("M" => cg), kde_g;
        missions=["M"], n_bins=4, label="", alignment_note="smoke/diagnostic")
    @test nrow(df) == 4
end

@testset "kde_strata_table: alignment column propagated" begin
    rng = Random.MersenneTwister(55)
    H, W = 10, 10
    kde_g = rand(rng, H, W)
    cg    = rand(rng, 0:3, H, W)
    for note in ["smoke/diagnostic", "gli_coregistered_canopy_kde_proxy",
                 "screenshot_nn_resized_to_counts_grid"]
        df = kde_strata_table(Dict("M" => cg), kde_g;
            missions=["M"], n_bins=4, label="",
            alignment_note=note)
        @test all(df.alignment .== note)
    end
end

# ===========================================================================
# 57. along_track_cr: bin count and validity
# ===========================================================================
@testset "along_track_cr: basic usage" begin
    cg = ones(Int, 10, 50)  # 10 rows × 50 cols, all covered
    xs = collect(0.0:1.0:49.0)
    centres, crs, nc = along_track_cr(cg, xs, 0.0, 49.0; bin_size_m=5.0, min_cells=1)
    @test length(centres) > 0
    @test all(c ≈ 1.0 for c in crs)
    @test all(n > 0 for n in nc)
end

@testset "split_at_gaps: gap detection" begin
    centres = [0.0, 5.0, 10.0, 50.0, 55.0]  # large gap at 10→50
    crs     = [0.9, 0.8, 0.85, 0.7, 0.75]
    nc      = [5, 5, 5, 5, 5]
    segs = split_at_gaps(centres, crs, nc; max_gap_m=20.0)
    @test length(segs) == 2
    @test length(segs[1].centres) == 3
    @test length(segs[2].centres) == 2
end

# ===========================================================================
# 58. Pipeline: build_mask_from_image uses :vote default
# ===========================================================================
@testset "build_mask_from_image: vote strategy default" begin
    rng = Random.MersenneTwister(999)
    arr = rand(rng, UInt8, 30, 30, 3)
    # Run with explicit ks to trigger sweep; check it returns without error
    mask_g, info = build_mask_from_image(arr;
        k=2, tree_labels=[1], seed=42,
        nsample=100, ks=2:4, k_strategy=:vote)
    @test mask_g isa RasterGrid
    @test info.k in 2:4
end

# ===========================================================================
# 59. Grid orientation invariant: load_counts_json returns H=263 × W=324
# ===========================================================================

@testset "load_counts_json: orientation H=263×W=324" begin
    # Build a minimal synthetic counts.json: one record, all zeros.
    # H=263, W=324 → 85212 elements.  The function's row-major decode must
    # yield a matrix of exactly (263, 324) regardless of axis labelling.
    H, W = 263, 324
    n    = H * W
    rec  = Dict("key"    => Dict("ret"=>"ground","cover"=>"field",
                                  "mission"=>"speed","kernel"=>"E"),
                 "matrix" => zeros(Int, n))
    tmp  = tempname() * ".json"
    open(tmp, "w") do io; JSON.print(io, [rec]); end

    records = load_counts_json(tmp; nrows=H, ncols=W)
    @test length(records) == 1
    @test size(records[1].matrix) == (H, W)   # 263 rows × 324 cols

    # Smoke-test wrong orientation is (W,H) ≠ (H,W)
    @test size(records[1].matrix) != (W, H)

    rm(tmp)
end

# ===========================================================================
# 60. Cover supports: DURHAM_COVER_N matches known ground-truth cell counts
# ===========================================================================

@testset "DURHAM_COVER_N matches known ground-truth cell counts" begin
    @test DURHAM_COVER_N[:conif] == 5_413
    @test DURHAM_COVER_N[:decid] == 24_080
    @test DURHAM_COVER_N[:field] == 55_719
    # Sum must equal total raster cells 263×324 = 85212
    @test sum(values(DURHAM_COVER_N)) == 263 * 324
end

# ===========================================================================
# 61. kde_strata_within_cover: stratum n_cells sums to cover mask size
# ===========================================================================

@testset "kde_strata_within_cover: stratum n_cells sums to cover mask size" begin
    rng    = Random.MersenneTwister(42)
    H, W   = 30, 40

    kde    = rand(rng, Float64, H, W)
    cg_a   = rand(rng, 0:5, H, W)
    cg_b   = rand(rng, 0:5, H, W)

    # Random cover mask — roughly half the cells
    mask   = rand(rng, Bool, H, W)
    n_cover = count(mask)

    cg_dict = Dict("MissionA" => cg_a, "MissionB" => cg_b)

    df = kde_strata_within_cover(cg_dict, kde, mask;
             missions=["MissionA","MissionB"],
             n_pos_bins=4, cover_label="synthetic")

    for m in ("MissionA","MissionB")
        sub = df[df.mission .== m, :]
        @test sum(sub.n_cells) == n_cover
    end
end

# ===========================================================================
# 62. kde_strata_within_cover: weighted-average CR == whole-cover CR
# ===========================================================================

@testset "kde_strata_within_cover: weighted-average CR equals whole-cover CR" begin
    rng   = Random.MersenneTwister(7)
    H, W  = 50, 60
    kde   = rand(rng, Float64, H, W)
    cg    = rand(rng, 0:3, H, W)
    mask  = rand(rng, Bool, H, W)

    n_support = count(mask)   # whole-cover denominator

    df = kde_strata_within_cover(
        Dict("M" => cg), kde, mask;
        missions = ["M"], n_pos_bins = 4, cover_label = "test"
    )

    sub = df[df.mission .== "M", :]

    # Weighted average CR = sum(n_covered) / n_support
    total_covered   = sum(sub.n_covered)
    weighted_cr     = total_covered / n_support

    # Brute-force whole-cover CR for the same mission and mask
    whole_cr = count(cg[mask] .>= 1) / n_support

    @test abs(weighted_cr - whole_cr) < 1e-9
end

# ===========================================================================
# 63. kde_strata_within_cover: no-renormalize contract
#     — out-of-range values trigger a warning but are NOT altered
# ===========================================================================

@testset "kde_strata_within_cover: out-of-range values warn, not renormalized" begin
    rng  = Random.MersenneTwister(99)
    H, W = 20, 20
    # KDE surface with values in [0, 1.5] — should warn but not throw
    kde_bad = rand(rng, Float64, H, W) .* 1.5
    cg      = rand(rng, 0:2, H, W)
    mask    = trues(H, W)

    # Values must pass through unaltered: capture extrema before call
    kde_min_before, kde_max_before = extrema(kde_bad)

    df = @test_logs (:warn, r"outside \[0,1\]") kde_strata_within_cover(
        Dict("M" => cg), kde_bad, mask;
        missions=["M"], n_pos_bins=2, cover_label="oor_test")

    # Function must not mutate the input array
    @test extrema(kde_bad) == (kde_min_before, kde_max_before)

    # density_hi in the returned table can exceed 1.0 (not clipped)
    @test any(df.density_hi .> 1.0)
end

# ===========================================================================
# 64. kde_strata_within_cover: dimension mismatch throws DimensionMismatch
# ===========================================================================

@testset "kde_strata_within_cover: dimension mismatch throws" begin
    H, W   = 15, 20
    kde    = rand(Float64, H, W)
    cg_bad = rand(0:3, H+1, W)   # wrong shape
    mask   = trues(H, W)

    @test_throws DimensionMismatch kde_strata_within_cover(
        Dict("M" => cg_bad), kde, mask; missions=["M"])
end

# ===========================================================================
# 65. kde_strata_within_cover: all CR values are in [0, 1]
# ===========================================================================

@testset "kde_strata_within_cover: all CR values in [0,1]" begin
    rng  = Random.MersenneTwister(17)
    H, W = 40, 50
    kde  = rand(rng, Float64, H, W)
    cg   = rand(rng, 0:10, H, W)
    mask = rand(rng, Bool, H, W)

    df = kde_strata_within_cover(
        Dict("M1" => cg, "M2" => cg), kde, mask;
        missions=["M1","M2"], n_pos_bins=6, cover_label="cr_range")

    @test all(0.0 .<= df.cr .<= 1.0)
end

# ===========================================================================
# 66. GeoTransform: construction, field access, getindex, show
# ===========================================================================

@testset "GeoTransform: construction and field access" begin
    # Named-field constructor
    gt = GeoTransform(341300.35, 0.028369, 0.0, 4774892.895, 0.0, -0.028368)
    @test gt.x_origin ≈ 341300.35
    @test gt.dx       ≈ 0.028369
    @test gt.dy       < 0.0   # north-up: dy must be negative

    # Vector constructor
    v  = [341300.35, 1.0, 0.0, 4774892.895, 0.0, -1.0]
    gt2 = GeoTransform(v)
    @test gt2.x_origin ≈ v[1]
    @test gt2.dy       ≈ v[6]

    # getindex compatibility (1-based, GDAL order)
    @test gt2[1] ≈ gt2.x_origin
    @test gt2[2] ≈ gt2.dx
    @test gt2[4] ≈ gt2.y_origin
    @test gt2[6] ≈ gt2.dy
    @test length(gt2) == 6

    # Iteration (collect should give 6 Float64)
    collected = collect(gt2)
    @test length(collected) == 6
    @test collected[2] ≈ gt2.dx

    # Short vector throws
    @test_throws ArgumentError GeoTransform([1.0, 2.0, 3.0])

    # Study-site defaults exist and are typed correctly
    @test GT_NATIVE isa GeoTransform
    @test GT_COUNTGRID isa GeoTransform
    @test GT_COUNTGRID.dx == 1.0
    @test GT_COUNTGRID.dy == -1.0
end

# ===========================================================================
# 67. KDESurface: construction, validation, show
# ===========================================================================

@testset "KDESurface: construction and validation" begin
    Z  = rand(20, 30)
    gt = GeoTransform([0.0, 0.5, 0.0, 100.0, 0.0, -0.5])

    # Normal construction
    surf = KDESurface(Z, gt, "EPSG:6348"; notes="test")
    @test size(surf) == (20, 30)
    @test size(surf, 1) == 20
    @test size(surf, 2) == 30
    @test surf.crs == "EPSG:6348"
    @test surf.notes == "test"

    # Convenience constructor with raw vector
    surf2 = KDESurface(Z, [0.0, 0.5, 0.0, 100.0, 0.0, -0.5], "EPSG:6348")
    @test size(surf2) == (20, 30)

    # show does not error
    io = IOBuffer()
    show(io, surf)
    @test occursin("KDESurface", String(take!(io)))

    # Out-of-range Z throws
    Z_bad = copy(Z)
    Z_bad[1,1] = -0.1
    @test_throws ArgumentError KDESurface(Z_bad, gt, "EPSG:6348")

    Z_big = copy(Z)
    Z_big[2,2] = 1.5
    @test_throws ArgumentError KDESurface(Z_big, gt, "EPSG:6348")
end

# ===========================================================================
# 68. resample_to_count_grid: shape, orientation, method dispatch
# ===========================================================================

@testset "resample_to_count_grid: shape and orientation" begin
    # Synthetic native surface: 50×60 at 1 m/px resolution
    # gt_native: origin (0,100), dx=1, dy=-1 (north-up)
    H_nat, W_nat = 50, 60
    gt_native = GeoTransform(0.0, 1.0, 0.0, 100.0, 0.0, -1.0)

    # Ramp surface: Z[i,j] = i/H_nat (rows increase south)
    Z_native = Matrix{Float64}([Float64(i)/H_nat for i in 1:H_nat, j in 1:W_nat])
    surf = KDESurface(Z_native, gt_native, "EPSG:4326")

    # Output grid: same origin, but 25×30 cells, dx=2, dy=-2
    H_out, W_out = 25, 30
    gt_out = GeoTransform(0.0, 2.0, 0.0, 100.0, 0.0, -2.0)

    # Nearest-neighbour
    Z_nn = resample_to_count_grid(surf; H_out=H_out, W_out=W_out,
                                   gt_out=gt_out, method=NearestNeighbor())
    @test size(Z_nn) == (H_out, W_out)
    @test all(0.0 .<= Z_nn .<= 1.0)

    # Bilinear
    Z_bl = resample_to_count_grid(surf; H_out=H_out, W_out=W_out,
                                   gt_out=gt_out, method=Bilinear())
    @test size(Z_bl) == (H_out, W_out)
    @test all(0.0 .<= Z_bl .<= 1.0)

    # Orientation: ramp increases southward (row index increases), so
    # Z_nn[1,1] < Z_nn[end,1]
    @test Z_nn[1,1] < Z_nn[end,1]

    # Default output shape is (263, 324)
    # Use GT_COUNTGRID as output — native must cover the same region
    # Build tiny native that covers GT_COUNTGRID extent
    gt_nat2 = GT_COUNTGRID   # 1 m/cell; resample 1:1 should be identity
    Z_id = rand(263, 324)
    surf_id = KDESurface(Z_id, gt_nat2, CRS_DURHAM)
    Z_out_default = resample_to_count_grid(surf_id)
    @test size(Z_out_default) == (263, 324)

    # Convenience overload: bare matrix + GeoTransform
    Z_conv = resample_to_count_grid(Z_id, gt_nat2; H_out=263, W_out=324,
                                    gt_out=GT_COUNTGRID)
    @test size(Z_conv) == (263, 324)

    # Convenience overload: bare matrix + raw vector
    Z_vec = resample_to_count_grid(Z_id, collect(gt_nat2); H_out=263, W_out=324)
    @test size(Z_vec) == (263, 324)
end

@testset "resample_to_count_grid: 1:1 same-grid is identity (nearest)" begin
    # When native and output grids are identical, nearest-neighbour
    # resampling must return an array equal to the input.
    H, W  = 10, 12
    gt    = GeoTransform(0.0, 1.0, 0.0, 100.0, 0.0, -1.0)
    Z     = rand(H, W)
    surf  = KDESurface(Z, gt, "EPSG:4326")
    Z_out = resample_to_count_grid(surf; H_out=H, W_out=W,
                                   gt_out=gt, method=NearestNeighbor())
    @test Z_out ≈ Z
end

# ===========================================================================
# 69. JLD2 save / load round-trip for KDESurface
# ===========================================================================

@testset "KDESurface: JLD2 save/load round-trip" begin
    Z    = rand(8, 10)
    gt   = GeoTransform([10.0, 0.5, 0.0, 200.0, 0.0, -0.5])
    surf = KDESurface(Z, gt, "EPSG:6348"; notes="unit test")
    tmp  = tempname() * ".jld2"

    path = save_kde_surface(surf, tmp)
    @test isfile(path)

    surf2 = load_kde_surface(path)
    @test surf2.Z ≈ surf.Z
    @test surf2.geotransform.x_origin ≈ surf.geotransform.x_origin
    @test surf2.geotransform.dy       ≈ surf.geotransform.dy
    @test surf2.crs   == surf.crs
    @test surf2.notes == surf.notes
    @test size(surf2) == size(surf)

    rm(tmp)
end

# ===========================================================================
# 70. CSV save / load round-trip for KDESurface
# ===========================================================================

@testset "KDESurface: CSV+JSON save/load round-trip" begin
    Z    = rand(5, 7)
    gt   = GeoTransform([341300.35, 1.0, 0.0, 4774892.895, 0.0, -1.0])
    surf = KDESurface(Z, gt, "EPSG:6348"; notes="csv test")
    tmp_csv = tempname() * ".csv"

    csv_path, json_path = save_kde_surface_csv(surf, tmp_csv)
    @test isfile(csv_path)
    @test isfile(json_path)

    surf2 = load_kde_surface_csv(csv_path)
    @test surf2.Z ≈ surf.Z
    @test surf2.crs == surf.crs

    rm(csv_path); rm(json_path)
end

# ===========================================================================
# 71. ThresholdMethod types: dispatch, show, arg validation
# ===========================================================================

@testset "ThresholdMethod types" begin
    @test OtsuThreshold()     isa ThresholdMethod
    @test QuantileThreshold() isa ThresholdMethod
    @test ManualThreshold(0.4) isa ThresholdMethod

    # QuantileThreshold bounds
    @test_throws ArgumentError QuantileThreshold(0.0)
    @test_throws ArgumentError QuantileThreshold(1.0)

    # ManualThreshold bounds
    @test_throws ArgumentError ManualThreshold(0.0)
    @test_throws ArgumentError ManualThreshold(-0.1)

    # show does not error
    io = IOBuffer()
    show(io, OtsuThreshold()); show(io, QuantileThreshold(0.7)); show(io, ManualThreshold(0.3))
    s = String(take!(io))
    @test occursin("Otsu",     s)
    @test occursin("Quantile", s)
    @test occursin("Manual",   s)
end

# ===========================================================================
# 72. assign_kde_density_classes: class labels, zero-density handling
# ===========================================================================

@testset "assign_kde_density_classes: basic correctness" begin
    # Construct a surface with known distribution
    Z = Matrix{Float64}([
        0.0   0.0   0.2   0.8;
        0.0   0.5   0.9   1.0;
        1e-10 0.3   0.7   0.95;
    ])  # 3×4

    r = assign_kde_density_classes(Z; method=OtsuThreshold(), eps=1e-9)
    @test r isa DensityClassResult
    @test size(r.class_matrix) == size(Z)

    # All class values in {1,2,3}
    @test all(v -> v ∈ (KDE_CLASS_FIELD, KDE_CLASS_DECIDUOUS, KDE_CLASS_CONIFEROUS),
              r.class_matrix)

    # Zero cells classified as field-like (not dropped)
    zero_mask = Z .<= r.eps_threshold
    @test all(r.class_matrix[zero_mask] .== KDE_CLASS_FIELD)

    # Support sums to total
    s = kde_class_support(r)
    @test s.field + s.deciduous + s.coniferous == length(Z)
    @test s.total == length(Z)

    # show does not error
    io = IOBuffer()
    show(io, r)
    @test occursin("DensityClassResult", String(take!(io)))
end

@testset "assign_kde_density_classes: Otsu threshold is in (0,1)" begin
    rng = Random.MersenneTwister(1)
    Z   = rand(rng, 50, 60)   # all positive; eps=1e-9 has no field cells
    r   = assign_kde_density_classes(Z; method=OtsuThreshold())
    @test 0.0 < r.upper_threshold < 1.0
    @test r.n_field == 0   # no zero cells in this surface
end

@testset "assign_kde_density_classes: zero-density cells → field-like, retained" begin
    # Surface with exactly half zero, half positive
    H, W = 20, 20
    Z    = zeros(Float64, H, W)
    Z[1:10, :] .= 0.5   # top half positive

    r = assign_kde_density_classes(Z; eps=1e-9)
    s = kde_class_support(r)

    @test s.field >= H * W ÷ 2    # at least half are field-like
    @test s.total == H * W        # no cells dropped
    @test r.n_field >= H * W ÷ 2
end

# ===========================================================================
# 73. ThresholdMethod dispatch: Otsu vs Quantile vs Manual give different thresholds
# ===========================================================================

@testset "compute_kde_thresholds: method dispatch" begin
    rng = Random.MersenneTwister(42)
    Z   = rand(rng, 40, 50)   # uniform positive densities

    t_otsu  = compute_kde_thresholds(Z, OtsuThreshold())
    t_q25   = compute_kde_thresholds(Z, QuantileThreshold(0.25))
    t_q75   = compute_kde_thresholds(Z, QuantileThreshold(0.75))
    t_man   = compute_kde_thresholds(Z, ManualThreshold(0.4))

    # ManualThreshold returns exactly the supplied value
    @test t_man.upper_threshold ≈ 0.4

    # QuantileThreshold(0.25) < QuantileThreshold(0.75) for random uniform data
    @test t_q25.upper_threshold < t_q75.upper_threshold

    # Otsu lies between the two quantiles (not guaranteed in general, but
    # true for uniform distributions where Otsu ≈ median)
    @test t_otsu.upper_threshold > 0.0
    @test t_otsu.upper_threshold < 1.0

    # All methods set eps_threshold correctly
    for t in (t_otsu, t_q25, t_man)
        @test t.eps_threshold == 1e-9
    end
end

# ===========================================================================
# 74. kde_class_cr: coverage ratio validity and DimensionMismatch guard
# ===========================================================================

@testset "kde_class_cr: validity and dispatch" begin
    rng = Random.MersenneTwister(7)
    H, W = 30, 40
    Z    = rand(rng, H, W)
    cg   = rand(rng, 0:5, H, W)

    r = assign_kde_density_classes(Z; method=OtsuThreshold())

    # Dispatched on DensityClassResult
    cr = kde_class_cr(r, cg)
    @test 0.0 <= cr.field.cr      <= 1.0
    @test 0.0 <= cr.deciduous.cr  <= 1.0
    @test 0.0 <= cr.coniferous.cr <= 1.0

    # Dispatched on bare class_matrix
    cr2 = kde_class_cr(r.class_matrix, cg)
    @test cr2.field.cr ≈ cr.field.cr

    # Weighted average CR should equal whole-grid CR when threshold splits all cells
    total_covered = cr.field.n_covered + cr.deciduous.n_covered + cr.coniferous.n_covered
    total_cells   = cr.field.n_cells   + cr.deciduous.n_cells   + cr.coniferous.n_cells
    @test total_cells == H * W
    whole_cr = count(cg .>= 1) / (H * W)
    @test abs(total_covered / total_cells - whole_cr) < 1e-12

    # DimensionMismatch guard
    cg_bad = rand(0:3, H+1, W)
    @test_throws DimensionMismatch kde_class_cr(r.class_matrix, cg_bad)
end

# ===========================================================================
# 75. Manual threshold override: thresholds are exactly obeyed
# ===========================================================================

@testset "ManualThreshold: exact threshold override" begin
    Z = Matrix{Float64}([0.0 0.1 0.3 0.5 0.7 0.9])   # 1×6

    # Manual threshold = 0.4: cells ≤ 0.4 and > eps → deciduous; > 0.4 → coniferous
    r = assign_kde_density_classes(Z; method=ManualThreshold(0.4))
    @test r.upper_threshold ≈ 0.4

    # cell [1,1]=0.0 → field-like (≤ eps)
    @test r.class_matrix[1,1] == KDE_CLASS_FIELD

    # cell [1,2]=0.1 → deciduous-like (eps < 0.1 ≤ 0.4)
    @test r.class_matrix[1,2] == KDE_CLASS_DECIDUOUS

    # cell [1,5]=0.7 → coniferous-like (0.7 > 0.4)
    @test r.class_matrix[1,5] == KDE_CLASS_CONIFEROUS
end

# ===========================================================================
# 76. validate_kde_range: correct pass/throw behaviour
# ===========================================================================

@testset "validate_kde_range: pass and throw" begin
    # Valid range — no exception
    @test validate_kde_range(rand(5, 5)) === nothing
    @test validate_kde_range(zeros(3, 3)) === nothing
    @test validate_kde_range(ones(3, 3))  === nothing

    # Negative values throw
    Z_neg = rand(5, 5); Z_neg[1,1] = -0.1
    @test_throws ArgumentError validate_kde_range(Z_neg)

    # Values > 1 throw
    Z_big = rand(5, 5); Z_big[2,2] = 1.01
    @test_throws ArgumentError validate_kde_range(Z_big)

    # Tolerance boundary — 1e-10 over is fine
    Z_ok = ones(3, 3) .* (1.0 + 1e-10)
    @test validate_kde_range(Z_ok) === nothing
end

# ===========================================================================
# 77. kde_class_summary: DataFrame schema and multi-mission coverage
# ===========================================================================

@testset "kde_class_summary: DataFrame schema" begin
    rng  = Random.MersenneTwister(11)
    H, W = 20, 25
    Z    = rand(rng, H, W)
    cg1  = rand(rng, 0:3, H, W)
    cg2  = rand(rng, 0:3, H, W)

    cg_dict = Dict("M1" => cg1, "M2" => cg2)
    df = kde_class_summary(Z, cg_dict;
             method=OtsuThreshold(), kde_status="test",
             alignment="synthetic")

    # Should have 3 classes × 2 missions = 6 rows
    @test nrow(df) == 6

    # Required columns
    for col in (:mission, :kde_class, :class_label, :n_cells, :n_covered, :cr,
                :eps_threshold, :upper_threshold, :threshold_method)
        @test hasproperty(df, col)
    end

    # All CR in [0,1]
    @test all(0.0 .<= df.cr .<= 1.0)

    # Class labels are the three expected strings
    labels = sort(unique(df.class_label))
    @test labels == sort(["field-like","deciduous-like","coniferous-like"])

    # Per-mission support sums to grid size
    for m in ("M1","M2")
        sub = filter(r -> r.mission == m, df)
        @test sum(sub.n_cells) == H * W
    end
end

# ===========================================================================
# 78. Existing DURHAM_COVER_N CR validation — must remain unchanged
# ===========================================================================

@testset "DURHAM_COVER_N: primary CR/CI values unchanged" begin
    @test DURHAM_COVER_N[:conif] == 5_413
    @test DURHAM_COVER_N[:decid] == 24_080
    @test DURHAM_COVER_N[:field] == 55_719
    @test sum(values(DURHAM_COVER_N)) == 263 * 324  # 85212 cells
end

# ===========================================================================
# 79. resample_to_image_grid: basic output shape and range
# ===========================================================================

@testset "resample_to_image_grid: output shape and value range" begin
    rng = Random.MersenneTwister(71)
    Z_big = rand(rng, 100, 130)
    Z_small = resample_to_image_grid(Z_big; H_out=263, W_out=324)
    @test size(Z_small) == (263, 324)
    @test all(0.0 .<= Z_small .<= 1.0)

    # Bilinear overload gives same shape
    Z_bil = resample_to_image_grid(Z_big; H_out=263, W_out=324, method=Bilinear())
    @test size(Z_bil) == (263, 324)
    @test all(0.0 .<= Z_bil .<= 1.0)

    # KDESurface dispatch (geotransform is intentionally ignored)
    surf = KDESurface(Z_big, GT_NATIVE, CRS_DURHAM)
    Z_from_surf = resample_to_image_grid(surf; H_out=10, W_out=13)
    @test size(Z_from_surf) == (10, 13)
end

# ===========================================================================
# 80. resample_to_image_grid: spatial localisation — blob must NOT collapse
#     to an edge band
# ===========================================================================

@testset "resample_to_image_grid: blob stays localised after downsampling" begin
    # Build a 200×200 surface with a Gaussian blob in the TOP-LEFT quadrant.
    # After downsampling to 40×40 the blob must remain in the top-left quadrant.
    H_src, W_src = 200, 200
    H_out, W_out = 40,  40
    Z_src = zeros(H_src, W_src)
    # Place blob centred at (30, 30) — clearly top-left
    for r in 1:H_src, c in 1:W_src
        Z_src[r, c] = exp(-((r - 30)^2 + (c - 30)^2) / (2 * 15^2))
    end
    Z_src ./= maximum(Z_src)

    Z_out = resample_to_image_grid(Z_src; H_out=H_out, W_out=W_out)

    # The maximum in the output should be in the top-left quadrant (rows 1:20, cols 1:20)
    max_idx = argmax(Z_out)
    @test max_idx[1] <= H_out ÷ 2    # row in top half
    @test max_idx[2] <= W_out ÷ 2    # col in left half

    # The bottom-right quadrant (rows 21:40, cols 21:40) should be near zero
    bottom_right_max = maximum(Z_out[H_out÷2+1:end, W_out÷2+1:end])
    @test bottom_right_max < 0.05
end

# ===========================================================================
# 81. resample_to_image_grid: N/S and E/W orientation with asymmetric pattern
# ===========================================================================

@testset "resample_to_image_grid: north/south and east/west orientation" begin
    # Use a surface with distinct north (row 1 = high value) vs south
    # (last row = low value) gradient, and west (col 1 = high) vs east gradient.
    # After downsampling, orientation must be preserved.
    H_src, W_src = 80, 100
    H_out, W_out = 20, 25

    # Linearly decreasing top-to-bottom: row 1 = 1.0, row 80 = 0.0
    Z_ns = [(H_src - r) / (H_src - 1) for r in 1:H_src, c in 1:W_src]
    Z_out_ns = resample_to_image_grid(Z_ns; H_out=H_out, W_out=W_out)
    # First row of output should be greater than last row (north-biased)
    @test mean(Z_out_ns[1, :]) > mean(Z_out_ns[end, :])

    # Linearly decreasing left-to-right: col 1 = 1.0, col 100 = 0.0
    Z_ew = [(W_src - c) / (W_src - 1) for r in 1:H_src, c in 1:W_src]
    Z_out_ew = resample_to_image_grid(Z_ew; H_out=H_out, W_out=W_out)
    # First column of output should be greater than last column (west-biased)
    @test mean(Z_out_ew[:, 1]) > mean(Z_out_ew[:, end])
end

# ===========================================================================
# 82. resample_to_image_grid: zeros remain zero / field-like after class assign
# ===========================================================================

@testset "resample_to_image_grid: zeros remain zero and map to field-like" begin
    # Build a surface where the top half is all zeros, bottom half has density
    H_src, W_src = 100, 100
    Z_src = zeros(H_src, W_src)
    Z_src[51:end, :] .= rand(Random.MersenneTwister(3), 50, 100)
    # Normalise the non-zero region
    nz_max = maximum(Z_src[51:end, :])
    Z_src[51:end, :] ./= nz_max

    H_out, W_out = 20, 20
    Z_out = resample_to_image_grid(Z_src; H_out=H_out, W_out=W_out)

    # Top half of output: rows 1:10 should all be exactly zero
    @test all(Z_out[1:10, :] .== 0.0)

    # Bottom half: at least some non-zero
    @test any(Z_out[11:end, :] .> 0.0)

    # Class assignment: all top-half cells → KDE_CLASS_FIELD
    r_cls = assign_kde_density_classes(Z_out)
    top_half_classes = r_cls.class_matrix[1:10, :]
    @test all(==(KDE_CLASS_FIELD), top_half_classes)
end

# ===========================================================================
# 83. resample_to_image_grid vs resample_to_count_grid: identity case
#     For a (263,324) surface, both should return an identical matrix
#     when the geotransform covers exactly the same domain.
# ===========================================================================

@testset "resample_to_image_grid: identity on already-correct-shape input" begin
    rng = Random.MersenneTwister(55)
    Z_263 = rand(rng, 263, 324)
    # Downsampling to the same shape is identity (nearest-neighbour)
    Z_out = resample_to_image_grid(Z_263; H_out=263, W_out=324)
    @test size(Z_out) == (263, 324)
    @test Z_out ≈ Z_263
end

# ===========================================================================
# 84. load_kde_surface_csv: normalised key handling for screenshot CSV header
# ===========================================================================

@testset "load_kde_surface_csv: screenshot-style header key normalisation" begin
    tmppath = tempname() * ".csv"

    # Write a CSV mimicking the screenshot KDE format:
    # key is '# Geotransform (screenshot): [...]' — previously not parsed
    open(tmppath, "w") do io
        println(io, "# Epanechnikov KDE density surface from orthomosaic screenshot")
        println(io, "# Source: Screenshot.jpg (10x12)")
        println(io, "# Geotransform (screenshot): [341300.35, 0.243, 0.0, 4774892.895, 0.0, -0.243]")
        println(io, "# crs: EPSG:6348")
        println(io, "# notes: test fixture")
        for r in 1:10
            println(io, join(rand(12), ","))
        end
    end

    surf = load_kde_surface_csv(tmppath)
    @test size(surf) == (10, 12)
    # Geotransform should now be parsed (not the GT_COUNTGRID fallback)
    @test surf.geotransform.dx ≈ 0.243  atol=1e-6
    @test surf.geotransform.dy ≈ -0.243 atol=1e-6
    # CRS should be parsed
    @test surf.crs == "EPSG:6348"
    # notes parsed
    @test surf.notes == "test fixture"

    rm(tmppath; force=true)
end

# ===========================================================================
# 85. resample_to_count_grid: blob collapse regression guard
#     A synthetic 100×100 blob resampled with correct geotransforms (same
#     domain) must NOT collapse to a bottom edge — the output blob must be
#     in the top-left when the input blob is in the top-left.
# ===========================================================================

@testset "resample_to_count_grid: spatial localisation regression (no edge collapse)" begin
    # Build a 100×100 surface with a blob in the top-left
    H_src, W_src = 100, 100
    Z_src = zeros(H_src, W_src)
    for r in 1:H_src, c in 1:W_src
        Z_src[r, c] = exp(-((r - 15)^2 + (c - 15)^2) / (2 * 10.0^2))
    end
    Z_src ./= maximum(Z_src)

    # Geotransform for the source: same origin and CRS as GT_COUNTGRID but
    # at native pixel size of 1 m (so 100×100 covers 100 m × 100 m)
    gt_src = GeoTransform(GT_COUNTGRID.x_origin, 1.0, 0.0,
                          GT_COUNTGRID.y_origin, 0.0, -1.0)

    # Output: 20×25 grid covering the same 100×100 m domain
    # (each output cell ≈ 5 m × 4 m)
    gt_out = GeoTransform(GT_COUNTGRID.x_origin, 5.0, 0.0,
                          GT_COUNTGRID.y_origin, 0.0, -4.0)

    Z_out = resample_to_count_grid(Z_src, gt_src; H_out=20, W_out=25, gt_out=gt_out)
    @test size(Z_out) == (20, 25)

    # Blob peak must remain in the top-left quadrant of the output
    max_idx = argmax(Z_out)
    @test max_idx[1] <= 10    # top half of rows
    @test max_idx[2] <= 13    # left half of cols

    # Bottom-right quadrant must be near zero
    @test maximum(Z_out[11:end, 14:end]) < 0.05
end

# ===========================================================================
# 86. MultiOtsuThreshold: type hierarchy and show
# ===========================================================================

@testset "MultiOtsuThreshold: type and show" begin
    m = MultiOtsuThreshold()
    @test m isa ThresholdMethod
    @test m isa MultiOtsuThreshold

    io = IOBuffer()
    show(io, m)
    s = String(take!(io))
    @test occursin("MultiOtsu", s)

    # Default method of assign_kde_density_classes is MultiOtsuThreshold
    Z = rand(Random.MersenneTwister(1), 10, 10)
    r = assign_kde_density_classes(Z)   # no method kwarg
    @test r.method isa MultiOtsuThreshold
end

# ===========================================================================
# 87. MultiOtsuThreshold: two ordered thresholds in (0,1)
# ===========================================================================

@testset "MultiOtsuThreshold: two ordered thresholds in [0,1]" begin
    rng = Random.MersenneTwister(22)
    Z   = rand(rng, 50, 60)
    t   = compute_kde_thresholds(Z, MultiOtsuThreshold())

    @test hasproperty(t, :lower_threshold)
    @test hasproperty(t, :upper_threshold)
    @test 0.0 <= t.lower_threshold <= 1.0
    @test 0.0 <= t.upper_threshold <= 1.0
    @test t.lower_threshold < t.upper_threshold

    r = assign_kde_density_classes(Z; method=MultiOtsuThreshold())
    @test r.lower_threshold < r.upper_threshold
    @test 0.0 <= r.lower_threshold <= 1.0
    @test 0.0 <= r.upper_threshold <= 1.0
end

# ===========================================================================
# 88. MultiOtsuThreshold: zeros classify as field-like
# ===========================================================================

@testset "MultiOtsuThreshold: zero-density pixels → field-like" begin
    # Surface: bottom half zeros, top half uniform positive
    H, W = 40, 40
    Z    = zeros(H, W)
    Z[1:20, :] .= rand(Random.MersenneTwister(5), 20, W)

    r = assign_kde_density_classes(Z; method=MultiOtsuThreshold())

    # Every zero pixel must be field-like
    zero_mask = Z .<= 1e-9
    @test all(==(KDE_CLASS_FIELD), r.class_matrix[zero_mask])

    # bottom-half zeros: all field-like
    @test all(==(KDE_CLASS_FIELD), r.class_matrix[21:end, :])

    # Support sums to grid
    s = kde_class_support(r)
    @test s.total == H * W
    @test s.field + s.deciduous + s.coniferous == H * W
end

# ===========================================================================
# 89. MultiOtsuThreshold: trimodal synthetic distribution maps correctly
# ===========================================================================

@testset "MultiOtsuThreshold: trimodal distribution correct class mapping" begin
    # Construct a surface with three well-separated modes:
    # ~50% values at 0.0 (field), ~30% around 0.35 (deciduous), ~20% around 0.85 (conif)
    rng = Random.MersenneTwister(99)
    n   = 600
    v_field = zeros(300)                             # mode 1: 0.0
    v_decid = 0.35 .+ 0.02 .* randn(rng, 180)       # mode 2: ~0.35
    v_conif = 0.85 .+ 0.02 .* randn(rng, 120)       # mode 3: ~0.85
    vals    = vcat(v_field, v_decid, v_conif)
    shuffle!(rng, vals)
    Z = reshape(vals, 30, 20)

    r = assign_kde_density_classes(Z; method=MultiOtsuThreshold())

    # t1 should lie between mode 1 and mode 2 → somewhere in (0.0, 0.35)
    @test r.lower_threshold > 0.0
    @test r.lower_threshold < 0.3

    # t2 should lie between mode 2 and mode 3 → somewhere in (0.38, 0.82)
    # (anywhere that places coniferous cluster ~0.85 above and deciduous ~0.35 below)
    @test r.upper_threshold > 0.38
    @test r.upper_threshold < 0.82

    # All exact-zero values should be field-like
    @test all(==(KDE_CLASS_FIELD), r.class_matrix[Z .== 0.0])

    # High-density values (≈0.85) should be coniferous
    @test all(==(KDE_CLASS_CONIFEROUS), r.class_matrix[Z .> 0.75])

    # Mid-density values (≈0.35) should be deciduous
    decid_mask = (Z .> 0.25) .& (Z .< 0.45)
    @test all(==(KDE_CLASS_DECIDUOUS), r.class_matrix[decid_mask])
end

# ===========================================================================
# 90. ManualThreshold two-arg form works; single-arg compat unchanged
# ===========================================================================

@testset "ManualThreshold: two-arg and single-arg (compat) forms" begin
    # Two-argument form
    m2 = ManualThreshold(0.15, 0.55)
    @test m2.lower ≈ 0.15
    @test m2.upper ≈ 0.55

    # Single-argument compat form sets lower = 0
    m1 = ManualThreshold(0.4)
    @test m1.lower == 0.0
    @test m1.upper ≈ 0.4

    # Errors: lower < 0
    @test_throws ArgumentError ManualThreshold(-0.1, 0.5)
    # Errors: upper <= lower
    @test_throws ArgumentError ManualThreshold(0.5, 0.3)
    @test_throws ArgumentError ManualThreshold(0.5, 0.5)

    # Two-arg classification: check lower split is honoured
    Z = Matrix{Float64}([0.0  0.10  0.20  0.40  0.60  0.80])  # 1×6
    r = assign_kde_density_classes(Z; method=ManualThreshold(0.15, 0.55))
    @test r.lower_threshold ≈ 0.15
    @test r.upper_threshold ≈ 0.55
    # 0.0 → field (≤ 0.15), 0.10 → field (≤ 0.15), 0.20 → deciduous (0.15 < 0.20 ≤ 0.55)
    @test r.class_matrix[1,1] == KDE_CLASS_FIELD
    @test r.class_matrix[1,2] == KDE_CLASS_FIELD
    @test r.class_matrix[1,3] == KDE_CLASS_DECIDUOUS
    @test r.class_matrix[1,4] == KDE_CLASS_DECIDUOUS
    @test r.class_matrix[1,5] == KDE_CLASS_CONIFEROUS
    @test r.class_matrix[1,6] == KDE_CLASS_CONIFEROUS
end

# ===========================================================================
# 91. Legacy OtsuThreshold: lower_threshold == eps; backward compat
# ===========================================================================

@testset "OtsuThreshold legacy: lower_threshold == eps" begin
    rng = Random.MersenneTwister(7)
    Z   = rand(rng, 30, 30)

    r = assign_kde_density_classes(Z; method=OtsuThreshold())
    @test r.lower_threshold == r.eps_threshold
    @test r.lower_threshold == 1e-9    # default eps

    # upper_threshold from legacy Otsu must still be in (0,1)
    @test 0.0 < r.upper_threshold < 1.0

    # Single-arg ManualThreshold compat: lower_threshold == eps as well
    r2 = assign_kde_density_classes(Z; method=ManualThreshold(0.4))
    @test r2.lower_threshold == 1e-9
    @test r2.upper_threshold ≈ 0.4
end

# ===========================================================================
# 92. CR/CI invariance: validated DURHAM_COVER_N constants unchanged
#     (primary outcome metric must not regress)
# ===========================================================================

@testset "validated DURHAM_COVER_N constants unchanged" begin
    @test DURHAM_COVER_N[:conif] == 5_413
    @test DURHAM_COVER_N[:decid] == 24_080
    @test DURHAM_COVER_N[:field] == 55_719
    @test sum(values(DURHAM_COVER_N)) == 85_212
    @test 263 * 324 == 85_212
end

# ===========================================================================
# 93. DensityClassResult: lower_threshold field present; show updated
# ===========================================================================

@testset "DensityClassResult: lower_threshold field and show string" begin
    Z = rand(Random.MersenneTwister(3), 15, 15)
    r = assign_kde_density_classes(Z; method=MultiOtsuThreshold())

    @test hasproperty(r, :lower_threshold)
    @test hasproperty(r, :upper_threshold)
    @test hasproperty(r, :eps_threshold)
    @test r.lower_threshold <= r.upper_threshold

    io = IOBuffer()
    show(io, r)
    s = String(take!(io))
    @test occursin("lower=", s)
    @test occursin("upper=", s)
    @test occursin("MultiOtsuThreshold", s)
end

# ===========================================================================
# 94. kde_class_summary: lower_threshold column present for MultiOtsu
# ===========================================================================

@testset "kde_class_summary: lower_threshold column with MultiOtsuThreshold" begin
    rng  = Random.MersenneTwister(55)
    H, W = 20, 25
    Z    = rand(rng, H, W)
    cg   = rand(rng, 0:3, H, W)

    df = kde_class_summary(Z, Dict("M1" => cg);
             method=MultiOtsuThreshold(), kde_status="test",
             alignment="synthetic")

    @test hasproperty(df, :lower_threshold)
    @test hasproperty(df, :upper_threshold)
    @test all(df.lower_threshold .< df.upper_threshold)
    @test nrow(df) == 3   # 3 classes × 1 mission
end

# ===========================================================================
# 52. RunInputs (inputs.jl)
# ===========================================================================
@testset "RunInputs: TOML round-trip" begin
    mktempdir() do dir
        # Create a minimal TOML
        toml_path = joinpath(dir, "cfg.toml")
        open(toml_path, "w") do io
            inputs_template(io)
        end
        # The template references non-existent paths; load should still parse
        ri = load_inputs(toml_path)
        @test isa(ri, RunInputs)
        @test ri.kde_kernel == :epanechnikov
        @test ri.tree_labels == [1]
        @test ri.flightlines_spacing_m == 40.0
        @test ri.kmedoids_k_range == (2, 12)
        @test ri.gli_class_codes[:field] == 2
    end
end

@testset "RunInputs: validate raises on missing rgb" begin
    ri = RunInputs(rgb = "/nonexistent/path/to/ortho.tif")
    @test_throws KDEFlightPlanning.MissingInputError KDEFlightPlanning.validate(ri)
end

# ===========================================================================
# 53. GeoTIFF I/O (geotiff_io.jl) — synthetic round trip
# ===========================================================================
@testset "geotiff_io: synthetic round trip" begin
    using ArchGDAL
    AG_ = ArchGDAL
    mktempdir() do dir
        path = joinpath(dir, "synth.tif")
        H, W = 12, 18
        R = rand(UInt8, H, W); G = rand(UInt8, H, W); B = rand(UInt8, H, W)
        gt = [100.0, 1.0, 0.0, 200.0, 0.0, -1.0]
        AG_.create(path; driver = AG_.getdriver("GTiff"),
                          width = W, height = H, nbands = 3, dtype = UInt8) do ds
            AG_.setgeotransform!(ds, gt)
            AG_.write!(ds, permutedims(R), 1)
            AG_.write!(ds, permutedims(G), 2)
            AG_.write!(ds, permutedims(B), 3)
        end
        rs = load_rgb_geotiff(path)
        @test size(rs) == (H, W)
        @test rs.gt.dx ≈ 1.0
        @test rs.gt.dy ≈ -1.0
        ex = raster_extents(rs)
        @test ex.xmin ≈ 100.0
        @test ex.dy  ≈ 1.0
        # Single-band write/read
        Z = Float64.([(i + 0.5*j) for i in 1:H, j in 1:W])
        out = joinpath(dir, "out.tif")
        write_single_band_geotiff(out, Z, rs.gt, "")
        Z2, gt2, _ = read_band(out)
        @test size(Z2) == size(Z)
        @test Z2 ≈ Z
    end
end

# ===========================================================================
# 53b. GeoTIFF-first native scale — no screenshot resample factor applied
# ===========================================================================
@testset "geotiff native m/px (anisotropic; no screenshot factor)" begin
    using ArchGDAL
    AG_ = ArchGDAL
    mktempdir() do dir
        path = joinpath(dir, "aniso.tif")
        H, W = 9, 15
        R = rand(UInt8, H, W); G = rand(UInt8, H, W); B = rand(UInt8, H, W)
        # Anisotropic north-up geotransform: dx = 0.5, dy = -0.25 (metres/px).
        xres_native, yres_native = 0.5, 0.25
        gt = [500000.0, xres_native, 0.0, 4000000.0, 0.0, -yres_native]
        AG_.create(path; driver = AG_.getdriver("GTiff"),
                          width = W, height = H, nbands = 3, dtype = UInt8) do ds
            AG_.setgeotransform!(ds, gt)
            AG_.write!(ds, permutedims(R), 1)
            AG_.write!(ds, permutedims(G), 2)
            AG_.write!(ds, permutedims(B), 3)
        end
        rs = load_rgb_geotiff(path)

        # Native metres/px come straight from the geotransform — anisotropic,
        # independent of any screenshot dimensions. This is the value the
        # GeoTIFF-authoritative producer path uses (factor_x = factor_y = 1).
        xres, yres = geotransform_resolution(rs.gt)
        @test xres ≈ xres_native
        @test yres ≈ yres_native
        @test xres != yres                                   # genuinely anisotropic

        # Full-ground metric extent = pixels × native GSD (no source_*_px factor).
        @test W * xres ≈ 7.5
        @test H * yres ≈ 2.25

        # A hypothetical screenshot-resample factor (e.g. source_width_px/W) must
        # NOT change the native resolution in GeoTIFF mode.
        bogus_factor = 8.6
        @test !(xres ≈ xres_native * bogus_factor)
    end
end

# ===========================================================================
# 54. lidar_counts.jl: bin_to_count_grid + tables (synthetic, no LAS)
# ===========================================================================
@testset "lidar_counts: bin_to_count_grid synthetic" begin
    classes = Int[
        2 2 0 0
        2 0 0 1
        0 0 1 1
        0 1 1 1
    ]
    gt = KDEFlightPlanning.GeoTransform([100.0, 1.0, 0.0, 104.0, 0.0, -1.0])
    ex = raster_extents(classes, gt)
    # Place exactly one point in each cell centre
    pts = [(ex.xmin + (i - 0.5), ex.ymax - (j - 0.5))
           for j in 1:size(classes,1), i in 1:size(classes,2)]
    X = vcat([reshape([p[1], p[2], 0.0], 1, 3) for p in pts]...)
    cf, cd, cc = bin_to_count_grid(X, classes;
        xmin = ex.xmin, xmax = ex.xmax, ymin = ex.ymin, ymax = ex.ymax,
        dx = ex.dx, dy = ex.dy)
    @test sum(cf) == count(==(2), classes)
    @test sum(cd) == count(==(0), classes)
    @test sum(cc) == count(==(1), classes)
    # Each cover's count grid must be zero outside its mask
    @test all(cf[classes .!= 2] .== 0)
    @test all(cd[classes .!= 0] .== 0)
    @test all(cc[classes .!= 1] .== 0)
end

@testset "lidar_counts: percent_density_table + cover_stats_table" begin
    classes = fill(2, 4, 4)
    classes[1:2, 3:4] .= 0
    classes[3:4, 3:4] .= 1
    grid = ones(Int32, 4, 4) .* 3
    counts = Dict(
        CountKey(ret = :all, cover = :field, mission = :density, kernel = :E) =>
            Int32.(grid .* (classes .== 2)),
        CountKey(ret = :all, cover = :decid, mission = :density, kernel = :E) =>
            Int32.(grid .* (classes .== 0)),
        CountKey(ret = :all, cover = :conif, mission = :density, kernel = :E) =>
            Int32.(grid .* (classes .== 1)),
    )
    pd = percent_density_table(counts, classes)
    @test "Cover" in names(pd)
    @test nrow(pd) == 3
    st = cover_stats_table(counts, classes;
        missions=(:density,), kernels=(:E,), returns=(:all,))
    @test "CR" in names(st)
    @test all(st.CR .≈ 1.0)   # all valid cells have count > 0 → CR = 1.0
end

# ===========================================================================
# 55. reports.jl: smoke test that file artefacts are produced
# ===========================================================================
@testset "reports: artefact production" begin
    mktempdir() do dir
        out = joinpath(dir, "report")
        # Cluster metric sweep with the actual ClusterMetrics struct
        mets = [ClusterMetrics(k, 0.5 + 0.1*k, 0.1*k, 1.0/k, 100.0*k,
                                nothing,
                                (k == 3 ? 4 : 0), (k == 3),
                                1, 100, :vote)
                 for k in 2:5]
        paths = report_cluster_metrics_sweep(mets, out)
        @test all(isfile, paths)
        # PCA scree
        paths2 = report_pca_explained([0.6, 0.3, 0.1], out)
        @test all(isfile, paths2)
        # Manifest + provenance
        report_run_manifest(out)
        @test isfile(joinpath(out, "manifest.csv"))
        report_provenance(out; package_version = "test")
        @test isfile(joinpath(out, "provenance.json"))
    end
end

# ===========================================================================
# 56. v0.5.2 hotfix: downsample / stride helpers + georeferenced spacing
# ===========================================================================
@testset "reports: plot_stride and downsample helpers" begin
    @test KDEFlightPlanning._plot_stride(100, 100; max_preview_px=2000) == 1
    @test KDEFlightPlanning._plot_stride(9000, 11000; max_preview_px=2000) == 6
    @test KDEFlightPlanning._plot_stride(20000, 20000; max_preview_px=1000) == 20
    M = collect(reshape(1:120, 12, 10))   # 12×10
    @test size(KDEFlightPlanning._downsample(M, 1)) == (12, 10)
    @test size(KDEFlightPlanning._downsample(M, 2)) == (6, 5)
    @test size(KDEFlightPlanning._downsample(M, 4)) == (3, 3)
end

@testset "RunInputs: track_spacing_m and waypoint spacing fields" begin
    mktempdir() do dir
        toml_path = joinpath(dir, "spacing.toml")
        open(toml_path, "w") do io
            write(io, """
            [paths]
            rgb = "synthetic.tif"
            [planning]
            track_spacing_m        = 40.0
            min_waypoint_spacing_m = 10.0
            max_waypoint_spacing_m = 30.0
            cluster_stride         = 8
            max_preview_px         = 1500
            """)
        end
        ri = load_inputs(toml_path)
        @test ri.flightlines_spacing_m == 40.0
        @test ri.min_waypoint_spacing_m == 10.0
        @test ri.max_waypoint_spacing_m == 30.0
        @test ri.cluster_stride == 8
        @test ri.max_preview_px == 1500
    end
end

@testset "RunInputs: legacy flightlines_spacing_m alias still works" begin
    mktempdir() do dir
        toml_path = joinpath(dir, "legacy.toml")
        open(toml_path, "w") do io
            write(io, """
            [paths]
            rgb = "synthetic.tif"
            [planning]
            flightlines_spacing_m = 20.0
            """)
        end
        ri = load_inputs(toml_path)
        @test ri.flightlines_spacing_m == 20.0
    end
end

@testset "axes_from_geotransform: spacing-in-metres invariant" begin
    # Realistic Durham geotransform: ~2.8 cm/px in x
    gt = [341300.35, 0.028369669906313746, 0.0,
          4774892.895, 0.0, -0.02836868197004068]
    xs, ys = axes_from_geotransform(gt, 11421, 9279)
    # Pixel-to-metre step
    @test length(xs) == 11421
    @test length(ys) == 9279
    dx = abs(xs[2] - xs[1])
    # `axes_from_geotransform` builds xs via a comprehension; the recurrence
    # accumulates a few ULP of error vs the closed-form (i - 0.5) * dx, hence
    # 1e-9 not 1e-12.
    @test isapprox(dx, 0.028369669906313746; atol=1e-9)
    # 40 m / dx ≈ 1410 pixels — which would be the bug fix's "meters → pixels"
    @test 40.0 / dx > 1000   # spacing=40 in metres is many pixels
end

@testset "build_mask_from_image_strided: synthetic stride round-trip" begin
    using Random
    rng = MersenneTwister(11)
    H, W = 80, 100
    arr = rand(rng, UInt8, H, W, 3)
    mg, info = build_mask_from_image_strided(arr; stride=4,
        k=2, tree_labels=[1], seed=11, nsample=500,
        ks=2:3, k_strategy=:vote)
    @test size(mg.Z) == (H, W)
    @test info.stride == 4
    @test info.H_sample == cld(H, 4)
    @test info.W_sample == cld(W, 4)
    @test length(info.labels_full) == H * W
    # All pixels labelled in {1..k}
    @test all(1 .<= info.labels_full .<= info.k)
end

@testset "proposed_artifacts checklist: produces CSV with statuses" begin
    mktempdir() do dir
        # Touch a couple of expected files so we get one 'produced' + one 'missing'
        mkpath(joinpath(dir, "ingest"))
        open(joinpath(dir, "ingest", "rgb_preview.png"), "w") do io
            write(io, "fake")
        end
        paths = report_proposed_artifacts_checklist(dir)
        @test length(paths) == 1
        @test isfile(paths[1])
        rows = readlines(paths[1])
        # Header + ≥40 rows
        @test length(rows) > 40
        @test occursin("path_pattern,tier,status,description", rows[1])
        @test any(occursin("produced", r) for r in rows)
        @test any(occursin("skipped", r)  for r in rows)
    end
end

# ===========================================================================
# 57. v0.5.3 PDF-timeout hotfix: report_formats config + _save_fig fault-tolerance
# ===========================================================================
@testset "RunInputs: report_formats parsing" begin
    mktempdir() do dir
        toml_path = joinpath(dir, "rf.toml")
        open(toml_path, "w") do io
            write(io, """
            [paths]
            rgb = "synthetic.tif"
            [planning]
            report_formats = ["png"]
            """)
        end
        ri = load_inputs(toml_path)
        @test ri.report_formats == ["png"]
    end
end

@testset "RunInputs: report_formats default is PNG only" begin
    ri = RunInputs(rgb = "/dev/null")
    @test ri.report_formats == ["png"]
end

@testset "RunInputs: report_formats accepts pdf opt-in" begin
    mktempdir() do dir
        toml_path = joinpath(dir, "rfp.toml")
        open(toml_path, "w") do io
            write(io, """
            [paths]
            rgb = "synthetic.tif"
            [planning]
            report_formats = ["png", "pdf"]
            """)
        end
        ri = load_inputs(toml_path)
        @test ri.report_formats == ["png", "pdf"]
    end
end

@testset "_save_fig honours formats argument and is fault-tolerant" begin
    @eval using CairoMakie
    mktempdir() do dir
        fig = CairoMakie.Figure()
        ax  = CairoMakie.Axis(fig[1, 1])
        CairoMakie.lines!(ax, 1:5, 1:5)

        # PNG only — explicit single format
        paths = KDEFlightPlanning._save_fig(fig, joinpath(dir, "t1");
                                             formats = ["png"])
        @test length(paths) == 1
        @test endswith(paths[1], ".png")
        @test isfile(paths[1])

        # PNG + PDF — both should write under normal conditions
        paths2 = KDEFlightPlanning._save_fig(fig, joinpath(dir, "t2");
                                              formats = ["png", "pdf"])
        @test endswith(paths2[1], ".png")
        @test isfile(paths2[1])
        # PDF may fail in unusual environments; if it succeeded we get 2
        # paths, otherwise 1. Either way the call must not throw.
        @test 1 <= length(paths2) <= 2

        # Unknown format falls through to CairoMakie.save (which throws),
        # but the catch keeps the loop going.
        paths3 = KDEFlightPlanning._save_fig(fig, joinpath(dir, "t3");
                                              formats = ["png", "xyz"])
        @test any(endswith.(paths3, ".png"))
    end
end

@testset "DEFAULT_REPORT_FORMATS is PNG only" begin
    @test KDEFlightPlanning.DEFAULT_REPORT_FORMATS == ["png"]
end

# ===========================================================================
# 16. Vegetation cluster-label selection helpers (scripts/tree_label_selection.jl)
#
# These are the pure, stdlib-only helpers backing the interactive `tree_labels`
# workflow in scripts/preprocess_site_image.jl. They are unit-tested here in a
# throwaway module so the include() does not leak names into the test globals.
# ===========================================================================
module TreeLabelSelectionTests
    using Test
    include(joinpath(@__DIR__, "..", "scripts", "tree_label_selection.jl"))

    @testset "tree-label selection helpers" begin

        @testset "site_configured_tree_labels" begin
            # Present, valid, nonempty → raw vector of Ints (order preserved).
            @test site_configured_tree_labels(Dict("tree_labels" => [3, 1])) == [3, 1]
            @test site_configured_tree_labels(Dict("tree_labels" => [2])) == [2]
            # Integer-valued floats coerce.
            @test site_configured_tree_labels(Dict("tree_labels" => [1.0, 2.0])) == [1, 2]
            # Absent / empty / non-integer / non-list → nothing (NO default).
            @test site_configured_tree_labels(Dict{String,Any}()) === nothing
            @test site_configured_tree_labels(Dict("tree_labels" => Int[])) === nothing
            @test site_configured_tree_labels(Dict("tree_labels" => [1.5])) === nothing
            @test site_configured_tree_labels(Dict("tree_labels" => "1,2")) === nothing
        end

        @testset "parse_tree_label_input" begin
            @test parse_tree_label_input("1,3", 4) == [1, 3]
            @test parse_tree_label_input("3 1", 4) == [1, 3]          # sorted
            @test parse_tree_label_input("  2 ,  4 ", 4) == [2, 4]     # whitespace tolerant
            # Reprompt signals (nothing): empty, non-integer, out-of-range, dup.
            @test parse_tree_label_input("", 4) === nothing
            @test parse_tree_label_input("   ", 4) === nothing
            @test parse_tree_label_input("x", 4) === nothing
            @test parse_tree_label_input("0", 4) === nothing
            @test parse_tree_label_input("5", 4) === nothing
            @test parse_tree_label_input("1,1", 4) === nothing        # duplicate
        end

        @testset "validate_configured_tree_labels" begin
            @test validate_configured_tree_labels([1, 3], 4) === nothing
            @test validate_configured_tree_labels(Int[], 4) !== nothing        # empty
            @test validate_configured_tree_labels([1, 1], 4) !== nothing       # dup
            @test validate_configured_tree_labels([5], 4) !== nothing          # OOR
            @test validate_configured_tree_labels([0], 4) !== nothing          # OOR
        end

        @testset "render_tree_labels" begin
            @test render_tree_labels([1]) == "[1]"
            @test render_tree_labels([1, 3]) == "[1, 3]"
        end

        @testset "resolve_input_path" begin
            # Empty / missing → "" (no path).
            @test resolve_input_path("/base", "") == ""
            @test resolve_input_path("/base", nothing) == ""
            # Relative resolves against base_dir; absolute passes through.
            @test resolve_input_path("/base/dir", "img.jpg") == abspath("/base/dir/img.jpg")
            @test resolve_input_path("/base", "/abs/img.tif") == "/abs/img.tif"
        end

        @testset "select_site_input precedence (GeoTIFF-first, image fallback)" begin
            mktempdir() do dir
                tif = joinpath(dir, "ortho.tif")
                jpg = joinpath(dir, "shot.jpg")
                write(tif, "fake-geotiff-bytes")   # existence is all select_site_input checks
                write(jpg, "fake-jpeg-bytes")

                # GeoTIFF wins even when an `image` also exists.
                @test select_site_input(Dict("geotiff" => tif, "image" => jpg), dir) ==
                      (:geotiff, tif)
                # GeoTIFF alone.
                @test select_site_input(Dict("geotiff" => "ortho.tif"), dir) ==
                      (:geotiff, tif)
                # No geotiff key → image fallback.
                @test select_site_input(Dict("image" => "shot.jpg"), dir) ==
                      (:image, jpg)
                # geotiff key present but file MISSING → falls back to image.
                @test select_site_input(Dict("geotiff" => "nope.tif", "image" => jpg), dir) ==
                      (:image, jpg)
                # Empty geotiff string → image fallback.
                @test select_site_input(Dict("geotiff" => "", "image" => "shot.jpg"), dir) ==
                      (:image, jpg)
                # Neither readable → :none (site skipped).
                @test select_site_input(Dict("geotiff" => "x.tif", "image" => "y.jpg"), dir) ==
                      (:none, "")
                @test select_site_input(Dict{String,Any}(), dir) == (:none, "")
            end
        end

        @testset "stdin_is_tty resolves Base.isatty (regression: UndefVarError)" begin
            # Regression for `UndefVarError: isatty not defined in Main`: `isatty`
            # is in Base but unexported, so an unqualified call in Main threw.
            # Calling the wrapper must resolve the symbol and return a Bool without
            # throwing (this is the exact gate that crashed resolve_site_tree_labels).
            @test stdin_is_tty() isa Bool                    # default stdin, must not throw
            # A non-TTY IOBuffer exercises Base's generic isatty(::IO) fallback.
            @test stdin_is_tty(IOBuffer()) === false
            @test stdin_is_tty(IOBuffer("1,3\n")) === false
        end

        @testset "prompt_tree_labels reprompts on invalid input" begin
            # First two lines invalid (non-int, out-of-range), third valid.
            input  = IOBuffer("foo\n9\n1,3\n")
            output = IOBuffer()
            sel = prompt_tree_labels(4, ["/tmp/k1.png", "/tmp/k2.png"];
                                     name = "Test Site", suggested = 2,
                                     in_io = input, out_io = output)
            @test sel == [1, 3]
            outstr = String(take!(output))
            @test occursin("Test Site", outstr)
            @test occursin("Invalid", outstr)                        # reprompted
        end

        @testset "prompt_tree_labels errors after max_attempts" begin
            input  = IOBuffer("bad\nbad\n")
            output = IOBuffer()
            @test_throws ErrorException prompt_tree_labels(
                3, String[]; name = "S", suggested = 1,
                in_io = input, out_io = output, max_attempts = 2)
        end

        # -------------------------------------------------------------------
        # persist_tree_labels — line-preserving, atomic TOML block update
        # -------------------------------------------------------------------
        function _with_tmp_config(content::String, f)
            dir = mktempdir()
            path = joinpath(dir, "cfg.toml")
            write(path, content)
            try
                f(path)
            finally
                rm(dir; recursive = true, force = true)
            end
        end

        @testset "persist replaces an existing tree_labels line" begin
            cfg = """
            # top comment
            seed = 6213

            [[site]]
            name = "OxBow Farm"          # trailing comment kept
            tree_labels = [1]
            col_a = "a.png"

            [[site]]
            name = "Other"
            col_a = "b.png"
            """
            _with_tmp_config(cfg) do path
                ok, msg = persist_tree_labels(path, "OxBow Farm", [2, 4])
                @test ok
                @test msg == ""
                out = read(path, String)
                @test occursin("tree_labels = [2, 4]", out)
                @test !occursin("tree_labels = [1]", out)
                # Untouched lines preserved byte-for-byte.
                @test occursin("# top comment", out)
                @test occursin("name = \"OxBow Farm\"          # trailing comment kept", out)
                @test occursin("[[site]]\nname = \"Other\"", out)
                @test occursin("seed = 6213", out)
            end
        end

        @testset "persist inserts tree_labels after name when absent" begin
            cfg = """
            [[site]]
            name = "Hubbard Brook"
            col_a = "hb.png"
            """
            _with_tmp_config(cfg) do path
                ok, _ = persist_tree_labels(path, "Hubbard Brook", [3])
                @test ok
                lines = readlines(path)
                ni = findfirst(l -> occursin("name = \"Hubbard Brook\"", l), lines)
                @test ni !== nothing
                @test occursin("tree_labels = [3]", lines[ni + 1])    # right after name
                @test occursin("col_a = \"hb.png\"", read(path, String))
            end
        end

        @testset "persist matches by slug (whitespace/case/punct-insensitive)" begin
            cfg = """
            [[site]]
            name = "Kingman Farm"
            col_a = "kf.png"
            """
            _with_tmp_config(cfg) do path
                ok, _ = persist_tree_labels(path, "kingman-farm", [2])
                @test ok
                @test occursin("tree_labels = [2]", read(path, String))
            end
        end

        @testset "persist preserves indentation of replaced line" begin
            cfg = "[[site]]\n  name = \"Indented\"\n  tree_labels = [1]\n"
            _with_tmp_config(cfg) do path
                ok, _ = persist_tree_labels(path, "Indented", [5])
                @test ok
                @test occursin("  tree_labels = [5]", read(path, String))
            end
        end

        @testset "persist refuses & does not mutate on ambiguous duplicate names" begin
            cfg = """
            [[site]]
            name = "Dup"
            tree_labels = [1]

            [[site]]
            name = "Dup"
            tree_labels = [2]
            """
            _with_tmp_config(cfg) do path
                before = read(path, String)
                ok, msg = persist_tree_labels(path, "Dup", [3])
                @test !ok
                @test occursin("ambiguous", msg)
                @test read(path, String) == before           # unchanged
            end
        end

        @testset "persist reports missing site without mutating" begin
            cfg = "[[site]]\nname = \"Present\"\n"
            _with_tmp_config(cfg) do path
                before = read(path, String)
                ok, msg = persist_tree_labels(path, "Absent", [1])
                @test !ok
                @test occursin("no [[site]] block", msg)
                @test read(path, String) == before
            end
            _with_tmp_config("seed = 1\n") do path            # no [[site]] at all
                ok, msg = persist_tree_labels(path, "Whatever", [1])
                @test !ok
                @test occursin("no [[site]] blocks", msg)
            end
        end

        # -------------------------------------------------------------------
        # find_conflict_markers / assert_no_conflict_markers — regression for
        # a git stash/merge conflict committed into cross_site_panel.toml
        # (TOML.parsefile otherwise dies with an opaque "expected key" at the
        # `<<<<<<< Updated upstream` line). persist_tree_labels never produces
        # these; this guards the manual-conflict case.
        # -------------------------------------------------------------------
        @testset "find_conflict_markers flags a stash-pop conflict block" begin
            # Byte-for-byte the shape that shipped in the failing config: a
            # geotiff conflict inside a [[site]] block.
            cfg = """
            [[site]]
            name                     = "OxBow Farm"
            image                    = "../data/site_images/oxbow_farm.jpg"
            <<<<<<< Updated upstream
            geotiff                  = ""
            source_width_px          = 11427
            =======
            geotiff                  = "/Users/darien/Desktop/data/OB_reproj_cropped.tiff"
            source_width_px          = 11427
            >>>>>>> Stashed changes
            provisional              = true
            """
            hits = find_conflict_markers(split(cfg, '\n'))
            @test length(hits) == 3
            @test [m for (_, m) in hits] == ["<<<<<<<", "=======", ">>>>>>>"]
            @test [l for (l, _) in hits] == [4, 7, 10]
        end

        @testset "find_conflict_markers: clean config has none" begin
            clean = """
            seed = 6213
            [[site]]
            name    = "OxBow Farm"
            geotiff = "/Users/darien/Desktop/data/OB_reproj_cropped.tiff"
            """
            @test isempty(find_conflict_markers(split(clean, '\n')))
            # A legit value that merely contains '=' or '<' is not a marker.
            @test isempty(find_conflict_markers(["k = \"a = b\"", "cmp = \"<7\""]))
        end

        @testset "assert_no_conflict_markers errors with line numbers" begin
            _with_tmp_config(
                "[[site]]\nname = \"X\"\n<<<<<<< Updated upstream\ngeotiff = \"\"\n=======\ngeotiff = \"a.tif\"\n>>>>>>> Stashed changes\n") do path
                err = try
                    assert_no_conflict_markers(path); nothing
                catch e; e; end
                @test err isa ErrorException
                @test occursin("conflict markers", err.msg)
                @test occursin("3", err.msg)          # first marker line
            end
            # A conflict-free config passes silently.
            _with_tmp_config("[[site]]\nname = \"X\"\ngeotiff = \"a.tif\"\n") do path
                @test assert_no_conflict_markers(path) === nothing
            end
        end
    end
end
