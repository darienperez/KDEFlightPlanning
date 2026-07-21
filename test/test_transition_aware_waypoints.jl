"""
    test/test_transition_aware_waypoints.jl

Focused unit/integration tests for the transition-aware (two-pass, event-driven)
waypoint placement added to `CurvatureGuidedSpeed`. Uses only synthetic
georeferenced `RasterGrid` data (metre axes, matching the UTM convention of the
real pipeline). Run standalone:

    julia --project=. test/test_transition_aware_waypoints.jl

or `include` it from `runtests.jl`.

Scenarios exercised
-------------------
1. Perfectly uniform density              → whole path at spacing_max.
2. Broad high-density plateau w/ edges     → quiet interior at max, edges refined.
3. Narrow transition (< spacing_max)       → event is NOT leapt over.
4. Smooth low-amplitude noise              → no spurious events (stays near max).
5. Multiple flight lines + turnarounds     → line_id correct, no dup corners.

Asserted properties
-------------------
- quiet regions use spacing near/exactly spacing_max (except boundaries),
- event interiors use spacing_min / fine spacing,
- ≥1 forced point immediately before and after each detected event,
- configured 10–30 m bounds are honoured (upper bound never exceeded),
- no consecutive duplicate coordinates,
- line_id nonzero and correct per survey line.
"""

using Test
using KDEFlightPlanning
using Statistics: mean, median

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

"""
Along-line consecutive spacings for one survey line.

A `line_id` group contains the horizontal survey sweep PLUS the (vertical)
turnaround transit that follows it. For spacing analysis we keep only the
horizontal sweep — the points sharing the group's modal y — so the vertical
transit hop does not masquerade as a survey-line gap.
"""
function _line_gaps(wps, lid)
    grp = [w for w in wps if w.line_id == lid]
    isempty(grp) && return Float64[], grp
    ykeys = round.([w.y for w in grp]; digits = 3)
    modal_y = argmax(v -> count(==(v), ykeys), unique(ykeys))
    line = sort([w for w in grp if round(w.y; digits = 3) == modal_y]; by = w -> w.x)
    length(line) < 2 && return Float64[], line
    gaps = [hypot(line[k+1].x - line[k].x, line[k+1].y - line[k].y)
            for k in 1:length(line)-1]
    return gaps, line
end

"x-coordinate of the midpoint of each along-line gap (for locating events)."
_gap_midx(line) = [(line[k].x + line[k+1].x) / 2 for k in 1:length(line)-1]

"Build a 1-line lawnmower path spanning [0,X] at y=y0."
function _single_line_path(X; y0 = 0.0)
    spec = LawnmowerSpec(xmin = 0.0, xmax = X, ymin = y0, ymax = y0,
                         spacing = 40.0, yaw_deg = 0.0, primary = :x, start = :low)
    lawnmower_from_extents(spec)
end

const SMIN = 10.0
const SMAX = 30.0

# A tanh edge centred at x0 with half-width w (in metres). Rows are constant in y.
_edge(x, x0, w) = 0.5 * (1 + tanh((x - x0) / w))

function _grid_from_profile(profile::Function; X = 300.0, Y = 120.0,
                            nx = 601, ny = 40)
    xs = collect(range(0.0, X; length = nx))
    ys = collect(range(0.0, Y; length = ny))
    Z  = zeros(nx > 0 ? length(ys) : 0, length(xs))
    for (i, x) in enumerate(xs)
        v = clamp(profile(x), 0.0, 1.0)
        @inbounds for j in 1:length(ys)
            Z[j, i] = v
        end
    end
    return RasterGrid(Z, xs, ys)
end

@testset "transition-aware waypoint placement" begin

    # -----------------------------------------------------------------------
    # 1. Perfectly uniform density → spacing_max everywhere (bounds honoured)
    # -----------------------------------------------------------------------
    @testset "uniform density → max spacing" begin
        grid = uniform_grid(0.5; nx = 60, ny = 20,
                            xmin = 0.0, xmax = 300.0, ymin = 0.0, ymax = 100.0)
        path = _single_line_path(300.0)
        strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
        wps = generate_waypoints(path, grid, strat;
                                 spacing_min = SMIN, spacing_max = SMAX,
                                 density_threshold = 0.5)
        gaps, line = _line_gaps(wps, 1)
        @test !isempty(gaps)
        # interior gaps are exactly spacing_max (only trailing boundary may differ)
        @test maximum(gaps) <= SMAX + 1e-6          # upper bound never exceeded
        @test count(g -> isapprox(g, SMAX; atol = 1e-3), gaps) >= length(gaps) - 1
        @test median(gaps) >= SMAX - 1e-3           # NOT median-normalised undershoot
    end

    # -----------------------------------------------------------------------
    # 2. Broad high-density plateau with entering/exiting edges
    # -----------------------------------------------------------------------
    @testset "broad plateau: quiet interior max, edges refined" begin
        x_lo, x_hi, hw = 110.0, 190.0, 4.0
        prof(x) = _edge(x, x_lo, hw) - _edge(x, x_hi, hw)   # ~1 inside [110,190]
        grid = _grid_from_profile(prof)
        path = _single_line_path(300.0)
        strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
        wps = generate_waypoints(path, grid, strat;
                                 spacing_min = SMIN, spacing_max = SMAX,
                                 density_threshold = 0.5, event_margin = SMIN)
        gaps, line = _line_gaps(wps, 1)
        midx = _gap_midx(line)

        @test maximum(gaps) <= SMAX + 1e-6                       # bound honoured
        # plateau deep interior (x∈[140,160]) is quiet → near max spacing
        interior = [gaps[k] for k in eachindex(gaps) if 140 <= midx[k] <= 160]
        @test !isempty(interior)
        @test mean(interior) >= SMAX - 3.0
        # edges get fine spacing
        near_edges = [gaps[k] for k in eachindex(gaps)
                      if abs(midx[k]-x_lo) <= 20 || abs(midx[k]-x_hi) <= 20]
        @test !isempty(near_edges)
        @test minimum(near_edges) <= SMIN + 1e-6
        # forced anticipatory + trailing point around each edge event
        xs_line = [w.x for w in line]
        has_pt(a, b) = any(x -> a - 1e-6 <= x <= b + 1e-6, xs_line)
        for x0 in (x_lo, x_hi)
            @test has_pt(x0 - SMIN - SMIN, x0)   # a point in the anticipatory band
            @test has_pt(x0, x0 + SMIN + SMIN)   # a point in the trailing band
        end
    end

    # -----------------------------------------------------------------------
    # 3. Narrow transition narrower than spacing_max → not leapt over
    # -----------------------------------------------------------------------
    @testset "narrow transition is not overshot" begin
        x0, hw = 150.0, 3.0                     # spike ~ 18 m wide < spacing_max=30
        prof(x) = exp(-0.5 * ((x - x0) / hw)^2)  # gaussian bump
        grid = _grid_from_profile(prof)
        path = _single_line_path(300.0)
        strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
        wps = generate_waypoints(path, grid, strat;
                                 spacing_min = SMIN, spacing_max = SMAX,
                                 density_threshold = 0.5, event_margin = SMIN)
        gaps, line = _line_gaps(wps, 1)
        midx = _gap_midx(line)
        # ≥1 waypoint lands inside the transition core [x0-2hw, x0+2hw]:
        # the legacy 30 m marcher could leap the whole 18 m band, placing none.
        core_pts = count(w -> x0 - 2hw <= w.x <= x0 + 2hw, line)
        @test core_pts >= 1
        # fine spacing present near the bump
        near = [gaps[k] for k in eachindex(gaps) if abs(midx[k] - x0) <= 25]
        @test !isempty(near) && minimum(near) <= SMIN + 1e-6
        @test maximum(gaps) <= SMAX + 1e-6
    end

    # -----------------------------------------------------------------------
    # 4. Smooth low-amplitude noise → no spurious events
    # -----------------------------------------------------------------------
    @testset "low-amplitude ripple stays near max" begin
        prof(x) = 0.5 + 0.02 * sin(2π * x / 60)   # amplitude 0.02, gentle
        grid = _grid_from_profile(prof)
        path = _single_line_path(300.0)
        strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
        # auto triggers should treat this as quiet (floors dominate)
        wps = generate_waypoints(path, grid, strat;
                                 spacing_min = SMIN, spacing_max = SMAX,
                                 density_threshold = nothing)  # no level at 0.5 crossing spuriously
        gaps, line = _line_gaps(wps, 1)
        @test maximum(gaps) <= SMAX + 1e-6
        @test median(gaps) >= SMAX - 3.0        # overwhelmingly max spacing
    end

    # -----------------------------------------------------------------------
    # 5. Multiple flight lines + turnarounds → line_id + no duplicate corners
    # -----------------------------------------------------------------------
    @testset "multi-line: line_id correct, no duplicate corners" begin
        x_lo, x_hi, hw = 110.0, 190.0, 4.0
        prof(x) = _edge(x, x_lo, hw) - _edge(x, x_hi, hw)
        grid = _grid_from_profile(prof; Y = 200.0, ny = 60)
        spec = LawnmowerSpec(xmin = 0.0, xmax = 300.0, ymin = 0.0, ymax = 160.0,
                             spacing = 40.0, yaw_deg = 0.0, primary = :x, start = :low)
        path = lawnmower_from_extents(spec)
        nlines = count(isodd, 1:length(path)-1)   # informative only
        strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
        wps = generate_waypoints(path, grid, strat;
                                 spacing_min = SMIN, spacing_max = SMAX,
                                 density_threshold = 0.5)

        # line_id nonzero everywhere
        @test all(w -> w.line_id >= 1, wps)
        ids = sort(unique(w.line_id for w in wps))
        @test length(ids) >= 5                       # 5 survey lines at 40 m over 160 m
        # each survey-line group is one horizontal sweep (single modal y) plus,
        # optionally, a vertical turnaround transit at a single constant x column.
        for lid in ids
            grp = [w for w in wps if w.line_id == lid]
            ykeys = round.([w.y for w in grp]; digits = 3)
            modal_y = argmax(v -> count(==(v), ykeys), unique(ykeys))
            sweep = [w for w in grp if round(w.y; digits = 3) == modal_y]
            transit = [w for w in grp if round(w.y; digits = 3) != modal_y]
            @test length(sweep) >= 2                        # a real horizontal sweep
            # transit points (if any) share one constant x → a clean vertical hop
            @test length(unique(round.(w.x for w in transit; digits = 3))) <= 1
        end
        # NO consecutive duplicate coordinates anywhere in the ordered path
        dups = count(k -> hypot(wps[k+1].x - wps[k].x, wps[k+1].y - wps[k].y) < 1e-6,
                     1:length(wps)-1)
        @test dups == 0
        # global upper bound within survey lines
        for lid in ids
            g, _ = _line_gaps(wps, lid)
            isempty(g) && continue
            @test maximum(g) <= SMAX + 1e-6
        end
    end

    # -----------------------------------------------------------------------
    # 6. Spacing-bounds single source of truth (RC1): arguments win
    # -----------------------------------------------------------------------
    @testset "spacing bounds honour arguments not struct defaults" begin
        grid = uniform_grid(0.5; nx = 50, ny = 12,
                            xmin = 0.0, xmax = 300.0, ymin = 0.0, ymax = 60.0)
        path = _single_line_path(300.0)
        # struct carries the OLD 2/20 defaults; arguments request 10/30
        strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0,
                                     spacing_min = 2.0, spacing_max = 20.0)
        wps = generate_waypoints(path, grid, strat;
                                 spacing_min = 10.0, spacing_max = 30.0)
        gaps, _ = _line_gaps(wps, 1)
        @test maximum(gaps) <= 30.0 + 1e-6
        @test maximum(gaps) > 20.0 + 1e-6      # would be impossible if struct 20 won
        @test all(g -> g >= 10.0 - 1e-6, gaps[1:end-1])  # min honoured (interior)
    end

    # -----------------------------------------------------------------------
    # 7. STRICT MINIMUM SPACING INVARIANT
    #     No two consecutive EMITTED waypoints (whole ordered sequence, not just
    #     per-line) are closer than spacing_min. Also asserts the max cap in
    #     quiet regions, correct line_id, no duplicates, refinement coverage,
    #     and a deterministic resolution for an infeasible (sub-min) event.
    # -----------------------------------------------------------------------
    @testset "strict minimum spacing invariant" begin
        TOL = 1e-6

        # all consecutive Euclidean gaps over the FULL ordered waypoint list
        _all_gaps(wps) = [hypot(wps[k+1].x - wps[k].x, wps[k+1].y - wps[k].y)
                          for k in 1:length(wps)-1]

        # ---- (a) helper-level unit tests: _span_interior never emits sub-min ----
        @testset "_span_interior gaps ∈ [min,max] by construction" begin
            spanpts(a, b; fine) = KDEFlightPlanning._span_interior(a, b, 10.0, 30.0; fine=fine)
            gapsof(a, b; fine) = diff(vcat(a, spanpts(a, b; fine=fine), b))
            for span in vcat(collect(10.5:0.37:120.0), [30.0, 60.0, 90.0, 35.0, 65.0, 31.0])
                gq = gapsof(0.0, span; fine=false)   # quiet
                ge = gapsof(0.0, span; fine=true)    # event
                @test all(g -> g >= 10.0 - TOL, gq)       # strict min (quiet)
                @test all(g -> g <= 30.0 + TOL, gq)       # max cap (quiet)
                @test all(g -> g >= 10.0 - TOL, ge)       # strict min (event)
                @test all(g -> g < 2*10.0 + TOL, ge)      # event step < 2·min (fine)
            end
        end

        # ---- (b) uniform: every whole-sequence gap ≥ min, ≤ max -----------------
        @testset "uniform → all gaps in [min,max]" begin
            grid = uniform_grid(0.5; nx = 80, ny = 20,
                                xmin = 0.0, xmax = 397.0, ymin = 0.0, ymax = 100.0)
            path = _single_line_path(397.0)   # not a multiple of 30 → forces remainder
            strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
            wps = generate_waypoints(path, grid, strat;
                                     spacing_min = SMIN, spacing_max = SMAX)
            g = _all_gaps(wps)
            @test minimum(g) >= SMIN - TOL
            @test maximum(g) <= SMAX + TOL
            @test all(w -> w.line_id >= 1, wps)
        end

        # ---- (c) broad plateau: strict min everywhere + refinement each side ----
        @testset "broad plateau: strict min + two-sided refinement" begin
            x_lo, x_hi, hw = 110.0, 190.0, 4.0
            prof(x) = _edge(x, x_lo, hw) - _edge(x, x_hi, hw)
            grid = _grid_from_profile(prof)
            path = _single_line_path(300.0)
            strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
            wps = generate_waypoints(path, grid, strat;
                                     spacing_min = SMIN, spacing_max = SMAX,
                                     density_threshold = 0.5, event_margin = SMIN)
            g = _all_gaps(wps)
            @test minimum(g) >= SMIN - TOL
            @test maximum(g) <= SMAX + TOL
            gaps, line = _line_gaps(wps, 1)
            @test all(gg -> gg >= SMIN - TOL, gaps)
            # ≥1 refinement waypoint on EACH side of each edge (feasible here)
            xs_line = [w.x for w in line]
            for x0 in (x_lo, x_hi)
                @test any(x -> x0 - 2SMIN - TOL <= x < x0, xs_line)  # anticipatory
                @test any(x -> x0 < x <= x0 + 2SMIN + TOL, xs_line)  # trailing
            end
        end

        # ---- (d) overlapping / adjacent events merge without sub-min gaps -------
        @testset "overlapping events: no sub-min at merged edges" begin
            # two bumps closer than 2·event_margin → their bands overlap/merge
            b1, b2, hw = 150.0, 168.0, 3.0
            prof(x) = exp(-0.5*((x-b1)/hw)^2) + exp(-0.5*((x-b2)/hw)^2)
            grid = _grid_from_profile(prof)
            path = _single_line_path(300.0)
            strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
            wps = generate_waypoints(path, grid, strat;
                                     spacing_min = SMIN, spacing_max = SMAX,
                                     density_threshold = 0.5, event_margin = SMIN)
            g = _all_gaps(wps)
            @test minimum(g) >= SMIN - TOL
            @test maximum(g) <= SMAX + TOL
        end

        # ---- (e) short residual segment (L just over min) -----------------------
        @testset "short residual segment degrades gracefully" begin
            # a path whose single segment is only slightly longer than min:
            grid = uniform_grid(0.5; nx = 20, ny = 8,
                                xmin = 0.0, xmax = 12.0, ymin = 0.0, ymax = 12.0)
            path = _single_line_path(12.0)     # L = 12  (> min=10, < max=30)
            strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
            wps = generate_waypoints(path, grid, strat;
                                     spacing_min = SMIN, spacing_max = SMAX)
            g = _all_gaps(wps)
            @test length(wps) == 2                 # only endpoints fit
            @test minimum(g) >= SMIN - TOL
            @test maximum(g) <= SMAX + TOL
        end

        # ---- (f) multi-line turns: strict min ACROSS corners, correct line_id ---
        @testset "multi-line turns: strict min across corners" begin
            x_lo, x_hi, hw = 110.0, 190.0, 4.0
            prof(x) = _edge(x, x_lo, hw) - _edge(x, x_hi, hw)
            grid = _grid_from_profile(prof; Y = 200.0, ny = 60)
            spec = LawnmowerSpec(xmin = 0.0, xmax = 300.0, ymin = 0.0, ymax = 160.0,
                                 spacing = 40.0, yaw_deg = 0.0, primary = :x, start = :low)
            path = lawnmower_from_extents(spec)
            strat = CurvatureGuidedSpeed(grid; vmin = 2.0, vmax = 8.0)
            wps = generate_waypoints(path, grid, strat;
                                     spacing_min = SMIN, spacing_max = SMAX,
                                     density_threshold = 0.5)
            g = _all_gaps(wps)
            @test minimum(g) >= SMIN - TOL             # incl. turn/corner gaps
            @test all(w -> w.line_id >= 1, wps)
            dups = count(x -> x < 1e-6, g)
            @test dups == 0
        end

        # ---- (g) INFEASIBLE narrow event: deterministic collapse to one point ---
        @testset "infeasible narrow event collapses (no sub-min)" begin
            # Directly exercise the placer with an event interval NARROWER than
            # spacing_min. It cannot host both edges without a sub-min gap, so it
            # is deterministically collapsed to a single representative point.
            L = 200.0
            narrow = [(100.0, 105.0)]              # width 5 < spacing_min 10
            pos = KDEFlightPlanning._place_positions(L, narrow;
                        spacing_min = 10.0, spacing_max = 30.0, min_step = 0.5)
            gaps = diff(pos)
            @test minimum(gaps) >= 10.0 - TOL      # strict min preserved
            @test maximum(gaps) <= 30.0 + TOL
            @test pos[1] == 0.0 && pos[end] == L   # endpoints retained
            # exactly one representative point lands in the collapsed band
            @test count(p -> 100.0 - TOL <= p <= 105.0 + TOL, pos) == 1
            # a wide (feasible) event, by contrast, keeps BOTH edges + interior
            wide = KDEFlightPlanning._place_positions(L, [(60.0, 120.0)];
                        spacing_min = 10.0, spacing_max = 30.0, min_step = 0.5)
            @test minimum(diff(wide)) >= 10.0 - TOL
            @test count(p -> 60.0 - TOL <= p <= 120.0 + TOL, wide) >= 3
        end
    end
end
