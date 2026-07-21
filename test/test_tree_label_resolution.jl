# ===========================================================================
# Portable tests for interactive vegetation-label resolution (metadata.jl).
#
# Standalone: run with
#     julia --project=. test/test_tree_label_resolution.jl
#
# Covers, using only injectable seams (in_io/out_io/is_tty/label_selector) and
# temporary directories — NO machine-specific absolute paths:
#   - parse_cluster_id_input / validate_tree_labels    (pure parsing/validation)
#   - interactive_tree_labels                          (reprompt + non-TTY)
#   - resolve_autok_tree_labels                        (no silent [1] default)
#   - build_mask_autok                                 (end-to-end on synthetic
#                                                        GeoTIFF via ArchGDAL)
#
# Regression note: these paths must never throw `UndefVarError: isatty` on
# Julia 1.12 — the TTY gate is qualified as `Base.isatty` in metadata.jl.
# ===========================================================================

using Test
using KDEFlightPlanning
using ArchGDAL

@testset "tree-label resolution (interactive selector)" begin

    # -----------------------------------------------------------------------
    # parse_cluster_id_input / validate_tree_labels
    # -----------------------------------------------------------------------
    @testset "parse_cluster_id_input / validate_tree_labels" begin
        @test parse_cluster_id_input("1,3", [1, 2, 3, 4]) == [1, 3]
        @test parse_cluster_id_input("3 1", [1, 2, 3]) == [1, 3]      # sorted
        @test parse_cluster_id_input("  2 , 3 ", [1, 2, 3]) == [2, 3] # whitespace ok
        # Reprompt signals (nothing): empty, non-int, not-available, duplicate.
        @test parse_cluster_id_input("", [1, 2]) === nothing
        @test parse_cluster_id_input("x", [1, 2]) === nothing
        @test parse_cluster_id_input("5", [1, 2, 3]) === nothing      # not available
        @test parse_cluster_id_input("1,1", [1, 2]) === nothing       # duplicate

        @test validate_tree_labels([2, 1], [1, 2, 3]) == [1, 2]       # sorted, unique
        @test_throws ErrorException validate_tree_labels(Int[], [1, 2])   # empty
        @test_throws ErrorException validate_tree_labels([9], [1, 2])     # out of range
        @test_throws ErrorException validate_tree_labels([1, 1], [1, 2])  # duplicate
        @test_throws ErrorException validate_tree_labels(nothing, [1, 2]) # nothing
    end

    # -----------------------------------------------------------------------
    # interactive_tree_labels — validate + reprompt with injected IO/TTY
    # -----------------------------------------------------------------------
    @testset "interactive_tree_labels: reprompt + non-TTY fallback" begin
        labels = repeat([1, 2, 3], inner = 4)   # clusters {1,2,3}, length 12
        H, W = 3, 4                              # H*W == length(labels)
        # `img` is a non-Colorant sentinel → save_cluster_overlays writes .txt
        # summaries (no image encoder), keeping the test deterministic.
        img = "synthetic-non-colorant"
        mktempdir() do dir
            # First two lines invalid (out-of-range, duplicate), third valid.
            inbuf  = IOBuffer("0\n2,2\n2 3\n")
            outbuf = IOBuffer()
            sel = interactive_tree_labels(img, labels, H, W;
                out_dir = dir, in_io = inbuf, out_io = outbuf, is_tty = true)
            @test sel == [2, 3]
            @test occursin("Invalid", String(take!(outbuf)))          # reprompted
        end
        mktempdir() do dir
            # Non-TTY → empty vector (caller must then require/resolve labels).
            @test interactive_tree_labels(img, labels, H, W;
                out_dir = dir, is_tty = false) == Int[]
        end
    end

    # -----------------------------------------------------------------------
    # resolve_autok_tree_labels — explicit / interactive / error contract
    # -----------------------------------------------------------------------
    @testset "resolve_autok_tree_labels: no silent default" begin
        labels = repeat([1, 2, 3], inner = 4)
        H, W, k = 3, 4, 3
        img = "synthetic-non-colorant"

        # Explicit, nonempty labels: used verbatim, selector never invoked.
        boom = (a...) -> error("selector must not run for explicit labels")
        @test resolve_autok_tree_labels([2], img, labels, k, H, W;
            interactive = true, is_tty = true, label_selector = boom) == [2]

        # Absent + TTY → injected selector result is validated and returned.
        @test resolve_autok_tree_labels(nothing, img, labels, k, H, W;
            interactive = true, is_tty = true,
            label_selector = (a...) -> [1, 3]) == [1, 3]

        # Absent + non-TTY → hard error (never guesses [1]).
        @test_throws ErrorException resolve_autok_tree_labels(nothing, img, labels, k, H, W;
            interactive = true, is_tty = false)
        # interactive=false enforces the same non-interactive contract.
        @test_throws ErrorException resolve_autok_tree_labels(nothing, img, labels, k, H, W;
            interactive = false, is_tty = true)

        # Out-of-range / empty selector results are rejected.
        @test_throws ErrorException resolve_autok_tree_labels(nothing, img, labels, k, H, W;
            interactive = true, is_tty = true, label_selector = (a...) -> [99])
        @test_throws ErrorException resolve_autok_tree_labels(nothing, img, labels, k, H, W;
            interactive = true, is_tty = true, label_selector = (a...) -> Int[])
    end

    # -----------------------------------------------------------------------
    # build_mask_autok — end-to-end label resolution on a synthetic GeoTIFF
    # -----------------------------------------------------------------------
    @testset "build_mask_autok: interactive label resolution" begin
        AG_ = ArchGDAL
        mktempdir() do dir
            path = joinpath(dir, "autok.tif")
            H, W = 24, 24
            R = Array{UInt8}(undef, H, W); G = similar(R); B = similar(R)
            for j in 1:H, i in 1:W
                if i <= W ÷ 2               # left half: green (vegetation-like)
                    R[j, i] = 0x20; G[j, i] = 0xC0; B[j, i] = 0x30
                else                        # right half: brown (bare-like)
                    R[j, i] = 0xA0; G[j, i] = 0x60; B[j, i] = 0x20
                end
            end
            gt = [0.0, 1.0, 0.0, Float64(H), 0.0, -1.0]
            AG_.create(path; driver = AG_.getdriver("GTiff"),
                              width = W, height = H, nbands = 3, dtype = UInt8) do ds
                AG_.setgeotransform!(ds, gt)
                AG_.write!(ds, permutedims(R), 1)
                AG_.write!(ds, permutedims(G), 2)
                AG_.write!(ds, permutedims(B), 3)
            end
            ov = joinpath(dir, "ov")

            # Explicit labels bypass the prompt entirely and land in info verbatim.
            _, info = build_mask_autok(path; ks = 2:3, nsample = 200,
                tree_labels = [2], label_selector = (a...) -> error("must not run"),
                isatty_fn = () -> true, overlay_outdir = ov, do_cleanup = false)
            @test info.tree_labels == [2]
            @test info.tree_labels_source == :explicit

            # Missing labels + TTY selector → returns labels; selector called ONCE.
            calls = Ref(0)
            sel = function (img, lf, H2, W2, k)
                calls[] += 1
                @test length(lf) == H2 * W2      # full-resolution labels handed in
                return [1]
            end
            _, info2 = build_mask_autok(path; ks = 2:3, nsample = 200,
                tree_labels = nothing, label_selector = sel,
                isatty_fn = () -> true, overlay_outdir = ov, do_cleanup = false)
            @test calls[] == 1
            @test info2.tree_labels == [1]
            @test info2.tree_labels_source == :interactive
            @test 2 <= info2.k <= 3

            # Out-of-range selector result is rejected.
            @test_throws ErrorException build_mask_autok(path; ks = 2:3, nsample = 200,
                tree_labels = nothing, label_selector = (a...) -> [info2.k + 5],
                isatty_fn = () -> true, overlay_outdir = ov, do_cleanup = false)

            # Missing labels + non-TTY → clear error, never a silent [1].
            @test_throws ErrorException build_mask_autok(path; ks = 2:3, nsample = 200,
                tree_labels = nothing, isatty_fn = () -> false,
                overlay_outdir = ov, do_cleanup = false)
        end
    end
end
