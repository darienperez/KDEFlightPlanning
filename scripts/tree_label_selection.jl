# ---------------------------------------------------------------------------
# scripts/tree_label_selection.jl
#
# Pure, dependency-light helpers for the cross-site preprocessing producer's
# vegetation-cluster label workflow:
#
#   • distinguishing an explicitly-configured `tree_labels` from an
#     absent/empty one (NO silent default),
#   • parsing + validating interactive cluster-id input (reprompt on invalid),
#   • a preview-and-readline prompt loop, and
#   • a line-preserving, atomic, targeted update of a single `[[site]]` block in
#     the run's TOML config (no whole-document TOML.print rewrite).
#
# These functions use ONLY the Julia stdlib (no CairoMakie / KDEFlightPlanning),
# so they can be unit-tested in isolation (see test/runtests.jl).
# ---------------------------------------------------------------------------

"""
    tree_labels_slug(name) -> String

Slug used to match a site robustly by name (mirrors `_slug` in the producer).
"""
tree_labels_slug(name::AbstractString) =
    lowercase(replace(strip(name), r"[^A-Za-z0-9]+" => "_")) |> s -> strip(s, '_')

"""
    site_configured_tree_labels(site) -> Union{Vector{Int}, Nothing}

Return the explicitly-configured, nonempty integer `tree_labels` from a parsed
`[[site]]` table, or `nothing` when the key is absent, empty, or not a list of
integers. Never falls back to a default — an absent/empty list is reported as
`nothing` so the caller can prompt (TTY) or fail (non-TTY).
"""
function site_configured_tree_labels(site)
    haskey(site, "tree_labels") || return nothing
    v = site["tree_labels"]
    v isa AbstractVector || return nothing
    isempty(v) && return nothing
    labs = Int[]
    for x in v
        xi = try
            xf = Float64(x)
            (isfinite(xf) && xf == round(xf)) ? Int(round(xf)) : return nothing
        catch
            return nothing
        end
        push!(labs, xi)
    end
    isempty(labs) && return nothing
    return labs
end

"""
    parse_tree_label_input(line, k) -> Union{Vector{Int}, Nothing}

Parse a comma/whitespace separated list of cluster ids. Returns a sorted vector
of unique integers within `1:k`, or `nothing` if the input is empty, contains a
non-integer, has any id outside `1:k`, or contains duplicates. `nothing` is the
signal for the prompt loop to reprompt.
"""
function parse_tree_label_input(line::AbstractString, k::Integer)
    toks = [t for t in split(line, r"[,\s]+") if !isempty(t)]
    isempty(toks) && return nothing
    labs = Int[]
    for t in toks
        v = tryparse(Int, t)
        v === nothing && return nothing
        (1 <= v <= k) || return nothing
        push!(labs, v)
    end
    length(unique(labs)) == length(labs) || return nothing   # reject duplicates
    return sort(labs)
end

"""
    validate_configured_tree_labels(labels, k) -> Union{Nothing, String}

Return `nothing` if `labels` is a valid nonempty set of unique ids within
`1:k`, otherwise a human-readable reason string. Used to fail clearly on a
mis-configured (non-interactive) `tree_labels` rather than silently proceeding.
"""
function validate_configured_tree_labels(labels::AbstractVector{<:Integer}, k::Integer)
    isempty(labels) && return "tree_labels is empty"
    length(unique(labels)) == length(labels) ||
        return "tree_labels has duplicate ids: $labels"
    bad = filter(v -> !(1 <= v <= k), labels)
    isempty(bad) || return "tree_labels $bad out of range for chosen k=$k (valid 1:$k)"
    return nothing
end

"""
    render_tree_labels(labels) -> String

TOML inline-array rendering, e.g. `[1, 3]`.
"""
render_tree_labels(labels::AbstractVector{<:Integer}) =
    string("[", join(labels, ", "), "]")

"""
    stdin_is_tty(io = stdin) -> Bool

Whether `io` is an interactive terminal. `isatty` lives in `Base` but is NOT
exported, so an unqualified `isatty(stdin)` in a script running in `Main` throws
`UndefVarError: isatty not defined`. This wrapper qualifies it as `Base.isatty`
(stable in the supported Julia ≥ 1.10) so the interactive-prompt gate works when
the producer is launched with `julia --project=. scripts/preprocess_site_image.jl`.
The `io` seam also lets the gate be exercised in unit tests (a non-TTY `IOBuffer`
returns `false` via Base's generic `isatty(::IO)` fallback).
"""
stdin_is_tty(io::IO = stdin) = Base.isatty(io)

"""
    prompt_tree_labels(k, overlay_paths; name, suggested, in_io=stdin, out_io=stdout)
        -> Vector{Int}

Preview the rendered cluster-overlay PNG paths, then read comma/whitespace
separated vegetation cluster ids from `in_io`, reprompting until the input
parses (via `parse_tree_label_input`) to a valid nonempty set of unique ids
within `1:k`. Returns the sorted ids.

The caller MUST gate this on an interactive TTY; the `in_io`/`out_io` seams
exist so the loop can be unit-tested with in-memory buffers.
"""
function prompt_tree_labels(k::Integer, overlay_paths::AbstractVector{<:AbstractString};
                            name::AbstractString, suggested::Integer,
                            in_io::IO = stdin, out_io::IO = stdout,
                            max_attempts::Integer = 100)
    println(out_io, "\n──── Vegetation cluster review — $name ────")
    println(out_io, "Chosen k = $k. Review the cluster overlays, then choose which")
    println(out_io, "cluster id(s) correspond to vegetation/canopy:")
    for p in overlay_paths
        println(out_io, "    • $p")
    end
    println(out_io, "Hint: greenest cluster (−mean CIELAB a*) is k=$suggested — ",
                    "verify visually; do NOT trust the hint blindly.")
    attempts = 0
    while attempts < max_attempts
        attempts += 1
        print(out_io, "Enter vegetation cluster id(s) in 1:$k, comma-separated (e.g. 1,3): ")
        flush(out_io)
        line = readline(in_io)
        sel = parse_tree_label_input(line, k)
        if sel === nothing
            println(out_io, "  ✗ Invalid — expected unique integers within 1:$k. Please try again.")
            continue
        end
        println(out_io, "  ✓ Selected tree_labels = $sel")
        return sel
    end
    error("prompt_tree_labels: no valid input after $max_attempts attempts.")
end

"""
    persist_tree_labels(config_path, site_name, labels) -> (ok::Bool, message::String)

Line-preserving, atomic update of the single `[[site]]` block whose `name`
matches `site_name` (exactly, or by slug). Replaces an existing `tree_labels =`
line — preserving its leading indentation and `key =` prefix — or inserts one
just after the block's `name = ...` line if absent. Writes to a temp file in the
SAME directory, then renames over the original, so every other line (comments,
ordering, unrelated site blocks, top-level keys) is preserved byte-for-byte. No
`TOML.print` round-trip.

Returns `(false, reason)` WITHOUT mutating anything when the site is missing or
when the name is ambiguous (multiple matching blocks). No backup file is written
(the config is VCS-tracked and the replace is atomic), which avoids accumulating
stray `.bak` files across runs/sites.
"""
function persist_tree_labels(config_path::AbstractString, site_name::AbstractString,
                             labels::AbstractVector{<:Integer})
    isfile(config_path) || return (false, "config not found: $config_path")
    lines = readlines(config_path)                       # line endings stripped

    site_hdr = r"^\s*\[\[\s*site\s*\]\]\s*$"
    any_tbl  = r"^\s*\["                                 # next table/aot header
    name_re  = r"^\s*name\s*=\s*\"(.*?)\"\s*(#.*)?$"

    starts = findall(i -> occursin(site_hdr, lines[i]), eachindex(lines))
    isempty(starts) && return (false, "no [[site]] blocks in $config_path")

    block_end(s) = begin
        stop = length(lines)
        for j in (s + 1):length(lines)
            if occursin(any_tbl, lines[j])
                stop = j - 1
                break
            end
        end
        stop
    end

    target_slug = tree_labels_slug(site_name)
    matches = Tuple{Int,Int}[]                           # (start, stop) inclusive
    for s in starts
        e = block_end(s)
        for j in s:e
            m = match(name_re, lines[j])
            if m !== nothing
                nm = m.captures[1]
                if nm == site_name || tree_labels_slug(nm) == target_slug
                    push!(matches, (s, e))
                end
                break                                    # only the first name per block
            end
        end
    end

    isempty(matches) &&
        return (false, "no [[site]] block with name/slug matching \"$site_name\"")
    length(matches) > 1 &&
        return (false, "ambiguous: $(length(matches)) [[site]] blocks match \"$site_name\" — refusing to mutate")

    s, e = matches[1]
    rendered = render_tree_labels(labels)

    tl_idx = 0
    for j in s:e
        if occursin(r"^\s*tree_labels\s*=", lines[j])
            tl_idx = j
            break
        end
    end

    if tl_idx != 0
        m = match(r"^(\s*tree_labels\s*=)([ \t]*)(.*)$", lines[tl_idx])
        prefix = m.captures[1]
        sp     = (m.captures[2] === nothing || isempty(m.captures[2])) ? " " : m.captures[2]
        lines[tl_idx] = string(prefix, sp, rendered)
    else
        insert_at = s
        indent = ""
        for j in s:e
            m = match(name_re, lines[j])
            if m !== nothing
                insert_at = j
                indent = match(r"^(\s*)", lines[j]).captures[1]
                break
            end
        end
        insert!(lines, insert_at + 1, string(indent, "tree_labels = ", rendered))
    end

    content = join(lines, "\n") * "\n"
    dir = dirname(abspath(config_path))
    tmppath, io = mktemp(dir)
    try
        write(io, content)
        close(io)
        mv(tmppath, abspath(config_path); force = true)
    catch err
        try; close(io); catch; end
        isfile(tmppath) && rm(tmppath; force = true)
        return (false, "atomic write failed: $(sprint(showerror, err))")
    end
    return (true, "")
end
