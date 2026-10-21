"""
    load_cover_masks(path; nrows=263, ncols=324,
                     class_codes=Dict(:field=>2, :decid=>0, :conif=>1),
                     flip_y=false) -> Dict{Symbol,BitMatrix}

Read exact cover masks from band 1 of a categorical GeoTIFF. Class values are
read without scaling or resampling; zero is a valid cover code. GDAL's band
validity mask and NoData value exclude invalid cells. A NoData value that is
also a configured cover code is rejected because those meanings conflict.

The returned masks have `(nrows, ncols)` shape. GDAL's `(columns, rows)` array
is transposed once, preserving the raster's native row order (north to south
for a north-up raster), as used by `bin_to_count_grid`. Set `flip_y=true` only
when an explicitly known legacy count-grid convention requires reversed rows.
This function never infers a row flip, resizes, reprojects, or reconstructs
missing cover cells from return counts. Unrequested categorical values are
excluded from the returned masks.
"""
function load_cover_masks(path::AbstractString;
                          nrows::Integer=263,
                          ncols::Integer=324,
                          class_codes::AbstractDict=Dict(:field=>2, :decid=>0, :conif=>1),
                          flip_y::Bool=false)::Dict{Symbol,BitMatrix}
    nrows > 0 && ncols > 0 ||
        throw(ArgumentError("nrows and ncols must be positive"))
    isempty(class_codes) && throw(ArgumentError("class_codes must not be empty"))
    all(k -> k isa Symbol, keys(class_codes)) ||
        throw(ArgumentError("class_codes keys must be cover-name Symbols"))
    all(v -> v isa Integer && !(v isa Bool), values(class_codes)) ||
        throw(ArgumentError("class_codes values must be integers"))
    length(unique(collect(values(class_codes)))) == length(class_codes) ||
        throw(ArgumentError("class_codes values must be distinct"))
    isfile(path) || throw(ArgumentError("Cover-class raster not found: $path"))

    return ArchGDAL.read(path) do ds
        band = ArchGDAL.getband(ds, 1)
        nodata = ArchGDAL.getnodatavalue(band)
        if nodata !== nothing && any(code -> nodata == code, values(class_codes))
            throw(ArgumentError(
                "Cover-class raster NoData value $nodata is also a configured " *
                "cover code; supply an unambiguous categorical raster"))
        end

        # ArchGDAL reads (columns, rows). Keep native row order after this
        # single transpose so the masks align with the count-grid reader.
        classes = permutedims(ArchGDAL.read(band))
        size(classes) == (nrows, ncols) || throw(DimensionMismatch(
            "Cover-class raster has shape $(size(classes)); expected " *
            "($nrows, $ncols) (rows, columns). No resizing is performed."))
        valid = BitMatrix(permutedims(ArchGDAL.read(ArchGDAL.getmaskband(band))) .!= 0)
        size(valid) == size(classes) ||
            throw(DimensionMismatch("Cover-class validity mask has the wrong shape"))
        if nodata !== nothing
            valid .&= .!isequal.(classes, nodata)
        end

        for i in eachindex(classes, valid)
            if valid[i]
                value = classes[i]
                (value isa Real && isfinite(value) && isinteger(value)) ||
                    throw(ArgumentError(
                        "Cover-class raster must contain integer-valued categories " *
                        "at valid cells; found $value"))
            end
        end

        masks = Dict{Symbol,BitMatrix}()
        for (cover, code) in class_codes
            mask = BitMatrix(valid .& (classes .== code))
            masks[cover] = flip_y ? BitMatrix(reverse(mask; dims=1)) : mask
        end
        return masks
    end
end
