# KDEFlightPlanning

Reproducible analysis code for the manuscript:

> **KDE-Guided Offline Variable-Speed Flight Planning for UAV LiDAR in Forested Terrain**
> Darien D. Perez Martin, Adam G. Hunsaker, Jennifer M. Jacobs, May-Win Thein

The package implements a KDE-guided adaptive flight-planning pipeline and the
statistical analysis used to evaluate it:

k-medoids clustering of an orthomosaic (CIELAB colorspace) → KDE density surface
(Epanechnikov kernel, Scott-style index bandwidth) → normalized speed map (2–8 m/s) with
curvature/gradient-based waypoint spacing → coverage-ratio (CR) evaluation with
block-bootstrap confidence intervals.

The primary outcome metric is the **coverage ratio (CR)** — the fraction of
1 m² cells with ≥1 ground return — evaluated across three cover types
(open field, deciduous forest, coniferous forest) at the Durham, NH study site
(same site as Sullivan et al. 2023, DOI 10.3390/rs15215091).

---

## Requirements

- **Julia ≥ 1.10** (`[compat]` floor in `Project.toml`); developed and tested on
  1.12.6. Use 1.12.x to match the bundled `Manifest.toml` exactly.
- Dependencies are pinned in `Project.toml` / `Manifest.toml`. Heavy deps include
  CairoMakie, ArchGDAL, LASDatasets, Clustering, and MultivariateStats.

## Setup

From the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This installs and precompiles the exact dependency versions from the manifest.
(First run downloads/precompiles several hundred packages and can take a while.)

Verify the package loads:

```bash
julia --project=. -e 'using KDEFlightPlanning; println("OK")'
```

---

## Running the whole pipeline (one command)

`scripts/run_pipeline.jl` is the **single entry point** for the entire analysis.
It orchestrates every stage in order and delegates to the package functions in
`src/` (no analysis logic is re-implemented in the script):

1. **Ingest → mask → KDE → speed map → waypoints** (data-driven; `run_from_config.jl`)
2. **Block-bootstrap CR / ΔCR confidence intervals** (`scripts/run_bootstrap.jl`;
   requires the original GLI cover masks for manuscript inference).
3. **Figures** (CairoMakie):
   - **3a** Characterisation figures — from the bundled ground-truth tables only.
   - **3b** Trajectory analysis + publication figures (Figures 1–3 + the main
     bootstrap CI figure) — **optional**; needs the large flown-track CSVs under
     `[paths.trajectory]`. Skipped with a clear message if they are absent.

### Usage

```bash
julia --project=. scripts/run_pipeline.jl [CONFIG] [options]
```

| Argument / option   | Meaning                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `CONFIG`            | Path to a `RunInputs` TOML config. Default: `config/run_durham.toml`.   |
| `--smoke`           | Synthetic planning demo and explicitly diagnostic bootstrap output.    |
| `--allow-approximate-masks` | Opt into diagnostic positive-return-union masks when the exact GLI is unavailable. |
| `--skip-bootstrap`  | Skip stage 2.                                                           |
| `--skip-figures`    | Skip stage 3.                                                           |
| `-h`, `--help`      | Print usage and exit.                                                   |

### Examples

```bash
# Offline synthetic smoke run — no large rasters/LAS needed:
julia --project=. scripts/run_pipeline.jl --smoke

# Full reproduction against your local data (paths set in the TOML):
julia --project=. scripts/run_pipeline.jl config/run_durham.toml

# Print help:
julia --project=. scripts/run_pipeline.jl --help
```

`--smoke` builds a small **synthetic GeoTIFF fixture** and runs the *real*,
config-driven planning pipeline against it, then runs the bootstrap and
characterisation-figure stages against the bundled ground-truth tables. It
needs no large data after dependencies are installed. Its approximate bootstrap
outputs are saved under `bootstrap/diagnostic/`, carry
`support_mode = approximate_positive_union`, and are **not manuscript CIs**.
The synthetic fixture never substitutes for the Durham GLI cover map.

If a required input declared in the config is missing, the affected stage is
reported as skipped (the planning stage prints a clear `MissingInputError`
naming the absent file/key). Whole-cover characterization figures can use the
bundled tables, but exact bootstrap inference requires the original cover map.
The requested bootstrap stage stops if that map is absent; use
`--skip-bootstrap` for summary figures only, or explicitly opt into diagnostics.
Unexpected errors in requested stages propagate instead of reporting a
successful reproduction.

---

## Configuration

Edit `config/run_durham.toml` (or copy it) to point at your local data. All
paths are resolved **relative to the config file's directory**, so the bundled
`../data/ground_truth/` references work out of the box. Key sections:

- `[paths]` — `rgb` orthomosaic GeoTIFF (required for the planning stage; a
  lower-resolution ortho is fine — set `cluster_stride` accordingly), optional
  `gli` cover-class GeoTIFF, the bundled `counts_json` / `stats_csv` /
  `percent_densities_csv` summary tables, and `outdir`.
- `[paths.waypoints]` — `Ed` = bundled `E_density_aware__waypoints_xy.csv`.
- `[paths.trajectory]` — **optional** flown-track CSVs (`const2` / `missionE` /
  `const8`); not bundled. Present → trajectory + publication figures generate.
- `[paths.lidar]` — **optional** raw LAS per mission; only needed to recompute
  `counts.json` from scratch (the bundled `counts.json` is authoritative).
- `[planning]` — planning parameters: `seed`, `track_spacing_m`,
  `speed_bounds_mps = [2.0, 8.0]`, `kde_kernel = "epanechnikov"`,
  `kmedoids_k_range`, waypoint-spacing bounds, `cluster_stride`, `tree_labels`,
  `report_formats`.
- `[gli_class_codes]` — GLI raster class codes for field / deciduous / coniferous.
- `[paths].gli_class_raster` — the original 263-row × 324-column categorical
  cover raster, including cells with zero returns in every mission. Required
  for exact bootstrap and cover-specific spatial analyses; it is not bundled.
- `[analysis].cover_mask_flip_y` — explicit row-order correction for a known
  legacy raster only (default `false`). Class values are read without scaling,
  resizing, or resampling; class `0` remains deciduous, not missing data.

The only path you must supply for the full planning stage is `[paths].rgb`; the
packaged config ships with a placeholder filename (the full-resolution Durham
ortho is not bundled).

---

## Inputs and outputs

**Bundled under `data/ground_truth/`** (derived inputs for the analyses):

- `counts.json` — cover × mission return-count grids (263×324). The legacy
  vectors are column-major, with north in the first row and no row flip;
  alignment is verified against the original GLI class raster. Every analysis
  uses the same decoder. The schema
  written by `write_count_grids_json` is also supported and records its shape.
- `stats_and_coverage.csv`, `percent_densities.csv` — manuscript summary tables.
- `E_density_aware__waypoints_xy.csv` — planned KDE-guided waypoints.

**You must supply** for the full planning stage:

- `[paths].rgb` — an orthomosaic GeoTIFF. The full-resolution Durham ortho is
  **not** bundled (multi-GB); a lower-resolution ortho is fine — set
  `cluster_stride` to taste.

**Additional inputs (not bundled):**

- The large flown-track trajectory CSVs (`[paths.trajectory]`, tens of MB each)
  — needed only for the trajectory/tracking figures (Figures 1–3 + the main
  bootstrap CI figure).
- Raw per-mission LAS files (`[paths.lidar]`, multi-GB) — needed only to
  recompute `counts.json` from scratch.
- GLI cover-class GeoTIFF / `gli_class_raster` — required for exact bootstrap
  support and fixed line-scan CR denominators. Counts alone cannot locate cells
  that received no returns in any mission; fixed cover-area totals cannot
  replace that missing spatial information.

The large raw rasters, LAS files, and trajectory CSVs are **not** required for
the test suite or the smoke run.

The original GLI raster reports EPSG:26919, whereas the package's Durham
planning defaults use EPSG:6348. Its affine origin also differs from the
hardcoded planning-grid default. Count-array alignment with the GLI classes
is verified, but absolute alignment with orthomosaics and trajectories must
be checked from their source metadata. The bootstrap uses the original
mask cells without reprojection; spatial scripts use its affine transform.

### Outputs (written under the config's `outdir`, e.g. `output/durham/`)

- KDE density surface, speed map, and per-mission waypoint CSVs (planning stage).
- `bootstrap/` — block-bootstrap CR / ΔCR confidence intervals
  (n = 5000, block_frac = 3%, seed = 42), using exact GLI support. Each CSV
  records `N`, `support_N`, and `support_mode`. The optional positive-union
  diagnostics use a separate `bootstrap/diagnostic/` directory.
- `figures/` — characterisation figures, plus (if the trajectory CSVs are
  supplied) the publication and tracking figures.
- `trajectory/` — trajectory-analysis CSVs (only when `[paths.trajectory]` is
  supplied).

---

## Repository layout

```
src/                 Package source (the analysis library)
  KDEFlightPlanning.jl   Module entry point: includes + exports
  kde*.jl                KDE surface, density classes, strata
  clustering.jl,         k-medoids, CIELAB features, PCA
    colorspace.jl, features.jl, pca.jl
  speedmap.jl, path.jl,  Speed map + waypoint / lawnmower planning
    waypoints.jl
  bootstrap.jl,          CR metrics + block-bootstrap CIs
    metrics.jl, count_statistics.jl
  geotiff_io.jl,         Raster / GeoTIFF / LAS / trajectory I/O
    lidar_counts.jl, trajectory.jl, along_track.jl, ...
  figures.jl             All figure generation (publication + KDE figures)
  reports.jl, inputs.jl  Reporting helpers + TOML config (RunInputs)

scripts/                 All runnable entry-point scripts (single unified folder)
  run_pipeline.jl        ★ single entry point for the whole pipeline
  run_from_config.jl     TOML-driven full pipeline engine (stage 1)
  run_bootstrap.jl       block-bootstrap CR / ΔCR confidence intervals (stage 2)
  gen_characterization_figures.jl   manuscript characterisation figures (stage 3a)
  trajectory_analysis.jl regenerate trajectory-analysis CSV outputs (stage 3b)
  make_figures.jl        publication figures from trajectory CSVs (stage 3b)
  ground_truth.jl        shared config-driven ground-truth discovery helper
  make_synthetic_geotiff_fixture.jl  synthetic GeoTIFF/GLI fixture for --smoke
  basic_workflow.jl      end-to-end API demo on a synthetic density surface
  reproduce_all.jl       DEPRECATED — thin redirect to run_pipeline.jl

config/run_durham.toml   Run configuration (Durham, NH) — relative paths
data/ground_truth/       Bundled manuscript summary tables (counts.json, …)
test/runtests.jl         Unit + regression test suite (synthetic + ground-truth)
output/                  Generated artefacts
```

---

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite uses synthetic `RasterGrid` fixtures throughout and does **not**
require the multi-GB raw data. Regression testsets that compare against the
ground-truth summaries default to the **bundled** copies under
`data/ground_truth/` (`counts.json`, `stats_and_coverage.csv`,
`E_density_aware__waypoints_xy.csv`) and skip gracefully (with a warning) if a
file is absent.

To point the regression tests at alternate copies of the ground-truth files
(e.g. freshly recomputed tables), set any of:

```bash
export KDE_TEST_COUNTS_JSON=/path/to/counts.json
export KDE_TEST_STATS_CSV=/path/to/stats_and_coverage.csv
export KDE_TEST_WAYPOINTS_CSV=/path/to/E_density_aware__waypoints_xy.csv
```

To additionally check all bundled count grids against the original cover
raster, run the suite with the optional private input:

```bash
KDEFLIGHTPLANNING_GLI_CLASS_RASTER=/path/to/original_gli.tif \
  julia --project=. -e 'using Pkg; Pkg.test()'
```

This check verifies cover areas, includes never-hit cells, and rejects a
reflected mask with otherwise identical cover totals.

---

## Reproducibility notes

- Block-bootstrap inference uses n = 5000 resamples, block_frac = 3%, seed = 42.
- Blocks intersecting the fixed cover mask are sampled with replacement until
  at least the original number of cover cells has been retained. Paired
  comparisons use the same sampled blocks for both missions. CR estimates and
  their intervals use the same spatial support, including never-hit cells.
- Point estimates are observed CR values or observed CR differences; bootstrap
  replicates provide percentile intervals. They are not reported as the mean
  of the bootstrap replicates.
- Characterization plots select missions by canonical CSV labels and kernel
  identifiers. The CSV `Mean` is already conditional on covered cells and is
  used directly, without dividing by CR.
- The exact GLI raster and large source datasets are not bundled. Exact
  manuscript CIs must be regenerated with the original mask and compared with
  the submitted values; synthetic or approximate outputs do not establish that
  comparison.
- Smoke/diagnostic outputs are labelled as such and must **not** be used for
  primary inferential claims; GLI remains the primary external cover
  ground-truth reference (Sullivan et al. 2023, DOI 10.3390/rs15215091).

