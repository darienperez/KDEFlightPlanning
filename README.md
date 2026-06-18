# KDEFlightPlanning

Reproducible analysis code for the manuscript:

> **KDE-Guided Offline Variable-Speed Flight Planning for UAV LiDAR in Forested Terrain**
> Darien D. Perez Martin, Adam G. Hunsaker, Jennifer M. Jacobs, May-Win Thein
> Target venue: *Remote Sensing* (MDPI), in preparation.

The package implements a KDE-guided adaptive flight-planning pipeline and the
statistical analysis used to evaluate it:

k-medoids clustering of an orthomosaic (CIELAB colorspace) → KDE density surface
(Epanechnikov kernel, Silverman's rule) → normalized speed map (2–8 m/s) with
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
2. **Block-bootstrap CR / ΔCR confidence intervals** (`scripts/run_bootstrap.jl`)
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
| `--smoke`           | Synthetic, no-large-data path (always succeeds offline).                |
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
needs no large data and always succeeds offline.

If a required input declared in the config is missing, the affected stage is
reported as skipped (the planning stage prints a clear `MissingInputError`
naming the absent file/key, and stages 2–3a still run from the bundled
ground-truth tables). **No results are invented** — missing large data simply
means the corresponding stage is skipped.

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

The only path you must supply for the full planning stage is `[paths].rgb`; the
packaged config ships with a placeholder filename (the full-resolution Durham
ortho is not bundled).

---

## Inputs and outputs

**Bundled under `data/ground_truth/`** (small; reproduce the bootstrap CIs and
characterisation figures with no extra inputs):

- `counts.json` — cover × mission ground-return count grids (263×324). Treated
  as **ground truth**; never recomputed from unavailable raw data.
- `stats_and_coverage.csv`, `percent_densities.csv` — manuscript summary tables.
- `E_density_aware__waypoints_xy.csv` — planned KDE-guided waypoints.

**You must supply** for the full planning stage:

- `[paths].rgb` — an orthomosaic GeoTIFF. The full-resolution Durham ortho is
  **not** bundled (multi-GB); a lower-resolution ortho is fine — set
  `cluster_stride` to taste.

**Optional (not bundled):**

- The large flown-track trajectory CSVs (`[paths.trajectory]`, tens of MB each)
  — needed only for the trajectory/tracking figures (Figures 1–3 + the main
  bootstrap CI figure).
- Raw per-mission LAS files (`[paths.lidar]`, multi-GB) — needed only to
  recompute `counts.json` from scratch.
- GLI cover-class GeoTIFF / `gli_class_raster` — optional external cover ground
  truth and fixed line-scan CR denominators.

The large raw rasters, LAS files, and trajectory CSVs are **not** required for
the test suite or the smoke run.

### Outputs (written under the config's `outdir`, e.g. `output/durham/`)

- KDE density surface, speed map, and per-mission waypoint CSVs (planning stage).
- `bootstrap/` — block-bootstrap CR / ΔCR confidence intervals
  (n = 5000, block_frac = 3%, seed = 42).
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

---

## Reproducibility notes

- Block-bootstrap inference uses n = 5000 resamples, block_frac = 3%, seed = 42.
- Smoke/diagnostic outputs are labelled as such and must **not** be used for
  primary inferential claims; GLI remains the primary external cover
  ground-truth reference (Sullivan et al. 2023, DOI 10.3390/rs15215091).
