#!/usr/bin/env python3
"""
scripts/fallback_preview_panel.py

*** VISUAL QA FALLBACK ONLY — NOT PIPELINE OUTPUT ***

This is a NumPy/Matplotlib re-implementation of the KDEFlightPlanning planning
stages, used only to eyeball the cross-site panel layout when a Julia runtime
is unavailable. It approximates — it does NOT reproduce — the package:

  (a) sRGB→CIELAB → k-means (proxy for k-medoids) → greenest-cluster mask
  (b) Epanechnikov-kernel KDE of the mask, normalised to [0, 1]
  (c) boustrophedon path coloured by inverse-linear 2–8 m/s speed

Differences from the real pipeline (why this is QA-only):
  • k-means, not k-medoids; no CIELAB feature standardisation / PCA / vote-k.
  • KDE bandwidth is a fixed fraction of the image, not Silverman/Scott auto.
  • Column (c) is ALWAYS image-space provisional here (spacing in pixels).
The authoritative artefacts come from scripts/preprocess_site_image.jl +
scripts/make_cross_site_panel.jl under Julia.

Durham native GSD (GT_NATIVE, ≈0.02837 m/px) is shown as an *assumed* relative
scale annotation only.

Usage:
    python3 scripts/fallback_preview_panel.py \
        --config config/cross_site_panel.toml \
        --out /home/user/workspace/cross_site_panel_preview.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import convolve

try:  # stdlib in 3.11+
    import tomllib
    def _load_toml(p):
        with open(p, "rb") as f:
            return tomllib.load(f)
except ModuleNotFoundError:  # pragma: no cover
    import toml
    def _load_toml(p):
        return toml.load(p)

# Durham native ortho GSD from src/kde_surface_io.jl GT_NATIVE, via
# geotransform_resolution = hypot(dx, y_rot), hypot(x_rot, dy).
GT_NATIVE = (341300.35, 0.028369669906313746, 0.0,
             4774892.895, 0.0, -0.02836868197004068)


def gt_resolution(gt):
    """(xres, yres) in metres/px, hypot of affine column/row axis vectors."""
    xres = np.hypot(gt[1], gt[4])  # per-column axis (GDAL 0-based GT[1],GT[4])
    yres = np.hypot(gt[2], gt[5])  # per-row axis    (GDAL 0-based GT[2],GT[5])
    return xres, yres


def srgb_to_lab(rgb):
    """rgb float [0,1] HxWx3 → CIELAB (L*,a*,b*), D65."""
    m = rgb > 0.04045
    lin = np.where(m, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    X = lin @ np.array([0.4124, 0.3576, 0.1805])
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    Z = lin @ np.array([0.0193, 0.1192, 0.9505])
    xr, yr, zr = X / 0.95047, Y / 1.0, Z / 1.08883

    def f(t):
        d = 6 / 29
        return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4 / 29)

    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def kmeans(X, k, seed=6213, iters=25):
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([X[lab == j].mean(0) if np.any(lab == j) else cent[j]
                        for j in range(k)])
        if np.allclose(new, cent):
            cent = new
            break
        cent = new
    return lab, cent


def epanechnikov_kernel(radius):
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    r2 = (xx ** 2 + yy ** 2) / (radius ** 2 + 1e-9)
    K = np.clip(1 - r2, 0, None)
    return K / K.sum()


def process(image_path, k=4, seed=6213, sample=6000):
    img = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64) / 255.0
    H, W = img.shape[:2]
    lab = srgb_to_lab(img)
    feats = lab.reshape(-1, 3)
    # standardise (like standardize_features!)
    mu, sig = feats.mean(0), feats.std(0) + 1e-9
    fz = (feats - mu) / sig
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fz), min(sample, len(fz)), replace=False)
    _, cent = kmeans(fz[idx], k, seed=seed)
    d = ((fz[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    labels = d.argmin(1).reshape(H, W)
    # greenest cluster = most negative mean a* (index 1 of Lab)
    green = [(-lab[..., 1][labels == j].mean()) if np.any(labels == j) else -1e9
             for j in range(k)]
    veg = int(np.argmax(green))
    mask = (labels == veg).astype(float)
    # (b) Epanechnikov KDE
    radius = max(4, int(round(min(H, W) * 0.03)))
    dens = convolve(mask, epanechnikov_kernel(radius), mode="nearest")
    dens = dens / (dens.max() + 1e-12)
    return dict(img=img, H=H, W=W, labels=labels, veg=veg, green=green,
                mask=mask, dens=dens, k=k)


def boustrophedon(dens, n_lines=11, vmin=2.0, vmax=8.0):
    """Image-space lawnmower path; speed = inverse-linear of local density."""
    H, W = dens.shape
    xs_lines = np.linspace(0.08 * W, 0.92 * W, n_lines)
    pts, spd = [], []
    for i, x in enumerate(xs_lines):
        ys = np.linspace(0.06 * H, 0.94 * H, 40)
        if i % 2:
            ys = ys[::-1]
        for y in ys:
            xi = int(np.clip(x, 0, W - 1))
            yi = int(np.clip(y, 0, H - 1))
            t = dens[yi, xi]
            pts.append((x, y))
            spd.append(vmax - (vmax - vmin) * t)  # dense → slow
    return np.array(pts), np.array(spd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/cross_site_panel.toml")
    ap.add_argument("--out", default="/home/user/workspace/cross_site_panel_preview.png")
    ap.add_argument("--repo", default=None,
                    help="repo root (defaults to the config file's parent's parent)")
    args = ap.parse_args()

    cfg = _load_toml(args.config)
    base = os.path.dirname(os.path.abspath(args.config))
    vmin = float(cfg.get("speed_min", 2.0))
    vmax = float(cfg.get("speed_max", 8.0))
    xr, yr = gt_resolution(GT_NATIVE)
    assumed_mpp = (xr + yr) / 2

    sites = cfg.get("site", [])
    rows = []
    for s in sites:
        img_rel = s.get("image")
        rows.append((s.get("name", "?"), img_rel, s))

    nrow = len(rows)
    fig, axes = plt.subplots(nrow, 3, figsize=(12, 3.4 * nrow),
                             constrained_layout=True)
    if nrow == 1:
        axes = axes[None, :]
    headers = ["(a) RGB + k-means veg overlay",
               "(b) Epanechnikov KDE surface",
               "(c) Boustrophedon by 2-8 m/s"]
    fig.suptitle("Cross-site planning panel — FALLBACK QA PREVIEW "
                 "(NumPy proxy, NOT KDEFlightPlanning pipeline output)",
                 fontsize=13, fontweight="bold", color="#7a1f1f")

    notes = {"generator": "scripts/fallback_preview_panel.py",
             "warning": "QA proxy (k-means/fixed-bw/image-space); not pipeline output.",
             "assumed_durham_gsd_m_per_px": assumed_mpp,
             "sites": []}

    sc_last = None
    for r, (name, img_rel, s) in enumerate(rows):
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(name, fontsize=12, fontweight="bold")
        if r == 0:
            for c in range(3):
                axes[r, c].set_title(headers[c], fontsize=11)

        if not img_rel:
            for c in range(3):
                axes[r, c].text(0.5, 0.5, f"{name}\n(no `image` — pipeline site)",
                                ha="center", va="center", fontsize=10,
                                transform=axes[r, c].transAxes)
                axes[r, c].set_facecolor("#e9e4dc")
            notes["sites"].append({"name": name, "status": "no image (pipeline-only row)"})
            continue

        path = img_rel if os.path.isabs(img_rel) else os.path.join(base, img_rel)
        path = os.path.normpath(path)
        res = process(path)
        H, W = res["H"], res["W"]

        # (a) overlay
        over = res["img"].copy()
        m = res["mask"].astype(bool)
        over[m] = 0.5 * over[m] + 0.5 * np.array([0.15, 0.85, 0.25])
        axes[r, 0].imshow(over)

        # (b) KDE
        im = axes[r, 1].imshow(res["dens"], cmap="viridis", vmin=0, vmax=1)

        # (c) waypoints
        pts, spd = boustrophedon(res["dens"], vmin=vmin, vmax=vmax)
        axes[r, 2].imshow(np.ones((H, W, 3)) * 0.96)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="cividis",
                            array=spd[:-1], norm=plt.Normalize(vmin, vmax),
                            linewidths=2)
        axes[r, 2].add_collection(lc)
        sc_last = axes[r, 2].scatter(pts[:, 0], pts[:, 1], c=spd, cmap="cividis",
                                     vmin=vmin, vmax=vmax, s=6)
        axes[r, 2].set_xlim(0, W); axes[r, 2].set_ylim(H, 0)
        span_m = W * assumed_mpp
        axes[r, 2].text(0.5, -0.06,
                        f"PROVISIONAL image-space  (assumed ≈{span_m:.0f} m wide "
                        f"@ {assumed_mpp:.4f} m/px)",
                        ha="center", va="top", transform=axes[r, 2].transAxes,
                        fontsize=8, color="red", fontweight="bold")

        notes["sites"].append({
            "name": name, "image": path, "width_px": W, "height_px": H,
            "k": res["k"], "suggested_veg_cluster_0based": res["veg"],
            "assumed_span_m": span_m,
            "column_c": "image-space provisional (pixels; assumed GSD annotation only)",
        })

    if sc_last is not None:
        cb = fig.colorbar(sc_last, ax=axes[:, 2], shrink=0.6, pad=0.02)
        cb.set_label("Target ground speed (m/s)")

    fig.savefig(args.out, dpi=150)
    print(f"[fallback] wrote {args.out}")
    notes_path = os.path.splitext(args.out)[0] + "_notes.json"
    with open(notes_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[fallback] wrote {notes_path}")


if __name__ == "__main__":
    main()
