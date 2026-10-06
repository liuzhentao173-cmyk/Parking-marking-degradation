#!/usr/bin/env python3
"""
Stage 5 - Measurement evaluation: uncertainty, maintenance grading and error
attribution of the degradation index D.

This script turns the four masks of an evaluation set into every number the
paper reports about the index as a *measurement*:

  * error decomposition and stage attribution        (paper Sec. 2.6.1, 5.2.3)
  * error by scene condition (shadow / severe wear)   (paper Sec. 5.2.4)
  * expanded uncertainty U = 2u at window, marking
    (object) and image support, Bland-Altman limits   (paper Sec. 5.3)
  * distribution against the five handbook ranks,
    the operational three-grade scale, grading
    accuracy and the repainting decision              (paper Sec. 5.4, Tables 7-8)

Masks (paper notation; internal tags in brackets)
-------------------------------------------------
  M_GT   [A1]  ground-truth marking footprint
  R_GT   [A2]  ground-truth residual paint
  M_Pred [A3]  Stage 1 output (01_segment_marking_roi.py)
  R_Pred [A4]  Stage 2 output (02_binarize_residual_paint.py)

The four folders must contain PNG masks with identical file names. Both the
{0,255} and the {0,1} encodings are accepted.

Index definitions (Eq. 5-7 of the paper)
----------------------------------------
  window  k :  U_k = M n W_k,  D_k = 1 - |R n U_k| / |U_k|
               (S = 512 px, stride 256 px, a side is valid when |U_k| >= 500 px,
                a window is valid when both the reference and the prediction are)
  image     :  D_img = 1 - sum_k |R n U_k| / sum_k |U_k| over the valid windows
  marking o :  connected markings of M_GT, extracted with the same eroded-seed
               rule as Stage 2;  D_GT(o) = 1 - |R_GT n o| / |o|,
               D_Pred(o) = 1 - |R_Pred n M_Pred n o| / |M_Pred n o|

The two cross-substituted realizations used for error attribution are
  D_Seg = 1 - |R_GT n M_Pred n W| / |M_Pred n W|   (segmentation error only)
  D_Bin = 1 - |R_Pred n M_GT n W| / |M_GT n W|     (binarization error only)

Optional inputs
---------------
  --window-csv      window_level_validation.csv from 04_downstream_gt_validation.py.
                    When given, the window table is read from it instead of being
                    recomputed (the two routes agree exactly).
  --stage2-objects  all_output_info.csv from 02_binarize_residual_paint.py, used to
                    flag windows crossed by a marking that Stage 2 found shadowed.
  --site-field N    underscore-separated field of the file name that identifies
                    the site; enables a per-site table with anonymised labels.

Confidence intervals resample whole parent images (cluster bootstrap), because
windows overlap by half their side and are not independent.

Usage
-----
  python 05_measurement_evaluation.py \
      --m-gt  data/M_GT  --r-gt  data/R_GT \
      --m-pred out/01/Pred_Mask  --r-pred out/02/03_final_mask \
      --window-csv out/04/window_level_validation.csv \
      --stage2-objects out/02/all_output_info.csv \
      --out out/05
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------------- #
# Constants (paper values)
# --------------------------------------------------------------------------- #
WINDOW_SIZE = 512
STRIDE = 256
MIN_UNIT_AREA = 500           # px, per side of a window
SEED_ERODE_KSIZE = 5          # as in 02_binarize_residual_paint.py
MIN_OBJECT_AREA = 40          # as in 02_binarize_residual_paint.py
MIN_PRED_OVERLAP = 40         # a marking must be recovered to be measurable

# Delamination-ratio bands of the Japanese road-marking handbook (JRMA 2012),
# Rank 5 (as constructed) .. Rank 1 (no longer recognisable).
HANDBOOK_EDGES = [0.03, 0.08, 0.23, 0.40]
HANDBOOK_LABELS = ["Rank 5", "Rank 4", "Rank 3", "Rank 2", "Rank 1"]

# Operational scale (paper Table 7): ranks that the measurement cannot resolve
# are merged; both boundaries are handbook boundaries.
GRADE_EDGES = [0.23, 0.40]
GRADE_LABELS = ["Grade I", "Grade II", "Grade III"]
GRADE_ACTIONS = ["Routine inspection", "Schedule repainting", "Repaint with priority"]
REPAINT_EDGE = 0.40

SEVERE_WEAR = 0.60            # D_GT above which a window counts as severely worn
MRE_MIN_GT = 0.10             # relative error evaluated only where D_GT > 0.10
TOLERANCE = 0.10              # share of measurements within +/-0.10


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def load_mask(path: Path) -> np.ndarray:
    img = np.array(Image.open(path).convert("L"))
    return img > (127 if img.max() > 1 else 0)


def window_starts(extent: int) -> list[int]:
    if extent <= WINDOW_SIZE:
        return [0]
    out = list(range(0, extent - WINDOW_SIZE + 1, STRIDE))
    if out[-1] + WINDOW_SIZE < extent:
        out.append(extent - WINDOW_SIZE)
    return out


def extract_objects(full: np.ndarray) -> list[np.ndarray]:
    """Eroded-seed connected components, identical to the Stage 2 rule."""
    full = full.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SEED_ERODE_KSIZE,) * 2)
    seed = cv2.erode(full, kernel, iterations=1)
    _, lab_full, _, _ = cv2.connectedComponentsWithStats(full, connectivity=8)
    n_seed, lab_seed, st_seed, _ = cv2.connectedComponentsWithStats(seed, connectivity=8)
    out, used = [], set()
    for sid in range(1, n_seed):
        if int(st_seed[sid, cv2.CC_STAT_AREA]) < MIN_OBJECT_AREA:
            continue
        labels = [int(v) for v in np.unique(lab_full[lab_seed == sid])
                  if int(v) != 0 and int(v) not in used]
        if not labels:
            continue
        obj = np.isin(lab_full, labels)
        if int(obj.sum()) < MIN_OBJECT_AREA:
            continue
        out.append(obj)
        used.update(labels)
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - half) / den, (centre + half) / den)


def band(values: np.ndarray, edges: list[float]) -> np.ndarray:
    """Band index: 0 = best. A value equal to an edge falls in the worse band."""
    return np.digitize(values, edges)


def site_of(name: str, field: int) -> str:
    raw = name.split("_")[field]
    for suffix in ("PLAN1", "PLAN2", "PLAN3"):
        raw = raw.replace(suffix, "")
    return raw


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def windows_from_masks(stem: str, m_gt, r_gt, m_pr, r_pr) -> list[dict]:
    rows = []
    h, w = m_gt.shape
    wid = 0
    for y in window_starts(h):
        for x in window_starts(w):
            wid += 1
            sl = (slice(y, y + WINDOW_SIZE), slice(x, x + WINDOW_SIZE))
            g, rg, p, rp = m_gt[sl], r_gt[sl], m_pr[sl], r_pr[sl]
            a_g, a_p = int(g.sum()), int(p.sum())
            if a_g < MIN_UNIT_AREA or a_p < MIN_UNIT_AREA:
                continue
            w_g, w_p = int((rg & g).sum()), int((rp & p).sum())
            rows.append({
                "parent": stem, "window_id": wid, "x": x, "y": y,
                "w": min(WINDOW_SIZE, w - x), "h": min(WINDOW_SIZE, h - y),
                "area_gt": a_g, "white_gt": w_g, "area_pred": a_p, "white_pred": w_p,
                "D_GT": 1 - w_g / a_g, "D_Pred": 1 - w_p / a_p,
                "D_Seg": 1 - int((rg & p).sum()) / a_p,
                "D_Bin": 1 - int((rp & g).sum()) / a_g,
                "weight": min(a_g, a_p),
            })
    return rows


def windows_from_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if r["pair_valid"] != "True":
                continue
            rows.append({
                "parent": r["parent"], "window_id": int(r["window_id"]),
                "x": int(_num(r["x"])), "y": int(_num(r["y"])),
                "w": int(_num(r["w"])), "h": int(_num(r["h"])),
                "area_gt": _num(r["GT_All_area"]), "white_gt": _num(r["GT_All_white"]),
                "area_pred": _num(r["Pred_area"]), "white_pred": _num(r["Pred_white"]),
                "D_GT": _num(r["GT_All_D"]), "D_Pred": _num(r["Pred_D"]),
                "D_Seg": _num(r["Seg_A2A3_D"]), "D_Bin": _num(r["Bin_A4A1_D"]),
                "weight": _num(r["pair_weight_area"]),
            })
    return [r for r in rows if not (math.isnan(r["D_GT"]) or math.isnan(r["D_Pred"]))]


def objects_from_masks(stem: str, m_gt, r_gt, m_pr, r_pr) -> list[dict]:
    rows = []
    for obj in extract_objects(m_gt):
        area = int(obj.sum())
        unit = obj & m_pr
        a_p = int(unit.sum())
        rows.append({
            "parent": stem, "area_px": area, "pred_area_px": a_p,
            "D_GT": 1 - int((r_gt & obj).sum()) / area,
            "D_Pred": (1 - int((r_pr & unit).sum()) / a_p)
            if a_p >= MIN_PRED_OVERLAP else float("nan"),
        })
    return rows


def images_from_windows(win: list[dict]) -> list[dict]:
    acc: dict[str, list[float]] = {}
    for r in win:
        a = acc.setdefault(r["parent"], [0.0, 0.0, 0.0, 0.0])
        a[0] += r["white_gt"]; a[1] += r["area_gt"]
        a[2] += r["white_pred"]; a[3] += r["area_pred"]
    return [{"parent": p, "D_GT": 1 - a[0] / a[1], "D_Pred": 1 - a[2] / a[3]}
            for p, a in sorted(acc.items())]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def metrology(gt: np.ndarray, pred: np.ndarray, support: str,
              weight: np.ndarray | None = None) -> dict:
    e = pred - gt
    sd = float(e.std(ddof=1))
    sel = gt > MRE_MIN_GT
    out = {
        "support": support, "n": int(e.size),
        "bias": float(e.mean()), "MAE": float(np.abs(e).mean()),
        "RMSE": float(math.sqrt((e ** 2).mean())),
        "aw_MAE": float((weight * np.abs(e)).sum() / weight.sum()) if weight is not None else float("nan"),
        "u": sd, "U_k2": 2 * sd,
        "LoA_lower": float(e.mean() - 1.96 * sd), "LoA_upper": float(e.mean() + 1.96 * sd),
        "MRE": float((np.abs(e[sel]) / gt[sel]).mean()) if sel.any() else float("nan"),
        "within_0.10": float((np.abs(e) <= TOLERANCE).mean()),
        "r": float(np.corrcoef(gt, pred)[0, 1]) if e.size > 2 else float("nan"),
    }
    return out


def grading(gt: np.ndarray, pred: np.ndarray, support: str) -> dict:
    g, p = band(gt, GRADE_EDGES), band(pred, GRADE_EDGES)
    n = g.size
    exact = int((g == p).sum())
    conf = np.zeros((3, 3), dtype=int)
    for a, b in zip(g, p):
        conf[a, b] += 1
    rep_ok = int(((gt >= REPAINT_EDGE) == (pred >= REPAINT_EDGE)).sum())
    lo, hi = wilson(exact, n)
    rlo, rhi = wilson(rep_ok, n)
    out = {
        "support": support, "n": n,
        "grade_accuracy": exact / n, "grade_ci_lo": lo, "grade_ci_hi": hi,
        "within_one_grade": float((np.abs(g - p) <= 1).mean()),
        "repaint_accuracy": rep_ok / n, "repaint_ci_lo": rlo, "repaint_ci_hi": rhi,
    }
    for i, lab in enumerate(GRADE_LABELS):
        col = conf[:, i].sum()
        row = conf[i, :].sum()
        out[f"precision_{lab}"] = conf[i, i] / col if col else float("nan")
        out[f"recall_{lab}"] = conf[i, i] / row if row else float("nan")
    out["_confusion"] = conf
    return out


def rank_precision(gt: np.ndarray, pred: np.ndarray) -> list[float]:
    g, p = band(gt, HANDBOOK_EDGES), band(pred, HANDBOOK_EDGES)
    out = []
    for i in range(len(HANDBOOK_LABELS)):
        col = int((p == i).sum())
        out.append(float(((g == i) & (p == i)).sum() / col) if col else float("nan"))
    return out


def distribution(values: np.ndarray) -> list[float]:
    b = band(values, HANDBOOK_EDGES)
    return [float((b == i).mean()) for i in range(len(HANDBOOK_LABELS))]


def decomposition(e_seg, e_bin, e_pred) -> dict:
    e_int = e_pred - e_seg - e_bin
    v_s, v_b = float(np.var(e_seg, ddof=1)), float(np.var(e_bin, ddof=1))
    cov2 = 2 * float(np.cov(e_seg, e_bin)[0, 1])
    tot = v_s + v_b + cov2
    return {
        "bias_seg": float(e_seg.mean()), "bias_bin": float(e_bin.mean()),
        "bias_pred": float(e_pred.mean()), "interaction": float(e_int.mean()),
        "MAE_seg": float(np.abs(e_seg).mean()), "MAE_bin": float(np.abs(e_bin).mean()),
        "MAE_pred": float(np.abs(e_pred).mean()),
        "var_seg": v_s, "var_bin": v_b, "cov_x2": cov2,
        "share_seg_pct": 100 * v_s / tot, "share_bin_pct": 100 * v_b / tot,
        "share_cov_pct": 100 * cov2 / tot,
    }


def cluster_bootstrap(parents: np.ndarray, arrays: tuple, fn, n_boot: int,
                      seed: int) -> dict[str, tuple[float, float]]:
    """Percentile CIs for every key returned by fn, resampling parent images."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(parents)
    idx = [np.flatnonzero(parents == p) for p in uniq]
    draws: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        sel = np.concatenate([idx[i] for i in pick])
        for k, v in fn(*(a[sel] for a in arrays)).items():
            draws.setdefault(k, []).append(v)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            for k, v in draws.items()}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m-gt", required=True, type=Path, help="M_GT folder (A1)")
    ap.add_argument("--r-gt", required=True, type=Path, help="R_GT folder (A2)")
    ap.add_argument("--m-pred", required=True, type=Path, help="M_Pred folder (A3)")
    ap.add_argument("--r-pred", required=True, type=Path, help="R_Pred folder (A4)")
    ap.add_argument("--window-csv", type=Path, default=None,
                    help="window_level_validation.csv from stage 04 (optional)")
    ap.add_argument("--stage2-objects", type=Path, default=None,
                    help="all_output_info.csv from stage 02 (optional, shadow flags)")
    ap.add_argument("--site-field", type=int, default=None,
                    help="file-name field identifying the site (optional)")
    ap.add_argument("--bootstrap", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, type=Path, help="output folder")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in args.m_gt.glob("*.png"))
    if not stems:
        raise SystemExit(f"no PNG masks found in {args.m_gt}")

    # ---- 1. build window / object / image tables ------------------------- #
    win, obj = [], []
    for stem in stems:
        paths = [d / f"{stem}.png" for d in (args.m_gt, args.r_gt, args.m_pred, args.r_pred)]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            print(f"  skipped {stem}: missing {missing}")
            continue
        m_gt, r_gt, m_pr, r_pr = (load_mask(p) for p in paths)
        if not (m_gt.shape == r_gt.shape == m_pr.shape == r_pr.shape):
            print(f"  skipped {stem}: mask shapes differ")
            continue
        r_gt = r_gt & m_gt                       # the reference lies inside M_GT
        obj += objects_from_masks(stem, m_gt, r_gt, m_pr, r_pr)
        if args.window_csv is None:
            win += windows_from_masks(stem, m_gt, r_gt, m_pr, r_pr)
    if args.window_csv is not None:
        win = [r for r in windows_from_csv(args.window_csv) if r["parent"] in set(stems)]
    img = images_from_windows(win)

    site_map: dict[str, str] = {}
    if args.site_field is not None:
        raw = sorted({site_of(s, args.site_field) for s in stems})
        site_map = {k: f"Site {chr(ord('A') + i)}" for i, k in enumerate(raw)}
    for table in (win, obj, img):
        for r in table:
            r["site"] = site_map.get(site_of(r["parent"], args.site_field), "") \
                if site_map else ""

    def col(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    w_gt, w_pr, w_wt = col(win, "D_GT"), col(win, "D_Pred"), col(win, "weight")
    w_par = np.array([r["parent"] for r in win])
    measured = [r for r in obj if not math.isnan(r["D_Pred"])]
    o_gt, o_pr = col(measured, "D_GT"), col(measured, "D_Pred")
    i_gt, i_pr = col(img, "D_GT"), col(img, "D_Pred")

    print("=" * 78)
    print(f"parents {len(img)}   windows {len(win)}   markings {len(obj)} "
          f"({len(obj) - len(measured)} missed entirely by segmentation)")

    # ---- 2. error decomposition (window support) ------------------------- #
    seg_ok = ~(np.isnan(col(win, "D_Seg")) | np.isnan(col(win, "D_Bin")))
    e_seg = (col(win, "D_Seg") - w_gt)[seg_ok]
    e_bin = (col(win, "D_Bin") - w_gt)[seg_ok]
    e_pred = (w_pr - w_gt)[seg_ok]
    dec = decomposition(e_seg, e_bin, e_pred)
    keys_ci = ("bias_seg", "bias_bin", "interaction",
               "share_seg_pct", "share_bin_pct", "share_cov_pct")
    ci = cluster_bootstrap(w_par[seg_ok], (e_seg, e_bin, e_pred),
                           lambda s, b, p: {k: decomposition(s, b, p)[k] for k in keys_ci},
                           args.bootstrap, args.seed)
    print("\nERROR DECOMPOSITION  e = e_seg + e_bin + e_int   (window support, "
          f"n = {int(seg_ok.sum())})")
    for k in ("bias_seg", "bias_bin", "bias_pred", "interaction",
              "MAE_seg", "MAE_bin", "MAE_pred"):
        extra = f"   95% CI [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]" if k in ci else ""
        print(f"  {k:<14}{dec[k]:+.4f}{extra}")
    for k in ("share_seg_pct", "share_bin_pct", "share_cov_pct"):
        print(f"  {k:<14}{dec[k]:6.1f}%   95% CI [{ci[k][0]:+.1f}, {ci[k][1]:+.1f}]")
    write_csv(args.out / "error_decomposition.csv",
              [{"quantity": k, "estimate": v,
                "ci_lo": ci.get(k, (float("nan"),) * 2)[0],
                "ci_hi": ci.get(k, (float("nan"),) * 2)[1]} for k, v in dec.items()])

    # ---- 3. error by scene condition ------------------------------------- #
    if args.stage2_objects is not None:
        boxes: dict[str, list[tuple[int, int, int, int]]] = {}
        with args.stage2_objects.open(encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                if r.get("shadow_crossing_detected") == "1":
                    x, y = int(_num(r["bbox_x"])), int(_num(r["bbox_y"]))
                    bw, bh = int(_num(r["bbox_w"])), int(_num(r["bbox_h"]))
                    boxes.setdefault(r["parent"], []).append((x, y, x + bw, y + bh))
        sub = [r for r, ok in zip(win, seg_ok) if ok]
        shadow = np.array([
            any(not (bx1 <= r["x"] or bx0 >= r["x"] + r["w"] or
                     by1 <= r["y"] or by0 >= r["y"] + r["h"])
                for bx0, by0, bx1, by1 in boxes.get(r["parent"], []))
            for r in sub])
        severe = np.array([r["D_GT"] > SEVERE_WEAR for r in sub])
        gt_sub = np.array([r["D_GT"] for r in sub])
        cases = []
        print("\nERROR BY SCENE CONDITION")
        print(f"  {'condition':<14}{'n':>6}{'mean D_GT':>11}{'MAE_seg':>9}"
              f"{'MAE_bin':>9}{'seg share':>11}")
        for name, sel in (("Shadow", shadow), ("Severe wear", severe),
                          ("Both", shadow & severe), ("Neither", ~shadow & ~severe),
                          ("All windows", np.ones_like(shadow))):
            if sel.sum() < 3:
                continue
            d = decomposition(e_seg[sel], e_bin[sel], e_pred[sel])
            cases.append({"condition": name, "n": int(sel.sum()),
                          "mean_D_GT": float(gt_sub[sel].mean()),
                          "MAE_seg": d["MAE_seg"], "MAE_bin": d["MAE_bin"],
                          "share_seg_pct": d["share_seg_pct"]})
            print(f"  {name:<14}{sel.sum():>6}{gt_sub[sel].mean():>11.3f}"
                  f"{d['MAE_seg']:>9.4f}{d['MAE_bin']:>9.4f}{d['share_seg_pct']:>10.1f}%")
        write_csv(args.out / "error_by_condition.csv", cases)

    # ---- 4. uncertainty at three supports -------------------------------- #
    met = [metrology(w_gt, w_pr, "window", w_wt),
           metrology(o_gt, o_pr, "marking"),
           metrology(i_gt, i_pr, "image")]
    print("\nMETROLOGICAL UNCERTAINTY  (u = SD of the error, U = 2u, k = 2)")
    print(f"  {'support':<9}{'n':>6}{'bias':>9}{'MAE':>8}{'RMSE':>8}{'U':>8}"
          f"{'MRE':>8}   limits of agreement")
    for m in met:
        print(f"  {m['support']:<9}{m['n']:>6}{m['bias']:>+9.4f}{m['MAE']:>8.4f}"
              f"{m['RMSE']:>8.4f}{m['U_k2']:>8.3f}{100 * m['MRE']:>7.1f}%"
              f"   [{m['LoA_lower']:+.3f}, {m['LoA_upper']:+.3f}]")
    print(f"  window aw-MAE {met[0]['aw_MAE']:.4f}")
    write_csv(args.out / "uncertainty.csv", met)

    # ---- 5. handbook distribution and band resolvability ----------------- #
    print("\nREFERENCE DISTRIBUTION OVER THE HANDBOOK RANKS")
    dist_rows = []
    for name, vals in (("window", w_gt), ("marking", col(obj, "D_GT")), ("image", i_gt)):
        d = distribution(vals)
        dist_rows.append({"support": name, "n": int(vals.size),
                          **{lab: v for lab, v in zip(HANDBOOK_LABELS, d)}})
        print(f"  {name:<8}(n={vals.size:>5})  " +
              "  ".join(f"{lab}: {100 * v:5.1f}%" for lab, v in zip(HANDBOOK_LABELS, d)))
    write_csv(args.out / "handbook_distribution.csv", dist_rows)

    U = met[0]["U_k2"]
    widths = ([HANDBOOK_EDGES[0]] + list(np.diff(HANDBOOK_EDGES)) + [1 - HANDBOOK_EDGES[-1]])
    g_widths = [GRADE_EDGES[0], GRADE_EDGES[1] - GRADE_EDGES[0], 1 - GRADE_EDGES[1]]
    rp = rank_precision(w_gt, w_pr)
    gw = grading(w_gt, w_pr, "window")
    print(f"\nBAND WIDTH AGAINST RESOLUTION  (window U = {U:.3f}, criterion: width > 2U)")
    res_rows = []
    for lab, wd, pr in zip(HANDBOOK_LABELS, widths, rp):
        res_rows.append({"band": lab, "width": wd, "width_over_2U": wd / (2 * U),
                         "precision_window": pr})
    for lab, wd in zip(GRADE_LABELS, g_widths):
        res_rows.append({"band": lab, "width": wd, "width_over_2U": wd / (2 * U),
                         "precision_window": gw[f"precision_{lab}"]})
    for r in res_rows:
        print(f"  {r['band']:<10}width {r['width']:.2f}   width/2U {r['width_over_2U']:.2f}"
              f"   precision {r['precision_window']:.3f}")
    write_csv(args.out / "band_resolution.csv", res_rows)

    # ---- 6. grading accuracy --------------------------------------------- #
    grades = [gw, grading(o_gt, o_pr, "marking"), grading(i_gt, i_pr, "image")]
    print("\nTHREE-GRADE SCALE  I: D < 0.23   II: 0.23-0.40   III: D >= 0.40")
    for g in grades:
        print(f"  {g['support']:<8}n={g['n']:>5}   grade {g['grade_accuracy']:.3f} "
              f"[{g['grade_ci_lo']:.3f}, {g['grade_ci_hi']:.3f}]   within one "
              f"{g['within_one_grade']:.3f}   repaint {g['repaint_accuracy']:.3f}")
    print("  window confusion (rows = reference, columns = measured):")
    for lab, row in zip(GRADE_LABELS, gw["_confusion"]):
        print(f"    {lab:<10}" + "".join(f"{v:>7d}" for v in row))
    write_csv(args.out / "grading_accuracy.csv",
              [{k: v for k, v in g.items() if not k.startswith("_")} for g in grades])

    # ---- 7. per-site table ----------------------------------------------- #
    if site_map:
        rows = []
        for s in sorted(site_map.values()):
            sel = np.array([r["site"] == s for r in win])
            if not sel.any():
                continue
            g, p, wt = w_gt[sel], w_pr[sel], w_wt[sel]
            rows.append({"site": s, "windows": int(sel.sum()), "mean_D_GT": float(g.mean()),
                         "grade_accuracy": float((band(g, GRADE_EDGES) == band(p, GRADE_EDGES)).mean()),
                         "repaint_accuracy": float(((g >= REPAINT_EDGE) == (p >= REPAINT_EDGE)).mean()),
                         "aw_MAE": float((wt * np.abs(p - g)).sum() / wt.sum())})
        write_csv(args.out / "per_site.csv", rows)

    # ---- 8. per-unit tables and summary ---------------------------------- #
    for rows, name in ((win, "windows"), (obj, "markings"), (img, "images")):
        for r in rows:
            r["grade_GT"] = GRADE_LABELS[int(band(np.array([r["D_GT"]]), GRADE_EDGES)[0])]
            r["grade_Pred"] = ("" if math.isnan(r["D_Pred"]) else
                               GRADE_LABELS[int(band(np.array([r["D_Pred"]]), GRADE_EDGES)[0])])
        write_csv(args.out / f"{name}.csv", rows)

    summary = {
        "n": {"parents": len(img), "windows": len(win), "markings": len(obj),
              "markings_measured": len(measured)},
        "decomposition": dec, "decomposition_ci95": ci,
        "uncertainty": {m["support"]: m for m in met},
        "grading": {g["support"]: {k: v for k, v in g.items() if not k.startswith("_")}
                    for g in grades},
        "scale": {"handbook_edges": HANDBOOK_EDGES, "grade_edges": GRADE_EDGES,
                  "grade_actions": dict(zip(GRADE_LABELS, GRADE_ACTIONS))},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nTables written to {args.out}")


if __name__ == "__main__":
    main()
