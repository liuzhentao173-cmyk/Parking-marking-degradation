# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

"""
Batch orchestrator for the main pipeline scripts 02 / 03 / 04.

For every (model, backbone) pair under
    <MODELS_ROOT>/{Model}/{Backbone}/Pred_Mask
this orchestrator runs, sequentially:
    02_binarize_residual_paint.py
    03_degradation_index_heatmap.py
    04_downstream_gt_validation.py

Image/GT folders are fixed by user spec. Pred_Mask lives in the per-backbone
subfolder and follows the existing naming convention. All outputs land next to
Pred_Mask so the existing layout is preserved.

Usage (PowerShell):
    python run_all_models.py
    python run_all_models.py --dry-run
    python run_all_models.py --models Segformer --backbones MitB0 MitB1
    python run_all_models.py --skip-02
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


def log(msg: str) -> None:
    """Print without breaking an active tqdm bar."""
    tqdm.write(msg)

# ============================================================
# Fixed inputs (per user spec)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_02 = SCRIPT_DIR.parent / "02_binarize_residual_paint.py"
SCRIPT_03 = SCRIPT_DIR.parent / "03_degradation_index_heatmap.py"
SCRIPT_04 = SCRIPT_DIR.parent / "04_downstream_gt_validation.py"

IMAGE_DIR   = Path(r"")  # TODO: set path
GT_MASK_DIR = Path(r"")     # M_GT
GT_BIN_DIR  = Path(r"")   # R_GT
MODELS_ROOT = Path(r"")  # TODO: set path

# Default model -> backbone mapping. Override at CLI if needed.
DEFAULT_TARGETS: Dict[str, List[str]] = {
    "FPN":           ["R34", "R50", "R101"],
    "UNetPlusPlus":  ["R34", "R50", "R101"],
    "DeepLabV3Plus": ["R34", "R50", "R101"],
    "Segformer":     ["MitB0", "MitB1", "MitB2", "MitB3", "MitB4", "MitB5"],
}

# Subfolder names placed under <Model>/<Backbone>/ for outputs of each stage.
OUT_02 = "02_wkdegmm_output"
OUT_03 = "03_blur_window_output"
OUT_04 = "04_validation_output"
# Stable subfolder name for script-02's run output. The script itself emits a
# timestamped folder; we rename it to this fixed name after each successful run,
# so that 03/04 always know where the R_Pred mask lives across re-runs.
STABLE_RUN_NAME = "wk_run"

# Preferred Python interpreters for subprocess execution, in priority order.
# The first one that exists is used as the default for --python. This means you
# can just hit ▶ / F5 in VSCode regardless of which interpreter VSCode picked,
# and the orchestrator will still dispatch 02/03/04 to an env that has cv2.
# Edit this list (or pass --python) if your conda lives elsewhere.
PREFERRED_PYTHONS: List[str] = [
    # TODO: optionally add absolute python interpreter paths here.
    # If left empty, the current interpreter (sys.executable) is used.
]


def auto_pick_python() -> str:
    for candidate in PREFERRED_PYTHONS:
        if Path(candidate).exists():
            return candidate
    return sys.executable


# ============================================================
# Helpers
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-run 02/03/04 for every model/backbone Pred_Mask folder.")
    p.add_argument("--models",     nargs="*", default=None, help="Subset of models to run (default: all).")
    p.add_argument("--backbones",  nargs="*", default=None, help="Subset of backbones to run (default: all per model).")
    p.add_argument("--models-root", type=str, default=str(MODELS_ROOT))
    p.add_argument("--image-dir",  type=str, default=str(IMAGE_DIR))
    p.add_argument("--gt-mask-dir", type=str, default=str(GT_MASK_DIR))
    p.add_argument("--gt-bin-dir", type=str, default=str(GT_BIN_DIR))
    p.add_argument("--skip-02",    action="store_true", help="Skip script 02 (assumes A4 already exists from a previous run).")
    p.add_argument("--skip-03",    action="store_true")
    p.add_argument("--skip-04",    action="store_true")
    p.add_argument("--continue-on-error", action="store_true", default=True, help="Keep going if one pair fails (default: True).")
    p.add_argument("--stop-on-error", action="store_true", help="Abort the whole batch on the first failure.")
    p.add_argument("--dry-run",    action="store_true", help="Print commands without executing.")
    p.add_argument("--apply-pred-opening", action="store_true", help="Forward --apply-pred-opening to script 02.")
    p.add_argument("--resize-masks-to-image", action="store_true", help="Forward to scripts 02/03/04.")
    p.add_argument("--python", type=str, default=auto_pick_python(),
                   help="Python interpreter used to launch 02/03/04. Default: first existing in PREFERRED_PYTHONS, "
                        "else current sys.executable. Use this to point at a conda env that has cv2/numpy/pandas/scipy/PIL installed.")
    p.add_argument("--skip-env-check", action="store_true",
                   help="Skip the pre-flight 'import cv2 numpy pandas scipy PIL tqdm' probe of --python.")
    return p.parse_args()


def probe_python_env(python_exe: str) -> Tuple[bool, str]:
    """Quickly verify the target Python can import every module 02/03/04 need.

    Returns (ok, message). Failing fast here saves the user from watching 15
    subprocesses all crash on the same ImportError.
    """
    probe = "import cv2, numpy, pandas, scipy, PIL, tqdm; import scipy.signal; print('ok')"
    try:
        proc = subprocess.run([python_exe, "-c", probe],
                              capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return False, f"interpreter not found: {python_exe}"
    except subprocess.TimeoutExpired:
        return False, "env probe timed out (>30s)"
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, proc.stdout.strip()


def discover_targets(models_root: Path, requested_models: Optional[List[str]], requested_backbones: Optional[List[str]]) -> List[Tuple[str, str, Path]]:
    """Resolve (model, backbone, pred_mask_dir) tuples to run.

    Falls back to DEFAULT_TARGETS for models the user didn't restrict, and verifies
    each Pred_Mask folder exists on disk.
    """
    targets: List[Tuple[str, str, Path]] = []
    model_list = requested_models if requested_models else list(DEFAULT_TARGETS.keys())
    for model in model_list:
        default_bbs = DEFAULT_TARGETS.get(model)
        if default_bbs is None:
            # Unknown model name: discover backbones from disk.
            model_dir = models_root / model
            if not model_dir.exists():
                print(f"[WARN] model folder missing, skipped: {model_dir}")
                continue
            default_bbs = [p.name for p in sorted(model_dir.iterdir()) if p.is_dir()]
        backbones = requested_backbones if requested_backbones else default_bbs
        for bb in backbones:
            pred_dir = models_root / model / bb / "Pred_Mask"
            if not pred_dir.exists():
                print(f"[WARN] Pred_Mask not found, skipped: {pred_dir}")
                continue
            targets.append((model, bb, pred_dir))
    return targets


def find_latest_02_run(output_root: Path) -> Optional[Path]:
    """Pick the freshest 'wk_*' run folder under script 02's output_root.

    Prefers the stable name (``wk_run``) if it exists, so post-renamed runs are
    found without scanning by mtime.
    """
    if not output_root.exists():
        return None
    stable = output_root / STABLE_RUN_NAME
    if stable.exists() and stable.is_dir():
        return stable
    cands = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("wk_")]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def promote_to_stable(output_root: Path) -> Optional[Path]:
    """Rename the freshest timestamped 'wk_*_<ts>' folder to STABLE_RUN_NAME.

    If a previous stable folder exists, it is removed first so we always end up
    with a single deterministic path. No-op if the latest folder is already the
    stable name.
    """
    if not output_root.exists():
        return None
    stable = output_root / STABLE_RUN_NAME
    # Find the freshest timestamped run that is NOT the stable folder itself.
    cands = [p for p in output_root.iterdir()
             if p.is_dir() and p.name.startswith("wk_") and p.name != STABLE_RUN_NAME]
    if not cands:
        return stable if stable.exists() else None
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    if stable.exists():
        shutil.rmtree(stable, ignore_errors=True)
    newest.rename(stable)
    return stable


def run_cmd(cmd: List[str], dry_run: bool, log_path: Path) -> int:
    """Run a command, mirroring stdout/stderr to a log file. Return exit code."""
    if dry_run:
        log(" $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as lf:
        lf.write(f"\n# === {datetime.now().isoformat()} ===\n".encode("utf-8"))
        lf.write(("$ " + " ".join(cmd) + "\n").encode("utf-8"))
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return int(proc.returncode)


# ============================================================
# Per-stage runners
# ============================================================
def run_stage_02(model: str, backbone: str, pred_dir: Path, image_dir: Path,
                 apply_opening: bool, resize: bool, dry_run: bool, batch_log_dir: Path,
                 python_exe: str) -> Optional[Path]:
    """Run script 02; return path to the R_Pred mask folder (03_final_mask) on success."""
    out_root = pred_dir.parent / OUT_02
    run_name = f"{model}_{backbone}_wkdegmm"
    cmd = [
        python_exe, str(SCRIPT_02),
        "--image-dir", str(image_dir),
        "--pred-mask-dir", str(pred_dir),
        "--output-root", str(out_root),
        "--run-name", run_name,
    ]
    if apply_opening:
        cmd.append("--apply-pred-opening")
    if resize:
        cmd.append("--resize-masks-to-image")

    log_path = batch_log_dir / f"{model}_{backbone}_02.log"
    rc = run_cmd(cmd, dry_run, log_path)
    if rc != 0:
        log(f"[ERROR] script 02 failed for {model}/{backbone} (rc={rc}); see {log_path}")
        return None

    if dry_run:
        return out_root / STABLE_RUN_NAME / "03_final_mask"

    # Rename the timestamped wk_*_<ts> -> wk_run so paths stay deterministic.
    stable = promote_to_stable(out_root)
    if stable is None:
        log(f"[ERROR] script 02 produced no wk_* run dir under {out_root}")
        return None
    a4_dir = stable / "03_final_mask"
    if not a4_dir.exists():
        log(f"[ERROR] script 02 run dir is missing 03_final_mask: {stable}")
        return None
    return a4_dir


def run_stage_03(model: str, backbone: str, pred_dir: Path, a4_dir: Path, image_dir: Path,
                 resize: bool, dry_run: bool, batch_log_dir: Path, python_exe: str) -> bool:
    out_dir = pred_dir.parent / OUT_03
    cmd = [
        python_exe, str(SCRIPT_03),
        "--image-dir", str(image_dir),
        "--pred-mask-dir", str(pred_dir),
        "--binary-mask-dir", str(a4_dir),
        "--output-dir", str(out_dir),
    ]
    if resize:
        cmd.append("--resize-masks-to-image")
    log_path = batch_log_dir / f"{model}_{backbone}_03.log"
    rc = run_cmd(cmd, dry_run, log_path)
    if rc != 0:
        log(f"[ERROR] script 03 failed for {model}/{backbone} (rc={rc}); see {log_path}")
        return False
    return True


def run_stage_04(model: str, backbone: str, pred_dir: Path, a4_dir: Path,
                 image_dir: Path, gt_mask_dir: Path, gt_bin_dir: Path,
                 resize: bool, dry_run: bool, batch_log_dir: Path, python_exe: str) -> bool:
    out_root = pred_dir.parent / OUT_04
    cmd = [
        python_exe, str(SCRIPT_04),
        "--image-dir", str(image_dir),
        "--gt-orig-dir", str(gt_mask_dir),
        "--gt-bin-dir", str(gt_bin_dir),
        "--pred-orig-dir", str(pred_dir),
        "--pred-bin-dir", str(a4_dir),
        "--output-root", str(out_root),
    ]
    if resize:
        cmd.append("--resize-masks-to-image")
    log_path = batch_log_dir / f"{model}_{backbone}_04.log"
    rc = run_cmd(cmd, dry_run, log_path)
    if rc != 0:
        log(f"[ERROR] script 04 failed for {model}/{backbone} (rc={rc}); see {log_path}")
        return False
    return True


def find_existing_a4(pred_dir: Path) -> Optional[Path]:
    """When --skip-02 is set, locate the existing R_Pred folder (prefers wk_run)."""
    out_root = pred_dir.parent / OUT_02
    latest = find_latest_02_run(out_root)
    if latest is None:
        return None
    a4 = latest / "03_final_mask"
    return a4 if a4.exists() else None


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()
    models_root = Path(args.models_root)
    image_dir   = Path(args.image_dir)
    gt_mask_dir = Path(args.gt_mask_dir)
    gt_bin_dir  = Path(args.gt_bin_dir)

    # Sanity checks
    for label, p in [("image_dir", image_dir), ("gt_mask_dir", gt_mask_dir),
                     ("gt_bin_dir", gt_bin_dir), ("models_root", models_root)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    for label, p in [("SCRIPT_02", SCRIPT_02), ("SCRIPT_03", SCRIPT_03), ("SCRIPT_04", SCRIPT_04)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    targets = discover_targets(models_root, args.models, args.backbones)
    if not targets:
        print("[FATAL] No (model, backbone) targets resolved.")
        sys.exit(2)

    python_exe = args.python
    if not args.skip_env_check and not args.dry_run:
        ok, msg = probe_python_env(python_exe)
        if not ok:
            print("=" * 100)
            print(f"[FATAL] Python interpreter '{python_exe}' is missing dependencies for 02/03/04.")
            print(f"        Probe output:\n{msg}")
            print("        Fix options:")
            print("          (a) activate the conda env you normally use, then re-run.")
            print("          (b) pass --python C:\\path\\to\\python.exe pointing at an env that has cv2/numpy/pandas/scipy/PIL/tqdm.")
            print("          (c) pass --skip-env-check to bypass this probe (not recommended).")
            print("=" * 100)
            sys.exit(3)
        print(f"[INFO] Env probe OK for: {python_exe}")

    # Single, deterministic batch log folder (no timestamp).
    # Logs are overwritten per (model, backbone) on each run; the previous
    # contents stay if you didn't re-run that pair.
    batch_log_dir = SCRIPT_DIR / "batch_logs"
    batch_log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Batch log dir : {batch_log_dir}")
    print(f"[INFO] Python        : {python_exe}")
    print(f"[INFO] Targets ({len(targets)}):")
    for m, bb, pd_ in targets:
        print(f"         - {m}/{bb}  ({pd_})")

    summary: List[Dict[str, str]] = []
    t0_total = time.time()

    # Stage-level granularity: 1 unit = 1 stage. Total = (stages not skipped) * N.
    active_stages = [s for s, skip in [("02", args.skip_02), ("03", args.skip_03), ("04", args.skip_04)] if not skip]
    total_units = max(1, len(active_stages) * len(targets))

    pbar = tqdm(total=total_units, desc="Pipeline", unit="stage", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]")

    for (model, backbone, pred_dir) in targets:
        tag = f"{model}/{backbone}"
        pbar.set_description(tag)
        t0 = time.time()
        status = {"model": model, "backbone": backbone, "pred_dir": str(pred_dir),
                  "stage02": "skipped", "stage03": "skipped", "stage04": "skipped",
                  "elapsed_sec": "0", "a4_dir": ""}

        # Stage 02
        a4_dir: Optional[Path] = None
        if args.skip_02:
            a4_dir = find_existing_a4(pred_dir)
            status["stage02"] = "skipped_existing" if a4_dir else "skipped_missing"
        else:
            pbar.set_postfix_str(f"{tag} 02 wkdegmm")
            t_stage = time.time()
            a4_dir = run_stage_02(model, backbone, pred_dir, image_dir,
                                  args.apply_pred_opening, args.resize_masks_to_image,
                                  args.dry_run, batch_log_dir, python_exe)
            status["stage02"] = "ok" if a4_dir else "fail"
            log(f"[{tag}] 02 {status['stage02']:>4}  {time.time() - t_stage:6.1f}s")
            pbar.update(1)

        if a4_dir is None and not args.dry_run:
            status["elapsed_sec"] = f"{time.time() - t0:.1f}"
            summary.append(status)
            # Bump the bar past the skipped stages so it still finishes at 100%.
            remaining_for_this_bb = sum(1 for s in active_stages if s != "02")
            if remaining_for_this_bb:
                pbar.update(remaining_for_this_bb)
            log(f"[SKIP 03/04] no A4 mask for {tag}")
            if args.stop_on_error:
                break
            continue
        status["a4_dir"] = str(a4_dir) if a4_dir else ""

        # Stage 03
        if not args.skip_03:
            pbar.set_postfix_str(f"{tag} 03 blur-window")
            t_stage = time.time()
            ok3 = run_stage_03(model, backbone, pred_dir, a4_dir, image_dir,
                               args.resize_masks_to_image, args.dry_run, batch_log_dir, python_exe)
            status["stage03"] = "ok" if ok3 else "fail"
            log(f"[{tag}] 03 {status['stage03']:>4}  {time.time() - t_stage:6.1f}s")
            pbar.update(1)
            if not ok3 and args.stop_on_error:
                summary.append(status)
                break

        # Stage 04
        if not args.skip_04:
            pbar.set_postfix_str(f"{tag} 04 validation")
            t_stage = time.time()
            ok4 = run_stage_04(model, backbone, pred_dir, a4_dir, image_dir,
                               gt_mask_dir, gt_bin_dir, args.resize_masks_to_image,
                               args.dry_run, batch_log_dir, python_exe)
            status["stage04"] = "ok" if ok4 else "fail"
            log(f"[{tag}] 04 {status['stage04']:>4}  {time.time() - t_stage:6.1f}s")
            pbar.update(1)
            if not ok4 and args.stop_on_error:
                summary.append(status)
                break

        status["elapsed_sec"] = f"{time.time() - t0:.1f}"
        summary.append(status)

        results_short = " ".join(
            f"{k}={v}" for k, v in [("02", status["stage02"]), ("03", status["stage03"]), ("04", status["stage04"])]
        )
        log(f"[done] {tag:<28} {results_short:<40} {status['elapsed_sec']}s")
    pbar.close()

    # Final report
    print("\n" + "#" * 100)
    print(f"BATCH SUMMARY (total {time.time() - t0_total:.1f}s)")
    print("#" * 100)
    hdr = f"{'model':14}{'backbone':10}{'02':18}{'03':6}{'04':6}{'sec':>8}"
    print(hdr)
    print("-" * len(hdr))
    ok_cnt = 0
    for s in summary:
        line = f"{s['model']:14}{s['backbone']:10}{s['stage02']:18}{s['stage03']:6}{s['stage04']:6}{s['elapsed_sec']:>8}"
        print(line)
        if s["stage02"] in ("ok", "skipped_existing") and s["stage03"] in ("ok", "skipped") and s["stage04"] in ("ok", "skipped"):
            ok_cnt += 1
    print(f"\nSucceeded: {ok_cnt}/{len(summary)}")
    print(f"Logs:      {batch_log_dir}")

    # Write a CSV summary (deterministic name, no timestamp).
    csv_path = batch_log_dir / "batch_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,backbone,pred_dir,stage02,stage03,stage04,a4_dir,elapsed_sec\n")
        for s in summary:
            f.write(",".join([s["model"], s["backbone"], s["pred_dir"],
                              s["stage02"], s["stage03"], s["stage04"],
                              s["a4_dir"], s["elapsed_sec"]]) + "\n")
    print(f"Summary CSV: {csv_path}")


if __name__ == "__main__":
    main()
