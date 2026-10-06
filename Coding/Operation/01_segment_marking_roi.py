# Mask notation (paper): M_GT = GT marking region, R_GT = GT residual paint,
# M_Pred = predicted marking region, R_Pred = predicted residual paint.
# Note: variable names and CSV column tags below still use A1/A2/A3/A4
# (A1=M_GT, A2=R_GT, A3=M_Pred, A4=R_Pred); CLI --help text likewise.
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ============================================================
# 01. SegFormer tiled inference -> full-size M_Pred Pred_Mask
# ------------------------------------------------------------
# Output convention:
#   Image/         stitched full-size RGB image, parent.png
#   Pred_Mask/     M_Pred = inference-original / predicted ROI, parent.png
#   Pred_Prob_NPY/ stitched probability map, parent.npy
#   Pred_Prob_PNG/ stitched probability map, parent.png 16-bit
#   GT_Mask/       M_GT = GT-original stitched mask, parent.png
#   Overlay/       RGB + Pred_Mask overlay
#   metadata/      CSV reports and run_config.json
# ============================================================

# -------------------------
# Defaults: edit here or use CLI
# -------------------------
DATASET_ROOT = Path(r"")  # TODO: set path
WEIGHT_PATH = Path(r"")  # TODO: set path
OUTPUT_ROOT = Path(r"")  # TODO: set path
HF_NAME = "nvidia/segformer-b5-finetuned-ade-640-640"
EVAL_SPLIT = "test"  # train / val / test
RUN_NAME = "SegFormer_MiT-B5_test"   

PROB_THRESHOLD = 0.50
PRED_OPEN_KSIZE = 3       # set <=1 to disable opening
NUM_RANDOM_PARENTS = -1   # <=0 means all parents
RANDOM_SEED = 42
CLEAR_OUTPUT_DIRS = False
SAVE_PROB_NPY = True
SAVE_PROB_PNG = True

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SegFormer tiled inference and stitching to full-size predicted masks")
    parser.add_argument("--dataset-root", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--weight-path", type=str, default=str(WEIGHT_PATH))
    parser.add_argument("--output-root", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--hf-name", type=str, default=HF_NAME)
    parser.add_argument("--split", type=str, default=EVAL_SPLIT, choices=["train", "val", "test"])
    parser.add_argument("--run-name", type=str, default=RUN_NAME)
    parser.add_argument("--prob-threshold", type=float, default=PROB_THRESHOLD)
    parser.add_argument("--pred-open-ksize", type=int, default=PRED_OPEN_KSIZE)
    parser.add_argument("--num-random-parents", type=int, default=NUM_RANDOM_PARENTS)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--clear-output-dirs", action="store_true", default=CLEAR_OUTPUT_DIRS)
    parser.add_argument("--no-prob-npy", action="store_true", help="Do not save stitched probability .npy files")
    parser.add_argument("--no-prob-png", action="store_true", help="Do not save stitched probability 16-bit png files")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS])


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def read_mask01(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return (arr > 127).astype(np.uint8)


def save_rgb(rgb: np.ndarray, out_path: Path) -> None:
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(out_path)


def save_binary_mask(mask01: np.ndarray, out_path: Path) -> None:
    Image.fromarray((mask01.astype(np.uint8) * 255), mode="L").save(out_path)


def save_prob_png(prob: np.ndarray, out_path: Path) -> None:
    prob_u16 = np.clip(prob, 0, 1)
    prob_u16 = (prob_u16 * 65535.0 + 0.5).astype(np.uint16)
    Image.fromarray(prob_u16, mode="I;16").save(out_path)


def find_matching_mask(img_path: Path, mask_dir: Path) -> Optional[Path]:
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        cand = mask_dir / f"{img_path.stem}{ext}"
        if cand.exists():
            return cand
    return None


def parse_tile_info(tile_path: Path) -> Dict[str, int | str]:
    """Parse parent name and tile coordinates from a tiled filename.

    Supported examples:
      SiteA_0002_0003__x000000_y000448.jpg
      parent_x000000_y000448.png
      parent-left000000-top000448.png
    """
    stem = tile_path.stem
    patterns = [
        r"^(?P<parent>.+?)__x(?P<x>\d+)_y(?P<y>\d+)$",
        r"^(?P<parent>.+?)[_-]x(?P<x>\d+)[_-]y(?P<y>\d+)$",
        r"^(?P<parent>.+?)[_-]left(?P<x>\d+)[_-]top(?P<y>\d+)$",
    ]
    for pat in patterns:
        m = re.match(pat, stem, flags=re.IGNORECASE)
        if m:
            d = m.groupdict()
            return {"parent": d["parent"], "x": int(d["x"]), "y": int(d["y"])}
    raise ValueError(f"Cannot parse tile coordinates from filename: {tile_path.name}")


def make_overlay(rgb: np.ndarray, mask01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    vis = rgb.copy().astype(np.float32)
    blue = np.zeros_like(vis)
    blue[..., 2] = 255
    keep = mask01.astype(bool)
    vis[keep] = vis[keep] * (1.0 - alpha) + blue[keep] * alpha
    return np.clip(vis, 0, 255).astype(np.uint8)


def strip_prefix_in_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned


def _looks_like_state_dict(d: dict) -> bool:
    if not isinstance(d, dict) or not d:
        return False
    n_tensor = sum(torch.is_tensor(v) for v in d.values())
    return n_tensor >= max(1, int(0.5 * len(d)))


def _find_state_dict_recursively(obj, path="root"):
    if isinstance(obj, torch.nn.Module):
        return obj.state_dict(), path + ".state_dict()"
    if isinstance(obj, dict):
        priority_keys = [
            "state_dict", "model_state_dict", "model_state", "model_weights",
            "network_state_dict", "net_state_dict", "model", "net", "weights", "ema_state_dict",
        ]
        for k in priority_keys:
            if k in obj:
                cand = obj[k]
                if isinstance(cand, torch.nn.Module):
                    return cand.state_dict(), f"{path}.{k}.state_dict()"
                if _looks_like_state_dict(cand):
                    return cand, f"{path}.{k}"
        if _looks_like_state_dict(obj):
            return obj, path
        for k, v in obj.items():
            if isinstance(v, dict):
                found = _find_state_dict_recursively(v, f"{path}.{k}")
                if found is not None:
                    return found
    return None


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    found = _find_state_dict_recursively(ckpt)
    if found is None:
        raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")
    state_dict, found_from = found
    state_dict = strip_prefix_in_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from: {ckpt_path}")
    print(f"  Found from      : {found_from}")
    print(f"  Missing keys    : {len(missing)}")
    print(f"  Unexpected keys : {len(unexpected)}")


def logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return torch.sigmoid(logits[:, 0])
    return torch.softmax(logits, dim=1)[:, 1]


@torch.no_grad()
def infer_tile(model, processor, rgb: np.ndarray) -> np.ndarray:
    pil_img = Image.fromarray(rgb)
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    logits = model(**inputs).logits
    logits = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False)
    prob = logits_to_prob(logits)[0].detach().cpu().numpy().astype(np.float32)
    return prob


def smooth_binary_mask(mask01: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return mask01.astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(np.uint8)


def init_canvas(h: int, w: int) -> Dict[str, np.ndarray]:
    return {
        "rgb_sum": np.zeros((h, w, 3), dtype=np.float32),
        "rgb_count": np.zeros((h, w, 1), dtype=np.float32),
        "prob_sum": np.zeros((h, w), dtype=np.float32),
        "prob_count": np.zeros((h, w), dtype=np.float32),
        "gt_sum": np.zeros((h, w), dtype=np.float32),
        "gt_count": np.zeros((h, w), dtype=np.float32),
    }


def build_parent_canvas(tile_records: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_h = max(r["y"] + r["h"] for r in tile_records)
    max_w = max(r["x"] + r["w"] for r in tile_records)
    canvas = init_canvas(max_h, max_w)

    for r in tile_records:
        y, x, h, w = int(r["y"]), int(r["x"]), int(r["h"]), int(r["w"])
        canvas["rgb_sum"][y:y+h, x:x+w] += r["rgb"].astype(np.float32)
        canvas["rgb_count"][y:y+h, x:x+w] += 1.0
        canvas["prob_sum"][y:y+h, x:x+w] += r["prob"].astype(np.float32)
        canvas["prob_count"][y:y+h, x:x+w] += 1.0
        canvas["gt_sum"][y:y+h, x:x+w] += r["gt"].astype(np.float32)
        canvas["gt_count"][y:y+h, x:x+w] += 1.0

    big_rgb = (canvas["rgb_sum"] / np.maximum(canvas["rgb_count"], 1.0)).clip(0, 255).astype(np.uint8)
    big_prob = (canvas["prob_sum"] / np.maximum(canvas["prob_count"], 1.0)).clip(0, 1).astype(np.float32)
    big_gt = ((canvas["gt_sum"] / np.maximum(canvas["gt_count"], 1.0)) >= 0.5).astype(np.uint8)
    return big_rgb, big_prob, big_gt


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    weight_path = Path(args.weight_path)
    output_root = Path(args.output_root)
    split = args.split
    prob_threshold = float(args.prob_threshold)
    pred_open_ksize = int(args.pred_open_ksize)
    save_prob_npy = not bool(args.no_prob_npy)
    save_prob_png_flag = not bool(args.no_prob_png)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    img_dir = dataset_root / split / "img"
    gt_mask_dir = dataset_root / split / "mask"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")
    if not gt_mask_dir.exists():
        print(f"[Warning] GT folder not found. Empty GT masks will be saved: {gt_mask_dir}")

    image_out_dir = output_root / "Image"
    pred_mask_dir = output_root / "Pred_Mask"
    pred_prob_npy_dir = output_root / "Pred_Prob_NPY"
    pred_prob_png_dir = output_root / "Pred_Prob_PNG"
    gt_mask_out_dir = output_root / "GT_Mask"
    overlay_dir = output_root / "Overlay"
    meta_dir = output_root / "metadata"

    dirs = [image_out_dir, pred_mask_dir, gt_mask_out_dir, overlay_dir, meta_dir]
    if save_prob_npy:
        dirs.append(pred_prob_npy_dir)
    if save_prob_png_flag:
        dirs.append(pred_prob_png_dir)

    for d in dirs:
        reset_dir(d) if args.clear_output_dirs else ensure_dir(d)

    img_paths = list_images(img_dir)
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {img_dir}")

    parent_to_imgpaths = defaultdict(list)
    skipped_parse_rows = []
    for img_path in img_paths:
        try:
            info = parse_tile_info(img_path)
            parent_to_imgpaths[str(info["parent"])].append(img_path)
        except Exception as e:
            skipped_parse_rows.append({"img_name": img_path.name, "reason": repr(e)})

    all_parents = sorted(parent_to_imgpaths.keys())
    if not all_parents:
        raise RuntimeError("No valid parent images parsed from tile filenames.")
    if args.num_random_parents and args.num_random_parents > 0:
        selected_parents = random.sample(all_parents, k=min(args.num_random_parents, len(all_parents)))
        selected_parents = sorted(selected_parents)
    else:
        selected_parents = all_parents

    processor = SegformerImageProcessor.from_pretrained(args.hf_name)
    model = SegformerForSemanticSegmentation.from_pretrained(args.hf_name, num_labels=2, ignore_mismatched_sizes=True)
    load_checkpoint(model, weight_path)
    model.to(DEVICE)
    model.eval()

    tile_rows = []
    parent_rows = []
    missing_gt_rows = []
    skipped_parent_rows = []

    for parent_name in tqdm(selected_parents, desc="Parent inference and stitching"):
        tile_records = []
        for img_path in sorted(parent_to_imgpaths[parent_name]):
            try:
                rgb = read_rgb(img_path)
                info = parse_tile_info(img_path)
                h, w = rgb.shape[:2]
                gt_path = find_matching_mask(img_path, gt_mask_dir) if gt_mask_dir.exists() else None
                if gt_path is not None:
                    gt = read_mask01(gt_path)
                    if gt.shape != (h, w):
                        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                else:
                    gt = np.zeros((h, w), dtype=np.uint8)
                    missing_gt_rows.append({"parent": parent_name, "img_name": img_path.name})

                prob = infer_tile(model, processor, rgb)
                pred_raw = (prob >= prob_threshold).astype(np.uint8)
                pred_smooth = smooth_binary_mask(pred_raw, pred_open_ksize)

                tile_records.append({
                    "parent": parent_name, "img_name": img_path.name,
                    "x": int(info["x"]), "y": int(info["y"]), "h": int(h), "w": int(w),
                    "rgb": rgb, "gt": gt, "prob": prob,
                })
                tile_rows.append({
                    "parent": parent_name, "img_name": img_path.name,
                    "x": int(info["x"]), "y": int(info["y"]), "tile_h": int(h), "tile_w": int(w),
                    "gt_pixels": int(gt.sum()),
                    "pred_raw_pixels_tile": int(pred_raw.sum()),
                    "pred_smooth_pixels_tile": int(pred_smooth.sum()),
                    "mean_prob_tile": float(prob.mean()),
                })
            except Exception as e:
                skipped_parent_rows.append({"parent": parent_name, "img_name": img_path.name, "reason": repr(e)})

        if not tile_records:
            skipped_parent_rows.append({"parent": parent_name, "img_name": "", "reason": "no_valid_tiles"})
            continue

        big_rgb, big_prob, big_gt = build_parent_canvas(tile_records)
        big_pred_raw = (big_prob >= prob_threshold).astype(np.uint8)
        big_pred = smooth_binary_mask(big_pred_raw, pred_open_ksize)
        overlay = make_overlay(big_rgb, big_pred)

        save_rgb(big_rgb, image_out_dir / f"{parent_name}.png")
        save_binary_mask(big_pred, pred_mask_dir / f"{parent_name}.png")
        save_binary_mask(big_gt, gt_mask_out_dir / f"{parent_name}.png")
        save_rgb(overlay, overlay_dir / f"{parent_name}_overlay.png")
        if save_prob_npy:
            np.save(pred_prob_npy_dir / f"{parent_name}.npy", big_prob.astype(np.float32))
        if save_prob_png_flag:
            save_prob_png(big_prob, pred_prob_png_dir / f"{parent_name}.png")

        parent_rows.append({
            "parent": parent_name,
            "tile_count": len(tile_records),
            "height": int(big_rgb.shape[0]),
            "width": int(big_rgb.shape[1]),
            "gt_pixels": int(big_gt.sum()),
            "pred_raw_pixels": int(big_pred_raw.sum()),
            "pred_pixels": int(big_pred.sum()),
            "mean_prob": float(big_prob.mean()),
            "prob_threshold": prob_threshold,
            "pred_open_ksize": pred_open_ksize,
        })

    pd.DataFrame(tile_rows).to_csv(meta_dir / "tile_level_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(parent_rows).to_csv(meta_dir / "parent_level_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_gt_rows).to_csv(meta_dir / "missing_gt_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_parse_rows).to_csv(meta_dir / "skipped_parse_report.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(skipped_parent_rows).to_csv(meta_dir / "skipped_parent_or_tile_report.csv", index=False, encoding="utf-8-sig")
    with open(meta_dir / "selected_parents.json", "w", encoding="utf-8") as f:
        json.dump(selected_parents, f, ensure_ascii=False, indent=2)

    config = {
        "MODE": "segformer_tiled_inference_to_full_masks",
        "RUN_NAME": args.run_name,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "DATASET_ROOT": str(dataset_root),
        "WEIGHT_PATH": str(weight_path),
        "OUTPUT_ROOT": str(output_root),
        "HF_NAME": args.hf_name,
        "EVAL_SPLIT": split,
        "DEVICE": DEVICE,
        "PROB_THRESHOLD": prob_threshold,
        "PRED_OPEN_KSIZE": pred_open_ksize,
        "NUM_RANDOM_PARENTS": args.num_random_parents,
        "RANDOM_SEED": args.random_seed,
        "SAVE_PROB_NPY": save_prob_npy,
        "SAVE_PROB_PNG": save_prob_png_flag,
        "OUTPUTS": {
            "A3_PRED_MASK_DIR": str(pred_mask_dir),
            "A1_GT_MASK_DIR": str(gt_mask_out_dir),
            "IMAGE_DIR": str(image_out_dir),
            "PRED_PROB_NPY_DIR": str(pred_prob_npy_dir) if save_prob_npy else "",
            "PRED_PROB_PNG_DIR": str(pred_prob_png_dir) if save_prob_png_flag else "",
            "OVERLAY_DIR": str(overlay_dir),
        },
    }
    with open(meta_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Saved parents: {len(parent_rows)} / {len(selected_parents)}")
    print(f"A3 Pred_Mask: {pred_mask_dir}")
    print(f"A1 GT_Mask  : {gt_mask_out_dir}")
    print(f"Image       : {image_out_dir}")
    print(f"Metadata    : {meta_dir}")


if __name__ == "__main__":
    main()
