"""
Step 2: Filter, remap IDs, validate polygon geometry, and merge into balanced dataset v2.
TRL 3 / TRL 4 Pipeline for YOLOv8n-seg defect segmentation.
"""

import os
import shutil
from pathlib import Path
from collections import Counter
import numpy as np

# Mapping rules from donor source IDs to target classes:
# 0: corrosion, 1: crack, 2: coating_damage
MAPPING_RULES = {
    "donor_cracks": {
        0: 1  # crack ID 0 -> target ID 1 (crack)
    },
    "donor_coatings": {
        0: 2, # peeling / paint_damage -> target ID 2 (coating_damage)
        1: 0, # rust -> target ID 0 (corrosion)
        2: 2  # scratch -> target ID 2 (coating_damage)
    }
}

# Determine base paths robustly whether executed from repo root or src/
BASE_DIR = Path(__file__).resolve().parent
if (BASE_DIR / "datasets").exists() or (BASE_DIR / "merged_dataset").exists():
    ROOT_DIR = BASE_DIR
else:
    ROOT_DIR = BASE_DIR.parent

OUTPUT_DIR = ROOT_DIR / "datasets" / "merged_dataset_v2"
OUT_IMG_DIR = OUTPUT_DIR / "images"
OUT_LBL_DIR = OUTPUT_DIR / "labels"

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

def process_file(src_img, src_lbl, prefix, class_map, is_background=False):
    dst_img_name = f"{prefix}_{src_img.name}"
    dst_lbl_name = f"{prefix}_{src_lbl.stem}.txt"
    
    # 1. Copy image
    shutil.copy2(src_img, OUT_IMG_DIR / dst_img_name)
    
    # 2. Process negative / clean background
    dst_lbl_path = OUT_LBL_DIR / dst_lbl_name
    if is_background or not src_lbl.exists():
        open(dst_lbl_path, "w").close()
        return
        
    valid_lines = []
    with open(src_lbl, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # polygon must contain class + at least 3 points (x, y)
                continue
            
            orig_cls = int(parts[0])
            if orig_cls not in class_map:
                continue
                
            target_cls = class_map[orig_cls]
            coords = [float(x) for x in parts[1:]]
            
            # Sanity checks coordinates in range [0, 1]
            coords_clamped = np.clip(coords, 0.0, 1.0)
            coords_str = " ".join([f"{c:.6f}" for c in coords_clamped])
            valid_lines.append(f"{target_cls} {coords_str}\n")
            
    with open(dst_lbl_path, "w") as f_out:
        f_out.writelines(valid_lines)

def run_merge():
    print(f"[*] Starting dataset merge into: {OUTPUT_DIR}")
    
    # 1. Transfer existing lab dataset (merged_dataset)
    current_dir_candidates = [
        ROOT_DIR / "merged_dataset",
        ROOT_DIR / "datasets" / "merged_dataset"
    ]
    current_dir = next((p for p in current_dir_candidates if p.exists()), None)
    if not current_dir:
        raise FileNotFoundError("Could not find merged_dataset directory.")
        
    print(f"--> [1/3] Merging lab dataset from {current_dir}...")
    lab_count = 0
    for img_path in (current_dir / "images").glob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            lbl_path = current_dir / "labels" / f"{img_path.stem}.txt"
            process_file(img_path, lbl_path, prefix="lab", class_map={0: 0, 1: 1, 2: 2})
            lab_count += 1
    print(f"    Added {lab_count} lab images.")

    # 2. Add crack donor (limit ~70-80 high-quality samples)
    crack_dir = ROOT_DIR / "donor_cracks" / "train"
    print(f"--> [2/3] Merging crack donors from {crack_dir}...")
    crack_count = 0
    for idx, img_path in enumerate((crack_dir / "images").glob("*.*")):
        if idx >= 80:
            break
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            lbl_path = crack_dir / "labels" / f"{img_path.stem}.txt"
            process_file(img_path, lbl_path, prefix="add_crack", class_map=MAPPING_RULES["donor_cracks"])
            crack_count += 1
    print(f"    Added {crack_count} crack donor images.")

    # 3. Add coating donor (limit ~70-80 samples)
    coat_dir = ROOT_DIR / "donor_coatings" / "train"
    print(f"--> [3/3] Merging coating donors from {coat_dir}...")
    coat_count = 0
    for idx, img_path in enumerate((coat_dir / "images").glob("*.*")):
        if idx >= 80:
            break
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            lbl_path = coat_dir / "labels" / f"{img_path.stem}.txt"
            process_file(img_path, lbl_path, prefix="add_coat", class_map=MAPPING_RULES["donor_coatings"])
            coat_count += 1
    print(f"    Added {coat_count} coating donor images.")

    total_images = len(list(OUT_IMG_DIR.glob("*.*")))
    print(f"--> Итого сформировано изображений в выборке: {total_images}")
    
    # Also create a symlink / copy at ROOT_DIR / merged_dataset_v2 for convenience
    alt_output = ROOT_DIR / "merged_dataset_v2"
    if not alt_output.exists():
        try:
            os.symlink(OUTPUT_DIR, alt_output, target_is_directory=True)
        except Exception:
            pass

    # Statistical verification of balanced distribution
    polygon_counts = Counter()
    clean_bg_count = 0
    total_labels = 0
    for lbl_file in OUT_LBL_DIR.glob("*.txt"):
        total_labels += 1
        lines = [l.strip() for l in lbl_file.read_text().splitlines() if l.strip()]
        if len(lines) == 0:
            clean_bg_count += 1
        else:
            for l in lines:
                cls_id = int(l.split()[0])
                polygon_counts[cls_id] += 1
                
    class_names = {0: "corrosion", 1: "crack", 2: "coating_damage"}
    print("\n================== ИТОГОВЫЙ БАЛАНС КЛАССОВ V2 ==================")
    print(f"Всего изображений: {total_images}")
    print(f"Чистых фоновых сэмплов (clean_bg): {clean_bg_count} ({clean_bg_count/total_images*100:.1f}%)")
    total_polygons = sum(polygon_counts.values())
    print(f"Всего полигонов сегментации: {total_polygons}")
    for cid, cname in class_names.items():
        cnt = polygon_counts[cid]
        pct = (cnt / total_polygons * 100) if total_polygons > 0 else 0
        print(f"  Класс {cid} ({cname}): {cnt} полигонов ({pct:.1f}%)")
    print("=================================================================\n")

if __name__ == "__main__":
    run_merge()
