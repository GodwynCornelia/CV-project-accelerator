"""
Dataset Merger & Harmonization Module.
Combines primary defect dataset (ds1) and secondary corrosion dataset (ds2)
into unified merged_dataset/ with verified polygon segmentation masks.
"""

import os
import shutil
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


UNIFIED_CLASSES = {
    0: 'corrosion',
    1: 'crack',
    2: 'coating_damage'
}


def validate_and_clean_polygon_line(line, default_class=0):
    """
    Validates a single YOLO segmentation polygon line:
    <class_id> x1 y1 x2 y2 ... xn yn
    Ensures coords are normalized [0.001, 0.999] and at least 3 vertices exist.
    """
    parts = line.strip().split()
    if len(parts) < 7:  # class + at least 3 points
        return None
        
    try:
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None
        
    if len(coords) % 2 != 0:
        coords = coords[:-1]
        
    if len(coords) < 6:
        return None
        
    # Clip coordinates to strictly valid YOLO normalized bounds
    cleaned_coords = [min(max(c, 0.001), 0.999) for c in coords]
    coords_str = " ".join([f"{c:.6f}" for c in cleaned_coords])
    return f"{cls_id} {coords_str}"


def merge_datasets(
    ds1_dir="dataset",
    ds2_dir="dataset_secondary",
    output_dir="merged_dataset"
):
    """
    Merges ds1 and ds2 into output_dir with ds1_ and ds2_ prefixes.
    Ensures unified class mapping and valid polygon segmentation annotations.
    """
    out_img_dir = os.path.join(output_dir, "images")
    out_lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)
    
    stats = {
        'total_images': 0,
        'ds1_images': 0,
        'ds2_images': 0,
        'background_negative_images': 0,
        'defect_instances': {c: 0 for c in UNIFIED_CLASSES.values()}
    }
    
    sources = [
        (ds1_dir, "ds1_"),
        (ds2_dir, "ds2_")
    ]
    
    print(f"[*] Starting dataset merge into '{output_dir}'...")
    
    for src_dir, prefix in sources:
        src_img_dir = os.path.join(src_dir, "images")
        src_lbl_dir = os.path.join(src_dir, "labels")
        
        if not os.path.exists(src_img_dir):
            print(f"[!] Warning: directory {src_img_dir} does not exist.")
            continue
            
        img_files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        for img_name in img_files:
            stem = Path(img_name).stem
            ext = Path(img_name).suffix
            
            new_img_name = f"{prefix}{stem}{ext}"
            new_lbl_name = f"{prefix}{stem}.txt"
            
            src_img_path = os.path.join(src_img_dir, img_name)
            src_lbl_path = os.path.join(src_lbl_dir, f"{stem}.txt")
            
            dst_img_path = os.path.join(out_img_dir, new_img_name)
            dst_lbl_path = os.path.join(out_lbl_dir, new_lbl_name)
            
            # Copy image
            shutil.copy2(src_img_path, dst_img_path)
            
            # Process and validate labels
            cleaned_lines = []
            if os.path.exists(src_lbl_path):
                with open(src_lbl_path, "r", encoding="utf-8") as f:
                    raw_lines = [l.strip() for l in f if l.strip()]
                    
                for line in raw_lines:
                    valid_line = validate_and_clean_polygon_line(line)
                    if valid_line:
                        parts = valid_line.split()
                        cls_id = int(parts[0])
                        # If class_id is within known classes, count it
                        if cls_id in UNIFIED_CLASSES:
                            stats['defect_instances'][UNIFIED_CLASSES[cls_id]] += 1
                        cleaned_lines.append(valid_line)
                        
            with open(dst_lbl_path, "w", encoding="utf-8") as f:
                if cleaned_lines:
                    f.write("\n".join(cleaned_lines) + "\n")
                else:
                    # Negative sample / background
                    f.write("")
                    stats['background_negative_images'] += 1

            stats['total_images'] += 1
            if prefix == "ds1_":
                stats['ds1_images'] += 1
            else:
                stats['ds2_images'] += 1
                
    # Save statistics
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    stats_path = os.path.join(reports_dir, "merged_dataset_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        
    print(f"\n[+] Merged dataset successfully created in '{output_dir}':")
    print(f"    - Total images: {stats['total_images']} ({stats['ds1_images']} from DS1, {stats['ds2_images']} from DS2)")
    print(f"    - Background / negative images: {stats['background_negative_images']}")
    print(f"    - Defect instances: {stats['defect_instances']}")
    
    # Plot updated defect distribution
    plot_merged_distribution(stats, os.path.join(reports_dir, "defect_distribution.png"))
    return stats


def plot_merged_distribution(stats, output_path="reports/defect_distribution.png"):
    """Generates distribution chart of merged dataset for reports."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Defect class counts
    classes = list(stats['defect_instances'].keys())
    counts = [stats['defect_instances'][c] for c in classes]
    colors = ['#c0392b', '#e67e22', '#2980b9']
    
    bars = axes[0].bar(classes, counts, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    axes[0].set_title(f"Распределение дефектов (Объединенная выборка: N={stats['total_images']})", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Количество инстансов полигонов", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., h + 1, f'{int(h)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Right: Source composition (DS1 vs DS2 vs Background negatives)
    categories = ['Первичный датасет (DS1)', 'Вторичный датасет (DS2)', 'Фоновые снимки (Clean BG)']
    cat_counts = [stats['ds1_images'], stats['ds2_images'], stats['background_negative_images']]
    cat_colors = ['#34495e', '#16a085', '#27ae60']
    
    bars2 = axes[1].bar(categories, cat_counts, color=cat_colors, edgecolor='black', alpha=0.85, width=0.55)
    axes[1].set_title("Состав выборки и подавление ложных срабатываний (FP)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Количество изображений", fontsize=11)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars2:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{int(h)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Updated distribution chart saved to '{output_path}'")


if __name__ == "__main__":
    merge_datasets()
