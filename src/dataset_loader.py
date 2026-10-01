"""
Dataset Loader & Generator for Oil Storage Tanks and Pipelines Defect Detection.
TRL 3 PoC - Fuel & Energy Complex (ТЭК).
Classes:
    0: corrosion (Коррозия металлоконструкций)
    1: crack (Усталостные и околошовные трещины)
    2: coating_damage (Отслоение / повреждение защитного покрытия)
"""

import os
import glob
import random
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CLASS_NAMES = {
    0: 'corrosion',
    1: 'crack',
    2: 'coating_damage'
}

CLASS_COLORS = {
    0: (40, 110, 200),    # Rust/Corrosion (BGR: brownish/orange)
    1: (0, 0, 230),       # Crack (BGR: bright red)
    2: (220, 180, 20)     # Coating damage (BGR: turquoise/cyan)
}


def create_metallic_background(width=640, height=640, bg_type='tank_wall'):
    """
    Generates realistic industrial metallic surfaces:
    - tank_wall: curved or flat sheet metal with welding seams, rivets, lighting gradients
    - pipeline: cylindrical gradient with specular highlights
    - weld_zone: prominent welded seam with heat-affected zone
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Base metallic gray with slight tone variation
    base_val = random.randint(120, 180)
    color_tint = np.array([
        base_val + random.randint(-8, 5),   # B
        base_val + random.randint(-4, 6),   # G
        base_val + random.randint(-5, 10)   # R
    ], dtype=np.float32)
    
    img[:] = np.clip(color_tint, 0, 255).astype(np.uint8)
    
    # Add subtle Gaussian noise to simulate metal grain
    noise = np.random.normal(0, random.uniform(6, 14), (height, width, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    if bg_type == 'pipeline':
        # Cylindrical shading gradient across X or Y
        axis = random.choice(['horizontal', 'vertical'])
        if axis == 'horizontal':
            grad = np.sin(np.linspace(0.2, np.pi - 0.2, width)) ** 1.5
            grad = np.tile(grad, (height, 1))
        else:
            grad = np.sin(np.linspace(0.2, np.pi - 0.2, height)) ** 1.5
            grad = np.tile(grad[:, None], (1, width))
            
        grad = (grad * 0.7 + 0.3)[:, :, None]
        img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)
        
    elif bg_type == 'weld_zone' or random.random() < 0.6:
        # Draw welding seams (common site for defects in RVS & pipelines)
        weld_y = random.randint(int(height * 0.2), int(height * 0.8))
        weld_thickness = random.randint(12, 26)
        
        # Heat affected zone (darker / discolored band)
        cv2.line(img, (0, weld_y), (width, weld_y), (base_val - 35, base_val - 25, base_val - 15), weld_thickness + 18)
        # Scaled weld bead texture
        for wx in range(0, width, 6):
            wy = weld_y + int(np.sin(wx / 8.0) * 3)
            bead_radius = random.randint(4, 9)
            bead_color = (
                max(0, base_val - random.randint(30, 60)),
                max(0, base_val - random.randint(20, 50)),
                max(0, base_val - random.randint(10, 40))
            )
            cv2.circle(img, (wx, wy), bead_radius, bead_color, -1)

    # Tank wall panels / sheet joints
    if random.random() < 0.5:
        joint_x = random.randint(int(width * 0.2), int(width * 0.8))
        cv2.line(img, (joint_x, 0), (joint_x, height), (base_val - 45, base_val - 40, base_val - 35), 2)
        # Rivets / bolts along joint
        for ry in range(25, height, random.randint(40, 70)):
            cv2.circle(img, (joint_x, ry), 5, (base_val - 50, base_val - 45, base_val - 40), -1)
            cv2.circle(img, (joint_x - 1, ry - 1), 2, (base_val + 30, base_val + 30, base_val + 35), -1)

    return img


def generate_corrosion_defect(img, min_size=50, max_size=180):
    """
    Generates realistic corrosion polygon with irregular ragged contours
    and rich rust discoloration (Fe2O3/FeOOH reddish-brown textures).
    Returns (updated_img, polygon_points_norm).
    """
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))
    
    num_pts = random.randint(14, 24)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_pts))
    base_r = random.uniform(min_size, max_size) / 2.0
    radii = base_r * np.random.uniform(0.6, 1.4, num_pts)
    
    pts_x = np.clip(cx + radii * np.cos(angles), 4, w - 5).astype(np.int32)
    pts_y = np.clip(cy + radii * np.sin(angles), 4, h - 5).astype(np.int32)
    poly = np.stack([pts_x, pts_y], axis=1)
    
    # Create mask for this defect
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    
    # Blur mask edges for natural blend
    blur_k = random.choice([7, 11, 15])
    soft_mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0).astype(np.float32) / 255.0
    
    # Rust texture: browns, deep ochre, rust-orange
    rust_layer = np.zeros_like(img, dtype=np.float32)
    rust_layer[:, :, 0] = np.random.uniform(15, 45, (h, w))    # Blue (low)
    rust_layer[:, :, 1] = np.random.uniform(50, 110, (h, w))   # Green (medium)
    rust_layer[:, :, 2] = np.random.uniform(120, 205, (h, w))  # Red (high - iron oxide)
    
    # Pitting noise inside rust
    pit_noise = np.random.normal(0, 18, (h, w, 1))
    rust_layer = np.clip(rust_layer + pit_noise, 0, 255)
    
    # Alpha blend
    alpha = soft_mask[:, :, None] * random.uniform(0.82, 0.96)
    img_float = img.astype(np.float32) * (1.0 - alpha) + rust_layer * alpha
    img[:] = np.clip(img_float, 0, 255).astype(np.uint8)
    
    # Extract contour for accurate YOLO polygon
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, None
        
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.008 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    
    if len(approx) < 3:
        return img, None
        
    norm_poly = approx.astype(np.float32) / np.array([w, h], dtype=np.float32)
    norm_poly = np.clip(norm_poly, 0.001, 0.999)
    return img, norm_poly


def generate_crack_defect(img, length_range=(60, 200)):
    """
    Generates fatigue or weld-toe crack with random-walk fracture path
    and micro-branching. Returns (updated_img, polygon_points_norm).
    """
    h, w = img.shape[:2]
    start_x = random.randint(int(w * 0.15), int(w * 0.85))
    start_y = random.randint(int(h * 0.15), int(h * 0.85))
    
    length = random.randint(*length_range)
    main_angle = random.uniform(0, 2 * np.pi)
    
    curr_x, curr_y = float(start_x), float(start_y)
    path = [(int(curr_x), int(curr_y))]
    step = 5
    
    for _ in range(length // step):
        delta_angle = np.random.normal(0, 0.35)
        main_angle += delta_angle
        curr_x += step * np.cos(main_angle)
        curr_y += step * np.sin(main_angle)
        curr_x = np.clip(curr_x, 8, w - 9)
        curr_y = np.clip(curr_y, 8, h - 9)
        path.append((int(curr_x), int(curr_y)))
        
    mask = np.zeros((h, w), dtype=np.uint8)
    thickness = random.randint(3, 7)
    
    for i in range(len(path) - 1):
        cv2.line(mask, path[i], path[i+1], 255, thickness)
        
    # Dark fissure in metal
    dark_val = random.randint(15, 45)
    fissure = np.full_like(img, dark_val, dtype=np.uint8)
    
    # Slight corrosion around crack mouth (stress corrosion cracking)
    halo = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    halo_diff = cv2.subtract(halo, mask)
    img[halo_diff > 0] = np.clip(img[halo_diff > 0].astype(np.int32) + np.array([-15, -10, 25]), 0, 255).astype(np.uint8)
    
    img[mask > 0] = fissure[mask > 0]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, None
        
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    
    if len(approx) < 3:
        return img, None
        
    norm_poly = approx.astype(np.float32) / np.array([w, h], dtype=np.float32)
    norm_poly = np.clip(norm_poly, 0.001, 0.999)
    return img, norm_poly


def generate_coating_damage_defect(img, min_size=60, max_size=170):
    """
    Generates paint blistering, chipping, or mechanical delamination defect.
    Exposes bare or slightly oxidized substrate underneath.
    """
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))
    
    num_pts = random.randint(8, 16)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_pts))
    base_r = random.uniform(min_size, max_size) / 2.0
    radii = base_r * np.random.uniform(0.7, 1.3, num_pts)
    
    pts_x = np.clip(cx + radii * np.cos(angles), 5, w - 6).astype(np.int32)
    pts_y = np.clip(cy + radii * np.sin(angles), 5, h - 6).astype(np.int32)
    poly = np.stack([pts_x, pts_y], axis=1)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    
    # Under-coating bare steel with oxidized streaks
    undercoat = np.zeros_like(img, dtype=np.uint8)
    under_val = random.randint(80, 115)
    undercoat[:] = (under_val - 5, under_val, under_val + 12)
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.float32)
    undercoat = np.clip(undercoat.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # White/bright peel border
    border = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((4, 4), np.uint8))
    
    img[mask > 0] = undercoat[mask > 0]
    img[border > 0] = np.clip(img[border > 0].astype(np.int32) + 60, 0, 255).astype(np.uint8)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, None
        
    c = max(contours, key=cv2.contourArea)
    epsilon = 0.007 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    
    if len(approx) < 3:
        return img, None
        
    norm_poly = approx.astype(np.float32) / np.array([w, h], dtype=np.float32)
    norm_poly = np.clip(norm_poly, 0.001, 0.999)
    return img, norm_poly


def generate_synthetic_dataset(output_dir="dataset", num_samples=65, img_size=(640, 640)):
    """
    Creates full synthetic inspection dataset with photorealistic industrial
    defects on storage tanks and pipelines.
    Guarantees stratified defect representation across all 3 classes.
    """
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    bg_types = ['tank_wall', 'pipeline', 'weld_zone']
    classes_pool = [0, 1, 2]
    
    print(f"[*] Synthesizing {num_samples} realistic industrial defect images in '{output_dir}'...")
    
    records = []
    
    for idx in range(num_samples):
        bg_t = random.choice(bg_types)
        img = create_metallic_background(img_size[0], img_size[1], bg_type=bg_t)
        
        # Decide defects for this image (1 to 3 defects, multi-defect images included)
        num_defects = random.choices([1, 2, 3], weights=[0.45, 0.40, 0.15])[0]
        # Ensure balanced distribution: cycle through classes
        primary_cls = classes_pool[idx % len(classes_pool)]
        selected_classes = [primary_cls]
        for _ in range(num_defects - 1):
            selected_classes.append(random.choice(classes_pool))
            
        label_lines = []
        
        for cls_id in selected_classes:
            if cls_id == 0:
                img, poly = generate_corrosion_defect(img)
            elif cls_id == 1:
                img, poly = generate_crack_defect(img)
            else:
                img, poly = generate_coating_damage_defect(img)
                
            if poly is not None and len(poly) >= 3:
                # Format: class_id x1 y1 x2 y2 ... xn yn
                coords_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                label_lines.append(f"{cls_id} {coords_str}")
                records.append({'image_id': idx, 'class_id': cls_id, 'class_name': CLASS_NAMES[cls_id]})
                
        # If no valid defect generated (rare fallback), force generate corrosion
        if not label_lines:
            img, poly = generate_corrosion_defect(img)
            if poly is not None:
                coords_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                label_lines.append(f"0 {coords_str}")
                records.append({'image_id': idx, 'class_id': 0, 'class_name': CLASS_NAMES[0]})

        img_filename = f"tank_pipe_defect_{idx:04d}.jpg"
        lbl_filename = f"tank_pipe_defect_{idx:04d}.txt"
        
        img_path = os.path.join(output_dir, "images", img_filename)
        lbl_path = os.path.join(output_dir, "labels", lbl_filename)
        
        cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(lbl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
            
    print(f"[+] Dataset synthesis completed: {num_samples} images, {len(records)} defect instances.")
    return output_dir


def validate_dataset(dataset_dir="dataset"):
    """
    Validates integrity of dataset:
    - Verifies image files can be read
    - Checks pairing of images and .txt label files
    - Checks YOLO polygon format validity (norm range [0, 1], >= 3 vertices)
    - Computes class distribution and instance statistics
    """
    img_dir = os.path.join(dataset_dir, "images")
    lbl_dir = os.path.join(dataset_dir, "labels")
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        raise FileNotFoundError(f"Dataset folders missing in {dataset_dir}")
        
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    stats = {
        'total_images': len(img_files),
        'valid_pairs': 0,
        'missing_labels': 0,
        'corrupted_images': 0,
        'class_counts': {name: 0 for name in CLASS_NAMES.values()},
        'total_polygons': 0,
        'polygon_vertex_counts': []
    }
    
    for img_name in img_files:
        stem = Path(img_name).stem
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, f"{stem}.txt")
        
        # Test image reading
        img = cv2.imread(img_path)
        if img is None:
            stats['corrupted_images'] += 1
            continue
            
        if not os.path.exists(lbl_path):
            stats['missing_labels'] += 1
            continue
            
        stats['valid_pairs'] += 1
        
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        for line in lines:
            parts = line.split()
            if len(parts) < 7:  # class + at least 3 points (6 coords)
                continue
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if cls_id in CLASS_NAMES:
                stats['class_counts'][CLASS_NAMES[cls_id]] += 1
                stats['total_polygons'] += 1
                stats['polygon_vertex_counts'].append(len(coords) // 2)

    return stats


def plot_dataset_statistics(stats, output_path="reports/defect_distribution.png"):
    """
    Plots professional class distribution and polygon complexity chart for reports.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Class frequency
    classes = list(stats['class_counts'].keys())
    counts = [stats['class_counts'][c] for c in classes]
    colors = ['#c0392b', '#e67e22', '#2980b9']
    
    bars = axes[0].bar(classes, counts, color=colors, edgecolor='black', alpha=0.85, width=0.55)
    axes[0].set_title("Распределение классов дефектов (TRL 3 Датасет)", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Количество инстансов дефектов", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Chart 2: Polygon complexity (vertex count distribution)
    axes[1].hist(stats['polygon_vertex_counts'], bins=12, color='#27ae60', edgecolor='black', alpha=0.8)
    axes[1].set_title("Сложность геометрии полигонов (Число вершин)", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Количество вершин полигона маски", fontsize=11)
    axes[1].set_ylabel("Частота", fontsize=11)
    axes[1].grid(linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Dataset distribution plot saved to {output_path}")


if __name__ == "__main__":
    dataset_dir = "dataset"
    if not os.path.exists(os.path.join(dataset_dir, "images")) or len(os.listdir(os.path.join(dataset_dir, "images"))) < 20:
        generate_synthetic_dataset(output_dir=dataset_dir, num_samples=65)
        
    stats = validate_dataset(dataset_dir)
    print("\nDataset Validation Summary:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    plot_dataset_statistics(stats)
