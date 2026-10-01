"""
Secondary Corrosion & Rust Dataset Generator for Metal Surface Inspection.
Specialized on:
1. Multi-scale corrosion & rust patterns on storage tanks & pipelines (Class 0: corrosion).
2. Clean metal backgrounds (negative samples) to suppress False Positives and boost Precision.
"""

import os
import random
import cv2
import numpy as np
from pathlib import Path


def generate_clean_background(width=640, height=640, surface_type='clean_paint'):
    """
    Generates negative samples (clean surfaces with no defects):
    - clean_paint: uniform or weathered industrial paint (white, light gray, industrial blue)
    - clean_steel: rolled steel with grinding marks, lighting gradient, specular reflection
    - clean_weld: pristine welding seam with no cracks or corrosion
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if surface_type == 'clean_paint':
        # Industrial tank paint: off-white, light gray, pale blue/beige
        palette = random.choice([
            (215, 220, 225),  # light industrial gray
            (235, 235, 230),  # tank white/cream
            (190, 180, 140),  # olive-beige primer
            (180, 170, 120)   # tank enamel
        ])
        noise = np.random.normal(0, random.uniform(3, 7), (height, width, 3)).astype(np.float32)
        base = np.full((height, width, 3), palette, dtype=np.float32)
        img = np.clip(base + noise, 0, 255).astype(np.uint8)
        
        # Subtle horizontal or vertical panel lap joint
        if random.random() < 0.6:
            pos = random.randint(int(height * 0.2), int(height * 0.8))
            cv2.line(img, (0, pos), (width, pos), (palette[0] - 25, palette[1] - 25, palette[2] - 25), 2)
            cv2.line(img, (0, pos + 1), (width, pos + 1), (palette[0] + 15, palette[1] + 15, palette[2] + 15), 1)

    elif surface_type == 'clean_steel':
        # Rolled bare metal with directional brushed finish and cylindrical shading
        base_gray = random.randint(140, 175)
        img[:] = (base_gray, base_gray, base_gray + random.randint(-4, 6))
        # Directional brush lines
        brush_noise = np.random.normal(0, 8, (height, width)).astype(np.float32)
        brush_noise = cv2.GaussianBlur(brush_noise, (1, 15), 0)
        img = np.clip(img.astype(np.float32) + brush_noise[:, :, None], 0, 255).astype(np.uint8)
        
        # Specular light glare / reflection (common source of false positives)
        glare_x = random.randint(int(width * 0.3), int(width * 0.7))
        glare_w = random.randint(80, 180)
        glare = np.exp(-((np.arange(width) - glare_x) ** 2) / (2 * (glare_w / 2.5) ** 2))
        glare = np.tile(glare[None, :, None], (height, 1, 3)) * random.uniform(40, 80)
        img = np.clip(img.astype(np.float32) + glare, 0, 255).astype(np.uint8)

    else:  # clean_weld
        base_gray = random.randint(130, 165)
        img[:] = base_gray
        weld_y = random.randint(int(height * 0.3), int(height * 0.7))
        # Pristine weld bead
        cv2.line(img, (0, weld_y), (width, weld_y), (base_gray - 30, base_gray - 20, base_gray - 10), 16)
        for wx in range(0, width, 5):
            bead_r = random.randint(4, 7)
            cv2.circle(img, (wx, weld_y), bead_r, (base_gray - 45, base_gray - 35, base_gray - 25), -1)
            cv2.circle(img, (wx, weld_y - 2), 2, (base_gray + 20, base_gray + 25, base_gray + 30), -1)

    return img


def generate_detailed_rust_defect(img, min_size=60, max_size=220):
    """
    Generates rich corrosion/rust polygon with irregular organic contour
    and multi-toned iron oxide textures.
    Returns (updated_img, polygon_norm).
    """
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.15), int(w * 0.85))
    cy = random.randint(int(h * 0.15), int(h * 0.85))
    
    num_pts = random.randint(16, 28)
    angles = np.sort(np.random.uniform(0, 2 * np.pi, num_pts))
    base_r = random.uniform(min_size, max_size) / 2.0
    radii = base_r * np.random.uniform(0.55, 1.45, num_pts)
    
    pts_x = np.clip(cx + radii * np.cos(angles), 5, w - 6).astype(np.int32)
    pts_y = np.clip(cy + radii * np.sin(angles), 5, h - 6).astype(np.int32)
    poly = np.stack([pts_x, pts_y], axis=1)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    
    # Soft feathering along boundary
    soft_mask = cv2.GaussianBlur(mask, (13, 13), 0).astype(np.float32) / 255.0
    
    # Multi-layered rust coloration (Fe2O3 red-brown, Fe3O4 dark, FeOOH orange-ochre)
    rust = np.zeros_like(img, dtype=np.float32)
    rust[:, :, 0] = np.random.uniform(10, 40, (h, w))    # B
    rust[:, :, 1] = np.random.uniform(45, 95, (h, w))    # G
    rust[:, :, 2] = np.random.uniform(130, 215, (h, w))  # R
    
    # Pitting roughness texture
    pit_noise = np.random.normal(0, 22, (h, w, 1))
    rust = np.clip(rust + pit_noise, 0, 255)
    
    # Blending
    alpha = soft_mask[:, :, None] * random.uniform(0.85, 0.98)
    img_float = img.astype(np.float32) * (1.0 - alpha) + rust * alpha
    img[:] = np.clip(img_float, 0, 255).astype(np.uint8)
    
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


def generate_secondary_dataset(output_dir="dataset_secondary", num_rust=35, num_clean_bg=15):
    """
    Builds the secondary dataset:
    - num_rust images containing dedicated multi-scale rust defects (Class 0: corrosion)
    - num_clean_bg negative samples (empty .txt) to train False Positive rejection
    Total: num_rust + num_clean_bg images.
    """
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    print(f"[*] Generating secondary rust dataset ({num_rust} rust + {num_clean_bg} clean backgrounds)...")
    
    # 1. Rust defect images
    surface_types = ['clean_paint', 'clean_steel', 'clean_weld']
    for i in range(num_rust):
        st = random.choice(surface_types)
        img = generate_clean_background(640, 640, surface_type=st)
        
        num_patches = random.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
        label_lines = []
        
        for _ in range(num_patches):
            img, poly = generate_detailed_rust_defect(img)
            if poly is not None and len(poly) >= 3:
                # Class 0: corrosion
                coords_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                label_lines.append(f"0 {coords_str}")
                
        img_filename = f"sec_rust_{i:04d}.jpg"
        lbl_filename = f"sec_rust_{i:04d}.txt"
        
        cv2.imwrite(os.path.join(img_dir, img_filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        with open(os.path.join(lbl_dir, lbl_filename), "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
            
    # 2. Clean background images (negative samples for FP suppression)
    for j in range(num_clean_bg):
        st = random.choice(surface_types)
        img = generate_clean_background(640, 640, surface_type=st)
        
        img_filename = f"sec_clean_bg_{j:04d}.jpg"
        lbl_filename = f"sec_clean_bg_{j:04d}.txt"
        
        cv2.imwrite(os.path.join(img_dir, img_filename), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        # Empty label file designates negative/background image in YOLO
        with open(os.path.join(lbl_dir, lbl_filename), "w", encoding="utf-8") as f:
            f.write("")
            
    print(f"[+] Secondary dataset ready in '{output_dir}': {num_rust + num_clean_bg} images total.")
    return output_dir


if __name__ == "__main__":
    generate_secondary_dataset()
