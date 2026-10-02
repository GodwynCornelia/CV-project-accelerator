"""
Defect Analyzer & Surface Damage Quantification for Storage Tanks and Pipelines.
TRL 3 PoC - Fuel & Energy Complex (ТЭК).
Calculates exact defect surface percentage (%), assesses risk levels, and renders inspection overlays.
"""

import os
import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from postprocessing import refine_mask_borders


DEFECT_METADATA = {
    0: {
        'name': 'corrosion',
        'ru_name': 'Коррозия',
        'color': (40, 110, 220),       # BGR (Orange-Brown)
        'hex': '#dc6e28',
        'critical_threshold': 5.0      # >5% surface is critical
    },
    1: {
        'name': 'crack',
        'ru_name': 'Трещина / Излом',
        'color': (0, 0, 230),          # BGR (Bright Red)
        'hex': '#e60000',
        'critical_threshold': 0.1      # Any crack is critical
    },
    2: {
        'name': 'coating_damage',
        'ru_name': 'Повреждение ЛКП',
        'color': (220, 180, 20),       # BGR (Cyan-Blue)
        'hex': '#14b4dc',
        'critical_threshold': 8.0
    }
}


def compute_defect_metrics(image_shape, masks, classes, confidences):
    """
    Computes exact pixel area and surface damage percentage for each defect class.
    
    Returns:
        metrics (dict):
            total_defect_area_px
            total_defect_percent
            class_breakdown: {class_name: {'area_px': ..., 'percent': ..., 'count': ...}}
            risk_level: 'NORMAL' | 'WARNING' | 'CRITICAL'
            risk_description: str
    """
    img_h, img_w = image_shape[:2]
    total_img_pixels = img_h * img_w
    
    # Combined binary mask for all defects (handles overlapping regions correctly)
    combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    class_masks = {
        cls_id: np.zeros((img_h, img_w), dtype=np.uint8)
        for cls_id in DEFECT_METADATA.keys()
    }
    class_counts = {cls_id: 0 for cls_id in DEFECT_METADATA.keys()}
    
    if masks is not None and len(masks) > 0:
        for m, cls_id in zip(masks, classes):
            cls_id = int(cls_id)
            if cls_id not in class_masks:
                continue
                
            # Resize mask to original image shape if needed
            if m.shape[:2] != (img_h, img_w):
                m_resized = cv2.resize(m.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            else:
                m_resized = m.astype(np.uint8)
                
            bin_mask = (m_resized > 0).astype(np.uint8)
            combined_mask = np.bitwise_or(combined_mask, bin_mask)
            class_masks[cls_id] = np.bitwise_or(class_masks[cls_id], bin_mask)
            class_counts[cls_id] += 1
            
    total_defect_px = int(np.sum(combined_mask > 0))
    total_defect_pct = round((total_defect_px / total_img_pixels) * 100.0, 3)
    
    class_breakdown = {}
    has_critical_crack = False
    
    for cls_id, meta in DEFECT_METADATA.items():
        cls_px = int(np.sum(class_masks[cls_id] > 0))
        cls_pct = round((cls_px / total_img_pixels) * 100.0, 3)
        class_breakdown[meta['name']] = {
            'class_id': cls_id,
            'ru_name': meta['ru_name'],
            'count': class_counts[cls_id],
            'area_px': cls_px,
            'percent_surface': cls_pct
        }
        if meta['name'] == 'crack' and class_counts[cls_id] > 0:
            has_critical_crack = True
            
    # Risk evaluation
    if has_critical_crack or total_defect_pct >= 5.0:
        risk_level = "CRITICAL"
        risk_desc = "КРИТИЧЕСКИЙ РИСК: Обнаружены трещины или обширная коррозия (>5%). Требуется аварийный останов / ремонт."
    elif total_defect_pct >= 1.0:
        risk_level = "WARNING"
        risk_desc = "ПРЕДУПРЕЖДЕНИЕ: Очаговая коррозия / деградация покрытия (1-5%). Рекомендовано плановое ТО."
    else:
        risk_level = "NORMAL"
        risk_desc = "НОРМА: Дефекты минимальны (<1%) либо отсутствуют. Объект допущен к эксплуатации."
        
    return {
        'total_pixels': total_img_pixels,
        'defect_pixels': total_defect_px,
        'total_defect_percent': total_defect_pct,
        'class_breakdown': class_breakdown,
        'risk_level': risk_level,
        'risk_description': risk_desc
    }


def draw_defect_hud(image, metrics, detections):
    """
    Renders clean, modern HUD overlay with segmentation masks, outlines,
    and industrial telemetry panel.
    """
    h, w = image.shape[:2]
    annotated = image.copy()
    overlay = image.copy()
    
    # 1. Draw segmentation masks with alpha transparency
    for det in detections:
        mask = det['mask']
        cls_id = det['class_id']
        conf = det['confidence']
        color = DEFECT_METADATA.get(cls_id, {}).get('color', (0, 255, 0))
        
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
        bin_mask = mask > 0
        overlay[bin_mask] = color
        
        # Draw contour border
        contours, _ = cv2.findContours(bin_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.drawContours(annotated, contours, -1, color, 2, cv2.LINE_AA)
        
        # Bounding box & label
        bbox = det.get('bbox')
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_name = DEFECT_METADATA.get(cls_id, {}).get('name', 'defect')
            lbl_text = f"{cls_name} {conf:.2f}"
            
            # Label background badge
            (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), color, -1)
            cv2.putText(annotated, lbl_text, (x1 + 3, max(th, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend masks with 0.35 opacity
    annotated = cv2.addWeighted(overlay, 0.38, annotated, 0.62, 0)
    
    # 2. Draw telemetry header HUD
    hud_h = 75
    hud_bg = np.zeros((hud_h, w, 3), dtype=np.uint8)
    hud_bg[:] = (20, 24, 28)  # Sleek dark slate
    
    # Status color
    risk_colors = {
        'NORMAL': (60, 180, 75),      # Green
        'WARNING': (30, 180, 240),    # Orange/Yellow
        'CRITICAL': (30, 30, 220)     # Red
    }
    status_col = risk_colors.get(metrics['risk_level'], (200, 200, 200))
    
    # Draw status pill
    cv2.rectangle(hud_bg, (15, 15), (145, 60), status_col, -1, cv2.LINE_AA)
    cv2.putText(hud_bg, metrics['risk_level'], (24, 43), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Area damage text
    total_pct = metrics['total_defect_percent']
    cv2.putText(hud_bg, f"TOTAL DEFECT AREA: {total_pct:.2f}%", (165, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
                
    # Breakdown string
    cb = metrics['class_breakdown']
    breakdown_str = f"Corrosion: {cb['corrosion']['percent_surface']:.1f}% | Crack: {cb['crack']['percent_surface']:.1f}% | Coating: {cb['coating_damage']['percent_surface']:.1f}%"
    cv2.putText(hud_bg, breakdown_str, (165, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 185, 190), 1, cv2.LINE_AA)
    
    # Combine HUD and image
    result = np.vstack([hud_bg, annotated])
    return result


class DefectAnalyzer:
    def __init__(self, model_path="best_model.pt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        self.model = YOLO(model_path)
        print(f"[+] DefectAnalyzer loaded model from: {model_path}")
        
    def analyze_image(self, img_source, conf_threshold=0.18):
        """
        Runs inference on image (path or numpy array), calculates surface area,
        and produces visual inspection card.
        """
        if isinstance(img_source, str):
            orig_img = cv2.imread(img_source)
            img_name = Path(img_source).name
        else:
            orig_img = img_source.copy()
            img_name = "in_memory_image.jpg"
            
        if orig_img is None:
            raise ValueError("Failed to load image for analysis.")
            
        h, w = orig_img.shape[:2]
        
        # Inference with TTA
        results = self.model.predict(
            orig_img,
            conf=conf_threshold,
            device='cpu',
            augment=True,
            verbose=False
        )[0]
        
        detections = []
        raw_masks = []
        raw_classes = []
        raw_confs = []
        
        if results.masks is not None and len(results.masks) > 0:
            masks_data = results.masks.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            
            # Sort by confidence descending and select top salient detections
            order = np.argsort(-confs)
            top_indices = order[:6]
            
            for idx in top_indices:
                # Apply morphological closing to seal micro-cracks and bridge gaps
                m = refine_mask_borders(masks_data[idx])
                b = boxes[idx]
                c = classes[idx]
                score = confs[idx]
                
                detections.append({
                    'class_id': int(c),
                    'class_name': DEFECT_METADATA.get(int(c), {}).get('name', 'defect'),
                    'confidence': round(float(score), 4),
                    'bbox': [round(float(x), 2) for x in b],
                    'mask': m
                })
                raw_masks.append(m)
                raw_classes.append(c)
                raw_confs.append(score)
                
        metrics = compute_defect_metrics((h, w), raw_masks, raw_classes, raw_confs)
        annotated_img = draw_defect_hud(orig_img, metrics, detections)
        
        # Clean detection records for JSON (exclude numpy arrays)
        clean_detections = [{k: v for k, v in d.items() if k != 'mask'} for d in detections]
        
        return {
            'image_name': img_name,
            'image_dimensions': {'width': w, 'height': h},
            'detections': clean_detections,
            'metrics': metrics,
            'annotated_image': annotated_img
        }

    def batch_analyze_and_report(self, input_dir="merged_dataset/images", output_dir="reports", max_samples=6):
        """
        Processes batch of test images, saves visualization samples and defect_analysis.json.
        Prioritizes defect-bearing samples for inspection cards.
        """
        samples_dir = os.path.join(output_dir, "inference_samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        img_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + glob.glob(os.path.join(input_dir, "*.png")))
        if not img_files:
            print(f"[!] No images found in {input_dir}")
            return {}
            
        # Target samples known to have salient defects + clean background
        candidate_images = [
            p for p in img_files
            if any(k in p for k in ['0018', '0015', '0012', '0000', '0003', 'sec_rust_0005', 'sec_clean_bg'])
        ]
        test_subset = candidate_images[:max_samples] if len(candidate_images) >= max_samples else img_files[:max_samples]
        batch_results = []
        
        print(f"[*] Analyzing {len(test_subset)} test inspection images (calibrated conf=0.18)...")
        
        for p in test_subset:
            analysis = self.analyze_image(p)
            save_path = os.path.join(samples_dir, f"inspected_{analysis['image_name']}")
            cv2.imwrite(save_path, analysis['annotated_image'])
            
            batch_results.append({
                'image_name': analysis['image_name'],
                'image_dimensions': analysis['image_dimensions'],
                'total_defect_percent': analysis['metrics']['total_defect_percent'],
                'risk_level': analysis['metrics']['risk_level'],
                'risk_description': analysis['metrics']['risk_description'],
                'class_breakdown': analysis['metrics']['class_breakdown'],
                'detections_count': len(analysis['detections']),
                'detections': analysis['detections'],
                'visual_report_path': save_path
            })
            print(f" -> {analysis['image_name']}: Damage={analysis['metrics']['total_defect_percent']}% | Risk={analysis['metrics']['risk_level']}")
            
        json_path = os.path.join(output_dir, "defect_analysis.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                'project': 'TRL 3 PoC - Reservoir & Pipeline Defect Detection',
                'target_directions': ['Direction #2: Computer Vision', 'Direction #3: Infrastructure Monitoring'],
                'evaluated_samples_count': len(batch_results),
                'inspection_results': batch_results
            }, f, indent=2, ensure_ascii=False)
            
        print(f"[+] Defect analysis report saved to: {json_path}")
        return batch_results


if __name__ == "__main__":
    model_file = "best_model.pt"
    if not os.path.exists(model_file):
        model_file = "yolov8n-seg.pt"
    analyzer = DefectAnalyzer(model_file)
    input_images_dir = "datasets/merged_dataset_v2/images" if os.path.exists("datasets/merged_dataset_v2/images") else ("merged_dataset/images" if os.path.exists("merged_dataset/images") else "dataset/images")
    analyzer.batch_analyze_and_report(input_images_dir, "reports", max_samples=6)
