"""
pseudo_labeling.py - Шаг 3 спецификации v4.
Пайплайн полуавтоматической псевдоразметки (Pseudo-Labeling / Self-Training).
Автоматическая разметка неразмеченных полетных кадров БПЛА высокоуверенными детекциями (conf >= 0.85)
с генерацией сегментационных полигонов в формате YOLOv8-seg.
"""

import os
import shutil
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def extract_polygon_coords(mask_contour, width, height):
    """Преобразует контур маски OpenCV в нормализованные координаты полигона YOLO (0.0..1.0)."""
    normalized = []
    for pt in mask_contour:
        x = np.clip(pt[0][0] / width, 0.0, 1.0)
        y = np.clip(pt[0][1] / height, 0.0, 1.0)
        normalized.extend([round(float(x), 6), round(float(y), 6)])
    return normalized


def generate_pseudo_labels(
    model_weights: str,
    unlabelled_img_dir: str,
    output_dir: str,
    conf_threshold: float = 0.85,
    save_empty_backgrounds: bool = True
):
    """
    Генерирует псевдоразметку для неразмеченных снимков БПЛА.
    
    Args:
        model_weights: путь к весам предобученной модели (best_model.pt или fold_0/best.pt)
        unlabelled_img_dir: директория с сырыми кадрами
        output_dir: директория для сохранения расширенного датасета
        conf_threshold: порог уверенности модели для принятия псевдометки (0.85)
        save_empty_backgrounds: сохранять ли пустые лейблы для кадров без дефектов (clean_bg)
    """
    model_path = Path(model_weights)
    if not model_path.exists():
        if Path("best_model.pt").exists():
            model_path = Path("best_model.pt")
            print(f"[!] Указанный путь не найден, используется {model_path}")
        else:
            raise FileNotFoundError(f"Веса модели не найдены: {model_weights}")

    model = YOLO(str(model_path))
    img_path = Path(unlabelled_img_dir)
    out_img_path = Path(output_dir) / "images"
    out_lbl_path = Path(output_dir) / "labels"
    
    out_img_path.mkdir(parents=True, exist_ok=True)
    out_lbl_path.mkdir(parents=True, exist_ok=True)
    
    # Сбор всех изображений
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    images = []
    for ext in extensions:
        images.extend(list(img_path.glob(ext)))
    images = sorted(list(set(images)))
    
    print(f"[*] Генерация псевдометок для {len(images)} кадров из {img_path} (порог conf >= {conf_threshold})...")
    
    labeled_count = 0
    clean_bg_count = 0
    total_polygons = 0
    
    for img_p in images:
        orig_img = cv2.imread(str(img_p))
        if orig_img is None:
            continue
        h, w = orig_img.shape[:2]
        
        results = model.predict(str(img_p), conf=conf_threshold, verbose=False)[0]
        label_lines = []
        
        # Если есть сегментационные маски
        if results.masks is not None and len(results.masks) > 0:
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            
            # Извлекаем полигоны xyn из Ultralytics
            polygons_xyn = results.masks.xyn
            for cls_id, conf, poly in zip(classes, confs, polygons_xyn):
                if conf < conf_threshold:
                    continue
                if len(poly) < 3:
                    continue
                
                coords_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                label_lines.append(f"{cls_id} {coords_str}")
                total_polygons += 1
        
        dest_img = out_img_path / img_p.name
        label_file = out_lbl_path / f"{img_p.stem}.txt"
        
        if len(label_lines) > 0:
            # Кадр содержит подтвержденные псевдодефекты
            cv2.imwrite(str(dest_img), orig_img)
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines) + "\n")
            labeled_count += 1
        elif save_empty_backgrounds and clean_bg_count < 15:
            # Кадр без дефектов - сохраняем как чистый фон (для подавления FP)
            cv2.imwrite(str(dest_img), orig_img)
            label_file.write_text("")
            clean_bg_count += 1
            
    print(f"\n[+] Псевдоразметка завершена:")
    print(f"  - Кадров с добавленными дефектами: {labeled_count}")
    print(f"  - Всего полигонов сгенерировано : {total_polygons}")
    print(f"  - Отрицательных фоновых кадров   : {clean_bg_count}")
    print(f"  - Итого изображений в выборке    : {len(list(out_img_path.glob('*.*')))}")
    print(f"  - Результаты сохранены в         : {output_dir}")
    
    return {
        'labeled_count': labeled_count,
        'total_polygons': total_polygons,
        'clean_bg_count': clean_bg_count,
        'output_dir': str(output_dir)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Пайплайн полуавтоматической псевдоразметки (v4)")
    parser.add_argument("--weights", type=str, default="best_model.pt", help="Веса модели-разметчика")
    parser.add_argument("--input", type=str, default="datasets/unlabelled_uav_frames", help="Каталог неразмеченных кадров")
    parser.add_argument("--output", type=str, default="datasets/augmented_with_pseudo", help="Каталог расширенного датасета")
    parser.add_argument("--conf", type=float, default=0.85, help="Порог уверенности для псевдоразметки")
    args = parser.parse_args()
    
    generate_pseudo_labels(
        model_weights=args.weights,
        unlabelled_img_dir=args.input,
        output_dir=args.output,
        conf_threshold=args.conf
    )
