"""
Шаг 2. Индустриальные аугментации под ТЭК (Albumentations).
Пайплайн искажений, имитирующий суровые погодные и технологические условия съемок с воздуха на нефтебазах и НПЗ:
- Моделирование солнечных бликов и глубоких теней (RandomBrightnessContrast, RandomShadow)
- Имитация микросмаза кадра при маневрировании дрона или порывах ветра (MotionBlur)
- Имитация локальных выпадений сенсора и шума (CoarseDropout)
"""

import albumentations as A
import numpy as np

def get_industrial_augmentation():
    """
    Пайплайн аугментаций под открытые индустриальные объекты ТЭК:
    - Моделирование солнечных бликов и глубоких теней
    - Имитация микросмаза кадра при маневрировании дрона
    """
    try:
        shadow_tf = A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=3, p=0.3)
    except Exception:
        shadow_tf = A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3), p=0.3)
        
    try:
        dropout_tf = A.CoarseDropout(max_holes=6, max_height=20, max_width=20, fill_value=0, p=0.2)
    except Exception:
        dropout_tf = A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(5, 20), hole_width_range=(5, 20), fill=0, p=0.2)
        
    return A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
        shadow_tf,
        A.MotionBlur(blur_limit=(3, 7), p=0.25),
        dropout_tf,
    ], p=0.7)

def apply_augmentation_to_image(img: np.ndarray) -> np.ndarray:
    """
    Применяет индустриальную трансформацию к входному изображению (BGR или RGB).
    """
    transform = get_industrial_augmentation()
    augmented = transform(image=img)
    return augmented['image']

if __name__ == "__main__":
    dummy = np.full((320, 320, 3), 128, dtype=np.uint8)
    res = apply_augmentation_to_image(dummy)
    print(f"[*] Custom industrial augmentation test successful. Output shape: {res.shape}")
