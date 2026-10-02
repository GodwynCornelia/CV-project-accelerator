"""
Шаг 3. Морфологическая постобработка масок (postprocessing.py).
Применение классических операторов компьютерного зрения (морфологического закрытия Morphological Closing)
к предсказанным нейросетью маскам дефектов:
- Компенсирует недостаточную емкость легковесной архитектуры YOLOv8n-seg
- Сшивает микроразрывы и устраняет пустоты внутри тонких протяженных трещин и сколов ЛКП
- Повышает Recall без увеличения размера нейросети и снижения FPS на борту БПЛА
"""

import cv2
import numpy as np

def refine_mask_borders(mask: np.ndarray) -> np.ndarray:
    """
    Морфологическое закрытие (Morphological Closing) для устранения разрывов 
    и пустых пикселей внутри контуров тонких трещин и сколов ЛКП.
    """
    is_binary_01 = False
    if mask.dtype != np.uint8:
        if mask.max() <= 1.0:
            is_binary_01 = True
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
            
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if is_binary_01:
        return (refined_mask > 127).astype(np.float32)
    return refined_mask

def batch_refine_masks(masks_list):
    """
    Пакетная постобработка списка бинарных или полутоновых масок.
    """
    return [refine_mask_borders(m) for m in masks_list]

if __name__ == "__main__":
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[40:60, 40:48] = 255
    test_mask[40:60, 52:60] = 255
    refined = refine_mask_borders(test_mask)
    print(f"[*] Morphological closing test success. Nonzero before: {np.count_nonzero(test_mask)}, after: {np.count_nonzero(refined)}")
