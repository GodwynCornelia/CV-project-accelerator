"""
train_v4_final.py - Шаг 5 спецификации v4.
Мультимасштабное обучение и высокая детализация (imgsz=1024, multi_scale=True).
Максимальное сохранение геометрии микротрещин околошовных зон и мелких сколов ЛКП.
Готовность к развертыванию на NVIDIA Jetson (TRL 4 / TRL 5).
"""

import os
import argparse
from pathlib import Path

# Защита от сбоя инструкции AVX2 в Polars
try:
    from ultralytics.engine.trainer import BaseTrainer

    class _MockSeries:
        def __init__(self, data=None): self._data = list(data) if data is not None else []
        def to_list(self): return self._data
        def max(self): return max(self._data) if self._data else 0.0

    class _MockRes:
        def __init__(self): self.columns = []
        def __getitem__(self, item): return _MockSeries()

    BaseTrainer.read_results_csv = lambda self: _MockRes()
except Exception:
    pass

from ultralytics import YOLO


def run_v4_training(
    model_weights="best_model.pt",
    data="data_v2.yaml",
    epochs=50,
    imgsz=1024,
    batch=8,
    project="runs/v4_optimized",
    name="v4_final_highres",
    device="cpu",
    export_onnx=False
):
    """
    Финальная доводка детектора YOLOv8n-seg:
    - Повышенное разрешение (imgsz=1024)
    - multi_scale=True для адаптации к перепадам высоты полета БПЛА
    - Сниженный lr0=0.0005 для тонкой сходимости
    - Подготовка экспортных весов для бортовых систем
    """
    model_path = Path(model_weights)
    if not model_path.exists():
        fallback_candidates = [
            "runs/v4_optimized/student_distilled/weights/best.pt",
            "runs/kfold_v3_group/fold_0_train/weights/best.pt",
            "yolov8n-seg.pt"
        ]
        for c in fallback_candidates:
            if Path(c).exists():
                model_path = Path(c)
                break
                
    print(f"[*] Старт финального мультимасштабного обучения v4 на базе {model_path} (imgsz={imgsz}, multi_scale=True)")
    model = YOLO(str(model_path))

    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        multi_scale=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        project=str(project),
        name=name,
        device=device,
        exist_ok=True,
        verbose=True
    )

    print("[+] Мультимасштабное обучение завершено.")
    
    # Валидация с высоким разрешением
    val_res = model.val(data=str(data), imgsz=imgsz, augment=True, conf=0.25, verbose=False)
    print(f"[*] Итоговый Mask mAP@0.50 (imgsz={imgsz}): {val_res.seg.map50:.4f}, mAP@0.50:0.95: {val_res.seg.map:.4f}")
    
    if export_onnx:
        print("[*] Экспорт модели в формат ONNX для последующей сборки TensorRT на NVIDIA Jetson...")
        try:
            model.export(format="onnx", imgsz=imgsz, dynamic=True)
            print("[+] Экспорт в ONNX успешно выполнен.")
        except Exception as e:
            print(f"[!] Экспорт ONNX пропущен: {e}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Финальное мультимасштабное обучение YOLOv8n-seg v4")
    parser.add_argument("--weights", type=str, default="best_model.pt", help="Исходные веса")
    parser.add_argument("--data", type=str, default="data_v2.yaml", help="Конфиг данных")
    parser.add_argument("--epochs", type=int, default=50, help="Эпохи доводки")
    parser.add_argument("--imgsz", type=int, default=1024, help="Разрешение")
    parser.add_argument("--batch", type=int, default=8, help="Батч")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство")
    parser.add_argument("--export", action="store_true", help="Экспорт в ONNX")
    args = parser.parse_args()

    run_v4_training(
        model_weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        export_onnx=args.export
    )
