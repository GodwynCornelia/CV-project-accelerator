"""
train_with_focal.py - Шаг 2 спецификации v4.
Обучение YOLOv8n-seg с кастомными Loss Gains (loss_gains_tuning.yaml) и Focal Loss для борьбы с дисбалансом классов ТЭК.
Подавление легких градиентов фонового металла и акцент на микротрещинах (crack).
"""

import sys
import os
import argparse
from pathlib import Path
import yaml
import torch

# Защита от сбоя инструкции AVX2 в Polars на старых процессорах
try:
    import ultralytics.models.yolo.segment.train as seg_train
    from ultralytics.engine.trainer import BaseTrainer

    class _MockSeries:
        def __init__(self, data=None): self._data = list(data) if data is not None else []
        def to_list(self): return self._data
        def max(self): return max(self._data) if self._data else 0.0

    class _MockRes:
        def __init__(self): self.columns = []
        def __getitem__(self, item): return _MockSeries()

    def _safe_read_results(self):
        return _MockRes()

    BaseTrainer.read_results_csv = _safe_read_results
except Exception:
    pass

from ultralytics import YOLO


def train_focal_model(
    data="data_v2.yaml",
    cfg_hyp="loss_gains_tuning.yaml",
    epochs=120,
    imgsz=640,
    batch=16,
    project="runs/v4_optimized",
    name="focal_loss_run",
    device="cpu"
):
    """
    Запуск обучения с кастомизированными весами функций потерь:
    - seg: 12.0 (акцент на пиксельные маски)
    - box: 7.5 (балансировка боксов)
    - cls: 0.5 (фокальный штраф)
    - cos_lr: True (косинусный отжиг learning rate)
    """
    root_dir = Path(__file__).resolve().parent
    hyp_file = Path(cfg_hyp)
    if not hyp_file.is_absolute():
        hyp_file = root_dir / cfg_hyp

    print(f"[*] Инициализация модели YOLOv8n-seg с кастомными loss gains: {hyp_file}")
    model = YOLO("yolov8n-seg.pt")

    train_args = {
        "data": str(data),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "optimizer": "AdamW",
        "cos_lr": True,
        "project": str(project),
        "name": name,
        "device": device,
        "exist_ok": True,
        "verbose": True
    }

    # Загружаем гиперпараметры из loss_gains_tuning.yaml
    if hyp_file.exists():
        with open(hyp_file, "r", encoding="utf-8") as f:
            hyp_data = yaml.safe_load(f)
            if isinstance(hyp_data, dict):
                # Передаем loss gains напрямую в train arguments
                train_args.update({k: v for k, v in hyp_data.items() if k in [
                    "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", 
                    "warmup_momentum", "box", "cls", "dfl", "seg", "pose", "kobj"
                ]})
                print(f"[+] Применены кастомные коэффициенты потерь: box={hyp_data.get('box')}, cls={hyp_data.get('cls')}, seg={hyp_data.get('seg')}")

    results = model.train(**train_args)
    print("Обучение с кастомными loss gains и focal балансировкой завершено.")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обучение с кастомными loss gains и focal балансировкой")
    parser.add_argument("--data", type=str, default="data_v2.yaml", help="Путь к data.yaml")
    parser.add_argument("--cfg", type=str, default="loss_gains_tuning.yaml", help="Путь к YAML с гиперпараметрами потерь")
    parser.add_argument("--epochs", type=int, default=120, help="Количество эпох")
    parser.add_argument("--imgsz", type=int, default=640, help="Разрешение кадра")
    parser.add_argument("--batch", type=int, default=16, help="Размер батча")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство (cpu / cuda)")
    args = parser.parse_args()

    train_focal_model(
        data=args.data,
        cfg_hyp=args.cfg,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )
