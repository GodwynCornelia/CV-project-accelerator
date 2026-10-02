"""
Шаг 1. Внедрение Group K-Fold (Честная валидация) и TTA (TRL 3 / TRL 4 v3).
- Группировка кадров по физическим объектам (GroupKFold) для исключения data leakage между соседними ракурсами
- Обучение на сбалансированной выборке merged_dataset_v2
- Индустриальные гиперпараметры под БПЛА (copy_paste=0.3, close_mosaic, AdamW, cache='ram')
- Шаг 4: Валидация с Test-Time Augmentation (augment=True)
- Шаг 5: Двухуровневое логирование и раздельный контроль классов (Corrosion, Crack, Coating Damage)
- Сохранение отчетов в reports/kfold_v3_group_metrics.csv и лучших весов в best_model.pt
"""

import os
import sys
import types
import io
import shutil
import argparse
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# Robust CPU AVX2-safe patch: avoid polars crash on Intel Ivy Bridge
def _safe_read_results_csv(self):
    try:
        if hasattr(self, 'csv') and self.csv.exists():
            df = pd.read_csv(self.csv)
            df.columns = [c.strip() for c in df.columns]
            return {col: df[col].tolist() for col in df.columns}
    except Exception:
        pass
    return {}

BaseTrainer.read_results_csv = _safe_read_results_csv

if 'polars' not in sys.modules:
    class _MockDataFrame:
        def __init__(self, *args, **kwargs): pass
    class _MockSeries:
        def __init__(self, *args, **kwargs): pass
    _polars_mock = types.ModuleType("polars")
    _polars_mock.DataFrame = _MockDataFrame
    _polars_mock.Series = _MockSeries
    def _mock_read_csv(source, **kwargs):
        src = io.BytesIO(source) if isinstance(source, bytes) else source
        _df = pd.read_csv(src)
        _df.columns = [c.strip() for c in _df.columns]
        class _MockRes:
            def to_dict(self, as_series=False):
                return {col: _df[col].tolist() for col in _df.columns}
        return _MockRes()
    _polars_mock.read_csv = _mock_read_csv
    sys.modules["polars"] = _polars_mock

def run_group_kfold_v3(epochs: int = 6, imgsz: int = 320, batch: int = 16, n_splits: int = 5):
    base_dir = Path(__file__).resolve().parent
    if (base_dir / "datasets" / "merged_dataset_v2").exists():
        root_dir = base_dir
    else:
        root_dir = base_dir.parent
        
    dataset_dir = root_dir / "datasets" / "merged_dataset_v2"
    if not dataset_dir.exists():
        dataset_dir = root_dir / "merged_dataset_v2"
        
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        
    img_files = sorted(list((dataset_dir / "images").glob("*.*")))
    print(f"[*] Found {len(img_files)} images in {dataset_dir / 'images'}")
    
    # Извлекаем группы из имен файлов для исключения утечки данных между кадрами одного объекта
    groups = []
    for p in img_files:
        parts = p.stem.split('_')
        num_str = ''.join([c for c in parts[-1] if c.isdigit()])
        num = int(num_str) if num_str else 0
        # Кластеризация смежных кадров инспекции одного резервуара / участка трубы
        subgroup = num // 6
        group_id = f"{parts[0]}_{parts[1]}_g{subgroup}"
        groups.append(group_id)
        
    unique_groups = len(set(groups))
    print(f"[*] Сформировано {unique_groups} уникальных физических групп объектов (GroupKFold)")
    
    gkf = GroupKFold(n_splits=n_splits)
    kfold_base = root_dir / "runs" / "kfold_v3_group"
    kfold_base.mkdir(parents=True, exist_ok=True)
    
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    best_overall_map50 = -1.0
    close_mosaic = min(3, max(1, epochs // 3))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(img_files, groups=groups)):
        print(f"\n==================== ЗАПУСК GROUP FOLD {fold} ====================")
        fold_dir = kfold_base / f"fold_{fold}"
        train_img_dir = fold_dir / "images" / "train"
        train_lbl_dir = fold_dir / "labels" / "train"
        val_img_dir = fold_dir / "images" / "val"
        val_lbl_dir = fold_dir / "labels" / "val"
        
        if fold_dir.exists():
            shutil.rmtree(fold_dir, ignore_errors=True)
            
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        for i in train_idx:
            p = img_files[i]
            shutil.copy2(p, train_img_dir / p.name)
            src_lbl = dataset_dir / "labels" / f"{p.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, train_lbl_dir / f"{p.stem}.txt")
            else:
                (train_lbl_dir / f"{p.stem}.txt").write_text("")
            
        for i in val_idx:
            p = img_files[i]
            shutil.copy2(p, val_img_dir / p.name)
            src_lbl = dataset_dir / "labels" / f"{p.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, val_lbl_dir / f"{p.stem}.txt")
            else:
                (val_lbl_dir / f"{p.stem}.txt").write_text("")
            
        fold_yaml = fold_dir / "fold.yaml"
        fold_yaml_data = {
            'path': str(fold_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'corrosion',
                1: 'crack',
                2: 'coating_damage'
            }
        }
        with open(fold_yaml, "w", encoding="utf-8") as f:
            yaml.dump(fold_yaml_data, f, sort_keys=False)
            
        # Инициализация сегментатора YOLOv8n-seg
        model = YOLO("yolov8n-seg.pt")
        
        model.train(
            data=str(fold_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            optimizer="AdamW",
            lr0=0.002,
            lrf=0.01,
            warmup_epochs=0.5,
            mosaic=1.0,
            copy_paste=0.3,
            close_mosaic=close_mosaic,
            cache='ram',
            project=str(kfold_base),
            name=f"fold_{fold}_train",
            verbose=False,
            device="cpu",
            exist_ok=True
        )
        
        weights_fold = kfold_base / f"fold_{fold}_train" / "weights" / "best.pt"
        eval_model = YOLO(str(weights_fold)) if weights_fold.exists() else model
        
        # Шаг 4: Валидация с Test-Time Augmentation (augment=True)
        # 1. Аналитическая валидация (conf=0.001) с TTA
        val_raw_tta = eval_model.val(data=str(fold_yaml), conf=0.001, iou=0.6, augment=True, split="val", imgsz=imgsz, verbose=False)
        
        # 2. Эксплуатационная валидация (conf=0.25 и conf=0.35) с TTA
        val_c25_tta = eval_model.val(data=str(fold_yaml), conf=0.25, iou=0.6, augment=True, split="val", imgsz=imgsz, verbose=False)
        val_c35_tta = eval_model.val(data=str(fold_yaml), conf=0.35, iou=0.6, augment=True, split="val", imgsz=imgsz, verbose=False)
        
        # Шаг 5: Двухуровневое логирование и раздельный контроль классов
        class_maps = getattr(val_raw_tta.seg, 'maps', None)
        c_corrosion = float(class_maps[0]) if class_maps is not None and len(class_maps) > 0 else float(val_raw_tta.seg.map)
        c_crack = float(class_maps[1]) if class_maps is not None and len(class_maps) > 1 else float(val_raw_tta.seg.map)
        c_coating = float(class_maps[2]) if class_maps is not None and len(class_maps) > 2 else float(val_raw_tta.seg.map)
        
        map50_mask = float(val_raw_tta.seg.map50)
        map50_95_mask = float(val_raw_tta.seg.map)
        prec_c25 = float(val_c25_tta.seg.mp)
        rec_c25 = float(val_c25_tta.seg.mr)
        prec_c35 = float(val_c35_tta.seg.mp)
        rec_c35 = float(val_c35_tta.seg.mr)
        box_map50 = float(val_raw_tta.box.map50)
        box_map50_95 = float(val_raw_tta.box.map)
        
        print(f"Group Fold {fold} Results (TTA): Mask mAP50={map50_mask:.4f}, mAP50-95={map50_95_mask:.4f}")
        print(f"  Раздельный контроль: Corrosion mAP50-95={c_corrosion:.4f} | Crack={c_crack:.4f} | Coating={c_coating:.4f}")
        print(f"  Эксплуатационный: @conf=0.25 P={prec_c25*100:.1f}%, R={rec_c25*100:.1f}% | @conf=0.35 P={prec_c35*100:.1f}%, R={rec_c35*100:.1f}%")
        
        if map50_mask > best_overall_map50:
            best_overall_map50 = map50_mask
            if weights_fold.exists():
                shutil.copy2(weights_fold, root_dir / "best_model.pt")
                print(f"[*] Новая лучшая модель зафиксирована в best_model.pt (Group mAP50={best_overall_map50:.4f})")
                
        results.append({
            "fold": fold,
            "mAP50_mask": round(map50_mask, 4),
            "mAP50_95_mask": round(map50_95_mask, 4),
            "precision": round(prec_c25, 4),
            "recall": round(rec_c25, 4),
            "precision_c35": round(prec_c35, 4),
            "recall_c35": round(rec_c35, 4),
            "corrosion_mAP50_95": round(c_corrosion, 4),
            "crack_mAP50_95": round(c_crack, 4),
            "coating_mAP50_95": round(c_coating, 4),
            "box_mAP50": round(box_map50, 4),
            "best_weights": str(weights_fold)
        })

    df_res = pd.DataFrame(results)
    out_csv = reports_dir / "kfold_v3_group_metrics.csv"
    df_res.to_csv(out_csv, index=False)
    
    stats_df = pd.DataFrame({
        "Metric": ["Mask mAP@0.50 (TTA)", "Mask mAP@0.50:0.95 (TTA)", "Precision @ conf=0.25", "Recall @ conf=0.25", "Precision @ conf=0.35", "Recall @ conf=0.35"],
        "Mean": [df_res["mAP50_mask"].mean(), df_res["mAP50_95_mask"].mean(), df_res["precision"].mean(), df_res["recall"].mean(), df_res["precision_c35"].mean(), df_res["recall_c35"].mean()],
        "Std": [df_res["mAP50_mask"].std(), df_res["mAP50_95_mask"].std(), df_res["precision"].std(), df_res["recall"].std(), df_res["precision_c35"].std(), df_res["recall_c35"].std()],
        "Min": [df_res["mAP50_mask"].min(), df_res["mAP50_95_mask"].min(), df_res["precision"].min(), df_res["recall"].min(), df_res["precision_c35"].min(), df_res["recall_c35"].min()],
        "Max": [df_res["mAP50_mask"].max(), df_res["mAP50_95_mask"].max(), df_res["precision"].max(), df_res["recall"].max(), df_res["precision_c35"].max(), df_res["recall_c35"].max()]
    })
    stats_df.to_csv(reports_dir / "kfold_v3_statistical_summary.csv", index=False)
    
    print("\n================ ИТОГОВЫЕ РЕЗУЛЬТАТЫ GROUP K-FOLD V3 (TTA) ================")
    print(df_res[["fold", "mAP50_mask", "mAP50_95_mask", "precision", "recall", "corrosion_mAP50_95", "crack_mAP50_95", "coating_mAP50_95"]])
    print("\nСводная статистика:")
    print(stats_df.to_string(index=False))
    print(f"\nGroup K-Fold обучение завершено. Метрики сохранены в: {out_csv}")
    
    # Генерация диаграммы устойчивости Group K-Fold
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].boxplot([df_res["mAP50_mask"], df_res["mAP50_95_mask"]], tick_labels=["mAP@0.50 (TTA)", "mAP@0.50:0.95 (TTA)"], patch_artist=True)
        axes[0].set_title("Group K-Fold Mask mAP (No Data Leakage)")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        axes[1].bar(["Corrosion", "Crack", "Coating Damage"], 
                    [df_res["corrosion_mAP50_95"].mean(), df_res["crack_mAP50_95"].mean(), df_res["coating_mAP50_95"].mean()],
                    color=['#dc6e28', '#e60000', '#14b4dc'], alpha=0.85)
        axes[1].set_title("Class-wise Mean Mask mAP@0.50:0.95")
        axes[1].set_ylabel("Mean mAP50-95")
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(reports_dir / "kfold_v3_group_boxplot.png", dpi=300)
        plt.close()
        print(f"[*] График сохранен в {reports_dir / 'kfold_v3_group_boxplot.png'}")
    except Exception as e:
        print(f"[!] Warning generating boxplot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6, help="Количество эпох на фолд")
    parser.add_argument("--imgsz", type=int, default=320, help="Размер входного кадра")
    parser.add_argument("--batch", type=int, default=16, help="Размер батча")
    parser.add_argument("--splits", type=int, default=5, help="Количество фолдов GroupKFold")
    args = parser.parse_args()
    
    run_group_kfold_v3(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, n_splits=args.splits)
