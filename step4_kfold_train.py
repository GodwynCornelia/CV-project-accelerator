"""
Step 4: Training and 5-Fold Cross-Validation on YOLOv8n-seg with optimized hyperparameters (TRL 3 / TRL 4).
Includes UAV model compensation techniques:
- copy_paste=0.3: rare crack and defect augmentation
- close_mosaic: fine-tuning polygon boundary masks
- conf=0.001 analytical validation (mAP50, mAP50-95)
- conf=0.25 and conf=0.35 operational UAV threshold calibrations
- Robust CPU AVX2-safe patch for Intel Ivy Bridge CPU
- UTF-8 YAML serialization for non-ASCII path support
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
from sklearn.model_selection import KFold
from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer

# Robust CPU AVX2-safe compatibility patch: avoid polars crash on older CPUs
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

def run_kfold_v2(epochs: int = 6, imgsz: int = 320, batch: int = 16, n_splits: int = 5):
    # Locate repository root and dataset dir
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
        
    img_files = np.array(sorted(list((dataset_dir / "images").glob("*.*"))))
    print(f"[*] Found {len(img_files)} images in {dataset_dir / 'images'}")
    
    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    kfold_base = root_dir / "runs" / "kfold_v2_runs"
    kfold_base.mkdir(parents=True, exist_ok=True)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    
    best_overall_map50 = -1.0
    best_weights_path = None
    
    close_mosaic = min(3, max(1, epochs // 3))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(img_files)):
        print(f"\n==================== ЗАПУСК FOLD {fold} ====================")
        fold_dir = kfold_base / f"fold_{fold}"
        fold_train_img = fold_dir / "images" / "train"
        fold_train_lbl = fold_dir / "labels" / "train"
        fold_val_img = fold_dir / "images" / "val"
        fold_val_lbl = fold_dir / "labels" / "val"
        
        # Clean existing fold directory if any
        if fold_dir.exists():
            shutil.rmtree(fold_dir, ignore_errors=True)
            
        for d in [fold_train_img, fold_train_lbl, fold_val_img, fold_val_lbl]:
            d.mkdir(parents=True, exist_ok=True)
            
        for p in img_files[train_idx]:
            shutil.copy2(p, fold_train_img / p.name)
            src_lbl = dataset_dir / "labels" / f"{p.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, fold_train_lbl / f"{p.stem}.txt")
            else:
                (fold_train_lbl / f"{p.stem}.txt").write_text("")
            
        for p in img_files[val_idx]:
            shutil.copy2(p, fold_val_img / p.name)
            src_lbl = dataset_dir / "labels" / f"{p.stem}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, fold_val_lbl / f"{p.stem}.txt")
            else:
                (fold_val_lbl / f"{p.stem}.txt").write_text("")
            
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
            
        # Initialize lightweight instance segmentation model
        model = YOLO("yolov8n-seg.pt")
        
        # Train with UAV compensation hyperparameters:
        # copy_paste=0.3: polygon copy-paste augmentation for crack and paint damage
        # close_mosaic: fine mask boundary fitting
        train_res = model.train(
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
        
        # Check fold best weights
        weights_fold = kfold_base / f"fold_{fold}_train" / "weights" / "best.pt"
        if not weights_fold.exists():
            weights_fold = fold_dir / "weights" / "best.pt"
            
        eval_model = YOLO(str(weights_fold)) if weights_fold.exists() else model
        
        # 1. Analytical validation (conf=0.001) for mAP50 and mAP50-95
        val_raw = eval_model.val(data=str(fold_yaml), conf=0.001, iou=0.6, split="val", imgsz=imgsz, verbose=False)
        
        # 2. Operational UAV validations
        val_c25 = eval_model.val(data=str(fold_yaml), conf=0.25, iou=0.6, split="val", imgsz=imgsz, verbose=False)
        val_c35 = eval_model.val(data=str(fold_yaml), conf=0.35, iou=0.6, split="val", imgsz=imgsz, verbose=False)
        
        map50_mask = float(val_raw.seg.map50)
        map50_95_mask = float(val_raw.seg.map)
        prec_c25 = float(val_c25.seg.mp)
        rec_c25 = float(val_c25.seg.mr)
        prec_c35 = float(val_c35.seg.mp)
        rec_c35 = float(val_c35.seg.mr)
        box_map50 = float(val_raw.box.map50)
        box_map50_95 = float(val_raw.box.map)
        
        print(f"Fold {fold} Results: Mask mAP50={map50_mask:.4f}, mAP50-95={map50_95_mask:.4f} | @conf=0.25 P={prec_c25*100:.1f}%, R={rec_c25*100:.1f}% | @conf=0.35 P={prec_c35*100:.1f}%, R={rec_c35*100:.1f}%")
        
        if map50_mask > best_overall_map50:
            best_overall_map50 = map50_mask
            best_weights_path = str(weights_fold)
            if weights_fold.exists():
                shutil.copy2(weights_fold, root_dir / "best_model.pt")
                print(f"[*] New best model saved to best_model.pt (mAP50={best_overall_map50:.4f})")
            
        results.append({
            "fold": fold,
            "mAP50_mask": round(map50_mask, 4),
            "mAP50_95_mask": round(map50_95_mask, 4),
            "precision_c25": round(prec_c25, 4),
            "recall_c25": round(rec_c25, 4),
            "precision_c35": round(prec_c35, 4),
            "recall_c35": round(rec_c35, 4),
            "box_mAP50": round(box_map50, 4),
            "box_mAP50_95": round(box_map50_95, 4),
            "best_weights": str(weights_fold)
        })

    # Summary report
    df_res = pd.DataFrame(results)
    out_csv = reports_dir / "kfold_v2_metrics_summary.csv"
    df_res.to_csv(out_csv, index=False)
    
    # Also save standard summary format
    stats_df = pd.DataFrame({
        "Metric": ["Mask mAP@0.50", "Mask mAP@0.50:0.95", "Precision @ conf=0.25", "Recall @ conf=0.25", "Precision @ conf=0.35", "Recall @ conf=0.35"],
        "Mean": [df_res["mAP50_mask"].mean(), df_res["mAP50_95_mask"].mean(), df_res["precision_c25"].mean(), df_res["recall_c25"].mean(), df_res["precision_c35"].mean(), df_res["recall_c35"].mean()],
        "Std": [df_res["mAP50_mask"].std(), df_res["mAP50_95_mask"].std(), df_res["precision_c25"].std(), df_res["recall_c25"].std(), df_res["precision_c35"].std(), df_res["recall_c35"].std()],
        "Min": [df_res["mAP50_mask"].min(), df_res["mAP50_95_mask"].min(), df_res["precision_c25"].min(), df_res["recall_c25"].min(), df_res["precision_c35"].min(), df_res["recall_c35"].min()],
        "Max": [df_res["mAP50_mask"].max(), df_res["mAP50_95_mask"].max(), df_res["precision_c25"].max(), df_res["recall_c25"].max(), df_res["precision_c35"].max(), df_res["recall_c35"].max()]
    })
    stats_df.to_csv(reports_dir / "kfold_v2_statistical_summary.csv", index=False)
    
    print("\n================ ИТОГОВЫЕ РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ V2 ================")
    print(df_res[["fold", "mAP50_mask", "mAP50_95_mask", "precision_c25", "recall_c25", "precision_c35", "recall_c35"]])
    print("\nСтатистика по метрикам:")
    print(stats_df.to_string(index=False))
    print(f"\nОтчет успешно сохранен в: {out_csv}")

    # Generate comparative boxplots
    try:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        axes[0].boxplot([df_res["mAP50_mask"], df_res["mAP50_95_mask"]], tick_labels=["mAP@0.50", "mAP@0.50:0.95"], patch_artist=True)
        axes[0].set_title("Mask mAP Distribution Across Folds (v2)")
        axes[0].set_ylabel("Score")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        
        axes[1].boxplot([df_res["precision_c25"], df_res["recall_c25"]], tick_labels=["Precision", "Recall"], patch_artist=True)
        axes[1].set_title("Operational Metrics at conf=0.25 (v2)")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        
        axes[2].boxplot([df_res["precision_c35"], df_res["recall_c35"]], tick_labels=["Precision", "Recall"], patch_artist=True)
        axes[2].set_title("Operational Metrics at conf=0.35 (v2)")
        axes[2].grid(True, linestyle="--", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(reports_dir / "kfold_v2_cv_metrics_boxplot.png", dpi=300)
        plt.close()
        print(f"[*] График сохранен в {reports_dir / 'kfold_v2_cv_metrics_boxplot.png'}")
    except Exception as e:
        print(f"[!] Warning generating boxplot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6, help="Number of epochs per fold")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size for training and validation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--splits", type=int, default=5, help="Number of KFold splits")
    args = parser.parse_args()
    
    run_kfold_v2(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, n_splits=args.splits)
