"""
train_with_focal.py - Шаг 2 спецификации v4.
Обучение YOLOv8n-seg с кастомными Loss Gains (loss_gains_tuning.yaml) и Focal Loss для борьбы с дисбалансом классов ТЭК.
Подавление легких градиентов фонового металла и акцент на микротрещинах (crack).
"""

from train_with_focal import train_focal_model

if __name__ == '__main__':
    train_focal_model()
