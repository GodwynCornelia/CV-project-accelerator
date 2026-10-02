"""
pseudo_labeling.py - Шаг 3 спецификации v4.
Пайплайн полуавтоматической псевдоразметки (Pseudo-Labeling / Self-Training).
"""

from pseudo_labeling import generate_pseudo_labels

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_model.pt")
    parser.add_argument("--input", type=str, default="datasets/unlabelled_uav_frames")
    parser.add_argument("--output", type=str, default="datasets/augmented_with_pseudo")
    parser.add_argument("--conf", type=float, default=0.85)
    args = parser.parse_args()
    
    generate_pseudo_labels(args.weights, args.input, args.output, args.conf)
