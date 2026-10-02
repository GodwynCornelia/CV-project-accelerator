"""
Generates balanced defect distribution comparison visualization (V1 vs V2).
Saves to reports/defect_distribution.png and reports/defect_distribution_v2.png.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_distribution_chart():
    os.makedirs("reports", exist_ok=True)
    
    classes = ['Corrosion (0)', 'Crack (1)', 'Coating Damage (2)']
    v1_polygons = [100, 31, 30]
    v2_polygons = [106, 99, 92]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # 1. Bar Chart: Polygon count comparison
    x = np.arange(len(classes))
    width = 0.35
    
    rects1 = axes[0].bar(x - width/2, v1_polygons, width, label='V1 Imbalanced (115 imgs)', color='#d9534f', alpha=0.85)
    rects2 = axes[0].bar(x + width/2, v2_polygons, width, label='V2 Balanced (275 imgs)', color='#2e6da4', alpha=0.85)
    
    axes[0].set_ylabel('Number of Segmentation Polygons')
    axes[0].set_title('Defect Class Polygon Counts: V1 vs V2 Expansion')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        axes[0].annotate(f'{h}',
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        axes[0].annotate(f'{h}',
                         xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontweight='bold', color='#1f497d')

    # 2. Donut Chart: V2 Defect Distribution
    colors = ['#dc6e28', '#e60000', '#14b4dc']
    explode = (0.02, 0.02, 0.02)
    wedges, texts, autotexts = axes[1].pie(
        v2_polygons, labels=classes, autopct='%1.1f%%',
        startangle=140, colors=colors, explode=explode,
        wedgeprops=dict(width=0.45, edgecolor='w')
    )
    for at in autotexts:
        at.set_color('white')
        at.set_weight('bold')
    axes[1].set_title('V2 Balanced Defect Ratio (~33% per class)\nClean BG: 39 images (14.2%)')
    
    plt.tight_layout()
    plt.savefig('reports/defect_distribution.png', dpi=300)
    plt.savefig('reports/defect_distribution_v2.png', dpi=300)
    plt.close()
    print("[*] Defect distribution charts generated successfully.")

if __name__ == "__main__":
    generate_comparison_distribution_chart()
