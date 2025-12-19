#!/usr/bin/env python3
"""
Generate publication-quality figures for VecOps-Bench VLDB 2026 paper.
Uses verified experimental results from December 2025 benchmarks.
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Verified experimental data from December 2025
CHURN_RESULTS = {
    'Milvus': {
        'cycles': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'recall': [97.84, 97.07, 96.28, 95.34, 94.57, 93.69, 92.96, 91.96, 91.03, 90.16, 89.28],
        'p50_ms': [15.48, 19.75, 19.13, 20.11, 19.32, 19.90, 20.82, 23.78, 19.93, 20.33, 20.95],
        'delete_s': [0, 21.55, 21.84, 23.84, 24.18, 21.32, 25.35, 23.19, 23.77, 21.51, 22.66],
        'insert_s': [0, 11.85, 15.34, 13.98, 14.84, 13.92, 14.64, 13.70, 14.74, 15.33, 14.86],
    },
    'pgvector': {
        'cycles': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'recall': [83.13, 82.47, 81.73, 80.87, 80.21, 79.50, 78.57, 77.57, 76.61, 75.85, 75.06],
        'p50_ms': [3.69, 3.74, 3.72, 3.67, 3.91, 3.96, 3.94, 4.07, 3.90, 4.03, 4.01],
        'delete_s': [0, 81.59, 79.16, 79.45, 72.48, 76.26, 111.99, 78.86, 137.65, 69.66, 70.50],
        'insert_s': [0, 1013.28, 1104.54, 1140.05, 1169.12, 1244.19, 1426.29, 1587.70, 1444.76, 1468.48, 1480.22],
    },
    'Chroma': {
        'cycles': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'recall': [88.99, 88.35, 87.58, 86.78, 86.14, 85.41, 84.87, 83.97, 83.11, 82.34, 81.61],
        'p50_ms': [2.90, 3.10, 3.13, 3.08, 3.01, 3.04, 3.04, 3.18, 3.00, 3.19, 3.05],
        'delete_s': [0, 700.96, 722.94, 743.03, 777.11, 630.93, 631.02, 648.04, 633.83, 639.73, 636.40],
        'insert_s': [0, 768.88, 753.44, 776.84, 721.06, 725.50, 736.18, 788.95, 835.63, 850.45, 860.48],
    },
    'Weaviate': {
        'cycles': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'recall': [81.80, 81.80, 81.80, 81.80, 81.80, 81.80, 81.68, 81.68, 80.91, 80.06, 79.06],
        'p50_ms': [248.52, 15.16, 2.86, 2.26, 2.15, 2.26, 2.20, 2.33, 3.14, 3.27, 71.94],
        'delete_s': [0, 80.68, 164.41, 58.89, 59.66, 62.30, 61.35, 59.08, 110.34, 168.15, 164.03],
        'insert_s': [0, 216.03, 40.30, 32.37, 32.57, 32.49, 32.06, 51.22, 51.70, 54.21, 510.02],
    },
}

SIFT1M_RESULTS = {
    'Milvus': {'recall_1': 98.38, 'recall_10': 98.75, 'recall_100': 97.14, 'p50_ms': 7.47},
    'Chroma': {'recall_1': 97.45, 'recall_10': 97.41, 'recall_100': 91.73, 'p50_ms': 1.23},
    'pgvector': {'recall_1': 95.86, 'recall_10': 94.41, 'recall_100': 40.64, 'p50_ms': 3.89},
}

# Colors for consistent styling
COLORS = {
    'Milvus': '#1f77b4',     # Blue
    'pgvector': '#ff7f0e',   # Orange
    'Chroma': '#2ca02c',     # Green
    'Weaviate': '#d62728',   # Red
    'Qdrant': '#9467bd',     # Purple
}

MARKERS = {
    'Milvus': 'o',
    'pgvector': 's',
    'Chroma': '^',
    'Weaviate': 'D',
}


def fig1_recall_degradation():
    """Figure 1: Recall@10 degradation over churn cycles (THE KEY FIGURE)"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for db, data in CHURN_RESULTS.items():
        ax.plot(data['cycles'], data['recall'],
                marker=MARKERS[db], markersize=6, linewidth=2,
                color=COLORS[db], label=db)

    ax.set_xlabel('Churn Cycle (100K deletes + 100K inserts per cycle)')
    ax.set_ylabel('Recall@10 (%)')
    ax.set_title('Recall Degradation Under Data Churn')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(70, 100)

    # Add degradation annotations
    for db, data in CHURN_RESULTS.items():
        degradation = data['recall'][0] - data['recall'][-1]
        ax.annotate(f'-{degradation:.1f}%',
                   xy=(10, data['recall'][-1]),
                   xytext=(10.3, data['recall'][-1]),
                   fontsize=9, color=COLORS[db])

    plt.tight_layout()
    plt.savefig('paper/figures/fig1_recall_degradation.pdf')
    plt.savefig('paper/figures/fig1_recall_degradation.png')
    plt.close()
    print('Generated: fig1_recall_degradation.pdf')


def fig2_degradation_bar():
    """Figure 2: Total degradation comparison (bar chart)"""
    fig, ax = plt.subplots(figsize=(6, 4))

    databases = list(CHURN_RESULTS.keys())
    degradations = [CHURN_RESULTS[db]['recall'][0] - CHURN_RESULTS[db]['recall'][-1]
                   for db in databases]
    colors = [COLORS[db] for db in databases]

    bars = ax.bar(databases, degradations, color=colors, edgecolor='black', linewidth=1)

    ax.set_ylabel('Recall Degradation (%)')
    ax.set_title('Total Recall Degradation After 10% Corpus Churn')
    ax.set_ylim(0, 12)

    # Add value labels
    for bar, val in zip(bars, degradations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('paper/figures/fig2_degradation_bar.pdf')
    plt.savefig('paper/figures/fig2_degradation_bar.png')
    plt.close()
    print('Generated: fig2_degradation_bar.pdf')


def fig3_churn_speed():
    """Figure 3: Churn operation speed comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))

    databases = list(CHURN_RESULTS.keys())

    # Average delete and insert times (excluding cycle 0)
    delete_times = []
    insert_times = []
    for db in databases:
        delete_times.append(np.mean(CHURN_RESULTS[db]['delete_s'][1:]))
        insert_times.append(np.mean(CHURN_RESULTS[db]['insert_s'][1:]))

    x = np.arange(len(databases))
    width = 0.35

    bars1 = ax.bar(x - width/2, delete_times, width, label='Delete 100K', color='#ff6b6b')
    bars2 = ax.bar(x + width/2, insert_times, width, label='Insert 100K', color='#4ecdc4')

    ax.set_ylabel('Time (seconds)')
    ax.set_title('Churn Operation Speed (per 100K vectors)')
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()
    ax.set_yscale('log')  # Log scale due to large differences

    plt.tight_layout()
    plt.savefig('paper/figures/fig3_churn_speed.pdf')
    plt.savefig('paper/figures/fig3_churn_speed.png')
    plt.close()
    print('Generated: fig3_churn_speed.pdf')


def fig4_sift1m_recall():
    """Figure 4: SIFT1M validation results"""
    fig, ax = plt.subplots(figsize=(8, 5))

    databases = list(SIFT1M_RESULTS.keys())
    x = np.arange(len(databases))
    width = 0.25

    recall_1 = [SIFT1M_RESULTS[db]['recall_1'] for db in databases]
    recall_10 = [SIFT1M_RESULTS[db]['recall_10'] for db in databases]
    recall_100 = [SIFT1M_RESULTS[db]['recall_100'] for db in databases]

    bars1 = ax.bar(x - width, recall_1, width, label='Recall@1', color='#1f77b4')
    bars2 = ax.bar(x, recall_10, width, label='Recall@10', color='#2ca02c')
    bars3 = ax.bar(x + width, recall_100, width, label='Recall@100', color='#ff7f0e')

    ax.set_ylabel('Recall (%)')
    ax.set_title('SIFT1M Validation Results')
    ax.set_xticks(x)
    ax.set_xticklabels(databases)
    ax.legend()
    ax.set_ylim(0, 105)

    # Highlight pgvector's low Recall@100
    ax.annotate('Low R@100', xy=(2 + width, 40.64),
               xytext=(2.3, 55), fontsize=9, color='red',
               arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    plt.savefig('paper/figures/fig4_sift1m_recall.pdf')
    plt.savefig('paper/figures/fig4_sift1m_recall.png')
    plt.close()
    print('Generated: fig4_sift1m_recall.pdf')


def fig5_latency_vs_recall():
    """Figure 5: Latency vs Recall trade-off at baseline"""
    fig, ax = plt.subplots(figsize=(6, 5))

    for db, data in CHURN_RESULTS.items():
        # Use baseline (cycle 0) values
        recall = data['recall'][0]
        # Use median p50 (excluding outliers like Weaviate's first cycle)
        p50 = np.median(data['p50_ms'][1:])  # Skip cycle 0 for more stable estimate

        ax.scatter(p50, recall, s=150, color=COLORS[db],
                  marker=MARKERS[db], label=db, edgecolors='black', linewidth=1)

    ax.set_xlabel('Median Query Latency (ms)')
    ax.set_ylabel('Baseline Recall@10 (%)')
    ax.set_title('Recall vs. Latency Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('paper/figures/fig5_latency_recall.pdf')
    plt.savefig('paper/figures/fig5_latency_recall.png')
    plt.close()
    print('Generated: fig5_latency_recall.pdf')


def main():
    # Create figures directory
    Path('paper/figures').mkdir(parents=True, exist_ok=True)

    print('Generating VLDB 2026 publication figures...')
    print('=' * 50)

    fig1_recall_degradation()  # THE KEY FIGURE
    fig2_degradation_bar()
    fig3_churn_speed()
    fig4_sift1m_recall()
    fig5_latency_vs_recall()

    print('=' * 50)
    print('All figures generated in paper/figures/')
    print()
    print('Summary of Key Results:')
    print('-' * 50)
    for db, data in CHURN_RESULTS.items():
        degradation = data['recall'][0] - data['recall'][-1]
        print(f'{db}: {data["recall"][0]:.2f}% -> {data["recall"][-1]:.2f}% ({degradation:.2f}% degradation)')


if __name__ == '__main__':
    main()
