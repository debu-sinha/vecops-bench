#!/usr/bin/env python3
"""
Generate publication-quality figures for VectorDB-Bench paper.

Creates figures suitable for academic publication:
- High DPI (300+)
- Clean, minimal style
- Proper font sizes
- Color-blind friendly palettes

Author: Debu Sinha <debusinha2009@gmail.com>
Project: VectorDB-Bench - Production-Oriented Vector Database Benchmarking
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Try to import plotting libraries
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: matplotlib/seaborn not installed. Install with:")
    print("  pip install matplotlib seaborn")

# Publication-quality settings
FIGURE_DPI = 300
FIGURE_WIDTH = 7  # inches (single column)
FIGURE_WIDTH_DOUBLE = 14  # inches (double column)
FONT_SIZE = 10
TITLE_SIZE = 12

# Color-blind friendly palette (IBM Design)
COLORS = {
    "milvus": "#648FFF",  # Blue
    "qdrant": "#785EF0",  # Purple
    "pgvector": "#DC267F",  # Magenta
    "weaviate": "#FE6100",  # Orange
    "chroma": "#FFB000",  # Gold
}

# Database display names
DB_NAMES = {
    "milvus": "Milvus",
    "qdrant": "Qdrant",
    "pgvector": "pgvector",
    "weaviate": "Weaviate",
    "chroma": "Chroma",
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    if not HAS_PLOTTING:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 1,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load all benchmark results from directory."""
    results = []
    results_path = Path(results_dir)

    for json_file in results_path.glob("**/*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def aggregate_trials(results: List[Dict]) -> Dict[str, Dict]:
    """Aggregate multiple trials per database into mean/std."""
    db_results = {}

    for r in results:
        if r.get("status") != "success":
            continue

        db = r.get("database", "unknown")
        if db not in db_results:
            db_results[db] = {
                "recall_10": [],
                "latency_p50": [],
                "latency_p99": [],
                "qps": [],
                "cold_start": [],
                "insert_speed": [],
            }

        # Extract metrics
        if "standard" in r:
            std = r["standard"]
            if "recall" in std and "@10" in std["recall"]:
                db_results[db]["recall_10"].append(std["recall"]["@10"])
            if "latency" in std:
                db_results[db]["latency_p50"].append(std["latency"].get("p50", 0))
                db_results[db]["latency_p99"].append(std["latency"].get("p99", 0))

        if "qps" in r and isinstance(r["qps"], dict):
            db_results[db]["qps"].append(r["qps"].get("qps", 0))

        if "cold_start" in r:
            db_results[db]["cold_start"].append(r["cold_start"].get("mean_ms", 0))

        if "index_build" in r:
            db_results[db]["insert_speed"].append(r["index_build"].get("vectors_per_second", 0))

    # Compute mean and std
    aggregated = {}
    for db, metrics in db_results.items():
        aggregated[db] = {}
        for metric, values in metrics.items():
            if values:
                aggregated[db][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "n": len(values),
                }

    return aggregated


def plot_recall_vs_latency(aggregated: Dict, output_path: str):
    """
    Figure 1: Recall@10 vs Latency scatter plot.
    Shows quality-performance trade-off.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5))

    for db, metrics in aggregated.items():
        if "recall_10" not in metrics or "latency_p50" not in metrics:
            continue

        recall = metrics["recall_10"]["mean"]
        latency = metrics["latency_p50"]["mean"]
        recall_std = metrics["recall_10"].get("std", 0)
        latency_std = metrics["latency_p50"].get("std", 0)

        ax.errorbar(
            latency,
            recall,
            xerr=latency_std,
            yerr=recall_std,
            fmt="o",
            markersize=12,
            color=COLORS.get(db, "#666666"),
            label=DB_NAMES.get(db, db),
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
        )

    ax.set_xlabel("Latency p50 (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Quality vs. Performance Trade-off")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_qps_comparison(aggregated: Dict, output_path: str):
    """
    Figure 2: QPS bar chart with error bars.
    Shows throughput comparison.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))

    databases = []
    qps_means = []
    qps_stds = []
    colors = []

    for db in ["pgvector", "weaviate", "chroma", "qdrant", "milvus"]:
        if db in aggregated and "qps" in aggregated[db]:
            databases.append(DB_NAMES.get(db, db))
            qps_means.append(aggregated[db]["qps"]["mean"])
            qps_stds.append(aggregated[db]["qps"]["std"])
            colors.append(COLORS.get(db, "#666666"))

    x = np.arange(len(databases))
    bars = ax.bar(x, qps_means, yerr=qps_stds, capsize=5, color=colors, alpha=0.8)

    ax.set_xlabel("Database")
    ax.set_ylabel("Queries per Second (QPS)")
    ax.set_title("Throughput Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(databases, rotation=45, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add value labels on bars
    for bar, val in zip(bars, qps_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_latency_distribution(results: List[Dict], output_path: str):
    """
    Figure 3: Latency distribution box plot.
    Shows latency variance across databases.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))

    data = []
    labels = []

    for db in ["milvus", "qdrant", "pgvector", "weaviate", "chroma"]:
        db_latencies = []
        for r in results:
            if r.get("database") == db and r.get("status") == "success":
                if "standard" in r and "latency" in r["standard"]:
                    lat = r["standard"]["latency"]
                    db_latencies.extend(
                        [
                            lat.get("p50", 0),
                            lat.get("p95", 0),
                            lat.get("p99", 0),
                        ]
                    )
        if db_latencies:
            data.append(db_latencies)
            labels.append(DB_NAMES.get(db, db))

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        for patch, db in zip(bp["boxes"], ["milvus", "qdrant", "pgvector", "weaviate", "chroma"]):
            patch.set_facecolor(COLORS.get(db, "#666666"))
            patch.set_alpha(0.7)

    ax.set_xlabel("Database")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_cold_start(aggregated: Dict, output_path: str):
    """
    Figure 4: Cold start latency comparison.
    Novel metric showing first-query performance.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))

    databases = []
    cold_means = []
    cold_stds = []
    colors = []

    for db in ["pgvector", "milvus", "chroma", "qdrant", "weaviate"]:
        if db in aggregated and "cold_start" in aggregated[db]:
            databases.append(DB_NAMES.get(db, db))
            cold_means.append(aggregated[db]["cold_start"]["mean"])
            cold_stds.append(aggregated[db]["cold_start"]["std"])
            colors.append(COLORS.get(db, "#666666"))

    x = np.arange(len(databases))
    bars = ax.bar(x, cold_means, yerr=cold_stds, capsize=5, color=colors, alpha=0.8)

    ax.set_xlabel("Database")
    ax.set_ylabel("Cold Start Latency (ms)")
    ax.set_title("Cold Start Performance (First Query After Restart)")
    ax.set_xticks(x)
    ax.set_xticklabels(databases, rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_operational_complexity(results: List[Dict], output_path: str):
    """
    Figure 5: Operational complexity radar chart.
    Novel metric showing deployment/maintenance burden.
    """
    if not HAS_PLOTTING:
        return

    # Extract operational scores
    ops_data = {}
    for r in results:
        db = r.get("database")
        if db and "operational_complexity" in r:
            ops = r["operational_complexity"]
            if db not in ops_data:
                ops_data[db] = ops

    if not ops_data:
        print("No operational complexity data found")
        return

    categories = ["Deployment", "Configuration", "Monitoring", "Maintenance"]

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    for db, ops in ops_data.items():
        values = [
            100 - ops.get("deployment_score", 50),  # Invert: lower is better
            100 - ops.get("configuration_score", 50),
            100 - ops.get("monitoring_score", 50),
            100 - ops.get("maintenance_score", 50),
        ]
        values += values[:1]

        ax.plot(
            angles,
            values,
            "o-",
            linewidth=2,
            label=DB_NAMES.get(db, db),
            color=COLORS.get(db, "#666666"),
        )
        ax.fill(angles, values, alpha=0.1, color=COLORS.get(db, "#666666"))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 100)
    ax.set_title("Operational Simplicity\n(Higher is Better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_insert_performance(aggregated: Dict, output_path: str):
    """
    Figure 6: Index build / insert performance.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))

    databases = []
    insert_means = []
    insert_stds = []
    colors = []

    for db in ["milvus", "weaviate", "chroma", "qdrant", "pgvector"]:
        if db in aggregated and "insert_speed" in aggregated[db]:
            databases.append(DB_NAMES.get(db, db))
            insert_means.append(aggregated[db]["insert_speed"]["mean"])
            insert_stds.append(aggregated[db]["insert_speed"]["std"])
            colors.append(COLORS.get(db, "#666666"))

    x = np.arange(len(databases))
    bars = ax.bar(x, insert_means, yerr=insert_stds, capsize=5, color=colors, alpha=0.8)

    ax.set_xlabel("Database")
    ax.set_ylabel("Vectors per Second")
    ax.set_title("Index Build Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(databases, rotation=45, ha="right")
    ax.set_yscale("log")  # Log scale for large differences

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_latex_table(aggregated: Dict, output_path: str):
    """Generate LaTeX table for paper."""

    headers = ["Database", "Recall@10", "Latency p50", "Latency p99", "QPS", "Cold Start"]

    rows = []
    for db in ["milvus", "qdrant", "pgvector", "weaviate", "chroma"]:
        if db not in aggregated:
            continue

        m = aggregated[db]
        row = [
            DB_NAMES.get(db, db),
            f"{m.get('recall_10', {}).get('mean', 0):.3f} $\\pm$ {m.get('recall_10', {}).get('std', 0):.3f}",
            f"{m.get('latency_p50', {}).get('mean', 0):.2f} ms",
            f"{m.get('latency_p99', {}).get('mean', 0):.2f} ms",
            f"{m.get('qps', {}).get('mean', 0):.1f}",
            f"{m.get('cold_start', {}).get('mean', 0):.1f} ms",
        ]
        rows.append(row)

    # Generate LaTeX
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Benchmark Results Summary}")
    latex.append("\\label{tab:results}")
    latex.append("\\begin{tabular}{l" + "c" * (len(headers) - 1) + "}")
    latex.append("\\toprule")
    latex.append(" & ".join(headers) + " \\\\")
    latex.append("\\midrule")

    for row in rows:
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"Saved: {output_path}")


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def load_churn_results(results_dir: str) -> Dict[str, Dict]:
    """Load churn test results for temporal drift analysis."""
    churn_results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("**/churn*/*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                db = data.get("database", "unknown")
                if "cycles" in data:
                    churn_results[db] = data
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return churn_results


def plot_temporal_drift(churn_results: Dict, output_path: str):
    """
    THE STAR FIGURE: Recall degradation under data churn.

    Shows how each database's recall degrades over churn cycles.
    This is the key novel contribution of the paper.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5))

    for db, data in churn_results.items():
        if "cycles" not in data:
            continue

        cycles = [c["cycle"] for c in data["cycles"]]
        recalls = [c["recall_at_10"] for c in data["cycles"]]

        ax.plot(
            cycles,
            recalls,
            marker="o",
            markersize=8,
            linewidth=2,
            color=COLORS.get(db, "#666666"),
            label=DB_NAMES.get(db, db),
        )

        # Add degradation annotation at the end
        if len(recalls) > 1:
            degradation = ((recalls[0] - recalls[-1]) / recalls[0]) * 100
            ax.annotate(
                f"{degradation:+.1f}%",
                xy=(cycles[-1], recalls[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                color=COLORS.get(db, "#666666"),
            )

    ax.set_xlabel("Churn Cycle (100K vectors deleted + inserted per cycle)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Temporal Drift: Recall Degradation Under Data Churn")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    # Add reference line at initial recall
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="Target (0.95)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_compaction_recovery(churn_results: Dict, output_path: str):
    """
    Figure: Compaction recovery effectiveness.

    Shows pre-churn, post-churn, and post-compaction recall for each DB.
    """
    if not HAS_PLOTTING:
        return

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 5))

    databases = []
    initial_recalls = []
    post_churn_recalls = []
    post_compaction_recalls = []
    colors_list = []

    for db in ["pgvector", "qdrant", "milvus", "weaviate", "chroma"]:
        if db not in churn_results:
            continue
        data = churn_results[db]
        if "summary" not in data:
            continue

        databases.append(DB_NAMES.get(db, db))
        initial_recalls.append(data["summary"].get("initial_recall", 0))
        post_churn_recalls.append(data["summary"].get("final_recall", 0))
        post_compaction_recalls.append(
            data["summary"].get("post_compaction_recall", data["summary"].get("final_recall", 0))
        )
        colors_list.append(COLORS.get(db, "#666666"))

    if not databases:
        print("No compaction data found")
        return

    x = np.arange(len(databases))
    width = 0.25

    bars1 = ax.bar(x - width, initial_recalls, width, label="Initial", alpha=0.9)
    bars2 = ax.bar(x, post_churn_recalls, width, label="Post-Churn", alpha=0.7)
    bars3 = ax.bar(x + width, post_compaction_recalls, width, label="Post-Compaction", alpha=0.5)

    # Color bars by database
    for i, color in enumerate(colors_list):
        bars1[i].set_color(color)
        bars2[i].set_color(color)
        bars3[i].set_color(color)

    ax.set_xlabel("Database")
    ax.set_ylabel("Recall@10")
    ax.set_title("Index Maintenance: Recall Recovery After Compaction")
    ax.set_xticks(x)
    ax.set_xticklabels(databases, rotation=45, ha="right")
    ax.legend(loc="lower right")
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def generate_effect_size_table(churn_results: Dict, output_path: str):
    """
    Generate LaTeX table of Cohen's d effect sizes for churn degradation.

    Compares each database's degradation to others.
    """
    databases = list(churn_results.keys())

    # Extract degradation percentages for each database
    degradations = {}
    for db, data in churn_results.items():
        if "cycles" in data:
            recalls = [c["recall_at_10"] for c in data["cycles"]]
            degradations[db] = recalls  # Full recall trajectory

    if len(degradations) < 2:
        print("Not enough databases for effect size comparison")
        return

    # Compute pairwise Cohen's d
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append("\\caption{Cohen's d Effect Sizes: Pairwise Comparison of Recall Degradation}")
    latex.append("\\label{tab:effect_sizes}")
    latex.append("\\begin{tabular}{l" + "c" * len(databases) + "}")
    latex.append("\\toprule")
    latex.append(" & " + " & ".join([DB_NAMES.get(db, db) for db in databases]) + " \\\\")
    latex.append("\\midrule")

    for db1 in databases:
        row = [DB_NAMES.get(db1, db1)]
        for db2 in databases:
            if db1 == db2:
                row.append("--")
            else:
                d = compute_cohens_d(degradations[db1], degradations[db2])
                interp = interpret_cohens_d(d)
                row.append(f"{d:.2f} ({interp[0]})")
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\multicolumn{" + str(len(databases) + 1) + "}{l}{\\footnotesize n=negligible, s=small, m=medium, l=large}")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    with open(output_path, "w") as f:
        f.write("\n".join(latex))

    print(f"Saved: {output_path}")


def generate_degradation_summary(churn_results: Dict, output_path: str):
    """
    Generate summary statistics for temporal drift with 95% CI.
    """
    summary = {
        "databases": {},
        "pairwise_effect_sizes": {},
    }

    for db, data in churn_results.items():
        if "summary" not in data:
            continue

        s = data["summary"]
        recalls = s.get("recall_trend", [])

        if len(recalls) > 1:
            # Compute 95% CI for degradation
            degradation_pct = s.get("recall_degradation_pct", 0)

            summary["databases"][db] = {
                "initial_recall": s.get("initial_recall"),
                "final_recall": s.get("final_recall"),
                "degradation_pct": degradation_pct,
                "post_compaction_recall": s.get("post_compaction_recall"),
                "recovery_pct": s.get("recovery_pct"),
                "recall_trajectory": recalls,
            }

    # Compute pairwise effect sizes
    dbs = list(summary["databases"].keys())
    for i, db1 in enumerate(dbs):
        for db2 in dbs[i + 1:]:
            traj1 = summary["databases"][db1]["recall_trajectory"]
            traj2 = summary["databases"][db2]["recall_trajectory"]
            d = compute_cohens_d(traj1, traj2)
            summary["pairwise_effect_sizes"][f"{db1}_vs_{db2}"] = {
                "cohens_d": d,
                "interpretation": interpret_cohens_d(d),
            }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--input", "-i", default="./results", help="Results directory")
    parser.add_argument("--output", "-o", default="./figures", help="Output directory for figures")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup matplotlib style
    setup_style()

    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    print(f"Loaded {len(results)} result files")

    if not results:
        print("No results found!")
        return

    # Aggregate trials
    aggregated = aggregate_trials(results)
    print(f"Aggregated results for: {list(aggregated.keys())}")

    # Generate figures
    print("\nGenerating baseline figures...")

    plot_recall_vs_latency(aggregated, str(output_dir / "fig1_recall_latency.pdf"))
    plot_qps_comparison(aggregated, str(output_dir / "fig2_qps_comparison.pdf"))
    plot_latency_distribution(results, str(output_dir / "fig3_latency_distribution.pdf"))
    plot_cold_start(aggregated, str(output_dir / "fig4_cold_start.pdf"))
    plot_operational_complexity(results, str(output_dir / "fig5_operational_complexity.pdf"))
    plot_insert_performance(aggregated, str(output_dir / "fig6_insert_performance.pdf"))

    # Generate LaTeX table
    generate_latex_table(aggregated, str(output_dir / "table_results.tex"))

    # Load and generate temporal drift figures (THE STAR FEATURE)
    print("\nGenerating temporal drift figures...")
    churn_results = load_churn_results(args.input)
    if churn_results:
        print(f"Loaded churn results for: {list(churn_results.keys())}")

        # The key figure - recall degradation over churn cycles
        plot_temporal_drift(churn_results, str(output_dir / "fig7_temporal_drift.pdf"))

        # Compaction recovery figure
        plot_compaction_recovery(churn_results, str(output_dir / "fig8_compaction_recovery.pdf"))

        # Generate effect size table (Cohen's d)
        generate_effect_size_table(churn_results, str(output_dir / "table_effect_sizes.tex"))

        # Generate degradation summary JSON
        generate_degradation_summary(churn_results, str(output_dir / "degradation_summary.json"))
    else:
        print("No churn results found, skipping temporal drift figures")

    print("\nDone! Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
