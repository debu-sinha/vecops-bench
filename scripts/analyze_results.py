#!/usr/bin/env python3
"""
VectorDB-Bench: Results analysis and visualization.

Production-oriented analysis including:
- Standard recall/latency visualizations
- Temporal drift degradation curves (NOVEL)
- Cost-performance Pareto frontiers (NOVEL)
- Operational complexity radar charts (NOVEL)
- Multi-trial statistical analysis with confidence intervals
- Statistical significance testing (t-test, Mann-Whitney U)
- Effect size calculation (Cohen's d)

Author: Debu Sinha <debusinha2009@gmail.com>
Project: VectorDB-Bench - Production-Oriented Vector Database Benchmarking

Usage:
    python scripts/analyze_results.py --results results/combined_*.json
    python scripts/analyze_results.py --results results/ --output analysis/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cost import compute_pareto_frontier


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """Load benchmark results from file or directory."""
    path = Path(results_path)

    if path.is_file():
        with open(path) as f:
            data = json.load(f)
            return data if isinstance(data, list) else [data]

    elif path.is_dir():
        all_results = []
        for file in path.glob("*.json"):
            with open(file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)
        return all_results

    else:
        raise ValueError(f"Invalid path: {results_path}")


def results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to a flat DataFrame."""
    rows = []

    for result in results:
        if result.get("status") != "success":
            continue

        row = {
            "database": result.get("database"),
            "dataset": result.get("dataset"),
            "num_documents": result.get("num_documents"),
            "embedding_dim": result.get("embedding_dim"),
        }

        # Standard metrics
        standard = result.get("standard", {})
        for k, v in standard.get("recall", {}).items():
            row[f"recall{k}"] = v
        for k, v in standard.get("precision", {}).items():
            row[f"precision{k}"] = v
        for k, v in standard.get("ndcg", {}).items():
            row[f"ndcg{k}"] = v

        latency = standard.get("latency", {})
        row["latency_p50"] = latency.get("p50")
        row["latency_p95"] = latency.get("p95")
        row["latency_p99"] = latency.get("p99")
        row["latency_mean"] = latency.get("mean")

        # QPS
        qps_data = result.get("qps", {})
        row["qps"] = qps_data.get("qps")

        # Cold start
        cold_start = result.get("cold_start", {})
        row["cold_start_mean_ms"] = cold_start.get("mean_ms")

        # Index build
        index_build = result.get("index_build", {})
        row["build_time_seconds"] = index_build.get("total_build_time_seconds")
        row["vectors_per_second"] = index_build.get("vectors_per_second")
        row["memory_delta_mb"] = index_build.get("memory_delta_mb")

        # Filtered search
        filtered = result.get("filtered_search", {})
        if filtered:
            row["filter_overhead_percent"] = filtered.get("filter_overhead_percent")

        # Cost analysis (NOVEL)
        cost = result.get("cost_analysis", {})
        if cost:
            row["cost_per_million_queries"] = cost.get("cost_per_million_queries_usd")
            row["cost_per_recall_point"] = cost.get("cost_per_recall_point")
            row["queries_per_dollar"] = cost.get("queries_per_dollar")
            row["avg_cpu_percent"] = cost.get("avg_cpu_percent")
            row["peak_memory_mb"] = cost.get("peak_memory_mb")

        # Operational complexity (NOVEL)
        ops = result.get("operational_complexity", {})
        if ops and "error" not in ops:
            row["ops_deployment"] = ops.get("deployment_score")
            row["ops_configuration"] = ops.get("configuration_score")
            row["ops_monitoring"] = ops.get("monitoring_score")
            row["ops_maintenance"] = ops.get("maintenance_score")
            row["ops_documentation"] = ops.get("documentation_score")
            row["ops_overall"] = ops.get("overall_score")
            row["ops_team_size"] = ops.get("recommended_team_size")

        rows.append(row)

    return pd.DataFrame(rows)


def extract_drift_data(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Extract temporal drift data from results."""
    drift_data = {}

    for result in results:
        if result.get("status") != "success":
            continue

        db_name = result.get("database")
        drift = result.get("temporal_drift", {})

        if drift:
            drift_data[db_name] = drift

    return drift_data


def plot_recall_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot recall@k comparison across databases."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, k in enumerate(["@1", "@10", "@100"]):
        col = f"recall{k}"
        if col in df.columns:
            ax = axes[idx]
            data = df[["database", col]].dropna()
            sns.barplot(data=data, x="database", y=col, ax=ax, palette="viridis")
            ax.set_title(f"Recall{k}")
            ax.set_xlabel("")
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1)
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "recall_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_latency_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot latency comparison across databases."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for grouped bar chart
    latency_cols = ["latency_p50", "latency_p95", "latency_p99"]
    available_cols = [c for c in latency_cols if c in df.columns]

    if not available_cols:
        return

    data = df[["database"] + available_cols].melt(
        id_vars=["database"],
        var_name="percentile",
        value_name="latency_ms"
    )

    sns.barplot(data=data, x="database", y="latency_ms", hue="percentile", ax=ax, palette="rocket")
    ax.set_title("Query Latency by Database")
    ax.set_xlabel("")
    ax.set_ylabel("Latency (ms)")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Percentile")

    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_qps_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot QPS comparison across databases."""
    if "qps" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = df[["database", "qps"]].dropna()
    sns.barplot(data=data, x="database", y="qps", ax=ax, palette="mako")
    ax.set_title("Queries Per Second (QPS)")
    ax.set_xlabel("")
    ax.set_ylabel("QPS")
    ax.tick_params(axis='x', rotation=45)

    # Add value labels
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}',
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_dir / "qps_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cold_start_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot cold start latency comparison."""
    if "cold_start_mean_ms" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    data = df[["database", "cold_start_mean_ms"]].dropna()
    sns.barplot(data=data, x="database", y="cold_start_mean_ms", ax=ax, palette="flare")
    ax.set_title("Cold Start Latency")
    ax.set_xlabel("")
    ax.set_ylabel("Time to First Query (ms)")
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "cold_start_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_build_time_comparison(df: pd.DataFrame, output_dir: Path):
    """Plot index build time comparison."""
    if "build_time_seconds" not in df.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Build time
    data = df[["database", "build_time_seconds"]].dropna()
    sns.barplot(data=data, x="database", y="build_time_seconds", ax=axes[0], palette="crest")
    axes[0].set_title("Index Build Time")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Time (seconds)")
    axes[0].tick_params(axis='x', rotation=45)

    # Vectors per second
    if "vectors_per_second" in df.columns:
        data = df[["database", "vectors_per_second"]].dropna()
        sns.barplot(data=data, x="database", y="vectors_per_second", ax=axes[1], palette="crest")
        axes[1].set_title("Index Build Throughput")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Vectors/Second")
        axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "build_time_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path):
    """Plot recall vs latency Pareto frontier."""
    if "recall@10" not in df.columns or "latency_p50" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    data = df[["database", "recall@10", "latency_p50"]].dropna()

    # Scatter plot
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for idx, (_, row) in enumerate(data.iterrows()):
        ax.scatter(row["latency_p50"], row["recall@10"],
                  s=200, c=[colors[idx]], label=row["database"], alpha=0.7)

    ax.set_xlabel("Latency p50 (ms)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall vs Latency Trade-off")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Ideal point annotation
    ax.annotate("← Better", xy=(0.1, 0.95), xycoords='axes fraction', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "pareto_frontier.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# NOVEL VISUALIZATIONS - Key differentiators for the paper
# ============================================================================

def plot_temporal_drift_curves(
    drift_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """
    Plot temporal drift degradation curves (NOVEL CONTRIBUTION).

    Shows how recall degrades over time as corpus evolves.
    """
    if not drift_data:
        print("No drift data available, skipping drift plots")
        return

    # Create figure with subplots for each drift pattern
    patterns = set()
    for db_data in drift_data.values():
        patterns.update(db_data.keys())

    patterns = sorted(patterns)
    n_patterns = len(patterns)

    if n_patterns == 0:
        return

    fig, axes = plt.subplots(1, n_patterns, figsize=(7 * n_patterns, 6))
    if n_patterns == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, len(drift_data)))

    for pattern_idx, pattern in enumerate(patterns):
        ax = axes[pattern_idx]

        for db_idx, (db_name, db_data) in enumerate(drift_data.items()):
            if pattern not in db_data:
                continue

            pattern_data = db_data[pattern]
            if "error" in pattern_data:
                continue

            recall_curve = pattern_data.get("recall_curve", [])
            if not recall_curve:
                continue

            timestamps = list(range(len(recall_curve)))

            ax.plot(
                timestamps,
                recall_curve,
                marker='o',
                linewidth=2,
                markersize=6,
                color=colors[db_idx],
                label=db_name
            )

        ax.set_xlabel("Time Step (Corpus Evolution)")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"Recall Degradation - {pattern.replace('_', ' ').title()} Drift")
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "temporal_drift_curves.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also create a degradation rate comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    db_names = []
    degradation_rates = {p: [] for p in patterns}

    for db_name, db_data in drift_data.items():
        db_names.append(db_name)
        for pattern in patterns:
            if pattern in db_data and "error" not in db_data[pattern]:
                rate = db_data[pattern].get("recall_degradation_rate", 0)
                degradation_rates[pattern].append(rate)
            else:
                degradation_rates[pattern].append(0)

    x = np.arange(len(db_names))
    width = 0.8 / len(patterns)

    for idx, pattern in enumerate(patterns):
        offset = (idx - len(patterns) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            degradation_rates[pattern],
            width,
            label=pattern.replace('_', ' ').title()
        )

    ax.set_xlabel("Database")
    ax.set_ylabel("Recall Degradation Rate (per timestamp)")
    ax.set_title("Recall Degradation Rate Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(db_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "drift_degradation_rates.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cost_performance_pareto(df: pd.DataFrame, output_dir: Path):
    """
    Plot cost vs recall Pareto frontier (NOVEL CONTRIBUTION).

    Identifies optimal databases for different budget constraints.
    """
    if "cost_per_million_queries" not in df.columns or "recall@10" not in df.columns:
        print("No cost data available, skipping cost-performance plots")
        return

    data = df[["database", "recall@10", "cost_per_million_queries", "qps"]].dropna()

    if data.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Cost vs Recall Pareto
    ax1 = axes[0]

    # Compute Pareto frontier
    points = list(zip(data["recall@10"].tolist(), [-c for c in data["cost_per_million_queries"].tolist()]))
    pareto_indices = compute_pareto_frontier(points, maximize_x=True, maximize_y=True)

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, (_, row) in enumerate(data.iterrows()):
        is_pareto = idx in pareto_indices
        marker = '*' if is_pareto else 'o'
        size = 300 if is_pareto else 150

        ax1.scatter(
            row["cost_per_million_queries"],
            row["recall@10"],
            s=size,
            c=[colors[idx]],
            marker=marker,
            label=row["database"],
            alpha=0.8,
            edgecolors='black' if is_pareto else 'none',
            linewidths=2 if is_pareto else 0
        )

    ax1.set_xlabel("Cost ($/Million Queries)")
    ax1.set_ylabel("Recall@10")
    ax1.set_title("Cost vs Recall Pareto Frontier")
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Add annotation for Pareto optimal
    ax1.annotate(
        "★ = Pareto Optimal",
        xy=(0.02, 0.98),
        xycoords='axes fraction',
        fontsize=10,
        verticalalignment='top'
    )

    # Plot 2: Value Score (Recall*QPS / Cost)
    ax2 = axes[1]

    # Compute value score
    data["value_score"] = (data["recall@10"] * data["qps"]) / data["cost_per_million_queries"]
    data_sorted = data.sort_values("value_score", ascending=True)

    colors_sorted = [colors[data.index.get_loc(idx)] for idx in data_sorted.index]

    bars = ax2.barh(
        data_sorted["database"],
        data_sorted["value_score"],
        color=colors_sorted
    )

    ax2.set_xlabel("Value Score (Recall × QPS / Cost)")
    ax2.set_title("Cost-Efficiency Value Score")
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, val in zip(bars, data_sorted["value_score"]):
        ax2.text(
            bar.get_width() + 0.01 * data_sorted["value_score"].max(),
            bar.get_y() + bar.get_height() / 2,
            f'{val:.2f}',
            va='center',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(output_dir / "cost_performance_pareto.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_operational_complexity_radar(df: pd.DataFrame, output_dir: Path):
    """
    Plot operational complexity radar chart (NOVEL CONTRIBUTION).

    Visualizes multi-dimensional operational burden.
    """
    ops_cols = ["ops_deployment", "ops_configuration", "ops_monitoring",
                "ops_maintenance", "ops_documentation"]

    if not all(col in df.columns for col in ops_cols):
        print("No operational complexity data, skipping radar plot")
        return

    data = df[["database"] + ops_cols].dropna()

    if data.empty:
        return

    # Number of variables
    categories = ["Deployment", "Configuration", "Monitoring", "Maintenance", "Documentation"]
    N = len(categories)

    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for idx, (_, row) in enumerate(data.iterrows()):
        values = [row[col] for col in ops_cols]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[idx], label=row["database"])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)

    # Set radial limits
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)

    ax.set_title("Operational Complexity (Lower = Simpler)", size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_dir / "operational_complexity_radar.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Also create a stacked bar chart for team size recommendations
    fig, ax = plt.subplots(figsize=(12, 6))

    team_size_order = ["solo", "small", "medium", "large"]
    team_size_colors = {"solo": "#2ecc71", "small": "#3498db", "medium": "#f39c12", "large": "#e74c3c"}

    if "ops_team_size" in df.columns:
        team_data = df[["database", "ops_overall", "ops_team_size"]].dropna()
        team_data = team_data.sort_values("ops_overall")

        bars = ax.barh(
            team_data["database"],
            team_data["ops_overall"],
            color=[team_size_colors.get(ts, "#95a5a6") for ts in team_data["ops_team_size"]]
        )

        ax.set_xlabel("Overall Operational Complexity Score")
        ax.set_title("Operational Complexity by Database")
        ax.grid(True, alpha=0.3, axis='x')

        # Add legend
        legend_elements = [
            Patch(facecolor=team_size_colors[ts], label=f"{ts.title()} Team")
            for ts in team_size_order if ts in team_data["ops_team_size"].values
        ]
        ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(output_dir / "operational_complexity_bars.png", dpi=150, bbox_inches='tight')
        plt.close()


def plot_comprehensive_comparison(df: pd.DataFrame, output_dir: Path):
    """
    Create a comprehensive multi-metric comparison plot.

    This is the key figure for the paper.
    """
    # Determine which metrics are available
    metric_configs = [
        ("recall@10", "Recall@10", True),  # (column, label, higher_is_better)
        ("latency_p50", "Latency p50 (ms)", False),
        ("qps", "QPS", True),
        ("cold_start_mean_ms", "Cold Start (ms)", False),
    ]

    # Add novel metrics if available
    if "cost_per_million_queries" in df.columns:
        metric_configs.append(("cost_per_million_queries", "$/M Queries", False))
    if "ops_overall" in df.columns:
        metric_configs.append(("ops_overall", "Ops Complexity", False))

    available_metrics = [(col, label, hib) for col, label, hib in metric_configs if col in df.columns]

    if len(available_metrics) < 2:
        return

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(5 * ((n_metrics + 1) // 2), 10))
    axes = axes.flatten()

    for idx, (col, label, higher_is_better) in enumerate(available_metrics):
        ax = axes[idx]

        data = df[["database", col]].dropna()
        data_sorted = data.sort_values(col, ascending=not higher_is_better)

        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(data_sorted))]

        bars = ax.barh(data_sorted["database"], data_sorted[col], color=colors)
        ax.set_xlabel(label)
        ax.set_title(f"{'↑' if higher_is_better else '↓'} {label}")
        ax.grid(True, alpha=0.3, axis='x')

    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Comprehensive Database Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_latex_table(df: pd.DataFrame, output_dir: Path):
    """Generate LaTeX table for paper."""
    # Select key columns
    cols = ["database", "recall@10", "latency_p50", "latency_p95", "qps",
            "cold_start_mean_ms", "build_time_seconds"]
    available_cols = [c for c in cols if c in df.columns]

    table_df = df[available_cols].copy()

    # Rename for paper
    rename_map = {
        "database": "Database",
        "recall@10": "Recall@10",
        "latency_p50": "Latency p50 (ms)",
        "latency_p95": "Latency p95 (ms)",
        "qps": "QPS",
        "cold_start_mean_ms": "Cold Start (ms)",
        "build_time_seconds": "Build Time (s)"
    }
    table_df = table_df.rename(columns=rename_map)

    # Generate LaTeX
    latex = table_df.to_latex(index=False, float_format="%.2f", escape=False)

    with open(output_dir / "results_table.tex", 'w') as f:
        f.write(latex)

    print(f"LaTeX table saved to: {output_dir / 'results_table.tex'}")

    # Generate extended table with novel metrics
    extended_cols = ["database", "recall@10", "latency_p50", "qps",
                     "cost_per_million_queries", "ops_overall"]
    available_extended = [c for c in extended_cols if c in df.columns]

    if len(available_extended) > 3:  # Has novel metrics
        extended_df = df[available_extended].copy()

        extended_rename = {
            "database": "Database",
            "recall@10": "Recall@10",
            "latency_p50": "Latency (ms)",
            "qps": "QPS",
            "cost_per_million_queries": "\\$/M Queries",
            "ops_overall": "Ops Score"
        }
        extended_df = extended_df.rename(columns=extended_rename)

        latex_ext = extended_df.to_latex(index=False, float_format="%.2f", escape=False)

        with open(output_dir / "results_table_extended.tex", 'w') as f:
            f.write(latex_ext)

        print(f"Extended LaTeX table saved to: {output_dir / 'results_table_extended.tex'}")


def generate_summary_report(
    df: pd.DataFrame,
    drift_data: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Generate markdown summary report."""
    report = []
    report.append("# VectorDB-Bench Results Summary\n")

    # Overview
    report.append("## Overview\n")
    report.append(f"- **Databases tested:** {', '.join(df['database'].unique())}")
    report.append(f"- **Dataset:** {df['dataset'].iloc[0] if 'dataset' in df.columns else 'N/A'}")
    report.append(f"- **Documents:** {df['num_documents'].iloc[0] if 'num_documents' in df.columns else 'N/A'}")
    report.append("")

    # Best performers
    report.append("## Key Findings\n")

    if "recall@10" in df.columns:
        best_recall = df.loc[df["recall@10"].idxmax()]
        report.append(f"- **Best Recall@10:** {best_recall['database']} ({best_recall['recall@10']:.4f})")

    if "latency_p50" in df.columns:
        best_latency = df.loc[df["latency_p50"].idxmin()]
        report.append(f"- **Lowest Latency (p50):** {best_latency['database']} ({best_latency['latency_p50']:.2f} ms)")

    if "qps" in df.columns:
        best_qps = df.loc[df["qps"].idxmax()]
        report.append(f"- **Highest QPS:** {best_qps['database']} ({best_qps['qps']:.1f})")

    if "cold_start_mean_ms" in df.columns:
        best_cold = df.loc[df["cold_start_mean_ms"].idxmin()]
        report.append(f"- **Fastest Cold Start:** {best_cold['database']} ({best_cold['cold_start_mean_ms']:.2f} ms)")

    # Novel metrics
    if "cost_per_million_queries" in df.columns:
        best_cost = df.loc[df["cost_per_million_queries"].idxmin()]
        report.append(f"- **Most Cost-Efficient:** {best_cost['database']} (${best_cost['cost_per_million_queries']:.2f}/M queries)")

    if "ops_overall" in df.columns:
        simplest = df.loc[df["ops_overall"].idxmin()]
        report.append(f"- **Simplest Operations:** {simplest['database']} (score: {simplest['ops_overall']:.0f})")

    report.append("")

    # Temporal Drift Analysis (NOVEL)
    if drift_data:
        report.append("## Temporal Drift Analysis (Novel)\n")
        report.append("This analysis measures how recall degrades as the corpus evolves over time.\n")

        for db_name, patterns in drift_data.items():
            report.append(f"### {db_name}\n")
            for pattern, data in patterns.items():
                if "error" in data:
                    report.append(f"- **{pattern}:** Error - {data['error']}")
                else:
                    degradation = data.get("recall_degradation_rate", 0)
                    half_life = data.get("recall_half_life", "N/A")
                    initial = data.get("initial_recall", 0)
                    final = data.get("final_recall", 0)
                    report.append(f"- **{pattern.replace('_', ' ').title()} Drift:**")
                    report.append(f"  - Initial Recall: {initial:.4f}")
                    report.append(f"  - Final Recall: {final:.4f}")
                    report.append(f"  - Degradation Rate: {degradation:.4f} per timestamp")
                    report.append(f"  - Half-Life: {half_life}")
            report.append("")

    # Cost Analysis (NOVEL)
    if "cost_per_million_queries" in df.columns:
        report.append("## Cost-Efficiency Analysis (Novel)\n")
        cost_cols = ["database", "cost_per_million_queries", "queries_per_dollar", "recall@10", "qps"]
        available = [c for c in cost_cols if c in df.columns]
        if available:
            report.append(df[available].to_markdown(index=False))
        report.append("")

    # Operational Complexity (NOVEL)
    if "ops_overall" in df.columns:
        report.append("## Operational Complexity Analysis (Novel)\n")
        report.append("Lower scores indicate simpler operations.\n")
        ops_cols = ["database", "ops_deployment", "ops_configuration", "ops_monitoring",
                   "ops_maintenance", "ops_documentation", "ops_overall", "ops_team_size"]
        available = [c for c in ops_cols if c in df.columns]
        if available:
            report.append(df[available].to_markdown(index=False))
        report.append("")

    # Full results table
    report.append("## Full Results\n")
    report.append(df.to_markdown(index=False))

    with open(output_dir / "summary_report.md", 'w') as f:
        f.write("\n".join(report))

    print(f"Summary report saved to: {output_dir / 'summary_report.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="VectorDB-Bench: Production-Oriented Results Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python scripts/analyze_results.py --results results/combined_*.json

  # Full analysis with all visualizations
  python scripts/analyze_results.py --results results/ --output analysis/
        """
    )
    parser.add_argument("--results", type=str, required=True, help="Path to results file or directory")
    parser.add_argument("--output", type=str, default="./analysis", help="Output directory for plots")
    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results}")
    results = load_results(args.results)
    print(f"Loaded {len(results)} benchmark results")

    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"Converted to DataFrame with {len(df)} rows")

    # Extract drift data
    drift_data = extract_drift_data(results)
    print(f"Found drift data for {len(drift_data)} databases")

    if df.empty:
        print("No successful results to analyze!")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.dpi'] = 150

    # Generate standard plots
    print("\nGenerating standard plots...")
    plot_recall_comparison(df, output_dir)
    plot_latency_comparison(df, output_dir)
    plot_qps_comparison(df, output_dir)
    plot_cold_start_comparison(df, output_dir)
    plot_build_time_comparison(df, output_dir)
    plot_pareto_frontier(df, output_dir)

    # Generate NOVEL plots (key differentiators)
    print("\nGenerating novel analysis plots...")
    plot_temporal_drift_curves(drift_data, output_dir)
    plot_cost_performance_pareto(df, output_dir)
    plot_operational_complexity_radar(df, output_dir)
    plot_comprehensive_comparison(df, output_dir)

    # Generate reports
    print("\nGenerating reports...")
    generate_latex_table(df, output_dir)
    generate_summary_report(df, drift_data, output_dir)

    # Save processed data
    df.to_csv(output_dir / "results_processed.csv", index=False)
    print(f"Processed data saved to: {output_dir / 'results_processed.csv'}")

    # Print summary of generated files
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")

    print(f"\nNovel contributions highlighted:")
    print("  - temporal_drift_curves.png: Recall degradation over corpus evolution")
    print("  - drift_degradation_rates.png: Degradation rate comparison")
    print("  - cost_performance_pareto.png: Cost-efficiency Pareto frontier")
    print("  - operational_complexity_radar.png: Multi-dimensional ops burden")
    print("  - comprehensive_comparison.png: All metrics side-by-side")


if __name__ == "__main__":
    main()
