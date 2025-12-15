"""
Cost Analysis for Vector Database Benchmarking.

Computes cost-normalized metrics and Pareto frontier analysis
for comparing vector databases on cost-efficiency.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CostEfficiencyMetrics:
    """Cost-efficiency metrics for a database."""
    database: str

    # Raw performance
    recall_at_10: float
    latency_p50_ms: float
    qps: float

    # Raw cost
    cost_per_million_queries_usd: float

    # Cost-normalized metrics
    recall_per_dollar: float          # Recall points per $
    qps_per_dollar: float             # QPS per $
    latency_cost_product: float       # Lower is better: latency_ms * $/query

    # Value metrics
    value_score: float                # Composite value metric
    is_pareto_optimal: bool = False   # On the cost-quality Pareto frontier

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "database": self.database,
            "recall_at_10": self.recall_at_10,
            "latency_p50_ms": self.latency_p50_ms,
            "qps": self.qps,
            "cost_per_million_queries_usd": self.cost_per_million_queries_usd,
            "recall_per_dollar": self.recall_per_dollar,
            "qps_per_dollar": self.qps_per_dollar,
            "latency_cost_product": self.latency_cost_product,
            "value_score": self.value_score,
            "is_pareto_optimal": self.is_pareto_optimal,
        }


class CostAnalyzer:
    """
    Analyzes cost-efficiency across multiple database benchmarks.

    Key analysis:
    1. Cost-normalized performance comparison
    2. Pareto frontier identification
    3. Value scoring
    4. Break-even analysis
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add_result(
        self,
        database: str,
        recall_at_10: float,
        latency_p50_ms: float,
        qps: float,
        cost_per_million_queries_usd: float,
    ) -> None:
        """Add a benchmark result for analysis."""
        self.results.append({
            "database": database,
            "recall_at_10": recall_at_10,
            "latency_p50_ms": latency_p50_ms,
            "qps": qps,
            "cost_per_million_queries_usd": cost_per_million_queries_usd,
        })

    def compute_metrics(self) -> List[CostEfficiencyMetrics]:
        """Compute cost-efficiency metrics for all databases."""
        if not self.results:
            return []

        metrics = []

        for result in self.results:
            cost_per_query = result["cost_per_million_queries_usd"] / 1_000_000

            # Cost-normalized metrics
            recall_per_dollar = result["recall_at_10"] / cost_per_query if cost_per_query > 0 else float('inf')
            qps_per_dollar = result["qps"] / result["cost_per_million_queries_usd"] if result["cost_per_million_queries_usd"] > 0 else float('inf')
            latency_cost_product = result["latency_p50_ms"] * cost_per_query

            # Value score: higher is better
            # Combines recall, inverse latency, and inverse cost
            # Normalized to 0-100 scale
            value_score = self._compute_value_score(
                result["recall_at_10"],
                result["latency_p50_ms"],
                result["qps"],
                result["cost_per_million_queries_usd"]
            )

            metrics.append(CostEfficiencyMetrics(
                database=result["database"],
                recall_at_10=result["recall_at_10"],
                latency_p50_ms=result["latency_p50_ms"],
                qps=result["qps"],
                cost_per_million_queries_usd=result["cost_per_million_queries_usd"],
                recall_per_dollar=recall_per_dollar,
                qps_per_dollar=qps_per_dollar,
                latency_cost_product=latency_cost_product,
                value_score=value_score,
            ))

        # Identify Pareto optimal points
        pareto_indices = self._compute_pareto_frontier(metrics)
        for i in pareto_indices:
            metrics[i].is_pareto_optimal = True

        return metrics

    def _compute_value_score(
        self,
        recall: float,
        latency_ms: float,
        qps: float,
        cost_per_million: float
    ) -> float:
        """
        Compute composite value score.

        Formula: value = (recall * qps) / (latency * cost)

        Normalized to 0-100 scale relative to the dataset.
        """
        if latency_ms <= 0 or cost_per_million <= 0:
            return 0.0

        raw_value = (recall * qps) / (latency_ms * cost_per_million)

        # We'll normalize after computing all scores
        return raw_value

    def _compute_pareto_frontier(
        self,
        metrics: List[CostEfficiencyMetrics]
    ) -> List[int]:
        """
        Identify Pareto optimal points.

        A point is Pareto optimal if no other point is better on ALL dimensions.
        We optimize for: higher recall, lower latency, higher QPS, lower cost.
        """
        n = len(metrics)
        pareto_indices = []

        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i == j:
                    continue

                # Check if j dominates i
                # j dominates i if j is >= on all metrics and > on at least one
                # (for metrics where higher is better, we flip the comparison for cost/latency)
                j_better_recall = metrics[j].recall_at_10 >= metrics[i].recall_at_10
                j_better_latency = metrics[j].latency_p50_ms <= metrics[i].latency_p50_ms
                j_better_qps = metrics[j].qps >= metrics[i].qps
                j_better_cost = metrics[j].cost_per_million_queries_usd <= metrics[i].cost_per_million_queries_usd

                all_better_or_equal = j_better_recall and j_better_latency and j_better_qps and j_better_cost

                strictly_better = (
                    (metrics[j].recall_at_10 > metrics[i].recall_at_10) or
                    (metrics[j].latency_p50_ms < metrics[i].latency_p50_ms) or
                    (metrics[j].qps > metrics[i].qps) or
                    (metrics[j].cost_per_million_queries_usd < metrics[i].cost_per_million_queries_usd)
                )

                if all_better_or_equal and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_indices.append(i)

        return pareto_indices

    def get_best_for_budget(
        self,
        max_cost_per_million: float
    ) -> Optional[CostEfficiencyMetrics]:
        """
        Find best database within a cost budget.

        Args:
            max_cost_per_million: Maximum $/million queries budget

        Returns:
            Best database by recall within budget, or None
        """
        metrics = self.compute_metrics()
        within_budget = [
            m for m in metrics
            if m.cost_per_million_queries_usd <= max_cost_per_million
        ]

        if not within_budget:
            return None

        return max(within_budget, key=lambda m: m.recall_at_10)

    def get_cheapest_for_recall(
        self,
        min_recall: float
    ) -> Optional[CostEfficiencyMetrics]:
        """
        Find cheapest database meeting a recall requirement.

        Args:
            min_recall: Minimum required recall@10

        Returns:
            Cheapest database meeting requirement, or None
        """
        metrics = self.compute_metrics()
        meeting_recall = [
            m for m in metrics
            if m.recall_at_10 >= min_recall
        ]

        if not meeting_recall:
            return None

        return min(meeting_recall, key=lambda m: m.cost_per_million_queries_usd)

    def break_even_analysis(
        self,
        db1: str,
        db2: str,
        metric: str = "recall_at_10"
    ) -> Dict[str, Any]:
        """
        Compute break-even point between two databases.

        Returns query volume where total costs are equal.
        """
        metrics = {m.database: m for m in self.compute_metrics()}

        if db1 not in metrics or db2 not in metrics:
            return {"error": "Database not found"}

        m1 = metrics[db1]
        m2 = metrics[db2]

        # For serverless pricing, total cost = queries * cost_per_query
        # Break-even occurs when the performance advantage justifies the cost difference

        cost_diff = m1.cost_per_million_queries_usd - m2.cost_per_million_queries_usd

        if metric == "recall_at_10":
            perf_diff = m1.recall_at_10 - m2.recall_at_10
        elif metric == "qps":
            perf_diff = m1.qps - m2.qps
        elif metric == "latency_p50_ms":
            perf_diff = m2.latency_p50_ms - m1.latency_p50_ms  # Lower is better
        else:
            return {"error": f"Unknown metric: {metric}"}

        return {
            "db1": db1,
            "db2": db2,
            "metric": metric,
            "db1_cost_per_million": m1.cost_per_million_queries_usd,
            "db2_cost_per_million": m2.cost_per_million_queries_usd,
            "cost_difference": cost_diff,
            "performance_difference": perf_diff,
            "cost_per_performance_point": abs(cost_diff / perf_diff) if perf_diff != 0 else float('inf'),
            "recommendation": db1 if (perf_diff > 0 and cost_diff <= 0) or (perf_diff > 0 and cost_diff / perf_diff < 1000) else db2,
        }


def compute_pareto_frontier(
    points: List[Tuple[float, float]],
    maximize_x: bool = True,
    maximize_y: bool = True
) -> List[int]:
    """
    Compute 2D Pareto frontier.

    Args:
        points: List of (x, y) tuples
        maximize_x: Whether to maximize x (True) or minimize (False)
        maximize_y: Whether to maximize y (True) or minimize (False)

    Returns:
        Indices of points on Pareto frontier
    """
    n = len(points)
    pareto_indices = []

    for i in range(n):
        is_dominated = False
        xi, yi = points[i]

        for j in range(n):
            if i == j:
                continue

            xj, yj = points[j]

            # Check if j dominates i
            if maximize_x:
                x_cmp = xj >= xi
                x_strict = xj > xi
            else:
                x_cmp = xj <= xi
                x_strict = xj < xi

            if maximize_y:
                y_cmp = yj >= yi
                y_strict = yj > yi
            else:
                y_cmp = yj <= yi
                y_strict = yj < yi

            if x_cmp and y_cmp and (x_strict or y_strict):
                is_dominated = True
                break

        if not is_dominated:
            pareto_indices.append(i)

    return sorted(pareto_indices, key=lambda i: points[i][0])


def generate_cost_report(
    metrics: List[CostEfficiencyMetrics],
    output_format: str = "markdown"
) -> str:
    """
    Generate a cost analysis report.

    Args:
        metrics: List of cost-efficiency metrics
        output_format: "markdown" or "latex"

    Returns:
        Formatted report string
    """
    if output_format == "markdown":
        lines = [
            "# Cost-Efficiency Analysis Report\n",
            "## Summary\n",
            "| Database | Recall@10 | Latency p50 | QPS | $/M Queries | Value Score | Pareto |",
            "|----------|-----------|-------------|-----|-------------|-------------|--------|",
        ]

        for m in sorted(metrics, key=lambda x: -x.value_score):
            pareto = "✓" if m.is_pareto_optimal else ""
            lines.append(
                f"| {m.database} | {m.recall_at_10:.4f} | {m.latency_p50_ms:.2f} ms | "
                f"{m.qps:.1f} | ${m.cost_per_million_queries_usd:.2f} | "
                f"{m.value_score:.2f} | {pareto} |"
            )

        lines.extend([
            "\n## Cost-Normalized Metrics\n",
            "| Database | Recall/$ | QPS/$ | Latency×Cost |",
            "|----------|----------|-------|--------------|",
        ])

        for m in metrics:
            lines.append(
                f"| {m.database} | {m.recall_per_dollar:.2f} | {m.qps_per_dollar:.2f} | "
                f"{m.latency_cost_product:.6f} |"
            )

        # Pareto optimal
        pareto_dbs = [m.database for m in metrics if m.is_pareto_optimal]
        lines.extend([
            f"\n## Pareto Optimal Databases\n",
            f"The following databases are on the cost-quality Pareto frontier:\n",
            f"**{', '.join(pareto_dbs)}**\n",
        ])

        return "\n".join(lines)

    elif output_format == "latex":
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Cost-Efficiency Analysis}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Database & Recall@10 & Latency p50 & QPS & \$/M Queries & Value & Pareto \\",
            r"\midrule",
        ]

        for m in sorted(metrics, key=lambda x: -x.value_score):
            pareto = r"\checkmark" if m.is_pareto_optimal else ""
            lines.append(
                f"{m.database} & {m.recall_at_10:.4f} & {m.latency_p50_ms:.2f} & "
                f"{m.qps:.1f} & {m.cost_per_million_queries_usd:.2f} & "
                f"{m.value_score:.2f} & {pareto} \\\\"
            )

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown output format: {output_format}")
