"""
Statistical Analysis Module

Provides proper statistical rigor for benchmark results:
- Confidence intervals
- Hypothesis testing (t-tests, ANOVA)
- Effect sizes (Cohen's d)
- Multiple comparison corrections
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy import stats


@dataclass
class StatisticalResult:
    """Result of statistical analysis."""
    metric: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "mean": self.mean,
            "std": self.std,
            "ci_95": [self.ci_lower, self.ci_upper],
            "n": self.n
        }


@dataclass
class ComparisonResult:
    """Result of pairwise comparison."""
    metric: str
    db_a: str
    db_b: str
    mean_a: float
    mean_b: float
    difference: float
    difference_percent: float
    t_statistic: float
    p_value: float
    cohens_d: float
    significant: bool
    interpretation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "comparison": f"{self.db_a} vs {self.db_b}",
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "difference": self.difference,
            "difference_percent": self.difference_percent,
            "t_statistic": self.t_statistic,
            "p_value": self.p_value,
            "cohens_d": self.cohens_d,
            "significant": self.significant,
            "interpretation": self.interpretation
        }


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float, float]:
    """
    Compute confidence interval for data.

    Returns:
        Tuple of (mean, std, ci_lower, ci_upper)
    """
    data_np = np.array(data)
    n = len(data_np)
    mean = np.mean(data_np)
    std = np.std(data_np, ddof=1)  # Sample std dev

    if n < 2:
        return mean, std, mean, mean

    # t-distribution for small samples
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_crit * (std / np.sqrt(n))

    return mean, std, mean - margin, mean + margin


def compute_stats_for_trials(
    trials: List[float],
    metric_name: str,
    confidence: float = 0.95
) -> StatisticalResult:
    """Compute statistics for a set of trial measurements."""
    mean, std, ci_lower, ci_upper = compute_confidence_interval(trials, confidence)

    return StatisticalResult(
        metric=metric_name,
        mean=mean,
        std=std,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n=len(trials)
    )


def check_normality(data: List[float], alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Check if data is approximately normally distributed using Shapiro-Wilk test.

    Note: Shapiro-Wilk is reliable for n < 5000. For larger samples,
    we use a random subsample.

    Returns:
        Tuple of (is_normal, p_value)
    """
    data_np = np.array(data)

    # Shapiro-Wilk test requires 3 <= n <= 5000
    if len(data_np) < 3:
        return True, 1.0  # Assume normal for tiny samples

    if len(data_np) > 5000:
        # Subsample for large datasets
        rng = np.random.default_rng(42)
        data_np = rng.choice(data_np, size=5000, replace=False)

    _, p_val = stats.shapiro(data_np)
    return p_val > alpha, float(p_val)


def two_sample_ttest(
    data_a: List[float],
    data_b: List[float],
    equal_var: bool = False
) -> Tuple[float, float]:
    """
    Perform Welch's t-test (unequal variance by default).

    Returns:
        Tuple of (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_ind(data_a, data_b, equal_var=equal_var)
    return float(t_stat), float(p_val)


def mann_whitney_u(
    data_a: List[float],
    data_b: List[float]
) -> Tuple[float, float]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).

    Use this when data is not normally distributed (common for latencies).

    Returns:
        Tuple of (u_statistic, p_value)
    """
    u_stat, p_val = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
    return float(u_stat), float(p_val)


def cohens_d(data_a: List[float], data_b: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large
    """
    n1, n2 = len(data_a), len(data_b)
    var1, var2 = np.var(data_a, ddof=1), np.var(data_b, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(data_a) - np.mean(data_b)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d value."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def compare_databases(
    metric_name: str,
    db_a_name: str,
    db_a_data: List[float],
    db_b_name: str,
    db_b_data: List[float],
    alpha: float = 0.05
) -> ComparisonResult:
    """
    Compare two databases on a metric with full statistical analysis.

    Automatically selects appropriate test based on normality:
    - If both datasets are normal: Welch's t-test
    - If either is non-normal: Mann-Whitney U test

    Returns:
        ComparisonResult with test results, effect size, and interpretation
    """
    mean_a = np.mean(db_a_data)
    mean_b = np.mean(db_b_data)
    diff = mean_a - mean_b
    diff_pct = (diff / mean_b) * 100 if mean_b != 0 else 0

    # Check normality of both datasets
    normal_a, norm_p_a = check_normality(db_a_data)
    normal_b, norm_p_b = check_normality(db_b_data)

    # Select appropriate test
    if normal_a and normal_b:
        # Both normal: use t-test
        t_stat, p_val = two_sample_ttest(db_a_data, db_b_data)
        test_used = "welch_t_test"
    else:
        # Non-normal: use Mann-Whitney U (non-parametric)
        t_stat, p_val = mann_whitney_u(db_a_data, db_b_data)
        test_used = "mann_whitney_u"

    d = cohens_d(db_a_data, db_b_data)

    significant = p_val < alpha
    effect = interpret_effect_size(d)

    # Generate interpretation with test info
    test_note = f" (using {test_used})" if test_used == "mann_whitney_u" else ""
    if significant:
        direction = "higher" if diff > 0 else "lower"
        interpretation = (
            f"{db_a_name} has significantly {direction} {metric_name} than {db_b_name} "
            f"(p={p_val:.4f}, {effect} effect size d={d:.2f}){test_note}"
        )
    else:
        interpretation = (
            f"No significant difference in {metric_name} between {db_a_name} and {db_b_name} "
            f"(p={p_val:.4f}){test_note}"
        )

    return ComparisonResult(
        metric=metric_name,
        db_a=db_a_name,
        db_b=db_b_name,
        mean_a=mean_a,
        mean_b=mean_b,
        difference=diff,
        difference_percent=diff_pct,
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=d,
        significant=significant,
        interpretation=interpretation
    )


def anova_multiple_databases(
    metric_name: str,
    db_data: Dict[str, List[float]]
) -> Dict[str, Any]:
    """
    One-way ANOVA for comparing multiple databases.

    Args:
        metric_name: Name of the metric
        db_data: Dict mapping database name to list of measurements

    Returns:
        Dict with ANOVA results and post-hoc analysis
    """
    groups = list(db_data.values())
    db_names = list(db_data.keys())

    # ANOVA
    f_stat, p_val = stats.f_oneway(*groups)

    result = {
        "metric": metric_name,
        "num_groups": len(groups),
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "significant": p_val < 0.05,
    }

    # If significant, do post-hoc pairwise comparisons
    if p_val < 0.05:
        pairwise = []
        for i, (name_a, data_a) in enumerate(db_data.items()):
            for name_b, data_b in list(db_data.items())[i+1:]:
                comparison = compare_databases(
                    metric_name, name_a, data_a, name_b, data_b
                )
                pairwise.append(comparison.to_dict())

        result["pairwise_comparisons"] = pairwise

        # Apply Bonferroni correction
        num_comparisons = len(pairwise)
        corrected_alpha = 0.05 / num_comparisons
        result["bonferroni_alpha"] = corrected_alpha

    return result


def aggregate_trial_results(
    all_results: List[Dict[str, Any]],
    metrics: List[str] = ["qps", "latency_p50", "recall_at_10"]
) -> Dict[str, Dict[str, StatisticalResult]]:
    """
    Aggregate results across multiple trials into statistical summaries.

    Args:
        all_results: List of result dicts from multiple trials
        metrics: List of metric names to aggregate

    Returns:
        Dict mapping database -> metric -> StatisticalResult
    """
    # Group by database
    db_trials: Dict[str, Dict[str, List[float]]] = {}

    for result in all_results:
        db_name = result.get("database", "unknown")
        if db_name not in db_trials:
            db_trials[db_name] = {m: [] for m in metrics}

        # Extract metrics
        for metric in metrics:
            value = _extract_metric(result, metric)
            if value is not None:
                db_trials[db_name][metric].append(value)

    # Compute statistics
    aggregated = {}
    for db_name, metrics_data in db_trials.items():
        aggregated[db_name] = {}
        for metric, values in metrics_data.items():
            if values:
                aggregated[db_name][metric] = compute_stats_for_trials(values, metric)

    return aggregated


def _extract_metric(result: Dict, metric: str) -> Optional[float]:
    """Extract a metric value from a result dict."""
    # Handle nested paths like "qps.qps" or "standard.latency.p50"
    parts = metric.split(".")
    value = result

    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return None

    return float(value) if isinstance(value, (int, float)) else None


def format_result_with_ci(
    stat: StatisticalResult,
    precision: int = 2
) -> str:
    """Format a statistical result as 'mean ± CI' string."""
    margin = (stat.ci_upper - stat.ci_lower) / 2
    return f"{stat.mean:.{precision}f} ± {margin:.{precision}f}"


def generate_latex_table(
    aggregated: Dict[str, Dict[str, StatisticalResult]],
    metrics: List[str],
    caption: str = "Benchmark Results"
) -> str:
    """Generate LaTeX table with statistical results."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{l" + "c" * len(metrics) + "}",
        "\\toprule",
        "\\textbf{Database} & " + " & ".join(f"\\textbf{{{m}}}" for m in metrics) + " \\\\",
        "\\midrule",
    ]

    for db_name, stats in aggregated.items():
        row = [db_name]
        for metric in metrics:
            if metric in stats:
                row.append(format_result_with_ci(stats[metric]))
            else:
                row.append("--")
        lines.append(" & ".join(row) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)
