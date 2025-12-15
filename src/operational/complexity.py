"""
Operational Complexity Scoring for Vector Databases.

This module quantifies the operational burden of deploying and
maintaining vector databases - a critical production factor
often ignored in benchmarks.

Dimensions:
1. Deployment Complexity - Steps to get running
2. Configuration Complexity - Tuning options and defaults
3. Monitoring & Observability - Built-in metrics and integrations
4. Maintenance Burden - Backup, upgrade, scaling procedures
5. Documentation Quality - Completeness and clarity
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum


class DeploymentMethod(Enum):
    """Supported deployment methods."""
    MANAGED_CLOUD = "managed_cloud"  # Fully managed SaaS
    DOCKER_SINGLE = "docker_single"  # Single Docker container
    DOCKER_COMPOSE = "docker_compose"  # Multi-container Docker
    KUBERNETES = "kubernetes"        # K8s deployment
    BINARY = "binary"               # Direct binary installation
    EMBEDDED = "embedded"           # In-process library


@dataclass
class OperationalMetrics:
    """Raw operational metrics for a database."""
    # Deployment
    deployment_methods: List[DeploymentMethod]
    min_deployment_steps: int        # Minimum steps to get running
    has_managed_option: bool         # SaaS/managed service available
    docker_image_size_mb: float      # Docker image size

    # Configuration
    required_config_params: int      # Params that MUST be set
    optional_config_params: int      # Optional tuning parameters
    has_sensible_defaults: bool      # Works well out of box
    config_documentation_quality: int  # 1-5 scale

    # Dependencies
    external_dependencies: List[str]  # etcd, minio, postgres, etc.
    min_memory_gb: float             # Minimum memory requirement
    min_cpu_cores: int               # Minimum CPU cores

    # Monitoring
    prometheus_metrics: bool         # Exposes Prometheus metrics
    builtin_dashboard: bool          # Has built-in monitoring UI
    structured_logging: bool         # JSON/structured logs
    health_check_endpoint: bool      # /health or similar

    # Maintenance
    online_backup: bool              # Can backup without downtime
    online_scaling: bool             # Can scale without downtime
    auto_recovery: bool              # Automatic failure recovery
    upgrade_documentation: int       # 1-5 scale

    # Documentation
    quickstart_time_minutes: int     # Time to complete quickstart
    api_documentation_quality: int   # 1-5 scale
    example_code_quality: int        # 1-5 scale
    community_size: int              # GitHub stars (proxy)

    # Support
    has_enterprise_support: bool
    community_response_hours: float  # Avg response time on issues


@dataclass
class OperationalComplexityScore:
    """Computed operational complexity scores."""
    database: str

    # Subscores (0-100, lower is simpler/better)
    deployment_score: float
    configuration_score: float
    monitoring_score: float
    maintenance_score: float
    documentation_score: float

    # Overall score (0-100, lower is simpler)
    overall_score: float

    # Recommendations
    best_for: List[str]          # Use cases this DB excels at
    challenges: List[str]        # Potential operational challenges
    recommended_team_size: str   # "solo", "small", "medium", "large"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "database": self.database,
            "deployment_score": self.deployment_score,
            "configuration_score": self.configuration_score,
            "monitoring_score": self.monitoring_score,
            "maintenance_score": self.maintenance_score,
            "documentation_score": self.documentation_score,
            "overall_score": self.overall_score,
            "best_for": self.best_for,
            "challenges": self.challenges,
            "recommended_team_size": self.recommended_team_size,
        }


# Pre-defined operational profiles for each database
DATABASE_OPERATIONAL_PROFILES: Dict[str, OperationalMetrics] = {
    "pinecone": OperationalMetrics(
        deployment_methods=[DeploymentMethod.MANAGED_CLOUD],
        min_deployment_steps=3,  # Sign up, create index, insert
        has_managed_option=True,
        docker_image_size_mb=0,  # No self-hosting

        required_config_params=2,  # API key, environment
        optional_config_params=5,
        has_sensible_defaults=True,
        config_documentation_quality=5,

        external_dependencies=[],
        min_memory_gb=0,  # Managed
        min_cpu_cores=0,

        prometheus_metrics=False,  # Cloud dashboard only
        builtin_dashboard=True,
        structured_logging=True,
        health_check_endpoint=True,

        online_backup=True,  # Managed by Pinecone
        online_scaling=True,
        auto_recovery=True,
        upgrade_documentation=5,

        quickstart_time_minutes=15,
        api_documentation_quality=5,
        example_code_quality=5,
        community_size=2000,

        has_enterprise_support=True,
        community_response_hours=24,
    ),

    "milvus": OperationalMetrics(
        deployment_methods=[
            DeploymentMethod.DOCKER_COMPOSE,
            DeploymentMethod.KUBERNETES,
            DeploymentMethod.MANAGED_CLOUD,
        ],
        min_deployment_steps=8,  # docker-compose with etcd, minio
        has_managed_option=True,  # Zilliz Cloud
        docker_image_size_mb=500,

        required_config_params=5,
        optional_config_params=50,  # Many tuning options
        has_sensible_defaults=True,
        config_documentation_quality=4,

        external_dependencies=["etcd", "minio/s3"],
        min_memory_gb=8,
        min_cpu_cores=4,

        prometheus_metrics=True,
        builtin_dashboard=False,  # Attu is separate
        structured_logging=True,
        health_check_endpoint=True,

        online_backup=True,
        online_scaling=True,  # Distributed architecture
        auto_recovery=True,
        upgrade_documentation=4,

        quickstart_time_minutes=30,
        api_documentation_quality=4,
        example_code_quality=4,
        community_size=25000,

        has_enterprise_support=True,  # Zilliz
        community_response_hours=48,
    ),

    "qdrant": OperationalMetrics(
        deployment_methods=[
            DeploymentMethod.DOCKER_SINGLE,
            DeploymentMethod.KUBERNETES,
            DeploymentMethod.MANAGED_CLOUD,
            DeploymentMethod.BINARY,
        ],
        min_deployment_steps=2,  # docker run
        has_managed_option=True,  # Qdrant Cloud
        docker_image_size_mb=150,

        required_config_params=1,  # Just port
        optional_config_params=20,
        has_sensible_defaults=True,
        config_documentation_quality=5,

        external_dependencies=[],  # Standalone
        min_memory_gb=2,
        min_cpu_cores=2,

        prometheus_metrics=True,
        builtin_dashboard=True,  # Web UI included
        structured_logging=True,
        health_check_endpoint=True,

        online_backup=True,
        online_scaling=True,  # Distributed mode
        auto_recovery=True,
        upgrade_documentation=5,

        quickstart_time_minutes=10,
        api_documentation_quality=5,
        example_code_quality=5,
        community_size=15000,

        has_enterprise_support=True,
        community_response_hours=24,
    ),

    "weaviate": OperationalMetrics(
        deployment_methods=[
            DeploymentMethod.DOCKER_SINGLE,
            DeploymentMethod.KUBERNETES,
            DeploymentMethod.MANAGED_CLOUD,
        ],
        min_deployment_steps=3,
        has_managed_option=True,  # Weaviate Cloud
        docker_image_size_mb=300,

        required_config_params=3,
        optional_config_params=30,
        has_sensible_defaults=True,
        config_documentation_quality=4,

        external_dependencies=[],
        min_memory_gb=4,
        min_cpu_cores=2,

        prometheus_metrics=True,
        builtin_dashboard=False,
        structured_logging=True,
        health_check_endpoint=True,

        online_backup=True,
        online_scaling=True,
        auto_recovery=True,
        upgrade_documentation=4,

        quickstart_time_minutes=15,
        api_documentation_quality=4,
        example_code_quality=4,
        community_size=8000,

        has_enterprise_support=True,
        community_response_hours=36,
    ),

    "pgvector": OperationalMetrics(
        deployment_methods=[
            DeploymentMethod.DOCKER_SINGLE,
            DeploymentMethod.MANAGED_CLOUD,  # RDS, Cloud SQL
            DeploymentMethod.BINARY,
        ],
        min_deployment_steps=4,  # Need PostgreSQL + extension
        has_managed_option=True,  # Via managed Postgres
        docker_image_size_mb=400,

        required_config_params=4,  # PG connection params
        optional_config_params=15,
        has_sensible_defaults=True,  # Inherits from PG
        config_documentation_quality=4,

        external_dependencies=["postgresql"],
        min_memory_gb=2,
        min_cpu_cores=2,

        prometheus_metrics=True,  # Via pg_stat
        builtin_dashboard=False,  # pgAdmin is separate
        structured_logging=True,
        health_check_endpoint=True,

        online_backup=True,  # pg_dump, pg_basebackup
        online_scaling=True,  # Postgres replication
        auto_recovery=True,  # With proper setup
        upgrade_documentation=4,

        quickstart_time_minutes=20,
        api_documentation_quality=3,  # Less vector-specific docs
        example_code_quality=3,
        community_size=7000,

        has_enterprise_support=True,  # Via Postgres vendors
        community_response_hours=48,
    ),

    "chroma": OperationalMetrics(
        deployment_methods=[
            DeploymentMethod.EMBEDDED,
            DeploymentMethod.DOCKER_SINGLE,
        ],
        min_deployment_steps=1,  # pip install
        has_managed_option=False,  # No managed service yet
        docker_image_size_mb=200,

        required_config_params=0,  # Works with defaults
        optional_config_params=10,
        has_sensible_defaults=True,
        config_documentation_quality=4,

        external_dependencies=[],
        min_memory_gb=1,
        min_cpu_cores=1,

        prometheus_metrics=False,
        builtin_dashboard=False,
        structured_logging=False,
        health_check_endpoint=True,

        online_backup=False,  # File-based
        online_scaling=False,  # Single-node only
        auto_recovery=False,
        upgrade_documentation=3,

        quickstart_time_minutes=5,
        api_documentation_quality=4,
        example_code_quality=4,
        community_size=10000,

        has_enterprise_support=False,
        community_response_hours=72,
    ),
}


def compute_complexity_score(metrics: OperationalMetrics, database: str) -> OperationalComplexityScore:
    """
    Compute operational complexity score from metrics.

    Scoring: 0-100 where LOWER is SIMPLER/BETTER.
    """
    # Deployment Score (0-100)
    deployment_score = 0
    deployment_score += min(metrics.min_deployment_steps * 5, 40)
    deployment_score += 0 if metrics.has_managed_option else 20
    deployment_score += len(metrics.external_dependencies) * 10
    deployment_score += min(metrics.docker_image_size_mb / 100, 20)
    deployment_score = min(deployment_score, 100)

    # Configuration Score (0-100)
    configuration_score = 0
    configuration_score += metrics.required_config_params * 10
    configuration_score += min(metrics.optional_config_params, 30)
    configuration_score += 0 if metrics.has_sensible_defaults else 30
    configuration_score += (5 - metrics.config_documentation_quality) * 5
    configuration_score = min(configuration_score, 100)

    # Monitoring Score (0-100) - lower is better (more monitoring = easier ops)
    monitoring_score = 100
    monitoring_score -= 25 if metrics.prometheus_metrics else 0
    monitoring_score -= 25 if metrics.builtin_dashboard else 0
    monitoring_score -= 25 if metrics.structured_logging else 0
    monitoring_score -= 25 if metrics.health_check_endpoint else 0
    monitoring_score = max(monitoring_score, 0)

    # Maintenance Score (0-100)
    maintenance_score = 0
    maintenance_score += 0 if metrics.online_backup else 20
    maintenance_score += 0 if metrics.online_scaling else 20
    maintenance_score += 0 if metrics.auto_recovery else 20
    maintenance_score += (5 - metrics.upgrade_documentation) * 8
    maintenance_score = min(maintenance_score, 100)

    # Documentation Score (0-100) - lower is better (better docs = easier ops)
    documentation_score = 100
    documentation_score -= min(100 - metrics.quickstart_time_minutes, 40)
    documentation_score -= metrics.api_documentation_quality * 10
    documentation_score -= metrics.example_code_quality * 10
    documentation_score = max(documentation_score, 0)

    # Overall Score (weighted average)
    weights = {
        "deployment": 0.25,
        "configuration": 0.20,
        "monitoring": 0.20,
        "maintenance": 0.25,
        "documentation": 0.10,
    }
    overall_score = (
        deployment_score * weights["deployment"] +
        configuration_score * weights["configuration"] +
        monitoring_score * weights["monitoring"] +
        maintenance_score * weights["maintenance"] +
        documentation_score * weights["documentation"]
    )

    # Generate recommendations
    best_for = []
    challenges = []

    if DeploymentMethod.MANAGED_CLOUD in metrics.deployment_methods:
        best_for.append("Teams wanting zero ops burden")
    if DeploymentMethod.EMBEDDED in metrics.deployment_methods:
        best_for.append("Local development and prototyping")
    if metrics.min_deployment_steps <= 3:
        best_for.append("Quick POCs and demos")
    if not metrics.external_dependencies:
        best_for.append("Simple deployments without dependencies")
    if metrics.online_scaling:
        best_for.append("Growing workloads requiring scale")

    if metrics.external_dependencies:
        challenges.append(f"Requires external services: {', '.join(metrics.external_dependencies)}")
    if not metrics.online_backup:
        challenges.append("No online backup - requires downtime for backups")
    if not metrics.prometheus_metrics:
        challenges.append("Limited observability - no Prometheus metrics")
    if metrics.min_memory_gb > 4:
        challenges.append(f"High memory requirement: {metrics.min_memory_gb}GB minimum")

    # Team size recommendation
    if overall_score < 25:
        team_size = "solo"
    elif overall_score < 50:
        team_size = "small"
    elif overall_score < 75:
        team_size = "medium"
    else:
        team_size = "large"

    return OperationalComplexityScore(
        database=database,
        deployment_score=deployment_score,
        configuration_score=configuration_score,
        monitoring_score=monitoring_score,
        maintenance_score=maintenance_score,
        documentation_score=documentation_score,
        overall_score=overall_score,
        best_for=best_for,
        challenges=challenges,
        recommended_team_size=team_size,
    )


def score_all_databases() -> Dict[str, OperationalComplexityScore]:
    """Compute complexity scores for all profiled databases."""
    scores = {}
    for db_name, metrics in DATABASE_OPERATIONAL_PROFILES.items():
        scores[db_name] = compute_complexity_score(metrics, db_name)
    return scores


def generate_complexity_report(output_format: str = "markdown") -> str:
    """Generate operational complexity report."""
    scores = score_all_databases()

    if output_format == "markdown":
        lines = [
            "# Operational Complexity Report\n",
            "Lower scores indicate simpler operations (0-100 scale).\n",
            "## Overall Ranking\n",
            "| Database | Overall | Deployment | Config | Monitoring | Maintenance | Docs | Team Size |",
            "|----------|---------|------------|--------|------------|-------------|------|-----------|",
        ]

        for db, score in sorted(scores.items(), key=lambda x: x[1].overall_score):
            lines.append(
                f"| {db} | {score.overall_score:.0f} | {score.deployment_score:.0f} | "
                f"{score.configuration_score:.0f} | {score.monitoring_score:.0f} | "
                f"{score.maintenance_score:.0f} | {score.documentation_score:.0f} | "
                f"{score.recommended_team_size} |"
            )

        lines.append("\n## Detailed Analysis\n")

        for db, score in sorted(scores.items(), key=lambda x: x[1].overall_score):
            lines.append(f"### {db.title()}\n")
            lines.append(f"**Overall Score:** {score.overall_score:.0f}/100\n")
            lines.append(f"**Best For:**")
            for item in score.best_for:
                lines.append(f"- {item}")
            lines.append(f"\n**Challenges:**")
            for item in score.challenges:
                lines.append(f"- {item}")
            lines.append("")

        return "\n".join(lines)

    else:
        raise ValueError(f"Unknown format: {output_format}")
