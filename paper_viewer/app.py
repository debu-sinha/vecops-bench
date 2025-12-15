#!/usr/bin/env python3
"""
VectorDB-Bench Paper Visualization App

A Streamlit application to visualize benchmark results and paper content.

Author: Debu Sinha <debusinha2009@gmail.com>

Usage:
    cd paper_viewer
    pip install streamlit plotly pandas
    streamlit run app.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="VectorDB-Bench Paper Viewer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .db-milvus { color: #648FFF; }
    .db-qdrant { color: #785EF0; }
    .db-pgvector { color: #DC267F; }
    .db-weaviate { color: #FE6100; }
    .db-chroma { color: #FFB000; }
</style>
""", unsafe_allow_html=True)

# Color scheme (color-blind friendly)
COLORS = {
    'milvus': '#648FFF',
    'qdrant': '#785EF0',
    'pgvector': '#DC267F',
    'weaviate': '#FE6100',
    'chroma': '#FFB000',
}

DB_NAMES = {
    'milvus': 'Milvus',
    'qdrant': 'Qdrant',
    'pgvector': 'pgvector',
    'weaviate': 'Weaviate',
    'chroma': 'Chroma',
}


def load_results(results_dir: str) -> List[Dict]:
    """Load all benchmark results from directory."""
    results = []
    results_path = Path(results_dir)

    if not results_path.exists():
        return results

    for json_file in results_path.glob('**/*.json'):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
        except Exception as e:
            st.warning(f"Could not load {json_file}: {e}")

    return results


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    """Convert results to DataFrame."""
    rows = []

    for r in results:
        if r.get('status') != 'success':
            continue

        row = {
            'database': r.get('database', 'unknown'),
            'num_documents': r.get('num_documents', 0),
            'timestamp': r.get('timestamp', ''),
        }

        # Standard metrics
        if 'standard' in r:
            std = r['standard']
            if 'recall' in std:
                row['recall@1'] = std['recall'].get('@1', 0)
                row['recall@10'] = std['recall'].get('@10', 0)
                row['recall@100'] = std['recall'].get('@100', 0)
            if 'latency' in std:
                row['latency_p50'] = std['latency'].get('p50', 0)
                row['latency_p95'] = std['latency'].get('p95', 0)
                row['latency_p99'] = std['latency'].get('p99', 0)

        # QPS
        if 'qps' in r and isinstance(r['qps'], dict):
            row['qps'] = r['qps'].get('qps', 0)

        # Cold start
        if 'cold_start' in r:
            row['cold_start_ms'] = r['cold_start'].get('mean_ms', 0)

        # Index build
        if 'index_build' in r:
            row['build_time_s'] = r['index_build'].get('total_build_time_seconds', 0)
            row['vectors_per_sec'] = r['index_build'].get('vectors_per_second', 0)
            row['memory_mb'] = r['index_build'].get('memory_delta_mb', 0)

        # Filtered search
        if 'filtered_search' in r:
            row['filter_overhead'] = r['filtered_search'].get('filter_overhead_percent', 0)

        # Operational complexity
        if 'operational_complexity' in r:
            ops = r['operational_complexity']
            row['ops_overall'] = ops.get('overall_score', 0)
            row['ops_deployment'] = ops.get('deployment_score', 0)
            row['ops_monitoring'] = ops.get('monitoring_score', 0)
            row['ops_maintenance'] = ops.get('maintenance_score', 0)

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_database(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate results by database with statistics."""
    if df.empty:
        return df

    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if 'database' in df.columns:
        agg_df = df.groupby('database')[numeric_cols].agg(['mean', 'std', 'count']).reset_index()
        agg_df.columns = ['_'.join(col).strip('_') for col in agg_df.columns]
        return agg_df
    return df


def main():
    # Header
    st.markdown('<div class="main-header">VectorDB-Bench</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Production-Oriented Vector Database Benchmarking</div>', unsafe_allow_html=True)
    st.markdown("**Author:** Debu Sinha (debusinha2009@gmail.com)")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "📈 Detailed Results", "📄 Paper Outline", "🔬 Methodology", "📁 Raw Data"]
    )

    # Load data - use absolute path to results
    default_results_dir = r"C:\Users\dsinh\research\vectordb-bench\results"

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Source")
    custom_path = st.sidebar.text_input("Results Directory", default_results_dir)

    results = load_results(custom_path)
    df = results_to_dataframe(results)

    st.sidebar.markdown(f"**Loaded:** {len(results)} benchmark runs")
    st.sidebar.markdown(f"**Successful:** {len(df)} results")

    # Pages
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "📈 Detailed Results":
        show_detailed_results(df)
    elif page == "📄 Paper Outline":
        show_paper_outline()
    elif page == "🔬 Methodology":
        show_methodology()
    elif page == "📁 Raw Data":
        show_raw_data(df, results)


def show_dashboard(df: pd.DataFrame):
    """Main dashboard with key metrics."""
    st.header("Benchmark Dashboard")

    if df.empty:
        st.warning("No benchmark results found. Please run benchmarks first or check the results directory.")
        st.info("Expected location: `./results/` relative to project root")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Databases Tested", df['database'].nunique())
    with col2:
        if 'qps' in df.columns:
            st.metric("Max QPS", f"{df['qps'].max():.0f}")
    with col3:
        if 'recall@10' in df.columns:
            st.metric("Max Recall@10", f"{df['recall@10'].max():.3f}")
    with col4:
        if 'latency_p50' in df.columns:
            st.metric("Min Latency (p50)", f"{df['latency_p50'].min():.2f} ms")

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("QPS Comparison")
        if 'qps' in df.columns:
            fig = px.bar(
                df.groupby('database')['qps'].mean().reset_index(),
                x='database',
                y='qps',
                color='database',
                color_discrete_map=COLORS,
                title="Queries Per Second (Higher is Better)"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recall@10 Comparison")
        if 'recall@10' in df.columns:
            fig = px.bar(
                df.groupby('database')['recall@10'].mean().reset_index(),
                x='database',
                y='recall@10',
                color='database',
                color_discrete_map=COLORS,
                title="Recall@10 (Higher is Better)"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Latency Distribution")
        if 'latency_p50' in df.columns:
            fig = px.box(
                df,
                x='database',
                y='latency_p50',
                color='database',
                color_discrete_map=COLORS,
                title="Query Latency p50 (Lower is Better)"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Cold Start Performance")
        if 'cold_start_ms' in df.columns:
            fig = px.bar(
                df.groupby('database')['cold_start_ms'].mean().reset_index(),
                x='database',
                y='cold_start_ms',
                color='database',
                color_discrete_map=COLORS,
                title="Cold Start Latency (Lower is Better)"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Quality vs Performance
    st.subheader("Quality vs. Performance Trade-off")
    if 'recall@10' in df.columns and 'latency_p50' in df.columns:
        fig = px.scatter(
            df,
            x='latency_p50',
            y='recall@10',
            color='database',
            color_discrete_map=COLORS,
            size='qps' if 'qps' in df.columns else None,
            title="Recall@10 vs Latency (bubble size = QPS)",
            labels={'latency_p50': 'Latency p50 (ms)', 'recall@10': 'Recall@10'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Operational Complexity Radar
    st.subheader("Operational Complexity (Novel Metric)")
    if 'ops_deployment' in df.columns:
        radar_cols = ['ops_deployment', 'ops_monitoring', 'ops_maintenance']
        available_cols = [c for c in radar_cols if c in df.columns]

        if available_cols:
            agg = df.groupby('database')[available_cols].mean().reset_index()

            fig = go.Figure()
            categories = ['Deployment', 'Monitoring', 'Maintenance'][:len(available_cols)]

            for _, row in agg.iterrows():
                db = row['database']
                values = [100 - row[c] for c in available_cols]  # Invert: higher = simpler
                values.append(values[0])  # Complete the loop

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    name=DB_NAMES.get(db, db),
                    fill='toself',
                    line=dict(color=COLORS.get(db, '#666666'))
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Operational Simplicity (Higher = Simpler to Operate)"
            )
            st.plotly_chart(fig, use_container_width=True)


def show_detailed_results(df: pd.DataFrame):
    """Detailed results view."""
    st.header("Detailed Results")

    if df.empty:
        st.warning("No results available")
        return

    # Database filter
    databases = df['database'].unique().tolist()
    selected_dbs = st.multiselect("Filter Databases", databases, default=databases)

    filtered_df = df[df['database'].isin(selected_dbs)]

    # Summary table
    st.subheader("Summary Statistics")
    agg_df = aggregate_by_database(filtered_df)
    st.dataframe(agg_df, use_container_width=True)

    # Individual metrics
    st.subheader("Metric Breakdown")

    metric_options = {
        'QPS': 'qps',
        'Recall@1': 'recall@1',
        'Recall@10': 'recall@10',
        'Recall@100': 'recall@100',
        'Latency p50': 'latency_p50',
        'Latency p95': 'latency_p95',
        'Latency p99': 'latency_p99',
        'Cold Start': 'cold_start_ms',
        'Insert Speed': 'vectors_per_sec',
        'Memory Usage': 'memory_mb',
        'Ops Complexity': 'ops_overall',
    }

    available_metrics = {k: v for k, v in metric_options.items() if v in filtered_df.columns}

    selected_metric = st.selectbox("Select Metric", list(available_metrics.keys()))
    metric_col = available_metrics[selected_metric]

    fig = px.bar(
        filtered_df.groupby('database')[metric_col].agg(['mean', 'std']).reset_index(),
        x='database',
        y='mean',
        error_y='std',
        color='database',
        color_discrete_map=COLORS,
        title=f"{selected_metric} by Database"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_paper_outline():
    """Display paper outline and structure."""
    st.header("Paper Outline")

    st.markdown("""
    ## VectorDB-Bench: A Production-Oriented Benchmark Suite for Vector Database Systems

    **Author:** Debu Sinha (debusinha2009@gmail.com)

    ---

    ### Abstract

    Vector databases have become critical infrastructure for AI/ML applications, powering semantic search,
    recommendation systems, and retrieval-augmented generation (RAG). While existing benchmarks focus
    primarily on recall-latency trade-offs, production deployments require understanding of additional
    dimensions including cold start performance, operational complexity, filtered search efficiency,
    and cost-per-query economics.

    We present VectorDB-Bench, a comprehensive benchmark suite that evaluates five leading vector
    databases (Milvus, Qdrant, pgvector, Weaviate, Chroma) across production-relevant metrics.

    ---

    ### 1. Introduction
    - Problem: Gap between academic benchmarks and production needs
    - Motivation: Real-world deployment considerations
    - Contributions:
      1. Novel metrics (cold start, operational complexity)
      2. Rigorous multi-trial methodology
      3. Open-source release

    ### 2. Related Work
    - ann-benchmarks (algorithm-level)
    - BEIR benchmark (datasets)
    - Existing vendor benchmarks (limitations)

    ### 3. Benchmark Design
    - Evaluated systems (Milvus, Qdrant, pgvector, Weaviate, Chroma)
    - Datasets (MS MARCO, NFCorpus, SciFact)
    - Standard metrics (Recall@k, NDCG, latency, QPS)
    - **Novel metrics:**
      - Cold start latency
      - Operational complexity scoring
      - Filtered search overhead

    ### 4. Results
    - Quality-performance trade-offs
    - Throughput comparison
    - Cold start analysis *(novel)*
    - Operational complexity *(novel)*
    - Scale testing (100K → 1M vectors)

    ### 5. Discussion
    - Key findings
    - Recommendations by use case
    - Limitations

    ### 6. Conclusion
    - Summary of contributions
    - Future work

    ---

    ### Target Venues
    - **SIGMOD/VLDB**: Database systems focus
    - **ICDE**: Industrial data engineering
    - **MLSys**: ML systems
    - **arXiv**: Preprint for visibility
    """)


def show_methodology():
    """Display benchmark methodology."""
    st.header("Methodology")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Evaluated Systems")
        st.markdown("""
        | Database | Index Type | Architecture |
        |----------|-----------|--------------|
        | **Milvus** | IVF_FLAT, HNSW | Distributed, cloud-native |
        | **Qdrant** | HNSW | Rust-based, single-node/cluster |
        | **pgvector** | IVFFlat, HNSW | PostgreSQL extension |
        | **Weaviate** | HNSW | GraphQL-native, modular |
        | **Chroma** | HNSW | Embedded-first, Python-native |
        """)

    with col2:
        st.subheader("Datasets")
        st.markdown("""
        | Dataset | Domain | Size |
        |---------|--------|------|
        | **MS MARCO** | General | 100K-1M passages |
        | **NFCorpus** | Medical | 3.6K documents |
        | **SciFact** | Scientific | 5K claims |

        All embedded using `sentence-transformers/all-mpnet-base-v2` (768 dim)
        """)

    st.subheader("Experimental Setup")
    st.markdown("""
    - **Hardware:** AWS c5.2xlarge (8 vCPU, 16GB RAM)
    - **Deployment:** Docker containers with pinned versions
    - **Trials:** 5 runs per configuration for statistical validity
    - **Warm-up:** 100 queries before measurement
    """)

    st.subheader("Metrics")

    tab1, tab2 = st.tabs(["Standard Metrics", "Novel Metrics"])

    with tab1:
        st.markdown("""
        - **Recall@k:** Fraction of relevant documents in top-k results
        - **NDCG@k:** Normalized discounted cumulative gain
        - **Latency (p50, p95, p99):** Query response time percentiles
        - **QPS:** Sustained queries per second under load
        """)

    with tab2:
        st.markdown("""
        - **Cold Start Latency:** Time from container start to first successful query
          - Critical for serverless/Lambda deployments
          - Measured as mean across 5 restart trials

        - **Operational Complexity Score:** Composite metric (0-100)
          - Deployment difficulty
          - Configuration complexity
          - Monitoring capabilities
          - Maintenance burden

        - **Filtered Search Overhead:** Latency increase when combining vector search with metadata filters

        - **Insert Throughput:** Vectors indexed per second during bulk load
        """)


def show_raw_data(df: pd.DataFrame, results: List[Dict]):
    """Show raw data for download."""
    st.header("Raw Data")

    if df.empty:
        st.warning("No data available")
        return

    st.subheader("Processed Results Table")
    st.dataframe(df, use_container_width=True)

    # Download buttons
    col1, col2 = st.columns(2)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "vectordb_bench_results.csv",
            "text/csv"
        )

    with col2:
        json_data = json.dumps(results, indent=2)
        st.download_button(
            "Download JSON",
            json_data,
            "vectordb_bench_results.json",
            "application/json"
        )

    st.subheader("Raw JSON Preview")
    if results:
        with st.expander("Show first result"):
            st.json(results[0])


if __name__ == '__main__':
    main()
