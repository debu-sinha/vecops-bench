"""Dataset loaders for BEIR and MTEB benchmarks."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


@dataclass
class Document:
    """Represents a document in the corpus."""

    doc_id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """Represents a query with relevance judgments."""

    query_id: str
    text: str
    relevant_docs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddedDataset:
    """Dataset with pre-computed embeddings."""

    name: str
    documents: List[Document]
    queries: List[Query]
    doc_embeddings: np.ndarray  # Shape: (num_docs, embedding_dim)
    query_embeddings: np.ndarray  # Shape: (num_queries, embedding_dim)
    embedding_model: str
    embedding_dim: int

    def __post_init__(self):
        assert len(self.documents) == len(
            self.doc_embeddings
        ), f"Document count mismatch: {len(self.documents)} vs {len(self.doc_embeddings)}"
        assert len(self.queries) == len(
            self.query_embeddings
        ), f"Query count mismatch: {len(self.queries)} vs {len(self.query_embeddings)}"

    def get_doc_id_to_idx(self) -> Dict[str, int]:
        """Get mapping from doc_id to index."""
        return {doc.doc_id: idx for idx, doc in enumerate(self.documents)}

    def get_query_id_to_idx(self) -> Dict[str, int]:
        """Get mapping from query_id to index."""
        return {q.query_id: idx for idx, q in enumerate(self.queries)}

    def sample(
        self, num_docs: Optional[int] = None, num_queries: Optional[int] = None
    ) -> "EmbeddedDataset":
        """Return a sampled subset of the dataset."""
        doc_indices = np.arange(len(self.documents))
        query_indices = np.arange(len(self.queries))

        if num_docs and num_docs < len(self.documents):
            doc_indices = np.random.choice(doc_indices, num_docs, replace=False)

        if num_queries and num_queries < len(self.queries):
            query_indices = np.random.choice(query_indices, num_queries, replace=False)

        # Filter documents
        sampled_docs = [self.documents[i] for i in doc_indices]
        sampled_doc_embeddings = self.doc_embeddings[doc_indices]

        # Filter queries and update relevant docs
        sampled_doc_ids = {doc.doc_id for doc in sampled_docs}
        sampled_queries = []
        sampled_query_indices = []

        for idx in query_indices:
            query = self.queries[idx]
            # Keep query only if it has at least one relevant doc in sampled set
            relevant_in_sample = [d for d in query.relevant_docs if d in sampled_doc_ids]
            if relevant_in_sample:
                sampled_queries.append(
                    Query(
                        query_id=query.query_id,
                        text=query.text,
                        relevant_docs=relevant_in_sample,
                        metadata=query.metadata,
                    )
                )
                sampled_query_indices.append(idx)

        sampled_query_embeddings = self.query_embeddings[sampled_query_indices]

        return EmbeddedDataset(
            name=f"{self.name}_sampled",
            documents=sampled_docs,
            queries=sampled_queries,
            doc_embeddings=sampled_doc_embeddings,
            query_embeddings=sampled_query_embeddings,
            embedding_model=self.embedding_model,
            embedding_dim=self.embedding_dim,
        )


class DatasetLoader(ABC):
    """Abstract base class for dataset loaders."""

    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load(self, dataset_name: str, **kwargs) -> Tuple[List[Document], List[Query]]:
        """Load documents and queries from the dataset."""
        pass

    def embed(
        self,
        documents: List[Document],
        queries: List[Query],
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32,
        cache: bool = True,
    ) -> EmbeddedDataset:
        """Embed documents and queries using sentence-transformers."""
        from sentence_transformers import SentenceTransformer

        # Check cache
        cache_key = self._get_cache_key(documents, queries, embedding_model)
        cache_path = self.cache_dir / f"{cache_key}.npz"

        if cache and cache_path.exists():
            print(f"Loading embeddings from cache: {cache_path}")
            data = np.load(cache_path)
            return EmbeddedDataset(
                name=cache_key,
                documents=documents,
                queries=queries,
                doc_embeddings=data["doc_embeddings"],
                query_embeddings=data["query_embeddings"],
                embedding_model=embedding_model,
                embedding_dim=data["doc_embeddings"].shape[1],
            )

        print(f"Loading embedding model: {embedding_model}")
        model = SentenceTransformer(embedding_model)

        # Embed documents
        print(f"Embedding {len(documents)} documents...")
        doc_texts = [doc.text for doc in documents]
        doc_embeddings = model.encode(
            doc_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        # Embed queries
        print(f"Embedding {len(queries)} queries...")
        query_texts = [q.text for q in queries]
        query_embeddings = model.encode(
            query_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True
        )

        # Cache embeddings
        if cache:
            print(f"Caching embeddings to: {cache_path}")
            np.savez(cache_path, doc_embeddings=doc_embeddings, query_embeddings=query_embeddings)

        return EmbeddedDataset(
            name=cache_key,
            documents=documents,
            queries=queries,
            doc_embeddings=doc_embeddings,
            query_embeddings=query_embeddings,
            embedding_model=embedding_model,
            embedding_dim=doc_embeddings.shape[1],
        )

    def _get_cache_key(
        self, documents: List[Document], queries: List[Query], embedding_model: str
    ) -> str:
        """Generate cache key based on dataset content and model."""
        content = f"{len(documents)}_{len(queries)}_{embedding_model}"
        # Add first and last doc IDs for uniqueness
        if documents:
            content += f"_{documents[0].doc_id}_{documents[-1].doc_id}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class BEIRLoader(DatasetLoader):
    """Loader for BEIR benchmark datasets."""

    AVAILABLE_DATASETS = [
        "msmarco",
        "trec-covid",
        "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "arguana",
        "webis-touche2020",
        "quora",
        "dbpedia-entity",
        "scidocs",
        "fever",
        "climate-fever",
        "scifact",
    ]

    def load(
        self,
        dataset_name: str,
        split: str = "test",
        max_docs: Optional[int] = None,
        max_queries: Optional[int] = None,
    ) -> Tuple[List[Document], List[Query]]:
        """Load a BEIR dataset."""
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Unknown BEIR dataset: {dataset_name}. " f"Available: {self.AVAILABLE_DATASETS}"
            )

        # Download dataset
        data_path = self.cache_dir / "beir" / dataset_name
        if not data_path.exists():
            print(f"Downloading BEIR dataset: {dataset_name}")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            util.download_and_unzip(url, str(self.cache_dir / "beir"))

        # Load data
        corpus, queries_raw, qrels = GenericDataLoader(str(data_path)).load(split=split)

        # Convert to our format
        documents = []
        for doc_id, doc_data in corpus.items():
            documents.append(
                Document(
                    doc_id=doc_id,
                    text=doc_data.get("text", ""),
                    title=doc_data.get("title"),
                    metadata={"source": "beir", "dataset": dataset_name},
                )
            )
            if max_docs and len(documents) >= max_docs:
                break

        # Build doc_id set for filtering queries
        doc_id_set = {doc.doc_id for doc in documents}

        queries = []
        for query_id, query_text in queries_raw.items():
            if query_id in qrels:
                relevant_docs = [
                    doc_id
                    for doc_id, score in qrels[query_id].items()
                    if score > 0 and doc_id in doc_id_set
                ]
                if relevant_docs:  # Only include queries with relevant docs
                    queries.append(
                        Query(
                            query_id=query_id,
                            text=query_text,
                            relevant_docs=relevant_docs,
                            metadata={"source": "beir", "dataset": dataset_name},
                        )
                    )
            if max_queries and len(queries) >= max_queries:
                break

        print(f"Loaded {dataset_name}: {len(documents)} documents, {len(queries)} queries")
        return documents, queries


class MTEBLoader(DatasetLoader):
    """Loader for MTEB retrieval tasks."""

    RETRIEVAL_TASKS = [
        "ArguAna",
        "ClimateFEVER",
        "CQADupstackRetrieval",
        "DBPedia",
        "FEVER",
        "FiQA2018",
        "HotpotQA",
        "MSMARCO",
        "NFCorpus",
        "NQ",
        "QuoraRetrieval",
        "SCIDOCS",
        "SciFact",
        "Touche2020",
        "TRECCOVID",
    ]

    def load(
        self,
        dataset_name: str,
        split: str = "test",
        max_docs: Optional[int] = None,
        max_queries: Optional[int] = None,
    ) -> Tuple[List[Document], List[Query]]:
        """Load an MTEB retrieval dataset."""
        from mteb import MTEB

        from datasets import load_dataset as hf_load_dataset

        # MTEB uses different naming conventions
        task_name = dataset_name

        # Load via HuggingFace datasets (MTEB backend)
        try:
            # Try loading corpus
            if dataset_name.lower() == "msmarco":
                corpus_data = hf_load_dataset("mteb/msmarco", "corpus", split="corpus")
                queries_data = hf_load_dataset("mteb/msmarco", "queries", split="queries")
            else:
                corpus_data = hf_load_dataset(
                    f"mteb/{dataset_name.lower()}", "corpus", split="corpus"
                )
                queries_data = hf_load_dataset(
                    f"mteb/{dataset_name.lower()}", "queries", split="queries"
                )
        except Exception as e:
            print(f"Failed to load MTEB dataset {dataset_name}: {e}")
            print("Falling back to BEIR loader...")
            beir_loader = BEIRLoader(cache_dir=str(self.cache_dir))
            return beir_loader.load(dataset_name.lower(), split, max_docs, max_queries)

        # Convert corpus
        documents = []
        for item in corpus_data:
            documents.append(
                Document(
                    doc_id=str(item.get("_id", item.get("id", len(documents)))),
                    text=item.get("text", ""),
                    title=item.get("title"),
                    metadata={"source": "mteb", "dataset": dataset_name},
                )
            )
            if max_docs and len(documents) >= max_docs:
                break

        # Convert queries (simplified - may need qrels loading)
        queries = []
        for item in queries_data:
            queries.append(
                Query(
                    query_id=str(item.get("_id", item.get("id", len(queries)))),
                    text=item.get("text", ""),
                    relevant_docs=[],  # Would need qrels
                    metadata={"source": "mteb", "dataset": dataset_name},
                )
            )
            if max_queries and len(queries) >= max_queries:
                break

        print(f"Loaded {dataset_name}: {len(documents)} documents, {len(queries)} queries")
        return documents, queries


def load_dataset(
    name: str,
    source: str = "beir",
    split: str = "test",
    max_docs: Optional[int] = None,
    max_queries: Optional[int] = None,
    embed: bool = True,
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    cache_dir: str = "./data/cache",
) -> EmbeddedDataset:
    """
    Convenience function to load and embed a dataset.

    Args:
        name: Dataset name (e.g., "scifact", "nfcorpus", "msmarco")
        source: "beir" or "mteb"
        split: Data split ("test", "train", "dev")
        max_docs: Maximum number of documents to load
        max_queries: Maximum number of queries to load
        embed: Whether to compute embeddings
        embedding_model: Sentence transformer model for embeddings
        cache_dir: Directory for caching downloads and embeddings

    Returns:
        EmbeddedDataset with documents, queries, and embeddings
    """
    if source.lower() == "beir":
        loader = BEIRLoader(cache_dir=cache_dir)
    elif source.lower() == "mteb":
        loader = MTEBLoader(cache_dir=cache_dir)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'beir' or 'mteb'")

    documents, queries = loader.load(name, split=split, max_docs=max_docs, max_queries=max_queries)

    if embed:
        return loader.embed(documents, queries, embedding_model=embedding_model, cache=True)
    else:
        # Return with empty embeddings
        return EmbeddedDataset(
            name=name,
            documents=documents,
            queries=queries,
            doc_embeddings=np.array([]),
            query_embeddings=np.array([]),
            embedding_model="",
            embedding_dim=0,
        )
