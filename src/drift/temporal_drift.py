"""
Temporal Drift Simulation for Vector Database Benchmarking.

This module simulates how document corpora evolve over time in production:
- Documents are updated (content changes)
- Documents are deleted (outdated, removed)
- New documents are added (fresh content)
- Embeddings may drift (re-embedding with updated models)

This allows measuring retrieval degradation over time - a critical
production concern not captured by static benchmarks.
"""

import copy
import hashlib
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class DriftType(Enum):
    """Types of corpus drift events."""

    UPDATE = "update"  # Document content modified
    DELETE = "delete"  # Document removed
    ADD = "add"  # New document added
    SEMANTIC_SHIFT = "semantic"  # Meaning changes (same words, different context)
    EMBEDDING_DRIFT = "embed"  # Re-embedded with different model/version


class DriftPattern(Enum):
    """Predefined drift patterns simulating real-world scenarios."""

    STABLE = "stable"  # Minimal changes (archival corpus)
    MODERATE = "moderate"  # Normal business updates
    HIGH_CHURN = "high_churn"  # Fast-moving content (news, social)
    SEASONAL = "seasonal"  # Periodic bursts of updates
    CATASTROPHIC = "catastrophic"  # Major corpus overhaul


@dataclass
class DriftEvent:
    """Records a single drift event."""

    doc_id: str
    drift_type: DriftType
    timestamp: int
    original_content: Optional[str] = None
    new_content: Optional[str] = None
    original_embedding: Optional[np.ndarray] = None
    new_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CorpusSnapshot:
    """Represents corpus state at a specific time point."""

    timestamp: int
    doc_ids: List[str]
    embeddings: np.ndarray
    texts: Optional[List[str]] = None
    metadata: Optional[List[Dict[str, Any]]] = None

    @property
    def size(self) -> int:
        return len(self.doc_ids)


class TemporalDriftSimulator:
    """
    Simulates temporal drift in a document corpus.

    This is the key novel contribution for the paper - measuring how
    retrieval performance degrades as the corpus evolves over time.
    """

    # Predefined drift configurations
    DRIFT_CONFIGS = {
        DriftPattern.STABLE: {
            "update_rate": 0.01,
            "delete_rate": 0.005,
            "add_rate": 0.01,
            "semantic_shift_rate": 0.001,
        },
        DriftPattern.MODERATE: {
            "update_rate": 0.05,
            "delete_rate": 0.02,
            "add_rate": 0.03,
            "semantic_shift_rate": 0.01,
        },
        DriftPattern.HIGH_CHURN: {
            "update_rate": 0.15,
            "delete_rate": 0.08,
            "add_rate": 0.10,
            "semantic_shift_rate": 0.05,
        },
        DriftPattern.SEASONAL: {
            "update_rate": 0.03,  # Base rate, multiplied during bursts
            "delete_rate": 0.01,
            "add_rate": 0.02,
            "semantic_shift_rate": 0.005,
            "burst_probability": 0.2,
            "burst_multiplier": 5.0,
        },
        DriftPattern.CATASTROPHIC: {
            "update_rate": 0.30,
            "delete_rate": 0.20,
            "add_rate": 0.25,
            "semantic_shift_rate": 0.10,
        },
    }

    def __init__(
        self,
        doc_ids: List[str],
        embeddings: np.ndarray,
        texts: Optional[List[str]] = None,
        seed: int = 42,
    ):
        """
        Initialize the drift simulator.

        Args:
            doc_ids: List of document IDs
            embeddings: Document embeddings (num_docs, embedding_dim)
            texts: Optional document texts for content-based drift
            seed: Random seed for reproducibility
        """
        self.original_doc_ids = doc_ids.copy()
        self.original_embeddings = embeddings.copy()
        self.original_texts = texts.copy() if texts else None

        self.current_doc_ids = doc_ids.copy()
        self.current_embeddings = embeddings.copy()
        self.current_texts = texts.copy() if texts else None

        self.embedding_dim = embeddings.shape[1]
        self.rng = np.random.default_rng(seed)
        self.random = random.Random(seed)

        self.current_timestamp = 0
        self.drift_history: List[DriftEvent] = []
        self.snapshots: List[CorpusSnapshot] = []

        # Save initial snapshot
        self._save_snapshot()

        # Counter for new document IDs
        self._new_doc_counter = 0

    def _save_snapshot(self) -> None:
        """Save current corpus state as a snapshot."""
        self.snapshots.append(
            CorpusSnapshot(
                timestamp=self.current_timestamp,
                doc_ids=self.current_doc_ids.copy(),
                embeddings=self.current_embeddings.copy(),
                texts=self.current_texts.copy() if self.current_texts else None,
            )
        )

    def _generate_drifted_embedding(
        self, original: np.ndarray, drift_magnitude: float = 0.1
    ) -> np.ndarray:
        """
        Generate a drifted version of an embedding.

        Simulates how an updated document would have a similar but
        not identical embedding.
        """
        noise = self.rng.normal(0, drift_magnitude, size=original.shape)
        drifted = original + noise
        # Normalize to unit length (for cosine similarity)
        drifted = drifted / np.linalg.norm(drifted)
        return drifted

    def _generate_new_embedding(self) -> np.ndarray:
        """Generate embedding for a new document."""
        # Sample from existing embeddings + noise to maintain distribution
        base_idx = self.rng.integers(len(self.current_embeddings))
        base = self.current_embeddings[base_idx]
        return self._generate_drifted_embedding(base, drift_magnitude=0.3)

    def simulate_update(self, doc_id: str, drift_magnitude: float = 0.1) -> DriftEvent:
        """Simulate updating a document's content."""
        if doc_id not in self.current_doc_ids:
            raise ValueError(f"Document {doc_id} not in corpus")

        idx = self.current_doc_ids.index(doc_id)
        original_embedding = self.current_embeddings[idx].copy()

        # Generate drifted embedding
        new_embedding = self._generate_drifted_embedding(original_embedding, drift_magnitude)
        self.current_embeddings[idx] = new_embedding

        event = DriftEvent(
            doc_id=doc_id,
            drift_type=DriftType.UPDATE,
            timestamp=self.current_timestamp,
            original_embedding=original_embedding,
            new_embedding=new_embedding,
        )
        self.drift_history.append(event)
        return event

    def simulate_delete(self, doc_id: str) -> DriftEvent:
        """Simulate deleting a document."""
        if doc_id not in self.current_doc_ids:
            raise ValueError(f"Document {doc_id} not in corpus")

        idx = self.current_doc_ids.index(doc_id)
        original_embedding = self.current_embeddings[idx].copy()

        # Remove from current corpus
        self.current_doc_ids.pop(idx)
        self.current_embeddings = np.delete(self.current_embeddings, idx, axis=0)
        if self.current_texts:
            self.current_texts.pop(idx)

        event = DriftEvent(
            doc_id=doc_id,
            drift_type=DriftType.DELETE,
            timestamp=self.current_timestamp,
            original_embedding=original_embedding,
        )
        self.drift_history.append(event)
        return event

    def simulate_add(self, doc_id: Optional[str] = None) -> DriftEvent:
        """Simulate adding a new document."""
        if doc_id is None:
            doc_id = f"new_doc_{self.current_timestamp}_{self._new_doc_counter}"
            self._new_doc_counter += 1

        if doc_id in self.current_doc_ids:
            raise ValueError(f"Document {doc_id} already exists")

        new_embedding = self._generate_new_embedding()

        self.current_doc_ids.append(doc_id)
        self.current_embeddings = np.vstack([self.current_embeddings, new_embedding])
        if self.current_texts:
            self.current_texts.append(f"[Synthetic document {doc_id}]")

        event = DriftEvent(
            doc_id=doc_id,
            drift_type=DriftType.ADD,
            timestamp=self.current_timestamp,
            new_embedding=new_embedding,
        )
        self.drift_history.append(event)
        return event

    def simulate_semantic_shift(self, doc_id: str) -> DriftEvent:
        """
        Simulate semantic shift - meaning changes significantly
        even if surface content is similar.
        """
        if doc_id not in self.current_doc_ids:
            raise ValueError(f"Document {doc_id} not in corpus")

        idx = self.current_doc_ids.index(doc_id)
        original_embedding = self.current_embeddings[idx].copy()

        # Larger drift for semantic shift
        new_embedding = self._generate_drifted_embedding(original_embedding, drift_magnitude=0.4)
        self.current_embeddings[idx] = new_embedding

        event = DriftEvent(
            doc_id=doc_id,
            drift_type=DriftType.SEMANTIC_SHIFT,
            timestamp=self.current_timestamp,
            original_embedding=original_embedding,
            new_embedding=new_embedding,
        )
        self.drift_history.append(event)
        return event

    def advance_time(
        self,
        pattern: DriftPattern = DriftPattern.MODERATE,
        custom_config: Optional[Dict[str, float]] = None,
    ) -> CorpusSnapshot:
        """
        Advance simulation by one time step with drift according to pattern.

        Args:
            pattern: Predefined drift pattern
            custom_config: Override specific drift rates

        Returns:
            CorpusSnapshot after drift
        """
        self.current_timestamp += 1

        # Get drift configuration
        config = self.DRIFT_CONFIGS[pattern].copy()
        if custom_config:
            config.update(custom_config)

        # Handle seasonal bursts
        in_burst = False
        if pattern == DriftPattern.SEASONAL:
            if self.rng.random() < config.get("burst_probability", 0.2):
                in_burst = True
                multiplier = config.get("burst_multiplier", 5.0)
                config["update_rate"] *= multiplier
                config["delete_rate"] *= multiplier
                config["add_rate"] *= multiplier

        # Calculate number of each operation
        n_docs = len(self.current_doc_ids)
        n_updates = int(n_docs * config["update_rate"])
        n_deletes = int(n_docs * config["delete_rate"])
        n_adds = int(len(self.original_doc_ids) * config["add_rate"])
        n_semantic = int(n_docs * config.get("semantic_shift_rate", 0))

        # Perform updates
        if n_updates > 0 and n_docs > 0:
            update_ids = self.random.sample(self.current_doc_ids, min(n_updates, n_docs))
            for doc_id in update_ids:
                if doc_id in self.current_doc_ids:  # May have been deleted
                    self.simulate_update(doc_id)

        # Perform deletes
        if n_deletes > 0 and len(self.current_doc_ids) > n_deletes:
            delete_ids = self.random.sample(
                self.current_doc_ids,
                min(n_deletes, len(self.current_doc_ids) - 1),  # Keep at least 1
            )
            for doc_id in delete_ids:
                if doc_id in self.current_doc_ids:
                    self.simulate_delete(doc_id)

        # Perform adds
        for _ in range(n_adds):
            self.simulate_add()

        # Perform semantic shifts
        if n_semantic > 0 and len(self.current_doc_ids) > 0:
            semantic_ids = self.random.sample(
                self.current_doc_ids, min(n_semantic, len(self.current_doc_ids))
            )
            for doc_id in semantic_ids:
                if doc_id in self.current_doc_ids:
                    self.simulate_semantic_shift(doc_id)

        # Save snapshot
        self._save_snapshot()

        return self.snapshots[-1]

    def run_simulation(
        self, num_steps: int, pattern: DriftPattern = DriftPattern.MODERATE
    ) -> List[CorpusSnapshot]:
        """
        Run drift simulation for multiple time steps.

        Args:
            num_steps: Number of time steps to simulate
            pattern: Drift pattern to use

        Returns:
            List of corpus snapshots at each time step
        """
        for _ in range(num_steps):
            self.advance_time(pattern=pattern)

        return self.snapshots

    def get_drift_statistics(self) -> Dict[str, Any]:
        """Get statistics about the drift simulation."""
        events_by_type = {}
        for dt in DriftType:
            events_by_type[dt.value] = sum(1 for e in self.drift_history if e.drift_type == dt)

        # Calculate corpus overlap with original
        original_set = set(self.original_doc_ids)
        current_set = set(self.current_doc_ids)
        overlap = len(original_set & current_set)

        return {
            "num_timestamps": self.current_timestamp,
            "total_drift_events": len(self.drift_history),
            "events_by_type": events_by_type,
            "original_corpus_size": len(self.original_doc_ids),
            "current_corpus_size": len(self.current_doc_ids),
            "corpus_overlap": overlap,
            "survival_rate": overlap / len(self.original_doc_ids) if self.original_doc_ids else 0,
            "churn_rate": 1 - (overlap / len(self.original_doc_ids))
            if self.original_doc_ids
            else 0,
        }

    def get_snapshot_at_time(self, timestamp: int) -> Optional[CorpusSnapshot]:
        """Get corpus snapshot at a specific timestamp."""
        for snapshot in self.snapshots:
            if snapshot.timestamp == timestamp:
                return snapshot
        return None

    def compute_embedding_drift(self, doc_id: str) -> Optional[float]:
        """
        Compute cumulative embedding drift for a document.

        Returns cosine distance from original to current embedding.
        """
        if doc_id not in self.original_doc_ids:
            return None
        if doc_id not in self.current_doc_ids:
            return None  # Document was deleted

        orig_idx = self.original_doc_ids.index(doc_id)
        curr_idx = self.current_doc_ids.index(doc_id)

        orig_emb = self.original_embeddings[orig_idx]
        curr_emb = self.current_embeddings[curr_idx]

        # Cosine distance
        cosine_sim = np.dot(orig_emb, curr_emb) / (
            np.linalg.norm(orig_emb) * np.linalg.norm(curr_emb)
        )
        return 1 - cosine_sim

    def get_average_embedding_drift(self) -> float:
        """Compute average embedding drift across surviving documents."""
        drifts = []
        for doc_id in self.original_doc_ids:
            drift = self.compute_embedding_drift(doc_id)
            if drift is not None:
                drifts.append(drift)

        return np.mean(drifts) if drifts else 0.0
