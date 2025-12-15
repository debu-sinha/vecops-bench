# Contributing to VecOps-Bench

Thank you for your interest in contributing to VecOps-Bench!

## How to Contribute

### Reporting Issues

- Use [GitHub Issues](https://github.com/debu-sinha/vecops-bench/issues)
- Include: OS, Python version, Docker version, error logs
- For benchmark discrepancies, include your hardware specs

### Pull Requests

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/my-feature`
3. **Make changes** following the style guide below
4. **Test** your changes locally
5. **Commit**: `git commit -m "Add: description of change"`
6. **Push**: `git push origin feature/my-feature`
7. **Open PR** against `main` branch

### Commit Message Format

```
Type: Short description

[Optional longer description]

Types:
- Add: New feature
- Fix: Bug fix
- Update: Modification to existing feature
- Docs: Documentation only
- Test: Adding tests
- Refactor: Code restructuring
```

---

## Code Style Guide

### Python

- **Formatter**: `black` (line length 100)
- **Linter**: `flake8`
- **Type hints**: Required for public functions
- **Docstrings**: Google style

```python
def compute_recall(
    retrieved: List[int],
    ground_truth: List[int],
    k: int = 10
) -> float:
    """Compute recall@k metric.

    Args:
        retrieved: List of retrieved document IDs
        ground_truth: List of relevant document IDs
        k: Number of top results to consider

    Returns:
        Recall score between 0 and 1
    """
    return len(set(retrieved[:k]) & set(ground_truth)) / min(k, len(ground_truth))
```

### Running Linters

```bash
# Format code
black src/ scripts/ --line-length 100

# Check style
flake8 src/ scripts/ --max-line-length 100

# Type checking
mypy src/ --ignore-missing-imports
```

---

## Adding a New Database Adapter

1. Create `src/databases/{database}_adapter.py`
2. Inherit from `VectorDBAdapter` base class
3. Implement required methods:

```python
from .base import VectorDBAdapter, QueryResult

class MyDatabaseAdapter(VectorDBAdapter):
    name = "mydatabase"

    def connect(self) -> None:
        """Connect to the database."""
        pass

    def create_collection(self, name: str, dimension: int) -> None:
        """Create a collection/index."""
        pass

    def insert(self, collection: str, vectors: np.ndarray, ids: List[int]) -> None:
        """Insert vectors into collection."""
        pass

    def search(self, collection: str, query: np.ndarray, top_k: int) -> QueryResult:
        """Search for similar vectors."""
        pass

    def disconnect(self) -> None:
        """Disconnect from database."""
        pass
```

4. Register in `src/databases/__init__.py`
5. Add to `docker-compose.yaml`
6. Update README.md

---

## Testing

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_adapters.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Requirements

- Each adapter must have unit tests
- Integration tests require Docker running
- Benchmark tests are slow (marked with `@pytest.mark.slow`)

---

## CI/CD Pipeline

GitHub Actions runs on every PR:

1. **Lint**: black, flake8, mypy
2. **Test**: pytest (unit tests only)
3. **Build**: Docker compose validation

Full benchmark runs are manual (too slow for CI).

---

## Areas for Contribution

- [ ] Additional database adapters (Pinecone, OpenSearch, Vespa)
- [ ] Distributed/cluster mode benchmarks
- [ ] Cost-per-query modeling
- [ ] Hybrid search (dense + sparse)
- [ ] Streaming ingestion benchmarks
- [ ] Multi-tenancy tests

---

## Questions?

- Open a [Discussion](https://github.com/debu-sinha/vecops-bench/discussions)
- Tag maintainer: @debu-sinha
