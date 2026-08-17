"""Synchronous persistence primitives for the benchmark registry."""

from .base import Base
from .engine import (
    DATABASE_URL_ENV,
    DEFAULT_DATABASE_URL,
    create_db_engine,
    create_session_factory,
    get_database_url,
)
from .errors import (
    InactiveDependencyError,
    InvalidStateTransitionError,
    RecordNotFoundError,
    RepositoryError,
    UniquenessConflictError,
)
from .repositories import (
    BenchmarkRunRepository,
    DatasetRepository,
    ModelRepository,
    ProviderEndpointRepository,
    SampleResultRepository,
)

__all__ = [
    "Base",
    "DATABASE_URL_ENV",
    "DEFAULT_DATABASE_URL",
    "create_db_engine",
    "create_session_factory",
    "get_database_url",
    "BenchmarkRunRepository",
    "DatasetRepository",
    "InactiveDependencyError",
    "InvalidStateTransitionError",
    "ModelRepository",
    "ProviderEndpointRepository",
    "RecordNotFoundError",
    "RepositoryError",
    "SampleResultRepository",
    "UniquenessConflictError",
]
