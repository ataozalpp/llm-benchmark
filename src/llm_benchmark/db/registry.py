"""Construction of the registry repository set."""

from __future__ import annotations

from dataclasses import dataclass

from .engine import create_db_engine, create_session_factory
from .repositories import (
    BenchmarkRunRepository,
    DatasetRepository,
    ModelRepository,
    ProviderEndpointRepository,
    SampleResultRepository,
)


@dataclass(frozen=True)
class RegistryRepositories:
    endpoints: ProviderEndpointRepository
    models: ModelRepository
    datasets: DatasetRepository
    runs: BenchmarkRunRepository
    samples: SampleResultRepository


def create_registry_repositories(database_url: str | None = None) -> RegistryRepositories:
    """Create repositories using the configured database without running migrations."""

    engine = create_db_engine(database_url)
    session_factory = create_session_factory(engine)
    return RegistryRepositories(
        endpoints=ProviderEndpointRepository(session_factory),
        models=ModelRepository(session_factory),
        datasets=DatasetRepository(session_factory),
        runs=BenchmarkRunRepository(session_factory),
        samples=SampleResultRepository(session_factory),
    )
