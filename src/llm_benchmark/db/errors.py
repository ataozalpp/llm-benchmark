"""Typed errors exposed by the persistence boundary."""

from __future__ import annotations


class RepositoryError(Exception):
    """Base class for persistence-boundary failures."""


class RecordNotFoundError(RepositoryError):
    def __init__(self, entity: str, record_id: object) -> None:
        super().__init__(f"{entity} record {record_id!r} was not found")
        self.entity = entity
        self.record_id = record_id


class UniquenessConflictError(RepositoryError):
    def __init__(self, entity: str, identity: str) -> None:
        super().__init__(f"{entity} already exists for identity: {identity}")
        self.entity = entity
        self.identity = identity


class InvalidStateTransitionError(RepositoryError):
    def __init__(self, current_status: str, requested_status: str) -> None:
        super().__init__(f"Run status cannot transition from {current_status!r} to {requested_status!r}")
        self.current_status = current_status
        self.requested_status = requested_status


class InactiveDependencyError(RepositoryError):
    def __init__(self, entity: str, record_id: object) -> None:
        super().__init__(f"{entity} record {record_id!r} is inactive")
        self.entity = entity
        self.record_id = record_id
