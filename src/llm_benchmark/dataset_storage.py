"""Framework-independent content-addressed storage for uploaded datasets."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

DatasetStorageFormat = Literal["csv", "jsonl"]

_SUPPORTED_FORMATS = frozenset({"csv", "jsonl"})
_STORAGE_KEY_PATTERN = re.compile(
    r"upload://sha256/(?P<digest>[0-9a-f]{64})\.(?P<file_format>csv|jsonl)\Z"
)


class DatasetStorageError(Exception):
    """Base class for safe local dataset storage failures."""


class DatasetStoragePolicyError(DatasetStorageError, ValueError):
    """Raised when storage policy values are invalid."""


class UnsupportedDatasetFormatError(DatasetStorageError, ValueError):
    """Raised when a file format is not supported."""


class DatasetFileTooLargeError(DatasetStorageError):
    """Raised when a stream exceeds the configured byte limit."""


class DatasetStreamError(DatasetStorageError):
    """Raised when the input stream cannot be read safely."""


class DatasetStorageWriteError(DatasetStorageError):
    """Raised when temporary storage cannot be created or written."""


class DatasetFinalizationError(DatasetStorageError):
    """Raised when a temporary file cannot be finalized atomically."""


class InvalidStorageKeyError(DatasetStorageError, ValueError):
    """Raised when an opaque storage key is malformed or unsupported."""


@dataclass(frozen=True)
class DatasetStoragePolicy:
    max_file_size_bytes: int = 10 * 1024 * 1024
    chunk_size_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if self.max_file_size_bytes <= 0:
            raise DatasetStoragePolicyError("max_file_size_bytes must be positive")
        if self.chunk_size_bytes <= 0:
            raise DatasetStoragePolicyError("chunk_size_bytes must be positive")


@dataclass(frozen=True)
class StoredDatasetFile:
    storage_key: str
    checksum_sha256: str
    size_bytes: int
    file_format: DatasetStorageFormat
    created_new_file: bool


class LocalDatasetStorage:
    """Store dataset streams beneath a constructor-injected local root."""

    def __init__(
        self,
        root: Path,
        policy: DatasetStoragePolicy | None = None,
    ) -> None:
        self._root = Path(root)
        self._policy = policy or DatasetStoragePolicy()

    def store(
        self,
        stream: BinaryIO,
        file_format: DatasetStorageFormat,
    ) -> StoredDatasetFile:
        normalized_format = self._validate_format(file_format)
        temporary_path: Path | None = None
        digest = hashlib.sha256()
        size_bytes = 0

        try:
            try:
                self._root.mkdir(parents=True, exist_ok=True)
                temporary = tempfile.NamedTemporaryFile(
                    mode="wb",
                    prefix=".dataset-upload-",
                    suffix=".tmp",
                    dir=self._root,
                    delete=False,
                )
                temporary_path = Path(temporary.name)
            except OSError as error:
                raise DatasetStorageWriteError("Dataset temporary storage could not be created") from error

            try:
                with temporary:
                    while True:
                        try:
                            chunk = stream.read(self._policy.chunk_size_bytes)
                        except (OSError, ValueError) as error:
                            raise DatasetStreamError("Dataset stream could not be read") from error
                        if not isinstance(chunk, (bytes, bytearray, memoryview)):
                            raise DatasetStreamError("Dataset stream must return bytes")
                        if not chunk:
                            break

                        chunk_bytes = bytes(chunk)
                        next_size = size_bytes + len(chunk_bytes)
                        if next_size > self._policy.max_file_size_bytes:
                            raise DatasetFileTooLargeError("Dataset exceeds the configured byte limit")
                        try:
                            temporary.write(chunk_bytes)
                        except OSError as error:
                            raise DatasetStorageWriteError("Dataset temporary storage could not be written") from error
                        digest.update(chunk_bytes)
                        size_bytes = next_size
            except DatasetStorageError:
                raise
            except OSError as error:
                raise DatasetStorageWriteError("Dataset temporary storage could not be closed") from error

            checksum = digest.hexdigest()
            storage_key = self._storage_key(checksum, normalized_format)
            final_path = self._path_from_parts(checksum, normalized_format)
            try:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                os.link(temporary_path, final_path)
                created_new_file = True
            except FileExistsError:
                created_new_file = False
            except OSError as error:
                raise DatasetFinalizationError("Dataset could not be finalized") from error

            return StoredDatasetFile(
                storage_key=storage_key,
                checksum_sha256=checksum,
                size_bytes=size_bytes,
                file_format=normalized_format,
                created_new_file=created_new_file,
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass

    def resolve(self, storage_key: str) -> Path:
        match = _STORAGE_KEY_PATTERN.fullmatch(storage_key)
        if match is None:
            raise InvalidStorageKeyError("Dataset storage key is invalid")
        return self._path_from_parts(
            match.group("digest"),
            match.group("file_format"),
        )

    @staticmethod
    def _validate_format(file_format: str) -> DatasetStorageFormat:
        if file_format not in _SUPPORTED_FORMATS:
            raise UnsupportedDatasetFormatError("Dataset format is unsupported")
        return file_format  # type: ignore[return-value]

    @staticmethod
    def _storage_key(checksum: str, file_format: DatasetStorageFormat) -> str:
        return f"upload://sha256/{checksum}.{file_format}"

    def _path_from_parts(self, checksum: str, file_format: str) -> Path:
        return self._root / "sha256" / f"{checksum}.{file_format}"
