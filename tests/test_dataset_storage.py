from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path

import pytest

from llm_benchmark.dataset_storage import (
    DatasetFileTooLargeError,
    DatasetFinalizationError,
    DatasetStorageFileMissingError,
    DatasetStorageIntegrityError,
    DatasetStoragePolicy,
    DatasetStoragePolicyError,
    DatasetStorageReadError,
    DatasetStorageWriteError,
    DatasetStreamError,
    InvalidDatasetChecksumError,
    InvalidStorageKeyError,
    LocalDatasetStorage,
    ParsedDatasetStorageKey,
    StoredDatasetFile,
    UnsupportedDatasetFormatError,
    parse_storage_key,
)


class BoundedReadStream:
    def __init__(self, content: bytes) -> None:
        self._content = content
        self._offset = 0
        self.requested_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.requested_sizes.append(size)
        if size <= 0:
            raise AssertionError("storage must request a positive bounded read")
        start = self._offset
        self._offset = min(len(self._content), start + size)
        return self._content[start:self._offset]


class FailingReadStream:
    def __init__(self, first_chunk: bytes) -> None:
        self._first_chunk = first_chunk
        self._read_count = 0

    def read(self, size: int = -1) -> bytes:
        self._read_count += 1
        if self._read_count == 1:
            return self._first_chunk[:size]
        raise OSError("private stream failure")


def _stored_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if path.is_file()]


@pytest.mark.parametrize(("file_format", "content"), [
    ("csv", b"sample_id,question\n001,Question\n"),
    ("jsonl", b'{"sample_id":"001"}\n'),
])
def test_store_preserves_exact_content_size_and_checksum(
    tmp_path: Path,
    file_format: str,
    content: bytes,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")

    result = storage.store(BytesIO(content), file_format)  # type: ignore[arg-type]

    checksum = hashlib.sha256(content).hexdigest()
    assert result == StoredDatasetFile(
        storage_key=f"upload://sha256/{checksum}.{file_format}",
        checksum_sha256=checksum,
        size_bytes=len(content),
        file_format=file_format,
        created_new_file=True,
    )
    assert storage.resolve(result.storage_key).read_bytes() == content
    assert str(tmp_path.resolve()) not in repr(result)


def test_store_uses_only_configured_bounded_reads(tmp_path: Path) -> None:
    content = b"abcdefghij"
    stream = BoundedReadStream(content)
    storage = LocalDatasetStorage(
        tmp_path / "datasets",
        DatasetStoragePolicy(max_file_size_bytes=20, chunk_size_bytes=3),
    )

    storage.store(stream, "csv")

    assert stream.requested_sizes
    assert set(stream.requested_sizes) == {3}


def test_exact_size_limit_is_accepted(tmp_path: Path) -> None:
    storage = LocalDatasetStorage(
        tmp_path / "datasets",
        DatasetStoragePolicy(max_file_size_bytes=4, chunk_size_bytes=2),
    )

    result = storage.store(BytesIO(b"1234"), "csv")

    assert result.size_bytes == 4


def test_one_byte_over_limit_is_rejected_and_temporary_file_is_cleaned(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(
        root,
        DatasetStoragePolicy(max_file_size_bytes=4, chunk_size_bytes=2),
    )

    with pytest.raises(DatasetFileTooLargeError):
        storage.store(BytesIO(b"12345"), "csv")

    assert _stored_files(root) == []


def test_stream_read_failure_cleans_temporary_file(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)

    with pytest.raises(DatasetStreamError):
        storage.store(FailingReadStream(b"partial"), "jsonl")  # type: ignore[arg-type]

    assert _stored_files(root) == []


def test_temporary_write_failure_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)
    real_named_temporary_file = tempfile.NamedTemporaryFile

    class FailingTemporaryFile:
        def __init__(self) -> None:
            self._file = real_named_temporary_file(
                mode="wb",
                prefix=".dataset-upload-",
                suffix=".tmp",
                dir=root,
                delete=False,
            )
            self.name = self._file.name

        def __enter__(self) -> "FailingTemporaryFile":
            return self

        def __exit__(self, *args: object) -> None:
            self._file.close()

        def write(self, content: bytes) -> int:
            raise OSError("private write failure")

    monkeypatch.setattr(
        "llm_benchmark.dataset_storage.tempfile.NamedTemporaryFile",
        lambda **kwargs: FailingTemporaryFile(),
    )

    with pytest.raises(DatasetStorageWriteError):
        storage.store(BytesIO(b"content"), "csv")

    assert _stored_files(root) == []


def test_finalization_failure_cleans_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)

    def fail_link(source: object, destination: object) -> None:
        raise OSError("private finalization failure")

    monkeypatch.setattr(os, "link", fail_link)

    with pytest.raises(DatasetFinalizationError):
        storage.store(BytesIO(b"content"), "csv")

    assert _stored_files(root) == []


def test_duplicate_content_reuses_existing_file_without_deleting_it(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)
    content = b"same content"

    first = storage.store(BytesIO(content), "jsonl")
    second = storage.store(BytesIO(content), "jsonl")

    assert first.created_new_file is True
    assert second.created_new_file is False
    assert second.storage_key == first.storage_key
    assert storage.resolve(first.storage_key).read_bytes() == content
    assert _stored_files(root) == [storage.resolve(first.storage_key)]


def test_failed_duplicate_stream_does_not_delete_existing_file(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)
    content = b"existing content"
    existing = storage.store(BytesIO(content), "jsonl")

    with pytest.raises(DatasetStreamError):
        storage.store(FailingReadStream(content), "jsonl")  # type: ignore[arg-type]

    final_path = storage.resolve(existing.storage_key)
    assert final_path.read_bytes() == content
    assert _stored_files(root) == [final_path]


def test_different_content_uses_different_files(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)

    first = storage.store(BytesIO(b"first"), "csv")
    second = storage.store(BytesIO(b"second"), "csv")

    assert first.storage_key != second.storage_key
    assert len(_stored_files(root)) == 2


def test_same_content_with_different_formats_uses_different_files(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)

    csv_result = storage.store(BytesIO(b"content"), "csv")
    jsonl_result = storage.store(BytesIO(b"content"), "jsonl")

    assert csv_result.checksum_sha256 == jsonl_result.checksum_sha256
    assert csv_result.storage_key != jsonl_result.storage_key
    assert len(_stored_files(root)) == 2


def test_stored_result_is_immutable(tmp_path: Path) -> None:
    result = LocalDatasetStorage(tmp_path / "datasets").store(BytesIO(b"content"), "csv")

    with pytest.raises(FrozenInstanceError):
        result.size_bytes = 0  # type: ignore[misc]


@pytest.mark.parametrize("invalid_format", ["xlsx", "CSV", "", "../csv"])
def test_unsupported_format_is_rejected_without_side_effects(
    tmp_path: Path,
    invalid_format: str,
) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)

    with pytest.raises(UnsupportedDatasetFormatError):
        storage.store(BytesIO(b"content"), invalid_format)  # type: ignore[arg-type]

    assert not root.exists()


@pytest.mark.parametrize(
    ("storage_key", "expected"),
    [
        (
            f"upload://sha256/{'a' * 64}.csv",
            ParsedDatasetStorageKey(
                digest="a" * 64,
                file_format="csv",
            ),
        ),
        (
            f"upload://sha256/{'b' * 64}.jsonl",
            ParsedDatasetStorageKey(
                digest="b" * 64,
                file_format="jsonl",
            ),
        ),
    ],
)
def test_parse_storage_key_returns_validated_parts(
    storage_key: str,
    expected: ParsedDatasetStorageKey,
) -> None:
    assert parse_storage_key(storage_key) == expected


@pytest.mark.parametrize(
    "storage_key",
    [
        "",
        "upload://sha256/abc.csv",
        f"upload://sha256/{'A' * 64}.csv",
        f"upload://sha256/{'a' * 64}.txt",
        f"upload://sha256/../{'a' * 64}.csv",
        f"upload://sha256/{'a' * 64}.csv/../other",
        f"file://sha256/{'a' * 64}.csv",
        f"upload://sha256/{'a' * 64}.csv?path=../other",
    ],
)
def test_parse_storage_key_rejects_invalid_values(
    storage_key: str,
) -> None:
    with pytest.raises(InvalidStorageKeyError):
        parse_storage_key(storage_key)


def test_parsed_storage_key_is_immutable() -> None:
    parsed = parse_storage_key(
        f"upload://sha256/{'a' * 64}.csv"
    )

    with pytest.raises(FrozenInstanceError):
        parsed.digest = "b" * 64  # type: ignore[misc]


@pytest.mark.parametrize(
    "storage_key",
    [
        "",
        "upload://sha256/abc.csv",
        f"upload://sha256/{'A' * 64}.csv",
        f"upload://sha256/{'a' * 64}.txt",
        f"upload://sha256/../{'a' * 64}.csv",
        f"upload://sha256/{'a' * 64}.csv/../other",
        f"file://sha256/{'a' * 64}.csv",
        f"upload://sha256/{'a' * 64}.csv?path=../other",
    ],
)


def test_resolve_rejects_malformed_or_traversal_like_keys(
    tmp_path: Path,
    storage_key: str,
) -> None:
    storage = LocalDatasetStorage(
        tmp_path / "datasets"
    )

    with pytest.raises(InvalidStorageKeyError):
        storage.resolve(storage_key)


@pytest.mark.parametrize(
    ("max_file_size_bytes", "chunk_size_bytes"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
)
def test_policy_values_must_be_positive(
    max_file_size_bytes: int,
    chunk_size_bytes: int,
) -> None:
    with pytest.raises(DatasetStoragePolicyError):
        DatasetStoragePolicy(
            max_file_size_bytes=max_file_size_bytes,
            chunk_size_bytes=chunk_size_bytes,
        )


def test_import_and_constructor_have_no_filesystem_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    LocalDatasetStorage(root)
    assert not root.exists()

    project_src = Path(__file__).parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_src)
    completed = subprocess.run(
        [sys.executable, "-c", "import llm_benchmark.dataset_storage"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(tmp_path.iterdir()) == []


def test_remove_deletes_only_resolved_stored_file(
        tmp_path: Path,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    stored = storage.store(
        BytesIO(b"dataset content"),
        "csv",
    )

    storage.remove(stored.storage_key)

    assert not storage.resolve(stored.storage_key).exists()


def test_remove_is_idempotent(
        tmp_path: Path
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    stored = storage.store(
        BytesIO(b"dataset content"),
        "jsonl",
    )

    storage.remove(stored.storage_key)
    storage.remove(stored.storage_key)

    assert not storage.resolve(stored.storage_key).exists()


@pytest.mark.parametrize("file_format", ["csv", "jsonl"])
def test_verify_accepts_valid_stored_content(
    tmp_path: Path,
    file_format: str,
) -> None:
    storage = LocalDatasetStorage(
        tmp_path / "datasets",
        DatasetStoragePolicy(chunk_size_bytes=3),
    )
    content = b"content spanning several bounded chunks"
    stored = storage.store(BytesIO(content), file_format)  # type: ignore[arg-type]

    verified = storage.verify(
        stored.storage_key,
        f"sha256:{stored.checksum_sha256}",
    )

    assert verified.read_bytes() == content


def test_verify_rejects_malformed_checksum_without_side_effects(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    storage = LocalDatasetStorage(root)

    with pytest.raises(InvalidDatasetChecksumError, match="metadata is invalid"):
        storage.verify(f"upload://sha256/{'a' * 64}.csv", "sha256:not-a-digest")

    assert not root.exists()


def test_verify_rejects_storage_key_checksum_mismatch(tmp_path: Path) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")

    with pytest.raises(DatasetStorageIntegrityError, match="storage key"):
        storage.verify(
            f"upload://sha256/{'a' * 64}.csv",
            f"sha256:{'b' * 64}",
        )


def test_verify_rejects_missing_and_tampered_files(tmp_path: Path) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    missing_key = f"upload://sha256/{'a' * 64}.csv"

    with pytest.raises(DatasetStorageFileMissingError, match="unavailable"):
        storage.verify(missing_key, f"sha256:{'a' * 64}")

    stored = storage.store(BytesIO(b"original"), "csv")
    storage.resolve(stored.storage_key).write_bytes(b"tampered private content")
    with pytest.raises(DatasetStorageIntegrityError, match="integrity validation") as captured:
        storage.verify(stored.storage_key, f"sha256:{stored.checksum_sha256}")
    assert "tampered" not in str(captured.value)
    assert str(tmp_path) not in str(captured.value)


def test_verify_maps_file_read_failure_to_safe_typed_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = LocalDatasetStorage(tmp_path / "datasets")
    stored = storage.store(BytesIO(b"content"), "jsonl")

    def fail_open(*_: object, **__: object) -> object:
        raise PermissionError("private path and content")

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(DatasetStorageReadError, match="could not be read") as captured:
        storage.verify(stored.storage_key, f"sha256:{stored.checksum_sha256}")
    assert "private" not in str(captured.value)


def test_verify_rejects_resolved_path_outside_storage_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    root.mkdir()
    outside_content = b"private external dataset content"
    digest = hashlib.sha256(outside_content).hexdigest()
    outside_path = tmp_path / "private-external.csv"
    outside_path.write_bytes(outside_content)
    storage = LocalDatasetStorage(root)

    monkeypatch.setattr(
        storage,
        "_path_from_parts",
        lambda *_: outside_path,
    )

    with pytest.raises(DatasetStorageIntegrityError, match="containment") as captured:
        storage.verify(
            f"upload://sha256/{digest}.csv",
            f"sha256:{digest}",
        )

    assert str(tmp_path) not in str(captured.value)
    assert "private external" not in str(captured.value)


@pytest.mark.parametrize("link_kind", ["file", "directory"])
def test_verify_rejects_symlink_escape_when_supported(
    tmp_path: Path,
    link_kind: str,
) -> None:
    root = tmp_path / "datasets"
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    content = b"external dataset content"
    digest = hashlib.sha256(content).hexdigest()
    outside_file = outside_root / f"{digest}.csv"
    outside_file.write_bytes(content)
    root.mkdir()

    try:
        if link_kind == "directory":
            (root / "sha256").symlink_to(
                outside_root,
                target_is_directory=True,
            )
        else:
            stored_directory = root / "sha256"
            stored_directory.mkdir()
            (stored_directory / f"{digest}.csv").symlink_to(outside_file)
    except OSError as error:
        pytest.skip(f"filesystem symlinks are unavailable: {type(error).__name__}")

    storage = LocalDatasetStorage(root)
    with pytest.raises(DatasetStorageIntegrityError, match="containment") as captured:
        storage.verify(
            f"upload://sha256/{digest}.csv",
            f"sha256:{digest}",
        )

    assert str(tmp_path) not in str(captured.value)
    assert "external dataset" not in str(captured.value)
