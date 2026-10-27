import asyncio
import zipfile

import pytest

from repository import manager


class FakeUpload:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    async def read(self, size):
        return next(self.chunks, b"")


def test_repository_id_resolution_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", tmp_path)

    for repository_id in ("..", "../outside", "..\\outside", "C:\\outside"):
        with pytest.raises(ValueError, match="Invalid repository ID"):
            manager.resolve_repository_path(repository_id)

    assert manager.resolve_repository_path("safe-repository") == (
        tmp_path / "safe-repository"
    )


def test_zip_extraction_rejects_traversal_and_cleans_partial_repository(
    tmp_path,
    monkeypatch,
):
    repository_root = tmp_path / "repositories"
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", repository_root)
    archive_path = tmp_path / "unsafe.zip"
    outside_path = tmp_path / "outside.txt"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("safe.txt", "safe content")
        archive.writestr("../outside.txt", "must not escape")

    with pytest.raises(ValueError, match="unsafe path"):
        manager.extract_repository(str(archive_path), "repository")

    assert not outside_path.exists()
    assert not (repository_root / "repository").exists()


def test_zip_extraction_preserves_repository_files(tmp_path, monkeypatch):
    repository_root = tmp_path / "repositories"
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", repository_root)
    archive_path = tmp_path / "safe.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("src/app.py", "def app():\n    return True\n")

    extracted_path = manager.extract_repository(
        str(archive_path),
        "repository",
    )

    assert (extracted_path / "src" / "app.py").read_text() == (
        "def app():\n    return True\n"
    )
    assert manager.get_supported_files(extracted_path) == [
        extracted_path / "src" / "app.py"
    ]


def test_zip_extraction_does_not_store_credentials_or_private_keys(
    tmp_path,
    monkeypatch,
):
    repository_root = tmp_path / "repositories"
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", repository_root)
    archive_path = tmp_path / "secrets.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("app.py", "print('safe')\n")
        archive.writestr(".env", "API_KEY=secret")
        archive.writestr("credentials.json", '{"token":"secret"}')
        archive.writestr("server.pem", "private key")
        archive.writestr(".env.example", "API_KEY=replace-me")

    extracted_path = manager.extract_repository(
        str(archive_path),
        "repository",
    )

    assert (extracted_path / "app.py").exists()
    assert (extracted_path / ".env.example").exists()
    assert not (extracted_path / ".env").exists()
    assert not (extracted_path / "credentials.json").exists()
    assert not (extracted_path / "server.pem").exists()


def test_zip_extraction_enforces_archive_entry_limit_and_cleans_partial_data(
    tmp_path,
    monkeypatch,
):
    repository_root = tmp_path / "repositories"
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", repository_root)
    monkeypatch.setattr(manager, "MAX_ARCHIVE_MEMBERS", 1)
    archive_path = tmp_path / "too-many-files.zip"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("one.txt", "one")
        archive.writestr("two.txt", "two")

    with pytest.raises(ValueError, match="too many entries"):
        manager.extract_repository(str(archive_path), "repository")

    assert not (repository_root / "repository").exists()


def test_empty_zip_extracts_to_an_empty_repository(tmp_path, monkeypatch):
    repository_root = tmp_path / "repositories"
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", repository_root)
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w"):
        pass

    extracted_path = manager.extract_repository(
        str(archive_path),
        "repository",
    )

    assert manager.get_supported_files(extracted_path) == []


def test_invalid_uploaded_zip_is_rejected_and_temporary_file_removed(
    tmp_path,
    monkeypatch,
):
    upload_dir = tmp_path / "uploads"
    repository_root = tmp_path / "repositories"
    monkeypatch.setattr(manager, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(manager, "BASE_REPOSITORY_DIR", repository_root)
    upload = FakeUpload([b"not a zip"])
    upload.filename = "broken.zip"

    with pytest.raises(ValueError, match="not a valid ZIP"):
        asyncio.run(
            manager.save_uploaded_repository(upload)
        )

    assert list(upload_dir.iterdir()) == []


def test_sensitive_files_are_not_indexed_but_env_templates_are(tmp_path):
    env_file = tmp_path / ".env"
    env_example = tmp_path / ".env.example"
    private_key = tmp_path / "server.pem"
    source_file = tmp_path / "app.py"

    for path in (env_file, env_example, private_key, source_file):
        path.write_text("placeholder")

    assert not manager.is_supported_file(env_file)
    assert manager.is_supported_file(env_example)
    assert not manager.is_supported_file(private_key)
    assert manager.is_supported_file(source_file)


def test_upload_stream_enforces_size_limit(tmp_path):
    upload = FakeUpload([b"1234", b"5678"])
    destination = tmp_path / "upload.zip"

    with pytest.raises(ValueError, match="maximum upload size"):
        asyncio.run(
            manager.stream_upload_to_disk(
                upload,
                destination,
                max_bytes=6,
            )
        )

    assert destination.read_bytes() == b"1234"
