import os
import shutil
import uuid
import zipfile
from pathlib import Path
from pathlib import PurePosixPath

from config import REPOSITORIES_DIR, DATA_DIR


# ============================================================
# DIRECTORIES
# ============================================================

BASE_REPOSITORY_DIR = REPOSITORIES_DIR
UPLOAD_DIR = DATA_DIR / "uploads"


# ============================================================
# DIRECTORY FILTERING
# ============================================================

IGNORED_DIRECTORIES = {
    ".git", ".hg", ".svn",
    "node_modules",
    "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "virtualenv",
    ".idea", ".vscode",
    "dist", "build", "out",
    ".next", ".nuxt",
    "coverage", ".coverage",
    "target", "vendor",
}


# ============================================================
# KNOWN FILE EXTENSIONS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".py", ".pyw", ".ipynb",
    ".js", ".jsx", ".ts", ".tsx",
    ".java",
    ".c", ".cpp", ".h", ".hpp",
    ".cs", ".go", ".rs", ".php",
    ".html", ".htm", ".css", ".scss", ".sass",
    ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf",
    ".md", ".rst", ".txt",
    ".sql", ".xml",
    ".sh", ".bash", ".bat", ".ps1",
}


IGNORED_FILES = {
    ".ds_store",
    "thumbs.db",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "secrets.json",
    "id_rsa",
    "id_ed25519",
}

SAFE_ENV_EXAMPLE_FILES = {
    ".env.example",
    ".env.sample",
    ".env.template",
}

SENSITIVE_FILE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}


# ============================================================
# FILE SIZE LIMITS
# ============================================================

MAX_UPLOAD_SIZE = 524_288_000
MAX_EXTRACTED_REPOSITORY_SIZE = 500 * 1024 * 1024
MAX_SINGLE_FILE_SIZE = 25 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 20_000


# ============================================================
# REPOSITORY ID
# ============================================================

def create_repository_id() -> str:
    """Create a unique repository ID."""
    return uuid.uuid4().hex[:12]


def resolve_repository_path(repository_id: str) -> Path:
    """Resolve a repository ID without permitting path traversal."""
    if (
        not isinstance(repository_id, str)
        or not repository_id
        or repository_id in {".", ".."}
        or "/" in repository_id
        or "\\" in repository_id
    ):
        raise ValueError("Invalid repository ID.")

    repository_root = BASE_REPOSITORY_DIR.resolve()
    repository_path = (repository_root / repository_id).resolve()

    if repository_path.parent != repository_root:
        raise ValueError("Invalid repository ID.")

    return repository_path


# ============================================================
# ZIP VALIDATION
# ============================================================

def validate_zip_file(zip_path: str) -> None:
    """Validate that the uploaded file is a ZIP archive."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Uploaded file is not a valid ZIP archive.")


# ============================================================
# SAFE ZIP PATH VALIDATION
# ============================================================

def is_safe_zip_path(
    extraction_root: Path,
    member_name: str,
) -> bool:
    """Prevent ZIP path traversal attacks."""

    if not member_name:
        return False

    normalized_name = member_name.replace("\\", "/")

    parts = PurePosixPath(normalized_name).parts
    if (
        normalized_name.startswith("/")
        or ":" in normalized_name
        or not parts
        or any(part in {".", ".."} for part in parts)
    ):
        return False

    target_path = (extraction_root / normalized_name).resolve()
    root_resolved = extraction_root.resolve()

    try:
        target_path.relative_to(root_resolved)
    except ValueError:
        return False

    return True


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def extract_repository(
    zip_path: str,
    repository_id: str,
) -> Path:
    """Safely extract repository files."""

    BASE_REPOSITORY_DIR.mkdir(parents=True, exist_ok=True)

    repository_path = resolve_repository_path(repository_id)

    if repository_path.exists():
        raise ValueError("Repository ID already exists.")

    repository_path.mkdir(parents=True, exist_ok=True)

    total_extracted_size = 0
    seen_members = set()

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            members = zip_ref.infolist()
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError(
                    "Repository archive contains too many entries."
                )

            for member in members:
                member_name = member.filename

                if not is_safe_zip_path(repository_path, member_name):
                    raise ValueError(
                        f"ZIP archive contains an unsafe path: {member_name}"
                    )

                normalized_name = PurePosixPath(
                    member_name.replace("\\", "/")
                ).as_posix().rstrip("/")
                duplicate_key = os.path.normcase(normalized_name)
                if not normalized_name or duplicate_key in seen_members:
                    raise ValueError(
                        "ZIP archive contains duplicate or empty paths."
                    )
                seen_members.add(duplicate_key)

                normalized_path = Path(*PurePosixPath(normalized_name).parts)
                normalized_parts = normalized_path.parts

                if any(
                    part.lower() in IGNORED_DIRECTORIES
                    for part in normalized_parts
                ):
                    continue

                filename = normalized_path.name.lower()
                if (
                    filename in IGNORED_FILES
                    or (
                        filename.startswith(".env")
                        and filename not in SAFE_ENV_EXAMPLE_FILES
                    )
                    or normalized_path.suffix.lower() in SENSITIVE_FILE_SUFFIXES
                ):
                    continue

                if member.is_dir():
                    (repository_path / member_name).mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    continue

                file_size = member.file_size

                if file_size > MAX_SINGLE_FILE_SIZE:
                    print(f"Skipping oversized file: {member_name}")
                    continue

                target_path = repository_path / member_name
                target_path.parent.mkdir(parents=True, exist_ok=True)

                extracted_file_size = 0
                with zip_ref.open(member, "r") as source:
                    with open(target_path, "wb") as destination:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break

                            extracted_file_size += len(chunk)
                            total_extracted_size += len(chunk)

                            if extracted_file_size > MAX_SINGLE_FILE_SIZE:
                                raise ValueError(
                                    "A repository file exceeds the 25 MB limit."
                                )

                            if total_extracted_size > MAX_EXTRACTED_REPOSITORY_SIZE:
                                raise ValueError(
                                    "Extracted repository exceeds the 500 MB limit."
                                )

                            destination.write(chunk)

        return repository_path

    except Exception:
        if repository_path.exists():
            shutil.rmtree(repository_path, ignore_errors=True)
        raise


# ============================================================
# PATH FILTERING
# ============================================================

def is_ignored_path(path: Path) -> bool:
    """Check whether a path contains an ignored directory."""
    return any(part in IGNORED_DIRECTORIES for part in path.parts)


# ============================================================
# CONTENT DETECTION
# ============================================================

def is_probably_text_file(file_path: Path) -> bool:
    """
    Detect likely text files without relying only on extensions.

    This is a lightweight heuristic. It does not guarantee that
    every discovered format can be parsed or indexed correctly.
    """

    try:
        with file_path.open("rb") as file:
            sample = file.read(8192)
    except OSError:
        return False

    # Empty files can be discovered.
    if not sample:
        return True

    # NUL bytes commonly indicate binary or UTF-16 content.
    if b"\x00" in sample:
        return False

    allowed_controls = {7, 8, 9, 10, 12, 13, 27}

    control_count = sum(
        1
        for byte in sample
        if byte < 32 and byte not in allowed_controls
    )

    return control_count / len(sample) <= 0.05


# ============================================================
# FILE SUPPORT CHECK
# ============================================================

def is_supported_file(file_path: Path) -> bool:
    """
    Allow known extensions and unfamiliar files that appear
    to contain readable text.
    """

    filename = file_path.name.lower()
    if (
        filename in IGNORED_FILES
        or (
            filename.startswith(".env")
            and filename not in SAFE_ENV_EXAMPLE_FILES
        )
        or file_path.suffix.lower() in SENSITIVE_FILE_SUFFIXES
    ):
        return False

    if is_ignored_path(file_path):
        return False

    try:
        if file_path.stat().st_size > MAX_SINGLE_FILE_SIZE:
            return False
    except OSError:
        return False

    # An unknown extension is not automatically rejected.
    return is_probably_text_file(file_path)


# ============================================================
# GET SUPPORTED FILES
# ============================================================

def get_supported_files(repository_path: Path) -> list[Path]:
    """Discover eligible files recursively."""

    supported_files = []

    for file_path in repository_path.rglob("*"):
        if not file_path.is_file():
            continue

        if not is_supported_file(file_path):
            continue

        supported_files.append(file_path)

    supported_files.sort(key=lambda path: str(path).lower())

    return supported_files


# ============================================================
# REPOSITORY INFORMATION
# ============================================================

def get_repository_info(repository_path: Path) -> dict:
    """Return repository metadata."""

    files = get_supported_files(repository_path)
    extensions = {}

    for file_path in files:
        extension = file_path.suffix.lower()
        extensions[extension] = extensions.get(extension, 0) + 1

    return {
        "repository_id": repository_path.name,
        "repository_path": str(repository_path),
        "file_count": len(files),
        "file_types": extensions,
        "files": [
            str(file_path.relative_to(repository_path))
            for file_path in files
        ],
    }


# ============================================================
# STREAM UPLOADED ZIP TO DISK
# ============================================================

async def stream_upload_to_disk(
    upload_file,
    destination_path: Path,
    max_bytes: int = MAX_UPLOAD_SIZE,
) -> None:
    """Stream the uploaded ZIP to disk in chunks."""

    total_bytes = 0

    with open(destination_path, "wb") as output_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)

            if not chunk:
                break

            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise ValueError(
                    "Uploaded archive exceeds the maximum upload size."
                )

            output_file.write(chunk)


# ============================================================
# SAVE UPLOADED REPOSITORY
# ============================================================

async def save_uploaded_repository(upload_file) -> dict:
    """Save, extract, and inspect an uploaded repository ZIP."""

    if not upload_file.filename:
        raise ValueError("Uploaded file has no filename.")

    filename = upload_file.filename.lower()

    if not filename.endswith(".zip"):
        raise ValueError("Only ZIP files are supported.")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    BASE_REPOSITORY_DIR.mkdir(parents=True, exist_ok=True)

    repository_id = create_repository_id()
    zip_path = UPLOAD_DIR / f"{repository_id}.zip"

    try:
        print(f"Saving uploaded repository: {upload_file.filename}")

        await stream_upload_to_disk(
            upload_file=upload_file,
            destination_path=zip_path,
        )

        validate_zip_file(str(zip_path))

        repository_path = extract_repository(
            zip_path=str(zip_path),
            repository_id=repository_id,
        )

        repository_info = get_repository_info(repository_path)

        print("Repository information collected.")
        print(f"Files: {repository_info['file_count']}")
        print(f"File types: {repository_info['file_types']}")

        return repository_info

    finally:
        if zip_path.exists():
            try:
                zip_path.unlink()
            except OSError:
                pass