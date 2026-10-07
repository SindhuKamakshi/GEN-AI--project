import shutil
import uuid
import zipfile
from pathlib import Path

from config import REPOSITORIES_DIR, DATA_DIR


# ============================================================
# DIRECTORIES
# ============================================================

BASE_REPOSITORY_DIR = REPOSITORIES_DIR
UPLOAD_DIR = DATA_DIR / "uploads"


# ============================================================
# DIRECTORY FILTERING
# ============================================================

# Directories that should not be indexed because they normally
# contain dependencies, generated files, caches, or Git metadata.
IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",

    "node_modules",

    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",

    ".venv",
    "venv",
    "env",
    "virtualenv",

    ".idea",
    ".vscode",

    "dist",
    "build",
    "out",
    ".next",
    ".nuxt",

    "coverage",
    ".coverage",

    "target",

    "vendor",
}


# ============================================================
# FILE FILTERING
# ============================================================

SUPPORTED_EXTENSIONS = {
    # Python
    ".py",
    ".pyw",

    # JavaScript / TypeScript
    ".js",
    ".jsx",
    ".ts",
    ".tsx",

    # Java
    ".java",

    # C / C++
    ".c",
    ".cpp",
    ".h",
    ".hpp",

    # C#
    ".cs",

    # Go
    ".go",

    # Rust
    ".rs",

    # PHP
    ".php",

    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",

    # Data / configuration
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",

    # Documentation
    ".md",
    ".rst",
    ".txt",

    # Database
    ".sql",

    # XML
    ".xml",

    # Shell
    ".sh",
    ".bash",
    ".bat",
    ".ps1",
}


# Files that should not be indexed even if their extension is
# technically supported.
IGNORED_FILES = {
    ".ds_store",
    "thumbs.db",
}


# ============================================================
# LIMITS
# ============================================================

# We intentionally do NOT impose the old 500 MB ZIP upload limit.
#
# The ZIP is streamed directly to disk, so the application does
# not need to load the entire archive into RAM.
#
# These limits protect the application from pathological ZIPs
# and extremely large individual files.
#
# Increase them later if necessary.

MAX_EXTRACTED_REPOSITORY_SIZE = 10 * 1024 * 1024 * 1024
# 10 GB maximum extracted repository size.

MAX_SINGLE_FILE_SIZE = 500 * 1024 * 1024
# 500 MB maximum for one individual file.


# ============================================================
# REPOSITORY ID
# ============================================================

def create_repository_id() -> str:
    """
    Create a short unique repository ID.
    """

    return uuid.uuid4().hex[:12]


# ============================================================
# ZIP VALIDATION
# ============================================================

def validate_zip_file(zip_path: str) -> None:
    """
    Validate that the uploaded file is a valid ZIP archive.
    """

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(
            "Uploaded file is not a valid ZIP archive."
        )


# ============================================================
# SAFE ZIP PATH VALIDATION
# ============================================================

def is_safe_zip_path(
    extraction_root: Path,
    member_name: str,
) -> bool:
    """
    Prevent ZIP-slip/path-traversal attacks.

    A malicious ZIP could contain something like:

        ../../some_file

    which could otherwise write files outside the repository
    directory.

    This function ensures every extracted file remains inside
    the intended repository directory.
    """

    if not member_name:
        return False

    # Convert ZIP separators to the local platform separator.
    normalized_name = member_name.replace("\\", "/")

    # Reject absolute paths.
    if normalized_name.startswith("/"):
        return False

    # Reject Windows drive paths such as C:/...
    if len(normalized_name) >= 2:
        if normalized_name[1] == ":":
            return False

    # Resolve the final extraction path.
    target_path = (
        extraction_root / normalized_name
    ).resolve()

    extraction_root_resolved = (
        extraction_root.resolve()
    )

    try:
        target_path.relative_to(
            extraction_root_resolved
        )
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
    """
    Safely extract a repository ZIP archive.

    Features:
    - ZIP-slip protection
    - extracted-size protection
    - individual-file-size protection
    - skips ignored directories
    - streams files from ZIP to disk
    """

    BASE_REPOSITORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    repository_path = (
        BASE_REPOSITORY_DIR / repository_id
    )

    # Remove an existing repository with the same ID.
    if repository_path.exists():
        shutil.rmtree(repository_path)

    repository_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_extracted_size = 0

    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as zip_ref:

            members = zip_ref.infolist()

            for member in members:

                member_name = member.filename

                # ------------------------------------------------
                # SAFE PATH CHECK
                # ------------------------------------------------

                if not is_safe_zip_path(
                    repository_path,
                    member_name,
                ):
                    raise ValueError(
                        "ZIP archive contains an unsafe file path: "
                        f"{member_name}"
                    )

                # ------------------------------------------------
                # NORMALIZE PATH
                # ------------------------------------------------

                normalized_parts = Path(
                    member_name.replace("\\", "/")
                ).parts

                # ------------------------------------------------
                # SKIP IGNORED DIRECTORIES
                # ------------------------------------------------

                if any(
                    part in IGNORED_DIRECTORIES
                    for part in normalized_parts
                ):
                    continue

                # ------------------------------------------------
                # DIRECTORY
                # ------------------------------------------------

                if member.is_dir():
                    directory_path = (
                        repository_path
                        / member_name
                    )

                    directory_path.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    continue

                # ------------------------------------------------
                # FILE SIZE PROTECTION
                # ------------------------------------------------

                file_size = member.file_size

                if file_size > MAX_SINGLE_FILE_SIZE:
                    print(
                        "Skipping oversized file: "
                        f"{member_name} "
                        f"({file_size / (1024 * 1024):.2f} MB)"
                    )
                    continue

                # ------------------------------------------------
                # TOTAL SIZE PROTECTION
                # ------------------------------------------------

                total_extracted_size += file_size

                if (
                    total_extracted_size
                    > MAX_EXTRACTED_REPOSITORY_SIZE
                ):
                    raise ValueError(
                        "Extracted repository is too large. "
                        "Maximum supported extracted size is "
                        "10 GB."
                    )

                # ------------------------------------------------
                # CREATE TARGET PATH
                # ------------------------------------------------

                target_path = (
                    repository_path
                    / member_name
                )

                target_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # ------------------------------------------------
                # STREAM FILE FROM ZIP TO DISK
                # ------------------------------------------------

                with zip_ref.open(
                    member,
                    "r",
                ) as source:

                    with open(
                        target_path,
                        "wb",
                    ) as destination:

                        while True:

                            chunk = source.read(
                                1024 * 1024
                            )

                            if not chunk:
                                break

                            destination.write(
                                chunk
                            )

        return repository_path

    except Exception:

        # Clean up partially extracted repository
        # if extraction fails.
        if repository_path.exists():
            shutil.rmtree(
                repository_path,
                ignore_errors=True,
            )

        raise


# ============================================================
# PATH FILTERING
# ============================================================

def is_ignored_path(
    path: Path,
) -> bool:
    """
    Return True if any directory in the path is ignored.
    """

    return any(
        part in IGNORED_DIRECTORIES
        for part in path.parts
    )


# ============================================================
# FILE SUPPORT CHECK
# ============================================================

def is_supported_file(
    file_path: Path,
) -> bool:
    """
    Determine whether a file should be processed by CodeLens.
    """

    # Ignore known unwanted filenames.
    if file_path.name.lower() in IGNORED_FILES:
        return False

    # Ignore files inside ignored directories.
    if is_ignored_path(file_path):
        return False

    # Only process supported text/code extensions.
    if (
        file_path.suffix.lower()
        not in SUPPORTED_EXTENSIONS
    ):
        return False

    # Ignore extremely large individual files.
    try:
        if (
            file_path.stat().st_size
            > MAX_SINGLE_FILE_SIZE
        ):
            return False
    except OSError:
        return False

    return True


# ============================================================
# GET SUPPORTED FILES
# ============================================================

def get_supported_files(
    repository_path: Path,
) -> list[Path]:
    """
    Return all supported source/documentation files.

    Files are discovered recursively without loading their
    contents into memory.
    """

    supported_files = []

    for file_path in repository_path.rglob("*"):

        if not file_path.is_file():
            continue

        if not is_supported_file(
            file_path
        ):
            continue

        supported_files.append(
            file_path
        )

    # Sort for deterministic results.
    supported_files.sort(
        key=lambda path: str(path).lower()
    )

    return supported_files


# ============================================================
# REPOSITORY INFORMATION
# ============================================================

def get_repository_info(
    repository_path: Path,
) -> dict:
    """
    Return metadata about the repository.
    """

    files = get_supported_files(
        repository_path
    )

    extensions = {}

    for file_path in files:

        extension = (
            file_path.suffix.lower()
        )

        if extension not in extensions:
            extensions[extension] = 0

        extensions[extension] += 1

    return {
        "repository_id": repository_path.name,
        "repository_path": str(
            repository_path
        ),
        "file_count": len(files),
        "file_types": extensions,
        "files": [
            str(
                file_path.relative_to(
                    repository_path
                )
            )
            for file_path in files
        ],
    }


# ============================================================
# STREAM UPLOADED ZIP TO DISK
# ============================================================

async def stream_upload_to_disk(
    upload_file,
    destination_path: Path,
) -> None:
    """
    Stream the uploaded ZIP file directly to disk.

    This avoids:

        await upload_file.read()

    which would load the entire ZIP into memory.

    The file is copied in 1 MB chunks.
    """

    with open(
        destination_path,
        "wb",
    ) as output_file:

        while True:

            chunk = await upload_file.read(
                1024 * 1024
            )

            if not chunk:
                break

            output_file.write(
                chunk
            )


# ============================================================
# SAVE UPLOADED REPOSITORY
# ============================================================

async def save_uploaded_repository(
    upload_file,
) -> dict:
    """
    Save and process an uploaded repository ZIP.

    Pipeline:

        Upload
          ↓
        Stream ZIP to disk
          ↓
        Validate ZIP
          ↓
        Safe extraction
          ↓
        File filtering
          ↓
        Repository metadata
    """

    if not upload_file.filename:
        raise ValueError(
            "Uploaded file has no filename."
        )

    filename = (
        upload_file.filename
        .lower()
    )

    if not filename.endswith(".zip"):
        raise ValueError(
            "Only ZIP files are supported."
        )

    # --------------------------------------------------------
    # CREATE REQUIRED DIRECTORIES
    # --------------------------------------------------------

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    BASE_REPOSITORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # CREATE REPOSITORY ID
    # --------------------------------------------------------

    repository_id = create_repository_id()

    zip_path = (
        UPLOAD_DIR
        / f"{repository_id}.zip"
    )

    try:

        # ----------------------------------------------------
        # STREAM UPLOAD TO DISK
        # ----------------------------------------------------

        print(
            f"Saving uploaded repository: "
            f"{upload_file.filename}"
        )

        await stream_upload_to_disk(
            upload_file=upload_file,
            destination_path=zip_path,
        )

        print(
            "Repository ZIP saved successfully."
        )

        # ----------------------------------------------------
        # VALIDATE ZIP
        # ----------------------------------------------------

        validate_zip_file(
            str(zip_path)
        )

        print(
            "ZIP validation successful."
        )

        # ----------------------------------------------------
        # EXTRACT REPOSITORY
        # ----------------------------------------------------

        repository_path = extract_repository(
            zip_path=str(zip_path),
            repository_id=repository_id,
        )

        print(
            f"Repository extracted to: "
            f"{repository_path}"
        )

        # ----------------------------------------------------
        # GET REPOSITORY INFORMATION
        # ----------------------------------------------------

        repository_info = get_repository_info(
            repository_path
        )

        print(
            "Repository information collected:"
        )

        print(
            f"Files: "
            f"{repository_info['file_count']}"
        )

        print(
            f"File types: "
            f"{repository_info['file_types']}"
        )

        return repository_info

    finally:

        # ----------------------------------------------------
        # REMOVE TEMPORARY ZIP
        # ----------------------------------------------------

        if zip_path.exists():

            try:
                zip_path.unlink()

            except OSError:
                pass