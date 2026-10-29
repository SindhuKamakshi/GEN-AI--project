import ast
import hashlib
import json
import logging
from pathlib import Path

from repository.notebook_utils import NotebookParseError, read_notebook_cells


logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

TEXT_CHUNK_SIZE = 80
TEXT_CHUNK_OVERLAP = 15

MAX_AST_FILE_SIZE_MB = 5
MAX_AST_FILE_SIZE_BYTES = MAX_AST_FILE_SIZE_MB * 1024 * 1024


# ============================================================
# LANGUAGE DETECTION
# ============================================================

LANGUAGE_MAP = {
    ".py": "Python",
    ".pyw": "Python",
    ".ipynb": "Jupyter Notebook",
    ".js": "JavaScript",
    ".jsx": "JavaScript React",
    ".ts": "TypeScript",
    ".tsx": "TypeScript React",
    ".java": "Java",
    ".c": "C",
    ".cpp": "C++",
    ".h": "C/C++ Header",
    ".hpp": "C++ Header",
    ".cs": "C#",
    ".go": "Go",
    ".rs": "Rust",
    ".php": "PHP",
    ".html": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".ini": "INI",
    ".md": "Markdown",
    ".rst": "reStructuredText",
    ".txt": "Text",
    ".sql": "SQL",
    ".xml": "XML",
    ".sh": "Shell",
    ".bat": "Batch",
    ".ps1": "PowerShell",
}


def get_language(file_path: Path) -> str:
    return LANGUAGE_MAP.get(file_path.suffix.lower(), "Unknown")


# ============================================================
# CHUNK ID
# ============================================================

def create_chunk_id(
    repository_id: str,
    file_path: str,
    start_line: int,
    end_line: int,
    name: str,
) -> str:
    raw_value = (
        f"{repository_id}|{file_path}|"
        f"{start_line}|{end_line}|{name}"
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"{repository_id}_chunk_{digest}"


# ============================================================
# SAFE FILE READING
# ============================================================

def read_file(file_path: Path) -> str:
    try:
        return file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except (OSError, UnicodeError):
        return ""


def get_file_size(file_path: Path) -> int:
    try:
        return file_path.stat().st_size
    except OSError:
        return 0


# ============================================================
# TEXT CHUNKING
# ============================================================

def create_text_chunks(
    content: str,
    file_path: str,
    repository_id: str,
    language: str,
) -> list[dict]:

    lines = content.splitlines()

    if not lines:
        return []

    chunks = []
    start = 0
    chunk_number = 1

    while start < len(lines):
        end = min(
            start + TEXT_CHUNK_SIZE,
            len(lines),
        )

        code = "\n".join(lines[start:end])

        if code.strip():
            name = f"chunk_{chunk_number}"

            chunk_id = create_chunk_id(
                repository_id=repository_id,
                file_path=file_path,
                start_line=start + 1,
                end_line=end,
                name=name,
            )

            chunks.append({
                "chunk_id": chunk_id,
                "content": code,
                "metadata": {
                    "repository_id": repository_id,
                    "file_path": file_path,
                    "language": language,
                    "chunk_type": "text_block",
                    "name": name,
                    "start_line": start + 1,
                    "end_line": end,
                },
            })

        if end >= len(lines):
            break

        start = end - TEXT_CHUNK_OVERLAP
        chunk_number += 1

    return chunks


# ============================================================
# PYTHON AST CHUNKING
# ============================================================

def create_python_chunks(
    content: str,
    file_path: str,
    repository_id: str,
) -> list[dict]:

    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, TypeError, MemoryError):
        return create_text_chunks(
            content,
            file_path,
            repository_id,
            "Python",
        )

    lines = content.splitlines()
    chunks = []

    for node in tree.body:
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        start_line = node.lineno

        decorators = getattr(node, "decorator_list", [])

        if decorators:
            start_line = min(
                start_line,
                *(decorator.lineno for decorator in decorators),
            )

        end_line = getattr(node, "end_lineno", start_line)

        code = "\n".join(
            lines[start_line - 1:end_line]
        )

        if not code.strip():
            continue

        name = node.name

        chunk_type = (
            "class"
            if isinstance(node, ast.ClassDef)
            else "function"
        )

        chunk_id = create_chunk_id(
            repository_id,
            file_path,
            start_line,
            end_line,
            name,
        )

        chunks.append({
            "chunk_id": chunk_id,
            "content": code,
            "metadata": {
                "repository_id": repository_id,
                "file_path": file_path,
                "language": "Python",
                "chunk_type": chunk_type,
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
            },
        })

    if not chunks and content.strip():
        return create_text_chunks(
            content,
            file_path,
            repository_id,
            "Python",
        )

    return chunks


# ============================================================
# JUPYTER NOTEBOOK CHUNKING
# ============================================================

def create_notebook_chunks(
    file_path: Path,
    relative_path: str,
    repository_id: str,
) -> list[dict]:
    """
    Extract Markdown, code, and raw cells from a notebook.
    Notebook code is treated as text and is never executed.
    """

    try:
        cells = read_notebook_cells(file_path)
    except NotebookParseError as exc:
        logger.warning(
            "Skipping malformed notebook %s: %s",
            relative_path,
            exc,
        )
        return []

    chunks = []

    for cell in cells:
        cell_number = cell["cell_number"]
        cell_type = cell["cell_type"]
        source = cell["source"]

        source = source.strip()

        if not source:
            continue

        if cell_type == "code":
            cell_language = "Python"
        elif cell_type == "markdown":
            cell_language = "Markdown"
        else:
            cell_language = "Text"

        cell_chunks = create_text_chunks(
            content=source,
            file_path=relative_path,
            repository_id=repository_id,
            language=cell_language,
        )

        for part_number, chunk in enumerate(
            cell_chunks,
            start=1,
        ):
            chunk["chunk_id"] = create_chunk_id(
                repository_id=repository_id,
                file_path=relative_path,
                start_line=cell_number,
                end_line=cell_number,
                name=(
                    f"cell_{cell_number}_"
                    f"part_{part_number}_{cell_type}"
                ),
            )

            chunk["content"] = (
                f"Notebook: {relative_path}\n"
                f"Cell: {cell_number}\n"
                f"Cell type: {cell_type}\n"
                f"Language: {cell_language}\n\n"
                f"{chunk['content']}"
            )

            chunk["metadata"].update({
                "language": "Jupyter Notebook",
                "chunk_type": "notebook_cell",
                "name": (
                    f"cell_{cell_number}_"
                    f"part_{part_number}_{cell_type}"
                ),
                "notebook_cell_type": cell_type,
                "cell_number": cell_number,
                "cell_part": part_number,
                "cell_language": cell_language,
                "start_line": cell_number,
                "end_line": cell_number,
                "cell_metadata_json": json.dumps(
                    cell["metadata"],
                    ensure_ascii=False,
                ),
            })

        chunks.extend(cell_chunks)

    return chunks


# ============================================================
# LARGE FILE CHUNKING
# ============================================================

def create_large_file_chunks(
    file_path: Path,
    relative_path: str,
    repository_id: str,
    language: str,
) -> list[dict]:

    chunks = []

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            buffer = []
            start_line = 1
            current_line = 0
            chunk_number = 1

            for line in file:
                current_line += 1
                buffer.append(line.rstrip("\r\n"))

                if len(buffer) >= TEXT_CHUNK_SIZE:
                    end_line = current_line
                    code = "\n".join(buffer)

                    if code.strip():
                        name = f"chunk_{chunk_number}"

                        chunks.append({
                            "chunk_id": create_chunk_id(
                                repository_id,
                                relative_path,
                                start_line,
                                end_line,
                                name,
                            ),
                            "content": code,
                            "metadata": {
                                "repository_id": repository_id,
                                "file_path": relative_path,
                                "language": language,
                                "chunk_type": "large_file_block",
                                "name": name,
                                "start_line": start_line,
                                "end_line": end_line,
                            },
                        })

                    overlap = buffer[-TEXT_CHUNK_OVERLAP:]

                    start_line = (
                        current_line - len(overlap) + 1
                    )

                    buffer = overlap
                    chunk_number += 1

            if buffer:
                end_line = current_line
                code = "\n".join(buffer)

                if code.strip():
                    name = f"chunk_{chunk_number}"

                    chunks.append({
                        "chunk_id": create_chunk_id(
                            repository_id,
                            relative_path,
                            start_line,
                            end_line,
                            name,
                        ),
                        "content": code,
                        "metadata": {
                            "repository_id": repository_id,
                            "file_path": relative_path,
                            "language": language,
                            "chunk_type": "large_file_block",
                            "name": name,
                            "start_line": start_line,
                            "end_line": end_line,
                        },
                    })

    except (OSError, UnicodeError):
        return []

    return chunks


# ============================================================
# CHUNK ONE FILE
# ============================================================

def chunk_file(
    file_path: Path,
    repository_root: Path,
    repository_id: str,
) -> list[dict]:

    try:
        relative_path = str(
            file_path.relative_to(repository_root)
        )
    except ValueError:
        return []

    suffix = file_path.suffix.lower()
    language = get_language(file_path)
    file_size = get_file_size(file_path)

    if file_size == 0:
        return []

    # Process notebooks before generic large-file handling.
    if suffix == ".ipynb":
        if file_size > MAX_AST_FILE_SIZE_BYTES:
            return []

        return create_notebook_chunks(
            file_path=file_path,
            relative_path=relative_path,
            repository_id=repository_id,
        )

    if file_size > MAX_AST_FILE_SIZE_BYTES:
        return create_large_file_chunks(
            file_path=file_path,
            relative_path=relative_path,
            repository_id=repository_id,
            language=language,
        )

    content = read_file(file_path)

    if not content.strip():
        return []

    if suffix in {".py", ".pyw"}:
        return create_python_chunks(
            content=content,
            file_path=relative_path,
            repository_id=repository_id,
        )

    return create_text_chunks(
        content=content,
        file_path=relative_path,
        repository_id=repository_id,
        language=language,
    )


# ============================================================
# CHUNK REPOSITORY
# ============================================================

def chunk_repository(
    repository_path: Path,
    repository_id: str,
    files: list[Path],
) -> list[dict]:

    all_chunks = []

    for file_path in files:
        file_chunks = chunk_file(
            file_path=file_path,
            repository_root=repository_path,
            repository_id=repository_id,
        )

        all_chunks.extend(file_chunks)

    return all_chunks