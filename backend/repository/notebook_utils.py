import json
from pathlib import Path


class NotebookParseError(ValueError):
    """Raised when a notebook does not have a readable cells structure."""


def read_notebook_cells(file_path: Path) -> list[dict]:
    """Read notebook source cells without loading or executing outputs."""
    try:
        with Path(file_path).open("r", encoding="utf-8") as notebook_file:
            notebook = json.load(notebook_file)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NotebookParseError("Notebook is not valid readable JSON.") from exc

    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise NotebookParseError("Notebook does not contain a valid cells list.")

    cells = []
    for cell_number, cell in enumerate(notebook["cells"], start=1):
        if not isinstance(cell, dict):
            continue

        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            continue

        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(part for part in source if isinstance(part, str))
        elif not isinstance(source, str):
            continue

        metadata = cell.get("metadata", {})
        cells.append({
            "cell_number": cell_number,
            "cell_type": cell_type,
            "source": source,
            "metadata": metadata if isinstance(metadata, dict) else {},
        })

    return cells
