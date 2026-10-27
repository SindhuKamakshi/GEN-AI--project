from datetime import datetime
from threading import RLock
from typing import Dict, Any


class ProgressService:
    """
    In-memory repository indexing progress tracker.

    Tracks the current processing state of each repository.
    """

    def __init__(self):
        self._progress: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def begin(self, repository_id: str) -> bool:
        """Atomically start indexing unless it is already in progress."""
        with self._lock:
            current = self._progress.get(repository_id)
            if current and current["status"] == "indexing":
                return False

            self._progress[repository_id] = {
                "repository_id": repository_id,
                "status": "indexing",
                "stage": "initialized",
                "progress": 0,
                "processed_files": 0,
                "total_files": 0,
                "processed_chunks": 0,
                "total_chunks": 0,
                "message": "Repository processing initialized.",
                "updated_at": datetime.utcnow().isoformat(),
            }
            return True

    def initialize(self, repository_id: str) -> None:
        """
        Initialize progress tracking for a repository.
        """

        with self._lock:
            self._progress[repository_id] = {
                "repository_id": repository_id,
                "status": "initialized",
                "stage": "initialized",
                "progress": 0,
                "processed_files": 0,
                "total_files": 0,
                "processed_chunks": 0,
                "total_chunks": 0,
                "message": "Repository processing initialized.",
                "updated_at": datetime.utcnow().isoformat(),
            }

    def update(
        self,
        repository_id: str,
        *,
        status: str | None = None,
        stage: str | None = None,
        progress: int | None = None,
        processed_files: int | None = None,
        total_files: int | None = None,
        processed_chunks: int | None = None,
        total_chunks: int | None = None,
        message: str | None = None,
    ) -> None:
        """
        Update one or more progress fields.
        """

        with self._lock:

            if repository_id not in self._progress:
                self.initialize(repository_id)

            current = self._progress[repository_id]

            if status is not None:
                current["status"] = status

            if stage is not None:
                current["stage"] = stage

            if progress is not None:
                current["progress"] = max(0, min(100, progress))

            if processed_files is not None:
                current["processed_files"] = processed_files

            if total_files is not None:
                current["total_files"] = total_files

            if processed_chunks is not None:
                current["processed_chunks"] = processed_chunks

            if total_chunks is not None:
                current["total_chunks"] = total_chunks

            if message is not None:
                current["message"] = message

            current["updated_at"] = datetime.utcnow().isoformat()

    def get(self, repository_id: str) -> Dict[str, Any] | None:
        """
        Get the current progress for a repository.
        """

        with self._lock:
            progress = self._progress.get(repository_id)

            if progress is None:
                return None

            return progress.copy()

    def complete(
        self,
        repository_id: str,
        *,
        processed_files: int | None = None,
        total_files: int | None = None,
        processed_chunks: int | None = None,
        total_chunks: int | None = None,
    ) -> None:
        """
        Mark repository processing as completed.
        """

        self.update(
            repository_id,
            status="completed",
            stage="completed",
            progress=100,
            processed_files=processed_files,
            total_files=total_files,
            processed_chunks=processed_chunks,
            total_chunks=total_chunks,
            message="Repository indexed successfully.",
        )

    def fail(
        self,
        repository_id: str,
        message: str,
    ) -> None:
        """
        Mark repository processing as failed.
        """

        self.update(
            repository_id,
            status="failed",
            stage="failed",
            message=message,
        )

    def remove(self, repository_id: str) -> None:
        """
        Remove progress information for a repository.
        """

        with self._lock:
            self._progress.pop(repository_id, None)


progress_service = ProgressService()