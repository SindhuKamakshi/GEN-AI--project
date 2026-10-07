from pathlib import Path
import json
import sqlite3

from config import DATA_DIR


KEYWORD_DB_PATH = DATA_DIR / "keyword_index.db"


class KeywordStore:
    """
    Persistent SQLite FTS5 keyword search store for CodeLens AI.

    This store complements ChromaDB:
    - ChromaDB -> semantic search
    - SQLite FTS5 -> keyword/BM25 search

    Each repository is isolated using repository_id.
    """

    def __init__(self, db_path: Path = KEYWORD_DB_PATH):
        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize_database()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path))

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize_database(self):
        """
        Create the FTS5 table if it does not already exist.
        """

        with self._connect() as connection:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS keyword_chunks
                USING fts5(
                    repository_id UNINDEXED,
                    chunk_id UNINDEXED,
                    file_path,
                    file_name,
                    content,
                    metadata_json UNINDEXED
                )
                """
            )

            connection.commit()

    def delete_repository(self, repository_id: str) -> None:
        """
        Delete all keyword-search data belonging to a repository.
        """

        if not repository_id:
            raise ValueError("Repository ID cannot be empty.")

        with self._connect() as connection:
            connection.execute(
                """
                DELETE FROM keyword_chunks
                WHERE repository_id = ?
                """,
                (repository_id,)
            )

            connection.commit()

    def add_chunks(
        self,
        repository_id: str,
        chunks: list[dict]
    ) -> int:
        """
        Add repository chunks to the FTS5 keyword index.
        """

        if not repository_id:
            raise ValueError("Repository ID cannot be empty.")

        if not chunks:
            return 0

        rows = []

        for index, chunk in enumerate(chunks):
            content = chunk.get("content", "")

            if not content:
                continue

            metadata = chunk.get("metadata", {})

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                chunk_id = f"{repository_id}_chunk_{index}"

            file_path = str(
                metadata.get("file_path", "")
            )

            file_name = str(
                metadata.get("file_name", "")
                or Path(file_path).name
            )

            rows.append(
                (
                    repository_id,
                    chunk_id,
                    file_path,
                    file_name,
                    content,
                    json.dumps(
                        metadata,
                        ensure_ascii=False
                    )
                )
            )

        if not rows:
            return 0

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO keyword_chunks (
                    repository_id,
                    chunk_id,
                    file_path,
                    file_name,
                    content,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows
            )

            connection.commit()

        return len(rows)

    def search(
        self,
        repository_id: str,
        query: str,
        top_k: int = 10
    ) -> list[dict]:
        """
        Perform keyword search using SQLite FTS5 BM25 ranking.

        SQLite FTS5's BM25 score is lower-is-better,
        so we preserve the raw BM25 score and also
        expose a normalized relevance score.
        """

        if not repository_id:
            raise ValueError("Repository ID cannot be empty.")

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        # Extract useful words from the user's query.
        terms = [
            term.strip()
            for term in query.split()
            if term.strip()
        ]

        if not terms:
            return []

        # FTS5 MATCH syntax needs protection from special
        # operators such as AND, OR, NOT, *, etc.
        safe_terms = []

        for term in terms:
            cleaned = "".join(
                character
                for character in term
                if character.isalnum()
                or character in "_-."
            )

            if cleaned:
                safe_terms.append(cleaned)

        if not safe_terms:
            return []

        # OR makes keyword retrieval more tolerant.
        match_query = " OR ".join(
            f'"{term}"'
            for term in safe_terms
        )

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    repository_id,
                    chunk_id,
                    file_path,
                    file_name,
                    content,
                    metadata_json,
                    bm25(keyword_chunks) AS bm25_score
                FROM keyword_chunks
                WHERE repository_id = ?
                  AND keyword_chunks MATCH ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (
                    repository_id,
                    match_query,
                    top_k
                )
            ).fetchall()

        results = []

        for rank, row in enumerate(rows, start=1):
            metadata = {}

            try:
                metadata = json.loads(
                    row["metadata_json"]
                )
            except (TypeError, json.JSONDecodeError):
                metadata = {}

            bm25_score = float(
                row["bm25_score"]
            )

            # BM25 from SQLite is normally negative.
            # Convert it into a positive relevance value
            # for easier downstream handling.
            relevance_score = 1.0 / (
                1.0 + abs(bm25_score)
            )

            results.append(
                {
                    "rank": rank,
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "metadata": metadata,
                    "file_path": row["file_path"],
                    "file_name": row["file_name"],
                    "bm25_score": bm25_score,
                    "keyword_score": relevance_score
                }
            )

        return results

    def get_repository_count(
        self,
        repository_id: str
    ) -> int:
        """
        Return the number of chunks indexed for a repository.
        """

        if not repository_id:
            raise ValueError("Repository ID cannot be empty.")

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM keyword_chunks
                WHERE repository_id = ?
                """,
                (repository_id,)
            ).fetchone()

        return int(row["count"])

    def repository_exists(
        self,
        repository_id: str
    ) -> bool:
        """
        Check whether keyword data exists for a repository.
        """

        return self.get_repository_count(repository_id) > 0