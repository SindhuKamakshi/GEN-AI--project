import hashlib
from pathlib import Path
from threading import Lock

from analysis.dependency_analyzer import DependencyAnalyzer


class DependencyService:
    """
    Provides a reusable interface over DependencyAnalyzer.

    The analyzer performs static AST analysis of a repository.
    This service caches the analysis result so that we do not
    rebuild the dependency graph for every user query.
    """

    def __init__(self):
        self._cache = {}
        self._fingerprints = {}
        self._lock = Lock()

    @staticmethod
    def _repository_fingerprint(repository_path: Path) -> str:
        digest = hashlib.sha256()

        for path in sorted(repository_path.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".py", ".ipynb"}:
                continue

            relative_parts = path.relative_to(repository_path).parts
            if any(
                part.lower() in DependencyAnalyzer.DEV_DIRS
                for part in relative_parts
            ):
                continue

            stat = path.stat()
            digest.update(path.relative_to(repository_path).as_posix().encode("utf-8"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))

        return digest.hexdigest()

    # =========================================================
    # Repository analysis
    # =========================================================

    def analyze_repository(
        self,
        repository_id: str,
        repository_path: str | Path,
    ) -> dict:
        """
        Analyze a repository and cache the result.

        Returns the complete DependencyAnalyzer result.
        """

        if not repository_id:
            raise ValueError(
                "Repository ID cannot be empty."
            )

        if not repository_path:
            raise ValueError(
                "Repository path cannot be empty."
            )

        repository_path = Path(repository_path)

        if not repository_path.exists():
            raise FileNotFoundError(
                f"Repository does not exist: {repository_path}"
            )

        fingerprint = self._repository_fingerprint(repository_path)

        with self._lock:

            if (
                repository_id in self._cache
                and self._fingerprints.get(repository_id) == fingerprint
            ):
                return self._cache[repository_id]

            analyzer = DependencyAnalyzer(
                repository_id=repository_id,
                repository_path=repository_path,
            )

            result = analyzer.analyze()

            self._cache[repository_id] = result
            self._fingerprints[repository_id] = fingerprint

            return result

    # =========================================================
    # Cache management
    # =========================================================

    def clear_repository(
        self,
        repository_id: str,
    ) -> None:
        """
        Remove one repository from the in-memory cache.
        """

        with self._lock:
            self._cache.pop(
                repository_id,
                None,
            )
            self._fingerprints.pop(repository_id, None)

    def clear_all(self) -> None:
        """
        Clear every cached dependency analysis.
        """

        with self._lock:
            self._cache.clear()
            self._fingerprints.clear()

    # =========================================================
    # Entity lookup
    # =========================================================

    def find_entities(
        self,
        repository_id: str,
        entity_name: str | None = None,
        file_path: str | None = None,
        repository_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Find entities by name and/or file path.
        """

        if not repository_id:
            raise ValueError(
                "Repository ID cannot be empty."
            )

        if repository_id not in self._cache:

            if repository_path is None:
                raise ValueError(
                    "Repository analysis is not cached. "
                    "repository_path is required."
                )

            self.analyze_repository(
                repository_id,
                repository_path,
            )

        result = self._cache[repository_id]

        matches = []

        for file_info in result.get(
            "files",
            [],
        ):

            for entity in file_info.get(
                "entities",
                [],
            ):

                if (
                    entity_name
                    and entity["name"] != entity_name
                ):
                    continue

                if (
                    file_path
                    and entity["file"] != file_path
                ):
                    continue

                matches.append(entity)

        return matches

    # =========================================================
    # Direct dependencies
    # =========================================================

    def get_entity_dependencies(
        self,
        repository_id: str,
        file_path: str,
        entity_name: str,
        repository_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Return entities directly used by the requested entity.
        """

        result = self._get_result(
            repository_id,
            repository_path,
        )

        dependencies = []

        for dependency in result.get(
            "entity_dependencies",
            [],
        ):

            if (
                dependency["from_file"] == file_path
                and dependency["from_entity"]
                == entity_name
            ):
                dependencies.append(
                    dependency
                )

        return dependencies

    # =========================================================
    # Reverse dependencies
    # =========================================================

    def get_entity_dependents(
        self,
        repository_id: str,
        file_path: str,
        entity_name: str,
        repository_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Return entities that depend on the requested entity.
        """

        result = self._get_result(
            repository_id,
            repository_path,
        )

        dependents = []

        for dependency in result.get(
            "entity_dependencies",
            [],
        ):

            if (
                dependency["to_file"] == file_path
                and dependency["to_entity"]
                == entity_name
            ):
                dependents.append(
                    dependency
                )

        return dependents

    # =========================================================
    # Related entities
    # =========================================================

    def expand_entity(
        self,
        repository_id: str,
        file_path: str,
        entity_name: str,
        repository_path: str | Path | None = None,
    ) -> dict:
        """
        Return both directions of dependency relationships
        for one entity.
        """

        dependencies = self.get_entity_dependencies(
            repository_id=repository_id,
            file_path=file_path,
            entity_name=entity_name,
            repository_path=repository_path,
        )

        dependents = self.get_entity_dependents(
            repository_id=repository_id,
            file_path=file_path,
            entity_name=entity_name,
            repository_path=repository_path,
        )

        return {
            "entity": {
                "file": file_path,
                "name": entity_name,
            },
            "dependencies": dependencies,
            "dependents": dependents,
        }

    # =========================================================
    # Retrieval expansion
    # =========================================================

    def expand_search_results(
        self,
        repository_id: str,
        search_results: list[dict],
        repository_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Expand retrieved code chunks with directly related
        dependency information.

        Existing search results are preserved.

        Dependency information is attached under:
            result["dependency_context"]
        """

        if not search_results:
            return []

        expanded_results = []

        seen_relationships = set()

        for result in search_results:

            enriched = dict(result)

            metadata = result.get(
                "metadata",
                {},
            )

            file_path = (
                metadata.get("file_path")
                or metadata.get("file")
                or metadata.get("path")
            )

            entity_name = (
                metadata.get("name")
                or metadata.get("entity_name")
            )

            dependency_context = {
                "dependencies": [],
                "dependents": [],
            }

            if file_path and entity_name:

                relationships = self.expand_entity(
                    repository_id=repository_id,
                    file_path=file_path,
                    entity_name=entity_name,
                    repository_path=repository_path,
                )

                for dependency in relationships[
                    "dependencies"
                ]:

                    relationship_key = (
                        dependency["from_file"],
                        dependency["from_entity"],
                        dependency["to_file"],
                        dependency["to_entity"],
                        dependency["relationship"],
                    )

                    if (
                        relationship_key
                        not in seen_relationships
                    ):
                        dependency_context[
                            "dependencies"
                        ].append(dependency)

                        seen_relationships.add(
                            relationship_key
                        )

                for dependent in relationships[
                    "dependents"
                ]:

                    relationship_key = (
                        dependent["from_file"],
                        dependent["from_entity"],
                        dependent["to_file"],
                        dependent["to_entity"],
                        dependent["relationship"],
                    )

                    if (
                        relationship_key
                        not in seen_relationships
                    ):
                        dependency_context[
                            "dependents"
                        ].append(dependent)

                        seen_relationships.add(
                            relationship_key
                        )

            enriched[
                "dependency_context"
            ] = dependency_context

            expanded_results.append(
                enriched
            )

        return expanded_results

    # =========================================================
    # Internal helper
    # =========================================================

    def _get_result(
        self,
        repository_id: str,
        repository_path: str | Path | None = None,
    ) -> dict:
        """
        Return cached analysis, analyzing the repository only
        when necessary.
        """

        if not repository_id:
            raise ValueError(
                "Repository ID cannot be empty."
            )

        if repository_id not in self._cache:

            if repository_path is None:
                raise ValueError(
                    "Repository analysis is not cached. "
                    "repository_path is required."
                )

            self.analyze_repository(
                repository_id,
                repository_path,
            )

        return self._cache[repository_id]