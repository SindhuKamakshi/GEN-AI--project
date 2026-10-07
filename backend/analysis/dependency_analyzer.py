import ast
from pathlib import Path
from collections import Counter, defaultdict, deque

from config import REPOSITORIES_DIR


class DependencyAnalyzer:
    """
    Static dependency analyzer for Python repositories.

    Detects:

    - Internal file dependencies
    - Reverse dependencies
    - External library dependencies
    - Dependency counts
    - Most depended-on files
    - Dependency graph
    - Direct dependency relationships
    - Basic transitive dependency information

    Repository code is never executed.
    """

    DEV_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "node_modules",
        "dist",
        "build",
        "coverage",
        ".idea",
        ".vscode",
    }

    STANDARD_LIBRARY = {
        "os",
        "sys",
        "json",
        "re",
        "math",
        "time",
        "datetime",
        "pathlib",
        "typing",
        "collections",
        "itertools",
        "functools",
        "random",
        "string",
        "logging",
        "warnings",
        "subprocess",
        "shutil",
        "uuid",
        "hashlib",
        "base64",
        "io",
        "ast",
        "csv",
        "zipfile",
        "tempfile",
        "statistics",
        "asyncio",
        "threading",
        "multiprocessing",
        "contextlib",
        "dataclasses",
        "enum",
        "inspect",
        "traceback",
        "unittest",
        "argparse",
        "glob",
        "pickle",
        "sqlite3",
    }

    def __init__(self, repository_id: str):
        if not repository_id:
            raise ValueError("repository_id is required")

        self.repository_id = repository_id
        self.repository_path = (
            REPOSITORIES_DIR / repository_id
        )

    # ---------------------------------------------------------
    # Repository validation
    # ---------------------------------------------------------

    def _validate_repository(self):
        if not self.repository_path.exists():
            raise FileNotFoundError(
                f"Repository '{self.repository_id}' was not found"
            )

        if not self.repository_path.is_dir():
            raise ValueError(
                f"Repository path is not a directory: "
                f"{self.repository_path}"
            )

    # ---------------------------------------------------------
    # File discovery
    # ---------------------------------------------------------

    def _iter_python_files(self):
        for path in self.repository_path.rglob("*.py"):

            relative_parts = path.relative_to(
                self.repository_path
            ).parts

            if any(
                part in self.DEV_DIRS
                for part in relative_parts
            ):
                continue

            yield path

    # ---------------------------------------------------------
    # Path / module helpers
    # ---------------------------------------------------------

    def _relative_path(self, path):
        return str(
            path.relative_to(
                self.repository_path
            )
        ).replace("\\", "/")

    @staticmethod
    def _module_name(relative_path):
        """
        Convert:

            rag.py
            utils/helper.py
            utils/__init__.py

        into:

            rag
            utils.helper
            utils
        """

        path = Path(relative_path)

        parts = list(path.parts)

        if not parts:
            return ""

        if parts[-1] == "__init__.py":
            parts = parts[:-1]

        else:
            parts[-1] = Path(
                parts[-1]
            ).stem

        return ".".join(parts)

    @staticmethod
    def _top_level_module(import_name):
        return import_name.split(".")[0]

    # ---------------------------------------------------------
    # Import extraction
    # ---------------------------------------------------------

    @staticmethod
    def _extract_imports(tree):
        imports = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):

                if node.module:

                    imports.append(
                        node.module
                    )

        return sorted(
            set(imports)
        )

    # ---------------------------------------------------------
    # Python file parsing
    # ---------------------------------------------------------

    def _parse_files(self):

        files = []
        failed_files = []

        for path in self._iter_python_files():

            relative_path = self._relative_path(
                path
            )

            module = self._module_name(
                relative_path
            )

            try:

                source = path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )

                tree = ast.parse(source)

                imports = self._extract_imports(
                    tree
                )

                files.append(
                    {
                        "path": relative_path,
                        "module": module,
                        "imports": imports,
                    }
                )

            except (
                SyntaxError,
                OSError,
                UnicodeDecodeError,
            ) as exc:

                failed_files.append(
                    {
                        "path": relative_path,
                        "error": str(exc),
                    }
                )

                files.append(
                    {
                        "path": relative_path,
                        "module": module,
                        "imports": [],
                    }
                )

        return files, failed_files

    # ---------------------------------------------------------
    # Internal module mapping
    # ---------------------------------------------------------

    def _build_module_map(self, files):

        module_map = {}

        for file_info in files:

            module = file_info["module"]

            if module:
                module_map[
                    module
                ] = file_info["path"]

        return module_map

    # ---------------------------------------------------------
    # Internal dependency detection
    # ---------------------------------------------------------

    def _find_internal_target(
        self,
        imported_module,
        module_map,
    ):
        """
        Resolve an import to a repository file.

        Example:

            import rag
            from rag import answer_question

        both resolve to:

            rag.py
        """

        # Exact match.
        if imported_module in module_map:

            return module_map[
                imported_module
            ]

        # Match parent module.
        for module_name, file_path in module_map.items():

            if (
                imported_module.startswith(
                    module_name + "."
                )
            ):
                return file_path

        # Match package import.
        for module_name, file_path in module_map.items():

            if module_name.startswith(
                imported_module + "."
            ):
                return file_path

        return None

    def _build_internal_dependencies(
        self,
        files,
        module_map,
    ):

        dependencies = []

        for file_info in files:

            source_file = file_info["path"]

            for imported_module in file_info[
                "imports"
            ]:

                target_file = (
                    self._find_internal_target(
                        imported_module,
                        module_map,
                    )
                )

                if target_file is None:
                    continue

                if target_file == source_file:
                    continue

                dependencies.append(
                    {
                        "from": source_file,
                        "to": target_file,
                        "import": imported_module,
                        "relationship": "internal_import",
                    }
                )

        # Remove duplicates.
        unique = {}

        for dependency in dependencies:

            key = (
                dependency["from"],
                dependency["to"],
                dependency["import"],
            )

            unique[key] = dependency

        return list(
            unique.values()
        )

    # ---------------------------------------------------------
    # External dependency detection
    # ---------------------------------------------------------

    def _build_external_dependencies(
        self,
        files,
        module_map,
    ):

        internal_top_levels = {
            module.split(".")[0]
            for module in module_map
        }

        external_by_file = defaultdict(set)

        for file_info in files:

            source_file = file_info["path"]

            for imported_module in file_info[
                "imports"
            ]:

                top_level = (
                    self._top_level_module(
                        imported_module
                    )
                )

                # Python standard library.
                if top_level in self.STANDARD_LIBRARY:
                    continue

                # Internal repository module.
                if top_level in internal_top_levels:
                    continue

                external_by_file[
                    source_file
                ].add(top_level)

        return {
            path: sorted(dependencies)
            for path, dependencies
            in external_by_file.items()
        }

    # ---------------------------------------------------------
    # Reverse dependency graph
    # ---------------------------------------------------------

    def _build_reverse_dependencies(
        self,
        internal_dependencies,
    ):

        reverse = defaultdict(list)

        for dependency in internal_dependencies:

            source = dependency["from"]
            target = dependency["to"]

            reverse[target].append(
                {
                    "file": source,
                    "import": dependency[
                        "import"
                    ],
                    "relationship": "depended_on_by",
                }
            )

        return {
            file_path: sorted(
                dependencies,
                key=lambda item: item["file"],
            )
            for file_path, dependencies
            in reverse.items()
        }

    # ---------------------------------------------------------
    # Direct dependency information
    # ---------------------------------------------------------

    def _build_file_dependency_summary(
        self,
        files,
        internal_dependencies,
        external_dependencies,
    ):

        depends_on = defaultdict(list)
        depended_on_by = defaultdict(list)

        for dependency in internal_dependencies:

            source = dependency["from"]
            target = dependency["to"]

            depends_on[source].append(
                target
            )

            depended_on_by[target].append(
                source
            )

        summaries = []

        for file_info in files:

            path = file_info["path"]

            summaries.append(
                {
                    "file": path,
                    "module": file_info[
                        "module"
                    ],
                    "internal_dependencies": sorted(
                        set(
                            depends_on.get(
                                path,
                                [],
                            )
                        )
                    ),
                    "depended_on_by": sorted(
                        set(
                            depended_on_by.get(
                                path,
                                [],
                            )
                        )
                    ),
                    "external_dependencies": (
                        external_dependencies.get(
                            path,
                            [],
                        )
                    ),
                    "dependency_count": len(
                        set(
                            depends_on.get(
                                path,
                                [],
                            )
                        )
                    ),
                    "dependent_count": len(
                        set(
                            depended_on_by.get(
                                path,
                                [],
                            )
                        )
                    ),
                }
            )

        return summaries

    # ---------------------------------------------------------
    # Most depended-on files
    # ---------------------------------------------------------

    def _build_dependency_hotspots(
        self,
        files,
        internal_dependencies,
    ):

        dependent_counts = Counter()

        for dependency in internal_dependencies:

            dependent_counts[
                dependency["to"]
            ] += 1

        hotspots = []

        file_paths = {
            file_info["path"]
            for file_info in files
        }

        for file_path in file_paths:

            count = dependent_counts.get(
                file_path,
                0,
            )

            hotspots.append(
                {
                    "file": file_path,
                    "dependent_count": count,
                }
            )

        hotspots.sort(
            key=lambda item: (
                -item["dependent_count"],
                item["file"],
            )
        )

        return hotspots

    # ---------------------------------------------------------
    # Dependency graph
    # ---------------------------------------------------------

    def _build_dependency_graph(
        self,
        files,
        internal_dependencies,
    ):

        nodes = []

        for file_info in files:

            nodes.append(
                {
                    "id": file_info["path"],
                    "label": Path(
                        file_info["path"]
                    ).name,
                    "module": file_info[
                        "module"
                    ],
                }
            )

        edges = []

        for dependency in internal_dependencies:

            edges.append(
                {
                    "source": dependency[
                        "from"
                    ],
                    "target": dependency[
                        "to"
                    ],
                    "relationship": dependency[
                        "relationship"
                    ],
                    "import": dependency[
                        "import"
                    ],
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    # ---------------------------------------------------------
    # Transitive dependency analysis
    # ---------------------------------------------------------

    def _build_transitive_dependencies(
        self,
        files,
        internal_dependencies,
    ):
        """
        Find indirect dependencies.

        Example:

            app.py → service.py
            service.py → database.py

        Then:

            app.py indirectly depends on database.py
        """

        graph = defaultdict(set)

        for dependency in internal_dependencies:

            graph[
                dependency["from"]
            ].add(
                dependency["to"]
            )

        file_paths = {
            file_info["path"]
            for file_info in files
        }

        result = {}

        for start_file in file_paths:

            visited = set()

            queue = deque(
                graph.get(
                    start_file,
                    set(),
                )
            )

            while queue:

                current = queue.popleft()

                if current == start_file:
                    continue

                if current in visited:
                    continue

                visited.add(current)

                for next_file in graph.get(
                    current,
                    set(),
                ):

                    if next_file not in visited:
                        queue.append(
                            next_file
                        )

            direct = graph.get(
                start_file,
                set(),
            )

            indirect = (
                visited - direct
            )

            result[start_file] = {
                "direct_dependencies": sorted(
                    direct
                ),
                "indirect_dependencies": sorted(
                    indirect
                ),
                "total_dependencies": len(
                    visited
                ),
            }

        return result

    # ---------------------------------------------------------
    # Main analysis
    # ---------------------------------------------------------

    def analyze(self):

        self._validate_repository()

        files, failed_files = (
            self._parse_files()
        )

        module_map = (
            self._build_module_map(
                files
            )
        )

        internal_dependencies = (
            self._build_internal_dependencies(
                files,
                module_map,
            )
        )

        external_dependencies = (
            self._build_external_dependencies(
                files,
                module_map,
            )
        )

        reverse_dependencies = (
            self._build_reverse_dependencies(
                internal_dependencies
            )
        )

        file_summaries = (
            self._build_file_dependency_summary(
                files,
                internal_dependencies,
                external_dependencies,
            )
        )

        hotspots = (
            self._build_dependency_hotspots(
                files,
                internal_dependencies,
            )
        )

        dependency_graph = (
            self._build_dependency_graph(
                files,
                internal_dependencies,
            )
        )

        transitive_dependencies = (
            self._build_transitive_dependencies(
                files,
                internal_dependencies,
            )
        )

        total_internal = len(
            internal_dependencies
        )

        total_external = sum(
            len(dependencies)
            for dependencies
            in external_dependencies.values()
        )

        return {
            "repository_id": self.repository_id,

            "total_python_files": len(files),

            "analyzed_files": len(files)
            - len(failed_files),

            "failed_files": failed_files,

            "total_internal_dependencies": (
                total_internal
            ),

            "total_external_dependencies": (
                total_external
            ),

            "internal_dependencies": (
                internal_dependencies
            ),

            "external_dependencies_by_file": (
                external_dependencies
            ),

            "reverse_dependencies": (
                reverse_dependencies
            ),

            "file_dependency_summary": (
                file_summaries
            ),

            "dependency_hotspots": hotspots,

            "transitive_dependencies": (
                transitive_dependencies
            ),

            "dependency_graph": (
                dependency_graph
            ),
        }