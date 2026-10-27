import ast
from pathlib import Path
from collections import Counter, defaultdict, deque

from config import REPOSITORIES_DIR
from repository.notebook_utils import NotebookParseError, read_notebook_cells


class DependencyAnalyzer:
    """
    Static dependency analyzer for Python repositories.

    Detects:

    File-level:
    - Internal file dependencies
    - Reverse dependencies
    - External library dependencies
    - Dependency counts
    - Most depended-on files
    - Dependency graph
    - Direct dependency relationships
    - Basic transitive dependency information

    Entity-level:
    - Function definitions
    - Class definitions
    - Function calls
    - Class references
    - Entity dependencies
    - Reverse entity dependencies
    - Entity dependency graph

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

    def __init__(
        self,
        repository_id: str,
        repository_path: str | Path | None = None,
    ):
        if not repository_id:
            raise ValueError("repository_id is required")

        self.repository_id = repository_id
        self.repository_path = Path(
            repository_path
            if repository_path is not None
            else REPOSITORIES_DIR / repository_id
        )

    # =========================================================
    # Repository validation
    # =========================================================

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

    # =========================================================
    # File discovery
    # =========================================================

    def _iter_python_files(self):
        for path in self.repository_path.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".ipynb"}:
                continue

            relative_parts = path.relative_to(
                self.repository_path
            ).parts

            if any(
                part.lower() in self.DEV_DIRS
                for part in relative_parts
            ):
                continue

            yield path

    # =========================================================
    # Path / module helpers
    # =========================================================

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
            parts[-1] = Path(parts[-1]).stem

        return ".".join(parts)

    @staticmethod
    def _top_level_module(import_name):
        return import_name.split(".")[0]

    # =========================================================
    # Import extraction
    # =========================================================

    @staticmethod
    def _extract_imports(tree):
        imports = []

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):

                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):

                if node.module:
                    imports.append(node.module)

        return sorted(set(imports))

    # =========================================================
    # Entity helpers
    # =========================================================

    @staticmethod
    def _get_entity_name(node):
        """
        Return the name of a function or class.
        """

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name

        if isinstance(node, ast.ClassDef):
            return node.name

        return None

    @staticmethod
    def _get_node_type(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "function"

        if isinstance(node, ast.ClassDef):
            return "class"

        return "unknown"

    @staticmethod
    def _get_line_range(node):
        start_line = getattr(node, "lineno", None)
        end_line = getattr(node, "end_lineno", None)

        return start_line, end_line

    # =========================================================
    # Entity extraction
    # =========================================================

    def _extract_entities(
        self,
        tree,
        relative_path,
    ):
        """
        Extract top-level and nested functions/classes.

        Each entity contains:

            file
            name
            type
            start_line
            end_line
            qualified_name
        """

        entities = []

        def visit_node(node, parent_name=None):

            if isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            ):

                entity_name = node.name

                if parent_name:
                    qualified_name = (
                        f"{parent_name}.{entity_name}"
                    )
                else:
                    qualified_name = entity_name

                start_line, end_line = (
                    self._get_line_range(node)
                )

                entities.append(
                    {
                        "file": relative_path,
                        "name": entity_name,
                        "type": self._get_node_type(node),
                        "start_line": start_line,
                        "end_line": end_line,
                        "qualified_name": qualified_name,
                    }
                )
                cell_number = getattr(
                    node,
                    "_codelens_cell_number",
                    None,
                )
                if cell_number is not None:
                    entities[-1]["cell_number"] = cell_number

                # Continue into nested entities.
                for child in ast.iter_child_nodes(node):
                    visit_node(
                        child,
                        qualified_name,
                    )

                return

            for child in ast.iter_child_nodes(node):
                visit_node(
                    child,
                    parent_name,
                )

        visit_node(tree)

        return entities

    # =========================================================
    # Python file parsing
    # =========================================================

    def _parse_files(self):

        files = []
        failed_files = []

        for path in self._iter_python_files():

            relative_path = self._relative_path(path)

            module = self._module_name(
                relative_path
            )

            try:
                if path.suffix.lower() == ".ipynb":
                    cells = read_notebook_cells(path)
                    module_trees = []
                    cell_errors = []

                    for cell in cells:
                        if cell["cell_type"] != "code" or not cell["source"].strip():
                            continue

                        try:
                            cell_tree = ast.parse(cell["source"])
                        except (SyntaxError, ValueError, TypeError):
                            cell_errors.append(cell["cell_number"])
                            continue

                        for node in ast.walk(cell_tree):
                            node._codelens_cell_number = cell["cell_number"]
                        module_trees.append(cell_tree)

                    tree = ast.Module(
                        body=[
                            node
                            for cell_tree in module_trees
                            for node in cell_tree.body
                        ],
                        type_ignores=[],
                    )
                    imports = sorted({
                        import_name
                        for cell_tree in module_trees
                        for import_name in self._extract_imports(cell_tree)
                    })
                    if cell_errors:
                        failed_files.append({
                            "path": relative_path,
                            "error": "Some notebook code cells could not be parsed.",
                            "cell_numbers": cell_errors,
                        })
                    if not module_trees and cell_errors:
                        tree = None
                        imports = []
                else:
                    source = path.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
                    tree = ast.parse(source)
                    imports = self._extract_imports(tree)

                entities = (
                    self._extract_entities(tree, relative_path)
                    if tree is not None
                    else []
                )

                files.append(
                    {
                        "path": relative_path,
                        "module": module,
                        "imports": imports,
                        "entities": entities,
                        "tree": tree,
                    }
                )

            except NotebookParseError:
                failed_files.append({
                    "path": relative_path,
                    "error": "Notebook is not valid readable JSON or has no cells list.",
                })
                files.append({
                    "path": relative_path,
                    "module": module,
                    "imports": [],
                    "entities": [],
                    "tree": None,
                })

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
                        "entities": [],
                        "tree": None,
                    }
                )

        return files, failed_files

    # =========================================================
    # Internal module mapping
    # =========================================================

    def _build_module_map(self, files):

        module_map = {}

        for file_info in files:

            module = file_info["module"]

            if module:
                module_map[module] = file_info["path"]

        return module_map

    # =========================================================
    # Internal dependency detection
    # =========================================================

    def _find_internal_target(
        self,
        imported_module,
        module_map,
    ):

        if imported_module in module_map:
            return module_map[imported_module]

        for module_name, file_path in module_map.items():

            if imported_module.startswith(
                module_name + "."
            ):
                return file_path

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

            for imported_module in file_info["imports"]:

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

        unique = {}

        for dependency in dependencies:

            key = (
                dependency["from"],
                dependency["to"],
                dependency["import"],
            )

            unique[key] = dependency

        return list(unique.values())

    # =========================================================
    # External dependency detection
    # =========================================================

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

            for imported_module in file_info["imports"]:

                top_level = (
                    self._top_level_module(
                        imported_module
                    )
                )

                if top_level in self.STANDARD_LIBRARY:
                    continue

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

    # =========================================================
    # Reverse dependency graph
    # =========================================================

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
                    "import": dependency["import"],
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

    # =========================================================
    # Direct file dependency information
    # =========================================================

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

            depends_on[source].append(target)
            depended_on_by[target].append(source)

        summaries = []

        for file_info in files:

            path = file_info["path"]

            summaries.append(
                {
                    "file": path,
                    "module": file_info["module"],
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

    # =========================================================
    # Dependency hotspots
    # =========================================================

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

    # =========================================================
    # File dependency graph
    # =========================================================

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
                    "module": file_info["module"],
                }
            )

        edges = []

        for dependency in internal_dependencies:

            edges.append(
                {
                    "source": dependency["from"],
                    "target": dependency["to"],
                    "relationship": dependency[
                        "relationship"
                    ],
                    "import": dependency["import"],
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    # =========================================================
    # Transitive dependency analysis
    # =========================================================

    def _build_transitive_dependencies(
        self,
        files,
        internal_dependencies,
    ):

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
                        queue.append(next_file)

            direct = graph.get(
                start_file,
                set(),
            )

            indirect = visited - direct

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

    # =========================================================
    # Entity lookup
    # =========================================================

    def _build_entity_index(self, files):

        entity_index = {}

        for file_info in files:

            for entity in file_info.get(
                "entities",
                [],
            ):

                key = (
                    entity["file"],
                    entity["name"],
                )

                entity_index[key] = entity

        return entity_index

    # =========================================================
    # Entity name lookup
    # =========================================================

    def _build_entity_name_index(self, files):

        name_index = defaultdict(list)

        for file_info in files:

            for entity in file_info.get(
                "entities",
                [],
            ):

                name_index[
                    entity["name"]
                ].append(entity)

        return name_index

    # =========================================================
    # Entity dependency extraction
    # =========================================================

    def _build_entity_dependencies(
        self,
        files,
    ):
        """
        Detect relationships such as:

            answer_question()
                -> retrieve_context()

            SomeClass
                -> OtherClass

        This is static AST analysis. It does not execute code.

        Because Python is dynamically typed, this analyzer uses
        conservative name matching rather than pretending to know
        runtime dispatch with certainty.
        """

        entity_name_index = (
            self._build_entity_name_index(files)
        )

        dependencies = []

        for file_info in files:

            tree = file_info.get("tree")

            if tree is None:
                continue
            # -----------------------------------------
            # Module-level entity imports
            # -----------------------------------------
            #
            # Detect relationships such as:
            #
            #     from rag import retrieve_context
            #
            #     test_retrieval.py
            #             |
            #             | imports_entity
            #             v
            #     rag.py::retrieve_context
            #
            # These relationships originate from the
            # module itself, so "__module__" is used
            # as the source entity.
            # -----------------------------------------

            for import_node in ast.iter_child_nodes(tree):

                if isinstance(import_node, ast.ImportFrom):

                    for alias in import_node.names:

                        imported_name = alias.name

                        if imported_name == "*":
                            continue

                        targets = (
                            entity_name_index.get(
                                imported_name,
                                [],
                            )
                        )

                        for target in targets:

                            if target["file"] == file_info["path"]:
                                continue

                            dependencies.append(
                                {
                                    "from_file": file_info[
                                        "path"
                                    ],
                                    "from_entity": "__module__",
                                    "from_entity_type": "module",
                                    "to_file": target[
                                        "file"
                                    ],
                                    "to_entity": target[
                                        "name"
                                    ],
                                    "to_entity_type": target[
                                        "type"
                                    ],
                                    "relationship": (
                                        "imports_entity"
                                    ),
                                    "start_line": (
                                        import_node.lineno
                                    ),
                                }
                            )

                elif isinstance(import_node, ast.Import):

                    for alias in import_node.names:

                        imported_root = (
                            alias.name.split(".")[0]
                        )

                        targets = (
                            entity_name_index.get(
                                imported_root,
                                [],
                            )
                        )

                        for target in targets:

                            if target["file"] == file_info["path"]:
                                continue

                            dependencies.append(
                                {
                                    "from_file": file_info[
                                        "path"
                                    ],
                                    "from_entity": "__module__",
                                    "from_entity_type": "module",
                                    "to_file": target[
                                        "file"
                                    ],
                                    "to_entity": target[
                                        "name"
                                    ],
                                    "to_entity_type": target[
                                        "type"
                                    ],
                                    "relationship": (
                                        "imports_entity"
                                    ),
                                    "start_line": (
                                        import_node.lineno
                                    ),
                                }
                            )
            entities = file_info.get(
                "entities",
                [],
            )

            # Map source locations to entities.
            entity_nodes = []

            def collect_nodes(node):

                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                    ),
                ):

                    start_line = getattr(
                        node,
                        "lineno",
                        None,
                    )

                    end_line = getattr(
                        node,
                        "end_lineno",
                        None,
                    )

                    matching_entity = None

                    for entity in entities:

                        if (
                            entity["name"] == node.name
                            and entity["start_line"]
                            == start_line
                        ):
                            matching_entity = entity
                            break

                    if matching_entity:
                        entity_nodes.append(
                            (
                                node,
                                matching_entity,
                            )
                        )

                for child in ast.iter_child_nodes(node):
                    collect_nodes(child)

            collect_nodes(tree)

            for node, source_entity in entity_nodes:

                source_name = source_entity["name"]
                source_file = source_entity["file"]

                # -----------------------------------------
                # Function / class calls
                # -----------------------------------------

                for child in ast.walk(node):

                    if isinstance(
                        child,
                        ast.Call,
                    ):

                        called_name = None

                        if isinstance(
                            child.func,
                            ast.Name,
                        ):
                            called_name = (
                                child.func.id
                            )

                        elif isinstance(
                            child.func,
                            ast.Attribute,
                        ):
                            called_name = (
                                child.func.attr
                            )

                        if not called_name:
                            continue

                        targets = (
                            entity_name_index.get(
                                called_name,
                                [],
                            )
                        )

                        for target in targets:

                            if (
                                target["file"]
                                == source_file
                                and target["name"]
                                == source_name
                            ):
                                continue

                            dependencies.append(
                                {
                                    "from_file": source_file,
                                    "from_entity": source_name,
                                    "from_entity_type": (
                                        source_entity[
                                            "type"
                                        ]
                                    ),
                                    "to_file": target[
                                        "file"
                                    ],
                                    "to_entity": target[
                                        "name"
                                    ],
                                    "to_entity_type": target[
                                        "type"
                                    ],
                                    "relationship": (
                                        "calls"
                                    ),
                                    "start_line": (
                                        source_entity[
                                            "start_line"
                                        ]
                                    ),
                                }
                            )

                # -----------------------------------------
                # Name references
                # -----------------------------------------

                for child in ast.walk(node):

                    if not isinstance(
                        child,
                        ast.Name,
                    ):
                        continue

                    referenced_name = child.id

                    targets = (
                        entity_name_index.get(
                            referenced_name,
                            [],
                        )
                    )

                    for target in targets:

                        if (
                            target["file"]
                            == source_file
                            and target["name"]
                            == source_name
                        ):
                            continue

                        # Avoid treating a function call as
                        # both "calls" and "references".
                        parent_call = None

                        for possible_call in ast.walk(node):

                            if isinstance(
                                possible_call,
                                ast.Call,
                            ):

                                if (
                                    isinstance(
                                        possible_call.func,
                                        ast.Name,
                                    )
                                    and possible_call.func
                                    is child
                                ):
                                    parent_call = (
                                        possible_call
                                    )
                                    break

                        if parent_call is not None:
                            continue

                        dependencies.append(
                            {
                                "from_file": source_file,
                                "from_entity": source_name,
                                "from_entity_type": (
                                    source_entity[
                                        "type"
                                    ]
                                ),
                                "to_file": target[
                                    "file"
                                ],
                                "to_entity": target[
                                    "name"
                                ],
                                "to_entity_type": target[
                                    "type"
                                ],
                                "relationship": (
                                    "references"
                                ),
                                "start_line": (
                                    source_entity[
                                        "start_line"
                                    ]
                                ),
                            }
                        )

        # Remove duplicates.
        unique = {}

        for dependency in dependencies:

            key = (
                dependency["from_file"],
                dependency["from_entity"],
                dependency["to_file"],
                dependency["to_entity"],
                dependency["relationship"],
            )

            unique[key] = dependency

        return list(unique.values())

    # =========================================================
    # Entity reverse dependencies
    # =========================================================

    def _build_entity_reverse_dependencies(
        self,
        entity_dependencies,
    ):

        reverse = defaultdict(list)

        for dependency in entity_dependencies:

            key = (
                dependency["to_file"],
                dependency["to_entity"],
            )

            reverse[key].append(
                {
                    "file": dependency[
                        "from_file"
                    ],
                    "entity": dependency[
                        "from_entity"
                    ],
                    "relationship": dependency[
                        "relationship"
                    ],
                }
            )

        return {
            f"{file}:{entity}": sorted(
                dependents,
                key=lambda item: (
                    item["file"],
                    item["entity"],
                ),
            )
            for (file, entity), dependents
            in reverse.items()
        }

    # =========================================================
    # Entity dependency summary
    # =========================================================

    def _build_entity_dependency_summary(
        self,
        files,
        entity_dependencies,
    ):

        depends_on = defaultdict(set)
        depended_on_by = defaultdict(set)

        for dependency in entity_dependencies:

            source = (
                dependency["from_file"],
                dependency["from_entity"],
            )

            target = (
                dependency["to_file"],
                dependency["to_entity"],
            )

            depends_on[source].add(target)
            depended_on_by[target].add(source)

        summaries = []

        for file_info in files:

            for entity in file_info.get(
                "entities",
                [],
            ):

                key = (
                    entity["file"],
                    entity["name"],
                )

                direct_dependencies = sorted(
                    [
                        {
                            "file": target[0],
                            "entity": target[1],
                        }
                        for target in depends_on.get(
                            key,
                            set(),
                        )
                    ],
                    key=lambda item: (
                        item["file"],
                        item["entity"],
                    ),
                )

                dependents = sorted(
                    [
                        {
                            "file": source[0],
                            "entity": source[1],
                        }
                        for source in depended_on_by.get(
                            key,
                            set(),
                        )
                    ],
                    key=lambda item: (
                        item["file"],
                        item["entity"],
                    ),
                )

                summary = {
                    "file": entity["file"],
                    "entity": entity["name"],
                    "type": entity["type"],
                    "qualified_name": entity["qualified_name"],
                    "start_line": entity["start_line"],
                    "end_line": entity["end_line"],
                    "dependencies": direct_dependencies,
                    "depended_on_by": dependents,
                    "dependency_count": len(direct_dependencies),
                    "dependent_count": len(dependents),
                }
                if "cell_number" in entity:
                    summary["cell_number"] = entity["cell_number"]
                summaries.append(summary)

        return summaries

    # =========================================================
    # Entity dependency graph
    # =========================================================

    def _build_entity_dependency_graph(
        self,
        files,
        entity_dependencies,
    ):

        nodes = []

        for file_info in files:

            for entity in file_info.get(
                "entities",
                [],
            ):

                entity_id = (
                    f"{entity['file']}::"
                    f"{entity['qualified_name']}"
                )

                nodes.append(
                    {
                        "id": entity_id,
                        "file": entity["file"],
                        "name": entity["name"],
                        "qualified_name": (
                            entity[
                                "qualified_name"
                            ]
                        ),
                        "type": entity["type"],
                        "start_line": entity[
                            "start_line"
                        ],
                        "end_line": entity[
                            "end_line"
                        ],
                    }
                )

        edges = []

        for dependency in entity_dependencies:

            source_id = (
                f"{dependency['from_file']}::"
                f"{dependency['from_entity']}"
            )

            target_id = (
                f"{dependency['to_file']}::"
                f"{dependency['to_entity']}"
            )

            edges.append(
                {
                    "source": source_id,
                    "target": target_id,
                    "relationship": dependency[
                        "relationship"
                    ],
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
        }

    # =========================================================
    # Main analysis
    # =========================================================

    def analyze(self):

        self._validate_repository()

        files, failed_files = (
            self._parse_files()
        )

        module_map = (
            self._build_module_map(files)
        )

        # ---------------------------------------------
        # File-level analysis
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Entity-level analysis
        # ---------------------------------------------

        entity_dependencies = (
            self._build_entity_dependencies(
                files
            )
        )

        entity_reverse_dependencies = (
            self._build_entity_reverse_dependencies(
                entity_dependencies
            )
        )

        entity_summaries = (
            self._build_entity_dependency_summary(
                files,
                entity_dependencies,
            )
        )

        entity_dependency_graph = (
            self._build_entity_dependency_graph(
                files,
                entity_dependencies,
            )
        )

        # ---------------------------------------------
        # Counts
        # ---------------------------------------------

        total_internal = len(
            internal_dependencies
        )

        total_external = sum(
            len(dependencies)
            for dependencies
            in external_dependencies.values()
        )

        total_entities = sum(
            len(
                file_info.get(
                    "entities",
                    [],
                )
            )
            for file_info in files
        )
        total_python_files = sum(
            Path(file_info["path"]).suffix.lower() != ".ipynb"
            for file_info in files
        )
        total_notebooks = sum(
            Path(file_info["path"]).suffix.lower() == ".ipynb"
            for file_info in files
        )

        # ---------------------------------------------
        # Final result
        # ---------------------------------------------

        return {
            "repository_id": self.repository_id,

            # File-level information
            "total_python_files": total_python_files,
            "total_notebooks": total_notebooks,

            "analyzed_files": sum(
                file_info.get("tree") is not None
                for file_info in files
            ),

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

            # Entity-level information
            "total_entities": total_entities,

            "entity_dependencies": (
                entity_dependencies
            ),

            "entity_reverse_dependencies": (
                entity_reverse_dependencies
            ),

            "entity_dependency_summary": (
                entity_summaries
            ),

            "entity_dependency_graph": (
                entity_dependency_graph
            ),
        }