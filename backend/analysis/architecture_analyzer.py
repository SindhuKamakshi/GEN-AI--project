import ast
from pathlib import Path
from collections import Counter, defaultdict

from config import REPOSITORIES_DIR


class ArchitectureAnalyzer:
    """
    Static architecture analyzer for Python repositories.

    Detects:
    - Python modules/files
    - Entry points
    - Internal imports
    - External dependencies
    - Frameworks
    - AI/ML/LLM technologies
    - Databases/vector databases
    - Architecture layers
    - File dependency graph

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

    ENTRY_POINT_NAMES = {
        "main.py",
        "app.py",
        "server.py",
        "run.py",
        "manage.py",
        "wsgi.py",
        "asgi.py",
    }

    FRAMEWORKS = {
        "fastapi": "FastAPI",
        "flask": "Flask",
        "django": "Django",
        "streamlit": "Streamlit",
        "gradio": "Gradio",
    }

    AI_ML_LIBRARIES = {
        "tensorflow",
        "torch",
        "pytorch",
        "keras",
        "sklearn",
        "scikit-learn",
        "transformers",
        "sentence_transformers",
        "langchain",
        "langgraph",
        "openai",
        "anthropic",
        "ollama",
        "chromadb",
        "faiss",
        "numpy",
        "pandas",
    }

    LLM_LIBRARIES = {
        "google",
        "google.generativeai",
        "openai",
        "anthropic",
        "ollama",
        "transformers",
        "langchain",
        "langgraph",
    }

    DATABASE_LIBRARIES = {
        "sqlite3": "SQLite",
        "sqlalchemy": "SQLAlchemy",
        "psycopg2": "PostgreSQL",
        "psycopg": "PostgreSQL",
        "pymysql": "MySQL",
        "mysql": "MySQL",
        "pymongo": "MongoDB",
        "redis": "Redis",
        "chromadb": "ChromaDB",
        "faiss": "FAISS",
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
    }

    LAYER_RULES = {
        "ui": {
            "names": {
                "app.py",
                "main.py",
                "frontend.py",
                "ui.py",
                "interface.py",
                "streamlit_app.py",
            },
            "keywords": {
                "ui",
                "frontend",
                "interface",
                "streamlit",
                "gradio",
                "dashboard",
            },
        },
        "api": {
            "names": {
                "api.py",
                "routes.py",
                "router.py",
                "views.py",
                "endpoints.py",
            },
            "keywords": {
                "api",
                "route",
                "router",
                "endpoint",
                "controller",
                "view",
            },
        },
        "service": {
            "names": {
                "service.py",
                "services.py",
                "pipeline.py",
                "rag.py",
                "engine.py",
            },
            "keywords": {
                "service",
                "pipeline",
                "engine",
                "processor",
                "manager",
            },
        },
        "data": {
            "names": {
                "database.py",
                "db.py",
                "models.py",
                "repository.py",
                "vector_store.py",
                "storage.py",
            },
            "keywords": {
                "database",
                "db",
                "model",
                "repository",
                "storage",
                "vector",
                "embedding",
            },
        },
        "config": {
            "names": {
                "config.py",
                "settings.py",
                "constants.py",
            },
            "keywords": {
                "config",
                "settings",
                "constant",
            },
        },
        "evaluation": {
            "names": {
                "evaluate.py",
                "evaluation.py",
                "evaluator.py",
            },
            "keywords": {
                "evaluate",
                "evaluation",
                "evaluator",
                "benchmark",
                "metrics",
            },
        },
        "testing": {
            "names": set(),
            "keywords": {
                "test",
                "tests",
            },
        },
    }

    def __init__(self, repository_id: str):
        if not repository_id:
            raise ValueError("repository_id is required")

        self.repository_id = repository_id
        self.repository_path = REPOSITORIES_DIR / repository_id

    def _validate_repository(self):
        if not self.repository_path.exists():
            raise FileNotFoundError(
                f"Repository '{self.repository_id}' was not found"
            )

        if not self.repository_path.is_dir():
            raise ValueError(
                f"Repository path is not a directory: {self.repository_path}"
            )

    def _iter_python_files(self):
        for path in self.repository_path.rglob("*.py"):
            relative_parts = path.relative_to(
                self.repository_path
            ).parts

            if any(part in self.DEV_DIRS for part in relative_parts):
                continue

            yield path

    def _relative_path(self, path):
        return str(
            path.relative_to(self.repository_path)
        ).replace("\\", "/")

    @staticmethod
    def _module_name(path):
        """
        Convert:

            package/utils.py
            package/__init__.py

        into Python-style module names.
        """

        parts = list(path.parts)

        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = Path(parts[-1]).stem

        return ".".join(parts)

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

    @staticmethod
    def _top_level_module(import_name):
        return import_name.split(".")[0]

    def _detect_entry_point(self, path, tree):
        filename = path.name.lower()

        if filename in self.ENTRY_POINT_NAMES:
            return True

        for node in ast.walk(tree):

            if isinstance(node, ast.If):

                test = node.test

                if (
                    isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "__name__"
                ):
                    return True

        return False

    def _detect_technologies(self, imports):
        frameworks = set()
        ai_ml = set()
        llm = set()
        databases = set()

        for import_name in imports:

            top_level = self._top_level_module(import_name)

            # Frameworks
            if top_level in self.FRAMEWORKS:
                frameworks.add(
                    self.FRAMEWORKS[top_level]
                )

            # AI / ML
            if (
                top_level in self.AI_ML_LIBRARIES
                or import_name in self.AI_ML_LIBRARIES
            ):
                ai_ml.add(import_name)

            # LLM technologies
            if (
                top_level in self.LLM_LIBRARIES
                or import_name in self.LLM_LIBRARIES
            ):
                llm.add(import_name)

            # Databases
            if top_level in self.DATABASE_LIBRARIES:
                databases.add(
                    self.DATABASE_LIBRARIES[top_level]
                )

        return {
            "frameworks": sorted(frameworks),
            "ai_ml": sorted(ai_ml),
            "llm": sorted(llm),
            "databases": sorted(databases),
        }

    def _classify_layer(self, path, imports):
        filename = path.name.lower()
        path_text = str(path).lower().replace("\\", "/")

        # Testing has priority.
        if (
            filename.startswith("test_")
            or filename.endswith("_test.py")
            or "/tests/" in path_text
            or "/test/" in path_text
        ):
            return "testing"

        # Evaluation has priority over generic service/other rules.
        if (
            filename in {
                "evaluate.py",
                "evaluation.py",
                "evaluator.py",
            }
            or filename.startswith("evaluate_")
            or filename.startswith("evaluation_")
            or filename.endswith("_evaluation.py")
            or "benchmark" in filename
            or "metrics" in filename
        ):
            return "evaluation"

        # Explicit filename rules.
        for layer, rules in self.LAYER_RULES.items():

            if filename in rules["names"]:
                return layer

        # Keyword rules.
        for layer, rules in self.LAYER_RULES.items():

            for keyword in rules["keywords"]:
                if (
                    keyword in filename
                    or keyword in path_text
                ):
                    return layer

        # UI framework imports.
        for import_name in imports:

            if import_name.startswith(
                ("streamlit", "gradio")
            ):
                return "ui"

        # API framework imports.
        for import_name in imports:

            if import_name.startswith(
                ("fastapi", "flask", "django")
            ):
                return "api"

        return "other"

    def _parse_file(self, path):
        relative_path = self._relative_path(path)

        try:
            source = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            tree = ast.parse(source)

        except (
            SyntaxError,
            OSError,
            UnicodeDecodeError,
        ) as exc:

            return {
                "path": relative_path,
                "module": self._module_name(
                    path.relative_to(
                        self.repository_path
                    )
                ),
                "imports": [],
                "entry_point": False,
                "layer": "other",
                "technologies": {},
                "parse_error": str(exc),
            }

        imports = self._extract_imports(tree)

        technologies = self._detect_technologies(
            imports
        )

        layer = self._classify_layer(
            path.relative_to(
                self.repository_path
            ),
            imports,
        )

        return {
            "path": relative_path,
            "module": self._module_name(
                path.relative_to(
                    self.repository_path
                )
            ),
            "imports": imports,
            "entry_point": self._detect_entry_point(
                path,
                tree,
            ),
            "layer": layer,
            "technologies": technologies,
            "parse_error": None,
        }

    def _build_internal_dependencies(self, files):
        """
        Determine which imports belong to this repository.

        Only imports that can be mapped to an analyzed
        repository module are considered internal.
        """

        module_to_file = {}

        for file_info in files:

            module = file_info["module"]

            if module:
                module_to_file[module] = (
                    file_info["path"]
                )

        dependencies = []

        for file_info in files:

            source_file = file_info["path"]

            for imported_module in file_info[
                "imports"
            ]:

                target_file = None

                # Exact match.
                if imported_module in module_to_file:
                    target_file = module_to_file[
                        imported_module
                    ]

                # Handle nested module relationships.
                if target_file is None:

                    for (
                        module_name,
                        file_path,
                    ) in module_to_file.items():

                        if (
                            imported_module.startswith(
                                module_name + "."
                            )
                            or module_name.startswith(
                                imported_module + "."
                            )
                        ):
                            target_file = file_path
                            break

                if target_file is None:
                    continue

                if target_file == source_file:
                    continue

                dependencies.append(
                    {
                        "from": source_file,
                        "to": target_file,
                        "import": imported_module,
                        "relationship": "imports",
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

    def _build_external_dependencies(
        self,
        files,
    ):
        """
        Identify external dependencies after internal
        repository modules have been excluded.
        """

        internal_modules = {
            file_info["module"].split(".")[0]
            for file_info in files
            if file_info["module"]
        }

        external = set()

        for file_info in files:

            for imported_module in file_info[
                "imports"
            ]:

                top_level = self._top_level_module(
                    imported_module
                )

                # Ignore Python standard library.
                if top_level in self.STANDARD_LIBRARY:
                    continue

                # Ignore repository modules.
                if top_level in internal_modules:
                    continue

                external.add(top_level)

        return sorted(external)

    def _build_graph(
        self,
        files,
        dependencies,
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
                    "layer": file_info["layer"],
                    "entry_point": file_info[
                        "entry_point"
                    ],
                }
            )

        edges = []

        for dependency in dependencies:

            edges.append(
                {
                    "source": dependency["from"],
                    "target": dependency["to"],
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

    def _build_layer_summary(self, files):
        summary = defaultdict(list)

        for file_info in files:

            summary[
                file_info["layer"]
            ].append(file_info["path"])

        return {
            layer: sorted(paths)
            for layer, paths in summary.items()
        }

    def _build_architecture_summary(
        self,
        files,
        dependencies,
        external_dependencies,
    ):
        layers = Counter(
            file_info["layer"]
            for file_info in files
        )

        frameworks = set()
        ai_ml = set()
        llm = set()
        databases = set()

        for file_info in files:

            technologies = file_info[
                "technologies"
            ]

            frameworks.update(
                technologies.get(
                    "frameworks",
                    [],
                )
            )

            ai_ml.update(
                technologies.get(
                    "ai_ml",
                    [],
                )
            )

            llm.update(
                technologies.get(
                    "llm",
                    [],
                )
            )

            databases.update(
                technologies.get(
                    "databases",
                    [],
                )
            )

        entry_points = [
            file_info["path"]
            for file_info in files
            if file_info["entry_point"]
        ]

        non_testing_layers = {
            layer: count
            for layer, count in layers.items()
            if layer != "testing"
        }

        if len(non_testing_layers) >= 2:
            architecture_style = (
                "layered_application"
            )
        else:
            architecture_style = (
                "simple_application"
            )

        return {
            "architecture_style": architecture_style,
            "entry_points": sorted(entry_points),
            "layers_detected": dict(layers),
            "frameworks": sorted(frameworks),
            "ai_ml_technologies": sorted(ai_ml),
            "llm_technologies": sorted(llm),
            "databases": sorted(databases),
            "internal_dependency_count": len(
                dependencies
            ),
            "external_dependency_count": len(
                external_dependencies
            ),
            "external_dependencies": (
                external_dependencies
            ),
        }

    def analyze(self):
        self._validate_repository()

        python_files = list(
            self._iter_python_files()
        )

        files = []
        failed_files = []

        for path in python_files:

            result = self._parse_file(path)

            if result["parse_error"]:

                failed_files.append(
                    {
                        "path": result["path"],
                        "error": result[
                            "parse_error"
                        ],
                    }
                )

            files.append(result)

        internal_dependencies = (
            self._build_internal_dependencies(
                files
            )
        )

        external_dependencies = (
            self._build_external_dependencies(
                files
            )
        )

        dependency_graph = self._build_graph(
            files,
            internal_dependencies,
        )

        layer_summary = (
            self._build_layer_summary(files)
        )

        architecture_summary = (
            self._build_architecture_summary(
                files,
                internal_dependencies,
                external_dependencies,
            )
        )

        return {
            "repository_id": self.repository_id,

            "total_python_files": len(
                python_files
            ),

            "analyzed_files": len(files),

            "failed_files": failed_files,

            "files": files,

            "internal_dependencies": (
                internal_dependencies
            ),

            "external_dependencies": (
                external_dependencies
            ),

            "dependency_graph": dependency_graph,

            "layer_summary": layer_summary,

            "architecture_summary": (
                architecture_summary
            ),
        }