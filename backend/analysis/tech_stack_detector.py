import ast
from pathlib import Path
import re

from repository.notebook_utils import NotebookParseError, read_notebook_cells


class TechnologyStackDetector:
    """
    Deterministic technology-stack detection.

    Detects technologies from:
    - Repository file extensions
    - Dependency/configuration files
    - Source-code imports
    - Project configuration
    """

    LANGUAGE_MAP = {
        ".py": "Python",
        ".pyw": "Python",
        ".ipynb": "Jupyter Notebook",
        ".js": "JavaScript",
        ".jsx": "JavaScript/React",
        ".ts": "TypeScript",
        ".tsx": "TypeScript/React",
        ".java": "Java",
        ".c": "C",
        ".cpp": "C++",
        ".cc": "C++",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".php": "PHP",
        ".rb": "Ruby",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".kts": "Kotlin",
        ".html": "HTML",
        ".htm": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".sass": "Sass",
        ".sql": "SQL",
        ".sh": "Shell",
        ".bash": "Shell",
    }

    CONFIGURATION_FILES = {
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pom.xml",
        "build.gradle",
        "build.gradle.kts",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "tsconfig.json",
        "vite.config.js",
        "vite.config.ts",
    }

    PACKAGE_MANAGERS = {
        "requirements.txt": "pip",
        "pyproject.toml": "pip / Poetry / PEP 621",
        "setup.py": "setuptools",
        "package.json": "npm",
        "package-lock.json": "npm",
        "yarn.lock": "Yarn",
        "pnpm-lock.yaml": "pnpm",
        "pom.xml": "Maven",
        "build.gradle": "Gradle",
        "build.gradle.kts": "Gradle",
    }

    # ------------------------------------------------------------
    # Dependency name → technology category
    # ------------------------------------------------------------

    PYTHON_TECHNOLOGIES = {
        "fastapi": ("frameworks", "FastAPI"),
        "flask": ("frameworks", "Flask"),
        "django": ("frameworks", "Django"),
        "streamlit": ("frameworks", "Streamlit"),
        "gradio": ("frameworks", "Gradio"),
        "uvicorn": ("frameworks", "Uvicorn"),
        "starlette": ("frameworks", "Starlette"),

        "sentence-transformers": (
            "ai_ml",
            "Sentence Transformers",
        ),
        "transformers": (
            "ai_ml",
            "Hugging Face Transformers",
        ),
        "torch": (
            "ai_ml",
            "PyTorch",
        ),
        "tensorflow": (
            "ai_ml",
            "TensorFlow",
        ),
        "keras": (
            "ai_ml",
            "Keras",
        ),
        "scikit-learn": (
            "ai_ml",
            "Scikit-learn",
        ),
        "sklearn": (
            "ai_ml",
            "Scikit-learn",
        ),
        "xgboost": (
            "ai_ml",
            "XGBoost",
        ),

        "google-genai": (
            "ai_ml",
            "Google Gemini",
        ),
        "google-generativeai": (
            "ai_ml",
            "Google Generative AI",
        ),
        "openai": (
            "ai_ml",
            "OpenAI",
        ),
        "anthropic": (
            "ai_ml",
            "Anthropic",
        ),
        "langchain": (
            "ai_ml",
            "LangChain",
        ),
        "langgraph": (
            "ai_ml",
            "LangGraph",
        ),

        "chromadb": (
            "vector_databases",
            "ChromaDB",
        ),
        "faiss-cpu": (
            "vector_databases",
            "FAISS",
        ),
        "faiss-gpu": (
            "vector_databases",
            "FAISS",
        ),
        "pinecone": (
            "vector_databases",
            "Pinecone",
        ),
        "weaviate-client": (
            "vector_databases",
            "Weaviate",
        ),
        "qdrant-client": (
            "vector_databases",
            "Qdrant",
        ),

        "sqlalchemy": (
            "databases",
            "SQLAlchemy",
        ),
        "psycopg2": (
            "databases",
            "PostgreSQL",
        ),
        "psycopg2-binary": (
            "databases",
            "PostgreSQL",
        ),
        "pymysql": (
            "databases",
            "MySQL",
        ),
        "mysql-connector-python": (
            "databases",
            "MySQL",
        ),
        "pymongo": (
            "databases",
            "MongoDB",
        ),
        "redis": (
            "databases",
            "Redis",
        ),

        "pandas": (
            "data_science",
            "Pandas",
        ),
        "numpy": (
            "data_science",
            "NumPy",
        ),
        "scipy": (
            "data_science",
            "SciPy",
        ),
        "matplotlib": (
            "data_science",
            "Matplotlib",
        ),

        "pymupdf": (
            "libraries",
            "PyMuPDF",
        ),
        "fitz": (
            "libraries",
            "PyMuPDF",
        ),
        "python-dotenv": (
            "libraries",
            "python-dotenv",
        ),
        "dotenv": (
            "libraries",
            "python-dotenv",
        ),
        "requests": (
            "libraries",
            "Requests",
        ),
        "httpx": (
            "libraries",
            "HTTPX",
        ),
        "pydantic": (
            "libraries",
            "Pydantic",
        ),

        "pytest": (
            "testing",
            "pytest",
        ),
        "unittest": (
            "testing",
            "unittest",
        ),
    }

    JAVASCRIPT_TECHNOLOGIES = {
        "react": (
            "frameworks",
            "React",
        ),
        "next": (
            "frameworks",
            "Next.js",
        ),
        "vue": (
            "frameworks",
            "Vue.js",
        ),
        "angular": (
            "frameworks",
            "Angular",
        ),
        "express": (
            "frameworks",
            "Express.js",
        ),
        "fastify": (
            "frameworks",
            "Fastify",
        ),
        "vite": (
            "build_tools",
            "Vite",
        ),
        "webpack": (
            "build_tools",
            "Webpack",
        ),
        "jest": (
            "testing",
            "Jest",
        ),
        "vitest": (
            "testing",
            "Vitest",
        ),
        "axios": (
            "libraries",
            "Axios",
        ),
    }

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------

    def __init__(self, repository_path: Path):
        self.repository_path = Path(repository_path)

        if not self.repository_path.exists():
            raise ValueError("Repository path does not exist.")

        if not self.repository_path.is_dir():
            raise ValueError(
                "Repository path must be a directory."
            )

    # ------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------

    def get_files(self) -> list[Path]:
        files = []

        ignored_directories = {
            ".git",
            ".hg",
            ".svn",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "env",
            "virtualenv",
            ".idea",
            ".vscode",
            "dist",
            "build",
            "out",
            "target",
            "coverage",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }

        for path in self.repository_path.rglob("*"):

            if not path.is_file():
                continue

            relative_parts = path.relative_to(
                self.repository_path
            ).parts

            if any(
                part.lower() in ignored_directories
                for part in relative_parts
            ):
                continue

            files.append(path)

        return sorted(files)

    # ------------------------------------------------------------
    # Languages
    # ------------------------------------------------------------

    def detect_languages(
        self,
        files: list[Path],
    ) -> list[str]:

        languages = set()

        for file_path in files:

            extension = file_path.suffix.lower()

            language = self.LANGUAGE_MAP.get(extension)

            if language:
                languages.add(language)

        return sorted(languages)

    # ------------------------------------------------------------
    # Configuration files
    # ------------------------------------------------------------

    def detect_configuration_files(
        self,
        files: list[Path],
    ) -> list[str]:

        result = []

        for file_path in files:

            filename = file_path.name.lower()

            if filename in self.CONFIGURATION_FILES:

                result.append(
                    str(
                        file_path.relative_to(
                            self.repository_path
                        )
                    )
                )

        return sorted(result)

    # ------------------------------------------------------------
    # Read text safely
    # ------------------------------------------------------------

    def read_text_file(
        self,
        file_path: Path,
        max_bytes: int = 2_000_000,
    ) -> str:

        try:

            if file_path.stat().st_size > max_bytes:
                return ""

            return file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except (OSError, UnicodeError):
            return ""

    # ------------------------------------------------------------
    # Extract dependency names from requirements.txt
    # ------------------------------------------------------------

    def parse_requirements(
        self,
        file_path: Path,
    ) -> set[str]:

        dependencies = set()

        content = self.read_text_file(file_path)

        for line in content.splitlines():

            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("-r "):
                continue

            if line.startswith("--"):
                continue

            # Remove environment markers.
            line = line.split(";")[0].strip()

            # Remove version operators.
            match = re.match(
                r"^([A-Za-z0-9_.-]+)",
                line,
            )

            if match:

                package = match.group(1).lower()

                dependencies.add(package)

        return dependencies

    # ------------------------------------------------------------
    # Extract imports from Python
    # ------------------------------------------------------------

    def detect_python_imports(
        self,
        file_path: Path,
    ) -> set[str]:

        imports = set()

        content = self.read_text_file(file_path)

        if not content:
            return imports

        for match in re.finditer(
            r"^\s*(?:from|import)\s+([A-Za-z0-9_.-]+)",
            content,
            flags=re.MULTILINE,
        ):

            module = match.group(1)

            root_module = module.split(".")[0]

            imports.add(root_module.lower())

        return imports

    def detect_notebook_imports(self, file_path: Path) -> set[str]:
        imports = set()

        try:
            cells = read_notebook_cells(file_path)
        except NotebookParseError:
            return imports

        for cell in cells:
            if cell["cell_type"] != "code":
                continue

            try:
                tree = ast.parse(cell["source"])
            except (SyntaxError, ValueError, TypeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(
                        alias.name.split(".")[0].lower()
                        for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0].lower())

        return imports

    # ------------------------------------------------------------
    # Extract JavaScript imports
    # ------------------------------------------------------------

    def detect_javascript_imports(
        self,
        file_path: Path,
    ) -> set[str]:

        imports = set()

        content = self.read_text_file(file_path)

        if not content:
            return imports

        patterns = [
            r'import\s+.*?\s+from\s+[\'"]([^\'"]+)',
            r'import\s*[\'"]([^\'"]+)',
            r'require\(\s*[\'"]([^\'"]+)',
        ]

        for pattern in patterns:

            for match in re.finditer(
                pattern,
                content,
            ):

                module = match.group(1)

                if module.startswith("."):
                    continue

                root_module = module.split("/")[0]

                if root_module.startswith("@"):
                    parts = root_module.split("/")

                    if len(parts) >= 2:
                        root_module = (
                            parts[0] + "/" + parts[1]
                        )

                imports.add(root_module.lower())

        return imports

    # ------------------------------------------------------------
    # Detect technologies from dependency names
    # ------------------------------------------------------------

    def classify_python_dependency(
        self,
        dependency: str,
        result: dict,
    ):

        mapping = self.PYTHON_TECHNOLOGIES.get(
            dependency.lower()
        )

        if not mapping:
            return

        category, technology = mapping

        result[category].add(technology)

    def classify_javascript_dependency(
        self,
        dependency: str,
        result: dict,
    ):

        mapping = self.JAVASCRIPT_TECHNOLOGIES.get(
            dependency.lower()
        )

        if not mapping:
            return

        category, technology = mapping

        result[category].add(technology)

    # ------------------------------------------------------------
    # Detect package manager
    # ------------------------------------------------------------

    def detect_package_managers(
        self,
        files: list[Path],
    ) -> list[str]:

        managers = set()

        for file_path in files:

            filename = file_path.name.lower()

            manager = self.PACKAGE_MANAGERS.get(
                filename
            )

            if manager:
                managers.add(manager)

        return sorted(managers)

    # ------------------------------------------------------------
    # Detect databases from repository files
    # ------------------------------------------------------------

    def detect_database_files(
        self,
        files: list[Path],
    ) -> list[str]:

        databases = set()

        for file_path in files:

            extension = file_path.suffix.lower()
            filename = file_path.name.lower()

            if extension == ".sqlite3":
                databases.add("SQLite")

            elif extension == ".sqlite":
                databases.add("SQLite")

            elif extension == ".db":
                databases.add("SQLite / Database file")

            if "postgres" in filename:
                databases.add("PostgreSQL")

            if "mysql" in filename:
                databases.add("MySQL")

            if "mongo" in filename:
                databases.add("MongoDB")

        return sorted(databases)

    # ------------------------------------------------------------
    # Main detection
    # ------------------------------------------------------------

    def detect(self) -> dict:

        files = self.get_files()

        result = {
            "languages": set(),
            "frameworks": set(),
            "ai_ml": set(),
            "vector_databases": set(),
            "databases": set(),
            "data_science": set(),
            "libraries": set(),
            "testing": set(),
            "build_tools": set(),
            "package_managers": set(),
        }

        # --------------------------------------------------------
        # Languages
        # --------------------------------------------------------

        result["languages"].update(
            self.detect_languages(files)
        )

        # --------------------------------------------------------
        # Configuration / package managers
        # --------------------------------------------------------

        result["package_managers"].update(
            self.detect_package_managers(files)
        )

        # --------------------------------------------------------
        # Process files
        # --------------------------------------------------------

        for file_path in files:

            filename = file_path.name.lower()
            extension = file_path.suffix.lower()

            # ----------------------------------------------------
            # requirements.txt
            # ----------------------------------------------------

            if filename == "requirements.txt":

                dependencies = self.parse_requirements(
                    file_path
                )

                for dependency in dependencies:

                    self.classify_python_dependency(
                        dependency,
                        result,
                    )

            # ----------------------------------------------------
            # Python source
            # ----------------------------------------------------

            elif extension in {".py", ".pyw"}:

                imports = self.detect_python_imports(
                    file_path
                )

                for dependency in imports:

                    self.classify_python_dependency(
                        dependency,
                        result,
                    )

            elif extension == ".ipynb":
                for dependency in self.detect_notebook_imports(file_path):
                    self.classify_python_dependency(dependency, result)

            # ----------------------------------------------------
            # JavaScript / TypeScript
            # ----------------------------------------------------

            elif extension in {
                ".js",
                ".jsx",
                ".ts",
                ".tsx",
            }:

                imports = self.detect_javascript_imports(
                    file_path
                )

                for dependency in imports:

                    self.classify_javascript_dependency(
                        dependency,
                        result,
                    )

            # ----------------------------------------------------
            # package.json
            # ----------------------------------------------------

            elif filename == "package.json":

                content = self.read_text_file(
                    file_path
                ).lower()

                for dependency in (
                    self.JAVASCRIPT_TECHNOLOGIES
                ):

                    if (
                        f'"{dependency}"' in content
                        or f"'{dependency}'" in content
                    ):

                        self.classify_javascript_dependency(
                            dependency,
                            result,
                        )

        # --------------------------------------------------------
        # Database files
        # --------------------------------------------------------

        result["databases"].update(
            self.detect_database_files(files)
        )

        # --------------------------------------------------------
        # Convert sets to sorted lists
        # --------------------------------------------------------

        return {
            key: sorted(value)
            for key, value in result.items()
        }