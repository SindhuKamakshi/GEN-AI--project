from pathlib import Path
from typing import List, Dict, Tuple


class ImportantFileAnalyzer:
    """
    Analyze a repository and identify files that are most useful
    for understanding the project.

    The analyzer prioritizes:
    1. Application entry points
    2. Core implementation files
    3. Primary documentation
    4. Configuration and dependency files
    5. Tests and evaluation files
    6. Infrastructure/generated storage
    7. Data and documents
    8. Low-value repository metadata
    """

    # ---------------------------------------------------------
    # Directories that should be ignored completely
    # ---------------------------------------------------------

    IGNORED_DIRECTORIES = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        "node_modules",
        "venv",
        ".venv",
        "env",
        "virtualenv",
        ".idea",
        ".vscode",
        "dist",
        "build",
        "out",
        "target",
        "coverage",
        ".coverage",
    }

    # ---------------------------------------------------------
    # Infrastructure directories
    #
    # These remain visible but are deliberately ranked low.
    # ---------------------------------------------------------

    INFRASTRUCTURE_DIRECTORIES = {
        "chroma_db",
        "chromadb",
        "vector_db",
        "vectorstore",
        "vector_store",
        "embeddings",
    }

    # ---------------------------------------------------------
    # Data / document directories
    # ---------------------------------------------------------

    DATA_DIRECTORIES = {
        "data",
        "dataset",
        "datasets",
        "documents",
        "uploads",
        "files",
        "assets",
        "models",
        "checkpoints",
    }

    # ---------------------------------------------------------
    # Source code extensions
    # ---------------------------------------------------------

    SOURCE_EXTENSIONS = {
        ".py",
        ".pyw",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
    }

    # ---------------------------------------------------------
    # Documentation
    # ---------------------------------------------------------

    DOCUMENTATION_EXTENSIONS = {
        ".md",
        ".rst",
        ".txt",
    }

    # ---------------------------------------------------------
    # Data/document files
    # ---------------------------------------------------------

    DATA_EXTENSIONS = {
        ".csv",
        ".json",
        ".jsonl",
        ".xml",
        ".yaml",
        ".yml",
        ".parquet",
        ".feather",
        ".xlsx",
        ".xls",
        ".pkl",
        ".pickle",
        ".npy",
        ".npz",
        ".pdf",
        ".docx",
        ".doc",
    }

    # ---------------------------------------------------------
    # Binary / generated artifacts
    # ---------------------------------------------------------

    BINARY_EXTENSIONS = {
        ".bin",
        ".so",
        ".dll",
        ".exe",
        ".dylib",
        ".whl",
    }

    # ---------------------------------------------------------
    # Database files
    # ---------------------------------------------------------

    DATABASE_EXTENSIONS = {
        ".db",
        ".sqlite",
        ".sqlite3",
    }

    # ---------------------------------------------------------
    # Important entry points
    # ---------------------------------------------------------

    ENTRY_POINT_FILES = {
        "app.py",
        "main.py",
        "server.py",
        "run.py",
        "manage.py",
        "index.js",
        "server.js",
        "main.js",
        "main.ts",
        "main.tsx",
        "application.py",
    }

    # ---------------------------------------------------------
    # Core implementation file names
    # ---------------------------------------------------------

    CORE_FILE_NAMES = {
        "rag.py",
        "pipeline.py",
        "service.py",
        "services.py",
        "api.py",
        "router.py",
        "routes.py",
        "controller.py",
        "controllers.py",
        "model.py",
        "models.py",
        "database.py",
        "db.py",
        "config.py",
        "settings.py",
        "utils.py",
        "inference.py",
        "embedding.py",
        "embeddings.py",
        "retriever.py",
        "retrieval.py",
        "llm.py",
        "generator.py",
    }

    # ---------------------------------------------------------
    # AI-related keywords
    #
    # These are applied ONLY to non-test implementation files.
    # ---------------------------------------------------------

    AI_FILE_KEYWORDS = {
        "rag",
        "llm",
        "embedding",
        "embed",
        "retrieval",
        "vector",
        "semantic",
        "agent",
        "inference",
        "prompt",
        "classifier",
        "machine_learning",
        "ml",
        "ai",
        "pipeline",
        "transformer",
    }

    # ---------------------------------------------------------
    # Configuration / dependency files
    # ---------------------------------------------------------

    CONFIG_FILE_NAMES = {
        "requirements.txt",
        "requirements-dev.txt",
        "pyproject.toml",
        "poetry.lock",
        "pipfile",
        "pipfile.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "compose.yml",
        "compose.yaml",
        "config.py",
        "settings.py",
        ".env.example",
        ".env.template",
    }

    # ---------------------------------------------------------
    # Primary documentation
    # ---------------------------------------------------------

    PRIMARY_DOCUMENTATION = {
        "readme.md",
        "readme.rst",
        "readme.txt",
    }

    # ---------------------------------------------------------
    # Low-value repository metadata
    # ---------------------------------------------------------

    LOW_VALUE_FILES = {
        ".gitignore",
        ".gitattributes",
        ".gitmodules",
        ".ds_store",
        "thumbs.db",
        ".gitkeep",
    }

    # ---------------------------------------------------------
    # Test naming patterns
    # ---------------------------------------------------------

    TEST_PREFIXES = (
        "test_",
    )

    TEST_SUFFIXES = (
        "_test.py",
        ".test.js",
        ".test.jsx",
        ".test.ts",
        ".test.tsx",
        ".spec.js",
        ".spec.jsx",
        ".spec.ts",
        ".spec.tsx",
    )

    TEST_KEYWORDS = {
        "test",
        "tests",
        "testing",
        "evaluation",
        "evaluate",
        "benchmark",
        "benchmarks",
    }

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(self, repository_path: Path):
        self.repository_path = Path(repository_path)

        if not self.repository_path.exists():
            raise ValueError("Repository path does not exist.")

        if not self.repository_path.is_dir():
            raise ValueError("Repository path must be a directory.")

    # ---------------------------------------------------------
    # File discovery
    # ---------------------------------------------------------

    def get_files(self) -> List[Path]:
        """
        Discover files while skipping generated development
        directories such as .git, venv, node_modules, etc.

        Important:
        chroma_db is NOT ignored because it is useful for
        repository understanding.
        """

        files = []

        for path in self.repository_path.rglob("*"):

            if not path.is_file():
                continue

            relative_path = path.relative_to(self.repository_path)

            parts_lower = {
                part.lower()
                for part in relative_path.parts
            }

            # Ignore development/generated directories.
            if parts_lower.intersection(
                self.IGNORED_DIRECTORIES
            ):
                continue

            files.append(path)

        return sorted(
            files,
            key=lambda path: str(
                path.relative_to(self.repository_path)
            ).lower(),
        )

    # ---------------------------------------------------------
    # Path helpers
    # ---------------------------------------------------------

    def get_relative_path(self, file_path: Path) -> str:
        return str(
            file_path.relative_to(self.repository_path)
        )

    def get_file_name(self, file_path: Path) -> str:
        return file_path.name

    def get_extension(self, file_path: Path) -> str:
        return file_path.suffix.lower()

    # ---------------------------------------------------------
    # Classification helpers
    # ---------------------------------------------------------

    def is_test_file(self, file_path: Path) -> bool:
        """
        Determine whether a file is primarily a test/evaluation file.
        """

        name = file_path.name.lower()
        stem = file_path.stem.lower()

        if any(
            name.startswith(prefix)
            for prefix in self.TEST_PREFIXES
        ):
            return True

        if any(
            name.endswith(suffix)
            for suffix in self.TEST_SUFFIXES
        ):
            return True

        if stem in {
            "evaluation",
            "evaluate",
            "benchmark",
            "benchmarks",
            "testing",
        }:
            return True

        return False

    def is_infrastructure_file(
        self,
        file_path: Path,
    ) -> bool:
        """
        Detect files belonging to infrastructure such as ChromaDB.
        """

        parts = {
            part.lower()
            for part in file_path.relative_to(
                self.repository_path
            ).parts
        }

        return bool(
            parts.intersection(
                self.INFRASTRUCTURE_DIRECTORIES
            )
        )

    def is_data_file(self, file_path: Path) -> bool:
        """
        Detect files located inside known data/document directories.
        """

        parts = {
            part.lower()
            for part in file_path.relative_to(
                self.repository_path
            ).parts
        }

        return bool(
            parts.intersection(
                self.DATA_DIRECTORIES
            )
        )

    def is_source_file(self, file_path: Path) -> bool:
        return (
            self.get_extension(file_path)
            in self.SOURCE_EXTENSIONS
        )

    def is_primary_documentation(
        self,
        file_path: Path,
    ) -> bool:
        return (
            file_path.name.lower()
            in self.PRIMARY_DOCUMENTATION
        )

    def is_config_file(self, file_path: Path) -> bool:
        return (
            file_path.name.lower()
            in self.CONFIG_FILE_NAMES
        )

    def is_low_value_file(self, file_path: Path) -> bool:
        return (
            file_path.name.lower()
            in self.LOW_VALUE_FILES
        )

    # ---------------------------------------------------------
    # AI/core keyword detection
    # ---------------------------------------------------------

    def contains_core_keyword(
        self,
        file_path: Path,
    ) -> bool:

        name = file_path.stem.lower()

        tokens = set(
            name.replace("-", "_")
            .replace(".", "_")
            .split("_")
        )

        return bool(
            tokens.intersection(
                self.AI_FILE_KEYWORDS
            )
        )

    # ---------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------

    def calculate_score(
        self,
        file_path: Path,
    ) -> Tuple[int, List[str]]:

        score = 0
        reasons = []

        name = file_path.name.lower()
        extension = self.get_extension(file_path)

        # -----------------------------------------------------
        # 1. Infrastructure / generated vector storage
        # -----------------------------------------------------

        if self.is_infrastructure_file(file_path):

            score += 5

            reasons.append(
                "Generated vector database or infrastructure storage."
            )

            if extension in self.BINARY_EXTENSIONS:
                reasons.append(
                    "Internal binary/index artifact."
                )

            if extension in self.DATABASE_EXTENSIONS:
                reasons.append(
                    "Database storage file."
                )

            return min(score, 100), reasons

        # -----------------------------------------------------
        # 2. Low-value metadata
        # -----------------------------------------------------

        if self.is_low_value_file(file_path):

            reasons.append(
                "Low-value repository metadata."
            )

            return 0, reasons

        # -----------------------------------------------------
        # 3. Environment file
        #
        # Do NOT treat .env as an important implementation file.
        # It may contain secrets.
        # -----------------------------------------------------

        if name == ".env":

            score += 5

            reasons.append(
                "Environment configuration or secret storage."
            )

            return score, reasons

        # -----------------------------------------------------
        # 4. Data / document storage
        # -----------------------------------------------------

        if self.is_data_file(file_path):

            score += 5

            reasons.append(
                "Data or document storage file."
            )

            return score, reasons

        # -----------------------------------------------------
        # 5. Binary artifacts
        # -----------------------------------------------------

        if extension in self.BINARY_EXTENSIONS:

            score += 3

            reasons.append(
                "Binary or generated artifact."
            )

            return score, reasons

        # -----------------------------------------------------
        # 6. Database files outside infrastructure directories
        # -----------------------------------------------------

        if extension in self.DATABASE_EXTENSIONS:

            score += 5

            reasons.append(
                "Database storage file."
            )

            return score, reasons

        # -----------------------------------------------------
        # 7. Primary documentation
        # -----------------------------------------------------

        if self.is_primary_documentation(file_path):

            score += 48

            reasons.append(
                "Primary project documentation."
            )

        elif extension in self.DOCUMENTATION_EXTENSIONS:

            score += 15

            reasons.append(
                "Project documentation."
            )

        # -----------------------------------------------------
        # 8. Configuration / dependency files
        # -----------------------------------------------------

        if self.is_config_file(file_path):

            score += 30

            reasons.append(
                "Important project configuration or dependency file."
            )

        # -----------------------------------------------------
        # 9. Test / evaluation classification
        #
        # This is intentionally checked BEFORE AI keywords.
        # Therefore test_retrieval.py will not receive the
        # same AI/core bonus as rag.py.
        # -----------------------------------------------------

        is_test = self.is_test_file(file_path)

        if is_test:

            score += 20

            reasons.append(
                "Contains project tests or evaluation logic."
            )

            # Test files are deliberately prevented from
            # receiving the normal AI/core implementation bonus.

        # -----------------------------------------------------
        # 10. Application entry point
        # -----------------------------------------------------

        if name in self.ENTRY_POINT_FILES:

            score += 75

            reasons.append(
                "Likely application entry point."
            )

        # -----------------------------------------------------
        # 11. Core implementation files
        #
        # Only non-test source files receive this bonus.
        # -----------------------------------------------------

        if (
            not is_test
            and name in self.CORE_FILE_NAMES
        ):

            score += 45

            reasons.append(
                "Likely core application or AI logic."
            )

        # -----------------------------------------------------
        # 12. AI/core keyword detection
        #
        # Again, only for actual implementation files.
        # -----------------------------------------------------

        if (
            not is_test
            and self.is_source_file(file_path)
            and self.contains_core_keyword(file_path)
        ):

            score += 20

            reasons.append(
                "Contains core application or AI-related logic."
            )

        # -----------------------------------------------------
        # 13. General source code
        # -----------------------------------------------------

        if self.is_source_file(file_path):

            score += 15

            reasons.append(
                "Contains project source code."
            )

        # -----------------------------------------------------
        # 14. Important directories
        # -----------------------------------------------------

        parent_parts = {
            part.lower()
            for part in file_path.relative_to(
                self.repository_path
            ).parent.parts
        }

        important_directory_names = {
            "src",
            "app",
            "backend",
            "frontend",
            "api",
            "core",
            "services",
            "components",
        }

        if parent_parts.intersection(
            important_directory_names
        ):

            score += 5

            reasons.append(
                "Located in an important project directory."
            )

        # -----------------------------------------------------
        # 15. File size
        #
        # Size is only a small supporting signal.
        # It should never dominate architectural importance.
        # -----------------------------------------------------

        try:

            file_size = file_path.stat().st_size

            if file_size > 10 * 1024:

                score += 3

                reasons.append(
                    "Contains a relatively large amount of content."
                )

        except OSError:
            pass

        # -----------------------------------------------------
        # Final score
        # -----------------------------------------------------

        score = min(score, 100)

        return score, reasons

    # ---------------------------------------------------------
    # Importance level
    # ---------------------------------------------------------

    def get_importance_level(
        self,
        score: int,
    ) -> str:

        if score >= 75:
            return "critical"

        if score >= 45:
            return "high"

        if score >= 20:
            return "medium"

        return "low"

    # ---------------------------------------------------------
    # Analyze one file
    # ---------------------------------------------------------

    def analyze_file(
        self,
        file_path: Path,
    ) -> Dict:

        score, reasons = self.calculate_score(
            file_path
        )

        return {
            "file_path": self.get_relative_path(
                file_path
            ),
            "file_name": self.get_file_name(
                file_path
            ),
            "extension": self.get_extension(
                file_path
            ),
            "importance_score": score,
            "importance_level": self.get_importance_level(
                score
            ),
            "reasons": reasons,
        }

    # ---------------------------------------------------------
    # Analyze repository
    # ---------------------------------------------------------

    def analyze(
        self,
        top_k: int = 20,
    ) -> List[Dict]:

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        files = self.get_files()

        results = [
            self.analyze_file(file_path)
            for file_path in files
        ]

        # Sort primarily by importance score.
        # If scores are equal, use file path for
        # deterministic ordering.
        results.sort(
            key=lambda item: (
                -item["importance_score"],
                item["file_path"].lower(),
            )
        )

        return results[:top_k]