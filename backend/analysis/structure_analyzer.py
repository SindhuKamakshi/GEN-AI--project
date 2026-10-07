from pathlib import Path
from typing import Dict, List, Any


class RepositoryStructureAnalyzer:
    """
    Analyze the structural organization of a software repository.

    The analyzer identifies:
    - directory hierarchy
    - files
    - file counts
    - source files
    - test files
    - documentation
    - configuration
    - data/document directories
    - infrastructure directories
    - repository depth
    """

    # ---------------------------------------------------------
    # Directories ignored from structural analysis
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
    # These remain visible.
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
    # Data directories
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
    # Source extensions
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
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
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
    # Configuration
    # ---------------------------------------------------------

    CONFIGURATION_FILES = {
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
        ".env",
        ".env.example",
        ".env.template",
    }

    # ---------------------------------------------------------
    # Test patterns
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

    TEST_NAMES = {
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

        self.repository_path = Path(
            repository_path
        )

        if not self.repository_path.exists():
            raise ValueError(
                "Repository path does not exist."
            )

        if not self.repository_path.is_dir():
            raise ValueError(
                "Repository path must be a directory."
            )

    # ---------------------------------------------------------
    # Path filtering
    # ---------------------------------------------------------

    def should_ignore_path(
        self,
        path: Path,
    ) -> bool:

        relative_path = path.relative_to(
            self.repository_path
        )

        for part in relative_path.parts:

            if part.lower() in self.IGNORED_DIRECTORIES:
                return True

        return False

    # ---------------------------------------------------------
    # File discovery
    # ---------------------------------------------------------

    def get_files(self) -> List[Path]:

        files = []

        for path in self.repository_path.rglob("*"):

            if not path.is_file():
                continue

            if self.should_ignore_path(path):
                continue

            files.append(path)

        return sorted(
            files,
            key=lambda path: str(
                path.relative_to(
                    self.repository_path
                )
            ).lower(),
        )

    # ---------------------------------------------------------
    # Directory discovery
    # ---------------------------------------------------------

    def get_directories(self) -> List[Path]:

        directories = []

        for path in self.repository_path.rglob("*"):

            if not path.is_dir():
                continue

            if self.should_ignore_path(path):
                continue

            directories.append(path)

        return sorted(
            directories,
            key=lambda path: str(
                path.relative_to(
                    self.repository_path
                )
            ).lower(),
        )

    # ---------------------------------------------------------
    # Relative path
    # ---------------------------------------------------------

    def relative_path(
        self,
        path: Path,
    ) -> str:

        return str(
            path.relative_to(
                self.repository_path
            )
        )

    # ---------------------------------------------------------
    # File classification
    # ---------------------------------------------------------

    def is_source_file(
        self,
        path: Path,
    ) -> bool:

        return (
            path.suffix.lower()
            in self.SOURCE_EXTENSIONS
        )

    def is_documentation_file(
        self,
        path: Path,
    ) -> bool:

        return (
            path.suffix.lower()
            in self.DOCUMENTATION_EXTENSIONS
        )

    def is_configuration_file(
        self,
        path: Path,
    ) -> bool:

        return (
            path.name.lower()
            in self.CONFIGURATION_FILES
        )

    def is_test_file(
        self,
        path: Path,
    ) -> bool:

        name = path.name.lower()
        stem = path.stem.lower()

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

        if stem in self.TEST_NAMES:
            return True

        return False

    def is_data_directory(
        self,
        path: Path,
    ) -> bool:

        return (
            path.name.lower()
            in self.DATA_DIRECTORIES
        )

    def is_infrastructure_directory(
        self,
        path: Path,
    ) -> bool:

        return (
            path.name.lower()
            in self.INFRASTRUCTURE_DIRECTORIES
        )

    # ---------------------------------------------------------
    # File type summary
    # ---------------------------------------------------------

    def build_file_type_summary(
        self,
        files: List[Path],
    ) -> Dict[str, int]:

        summary: Dict[str, int] = {}

        for file_path in files:

            extension = file_path.suffix.lower()

            if not extension:
                extension = "[no extension]"

            summary[extension] = (
                summary.get(extension, 0) + 1
            )

        return dict(
            sorted(
                summary.items(),
                key=lambda item: (
                    -item[1],
                    item[0],
                ),
            )
        )

    # ---------------------------------------------------------
    # Directory tree
    # ---------------------------------------------------------

    def build_tree(
        self,
    ) -> Dict[str, Any]:
        """
        Build a nested directory/file representation.
        """

        root = {
            "name": self.repository_path.name,
            "type": "directory",
            "path": ".",
            "children": [],
        }

        nodes = {
            ".": root
        }

        directories = self.get_directories()

        # Create directory nodes first.
        for directory in directories:

            relative = directory.relative_to(
                self.repository_path
            )

            parts = relative.parts

            parent_key = "."

            for index in range(
                1,
                len(parts) + 1,
            ):

                current_parts = parts[:index]

                current_key = str(
                    Path(*current_parts)
                )

                if current_key not in nodes:

                    node = {
                        "name": parts[index - 1],
                        "type": "directory",
                        "path": current_key,
                        "children": [],
                    }

                    nodes[current_key] = node

                    nodes[parent_key][
                        "children"
                    ].append(node)

                parent_key = current_key

        # Add files.
        files = self.get_files()

        for file_path in files:

            relative = file_path.relative_to(
                self.repository_path
            )

            parts = relative.parts

            if len(parts) == 1:

                parent_key = "."

            else:

                parent_key = str(
                    Path(*parts[:-1])
                )

            file_node = {
                "name": file_path.name,
                "type": "file",
                "path": str(relative),
            }

            if parent_key in nodes:

                nodes[parent_key][
                    "children"
                ].append(file_node)

            else:

                # Defensive fallback.
                root["children"].append(
                    file_node
                )

        return root

    # ---------------------------------------------------------
    # Directory summaries
    # ---------------------------------------------------------

    def build_directory_summaries(
        self,
        files: List[Path],
        directories: List[Path],
    ) -> List[Dict[str, Any]]:

        summaries = []

        for directory in directories:

            relative_directory = (
                directory.relative_to(
                    self.repository_path
                )
            )

            directory_files = [
                file_path
                for file_path in files
                if directory
                in file_path.parents
            ]

            source_count = sum(
                1
                for file_path in directory_files
                if self.is_source_file(file_path)
            )

            test_count = sum(
                1
                for file_path in directory_files
                if self.is_test_file(file_path)
            )

            documentation_count = sum(
                1
                for file_path in directory_files
                if self.is_documentation_file(
                    file_path
                )
            )

            configuration_count = sum(
                1
                for file_path in directory_files
                if self.is_configuration_file(
                    file_path
                )
            )

            if self.is_infrastructure_directory(
                directory
            ):
                category = "infrastructure"

            elif self.is_data_directory(
                directory
            ):
                category = "data"

            elif test_count > 0:
                category = "testing"

            elif source_count > 0:
                category = "source"

            elif documentation_count > 0:
                category = "documentation"

            else:
                category = "other"

            summaries.append(
                {
                    "path": str(
                        relative_directory
                    ),
                    "name": directory.name,
                    "category": category,
                    "file_count": len(
                        directory_files
                    ),
                    "source_file_count": source_count,
                    "test_file_count": test_count,
                    "documentation_file_count": (
                        documentation_count
                    ),
                    "configuration_file_count": (
                        configuration_count
                    ),
                }
            )

        return summaries

    # ---------------------------------------------------------
    # Root-level files
    # ---------------------------------------------------------

    def get_root_files(
        self,
        files: List[Path],
    ) -> List[str]:

        root_files = []

        for file_path in files:

            relative = file_path.relative_to(
                self.repository_path
            )

            if len(relative.parts) == 1:

                root_files.append(
                    file_path.name
                )

        return sorted(
            root_files,
            key=str.lower,
        )

    # ---------------------------------------------------------
    # Category lists
    # ---------------------------------------------------------

    def get_category_files(
        self,
        files: List[Path],
    ) -> Dict[str, List[str]]:

        source_files = []
        test_files = []
        documentation_files = []
        configuration_files = []

        for file_path in files:

            relative = self.relative_path(
                file_path
            )

            if self.is_test_file(file_path):

                test_files.append(relative)

            elif self.is_source_file(file_path):

                source_files.append(relative)

            if self.is_documentation_file(
                file_path
            ):

                documentation_files.append(
                    relative
                )

            if self.is_configuration_file(
                file_path
            ):

                configuration_files.append(
                    relative
                )

        return {
            "source_files": sorted(
                source_files,
                key=str.lower,
            ),
            "test_files": sorted(
                test_files,
                key=str.lower,
            ),
            "documentation_files": sorted(
                documentation_files,
                key=str.lower,
            ),
            "configuration_files": sorted(
                configuration_files,
                key=str.lower,
            ),
        }

    # ---------------------------------------------------------
    # Maximum repository depth
    # ---------------------------------------------------------

    def calculate_max_depth(
        self,
        files: List[Path],
    ) -> int:

        if not files:
            return 0

        return max(
            len(
                file_path.relative_to(
                    self.repository_path
                ).parts
            )
            for file_path in files
        )

    # ---------------------------------------------------------
    # Analyze
    # ---------------------------------------------------------

    def analyze(self) -> Dict[str, Any]:

        files = self.get_files()
        directories = self.get_directories()

        categories = self.get_category_files(
            files
        )

        infrastructure_directories = [
            self.relative_path(directory)
            for directory in directories
            if self.is_infrastructure_directory(
                directory
            )
        ]

        data_directories = [
            self.relative_path(directory)
            for directory in directories
            if self.is_data_directory(directory)
        ]

        testing_directories = [
            self.relative_path(directory)
            for directory in directories
            if directory.name.lower()
            in {
                "test",
                "tests",
                "__tests__",
            }
        ]

        return {
            "repository_name": (
                self.repository_path.name
            ),
            "total_files": len(files),
            "total_directories": len(
                directories
            ),
            "maximum_depth": self.calculate_max_depth(
                files
            ),
            "root_files": self.get_root_files(
                files
            ),
            "file_types": self.build_file_type_summary(
                files
            ),
            "source_file_count": len(
                categories["source_files"]
            ),
            "test_file_count": len(
                categories["test_files"]
            ),
            "documentation_file_count": len(
                categories["documentation_files"]
            ),
            "configuration_file_count": len(
                categories["configuration_files"]
            ),
            "source_files": categories[
                "source_files"
            ],
            "test_files": categories[
                "test_files"
            ],
            "documentation_files": categories[
                "documentation_files"
            ],
            "configuration_files": categories[
                "configuration_files"
            ],
            "infrastructure_directories": sorted(
                infrastructure_directories,
                key=str.lower,
            ),
            "data_directories": sorted(
                data_directories,
                key=str.lower,
            ),
            "testing_directories": sorted(
                testing_directories,
                key=str.lower,
            ),
            "directory_summaries": (
                self.build_directory_summaries(
                    files,
                    directories,
                )
            ),
            "tree": self.build_tree(),
        }