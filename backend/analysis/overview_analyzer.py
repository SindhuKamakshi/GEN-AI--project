from pathlib import Path
from collections import Counter


class RepositoryOverviewAnalyzer:
    """
    Deterministic repository analysis.

    This analyzer does not use an LLM.
    It inspects the repository structure and files directly.
    """

    LANGUAGE_MAP = {
        ".py": "Python",
        ".pyw": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript/React",
        ".ts": "TypeScript",
        ".tsx": "TypeScript/React",
        ".java": "Java",
        ".c": "C",
        ".cpp": "C++",
        ".cc": "C++",
        ".h": "C/C++ Header",
        ".hpp": "C++ Header",
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
        ".bat": "Batch",
        ".ps1": "PowerShell",
        ".xml": "XML",
        ".json": "JSON",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".toml": "TOML",
        ".ini": "INI",
        ".cfg": "Configuration",
        ".conf": "Configuration",
        ".md": "Markdown",
        ".rst": "reStructuredText",
        ".txt": "Text",
    }

    DOCUMENTATION_FILES = {
        "readme",
        "readme.md",
        "readme.txt",
        "readme.rst",
        "documentation.md",
        "docs.md",
        "changelog.md",
        "contributing.md",
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
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        ".env.example",
        "tsconfig.json",
        "vite.config.js",
        "vite.config.ts",
    }

    ENTRY_POINT_FILES = {
        "main.py",
        "app.py",
        "server.py",
        "run.py",
        "manage.py",
        "index.js",
        "server.js",
        "app.js",
        "main.js",
        "main.ts",
        "index.ts",
        "main.java",
        "application.java",
        "program.cs",
    }

    TEST_NAME_PATTERNS = (
        "test_",
        "tests_",
        "_test",
        "test.",
    )

    IGNORED_DIRECTORIES = {
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

    def __init__(self, repository_path: Path):
        self.repository_path = Path(repository_path)

        if not self.repository_path.exists():
            raise ValueError("Repository path does not exist.")

        if not self.repository_path.is_dir():
            raise ValueError("Repository path must be a directory.")

    # ------------------------------------------------------------
    # FILE DISCOVERY
    # ------------------------------------------------------------

    def get_files(self) -> list[Path]:
        """
        Return repository files while ignoring generated,
        dependency and version-control directories.
        """

        files = []

        for path in self.repository_path.rglob("*"):

            if not path.is_file():
                continue

            relative_parts = path.relative_to(
                self.repository_path
            ).parts

            if any(
                part.lower() in self.IGNORED_DIRECTORIES
                for part in relative_parts
            ):
                continue

            files.append(path)

        return sorted(files)

    # ------------------------------------------------------------
    # BASIC INFORMATION
    # ------------------------------------------------------------

    def calculate_total_size(self, files: list[Path]) -> int:
        """
        Calculate total repository file size in bytes.
        """

        total_size = 0

        for file_path in files:
            try:
                total_size += file_path.stat().st_size
            except OSError:
                continue

        return total_size

    # ------------------------------------------------------------
    # FILE TYPES
    # ------------------------------------------------------------

    def detect_file_types(
        self,
        files: list[Path],
    ) -> dict[str, int]:

        extensions = Counter()

        for file_path in files:
            extension = file_path.suffix.lower()

            if extension:
                extensions[extension] += 1
            else:
                extensions["[no extension]"] += 1

        return dict(
            sorted(
                extensions.items(),
                key=lambda item: (-item[1], item[0])
            )
        )

    # ------------------------------------------------------------
    # LANGUAGES
    # ------------------------------------------------------------

    def detect_languages(
        self,
        files: list[Path],
    ) -> dict[str, int]:

        languages = Counter()

        for file_path in files:

            extension = file_path.suffix.lower()

            language = self.LANGUAGE_MAP.get(extension)

            if language:
                languages[language] += 1

        return dict(
            sorted(
                languages.items(),
                key=lambda item: (-item[1], item[0])
            )
        )

    # ------------------------------------------------------------
    # DIRECTORY STRUCTURE
    # ------------------------------------------------------------

    def detect_directories(
        self,
        files: list[Path],
    ) -> list[str]:

        directories = set()

        for file_path in files:

            relative_path = file_path.relative_to(
                self.repository_path
            )

            parent = relative_path.parent

            if str(parent) != ".":
                directories.add(str(parent))

        return sorted(directories)

    # ------------------------------------------------------------
    # DOCUMENTATION
    # ------------------------------------------------------------

    def detect_documentation_files(
        self,
        files: list[Path],
    ) -> list[str]:

        documentation = []

        for file_path in files:

            filename = file_path.name.lower()

            if (
                filename in self.DOCUMENTATION_FILES
                or file_path.suffix.lower() in {
                    ".md",
                    ".rst",
                }
            ):
                documentation.append(
                    str(
                        file_path.relative_to(
                            self.repository_path
                        )
                    )
                )

        return sorted(documentation)

    # ------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------

    def detect_configuration_files(
        self,
        files: list[Path],
    ) -> list[str]:

        configuration = []

        for file_path in files:

            filename = file_path.name.lower()

            if filename in self.CONFIGURATION_FILES:

                configuration.append(
                    str(
                        file_path.relative_to(
                            self.repository_path
                        )
                    )
                )

        return sorted(configuration)

    # ------------------------------------------------------------
    # TEST FILES
    # ------------------------------------------------------------

    def detect_test_files(
        self,
        files: list[Path],
    ) -> list[str]:

        test_files = []

        for file_path in files:

            filename = file_path.name.lower()

            is_test = (
                filename.startswith("test_")
                or filename.endswith("_test.py")
                or filename.endswith(".test.js")
                or filename.endswith(".test.jsx")
                or filename.endswith(".test.ts")
                or filename.endswith(".test.tsx")
                or filename.endswith(".spec.js")
                or filename.endswith(".spec.ts")
                or "tests" in {
                    part.lower()
                    for part in file_path.relative_to(
                        self.repository_path
                    ).parts
                }
            )

            if is_test:
                test_files.append(
                    str(
                        file_path.relative_to(
                            self.repository_path
                        )
                    )
                )

        return sorted(test_files)

    # ------------------------------------------------------------
    # ENTRY POINT CANDIDATES
    # ------------------------------------------------------------

    def detect_entry_points(
        self,
        files: list[Path],
    ) -> list[str]:

        entry_points = []

        for file_path in files:

            filename = file_path.name.lower()

            if filename in self.ENTRY_POINT_FILES:

                entry_points.append(
                    str(
                        file_path.relative_to(
                            self.repository_path
                        )
                    )
                )

        return sorted(entry_points)

    # ------------------------------------------------------------
    # MAIN ANALYSIS
    # ------------------------------------------------------------

    def analyze(self) -> dict:

        files = self.get_files()

        total_size = self.calculate_total_size(files)

        return {
            "total_files": len(files),
            "total_size_bytes": total_size,
            "file_types": self.detect_file_types(files),
            "languages": self.detect_languages(files),
            "directories": self.detect_directories(files),
            "documentation_files": self.detect_documentation_files(
                files
            ),
            "configuration_files": self.detect_configuration_files(
                files
            ),
            "test_files": self.detect_test_files(files),
            "entry_point_candidates": self.detect_entry_points(
                files
            ),
        }