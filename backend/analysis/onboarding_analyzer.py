from pathlib import Path
import inspect

from config import REPOSITORIES_DIR

from analysis.overview_analyzer import RepositoryOverviewAnalyzer
from analysis.tech_stack_detector import TechnologyStackDetector
from analysis.importance_analyzer import ImportantFileAnalyzer
from analysis.architecture_analyzer import ArchitectureAnalyzer
from analysis.dependency_analyzer import DependencyAnalyzer


class OnboardingAnalyzer:
    """
    Generates a developer onboarding guide from static
    repository analysis.

    Combines:
    - Repository overview
    - Technology stack
    - Important files
    - Architecture
    - Dependencies
    - Learning path

    This analyzer does not execute repository code.
    """

    def __init__(self, repository_id: str):

        if not repository_id:
            raise ValueError(
                "repository_id is required"
            )

        self.repository_id = repository_id

        # Convert repository ID into the actual repository path.
        self.repository_path = (
            REPOSITORIES_DIR / repository_id
        )

        if not self.repository_path.exists():
            raise ValueError(
                "Repository path does not exist."
            )

        if not self.repository_path.is_dir():
            raise ValueError(
                "Repository path must be a directory."
            )

        # -----------------------------------------------------
        # Initialize existing analyzers
        # -----------------------------------------------------

        self.overview_analyzer = self._create_analyzer(
            RepositoryOverviewAnalyzer
        )

        self.tech_stack_detector = self._create_analyzer(
            TechnologyStackDetector
        )

        self.importance_analyzer = self._create_analyzer(
            ImportantFileAnalyzer
        )

        self.architecture_analyzer = self._create_analyzer(
            ArchitectureAnalyzer
        )

        self.dependency_analyzer = self._create_analyzer(
            DependencyAnalyzer
        )

    # =========================================================
    # ANALYZER INITIALIZATION
    # =========================================================

    def _create_analyzer(self, analyzer_class):
        """
        Existing analyzers may accept either:

            repository_id

        or:

            repository_path

        Detect the constructor parameter and provide
        the appropriate value.
        """

        signature = inspect.signature(
            analyzer_class.__init__
        )

        parameters = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.name != "self"
        ]

        # Constructor takes no arguments.
        if not parameters:
            return analyzer_class()

        parameter_name = parameters[0].name.lower()

        # Example:
        # repository_path
        if "path" in parameter_name:
            return analyzer_class(
                self.repository_path
            )

        # Example:
        # repository_id
        if "id" in parameter_name:
            return analyzer_class(
                self.repository_id
            )

        # Safe fallback.
        return analyzer_class(
            self.repository_path
        )

    # =========================================================
    # UTILITY METHODS
    # =========================================================

    @staticmethod
    def _is_test_file(file_path):
        """
        Determine whether a file is a test file.
        """

        normalized_path = str(
            file_path
        ).replace("\\", "/")

        file_name = Path(
            file_path
        ).name.lower()

        return (
            file_name.startswith("test_")
            or file_name.endswith("_test.py")
            or "/tests/" in normalized_path
        )

    @staticmethod
    def _is_evaluation_file(file_path):
        """
        Determine whether a file is an evaluation,
        benchmark, or metrics file.
        """

        file_name = Path(
            file_path
        ).name.lower()

        return (
            file_name.startswith("evaluate")
            or file_name.startswith("evaluation")
            or "benchmark" in file_name
            or "metrics" in file_name
        )

    # =========================================================
    # IMPORTANT FILES
    # =========================================================

    def _build_important_files(
        self,
        importance
    ):
        """
        Convert the existing ImportantFileAnalyzer output
        into the onboarding format.

        Supports both:

            [
                {...},
                {...}
            ]

        and:

            {
                "important_files": [...]
            }
        """

        if isinstance(
            importance,
            list
        ):

            important_list = importance

        elif isinstance(
            importance,
            dict
        ):

            important_list = importance.get(
                "important_files",
                []
            )

        else:

            important_list = []

        important_files = []

        for file_info in important_list:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            level = file_info.get(
                "importance_level"
            )

            if level not in {
                "critical",
                "high"
            }:
                continue

            file_path = file_info.get(
                "file_path"
            )

            if file_path is None:
                file_path = file_info.get(
                    "file"
                )

            score = file_info.get(
                "importance_score"
            )

            if score is None:
                score = file_info.get(
                    "score",
                    0
                )

            reasons = file_info.get(
                "reasons",
                []
            )

            important_files.append(
                {
                    "file": file_path,
                    "importance_level": level,
                    "score": score,
                    "reasons": reasons,
                }
            )

        important_files.sort(
            key=lambda item: (
                -item["score"],
                item["file"] or ""
            )
        )

        return important_files

    # =========================================================
    # ARCHITECTURE FILES
    # =========================================================

    @staticmethod
    def _get_architecture_files(
        architecture
    ):
        """
        Extract the file list from architecture analysis.

        Supports both dictionary and list formats.
        """

        if isinstance(
            architecture,
            dict
        ):

            return architecture.get(
                "files",
                []
            )

        if isinstance(
            architecture,
            list
        ):

            return architecture

        return []

    # =========================================================
    # ENTRY POINTS
    # =========================================================

    def _find_entry_points(
        self,
        architecture
    ):

        files = self._get_architecture_files(
            architecture
        )

        entry_points = []

        for file_info in files:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            if file_info.get(
                "entry_point"
            ):

                file_path = file_info.get(
                    "path"
                )

                if file_path:
                    entry_points.append(
                        file_path
                    )

        return sorted(
            set(entry_points)
        )

    # =========================================================
    # CORE FILES
    # =========================================================

    def _identify_core_files(
        self,
        architecture,
        dependency_result
    ):
        """
        Identify files that are central to understanding
        the repository.
        """

        core_files = []

        architecture_files = (
            self._get_architecture_files(
                architecture
            )
        )

        # -----------------------------------------------------
        # Entry points
        # -----------------------------------------------------

        for file_info in architecture_files:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            if file_info.get(
                "entry_point"
            ):

                file_path = file_info.get(
                    "path"
                )

                if file_path:

                    core_files.append(
                        {
                            "file": file_path,
                            "role": "entry_point",
                            "reason": (
                                "Application entry point."
                            ),
                        }
                    )

        # -----------------------------------------------------
        # Service layer
        # -----------------------------------------------------

        for file_info in architecture_files:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            if file_info.get(
                "layer"
            ) != "service":
                continue

            file_path = file_info.get(
                "path"
            )

            if not file_path:
                continue

            if any(
                item["file"] == file_path
                for item in core_files
            ):
                continue

            core_files.append(
                {
                    "file": file_path,
                    "role": "core_service",
                    "reason": (
                        "Core service or processing "
                        "module."
                    ),
                }
            )

        # -----------------------------------------------------
        # Dependency hotspots
        # -----------------------------------------------------

        if isinstance(
            dependency_result,
            dict
        ):

            hotspots = dependency_result.get(
                "dependency_hotspots",
                []
            )

        else:

            hotspots = []

        for hotspot in hotspots:

            if not isinstance(
                hotspot,
                dict
            ):
                continue

            dependent_count = hotspot.get(
                "dependent_count",
                0
            )

            if dependent_count < 2:
                continue

            file_path = hotspot.get(
                "file"
            )

            if not file_path:
                continue

            if any(
                item["file"] == file_path
                for item in core_files
            ):
                continue

            core_files.append(
                {
                    "file": file_path,
                    "role": "dependency_hotspot",
                    "reason": (
                        f"Used by "
                        f"{dependent_count} "
                        f"other files."
                    ),
                }
            )

        return core_files

    # =========================================================
    # ARCHITECTURE SUMMARY
    # =========================================================

    def _build_architecture_summary(
        self,
        architecture
    ):

        if not isinstance(
            architecture,
            dict
        ):

            return {
                "architecture_style": None,
                "entry_points": [],
                "layers": {},
                "frameworks": [],
                "ai_ml_technologies": [],
                "llm_technologies": [],
                "databases": [],
            }

        summary = architecture.get(
            "architecture_summary",
            {}
        )

        if not isinstance(
            summary,
            dict
        ):
            summary = {}

        return {
            "architecture_style": (
                summary.get(
                    "architecture_style"
                )
            ),
            "entry_points": (
                summary.get(
                    "entry_points",
                    []
                )
            ),
            "layers": (
                summary.get(
                    "layers_detected",
                    {}
                )
            ),
            "frameworks": (
                summary.get(
                    "frameworks",
                    []
                )
            ),
            "ai_ml_technologies": (
                summary.get(
                    "ai_ml_technologies",
                    []
                )
            ),
            "llm_technologies": (
                summary.get(
                    "llm_technologies",
                    []
                )
            ),
            "databases": (
                summary.get(
                    "databases",
                    []
                )
            ),
        }

    # =========================================================
    # TECHNOLOGY SUMMARY
    # =========================================================

    def _build_technology_summary(
        self,
        tech_stack
    ):

        if not isinstance(
            tech_stack,
            dict
        ):

            return {
                "languages": [],
                "frameworks": [],
                "ai_ml": [],
                "vector_databases": [],
                "databases": [],
                "data_science": [],
                "libraries": [],
                "testing": [],
                "build_tools": [],
                "package_managers": [],
            }

        return {
            "languages": (
                tech_stack.get(
                    "languages",
                    []
                )
            ),
            "frameworks": (
                tech_stack.get(
                    "frameworks",
                    []
                )
            ),
            "ai_ml": (
                tech_stack.get(
                    "ai_ml",
                    []
                )
            ),
            "vector_databases": (
                tech_stack.get(
                    "vector_databases",
                    []
                )
            ),
            "databases": (
                tech_stack.get(
                    "databases",
                    []
                )
            ),
            "data_science": (
                tech_stack.get(
                    "data_science",
                    []
                )
            ),
            "libraries": (
                tech_stack.get(
                    "libraries",
                    []
                )
            ),
            "testing": (
                tech_stack.get(
                    "testing",
                    []
                )
            ),
            "build_tools": (
                tech_stack.get(
                    "build_tools",
                    []
                )
            ),
            "package_managers": (
                tech_stack.get(
                    "package_managers",
                    []
                )
            ),
        }

    # =========================================================
    # LEARNING PATH
    # =========================================================

    def _build_learning_path(
        self,
        architecture,
        importance
    ):
        """
        Build a recommended order for a new developer
        to understand the repository.
        """

        steps = []

        architecture_files = (
            self._get_architecture_files(
                architecture
            )
        )

        # -----------------------------------------------------
        # Step 1: Entry points
        # -----------------------------------------------------

        entry_points = (
            self._find_entry_points(
                architecture
            )
        )

        for file_path in entry_points:

            steps.append(
                {
                    "step": len(steps) + 1,
                    "file": file_path,
                    "stage": "entry_point",
                    "reason": (
                        "Start here to understand "
                        "how the application begins."
                    ),
                }
            )

        # -----------------------------------------------------
        # Step 2: Core service files
        # -----------------------------------------------------

        service_files = []

        for file_info in architecture_files:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            if file_info.get(
                "layer"
            ) == "service":

                file_path = file_info.get(
                    "path"
                )

                if file_path:
                    service_files.append(
                        file_path
                    )

        for file_path in sorted(
            set(service_files)
        ):

            if any(
                step["file"] == file_path
                for step in steps
            ):
                continue

            steps.append(
                {
                    "step": len(steps) + 1,
                    "file": file_path,
                    "stage": "core_logic",
                    "reason": (
                        "Understand the core "
                        "application logic."
                    ),
                }
            )

        # -----------------------------------------------------
        # Step 3: Important files
        # -----------------------------------------------------

        important_files = (
            self._build_important_files(
                importance
            )
        )

        for item in important_files:

            file_path = item.get(
                "file"
            )

            if not file_path:
                continue

            if any(
                step["file"] == file_path
                for step in steps
            ):
                continue

            steps.append(
                {
                    "step": len(steps) + 1,
                    "file": file_path,
                    "stage": "important_file",
                    "reason": (
                        "Important repository file "
                        "identified by static analysis."
                    ),
                }
            )

        # -----------------------------------------------------
        # Step 4: Tests
        # -----------------------------------------------------

        test_files = []

        for file_info in architecture_files:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            file_path = file_info.get(
                "path"
            )

            if (
                file_path
                and self._is_test_file(
                    file_path
                )
            ):

                test_files.append(
                    file_path
                )

        for file_path in sorted(
            set(test_files)
        ):

            if any(
                step["file"] == file_path
                for step in steps
            ):
                continue

            steps.append(
                {
                    "step": len(steps) + 1,
                    "file": file_path,
                    "stage": "testing",
                    "reason": (
                        "Use the tests to understand "
                        "expected behavior."
                    ),
                }
            )

        # -----------------------------------------------------
        # Step 5: Evaluation files
        # -----------------------------------------------------

        evaluation_files = []

        for file_info in architecture_files:

            if not isinstance(
                file_info,
                dict
            ):
                continue

            file_path = file_info.get(
                "path"
            )

            if (
                file_path
                and self._is_evaluation_file(
                    file_path
                )
            ):

                evaluation_files.append(
                    file_path
                )

        for file_path in sorted(
            set(evaluation_files)
        ):

            if any(
                step["file"] == file_path
                for step in steps
            ):
                continue

            steps.append(
                {
                    "step": len(steps) + 1,
                    "file": file_path,
                    "stage": "evaluation",
                    "reason": (
                        "Review evaluation or "
                        "benchmark logic after "
                        "understanding the main system."
                    ),
                }
            )

        return steps

    # =========================================================
    # KEY CONCEPTS
    # =========================================================

    def _build_key_concepts(
        self,
        technology_summary,
        architecture_summary
    ):

        concepts = []

        # Languages
        for language in technology_summary.get(
            "languages",
            []
        ):

            concepts.append(
                {
                    "concept": language,
                    "category": "language",
                }
            )

        # Frameworks
        for framework in technology_summary.get(
            "frameworks",
            []
        ):

            concepts.append(
                {
                    "concept": framework,
                    "category": "framework",
                }
            )

        # AI/ML
        for technology in technology_summary.get(
            "ai_ml",
            []
        ):

            concepts.append(
                {
                    "concept": technology,
                    "category": "ai_ml",
                }
            )

        # Vector databases
        for database in technology_summary.get(
            "vector_databases",
            []
        ):

            concepts.append(
                {
                    "concept": database,
                    "category": "vector_database",
                }
            )

        # Databases
        for database in technology_summary.get(
            "databases",
            []
        ):

            concepts.append(
                {
                    "concept": database,
                    "category": "database",
                }
            )

        # LLM technologies
        for technology in architecture_summary.get(
            "llm_technologies",
            []
        ):

            concepts.append(
                {
                    "concept": technology,
                    "category": "llm",
                }
            )

        # Remove duplicates
        unique = {}

        for concept in concepts:

            key = (
                concept["category"],
                concept["concept"]
            )

            unique[key] = concept

        return list(
            unique.values()
        )

    # =========================================================
    # REPOSITORY GUIDE
    # =========================================================

    def _build_repository_guide(
        self,
        overview,
        architecture_summary,
        learning_path
    ):

        if not isinstance(
            overview,
            dict
        ):
            overview = {}

        repository_name = overview.get(
            "repository_name",
            self.repository_id
        )

        total_files = overview.get(
            "total_files",
            0
        )

        return {
            "repository_name": repository_name,
            "total_files": total_files,
            "architecture_style": (
                architecture_summary.get(
                    "architecture_style"
                )
            ),
            "start_with": (
                learning_path[0]["file"]
                if learning_path
                else None
            ),
            "recommended_first_files": [
                step["file"]
                for step in learning_path[:5]
            ],
        }

    # =========================================================
    # MAIN ANALYSIS
    # =========================================================

    def analyze(self):

        # -----------------------------------------------------
        # Run existing analyzers
        # -----------------------------------------------------

        overview = (
            self.overview_analyzer.analyze()
        )

        tech_stack = (
            self.tech_stack_detector.detect()
        )

        importance = (
            self.importance_analyzer.analyze()
        )

        architecture = (
            self.architecture_analyzer.analyze()
        )

        dependency_result = (
            self.dependency_analyzer.analyze()
        )

        # -----------------------------------------------------
        # Build onboarding sections
        # -----------------------------------------------------

        architecture_summary = (
            self._build_architecture_summary(
                architecture
            )
        )

        technology_summary = (
            self._build_technology_summary(
                tech_stack
            )
        )

        important_files = (
            self._build_important_files(
                importance
            )
        )

        core_files = (
            self._identify_core_files(
                architecture,
                dependency_result
            )
        )

        learning_path = (
            self._build_learning_path(
                architecture,
                importance
            )
        )

        key_concepts = (
            self._build_key_concepts(
                technology_summary,
                architecture_summary
            )
        )

        repository_guide = (
            self._build_repository_guide(
                overview,
                architecture_summary,
                learning_path
            )
        )

        # -----------------------------------------------------
        # Dependency summary
        # -----------------------------------------------------

        if isinstance(
            dependency_result,
            dict
        ):

            dependency_summary = {
                "total_internal_dependencies": (
                    dependency_result.get(
                        "total_internal_dependencies",
                        0
                    )
                ),
                "total_external_dependencies": (
                    dependency_result.get(
                        "total_external_dependencies",
                        0
                    )
                ),
                "dependency_hotspots": (
                    dependency_result.get(
                        "dependency_hotspots",
                        []
                    )
                ),
            }

        else:

            dependency_summary = {
                "total_internal_dependencies": 0,
                "total_external_dependencies": 0,
                "dependency_hotspots": [],
            }

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        return {
            "repository_id": (
                self.repository_id
            ),

            "repository_guide": (
                repository_guide
            ),

            "technology_summary": (
                technology_summary
            ),

            "architecture_summary": (
                architecture_summary
            ),

            "important_files": (
                important_files
            ),

            "core_files": (
                core_files
            ),

            "key_concepts": (
                key_concepts
            ),

            "learning_path": (
                learning_path
            ),

            "dependency_summary": (
                dependency_summary
            ),
        }