from pathlib import Path

from analysis.dependency_analyzer import DependencyAnalyzer


class ImpactAnalyzer:
    """
    Static impact analyzer for Python repositories.

    Determines:
    - Direct dependents
    - Indirect dependents
    - Direct dependencies
    - Indirect dependencies
    - Affected tests
    - Affected entry points
    - External dependencies
    - Impact level
    - Impact reasons

    Repository code is never executed.
    """

    def __init__(self, repository_id: str):
        if not repository_id:
            raise ValueError("repository_id is required")

        self.repository_id = repository_id

        self.dependency_analyzer = DependencyAnalyzer(
            repository_id
        )

    # ---------------------------------------------------------
    # Graph construction
    # ---------------------------------------------------------

    def _build_reverse_graph(self, internal_dependencies):
        """
        Build:

            target -> files that depend on target
        """

        reverse_graph = {}

        for dependency in internal_dependencies:

            source = dependency["from"]
            target = dependency["to"]

            reverse_graph.setdefault(
                target,
                set(),
            ).add(source)

        return reverse_graph

    def _build_forward_graph(self, internal_dependencies):
        """
        Build:

            source -> files that source depends on
        """

        forward_graph = {}

        for dependency in internal_dependencies:

            source = dependency["from"]
            target = dependency["to"]

            forward_graph.setdefault(
                source,
                set(),
            ).add(target)

        return forward_graph

    # ---------------------------------------------------------
    # Recursive dependency traversal
    # ---------------------------------------------------------

    def _find_all_dependents(
        self,
        file_path,
        reverse_graph,
    ):
        """
        Find every file that could be affected by
        a change to file_path.

        Includes both direct and indirect dependents.
        """

        visited = set()
        queue = list(
            reverse_graph.get(
                file_path,
                set(),
            )
        )

        direct = set(queue)

        while queue:

            current = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            for dependent in reverse_graph.get(
                current,
                set(),
            ):

                if dependent not in visited:
                    queue.append(dependent)

        indirect = visited - direct

        return (
            sorted(direct),
            sorted(indirect),
        )

    # ---------------------------------------------------------
    # Dependency traversal
    # ---------------------------------------------------------

    def _find_all_dependencies(
        self,
        file_path,
        forward_graph,
    ):
        """
        Find every file that the selected file depends on.
        """

        visited = set()

        queue = list(
            forward_graph.get(
                file_path,
                set(),
            )
        )

        direct = set(queue)

        while queue:

            current = queue.pop(0)

            if current in visited:
                continue

            visited.add(current)

            for dependency in forward_graph.get(
                current,
                set(),
            ):

                if dependency not in visited:
                    queue.append(
                        dependency
                    )

        indirect = visited - direct

        return (
            sorted(direct),
            sorted(indirect),
        )

    # ---------------------------------------------------------
    # File lookup
    # ---------------------------------------------------------

    def _find_file_info(
        self,
        files,
        file_path,
    ):
        for file_info in files:

            if file_info["path"] == file_path:
                return file_info

        return None

    # ---------------------------------------------------------
    # Impact classification
    # ---------------------------------------------------------

    def _classify_impact(
        self,
        target_file,
        direct_dependents,
        indirect_dependents,
        affected_tests,
        affected_entry_points,
    ):
        """
        Determine overall impact level.

        HIGH:
            - entry point is affected
            - many direct dependents
            - tests and application files are affected

        MEDIUM:
            - one or more dependents
            - or multiple indirect dependents

        LOW:
            - no dependents
        """

        reasons = []

        dependent_count = (
            len(direct_dependents)
            + len(indirect_dependents)
        )

        target_name = Path(
            target_file
        ).name.lower()

        # Entry-point target.
        if target_name in {
            "app.py",
            "main.py",
            "server.py",
            "run.py",
            "manage.py",
        }:
            reasons.append(
                "The file is an application entry point."
            )

        if affected_entry_points:
            reasons.append(
                "Changes can propagate to application entry points."
            )

        if len(direct_dependents) >= 3:
            reasons.append(
                "The file has three or more direct dependents."
            )

        elif len(direct_dependents) >= 1:
            reasons.append(
                "The file is directly used by other files."
            )

        if indirect_dependents:
            reasons.append(
                "The file has indirect dependents."
            )

        if affected_tests:
            reasons.append(
                "One or more test files may be affected."
            )

        # Classification.
        if (
            affected_entry_points
            or len(direct_dependents) >= 3
            or (
                affected_tests
                and direct_dependents
            )
        ):
            impact_level = "HIGH"

        elif dependent_count > 0:
            impact_level = "MEDIUM"

        else:
            impact_level = "LOW"

        if not reasons:
            reasons.append(
                "No dependent files were detected."
            )

        return {
            "impact_level": impact_level,
            "reasons": reasons,
        }

    # ---------------------------------------------------------
    # Analyze a specific file
    # ---------------------------------------------------------

    def analyze_file(
        self,
        file_path,
    ):
        if not file_path:
            raise ValueError(
                "file_path is required"
            )

        analysis = (
            self.dependency_analyzer.analyze()
        )

        files = analysis["file_dependency_summary"]

        target = self._find_file_info(
            analysis.get(
                "files",
                [],
            ),
            file_path,
        )

        if target is None:

            # DependencyAnalyzer doesn't expose
            # its raw file list in the returned
            # structure, so use the summary as
            # a fallback existence check.
            summary_target = next(
                (
                    item
                    for item in files
                    if item["file"] == file_path
                ),
                None,
            )

            if summary_target is None:
                raise FileNotFoundError(
                    f"File '{file_path}' was not found "
                    f"in repository '{self.repository_id}'"
                )

        internal_dependencies = analysis[
            "internal_dependencies"
        ]

        external_dependencies = analysis[
            "external_dependencies_by_file"
        ]

        reverse_graph = (
            self._build_reverse_graph(
                internal_dependencies
            )
        )

        forward_graph = (
            self._build_forward_graph(
                internal_dependencies
            )
        )

        direct_dependents, indirect_dependents = (
            self._find_all_dependents(
                file_path,
                reverse_graph,
            )
        )

        direct_dependencies, indirect_dependencies = (
            self._find_all_dependencies(
                file_path,
                forward_graph,
            )
        )

        # -----------------------------------------------------
        # Determine affected tests
        # -----------------------------------------------------

        all_affected = set(
            direct_dependents
            + indirect_dependents
        )

        affected_tests = sorted(
            file
            for file in all_affected
            if (
                Path(file).name.startswith(
                    "test_"
                )
                or Path(file).name.endswith(
                    "_test.py"
                )
                or "/tests/" in file.replace(
                    "\\",
                    "/",
                )
            )
        )

        # -----------------------------------------------------
        # Determine affected entry points
        # -----------------------------------------------------

        entry_point_names = {
            "app.py",
            "main.py",
            "server.py",
            "run.py",
            "manage.py",
            "wsgi.py",
            "asgi.py",
        }

        affected_entry_points = sorted(
            file
            for file in all_affected
            if Path(file).name.lower()
            in entry_point_names
        )

        # -----------------------------------------------------
        # External dependencies
        # -----------------------------------------------------

        external_dependencies_for_file = (
            external_dependencies.get(
                file_path,
                [],
            )
        )

        # -----------------------------------------------------
        # Target metadata
        # -----------------------------------------------------

        target_summary = next(
            (
                item
                for item in files
                if item["file"] == file_path
            ),
            None,
        )

        impact = self._classify_impact(
            file_path,
            direct_dependents,
            indirect_dependents,
            affected_tests,
            affected_entry_points,
        )

        return {
            "repository_id": self.repository_id,

            "file": file_path,

            "module": (
                target_summary["module"]
                if target_summary
                else None
            ),

            "impact": impact,

            "direct_dependencies": (
                direct_dependencies
            ),

            "indirect_dependencies": (
                indirect_dependencies
            ),

            "direct_dependents": (
                direct_dependents
            ),

            "indirect_dependents": (
                indirect_dependents
            ),

            "affected_tests": affected_tests,

            "affected_entry_points": (
                affected_entry_points
            ),

            "external_dependencies": (
                external_dependencies_for_file
            ),

            "direct_dependent_count": len(
                direct_dependents
            ),

            "indirect_dependent_count": len(
                indirect_dependents
            ),

            "total_affected_files": len(
                all_affected
            ),
        }

    # ---------------------------------------------------------
    # Main analysis
    # ---------------------------------------------------------

    def analyze(self, file_path=None):
        """
        Analyze one file.

        If file_path is omitted, return a repository-level
        impact summary.
        """

        if file_path:
            return self.analyze_file(
                file_path
            )

        analysis = (
            self.dependency_analyzer.analyze()
        )

        internal_dependencies = analysis[
            "internal_dependencies"
        ]

        reverse_graph = (
            self._build_reverse_graph(
                internal_dependencies
            )
        )

        file_paths = {
            item["file"]
            for item in analysis[
                "file_dependency_summary"
            ]
        }

        impact_summary = []

        for file_path in sorted(
            file_paths
        ):

            direct_dependents, indirect_dependents = (
                self._find_all_dependents(
                    file_path,
                    reverse_graph,
                )
            )

            affected = set(
                direct_dependents
                + indirect_dependents
            )

            entry_point_names = {
                "app.py",
                "main.py",
                "server.py",
                "run.py",
                "manage.py",
                "wsgi.py",
                "asgi.py",
            }

            affected_entry_points = [
                file
                for file in affected
                if Path(file).name.lower()
                in entry_point_names
            ]

            affected_tests = [
                file
                for file in affected
                if (
                    Path(file).name.startswith(
                        "test_"
                    )
                    or Path(file).name.endswith(
                        "_test.py"
                    )
                )
            ]

            classification = (
                self._classify_impact(
                    file_path,
                    direct_dependents,
                    indirect_dependents,
                    affected_tests,
                    affected_entry_points,
                )
            )

            impact_summary.append(
                {
                    "file": file_path,
                    "impact_level": classification[
                        "impact_level"
                    ],
                    "direct_dependent_count": len(
                        direct_dependents
                    ),
                    "indirect_dependent_count": len(
                        indirect_dependents
                    ),
                    "total_affected_files": len(
                        affected
                    ),
                    "affected_tests": sorted(
                        affected_tests
                    ),
                    "affected_entry_points": sorted(
                        affected_entry_points
                    ),
                    "reasons": classification[
                        "reasons"
                    ],
                }
            )

        impact_order = {
            "HIGH": 0,
            "MEDIUM": 1,
            "LOW": 2,
        }

        impact_summary.sort(
            key=lambda item: (
                impact_order.get(
                    item["impact_level"],
                    99,
                ),
                -item[
                    "total_affected_files"
                ],
                item["file"],
            )
        )

        return {
            "repository_id": self.repository_id,
            "total_files_analyzed": len(
                file_paths
            ),
            "files_with_impact": impact_summary,
        }