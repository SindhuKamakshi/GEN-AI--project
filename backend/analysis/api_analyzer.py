import ast
from pathlib import Path
from collections import Counter

from config import REPOSITORIES_DIR


class APIAnalyzer:
    """
    Discovers REST APIs and application interfaces inside a repository.

    Currently supports:
    - FastAPI
    - Flask
    - Streamlit

    The analyzer uses Python AST instead of executing repository code.
    """

    DEV_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "coverage",
        ".idea",
        ".vscode",
    }

    PYTHON_EXTENSIONS = {".py"}

    HTTP_METHODS = {
        "get",
        "post",
        "put",
        "patch",
        "delete",
        "head",
        "options",
    }

    STREAMLIT_COMPONENTS = {
        # Input components
        "button": "button",
        "file_uploader": "file_upload",
        "text_input": "text_input",
        "text_area": "text_area",
        "chat_input": "chat_input",
        "selectbox": "selectbox",
        "multiselect": "multiselect",
        "radio": "radio",
        "checkbox": "checkbox",
        "slider": "slider",
        "select_slider": "select_slider",
        "number_input": "number_input",
        "date_input": "date_input",
        "time_input": "time_input",
        "color_picker": "color_picker",

        # Forms
        "form": "form",
        "form_submit_button": "form_submit_button",

        # Output / interaction
        "download_button": "download_button",
        "dataframe": "dataframe",
        "table": "table",
        "write": "output",
        "markdown": "output",
        "text": "output",
        "title": "output",
        "header": "output",
        "subheader": "output",
        "success": "status_output",
        "error": "status_output",
        "warning": "status_output",
        "info": "status_output",

        # Layout/navigation
        "sidebar": "sidebar",
        "tabs": "tabs",
        "columns": "layout",
        "expander": "layout",
        "container": "layout",
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
        """
        Yield Python files while skipping development/build directories.
        """

        for path in self.repository_path.rglob("*.py"):
            relative_parts = path.relative_to(self.repository_path).parts

            if any(part in self.DEV_DIRS for part in relative_parts):
                continue

            yield path

    @staticmethod
    def _get_call_name(node):
        """
        Extract a callable name from an AST Call node.

        Examples:
            st.button(...) -> st.button
            app.get(...)   -> app.get
            button(...)    -> button
        """

        if not isinstance(node, ast.Call):
            return None

        func = node.func

        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name):
                return f"{func.value.id}.{func.attr}"

            if isinstance(func.value, ast.Attribute):
                parts = []

                current = func

                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value

                if isinstance(current, ast.Name):
                    parts.append(current.id)

                return ".".join(reversed(parts))

        elif isinstance(func, ast.Name):
            return func.id

        return None

    @staticmethod
    def _get_function_context(tree, line_number):
        """
        Return the function/method containing the given line.
        """

        context = None

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)

                if start is None or end is None:
                    continue

                if start <= line_number <= end:
                    context = node.name

        return context

    @staticmethod
    def _detect_frameworks(tree):
        """
        Detect frameworks from imports.
        """

        frameworks = set()

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "streamlit":
                        frameworks.add("Streamlit")

                    elif alias.name == "fastapi":
                        frameworks.add("FastAPI")

                    elif alias.name == "flask":
                        frameworks.add("Flask")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""

                if module == "streamlit" or module.startswith("streamlit."):
                    frameworks.add("Streamlit")

                elif module == "fastapi" or module.startswith("fastapi."):
                    frameworks.add("FastAPI")

                elif module == "flask" or module.startswith("flask."):
                    frameworks.add("Flask")

        return sorted(frameworks)

    @staticmethod
    def _detect_streamlit_aliases(tree):
        """
        Detect aliases used for the streamlit package.

        Supports:

            import streamlit as st
            import streamlit
        """

        aliases = {"st"}

        for node in ast.walk(tree):

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "streamlit":
                        aliases.add(alias.asname or "streamlit")

        return aliases

    @staticmethod
    def _extract_literal_argument(call_node, index=0):
        """
        Safely extract a simple literal argument.

        This is intentionally conservative and never executes code.
        """

        if len(call_node.args) <= index:
            return None

        argument = call_node.args[index]

        if isinstance(argument, ast.Constant):
            value = argument.value

            if isinstance(value, (str, int, float, bool)):
                return value

        return None

    def _discover_rest_endpoint(self, node, tree, file_path):
        """
        Detect FastAPI/Flask style route decorators.
        """

        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None

        for decorator in node.decorator_list:

            if not isinstance(decorator, ast.Call):
                continue

            if not isinstance(decorator.func, ast.Attribute):
                continue

            method = decorator.func.attr.lower()

            if method not in self.HTTP_METHODS:
                continue

            framework = None

            if isinstance(decorator.func.value, ast.Name):
                object_name = decorator.func.value.id.lower()

                if object_name in {
                    "app",
                    "api",
                    "router",
                    "application",
                    "blueprint",
                }:
                    framework = "FastAPI/Flask"

            if framework is None:
                continue

            path = self._extract_literal_argument(decorator, 0)

            if path is None:
                path = "/"

            relative_path = str(
                file_path.relative_to(self.repository_path)
            ).replace("\\", "/")

            return {
                "interface_type": "rest_endpoint",
                "method": method.upper(),
                "path": path,
                "file": relative_path,
                "line": node.lineno,
                "function": node.name,
                "framework": framework,
            }

        return None

    def _discover_streamlit_component(
        self,
        node,
        tree,
        file_path,
        streamlit_aliases,
    ):
        """
        Detect Streamlit UI/application components.
        """

        if not isinstance(node, ast.Call):
            return None

        if not isinstance(node.func, ast.Attribute):
            return None

        object_node = node.func.value

        if not isinstance(object_node, ast.Name):
            return None

        alias = object_node.id

        if alias not in streamlit_aliases:
            return None

        component = node.func.attr

        if component not in self.STREAMLIT_COMPONENTS:
            return None

        relative_path = str(
            file_path.relative_to(self.repository_path)
        ).replace("\\", "/")

        component_type = self.STREAMLIT_COMPONENTS[component]

        label = self._extract_literal_argument(node, 0)

        return {
            "interface_type": "application_component",
            "framework": "Streamlit",
            "component": component,
            "component_type": component_type,
            "label": label,
            "file": relative_path,
            "line": node.lineno,
            "function": self._get_function_context(
                tree,
                node.lineno,
            ),
        }

    def _analyze_file(self, file_path):
        """
        Analyze a single Python file.
        """

        result = {
            "endpoints": [],
            "application_interfaces": [],
            "frameworks": [],
        }

        try:
            source = file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

            tree = ast.parse(source)

        except (SyntaxError, UnicodeDecodeError, OSError):
            return result

        result["frameworks"] = self._detect_frameworks(tree)

        streamlit_aliases = self._detect_streamlit_aliases(tree)

        for node in ast.walk(tree):

            # REST API discovery
            endpoint = self._discover_rest_endpoint(
                node,
                tree,
                file_path,
            )

            if endpoint:
                result["endpoints"].append(endpoint)

            # Streamlit application interface discovery
            interface = self._discover_streamlit_component(
                node,
                tree,
                file_path,
                streamlit_aliases,
            )

            if interface:
                result["application_interfaces"].append(interface)

        return result

    def analyze(self):
        """
        Analyze the complete repository.
        """

        self._validate_repository()

        python_files = list(self._iter_python_files())

        endpoints = []
        application_interfaces = []

        frameworks = Counter()

        analyzed_python_files = 0
        failed_python_files = 0

        files_with_endpoints = set()
        files_with_interfaces = set()

        for file_path in python_files:

            result = self._analyze_file(file_path)

            analyzed_python_files += 1

            if result["frameworks"]:
                frameworks.update(result["frameworks"])

            if result["endpoints"]:
                endpoints.extend(result["endpoints"])

                for endpoint in result["endpoints"]:
                    files_with_endpoints.add(endpoint["file"])

            if result["application_interfaces"]:
                application_interfaces.extend(
                    result["application_interfaces"]
                )

                for interface in result["application_interfaces"]:
                    files_with_interfaces.add(interface["file"])

        http_methods = Counter(
            endpoint["method"]
            for endpoint in endpoints
        )

        interface_types = Counter(
            interface["component_type"]
            for interface in application_interfaces
        )

        interface_frameworks = Counter(
            interface["framework"]
            for interface in application_interfaces
        )

        return {
            "repository_id": self.repository_id,

            "total_python_files": len(python_files),

            "analyzed_python_files": analyzed_python_files,

            "failed_python_files": failed_python_files,

            # REST API information
            "total_endpoints": len(endpoints),

            "frameworks": dict(frameworks),

            "http_methods": dict(http_methods),

            "files_with_apis": sorted(files_with_endpoints),

            "endpoints": endpoints,

            # Application interface information
            "total_application_interfaces": len(
                application_interfaces
            ),

            "application_interface_frameworks": dict(
                interface_frameworks
            ),

            "application_interface_types": dict(
                interface_types
            ),

            "files_with_application_interfaces": sorted(
                files_with_interfaces
            ),

            "application_interfaces": application_interfaces,
        }