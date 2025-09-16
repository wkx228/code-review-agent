# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Code analysis tool for code review."""

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class CodeAnalysisTool(Tool):
    """Tool for analyzing Python code structure and generating compatibility reports."""

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)

    @override
    def get_name(self) -> str:
        return "code_analysis_tool"

    @override
    def get_description(self) -> str:
        return """Analyze Python code structure and generate compatibility reports.
        
        This tool can:
        - Parse Python AST and analyze code structure
        - Extract public APIs and interfaces
        - Analyze dependency relationships
        - Generate compatibility reports
        - Compare code structures between versions
        """

    @override
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="analysis_type",
                type="string",
                description="Type of analysis to perform",
                enum=["api_extraction", "structure_analysis", "dependency_analysis", "compatibility_report"],
                required=True,
            ),
            ToolParameter(
                name="source_code",
                type="string",
                description="Source code content to analyze",
                required=False,
            ),
            ToolParameter(
                name="target_code",
                type="string", 
                description="Target code content for comparison",
                required=False,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file being analyzed",
                required=False,
            ),
            ToolParameter(
                name="include_private",
                type="boolean",
                description="Whether to include private members in analysis",
                required=False,
            ),
            ToolParameter(
                name="output_format",
                type="string",
                description="Output format for the analysis",
                enum=["json", "markdown", "text"],
                required=False,
            ),
            ToolParameter(
                name="analysis_scope",
                type="array",
                description="Scope of analysis",
                items={"type": "string", "enum": ["functions", "classes", "imports", "variables", "decorators"]},
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            analysis_type = arguments["analysis_type"]
            source_code = arguments.get("source_code")
            target_code = arguments.get("target_code")
            file_path = arguments.get("file_path", "unknown.py")
            include_private = arguments.get("include_private", False)
            output_format = arguments.get("output_format", "text")
            analysis_scope = arguments.get("analysis_scope", ["functions", "classes", "imports"])

            if analysis_type == "api_extraction":
                if not source_code:
                    return ToolExecResult(
                        error="source_code is required for api_extraction",
                        error_code=1,
                    )
                result = self._extract_public_api(source_code, file_path, include_private, analysis_scope)
                
            elif analysis_type == "structure_analysis":
                if not source_code:
                    return ToolExecResult(
                        error="source_code is required for structure_analysis",
                        error_code=1,
                    )
                result = self._analyze_code_structure(source_code, file_path, analysis_scope)
                
            elif analysis_type == "dependency_analysis":
                if not source_code:
                    return ToolExecResult(
                        error="source_code is required for dependency_analysis",
                        error_code=1,
                    )
                result = self._analyze_dependencies(source_code, file_path)
                
            elif analysis_type == "compatibility_report":
                if not source_code or not target_code:
                    return ToolExecResult(
                        error="Both source_code and target_code are required for compatibility_report",
                        error_code=1,
                    )
                result = self._generate_compatibility_report(
                    source_code, target_code, file_path, include_private
                )
                
            else:
                return ToolExecResult(
                    error=f"Unknown analysis type: {analysis_type}",
                    error_code=1,
                )

            # Format output
            formatted_result = self._format_output(result, output_format)
            return ToolExecResult(output=formatted_result)

        except Exception as e:
            return ToolExecResult(
                error=f"Error during code analysis: {str(e)}",
                error_code=1,
            )

    def _extract_public_api(
        self, 
        code: str, 
        file_path: str, 
        include_private: bool,
        analysis_scope: List[str]
    ) -> Dict[str, Any]:
        """Extract public API elements from code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}

        api_elements = {
            "file_path": file_path,
            "functions": {},
            "classes": {},
            "imports": [],
            "variables": [],
            "constants": [],
            "decorators": []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and "functions" in analysis_scope:
                if include_private or not node.name.startswith('_'):
                    api_elements["functions"][node.name] = self._extract_function_info(node)
            
            elif isinstance(node, ast.ClassDef) and "classes" in analysis_scope:
                if include_private or not node.name.startswith('_'):
                    api_elements["classes"][node.name] = self._extract_class_info(node, include_private)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)) and "imports" in analysis_scope:
                api_elements["imports"].append(self._extract_import_info(node))
            
            elif isinstance(node, ast.Assign) and "variables" in analysis_scope:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if include_private or not target.id.startswith('_'):
                            var_info = {
                                "name": target.id,
                                "type": self._infer_type(node.value),
                                "line": node.lineno
                            }
                            if target.id.isupper():
                                api_elements["constants"].append(var_info)
                            else:
                                api_elements["variables"].append(var_info)

        return api_elements

    def _extract_function_info(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Extract detailed function information."""
        info = {
            "name": node.name,
            "line": node.lineno,
            "arguments": [],
            "defaults": [],
            "decorators": [],
            "docstring": None,
            "return_annotation": None,
            "is_async": isinstance(node, ast.AsyncFunctionDef)
        }

        # Extract arguments
        for arg in node.args.args:
            arg_info = {"name": arg.arg}
            if arg.annotation:
                arg_info["annotation"] = ast.unparse(arg.annotation)
            info["arguments"].append(arg_info)

        # Extract defaults
        for default in node.args.defaults:
            info["defaults"].append(ast.unparse(default))

        # Extract decorators
        for decorator in node.decorator_list:
            info["decorators"].append(ast.unparse(decorator))

        # Extract docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            info["docstring"] = node.body[0].value.value

        # Extract return annotation
        if node.returns:
            info["return_annotation"] = ast.unparse(node.returns)

        return info

    def _extract_class_info(self, node: ast.ClassDef, include_private: bool) -> Dict[str, Any]:
        """Extract detailed class information."""
        info = {
            "name": node.name,
            "line": node.lineno,
            "base_classes": [],
            "decorators": [],
            "methods": {},
            "attributes": [],
            "docstring": None
        }

        # Extract base classes
        for base in node.bases:
            info["base_classes"].append(ast.unparse(base))

        # Extract decorators
        for decorator in node.decorator_list:
            info["decorators"].append(ast.unparse(decorator))

        # Extract methods and attributes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if include_private or not item.name.startswith('_'):
                    info["methods"][item.name] = self._extract_function_info(item)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        if include_private or not target.id.startswith('_'):
                            info["attributes"].append({
                                "name": target.id,
                                "type": self._infer_type(item.value),
                                "line": item.lineno
                            })

        # Extract docstring
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            info["docstring"] = node.body[0].value.value

        return info

    def _extract_import_info(self, node) -> Dict[str, Any]:
        """Extract import information."""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "line": node.lineno,
                "names": [(alias.name, alias.asname) for alias in node.names]
            }
        else:  # ast.ImportFrom
            return {
                "type": "from_import", 
                "line": node.lineno,
                "module": node.module,
                "names": [(alias.name, alias.asname) for alias in node.names],
                "level": node.level
            }

    def _infer_type(self, node: ast.AST) -> str:
        """Infer the type of a value from AST node."""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return node.func.id
            else:
                return "unknown"
        else:
            return "unknown"

    def _analyze_code_structure(self, code: str, file_path: str, analysis_scope: List[str]) -> Dict[str, Any]:
        """Analyze the overall structure of the code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}

        structure = {
            "file_path": file_path,
            "total_lines": len(code.splitlines()),
            "complexity_metrics": {},
            "structure_summary": {}
        }

        # Count different types of nodes
        node_counts = {}
        function_complexity = {}
        max_nesting_depth = 0

        for node in ast.walk(tree):
            node_type = type(node).__name__
            node_counts[node_type] = node_counts.get(node_type, 0) + 1
            
            # Calculate cyclomatic complexity for functions
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_complexity(node)
                function_complexity[node.name] = complexity
            
            # Track nesting depth
            depth = self._calculate_nesting_depth(node)
            max_nesting_depth = max(max_nesting_depth, depth)

        structure["complexity_metrics"] = {
            "function_count": node_counts.get("FunctionDef", 0),
            "class_count": node_counts.get("ClassDef", 0),
            "max_function_complexity": max(function_complexity.values()) if function_complexity else 0,
            "avg_function_complexity": sum(function_complexity.values()) / len(function_complexity) if function_complexity else 0,
            "max_nesting_depth": max_nesting_depth,
            "function_complexity_details": function_complexity
        }

        structure["structure_summary"] = {
            "imports": node_counts.get("Import", 0) + node_counts.get("ImportFrom", 0),
            "functions": node_counts.get("FunctionDef", 0),
            "classes": node_counts.get("ClassDef", 0),
            "if_statements": node_counts.get("If", 0),
            "for_loops": node_counts.get("For", 0),
            "while_loops": node_counts.get("While", 0),
            "try_blocks": node_counts.get("Try", 0)
        }

        return structure

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity

    def _calculate_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth of a node."""
        max_depth = depth
        
        if isinstance(node, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
            depth += 1
        
        for child in ast.iter_child_nodes(node):
            child_depth = self._calculate_nesting_depth(child, depth)
            max_depth = max(max_depth, child_depth)
        
        return max_depth

    def _analyze_dependencies(self, code: str, file_path: str) -> Dict[str, Any]:
        """Analyze dependencies and imports in the code."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return {"error": f"Syntax error: {str(e)}"}

        dependencies = {
            "file_path": file_path,
            "external_imports": [],
            "internal_imports": [],
            "standard_library": [],
            "unknown_imports": [],
            "import_summary": {}
        }

        # Known standard library modules (partial list)
        stdlib_modules = {
            'os', 'sys', 'json', 'ast', 'typing', 'pathlib', 'collections',
            'itertools', 'functools', 'datetime', 'math', 'random', 're',
            'urllib', 'http', 'asyncio', 'threading', 'multiprocessing'
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in stdlib_modules:
                        dependencies["standard_library"].append(alias.name)
                    elif module_name.startswith('.') or 'trae_agent' in module_name:
                        dependencies["internal_imports"].append(alias.name)
                    else:
                        dependencies["external_imports"].append(alias.name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in stdlib_modules:
                        dependencies["standard_library"].append(node.module)
                    elif node.level > 0 or 'trae_agent' in node.module:
                        dependencies["internal_imports"].append(node.module)
                    else:
                        dependencies["external_imports"].append(node.module)

        # Create summary
        dependencies["import_summary"] = {
            "total_imports": len(dependencies["external_imports"]) + 
                           len(dependencies["internal_imports"]) + 
                           len(dependencies["standard_library"]),
            "external_count": len(set(dependencies["external_imports"])),
            "internal_count": len(set(dependencies["internal_imports"])),
            "stdlib_count": len(set(dependencies["standard_library"]))
        }

        return dependencies

    def _generate_compatibility_report(
        self, 
        old_code: str, 
        new_code: str, 
        file_path: str,
        include_private: bool
    ) -> Dict[str, Any]:
        """Generate a compatibility report between two code versions."""
        try:
            old_api = self._extract_public_api(old_code, file_path, include_private, 
                                              ["functions", "classes", "variables"])
            new_api = self._extract_public_api(new_code, file_path, include_private,
                                              ["functions", "classes", "variables"])
        except Exception as e:
            return {"error": f"Error parsing code: {str(e)}"}

        report = {
            "file_path": file_path,
            "compatibility_status": "compatible",
            "changes": {
                "removed": {},
                "added": {},
                "modified": {}
            },
            "recommendations": []
        }

        # Check function changes
        old_functions = set(old_api.get("functions", {}).keys())
        new_functions = set(new_api.get("functions", {}).keys())
        
        removed_functions = old_functions - new_functions
        added_functions = new_functions - old_functions
        
        if removed_functions:
            report["changes"]["removed"]["functions"] = list(removed_functions)
            report["compatibility_status"] = "breaking_changes"
            report["recommendations"].append("Consider deprecating functions instead of removing them")
        
        if added_functions:
            report["changes"]["added"]["functions"] = list(added_functions)

        # Check class changes
        old_classes = set(old_api.get("classes", {}).keys())
        new_classes = set(new_api.get("classes", {}).keys())
        
        removed_classes = old_classes - new_classes
        added_classes = new_classes - old_classes
        
        if removed_classes:
            report["changes"]["removed"]["classes"] = list(removed_classes)
            report["compatibility_status"] = "breaking_changes"
            report["recommendations"].append("Consider deprecating classes instead of removing them")
        
        if added_classes:
            report["changes"]["added"]["classes"] = list(added_classes)

        # Check for signature changes in existing functions
        common_functions = old_functions & new_functions
        for func_name in common_functions:
            old_func = old_api["functions"][func_name]
            new_func = new_api["functions"][func_name]
            
            if len(old_func["arguments"]) != len(new_func["arguments"]):
                if "modified" not in report["changes"]:
                    report["changes"]["modified"] = {}
                if "functions" not in report["changes"]["modified"]:
                    report["changes"]["modified"]["functions"] = {}
                
                report["changes"]["modified"]["functions"][func_name] = "argument_count_changed"
                if len(old_func["arguments"]) > len(new_func["arguments"]):
                    report["compatibility_status"] = "breaking_changes"

        # Generate final assessment
        if report["compatibility_status"] == "breaking_changes":
            report["recommendations"].append("Review all breaking changes before releasing")
        elif report["changes"]["added"]:
            report["compatibility_status"] = "backward_compatible"
            report["recommendations"].append("New features added - update version appropriately")

        return report

    def _format_output(self, data: Dict[str, Any], output_format: str) -> str:
        """Format the output according to the specified format."""
        if output_format == "json":
            return json.dumps(data, indent=2)
        elif output_format == "markdown":
            return self._format_as_markdown(data)
        else:  # text
            return self._format_as_text(data)

    def _format_as_markdown(self, data: Dict[str, Any]) -> str:
        """Format data as markdown."""
        lines = [f"# Code Analysis Report", ""]
        
        if "file_path" in data:
            lines.extend([f"**File:** {data['file_path']}", ""])
        
        if "functions" in data:
            lines.extend(["## Functions", ""])
            for name, info in data["functions"].items():
                lines.append(f"### {name}")
                lines.append(f"- Line: {info.get('line', 'unknown')}")
                if info.get("docstring"):
                    lines.append(f"- Description: {info['docstring'][:100]}...")
                lines.append("")
        
        if "classes" in data:
            lines.extend(["## Classes", ""])
            for name, info in data["classes"].items():
                lines.append(f"### {name}")
                lines.append(f"- Line: {info.get('line', 'unknown')}")
                lines.append(f"- Methods: {len(info.get('methods', {}))}")
                lines.append("")
        
        return "\n".join(lines)

    def _format_as_text(self, data: Dict[str, Any]) -> str:
        """Format data as plain text."""
        lines = ["Code Analysis Report", "=" * 20, ""]
        
        if "file_path" in data:
            lines.extend([f"File: {data['file_path']}", ""])
        
        if "functions" in data:
            lines.extend([f"Functions ({len(data['functions'])}):", ""])
            for name, info in data["functions"].items():
                lines.append(f"  - {name} (line {info.get('line', '?')})")
            lines.append("")
        
        if "classes" in data:
            lines.extend([f"Classes ({len(data['classes'])}):", ""])
            for name, info in data["classes"].items():
                lines.append(f"  - {name} (line {info.get('line', '?')})")
            lines.append("")
        
        if "complexity_metrics" in data:
            metrics = data["complexity_metrics"]
            lines.extend(["Complexity Metrics:", ""])
            lines.append(f"  - Function count: {metrics.get('function_count', 0)}")
            lines.append(f"  - Class count: {metrics.get('class_count', 0)}")
            lines.append(f"  - Max complexity: {metrics.get('max_function_complexity', 0)}")
            lines.append("")
        
        return "\n".join(lines)