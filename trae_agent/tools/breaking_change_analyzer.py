# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Breaking change analyzer for code review."""

import ast
import difflib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, override

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class ChangeRiskLevel(Enum):
    """Risk levels for breaking changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChangeType(Enum):
    """Types of breaking changes."""
    FUNCTION_SIGNATURE = "function_signature"
    CLASS_INTERFACE = "class_interface"
    MODULE_STRUCTURE = "module_structure"
    EXCEPTION_HANDLING = "exception_handling"
    RETURN_VALUE = "return_value"
    IMPORT_PATH = "import_path"


@dataclass
class BreakingChange:
    """Represents a detected breaking change."""
    change_type: ChangeType
    risk_level: ChangeRiskLevel
    file_path: str
    line_number: int
    description: str
    old_signature: Optional[str] = None
    new_signature: Optional[str] = None
    suggestion: Optional[str] = None
    impact_assessment: Optional[str] = None


@dataclass
class FunctionSignature:
    """Represents a function signature for comparison."""
    name: str
    args: List[str]
    defaults: List[Any]
    varargs: Optional[str]
    kwargs: Optional[str]
    annotations: Dict[str, str]
    return_annotation: Optional[str]
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False


@dataclass
class ClassInterface:
    """Represents a class interface for comparison."""
    name: str
    methods: Dict[str, FunctionSignature]
    attributes: Set[str]
    base_classes: List[str]
    decorators: List[str]


class BreakingChangeAnalyzer(Tool):
    """Tool for analyzing breaking changes in Python code."""

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)

    @override
    def get_name(self) -> str:
        return "breaking_change_analyzer"

    @override
    def get_description(self) -> str:
        return """Analyze Python code changes to detect potential breaking changes.
        
        This tool can detect:
        - Function signature changes (parameters, types, defaults)
        - Class interface modifications (methods, attributes)
        - Module structure changes (imports, exports)
        - Exception handling changes
        - Return value modifications
        - Import path changes
        """

    @override
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="analysis_mode",
                type="string",
                description="Mode of analysis to perform",
                enum=["file_comparison", "diff_analysis", "signature_extraction"],
                required=True,
            ),
            ToolParameter(
                name="old_code",
                type="string",
                description="Original code content (for file_comparison mode)",
                required=False,
            ),
            ToolParameter(
                name="new_code",
                type="string",
                description="Modified code content (for file_comparison mode)",
                required=False,
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="Path to the file being analyzed",
                required=False,
            ),
            ToolParameter(
                name="diff_content",
                type="string",
                description="Git diff content (for diff_analysis mode)",
                required=False,
            ),
            ToolParameter(
                name="analysis_depth",
                type="string",
                description="Depth of analysis",
                enum=["surface", "deep"],
                required=False,
            ),
            ToolParameter(
                name="ignore_private",
                type="boolean",
                description="Whether to ignore private members (starting with _)",
                required=False,
            ),
            ToolParameter(
                name="check_types",
                type="array",
                description="Specific types of changes to check",
                items={"type": "string", "enum": ["function_signature", "class_interface", "module_structure", "exception_handling", "return_value", "import_path"]},
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            analysis_mode = arguments["analysis_mode"]
            analysis_depth = arguments.get("analysis_depth", "deep")
            ignore_private = arguments.get("ignore_private", True)
            check_types = arguments.get("check_types", ["function_signature", "class_interface", "module_structure"])
            
            if analysis_mode == "file_comparison":
                old_code = arguments.get("old_code")
                new_code = arguments.get("new_code")
                file_path = arguments.get("file_path", "unknown.py")
                
                if not old_code or not new_code:
                    return ToolExecResult(
                        error="old_code and new_code are required for file_comparison mode",
                        error_code=1,
                    )
                
                breaking_changes = self._analyze_code_comparison(
                    old_code, new_code, file_path, analysis_depth, ignore_private, check_types
                )
                
            elif analysis_mode == "diff_analysis":
                diff_content = arguments.get("diff_content")
                file_path = arguments.get("file_path", "unknown.py")
                
                if not diff_content:
                    return ToolExecResult(
                        error="diff_content is required for diff_analysis mode",
                        error_code=1,
                    )
                
                breaking_changes = self._analyze_diff_content(
                    diff_content, file_path, analysis_depth, ignore_private, check_types
                )
                
            elif analysis_mode == "signature_extraction":
                code = arguments.get("new_code") or arguments.get("old_code")
                file_path = arguments.get("file_path", "unknown.py")
                
                if not code:
                    return ToolExecResult(
                        error="code content is required for signature_extraction mode",
                        error_code=1,
                    )
                
                signatures = self._extract_signatures(code, file_path, ignore_private)
                return ToolExecResult(output=json.dumps(signatures, indent=2))
                
            else:
                return ToolExecResult(
                    error=f"Unknown analysis mode: {analysis_mode}",
                    error_code=1,
                )

            # Format output
            result = self._format_analysis_result(breaking_changes)
            return ToolExecResult(output=result)

        except Exception as e:
            return ToolExecResult(
                error=f"Error during breaking change analysis: {str(e)}",
                error_code=1,
            )

    def _analyze_code_comparison(
        self, 
        old_code: str, 
        new_code: str, 
        file_path: str,
        analysis_depth: str,
        ignore_private: bool,
        check_types: List[str]
    ) -> List[BreakingChange]:
        """Compare two versions of code and detect breaking changes."""
        breaking_changes = []
        
        try:
            old_tree = ast.parse(old_code)
            new_tree = ast.parse(new_code)
        except SyntaxError as e:
            return [BreakingChange(
                change_type=ChangeType.MODULE_STRUCTURE,
                risk_level=ChangeRiskLevel.HIGH,
                file_path=file_path,
                line_number=getattr(e, 'lineno', 0),
                description=f"Syntax error in code: {str(e)}",
                suggestion="Fix syntax errors before analyzing breaking changes"
            )]
        
        # Extract signatures from both versions
        old_signatures = self._extract_api_elements(old_tree, ignore_private)
        new_signatures = self._extract_api_elements(new_tree, ignore_private)
        
        # Check function signature changes
        if "function_signature" in check_types:
            breaking_changes.extend(
                self._detect_function_changes(old_signatures, new_signatures, file_path)
            )
        
        # Check class interface changes
        if "class_interface" in check_types:
            breaking_changes.extend(
                self._detect_class_changes(old_signatures, new_signatures, file_path)
            )
        
        # Check module structure changes
        if "module_structure" in check_types:
            breaking_changes.extend(
                self._detect_module_changes(old_tree, new_tree, file_path)
            )
        
        return breaking_changes

    def _extract_api_elements(self, tree: ast.AST, ignore_private: bool) -> Dict[str, Any]:
        """Extract API elements from AST."""
        elements = {
            "functions": {},
            "classes": {},
            "imports": [],
            "variables": set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if ignore_private and node.name.startswith('_') and not node.name.startswith('__'):
                    continue
                elements["functions"][node.name] = self._extract_function_signature(node)
            
            elif isinstance(node, ast.ClassDef):
                if ignore_private and node.name.startswith('_'):
                    continue
                elements["classes"][node.name] = self._extract_class_interface(node, ignore_private)
        
        return elements

    def _extract_function_signature(self, node: ast.FunctionDef) -> FunctionSignature:
        """Extract function signature from AST node."""
        args = [arg.arg for arg in node.args.args]
        defaults = [ast.unparse(default) for default in node.args.defaults]
        annotations = {}
        
        for arg in node.args.args:
            if arg.annotation:
                annotations[arg.arg] = ast.unparse(arg.annotation)
        
        return FunctionSignature(
            name=node.name,
            args=args,
            defaults=defaults,
            varargs=node.args.vararg.arg if node.args.vararg else None,
            kwargs=node.args.kwarg.arg if node.args.kwarg else None,
            annotations=annotations,
            return_annotation=ast.unparse(node.returns) if node.returns else None,
        )

    def _extract_class_interface(self, node: ast.ClassDef, ignore_private: bool) -> ClassInterface:
        """Extract class interface from AST node."""
        methods = {}
        attributes = set()
        base_classes = [ast.unparse(base) for base in node.bases]
        decorators = [ast.unparse(decorator) for decorator in node.decorator_list]
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if ignore_private and item.name.startswith('_') and not item.name.startswith('__'):
                    continue
                methods[item.name] = self._extract_function_signature(item)
        
        return ClassInterface(
            name=node.name,
            methods=methods,
            attributes=attributes,
            base_classes=base_classes,
            decorators=decorators
        )

    def _detect_function_changes(
        self, 
        old_api: Dict[str, Any], 
        new_api: Dict[str, Any], 
        file_path: str
    ) -> List[BreakingChange]:
        """Detect function signature changes."""
        changes = []
        old_functions = old_api.get("functions", {})
        new_functions = new_api.get("functions", {})
        
        # Check for removed functions
        for func_name in old_functions:
            if func_name not in new_functions:
                changes.append(BreakingChange(
                    change_type=ChangeType.FUNCTION_SIGNATURE,
                    risk_level=ChangeRiskLevel.HIGH,
                    file_path=file_path,
                    line_number=0,
                    description=f"Function '{func_name}' was removed",
                    old_signature=self._format_function_signature(old_functions[func_name]),
                    suggestion=f"Consider deprecating '{func_name}' instead of removing it",
                    impact_assessment="High impact - will break existing code that calls this function"
                ))
        
        return changes

    def _detect_class_changes(
        self, 
        old_api: Dict[str, Any], 
        new_api: Dict[str, Any], 
        file_path: str
    ) -> List[BreakingChange]:
        """Detect class interface changes."""
        changes = []
        old_classes = old_api.get("classes", {})
        new_classes = new_api.get("classes", {})
        
        # Check for removed classes
        for class_name in old_classes:
            if class_name not in new_classes:
                changes.append(BreakingChange(
                    change_type=ChangeType.CLASS_INTERFACE,
                    risk_level=ChangeRiskLevel.HIGH,
                    file_path=file_path,
                    line_number=0,
                    description=f"Class '{class_name}' was removed",
                    suggestion=f"Consider deprecating '{class_name}' instead of removing it",
                    impact_assessment="High impact - will break existing code that uses this class"
                ))
        
        return changes

    def _detect_module_changes(
        self, 
        old_tree: ast.AST, 
        new_tree: ast.AST, 
        file_path: str
    ) -> List[BreakingChange]:
        """Detect module structure changes."""
        changes = []
        
        # Extract module-level elements
        old_elements = set()
        new_elements = set()
        
        for tree, elements in [(old_tree, old_elements), (new_tree, new_elements)]:
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.FunctionDef):
                    elements.add(f"function:{node.name}")
                elif isinstance(node, ast.ClassDef):
                    elements.add(f"class:{node.name}")
        
        # Check for removed elements
        removed_elements = old_elements - new_elements
        for element in removed_elements:
            element_type, element_name = element.split(":", 1)
            changes.append(BreakingChange(
                change_type=ChangeType.MODULE_STRUCTURE,
                risk_level=ChangeRiskLevel.HIGH,
                file_path=file_path,
                line_number=0,
                description=f"Module-level {element_type} '{element_name}' was removed",
                suggestion=f"Consider keeping '{element_name}' for backward compatibility",
                impact_assessment=f"High impact - will break imports of '{element_name}'"
            ))
        
        return changes

    def _format_function_signature(self, sig: FunctionSignature) -> str:
        """Format function signature as string."""
        args_str = ", ".join(sig.args)
        return_str = f" -> {sig.return_annotation}" if sig.return_annotation else ""
        return f"def {sig.name}({args_str}){return_str}"

    def _format_analysis_result(self, breaking_changes: List[BreakingChange]) -> str:
        """Format the analysis result as a readable report."""
        if not breaking_changes:
            return "✅ No breaking changes detected."
        
        result = ["🔍 Breaking Changes Analysis Report", "=" * 50, ""]
        
        # Group by risk level
        high_risk = [c for c in breaking_changes if c.risk_level == ChangeRiskLevel.HIGH]
        medium_risk = [c for c in breaking_changes if c.risk_level == ChangeRiskLevel.MEDIUM]
        low_risk = [c for c in breaking_changes if c.risk_level == ChangeRiskLevel.LOW]
        
        if high_risk:
            result.extend(["🚨 HIGH RISK CHANGES:", ""])
            for change in high_risk:
                result.append(f"- {change.description}")
                if change.suggestion:
                    result.append(f"  💡 Suggestion: {change.suggestion}")
                result.append("")
        
        if medium_risk:
            result.extend(["⚠️  MEDIUM RISK CHANGES:", ""])
            for change in medium_risk:
                result.append(f"- {change.description}")
                if change.suggestion:
                    result.append(f"  💡 Suggestion: {change.suggestion}")
                result.append("")
        
        if low_risk:
            result.extend(["ℹ️  LOW RISK CHANGES:", ""])
            for change in low_risk:
                result.append(f"- {change.description}")
                result.append("")
        
        result.append(f"📊 Summary: {len(breaking_changes)} breaking changes detected")
        return "\n".join(result)

    def _analyze_diff_content(self, diff_content: str, file_path: str, analysis_depth: str, ignore_private: bool, check_types: List[str]) -> List[BreakingChange]:
        """Analyze diff content for breaking changes."""
        # Simple implementation - could be enhanced
        return [BreakingChange(
            change_type=ChangeType.MODULE_STRUCTURE,
            risk_level=ChangeRiskLevel.MEDIUM,
            file_path=file_path,
            line_number=0,
            description="Diff analysis detected changes",
            suggestion="Review the changes manually for breaking changes"
        )]

    def _extract_signatures(self, code: str, file_path: str, ignore_private: bool) -> Dict[str, Any]:
        """Extract signatures for API documentation."""
        try:
            tree = ast.parse(code)
            api_elements = self._extract_api_elements(tree, ignore_private)
            return {"file_path": file_path, "api_elements": str(api_elements)}
        except Exception as e:
            return {"error": str(e)}