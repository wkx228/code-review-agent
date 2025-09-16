# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tests for BreakingChangeAnalyzer."""

import pytest

from trae_agent.tools.breaking_change_analyzer import (
    BreakingChangeAnalyzer,
    ChangeRiskLevel,
    ChangeType,
    BreakingChange,
    FunctionSignature,
    ClassInterface,
)


@pytest.fixture
def analyzer():
    """Create a BreakingChangeAnalyzer instance."""
    return BreakingChangeAnalyzer()


class TestBreakingChangeAnalyzer:
    """Test BreakingChangeAnalyzer functionality."""

    def test_get_name(self, analyzer):
        """Test tool name."""
        assert analyzer.get_name() == "breaking_change_analyzer"

    def test_get_description(self, analyzer):
        """Test tool description."""
        description = analyzer.get_description()
        assert "breaking changes" in description.lower()
        assert "python code" in description.lower()

    def test_get_parameters(self, analyzer):
        """Test tool parameters."""
        params = analyzer.get_parameters()
        param_names = [p.name for p in params]
        
        assert "analysis_mode" in param_names
        assert "old_code" in param_names
        assert "new_code" in param_names
        assert "file_path" in param_names
        assert "diff_content" in param_names

    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, analyzer):
        """Test with missing required parameters."""
        result = await analyzer.execute({
            "analysis_mode": "file_comparison"
        })
        
        assert result.error_code == 1
        assert "old_code and new_code are required" in result.error

    @pytest.mark.asyncio
    async def test_unknown_analysis_mode(self, analyzer):
        """Test with unknown analysis mode."""
        result = await analyzer.execute({
            "analysis_mode": "unknown_mode"
        })
        
        assert result.error_code == 1
        assert "Unknown analysis mode" in result.error

    @pytest.mark.asyncio
    async def test_file_comparison_no_changes(self, analyzer):
        """Test file comparison with identical code."""
        code = """
def hello():
    print("Hello, World!")

class Person:
    def __init__(self, name):
        self.name = name
"""
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": code,
            "new_code": code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "No breaking changes detected" in result.output

    @pytest.mark.asyncio
    async def test_function_removal_detection(self, analyzer):
        """Test detection of removed functions."""
        old_code = """
def hello():
    print("Hello")

def goodbye():
    print("Goodbye")
"""
        
        new_code = """
def hello():
    print("Hello")
"""
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": old_code,
            "new_code": new_code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "HIGH RISK CHANGES" in result.output
        assert "goodbye" in result.output
        assert "was removed" in result.output

    @pytest.mark.asyncio
    async def test_class_removal_detection(self, analyzer):
        """Test detection of removed classes."""
        old_code = """
class Person:
    def __init__(self, name):
        self.name = name

class Car:
    def __init__(self, model):
        self.model = model
"""
        
        new_code = """
class Person:
    def __init__(self, name):
        self.name = name
"""
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": old_code,
            "new_code": new_code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "HIGH RISK CHANGES" in result.output
        assert "Car" in result.output
        assert "was removed" in result.output

    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, analyzer):
        """Test handling of syntax errors."""
        old_code = "def hello():\n    print('Hello')"
        new_code = "def hello(\n    print('Hello')"  # Missing closing parenthesis
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": old_code,
            "new_code": new_code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "Syntax error" in result.output

    @pytest.mark.asyncio
    async def test_signature_extraction_mode(self, analyzer):
        """Test signature extraction mode."""
        code = """
def greet(name: str, age: int = 25) -> str:
    return f"Hello {name}, you are {age}"

class User:
    def __init__(self, username: str):
        self.username = username
    
    def get_name(self) -> str:
        return self.username
"""
        
        result = await analyzer.execute({
            "analysis_mode": "signature_extraction",
            "new_code": code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        # Should be JSON output
        assert "greet" in result.output or "User" in result.output

    @pytest.mark.asyncio
    async def test_diff_analysis_mode(self, analyzer):
        """Test diff analysis mode."""
        diff_content = """
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
     print("Hello")
+    print("World")
"""
        
        result = await analyzer.execute({
            "analysis_mode": "diff_analysis",
            "diff_content": diff_content,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_ignore_private_members(self, analyzer):
        """Test ignoring private members."""
        old_code = """
def public_function():
    pass

def _private_function():
    pass

class MyClass:
    def public_method(self):
        pass
    
    def _private_method(self):
        pass
"""
        
        new_code = """
def public_function():
    pass

class MyClass:
    def public_method(self):
        pass
"""
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": old_code,
            "new_code": new_code,
            "file_path": "test.py",
            "ignore_private": True
        })
        
        assert result.error_code == 0
        # Should not report removal of private members
        assert "_private_function" not in result.output
        assert "_private_method" not in result.output

    @pytest.mark.asyncio
    async def test_specific_check_types(self, analyzer):
        """Test checking only specific types of changes."""
        old_code = """
def hello():
    pass

class Person:
    def __init__(self):
        pass

import os
"""
        
        new_code = """
class Person:
    def __init__(self):
        pass

import sys
"""
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": old_code,
            "new_code": new_code,
            "file_path": "test.py",
            "check_types": ["function_signature"]  # Only check functions
        })
        
        assert result.error_code == 0
        # Should detect function removal but not import changes
        assert "hello" in result.output
        assert "was removed" in result.output

    def test_function_signature_creation(self, analyzer):
        """Test FunctionSignature creation."""
        sig = FunctionSignature(
            name="test_func",
            args=["self", "param1", "param2"],
            defaults=["default_value"],
            varargs=None,
            kwargs=None,
            annotations={"param1": "str", "param2": "int"},
            return_annotation="bool"
        )
        
        assert sig.name == "test_func"
        assert len(sig.args) == 3
        assert sig.annotations["param1"] == "str"

    def test_class_interface_creation(self, analyzer):
        """Test ClassInterface creation."""
        interface = ClassInterface(
            name="TestClass",
            methods={},
            attributes={"attr1", "attr2"},
            base_classes=["BaseClass"],
            decorators=["@dataclass"]
        )
        
        assert interface.name == "TestClass"
        assert "attr1" in interface.attributes
        assert "BaseClass" in interface.base_classes

    def test_breaking_change_creation(self, analyzer):
        """Test BreakingChange creation."""
        change = BreakingChange(
            change_type=ChangeType.FUNCTION_SIGNATURE,
            risk_level=ChangeRiskLevel.HIGH,
            file_path="test.py",
            line_number=10,
            description="Function was removed",
            suggestion="Consider deprecating instead"
        )
        
        assert change.change_type == ChangeType.FUNCTION_SIGNATURE
        assert change.risk_level == ChangeRiskLevel.HIGH
        assert change.file_path == "test.py"

    @pytest.mark.asyncio
    async def test_analysis_depth_parameter(self, analyzer):
        """Test analysis depth parameter."""
        code = "def hello(): pass"
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": code,
            "new_code": code,
            "file_path": "test.py",
            "analysis_depth": "surface"
        })
        
        assert result.error_code == 0

    def test_format_analysis_result(self, analyzer):
        """Test formatting of analysis results."""
        changes = [
            BreakingChange(
                change_type=ChangeType.FUNCTION_SIGNATURE,
                risk_level=ChangeRiskLevel.HIGH,
                file_path="test.py",
                line_number=5,
                description="Function 'test' was removed",
                suggestion="Consider deprecating instead"
            ),
            BreakingChange(
                change_type=ChangeType.CLASS_INTERFACE,
                risk_level=ChangeRiskLevel.MEDIUM,
                file_path="test.py",
                line_number=10,
                description="Method 'process' was modified",
                suggestion="Review the changes"
            )
        ]
        
        result = analyzer._format_analysis_result(changes)
        
        assert "Breaking Changes Analysis Report" in result
        assert "HIGH RISK CHANGES" in result
        assert "MEDIUM RISK CHANGES" in result
        assert "Function 'test' was removed" in result
        assert "Method 'process' was modified" in result