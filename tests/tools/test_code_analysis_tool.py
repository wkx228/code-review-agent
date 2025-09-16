# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tests for CodeAnalysisTool."""

import json
import pytest

from trae_agent.tools.code_analysis_tool import CodeAnalysisTool


@pytest.fixture
def analysis_tool():
    """Create a CodeAnalysisTool instance."""
    return CodeAnalysisTool()


class TestCodeAnalysisTool:
    """Test CodeAnalysisTool functionality."""

    def test_get_name(self, analysis_tool):
        """Test tool name."""
        assert analysis_tool.get_name() == "code_analysis_tool"

    def test_get_description(self, analysis_tool):
        """Test tool description."""
        description = analysis_tool.get_description()
        assert "code structure" in description.lower()
        assert "compatibility" in description.lower()

    def test_get_parameters(self, analysis_tool):
        """Test tool parameters."""
        params = analysis_tool.get_parameters()
        param_names = [p.name for p in params]
        
        assert "analysis_type" in param_names
        assert "source_code" in param_names
        assert "target_code" in param_names
        assert "file_path" in param_names
        assert "include_private" in param_names
        assert "output_format" in param_names

    @pytest.mark.asyncio
    async def test_missing_source_code(self, analysis_tool):
        """Test with missing source code."""
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction"
        })
        
        assert result.error_code == 1
        assert "source_code is required" in result.error

    @pytest.mark.asyncio
    async def test_unknown_analysis_type(self, analysis_tool):
        """Test with unknown analysis type."""
        result = await analysis_tool.execute({
            "analysis_type": "unknown_type",
            "source_code": "def hello(): pass"
        })
        
        assert result.error_code == 1
        assert "Unknown analysis type" in result.error

    @pytest.mark.asyncio
    async def test_api_extraction_basic(self, analysis_tool):
        """Test basic API extraction."""
        code = """
def public_function(param1: str, param2: int = 10) -> str:
    '''A public function.'''
    return f"{param1}: {param2}"

def _private_function():
    pass

class PublicClass:
    '''A public class.'''
    
    def __init__(self, name: str):
        self.name = name
        self._private_attr = "private"
    
    def public_method(self) -> str:
        return self.name
    
    def _private_method(self):
        pass

CONSTANT = "value"
variable = 42
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "include_private": False,
            "analysis_scope": ["functions", "classes", "variables"]
        })
        
        assert result.error_code == 0
        assert "public_function" in result.output
        assert "PublicClass" in result.output
        assert "_private_function" not in result.output
        assert "_private_method" not in result.output

    @pytest.mark.asyncio
    async def test_api_extraction_with_private(self, analysis_tool):
        """Test API extraction including private members."""
        code = """
def public_function():
    pass

def _private_function():
    pass

class TestClass:
    def public_method(self):
        pass
    
    def _private_method(self):
        pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "include_private": True,
            "analysis_scope": ["functions", "classes"]
        })
        
        assert result.error_code == 0
        assert "_private_function" in result.output
        assert "_private_method" in result.output

    @pytest.mark.asyncio
    async def test_structure_analysis(self, analysis_tool):
        """Test code structure analysis."""
        code = """
import os
import sys
from typing import List

def function1():
    if True:
        for i in range(10):
            while i > 0:
                try:
                    print(i)
                except Exception:
                    pass
                i -= 1

def function2():
    pass

class MyClass:
    def method1(self):
        if True:
            return True
        return False

    def method2(self):
        pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "structure_analysis",
            "source_code": code,
            "file_path": "test.py",
            "analysis_scope": ["functions", "classes"]
        })
        
        assert result.error_code == 0
        assert "complexity_metrics" in result.output
        assert "structure_summary" in result.output
        assert "function_count" in result.output
        assert "class_count" in result.output

    @pytest.mark.asyncio
    async def test_dependency_analysis(self, analysis_tool):
        """Test dependency analysis."""
        code = """
import os
import sys
from typing import List, Dict
from pathlib import Path
import requests
from mymodule import custom_function
from .relative_module import helper
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "dependency_analysis",
            "source_code": code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "external_imports" in result.output
        assert "internal_imports" in result.output
        assert "standard_library" in result.output
        assert "import_summary" in result.output

    @pytest.mark.asyncio
    async def test_compatibility_report(self, analysis_tool):
        """Test compatibility report generation."""
        old_code = """
def old_function(param1, param2):
    return param1 + param2

class OldClass:
    def method1(self):
        pass
    
    def method2(self):
        pass
"""
        
        new_code = """
def old_function(param1, param2, param3=None):
    return param1 + param2

class OldClass:
    def method1(self):
        pass
    
    def new_method(self):
        pass

def new_function():
    pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "compatibility_report",
            "source_code": old_code,
            "target_code": new_code,
            "file_path": "test.py",
            "include_private": False
        })
        
        assert result.error_code == 0
        assert "compatibility_status" in result.output
        assert "changes" in result.output
        assert "recommendations" in result.output

    @pytest.mark.asyncio
    async def test_json_output_format(self, analysis_tool):
        """Test JSON output format."""
        code = """
def test_function():
    pass

class TestClass:
    pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "output_format": "json",
            "analysis_scope": ["functions", "classes"]
        })
        
        assert result.error_code == 0
        # Should be valid JSON
        json.loads(result.output)  # This will raise if not valid JSON

    @pytest.mark.asyncio
    async def test_markdown_output_format(self, analysis_tool):
        """Test Markdown output format."""
        code = """
def test_function():
    '''Test function.'''
    pass

class TestClass:
    '''Test class.'''
    def test_method(self):
        pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "output_format": "markdown",
            "analysis_scope": ["functions", "classes"]
        })
        
        assert result.error_code == 0
        assert "# Code Analysis Report" in result.output
        assert "## Functions" in result.output
        assert "## Classes" in result.output

    @pytest.mark.asyncio
    async def test_syntax_error_handling(self, analysis_tool):
        """Test handling of syntax errors."""
        invalid_code = """
def incomplete_function(
    # Missing closing parenthesis and colon
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": invalid_code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "error" in result.output.lower()

    @pytest.mark.asyncio
    async def test_function_complexity_calculation(self, analysis_tool):
        """Test function complexity calculation."""
        complex_code = """
def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                try:
                    result = process(i)
                    if result:
                        return result
                    else:
                        continue
                except ValueError:
                    pass
                except TypeError:
                    break
            elif i % 3 == 0:
                while i > 0:
                    i -= 1
    return None

def simple_function():
    return "simple"
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "structure_analysis",
            "source_code": complex_code,
            "file_path": "test.py"
        })
        
        assert result.error_code == 0
        assert "max_function_complexity" in result.output
        assert "function_complexity_details" in result.output

    @pytest.mark.asyncio
    async def test_async_function_detection(self, analysis_tool):
        """Test detection of async functions."""
        code = """
async def async_function():
    await some_operation()
    return "done"

def sync_function():
    return "sync"
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "analysis_scope": ["functions"]
        })
        
        assert result.error_code == 0
        assert "async_function" in result.output
        assert "sync_function" in result.output

    @pytest.mark.asyncio
    async def test_decorator_detection(self, analysis_tool):
        """Test detection of decorators."""
        code = """
@property
def getter_method(self):
    return self._value

@staticmethod
def static_method():
    pass

@classmethod  
def class_method(cls):
    pass

@custom_decorator
@another_decorator
def decorated_function():
    pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "analysis_scope": ["functions"]
        })
        
        assert result.error_code == 0
        assert "decorators" in result.output

    @pytest.mark.asyncio
    async def test_type_annotation_extraction(self, analysis_tool):
        """Test extraction of type annotations."""
        code = """
from typing import List, Dict, Optional

def typed_function(
    name: str,
    age: int,
    scores: List[float],
    metadata: Dict[str, str],
    optional_param: Optional[str] = None
) -> bool:
    return True

class TypedClass:
    def __init__(self, value: int):
        self.value: int = value
    
    def get_value(self) -> int:
        return self.value
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "analysis_scope": ["functions", "classes"]
        })
        
        assert result.error_code == 0
        assert "annotation" in result.output

    @pytest.mark.asyncio
    async def test_inheritance_detection(self, analysis_tool):
        """Test detection of class inheritance."""
        code = """
class BaseClass:
    pass

class ChildClass(BaseClass):
    pass

class MultipleInheritance(BaseClass, object):
    pass
"""
        
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": code,
            "file_path": "test.py",
            "analysis_scope": ["classes"]
        })
        
        assert result.error_code == 0
        assert "base_classes" in result.output
        assert "BaseClass" in result.output