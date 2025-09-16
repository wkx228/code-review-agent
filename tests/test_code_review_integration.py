# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Integration tests for end-to-end code review workflow."""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
import git

from trae_agent.agent.code_review_agent import CodeReviewAgent
from trae_agent.tools.git_diff_tool import GitDiffTool
from trae_agent.tools.breaking_change_analyzer import BreakingChangeAnalyzer
from trae_agent.tools.code_analysis_tool import CodeAnalysisTool
from trae_agent.utils.config import TraeAgentConfig


@pytest.fixture
def mock_config():
    """Create a mock AgentConfig for testing."""
    from trae_agent.utils.config import AgentConfig, ModelConfig, ModelProviderConfig, ModelProvider
    
    # Create model provider config
    model_provider_config = ModelProviderConfig(
        provider=ModelProvider.ANTHROPIC,
        api_key="test_key",
        base_url=None,
        api_version=None
    )
    
    # Create model config
    model_config = ModelConfig(
        model_provider=model_provider_config,
        model="test_model",
        max_tokens=8192,
        temperature=0.3,
        top_p=1.0,
        top_k=0,
        max_retries=3,
        parallel_tool_calls=True
    )
    
    # Create agent config
    config = AgentConfig(
        model=model_config,
        max_steps=50,
        tools=[
            "git_diff_tool",
            "breaking_change_analyzer",
            "code_analysis_tool",
            "task_done"
        ]
    )
    
    return config


@pytest.fixture
def temp_git_repo_with_changes():
    """Create a temporary git repository with code changes."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.init(temp_dir)
        
        # Create initial version
        initial_code = '''
def calculate_sum(a, b):
    """Calculate sum of two numbers."""
    return a + b

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def get_history(self):
        return self.history
'''
        
        test_file = Path(temp_dir) / "calculator.py"
        test_file.write_text(initial_code)
        
        # Add and commit initial version
        repo.index.add([str(test_file)])
        repo.index.commit("Initial commit")
        
        # Create modified version with breaking changes
        modified_code = '''
def calculate_sum(a, b, c=0):
    """Calculate sum of numbers."""
    return a + b + c

class Calculator:
    """An advanced calculator class."""
    
    def __init__(self, precision=2):
        self.history = []
        self.precision = precision
    
    def add(self, a, b):
        result = round(a + b, self.precision)
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = round(a * b, self.precision)
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def clear_history(self):
        self.history.clear()
'''
        
        test_file.write_text(modified_code)
        
        # Create another test file
        new_file = Path(temp_dir) / "utils.py"
        new_file.write_text('''
def helper_function():
    """A helper function."""
    pass

class Helper:
    def process(self):
        pass
''')
        
        yield temp_dir


class TestCodeReviewIntegration:
    """Integration tests for the complete code review workflow."""

    @pytest.mark.asyncio
    async def test_git_diff_tool_integration(self, temp_git_repo_with_changes):
        """Test GitDiffTool integration with real git repository."""
        git_tool = GitDiffTool()
        
        # Test unstaged changes
        result = await git_tool.execute({
            "repository_path": temp_git_repo_with_changes,
            "analysis_type": "unstaged_changes",
            "file_pattern": "*.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        assert "calculator.py" in result.output
        assert "Modified files" in result.output

    @pytest.mark.asyncio
    async def test_breaking_change_analyzer_integration(self):
        """Test BreakingChangeAnalyzer with realistic code changes."""
        analyzer = BreakingChangeAnalyzer()
        
        old_code = '''
def process_data(data):
    return data.upper()

class DataProcessor:
    def __init__(self):
        self.processed = []
    
    def process(self, item):
        return item * 2
    
    def get_results(self):
        return self.processed
'''
        
        new_code = '''
def process_data(data, encoding="utf-8"):
    return data.upper().encode(encoding)

class DataProcessor:
    def __init__(self, batch_size=10):
        self.processed = []
        self.batch_size = batch_size
    
    def process_batch(self, items):
        return [item * 2 for item in items]
    
    def get_results(self):
        return self.processed
'''
        
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": old_code,
            "new_code": new_code,
            "file_path": "processor.py",
            "check_types": ["function_signature", "class_interface"]
        })
        
        assert result.error_code == 0
        assert "breaking changes" in result.output.lower()
        # Should detect removed method 'process'
        assert "process" in result.output

    @pytest.mark.asyncio
    async def test_code_analysis_tool_integration(self):
        """Test CodeAnalysisTool with complex code structures."""
        analysis_tool = CodeAnalysisTool()
        
        complex_code = '''
import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class User:
    """User data class."""
    name: str
    age: int
    email: Optional[str] = None

class UserManager:
    """Manages user operations."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
    
    @property
    def user_count(self) -> int:
        return len(self.users)
    
    async def add_user(self, user: User) -> bool:
        """Add a user to the system."""
        if user.name in self.users:
            return False
        
        self.users[user.name] = user
        return True
    
    def get_user(self, name: str) -> Optional[User]:
        """Get user by name."""
        return self.users.get(name)
    
    def _validate_user(self, user: User) -> bool:
        """Private validation method."""
        return bool(user.name and user.age > 0)

def create_admin_user() -> User:
    """Create an admin user."""
    return User("admin", 30, "admin@example.com")
'''
        
        # Test API extraction
        result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": complex_code,
            "file_path": "user_manager.py",
            "include_private": False,
            "output_format": "json"
        })
        
        assert result.error_code == 0
        assert "UserManager" in result.output
        assert "add_user" in result.output
        assert "_validate_user" not in result.output  # Private method excluded

    @pytest.mark.asyncio  
    async def test_tools_workflow_integration(self, temp_git_repo_with_changes):
        """Test the integration of multiple tools in a workflow."""
        # Step 1: Analyze git changes
        git_tool = GitDiffTool()
        git_result = await git_tool.execute({
            "repository_path": temp_git_repo_with_changes,
            "analysis_type": "modified_files",
            "file_pattern": "*.py"
        })
        
        assert git_result.error_code == 0
        
        # Step 2: Get specific file changes
        file_result = await git_tool.execute({
            "repository_path": temp_git_repo_with_changes,
            "analysis_type": "file_changes",
            "target_file": "calculator.py"
        })
        
        assert file_result.error_code == 0
        
        # Step 3: Analyze code structure
        analysis_tool = CodeAnalysisTool()
        
        current_code = (Path(temp_git_repo_with_changes) / "calculator.py").read_text()
        structure_result = await analysis_tool.execute({
            "analysis_type": "structure_analysis",
            "source_code": current_code,
            "file_path": "calculator.py"
        })
        
        assert structure_result.error_code == 0
        assert "complexity_metrics" in structure_result.output

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling across tool integrations."""
        git_tool = GitDiffTool()
        
        # Test with non-existent repository
        result = await git_tool.execute({
            "repository_path": "/non/existent/path",
            "analysis_type": "unstaged_changes"
        })
        
        assert result.error_code == 1
        assert "does not exist" in result.error
        
        # Test breaking change analyzer with invalid code
        analyzer = BreakingChangeAnalyzer()
        result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": "def valid(): pass",
            "new_code": "def invalid(\n  # syntax error",
            "file_path": "test.py"
        })
        
        assert result.error_code == 0  # Should handle gracefully
        assert "syntax error" in result.output.lower()

    @pytest.mark.asyncio
    async def test_compatibility_analysis_workflow(self):
        """Test end-to-end compatibility analysis workflow."""
        # Simulate a real API evolution scenario
        v1_api = '''
class APIClient:
    def __init__(self, endpoint):
        self.endpoint = endpoint
    
    def get_data(self, id):
        """Get data by ID."""
        return {"id": id, "data": "value"}
    
    def post_data(self, data):
        """Post data."""
        return {"status": "created"}
    
    def delete_data(self, id):
        """Delete data by ID."""
        return {"status": "deleted"}
'''
        
        v2_api = '''
class APIClient:
    def __init__(self, endpoint, timeout=30):
        self.endpoint = endpoint
        self.timeout = timeout
    
    def get_data(self, id, include_metadata=False):
        """Get data by ID with optional metadata."""
        result = {"id": id, "data": "value"}
        if include_metadata:
            result["metadata"] = {"version": "2.0"}
        return result
    
    def post_data(self, data, validate=True):
        """Post data with validation."""
        if validate and not data:
            raise ValueError("Data cannot be empty")
        return {"status": "created", "validated": validate}
    
    def update_data(self, id, data):
        """Update existing data."""
        return {"status": "updated", "id": id}
    
    # delete_data method removed - BREAKING CHANGE
'''
        
        # Step 1: Analyze breaking changes
        analyzer = BreakingChangeAnalyzer()
        breaking_result = await analyzer.execute({
            "analysis_mode": "file_comparison",
            "old_code": v1_api,
            "new_code": v2_api,
            "file_path": "api_client.py",
            "check_types": ["function_signature", "class_interface"]
        })
        
        assert breaking_result.error_code == 0
        assert "breaking changes" in breaking_result.output.lower()
        assert "delete_data" in breaking_result.output  # Should detect removed method
        
        # Step 2: Generate compatibility report
        analysis_tool = CodeAnalysisTool()
        compat_result = await analysis_tool.execute({
            "analysis_type": "compatibility_report",
            "source_code": v1_api,
            "target_code": v2_api,
            "file_path": "api_client.py"
        })
        
        assert compat_result.error_code == 0
        assert "compatibility" in compat_result.output
        assert "breaking_changes" in compat_result.output

    @pytest.mark.asyncio
    async def test_large_codebase_analysis(self):
        """Test analysis of larger, more realistic codebase."""
        # Simulate a larger module
        large_module = '''
"""
A comprehensive data processing module.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ProcessingConfig:
    batch_size: int = 100
    max_retries: int = 3
    timeout: float = 30.0
    parallel_workers: int = 4

class ProcessorProtocol(Protocol):
    async def process(self, data: Any) -> Any: ...

class BaseProcessor(ABC):
    """Abstract base processor."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.status = ProcessingStatus.PENDING
    
    @abstractmethod
    async def process_item(self, item: Any) -> Any:
        """Process a single item."""
        pass
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Process a batch of items."""
        results = []
        for item in items:
            try:
                result = await self.process_item(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process item: {e}")
                results.append(None)
        return results

class DataProcessor(BaseProcessor):
    """Concrete data processor implementation."""
    
    def __init__(self, config: ProcessingConfig, validator=None):
        super().__init__(config)
        self.validator = validator
        self.processed_count = 0
    
    async def process_item(self, item: Any) -> Any:
        """Process a single data item."""
        if self.validator and not self.validator(item):
            raise ValueError(f"Invalid item: {item}")
        
        # Simulate processing
        await asyncio.sleep(0.01)
        self.processed_count += 1
        return f"processed_{item}"
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "processed_count": self.processed_count,
            "batch_size": self.config.batch_size
        }

class ProcessorManager:
    """Manages multiple processors."""
    
    def __init__(self):
        self.processors: Dict[str, BaseProcessor] = {}
        self._lock = asyncio.Lock()
    
    def register_processor(self, name: str, processor: BaseProcessor) -> None:
        """Register a processor."""
        self.processors[name] = processor
    
    async def process_with_processor(
        self, 
        processor_name: str, 
        data: List[Any]
    ) -> Optional[List[Any]]:
        """Process data with specific processor."""
        async with self._lock:
            processor = self.processors.get(processor_name)
            if not processor:
                return None
            
            return await processor.process_batch(data)
    
    def list_processors(self) -> List[str]:
        """List available processors."""
        return list(self.processors.keys())

# Utility functions
def validate_data(data: Any) -> bool:
    """Validate input data."""
    return data is not None and str(data).strip()

async def create_default_processor(config: Optional[ProcessingConfig] = None) -> DataProcessor:
    """Create a default processor instance."""
    if config is None:
        config = ProcessingConfig()
    
    return DataProcessor(config, validator=validate_data)
'''
        
        # Test comprehensive analysis
        analysis_tool = CodeAnalysisTool()
        
        # Test structure analysis
        structure_result = await analysis_tool.execute({
            "analysis_type": "structure_analysis",
            "source_code": large_module,
            "file_path": "data_processor.py"
        })
        
        assert structure_result.error_code == 0
        assert "complexity_metrics" in structure_result.output
        assert "function_count" in structure_result.output
        assert "class_count" in structure_result.output
        
        # Test API extraction
        api_result = await analysis_tool.execute({
            "analysis_type": "api_extraction",
            "source_code": large_module,
            "file_path": "data_processor.py",
            "include_private": False,
            "analysis_scope": ["functions", "classes", "variables"]
        })
        
        assert api_result.error_code == 0
        assert "DataProcessor" in api_result.output
        assert "ProcessorManager" in api_result.output
        assert "ProcessingStatus" in api_result.output
        
        # Test dependency analysis
        dep_result = await analysis_tool.execute({
            "analysis_type": "dependency_analysis",
            "source_code": large_module,
            "file_path": "data_processor.py"
        })
        
        assert dep_result.error_code == 0
        assert "import_summary" in dep_result.output
        assert "external_imports" in dep_result.output
        assert "standard_library" in dep_result.output

    @pytest.mark.asyncio
    async def test_real_world_breaking_changes(self):
        """Test detection of real-world breaking change patterns."""
        scenarios = [
            {
                "name": "Function parameter removal",
                "old": "def func(a, b, c): return a + b + c",
                "new": "def func(a, b): return a + b",
                "expected": ["parameter", "removed"]
            },
            {
                "name": "Method signature change",
                "old": "class A:\n    def method(self, x): pass",
                "new": "class A:\n    def method(self, x, y): pass",
                "expected": ["method", "signature"]
            },
            {
                "name": "Class removal", 
                "old": "class OldClass: pass\nclass NewClass: pass",
                "new": "class NewClass: pass",
                "expected": ["OldClass", "removed"]
            },
            {
                "name": "Return type change",
                "old": "def func() -> int: return 1",
                "new": "def func() -> str: return '1'",
                "expected": ["return", "type"]
            }
        ]
        
        analyzer = BreakingChangeAnalyzer()
        
        for scenario in scenarios:
            result = await analyzer.execute({
                "analysis_mode": "file_comparison",
                "old_code": scenario["old"],
                "new_code": scenario["new"],
                "file_path": f"{scenario['name']}.py"
            })
            
            assert result.error_code == 0
            
            # Check that expected keywords are in the output
            for keyword in scenario["expected"]:
                assert keyword.lower() in result.output.lower(), \
                    f"Expected '{keyword}' in output for scenario '{scenario['name']}'"

    def test_tool_parameter_validation(self):
        """Test parameter validation across all tools."""
        tools = [GitDiffTool(), BreakingChangeAnalyzer(), CodeAnalysisTool()]
        
        for tool in tools:
            params = tool.get_parameters()
            
            # All tools should have basic metadata
            assert tool.get_name()
            assert tool.get_description()
            assert isinstance(params, list)
            
            # Check parameter structure
            for param in params:
                assert hasattr(param, 'name')
                assert hasattr(param, 'type')
                assert hasattr(param, 'description')
                assert hasattr(param, 'required')