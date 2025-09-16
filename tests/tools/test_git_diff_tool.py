# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tests for GitDiffTool."""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import git

from trae_agent.tools.git_diff_tool import GitDiffTool


@pytest.fixture
def git_tool():
    """Create a GitDiffTool instance."""
    return GitDiffTool()


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = git.Repo.init(temp_dir)
        
        # Create a test file
        test_file = Path(temp_dir) / "test.py"
        test_file.write_text("def hello():\n    print('Hello, World!')\n")
        
        # Add and commit
        repo.index.add([str(test_file)])
        repo.index.commit("Initial commit")
        
        # Make some changes
        test_file.write_text("def hello():\n    print('Hello, World! Modified')\n\ndef new_function():\n    pass\n")
        
        yield temp_dir


class TestGitDiffTool:
    """Test GitDiffTool functionality."""

    def test_get_name(self, git_tool):
        """Test tool name."""
        assert git_tool.get_name() == "git_diff_tool"

    def test_get_description(self, git_tool):
        """Test tool description."""
        description = git_tool.get_description()
        assert "git repository changes" in description
        assert "diff information" in description

    def test_get_parameters(self, git_tool):
        """Test tool parameters."""
        params = git_tool.get_parameters()
        param_names = [p.name for p in params]
        
        assert "repository_path" in param_names
        assert "analysis_type" in param_names
        assert "file_pattern" in param_names
        assert "target_file" in param_names

    @pytest.mark.asyncio
    async def test_invalid_repository_path(self, git_tool):
        """Test with invalid repository path."""
        result = await git_tool.execute({
            "repository_path": "/nonexistent/path",
            "analysis_type": "unstaged_changes"
        })
        
        assert result.error_code == 1
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_not_git_repository(self, git_tool):
        """Test with path that's not a git repository."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = await git_tool.execute({
                "repository_path": temp_dir,
                "analysis_type": "unstaged_changes"
            })
            
            assert result.error_code == 1
            assert "not a valid git repository" in result.error

    @pytest.mark.asyncio
    async def test_unstaged_changes_analysis(self, git_tool, temp_git_repo):
        """Test unstaged changes analysis."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "unstaged_changes",
            "file_pattern": "*.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        assert "Modified files" in result.output
        assert "test.py" in result.output

    @pytest.mark.asyncio
    async def test_modified_files_analysis(self, git_tool, temp_git_repo):
        """Test modified files analysis."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "modified_files",
            "file_pattern": "*.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        assert "GIT STATUS SUMMARY" in result.output

    @pytest.mark.asyncio
    async def test_file_changes_analysis(self, git_tool, temp_git_repo):
        """Test file changes analysis."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "file_changes",
            "target_file": "test.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        assert "Changes in file: test.py" in result.output

    @pytest.mark.asyncio
    async def test_file_changes_missing_target_file(self, git_tool, temp_git_repo):
        """Test file changes analysis without target file."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "file_changes"
        })
        
        assert result.error_code == 1
        assert "target_file is required" in result.error

    @pytest.mark.asyncio
    async def test_line_diff_analysis(self, git_tool, temp_git_repo):
        """Test line diff analysis."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "line_diff",
            "target_file": "test.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        assert "Line-by-line diff" in result.output

    @pytest.mark.asyncio
    async def test_line_diff_missing_target_file(self, git_tool, temp_git_repo):
        """Test line diff analysis without target file."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "line_diff"
        })
        
        assert result.error_code == 1
        assert "target_file is required" in result.error

    @pytest.mark.asyncio
    async def test_unknown_analysis_type(self, git_tool, temp_git_repo):
        """Test with unknown analysis type."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "unknown_type"
        })
        
        assert result.error_code == 1
        assert "Unknown analysis type" in result.error

    @pytest.mark.asyncio
    async def test_include_untracked_files(self, git_tool, temp_git_repo):
        """Test including untracked files."""
        # Create an untracked file
        untracked_file = Path(temp_git_repo) / "untracked.py"
        untracked_file.write_text("# This is untracked\n")
        
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "unstaged_changes",
            "include_untracked": True,
            "file_pattern": "*.py"
        })
        
        assert result.error_code == 0
        assert result.output is not None
        assert "Untracked files" in result.output
        assert "untracked.py" in result.output

    @pytest.mark.asyncio
    async def test_file_pattern_filtering(self, git_tool, temp_git_repo):
        """Test file pattern filtering."""
        # Create a non-Python file
        other_file = Path(temp_git_repo) / "README.md"
        other_file.write_text("# README\n")
        
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "unstaged_changes",
            "file_pattern": "*.md",
            "include_untracked": True
        })
        
        assert result.error_code == 0
        assert result.output is not None
        # Should include README.md but not test.py
        assert "README.md" in result.output
        assert "test.py" not in result.output or "Modified files (0)" in result.output

    @pytest.mark.asyncio
    async def test_base_commit_parameter(self, git_tool, temp_git_repo):
        """Test using a specific base commit."""
        result = await git_tool.execute({
            "repository_path": temp_git_repo,
            "analysis_type": "file_changes",
            "target_file": "test.py",
            "base_commit": "HEAD"
        })
        
        assert result.error_code == 0
        assert result.output is not None