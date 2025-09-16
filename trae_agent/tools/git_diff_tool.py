# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Git diff analysis tool for code review."""

import difflib
import os
from pathlib import Path
from typing import Any, Dict, List, override

import git
from git import Repo

from trae_agent.tools.base import Tool, ToolCallArguments, ToolExecResult, ToolParameter


class GitDiffTool(Tool):
    """Tool for analyzing git differences in a repository."""

    def __init__(self, model_provider: str | None = None):
        super().__init__(model_provider)

    @override
    def get_name(self) -> str:
        return "git_diff_tool"

    @override
    def get_description(self) -> str:
        return """Analyze git repository changes and provide detailed diff information.
        
        This tool can:
        - Get unstaged changes in the working directory
        - Analyze specific file changes
        - Get list of modified files
        - Extract line-by-line differences
        """

    @override
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="repository_path",
                type="string",
                description="Absolute path to the git repository",
                required=True,
            ),
            ToolParameter(
                name="analysis_type",
                type="string",
                description="Type of analysis to perform",
                enum=["unstaged_changes", "file_changes", "modified_files", "line_diff"],
                required=True,
            ),
            ToolParameter(
                name="file_pattern",
                type="string",
                description="File pattern to filter (e.g., '*.py' for Python files)",
                required=False,
            ),
            ToolParameter(
                name="target_file",
                type="string",
                description="Specific file to analyze (required for file_changes and line_diff)",
                required=False,
            ),
            ToolParameter(
                name="include_untracked",
                type="boolean",
                description="Whether to include untracked files in analysis",
                required=False,
            ),
            ToolParameter(
                name="base_commit",
                type="string",
                description="Base commit to compare against (default: HEAD)",
                required=False,
            ),
        ]

    @override
    async def execute(self, arguments: ToolCallArguments) -> ToolExecResult:
        try:
            repository_path = arguments["repository_path"]
            analysis_type = arguments["analysis_type"]
            file_pattern = arguments.get("file_pattern", "*.py")
            target_file = arguments.get("target_file")
            include_untracked = arguments.get("include_untracked", False)
            base_commit = arguments.get("base_commit", "HEAD")

            if not os.path.exists(repository_path):
                return ToolExecResult(
                    error=f"Repository path does not exist: {repository_path}",
                    error_code=1,
                )

            try:
                repo = Repo(repository_path)
            except git.exc.InvalidGitRepositoryError:
                return ToolExecResult(
                    error=f"Path is not a valid git repository: {repository_path}",
                    error_code=1,
                )

            if analysis_type == "unstaged_changes":
                result = self._get_unstaged_changes(repo, file_pattern, include_untracked)
            elif analysis_type == "file_changes":
                if not target_file:
                    return ToolExecResult(
                        error="target_file is required for file_changes analysis",
                        error_code=1,
                    )
                result = self._analyze_file_changes(repo, target_file, base_commit)
            elif analysis_type == "modified_files":
                result = self._get_modified_files(repo, file_pattern, include_untracked)
            elif analysis_type == "line_diff":
                if not target_file:
                    return ToolExecResult(
                        error="target_file is required for line_diff analysis",
                        error_code=1,
                    )
                result = self._get_line_diff(repo, target_file, base_commit)
            else:
                return ToolExecResult(
                    error=f"Unknown analysis type: {analysis_type}",
                    error_code=1,
                )

            return ToolExecResult(output=result)

        except Exception as e:
            return ToolExecResult(
                error=f"Error during git analysis: {str(e)}",
                error_code=1,
            )

    def _get_unstaged_changes(self, repo: Repo, file_pattern: str, include_untracked: bool) -> str:
        """Get all unstaged changes in the repository."""
        changes = []
        
        # Get modified files
        modified_files = [item.a_path for item in repo.index.diff(None)]
        
        # Filter by pattern
        if file_pattern:
            import fnmatch
            modified_files = [f for f in modified_files if fnmatch.fnmatch(f, file_pattern)]
        
        changes.append(f"Modified files ({len(modified_files)}):")
        for file_path in modified_files:
            changes.append(f"  - {file_path}")
        
        # Get untracked files if requested
        if include_untracked:
            untracked_files = repo.untracked_files
            if file_pattern:
                import fnmatch
                untracked_files = [f for f in untracked_files if fnmatch.fnmatch(f, file_pattern)]
            
            changes.append(f"\nUntracked files ({len(untracked_files)}):")
            for file_path in untracked_files:
                changes.append(f"  - {file_path}")
        
        # Get diff summary
        try:
            diff_output = repo.git.diff("--stat")
            if diff_output:
                changes.append(f"\nDiff summary:\n{diff_output}")
        except Exception:
            pass
        
        return "\n".join(changes)

    def _analyze_file_changes(self, repo: Repo, target_file: str, base_commit: str) -> str:
        """Analyze changes in a specific file."""
        try:
            file_path = Path(repo.working_dir) / target_file
            
            if not file_path.exists():
                return f"File does not exist: {target_file}"
            
            # Get the diff for the specific file
            try:
                if base_commit == "HEAD":
                    # Compare working directory with last commit
                    diff_output = repo.git.diff("HEAD", "--", target_file)
                else:
                    # Compare with specific commit
                    diff_output = repo.git.diff(base_commit, "HEAD", "--", target_file)
            except Exception:
                # If no commits exist or file is untracked
                diff_output = "File is new or untracked"
            
            result = [f"Changes in file: {target_file}"]
            
            if diff_output:
                result.append(f"\nDiff output:\n{diff_output}")
            else:
                result.append("\nNo changes detected in this file.")
            
            # Add file statistics
            try:
                lines = file_path.read_text(encoding='utf-8').splitlines()
                result.append(f"\nFile statistics:")
                result.append(f"  - Total lines: {len(lines)}")
                result.append(f"  - File size: {file_path.stat().st_size} bytes")
            except Exception as e:
                result.append(f"\nCould not read file statistics: {e}")
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error analyzing file {target_file}: {str(e)}"

    def _get_modified_files(self, repo: Repo, file_pattern: str, include_untracked: bool) -> str:
        """Get list of modified files with basic information."""
        result = []
        
        # Get staged changes
        staged_files = [item.a_path for item in repo.index.diff("HEAD")]
        
        # Get unstaged changes
        unstaged_files = [item.a_path for item in repo.index.diff(None)]
        
        # Apply file pattern filter
        if file_pattern:
            import fnmatch
            staged_files = [f for f in staged_files if fnmatch.fnmatch(f, file_pattern)]
            unstaged_files = [f for f in unstaged_files if fnmatch.fnmatch(f, file_pattern)]
        
        result.append("=== GIT STATUS SUMMARY ===")
        
        if staged_files:
            result.append(f"\nStaged files ({len(staged_files)}):")
            for file_path in staged_files:
                result.append(f"  M {file_path}")
        
        if unstaged_files:
            result.append(f"\nUnstaged files ({len(unstaged_files)}):")
            for file_path in unstaged_files:
                result.append(f"  M {file_path}")
        
        if include_untracked:
            untracked_files = repo.untracked_files
            if file_pattern:
                import fnmatch
                untracked_files = [f for f in untracked_files if fnmatch.fnmatch(f, file_pattern)]
            
            if untracked_files:
                result.append(f"\nUntracked files ({len(untracked_files)}):")
                for file_path in untracked_files:
                    result.append(f"  ? {file_path}")
        
        if not staged_files and not unstaged_files and (not include_untracked or not repo.untracked_files):
            result.append("\nNo changes detected.")
        
        return "\n".join(result)

    def _get_line_diff(self, repo: Repo, target_file: str, base_commit: str) -> str:
        """Get detailed line-by-line diff for a specific file."""
        try:
            file_path = Path(repo.working_dir) / target_file
            
            if not file_path.exists():
                return f"File does not exist: {target_file}"
            
            # Get current file content
            try:
                current_content = file_path.read_text(encoding='utf-8').splitlines(keepends=True)
            except Exception as e:
                return f"Error reading current file: {e}"
            
            # Get previous version
            try:
                if base_commit == "HEAD":
                    # Get the committed version
                    previous_content = repo.git.show(f"HEAD:{target_file}").splitlines(keepends=True)
                else:
                    # Get specific commit version
                    previous_content = repo.git.show(f"{base_commit}:{target_file}").splitlines(keepends=True)
            except Exception:
                # File is new or doesn't exist in previous commit
                previous_content = []
            
            # Generate unified diff
            diff_lines = list(difflib.unified_diff(
                previous_content,
                current_content,
                fromfile=f"a/{target_file}",
                tofile=f"b/{target_file}",
                lineterm=""
            ))
            
            if not diff_lines:
                return f"No differences found in {target_file}"
            
            result = [f"Line-by-line diff for: {target_file}"]
            result.append("=" * 50)
            result.extend(diff_lines)
            
            return "\n".join(result)
            
        except Exception as e:
            return f"Error generating line diff for {target_file}: {str(e)}"