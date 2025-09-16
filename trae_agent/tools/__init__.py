# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tools module for Trae Agent."""

from trae_agent.tools.base import Tool, ToolCall, ToolExecutor, ToolResult
from trae_agent.tools.bash_tool import BashTool
from trae_agent.tools.breaking_change_analyzer import BreakingChangeAnalyzer
from trae_agent.tools.ckg_tool import CKGTool
from trae_agent.tools.code_analysis_tool import CodeAnalysisTool
from trae_agent.tools.edit_tool import TextEditorTool
from trae_agent.tools.git_diff_tool import GitDiffTool
from trae_agent.tools.json_edit_tool import JSONEditTool
from trae_agent.tools.sequential_thinking_tool import SequentialThinkingTool
from trae_agent.tools.task_done_tool import TaskDoneTool

__all__ = [
    "Tool",
    "ToolResult",
    "ToolCall",
    "ToolExecutor",
    "BashTool",
    "BreakingChangeAnalyzer",
    "CodeAnalysisTool",
    "GitDiffTool",
    "TextEditorTool",
    "JSONEditTool",
    "SequentialThinkingTool",
    "TaskDoneTool",
    "CKGTool",
]

tools_registry: dict[str, type[Tool]] = {
    "bash": BashTool,
    "str_replace_based_edit_tool": TextEditorTool,
    "json_edit_tool": JSONEditTool,
    "sequentialthinking": SequentialThinkingTool,
    "task_done": TaskDoneTool,
    "ckg": CKGTool,
    "git_diff_tool": GitDiffTool,
    "breaking_change_analyzer": BreakingChangeAnalyzer,
    "code_analysis_tool": CodeAnalysisTool,
}
