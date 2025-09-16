# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Code Review Agent for detecting breaking changes and providing review feedback."""

import os
from typing import override

from trae_agent.agent.base_agent import BaseAgent
from trae_agent.prompt.code_review_prompt import CODE_REVIEW_SYSTEM_PROMPT, CODE_REVIEW_TASK_PROMPT
from trae_agent.tools.base import ToolResult
from trae_agent.utils.config import AgentConfig
from trae_agent.utils.llm_clients.llm_basics import LLMMessage, LLMResponse

# Tool names specific to code review
CodeReviewToolNames = [
    "git_diff_tool",
    "breaking_change_analyzer", 
    "code_analysis_tool",
    "str_replace_based_edit_tool",
    "sequentialthinking",
    "task_done",
    "bash",
]


class CodeReviewAgent(BaseAgent):
    """Specialized agent for code review and breaking change detection."""

    def __init__(
        self,
        agent_config: AgentConfig,
        docker_config: dict | None = None,
        docker_keep: bool = True,
    ):
        """Initialize CodeReviewAgent.

        Args:
            agent_config: Configuration for the agent
            docker_config: Optional Docker configuration
            docker_keep: Whether to keep Docker container after completion
        """
        super().__init__(agent_config, docker_config, docker_keep)
        
        # Code review specific configuration
        self.analysis_scope: str = "all"  # all, functions, classes, etc.
        self.focus_areas: list[str] = ["breaking_changes", "compatibility", "api_stability"]
        self.risk_threshold: str = "medium"  # low, medium, high
        self.include_suggestions: bool = True
        self.project_path: str = ""

    @override
    def new_task(
        self,
        task: str,
        extra_args: dict[str, str] | None = None,
        tool_names: list[str] | None = None,
    ):
        """Create a new code review task."""
        self._task: str = task

        # Use code review specific tools if not provided
        if tool_names is None:
            tool_names = CodeReviewToolNames

        # Initialize tools if not already done
        if len(self._tools) == 0:
            provider = self._model_config.model_provider.provider
            from trae_agent.tools import tools_registry
            self._tools = [
                tools_registry[tool_name](model_provider=provider) for tool_name in tool_names
            ]

        self._initial_messages: list[LLMMessage] = []
        self._initial_messages.append(LLMMessage(role="system", content=self.get_system_prompt()))

        # Process extra arguments for code review
        if not extra_args:
            raise ValueError("Project path is required for code review.")
        
        if "project_path" not in extra_args:
            raise ValueError("Project path is required")

        self.project_path = extra_args.get("project_path", "")
        
        # Code review specific parameters
        self.analysis_scope = extra_args.get("analysis_scope", "all")
        self.focus_areas = extra_args.get("focus_areas", "breaking_changes,compatibility").split(",")
        self.risk_threshold = extra_args.get("risk_threshold", "medium")
        self.include_suggestions = extra_args.get("include_suggestions", "true").lower() == "true"

        # Format the task prompt
        user_message = CODE_REVIEW_TASK_PROMPT.format(
            repository_path=self.project_path,
            analysis_scope=self.analysis_scope,
            focus_areas=", ".join(self.focus_areas)
        )

        if self.docker_manager and self.docker_manager.container_workspace:
            user_message = user_message.replace(self.project_path, self.docker_manager.container_workspace)

        # Add custom task description if provided
        if task and task.strip():
            user_message += f"\n\nAdditional Instructions:\n{task}"

        self._initial_messages.append(LLMMessage(role="user", content=user_message))

        # If trajectory recorder is set, start recording
        if self._trajectory_recorder:
            self._trajectory_recorder.start_recording(
                task=f"Code Review: {task}" if task else "Code Review Analysis",
                provider=self._llm_client.provider.value,
                model=self._model_config.model,
                max_steps=self._max_steps,
            )

    def get_system_prompt(self) -> str:
        """Get the system prompt for CodeReviewAgent."""
        return CODE_REVIEW_SYSTEM_PROMPT

    @override
    def reflect_on_result(self, tool_results: list[ToolResult]) -> str | None:
        """Reflect on tool results and provide guidance for code review."""
        if not tool_results:
            return None

        # Count tool usage for reflection
        tool_usage = {}
        errors = []
        
        for result in tool_results:
            tool_usage[result.name] = tool_usage.get(result.name, 0) + 1
            if not result.success and result.error:
                errors.append(f"{result.name}: {result.error}")

        reflection_parts = []

        # Check if core code review tools were used
        if "git_diff_tool" not in tool_usage:
            reflection_parts.append(
                "Consider using git_diff_tool to analyze repository changes first."
            )

        if "breaking_change_analyzer" not in tool_usage:
            reflection_parts.append(
                "Use breaking_change_analyzer to detect potential breaking changes in the code."
            )

        # Check for errors
        if errors:
            reflection_parts.append(
                f"Address the following tool errors: {'; '.join(errors[:3])}"
            )

        # Provide guidance based on analysis completeness
        if len(tool_usage) < 2:
            reflection_parts.append(
                "Ensure comprehensive analysis by using multiple code review tools."
            )

        if reflection_parts:
            return "Review Analysis Guidance: " + " ".join(reflection_parts)

        return None

    @override
    def llm_indicates_task_completed(self, llm_response: LLMResponse) -> bool:
        """Check if the LLM indicates that the task is completed."""
        if llm_response.tool_calls is None:
            return False
        return any(tool_call.name == "task_done" for tool_call in llm_response.tool_calls)

    @override
    def _is_task_completed(self, llm_response: LLMResponse) -> bool:
        """Enhanced task completion detection for code review."""
        # For code review, we consider the task complete when analysis is done
        # and a report has been generated
        return True

    @override
    def task_incomplete_message(self) -> str:
        """Return a message indicating that the code review task is incomplete."""
        return "Code review analysis is incomplete. Please ensure all repository changes are analyzed and a comprehensive report is generated."

    @override
    async def cleanup_mcp_clients(self) -> None:
        """Clean up MCP clients (not used in code review agent)."""
        pass

    def analyze_breaking_changes(self, repository_path: str) -> dict:
        """Analyze breaking changes in the repository.
        
        This is a helper method that can be called programmatically.
        """
        return {
            "repository_path": repository_path,
            "analysis_scope": self.analysis_scope,
            "focus_areas": self.focus_areas,
            "status": "analysis_required",
            "message": "Use the execute_task method to perform the full analysis"
        }

    def generate_review_report(self, analysis_results: dict) -> str:
        """Generate a formatted review report from analysis results."""
        report_lines = [
            "# Code Review Report",
            "=" * 50,
            "",
            f"**Repository:** {analysis_results.get('repository_path', 'Unknown')}",
            f"**Analysis Scope:** {self.analysis_scope}",
            f"**Focus Areas:** {', '.join(self.focus_areas)}",
            f"**Risk Threshold:** {self.risk_threshold}",
            "",
        ]

        # Add analysis results
        if "breaking_changes" in analysis_results:
            changes = analysis_results["breaking_changes"]
            report_lines.extend([
                "## Breaking Changes Analysis",
                "",
                f"Total breaking changes detected: {len(changes)}",
                ""
            ])

            # Group by risk level
            high_risk = [c for c in changes if c.get("risk_level") == "high"]
            medium_risk = [c for c in changes if c.get("risk_level") == "medium"]
            low_risk = [c for c in changes if c.get("risk_level") == "low"]

            if high_risk:
                report_lines.extend(["### 🚨 High Risk Changes", ""])
                for change in high_risk:
                    report_lines.append(f"- {change.get('description', 'Unknown change')}")
                report_lines.append("")

            if medium_risk:
                report_lines.extend(["### ⚠️ Medium Risk Changes", ""])
                for change in medium_risk:
                    report_lines.append(f"- {change.get('description', 'Unknown change')}")
                report_lines.append("")

            if low_risk:
                report_lines.extend(["### ℹ️ Low Risk Changes", ""])
                for change in low_risk:
                    report_lines.append(f"- {change.get('description', 'Unknown change')}")
                report_lines.append("")

        # Add recommendations if enabled
        if self.include_suggestions and "recommendations" in analysis_results:
            report_lines.extend([
                "## Recommendations",
                ""
            ])
            for rec in analysis_results["recommendations"]:
                report_lines.append(f"- {rec}")
            report_lines.append("")

        # Add compatibility assessment
        if "compatibility" in analysis_results:
            compatibility = analysis_results["compatibility"]
            report_lines.extend([
                "## Compatibility Assessment",
                "",
                f"**Status:** {compatibility.get('status', 'Unknown')}",
                f"**Impact:** {compatibility.get('impact', 'Not assessed')}",
                ""
            ])

        report_lines.extend([
            "---",
            "*Report generated by CodeReviewAgent*"
        ])

        return "\n".join(report_lines)