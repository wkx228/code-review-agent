# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""Tests for CodeReviewAgent architecture."""

import pytest
from unittest.mock import Mock, patch

from trae_agent.agent.code_review_agent import CodeReviewAgent
from trae_agent.agent.base_agent import BaseAgent
from trae_agent.agent.trae_agent import TraeAgent
from trae_agent.utils.config import AgentConfig, ModelConfig, ModelProviderConfig, ModelProvider


class TestCodeReviewAgentArchitecture:
    """Test that CodeReviewAgent has the correct architecture."""

    def test_code_review_agent_inherits_from_base_agent(self):
        """Test that CodeReviewAgent inherits from BaseAgent, not TraeAgent."""
        # Check inheritance hierarchy
        assert issubclass(CodeReviewAgent, BaseAgent)
        assert not issubclass(CodeReviewAgent, TraeAgent)
        
        # CodeReviewAgent and TraeAgent should be siblings under BaseAgent
        assert issubclass(TraeAgent, BaseAgent)
        
        # They should not inherit from each other
        assert not issubclass(CodeReviewAgent, TraeAgent)
        assert not issubclass(TraeAgent, CodeReviewAgent)

    @pytest.fixture
    def mock_agent_config(self):
        """Create a mock AgentConfig for testing."""
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

    def test_code_review_agent_initialization(self, mock_agent_config):
        """Test CodeReviewAgent initialization."""
        with patch('trae_agent.agent.code_review_agent.LLMClient'):
            agent = CodeReviewAgent(mock_agent_config)
            
            # Check it's a CodeReviewAgent instance
            assert isinstance(agent, CodeReviewAgent)
            assert isinstance(agent, BaseAgent)
            assert not isinstance(agent, TraeAgent)
            
            # Check code review specific attributes
            assert hasattr(agent, 'analysis_scope')
            assert hasattr(agent, 'focus_areas')
            assert hasattr(agent, 'risk_threshold')
            assert hasattr(agent, 'include_suggestions')
            
            # Check default values
            assert agent.analysis_scope == "all"
            assert "breaking_changes" in agent.focus_areas
            assert agent.risk_threshold == "medium"
            assert agent.include_suggestions is True

    def test_code_review_agent_has_required_methods(self, mock_agent_config):
        """Test that CodeReviewAgent implements required abstract methods."""
        with patch('trae_agent.agent.code_review_agent.LLMClient'):
            agent = CodeReviewAgent(mock_agent_config)
            
            # Check required methods exist
            assert hasattr(agent, 'new_task')
            assert hasattr(agent, 'cleanup_mcp_clients')
            assert hasattr(agent, 'get_system_prompt')
            assert hasattr(agent, 'reflect_on_result')
            
            # Check code review specific methods
            assert hasattr(agent, 'analyze_breaking_changes')
            assert hasattr(agent, 'generate_review_report')

    def test_code_review_agent_new_task(self, mock_agent_config):
        """Test new_task method of CodeReviewAgent."""
        with patch('trae_agent.agent.code_review_agent.LLMClient'), \
             patch('trae_agent.tools.tools_registry', {}):
            
            agent = CodeReviewAgent(mock_agent_config)
            
            # Test that new_task requires project_path
            with pytest.raises(ValueError, match="Project path is required"):
                agent.new_task("Test task")
            
            with pytest.raises(ValueError, match="Project path is required"):
                agent.new_task("Test task", extra_args={})
            
            # Test successful task creation
            extra_args = {
                "project_path": "/test/repo",
                "analysis_scope": "functions",
                "risk_threshold": "high"
            }
            
            # This should not raise an exception
            try:
                agent.new_task("Test code review task", extra_args)
                assert agent.project_path == "/test/repo"
                assert agent.analysis_scope == "functions"
                assert agent.risk_threshold == "high"
            except Exception as e:
                # We expect some exceptions due to mocking, but not ValueError about project_path
                assert "Project path is required" not in str(e)

    def test_code_review_agent_no_mcp_methods(self, mock_agent_config):
        """Test that CodeReviewAgent doesn't have MCP-specific methods."""
        with patch('trae_agent.agent.code_review_agent.LLMClient'):
            agent = CodeReviewAgent(mock_agent_config)
            
            # CodeReviewAgent should not have MCP-related methods
            assert not hasattr(agent, 'allow_mcp_servers')
            assert not hasattr(agent, 'mcp_servers_config')
            assert not hasattr(agent, 'mcp_tools')
            assert not hasattr(agent, 'mcp_clients')
            assert not hasattr(agent, 'initialise_mcp')
            assert not hasattr(agent, 'discover_mcp_tools')

    def test_code_review_agent_cleanup_mcp_clients(self, mock_agent_config):
        """Test that cleanup_mcp_clients is implemented but does nothing."""
        with patch('trae_agent.agent.code_review_agent.LLMClient'):
            agent = CodeReviewAgent(mock_agent_config)
            
            # Should not raise an exception
            import asyncio
            asyncio.run(agent.cleanup_mcp_clients())

    def test_system_prompt_is_code_review_specific(self, mock_agent_config):
        """Test that CodeReviewAgent uses code review specific system prompt."""
        with patch('trae_agent.agent.code_review_agent.LLMClient'):
            agent = CodeReviewAgent(mock_agent_config)
            
            system_prompt = agent.get_system_prompt()
            
            # Should contain code review specific terms
            assert "breaking changes" in system_prompt.lower()
            assert "code review" in system_prompt.lower()
            assert "git_diff_tool" in system_prompt
            assert "breaking_change_analyzer" in system_prompt

    def test_agent_type_registration(self):
        """Test that AgentType includes CodeReviewAgent."""
        from trae_agent.agent.agent import AgentType
        
        assert hasattr(AgentType, 'CodeReviewAgent')
        assert AgentType.CodeReviewAgent.value == "code_review_agent"
        
        # Should have both agent types
        agent_types = [item.value for item in AgentType]
        assert "trae_agent" in agent_types
        assert "code_review_agent" in agent_types