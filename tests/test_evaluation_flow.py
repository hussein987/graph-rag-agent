"""
Test the evaluation flow to ensure critic evaluates final responses correctly.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.graph_agent import ZeroHallucinationAgent
from src.agents.nodes import AgentState, planner, executor, critic, response_formatter
from src.evaluation.verification import VerificationResult
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json


class TestEvaluationFlow:
    """Test the complete evaluation flow with state logging."""

    def test_retrieval_query_flow(self):
        """Test a complete retrieval query flow with state logging."""

        # Mock the LLM and tools for testing
        mock_llm = Mock()
        mock_llm_with_tools = Mock()

        # Mock planner response
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = Mock(type="knowledge_query")
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock executor response with tool call
        mock_tool_response = Mock()
        mock_tool_response.content = ""  # Empty content for tool call
        mock_tool_response.tool_calls = [
            {"name": "retrieve_knowledge", "args": {"query": "test query"}}
        ]
        mock_llm_with_tools.invoke.return_value = mock_tool_response

        # Test planner
        initial_state = {
            "messages": [HumanMessage(content="What is the capital of Finland?")],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }


        # Test planner
        planner_result = planner(initial_state, mock_llm)
        initial_state.update(planner_result)


        # Test executor
        executor_result = executor(initial_state, mock_llm_with_tools)
        initial_state.update(executor_result)


        # Simulate tool execution by adding tool message
        tool_message = ToolMessage(
            content="Helsinki is the capital of Finland. It's located in southern Finland...",
            tool_call_id="test_call_id",
        )
        initial_state["messages"].append(tool_message)


        # Test response formatter
        mock_llm.invoke.return_value = Mock(
            content="Helsinki is the capital of Finland, located in the southern part of the country."
        )

        formatter_result = response_formatter(initial_state, mock_llm)
        # Simulate LangGraph behavior: append messages, don't replace
        if formatter_result.get("messages"):
            initial_state["messages"].extend(formatter_result["messages"])
        initial_state.update(
            {k: v for k, v in formatter_result.items() if k != "messages"}
        )


        # Test critic
        # Mock the structured LLM output
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = VerificationResult(
            reflects_context=True,
            addresses_query=True,
            is_clarification_question=False,
            confidence=0.85,
            issues=[],
            rag_score=0.8,
            consistency_score=0.9,
            should_retry=False,
            retry_reason="",
        )

        # Mock the with_structured_output method
        mock_llm.with_structured_output.return_value = mock_structured_llm

        critic_result = critic(initial_state, mock_llm)
        initial_state.update(critic_result)


        # Verify the structured LLM was called with messages
        mock_structured_llm.invoke.assert_called_once()
        call_args = mock_structured_llm.invoke.call_args[0][
            0
        ]  # Get first positional argument


        # Assertions
        assert len(call_args) > 1  # Should have system message + conversation messages
        assert (
            "Helsinki" in call_args[-1].content
        )  # Final response should contain Helsinki
        assert initial_state["evaluation"]["should_retry"] == False
        assert len(initial_state["evaluation"]["evaluations"]) == 1
        assert initial_state["evaluation"]["verification_result"].confidence == 0.85

    def test_weather_query_no_evaluation(self):
        """Test weather query doesn't trigger evaluation."""

        mock_llm = Mock()
        mock_llm_with_tools = Mock()

        # Mock planner response
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = Mock(type="knowledge_query")
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock executor response with weather tool call
        mock_tool_response = Mock()
        mock_tool_response.content = ""
        mock_tool_response.tool_calls = [
            {"name": "get_weather", "args": {"location": "Helsinki"}}
        ]
        mock_llm_with_tools.invoke.return_value = mock_tool_response

        initial_state = {
            "messages": [HumanMessage(content="What's the weather in Helsinki?")],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }

        # Execute flow
        planner_result = planner(initial_state, mock_llm)
        initial_state.update(planner_result)

        executor_result = executor(initial_state, mock_llm_with_tools)
        initial_state.update(executor_result)

        # Simulate weather tool execution
        tool_message = ToolMessage(
            content="Temperature: 22°C, Sunny, Wind: 5 km/h",
            tool_call_id="weather_call_id",
        )
        initial_state["messages"].append(tool_message)

        # Response formatter
        mock_llm.invoke.return_value = Mock(
            content="The current weather in Helsinki is 22°C and sunny with light winds."
        )
        formatter_result = response_formatter(initial_state, mock_llm)
        # Simulate LangGraph behavior: append messages, don't replace
        if formatter_result.get("messages"):
            initial_state["messages"].extend(formatter_result["messages"])
        initial_state.update(
            {k: v for k, v in formatter_result.items() if k != "messages"}
        )

        # Critic should skip evaluation
        critic_result = critic(initial_state, mock_llm)
        initial_state.update(critic_result)


        # Assertions
        assert "get_weather" in initial_state["tools_used"]
        assert "retrieve_knowledge" not in initial_state["tools_used"]
        assert initial_state["evaluation"]["should_retry"] == False
        assert len(initial_state["evaluation"]["evaluations"]) == 0

    def test_casual_chat_no_evaluation(self):
        """Test casual chat doesn't trigger evaluation."""

        mock_llm = Mock()

        # Mock structured output for casual chat
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = Mock(type="casual_chat")
        mock_llm.with_structured_output.return_value = mock_structured_llm

        initial_state = {
            "messages": [HumanMessage(content="Hello there!")],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }

        # Execute planner
        planner_result = planner(initial_state, mock_llm)
        initial_state.update(planner_result)


        # Assertions
        assert initial_state["query_type"] == "casual_chat"
        # Casual chat should go directly to direct_chat, not through evaluation

    def test_low_confidence_retry_flow(self):
        """Test low confidence triggers retry."""

        mock_llm = Mock()
        mock_llm_with_tools = Mock()

        # Mock planner response
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = Mock(type="knowledge_query")
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock executor response with retrieval tool call
        mock_tool_response = Mock()
        mock_tool_response.content = ""
        mock_tool_response.tool_calls = [
            {"name": "retrieve_knowledge", "args": {"query": "test query"}}
        ]
        mock_llm_with_tools.invoke.return_value = mock_tool_response

        initial_state = {
            "messages": [HumanMessage(content="What is quantum computing?")],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }

        # Execute planner and executor
        planner_result = planner(initial_state, mock_llm)
        initial_state.update(planner_result)

        executor_result = executor(initial_state, mock_llm_with_tools)
        initial_state.update(executor_result)

        # Add tool message
        tool_message = ToolMessage(
            content="Quantum computing uses quantum bits...",
            tool_call_id="test_call_id",
        )
        initial_state["messages"].append(tool_message)

        # Response formatter
        mock_llm.invoke.return_value = Mock(
            content="Quantum computing is a complex field that I'm not sure about."
        )
        formatter_result = response_formatter(initial_state, mock_llm)
        # Simulate LangGraph behavior: append messages, don't replace
        if formatter_result.get("messages"):
            initial_state["messages"].extend(formatter_result["messages"])
        initial_state.update(
            {k: v for k, v in formatter_result.items() if k != "messages"}
        )

        # Critic with low confidence
        # Mock the structured LLM output for low confidence
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = VerificationResult(
            reflects_context=False,
            addresses_query=False,
            is_clarification_question=False,
            confidence=0.2,
            issues=["Response seems uncertain", "Context doesn't match query well"],
            rag_score=0.3,
            consistency_score=0.4,
            should_retry=True,
            retry_reason="response_poor",
        )

        # Mock the with_structured_output method
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Add an ID to the AI message so RemoveMessage can work
        final_ai_message = initial_state["messages"][-1]
        final_ai_message.id = "test_ai_message_id"

        critic_result = critic(initial_state, mock_llm)


        # Assertions
        assert critic_result["evaluation"]["should_retry"] == True
        assert critic_result["retry_count"] == 1
        assert critic_result["evaluation"]["verification_result"].confidence == 0.2
        assert (
            critic_result["evaluation"]["verification_result"].retry_reason
            == "response_poor"
        )

        # Check that RemoveMessage was created
        if critic_result.get("messages"):
            from langchain_core.messages import RemoveMessage

            assert len(critic_result["messages"]) == 1
            assert isinstance(critic_result["messages"][0], RemoveMessage)
            assert critic_result["messages"][0].id == "test_ai_message_id"

    def test_clarification_question_no_retry(self):
        """Test clarification questions are not retried."""

        mock_llm = Mock()
        mock_llm_with_tools = Mock()

        # Mock planner response
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = Mock(type="knowledge_query")
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock executor response with retrieval tool call
        mock_tool_response = Mock()
        mock_tool_response.content = ""
        mock_tool_response.tool_calls = [
            {"name": "retrieve_knowledge", "args": {"query": "test query"}}
        ]
        mock_llm_with_tools.invoke.return_value = mock_tool_response

        initial_state = {
            "messages": [HumanMessage(content="Tell me about weather in Finland")],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }

        # Execute planner and executor
        planner_result = planner(initial_state, mock_llm)
        initial_state.update(planner_result)

        executor_result = executor(initial_state, mock_llm_with_tools)
        initial_state.update(executor_result)

        # Add tool message
        tool_message = ToolMessage(
            content="Weather information for Finland requires specific location...",
            tool_call_id="test_call_id",
        )
        initial_state["messages"].append(tool_message)

        # Response formatter with clarification question
        mock_llm.invoke.return_value = Mock(
            content="Could you please specify which city in Finland you would like to know the weather for?"
        )
        formatter_result = response_formatter(initial_state, mock_llm)
        if formatter_result.get("messages"):
            initial_state["messages"].extend(formatter_result["messages"])
        initial_state.update(
            {k: v for k, v in formatter_result.items() if k != "messages"}
        )

        # Critic recognizes this as a clarification question (should not retry)
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = VerificationResult(
            reflects_context=True,
            addresses_query=True,
            is_clarification_question=True,
            confidence=0.9,  # High confidence for clarification questions
            issues=[],
            rag_score=0.7,
            consistency_score=0.8,
            should_retry=False,  # Should not retry clarification questions
            retry_reason="",
        )

        mock_llm.with_structured_output.return_value = mock_structured_llm

        critic_result = critic(initial_state, mock_llm)


        # Assertions
        assert (
            critic_result["evaluation"]["verification_result"].is_clarification_question
            == True
        )
        assert critic_result["evaluation"]["should_retry"] == False
        assert critic_result["evaluation"]["verification_result"].confidence == 0.9
        assert critic_result.get("messages") is None  # No messages should be removed

    def test_max_retries_no_message_removal(self):
        """Test max retries reached doesn't remove messages."""

        mock_llm = Mock()
        mock_llm_with_tools = Mock()

        # Mock planner response
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = Mock(type="knowledge_query")
        mock_llm.with_structured_output.return_value = mock_structured_llm

        # Mock executor response with retrieval tool call
        mock_tool_response = Mock()
        mock_tool_response.content = ""
        mock_tool_response.tool_calls = [
            {"name": "retrieve_knowledge", "args": {"query": "test query"}}
        ]
        mock_llm_with_tools.invoke.return_value = mock_tool_response

        initial_state = {
            "messages": [HumanMessage(content="What is quantum computing?")],
            "query_type": "",
            "retry_count": 2,  # Already at max retries
            "evaluation": {},
            "tools_used": [],
        }

        # Execute planner and executor
        planner_result = planner(initial_state, mock_llm)
        initial_state.update(planner_result)

        executor_result = executor(initial_state, mock_llm_with_tools)
        initial_state.update(executor_result)

        # Add tool message
        tool_message = ToolMessage(
            content="Quantum computing uses quantum bits...",
            tool_call_id="test_call_id",
        )
        initial_state["messages"].append(tool_message)

        # Response formatter
        mock_llm.invoke.return_value = Mock(
            content="Quantum computing is a complex field..."
        )
        formatter_result = response_formatter(initial_state, mock_llm)
        if formatter_result.get("messages"):
            initial_state["messages"].extend(formatter_result["messages"])
        initial_state.update(
            {k: v for k, v in formatter_result.items() if k != "messages"}
        )

        # Critic with low confidence but max retries reached
        mock_structured_llm = Mock()
        mock_structured_llm.invoke.return_value = VerificationResult(
            reflects_context=False,
            addresses_query=False,
            is_clarification_question=False,
            confidence=0.2,
            issues=["Response seems uncertain"],
            rag_score=0.3,
            consistency_score=0.4,
            should_retry=True,
            retry_reason="response_poor",
        )

        mock_llm.with_structured_output.return_value = mock_structured_llm

        critic_result = critic(initial_state, mock_llm)


        # Assertions
        assert (
            critic_result["evaluation"]["should_retry"] == True
        )  # Would retry but can't
        assert initial_state["retry_count"] == 2  # At max retries
        assert (
            critic_result.get("messages") is None
        )  # No messages should be removed when max retries reached


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
