from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model
from typing import Dict, Any
from src.agents.nodes import (
    AgentState,
    planner,
    direct_chat,
    executor,
    critic,
    response_formatter,
)
from src.tools.weather_tool import get_weather
from src.tools.retrieval_tool import create_retrieval_tools
from src.utils.config import config


class ZeroHallucinationAgent:
    """
    Main LangGraph agent implementing 3-score evaluation for hallucination prevention.
    Uses Retrieval Confidence, Source Consistency, and LLM Verification.
    """

    def __init__(self):
        """Initialize the agent with knowledge base components loaded once."""
        print("🔧 Initializing Zero-Hallucination Agent...")

        # Initialize LLM with tools
        self.llm = init_chat_model(
            model=config.llm_model_name,
            api_key=config.openai_api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # Available tools (including GraphRAG retrieval tools)
        self.retrieval_tools = create_retrieval_tools()
        # Only use GraphRAG tools (local and global search), not vector search
        self.graph_retrieval_tools = [
            tool
            for tool in self.retrieval_tools
            if tool.name in ["local_search", "global_search"]
        ]
        self.tools = [get_weather] + self.graph_retrieval_tools
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create tool node
        self.tool_node = ToolNode(self.tools)

        # Build the graph
        self.graph = self._build_graph()

        print("✅ Agent initialization complete!")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(AgentState, input_schema=MessagesState)

        # Add nodes (tool-based workflow)
        workflow.add_node("planner", lambda state: planner(state, self.llm))
        workflow.add_node("direct_chat", lambda state: direct_chat(state, self.llm))
        workflow.add_node(
            "executor", lambda state: executor(state, self.llm_with_tools)
        )
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("critic", lambda state: critic(state, self.llm))
        workflow.add_node(
            "response_formatter", lambda state: response_formatter(state, self.llm)
        )

        # Add edges - define the tool-based workflow
        # Planner-router determines the flow
        workflow.add_conditional_edges(
            "planner",
            self._route_query,
            {
                "casual_chat": "direct_chat",
                "knowledge_query": "executor",
            },
        )

        # Direct chat goes straight to END
        workflow.add_edge("direct_chat", END)

        # Conditional edge from executor - check if tools need to be called
        workflow.add_conditional_edges(
            "executor",
            self._should_call_tools,
            {
                "call_tools": "tools",
                "continue": "response_formatter",
            },
        )

        # After tools, go to response_formatter to format tool outputs
        workflow.add_edge("tools", "response_formatter")

        # After response_formatter, go to critic for final evaluation
        workflow.add_edge("response_formatter", "critic")

        # Conditional edge from critic - retry or finish
        workflow.add_conditional_edges(
            "critic",
            self._should_retry,
            {
                "retry": "planner",  # Go back to planning for retry
                "finish": END,
            },
        )

        # Set entry point
        workflow.set_entry_point("planner")

        return workflow.compile()

    def _route_query(self, state: AgentState) -> str:
        """Route query based on classification from router node."""
        query_type = state.get("query_type", "knowledge_query")
        return query_type

    def _should_call_tools(self, state: AgentState) -> str:
        """Determine whether to call tools based on the last message."""
        messages = state.get("messages", [])
        if messages and hasattr(messages[-1], "tool_calls") and messages[-1].tool_calls:
            return "call_tools"
        return "continue"

    def _should_retry(self, state: AgentState) -> str:
        """Determine whether to retry or finish based on critic evaluation."""
        should_retry = state.get("evaluation", {}).get("should_retry", False)
        retry_count = state.get("retry_count", 0)

        # Retry if critic recommends it and we haven't exceeded max retries
        if should_retry and retry_count < config.max_retries:
            return "retry"
        else:
            return "finish"

    def run(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """
        Run the agent on a given query.

        Args:
            query: User input query
            thread_id: Thread ID for conversation persistence

        Returns:
            Dictionary with response and metadata
        """
        # Initialize state
        initial_state: AgentState = {
            "messages": [("user", query)],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }

        # Run the graph with conversation thread
        configuration = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke(initial_state, configuration)

        # Return structured response
        return {
            "messages": final_state.get("messages", []),
            "confidence_scores": final_state.get("evaluation", {}).get(
                "evaluations", []
            ),
            "retry_count": final_state.get("retry_count", 0),
            "tools_used": final_state.get("tools_used", []),
            "metadata": {
                "total_evaluations": len(
                    final_state.get("evaluation", {}).get("evaluations", [])
                ),
                "low_confidence_warning": final_state.get("retry_count", 0)
                >= config.max_retries,
                "graph_facts_used": len(
                    [
                        e
                        for e in final_state.get("evaluation", {}).get(
                            "evaluations", []
                        )
                        if e.get("extracted_facts")
                    ]
                ),
            },
        }

    async def arun(self, query: str, thread_id: str = "default") -> Dict[str, Any]:
        """Async version of run method."""
        # Initialize state
        initial_state: AgentState = {
            "messages": [("user", query)],
            "query_type": "",
            "retry_count": 0,
            "evaluation": {},
            "tools_used": [],
        }

        # Run the graph asynchronously with conversation thread
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await self.graph.ainvoke(initial_state, config)

        # Return structured response
        return {
            "messages": final_state.get("messages", []),
            "confidence_scores": final_state.get("evaluation", {}).get(
                "evaluations", []
            ),
            "retry_count": final_state.get("retry_count", 0),
            "tools_used": final_state.get("tools_used", []),
            "metadata": {
                "total_evaluations": len(
                    final_state.get("evaluation", {}).get("evaluations", [])
                ),
                "low_confidence_warning": final_state.get("retry_count", 0)
                >= config.max_retries,
                "graph_facts_used": len(
                    [
                        e
                        for e in final_state.get("evaluation", {}).get(
                            "evaluations", []
                        )
                        if e.get("extracted_facts")
                    ]
                ),
            },
        }


# Factory function for easy instantiation
def create_agent() -> ZeroHallucinationAgent:
    """Create and return a new ZeroHallucinationAgent instance."""
    return ZeroHallucinationAgent()


graph = create_agent().graph
