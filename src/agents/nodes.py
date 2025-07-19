"""
Agent node implementations for the Zero-Hallucination AI Agent.

This module contains all the individual node functions that make up the LangGraph workflow,
including routing, planning, retrieval, execution, and response formatting.
"""

from typing import Dict, Any, List
from langgraph.graph import MessagesState
from langchain_core.messages import AIMessage, RemoveMessage
from src.utils.config import config, get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class QueryClassification(BaseModel):
    """Structured output for query classification"""

    type: str = Field(
        description="Query type: 'casual_chat' or 'knowledge_query'",
        enum=["casual_chat", "knowledge_query"],
    )


def _remove_tool_call_sequence(messages: List, tool_name: str) -> List[RemoveMessage]:
    """
    Remove a tool call sequence (AI tool call + tool message + AI response) for retry.

    Args:
        messages: List of conversation messages
        tool_name: Name of the tool call to remove sequence for

    Returns:
        List of RemoveMessage objects to remove the sequence
    """
    from langchain_core.messages import AIMessage, ToolMessage

    messages_to_remove = []

    # Find the tool call sequence in reverse order
    for i in reversed(range(len(messages))):
        msg = messages[i]

        # Look for AI message with tool calls for the specified tool
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                if tool_call.get("name") == tool_name:
                    # Found the AI tool call message
                    if msg.id:
                        messages_to_remove.append(RemoveMessage(id=msg.id))

                    # Look for the corresponding tool message
                    for j in range(i + 1, len(messages)):
                        tool_msg = messages[j]
                        if isinstance(
                            tool_msg, ToolMessage
                        ) and tool_msg.tool_call_id == tool_call.get("id"):
                            if tool_msg.id:
                                messages_to_remove.append(RemoveMessage(id=tool_msg.id))

                            # Look for the AI response after the tool message
                            for k in range(j + 1, len(messages)):
                                response_msg = messages[k]
                                if (
                                    isinstance(response_msg, AIMessage)
                                    and response_msg.content
                                ):
                                    if response_msg.id:
                                        messages_to_remove.append(
                                            RemoveMessage(id=response_msg.id)
                                        )
                                    break
                            break
                    break

    return messages_to_remove


class AgentState(MessagesState):
    """
    State object passed between LangGraph nodes.

    Inherits from MessagesState to provide conversation history management.

    Attributes:
        query_type: Classification of the query ('casual_chat' or 'knowledge_query')
        retry_count: Number of retry attempts for the current query
        evaluation: Results from the critic evaluation
        tools_used: List of tools that were used during processing
    """

    # `messages` is inherited from MessagesState
    query_type: str  # 'casual_chat' or 'knowledge_query'
    retry_count: int
    evaluation: Dict[str, Any]
    tools_used: List[str]


def planner(state: AgentState, llm) -> Dict[str, Any]:
    """
    Combined planner and router that classifies queries and creates appropriate plans.
    Routes to direct_chat for casual conversation or creates knowledge tasks.

    Args:
        state: Current agent state
        llm: Language model for planning and routing

    Returns:
        Dictionary with query_type and plan (if knowledge_query)
    """
    logger.info("Starting planner node")
    try:
        from langchain_core.messages import SystemMessage

        system_msg = SystemMessage(
            content="""
        You are an intelligent agent planner and router. Analyze the user's message and classify it:
        
        1. 'casual_chat': Greetings, small talk, simple questions that don't require specific knowledge retrieval
        2. 'knowledge_query': Questions that require specific information, facts, or detailed analysis
        
        Examples:
        - "Hi there!" → casual_chat
        - "Thanks!" → casual_chat
        - "What's the weather in Helsinki?" → knowledge_query
        - "Tell me about Finnish culture" → knowledge_query
        
        Consider the full conversation history for context.
        """
        )

        # Include conversation history for context
        messages = [system_msg] + state["messages"]

        # Use structured output to avoid generating AI messages directly
        structured_llm = llm.with_structured_output(QueryClassification)
        result = structured_llm.invoke(messages)

        if result.type == "casual_chat":
            logger.info("Query classified as casual_chat")
            return {"query_type": "casual_chat"}
        elif result.type == "knowledge_query":
            logger.info("Query classified as knowledge_query")
            return {"query_type": "knowledge_query"}
        else:
            logger.warning("Unknown query type, defaulting to knowledge_query")
            return {"query_type": "knowledge_query"}

    except Exception as e:
        logger.error(f"Error in planner node: {e}, falling back to knowledge_query")
        return {"query_type": "knowledge_query"}


def direct_chat(state: AgentState, llm) -> Dict[str, Any]:
    """
    Handle casual conversations directly without knowledge retrieval.

    Args:
        state: Current agent state
        llm: Language model for response generation

    Returns:
        Dictionary with AI message response
    """
    logger.info("Starting direct_chat node")
    try:
        from langchain_core.messages import SystemMessage

        system_msg = SystemMessage(
            content="""
        You are a helpful AI assistant. Respond to the user's message in a friendly, conversational way.
        Keep your response concise and natural. You have access to the full conversation history,
        so you can refer to previous messages when appropriate.
        """
        )

        # Include full conversation history for context
        messages = [system_msg] + state["messages"]
        response = llm.invoke(messages)
        ai_message = AIMessage(content=response.content)
        logger.info("Direct chat response generated successfully")
        return {"messages": [ai_message]}
    except Exception as e:
        logger.error(f"Error in direct_chat node: {e}, using fallback response")
        ai_message = AIMessage(content="I'm here to help! How can I assist you today?")
        return {"messages": [ai_message]}


def executor(state: AgentState, llm_with_tools) -> Dict[str, Any]:
    """
    Execute the user's query using LLM with available tools (retrieval, weather, etc.).
    The LLM will automatically call appropriate tools based on the conversation.
    """
    logger.info("Starting executor node")
    try:
        from langchain_core.messages import SystemMessage

        # Check if there's a previous evaluation result to learn from
        previous_evaluation = state.get("previous_evaluation")

        base_content = """
            You are a helpful AI assistant with access to powerful GraphRAG tools:
            
            1. local_search: GraphRAG local search for entity-focused, detailed responses with graph traversal
            2. global_search: GraphRAG global search for high-level summaries across the entire knowledge base
            3. get_weather: Get current weather information for any location
            
            Use these tools when needed to provide accurate, helpful responses. Consider the full conversation 
            history to understand context and provide relevant information.
            
            TOOL SELECTION GUIDANCE:
            - Use local_search for specific questions about particular entities, people, places, or detailed information
            - Use global_search for broad questions requiring summaries across multiple topics or high-level overviews
            - Use get_weather for current weather information
            - You can use multiple tools in sequence if needed
            
            Examples:
            - "Tell me about John Smith" → local_search (specific entity)
            - "What are the main themes in the documents?" → global_search (broad summary)
            - "Weather in Helsinki" → get_weather
            """

        # Add previous evaluation context if available
        if previous_evaluation:
            retry_guidance = f"""
            
            IMPORTANT - PREVIOUS ATTEMPT FEEDBACK:
            This is a retry attempt. The previous response was evaluated as needing improvement:
            - Confidence: {previous_evaluation.get('confidence', 'N/A')}
            - Issues: {', '.join(previous_evaluation.get('issues', []))}
            - RAG Score: {previous_evaluation.get('rag_score', 'N/A')}
            - Consistency Score: {previous_evaluation.get('consistency_score', 'N/A')}
            - Retry Reason: {previous_evaluation.get('retry_reason', 'N/A')}
            
            Based on this feedback, please:
            1. Address the specific issues mentioned above
            2. If retry_reason is "context_insufficient", try different search terms or broader queries
            3. If retry_reason is "response_poor", improve the response quality and clarity
            4. Ensure your response is more accurate and comprehensive
            """
            base_content += retry_guidance

        system_msg = SystemMessage(content=base_content)

        # Include conversation history for context
        messages = [system_msg] + state["messages"]
        response = llm_with_tools.invoke(messages)

        # Always add the response to messages (including tool calls)
        state["messages"].append(response)

        # Track tool usage
        tools_used = state.get("tools_used", [])

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                tools_used.append(tool_call["name"])
                logger.info(f"Tool called: {tool_call['name']}")

        return {"tools_used": tools_used}

    except Exception as e:
        logger.error(f"Error in executor node: {e}")
        return {"tools_used": state.get("tools_used", [])}


def critic(state: AgentState, llm) -> Dict[str, Any]:
    """
    Evaluate final responses using LLM verification only.
    This runs AFTER tools have been called and responses are complete.
    """
    logger.info("Starting critic node")
    from src.evaluation.verification import VerificationResult
    from langchain_core.messages import SystemMessage, AIMessage

    # Check if retrieval was used
    tools_used = state.get("tools_used", [])
    retrieval_used = "local_search" in tools_used or "global_search" in tools_used

    # If no retrieval was used, skip evaluation
    if not retrieval_used:
        logger.info("No retrieval used, skipping evaluation")
        return {"evaluation": {"should_retry": False, "evaluations": []}}

    # Get messages - the conversation context
    messages = state.get("messages", [])

    # Find the final AI response (last AI message)
    final_response = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_response = msg.content
            break

    # Only evaluate if we have a response
    if not final_response:
        logger.info("Missing final response, skipping evaluation")
        return {"evaluation": {"should_retry": False, "evaluations": []}}

    logger.info(f"Evaluating response: '{final_response[:50]}...'")

    # Create system message for evaluation
    system_msg = SystemMessage(
        content="""
        You are evaluating the quality of an AI assistant's response in this conversation.
        
        Please evaluate the final AI response based on the conversation context:
        1. Does the response accurately reflect any retrieved context?
        2. Does the response address the user's query?
        3. Is the response asking for clarification from the user?
        4. What is your overall confidence in this response (0-1)?
        5. Are there any issues or concerns?
        6. Calculate RAG score (0-1): How relevant is any retrieved context to the user's query?
        7. Calculate consistency score (0-1): How well does the response align with the context?
        8. Should this response be retried? If yes, provide the reason.
        
        IMPORTANT EVALUATION GUIDELINES:
        - CLARIFICATION QUESTIONS ARE VALID RESPONSES: If the response is asking for clarification (e.g., "Could you please specify which location in Finland you would like to know the weather for?"), this is a GOOD response when the user query is ambiguous. Mark is_clarification_question as true and should_retry as false.
        - Only mark reflects_context as true if the response clearly aligns with the context.
        - Only mark addresses_query as true if the response actually answers what the user asked OR asks for appropriate clarification.
        - Clarification questions should have high confidence scores (0.8+) as they are appropriate responses to ambiguous queries.
        
        For retry decisions, consider:
        - Low relevance of context to query (low RAG score)
        - Response doesn't align with context (low consistency score)
        - Response doesn't address the query AND is not a clarification question
        - Response contains potential inaccuracies
        
        DO NOT RETRY if:
        - The response is asking for clarification (is_clarification_question = true)
        - The response appropriately addresses the user's query
        
        If you determine a retry is needed, specify whether:
        - "context_insufficient" - need to retrieve more/different context
        - "response_poor" - context is good but response quality is poor
        """
    )

    # Run LLM verification with full conversation context
    try:
        structured_llm = llm.with_structured_output(VerificationResult)
        verification_result = structured_llm.invoke([system_msg] + messages)
    except Exception as e:
        logger.error(f"LLM verification failed: {e}")
        # Return conservative fallback
        verification_result = VerificationResult(
            reflects_context=False,
            addresses_query=False,
            is_clarification_question=False,
            confidence=0.3,
            issues=[f"Verification failed: {str(e)}"],
            rag_score=0.3,
            consistency_score=0.3,
            should_retry=True,
            retry_reason="Verification failed",
        )

    should_retry = verification_result.should_retry
    retry_count = state.get("retry_count", 0)

    if should_retry:
        logger.warning(
            f"Low confidence detected - "
            f"RAG: {verification_result.rag_score:.3f}, "
            f"Consistency: {verification_result.consistency_score:.3f}, "
            f"Confidence: {verification_result.confidence:.3f}, "
            f"Reason: {verification_result.retry_reason}"
        )

    evaluation_result = {
        "evaluations": [verification_result.model_dump()],
        "should_retry": should_retry,
        "retry_count": retry_count,
        "verification_result": verification_result,
    }

    # Decide whether to retry or finalize
    if should_retry and retry_count < config.max_retries:
        logger.info(f"Triggering retry {retry_count + 1}/{config.max_retries}")

        # Smart message removal based on verification result
        messages_to_remove = []
        retry_reason = verification_result.retry_reason

        if retry_reason == "context_insufficient":
            # Remove the entire retrieval tool call sequence (AI tool call + tool message + AI response)
            logger.info("Removing tool call sequence for context re-retrieval")
            # Check which retrieval tool was used and remove its sequence
            if "local_search" in tools_used:
                messages_to_remove = _remove_tool_call_sequence(
                    messages, "local_search"
                )
            elif "global_search" in tools_used:
                messages_to_remove = _remove_tool_call_sequence(
                    messages, "global_search"
                )
        elif retry_reason == "response_poor":
            # Remove only the last AI response (context is sufficient)
            logger.info("Removing only the last response (context is sufficient)")
            if final_response:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content == final_response:
                        if msg.id:
                            messages_to_remove.append(RemoveMessage(id=msg.id))
                        break
        else:
            # Default: remove the last AI response
            logger.info("Default retry: removing last response")
            if final_response:
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage) and msg.content == final_response:
                        if msg.id:
                            messages_to_remove.append(RemoveMessage(id=msg.id))
                        break

        return {
            "evaluation": evaluation_result,
            "retry_count": retry_count + 1,
            "messages": messages_to_remove,
            "previous_evaluation": verification_result.model_dump(),  # Pass evaluation to next run
        }
    else:
        if should_retry:
            logger.warning(
                f"Max retries ({config.max_retries}) reached, keeping current response despite low confidence"
            )
        else:
            logger.info("Evaluation passed, proceeding to response formatting")
        # Don't remove messages when max retries reached - keep the current response for the user
        return {"evaluation": evaluation_result}


def response_formatter(state: AgentState, llm) -> Dict[str, Any]:
    """
    Format tool outputs if needed, otherwise pass through.
    Only formats if the last message contains unprocessed tool calls.
    If there's already a response from executor (no tool calls), use it directly.
    """
    from langchain_core.messages import ToolMessage, SystemMessage, AIMessage

    logger.info("Starting response_formatter node")
    try:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        last_message = messages[-1]

        # Check if we have tool calls that need formatting
        has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

        # Check if we have recent tool messages that need formatting
        recent_tool_messages = [
            msg for msg in messages[-5:] if isinstance(msg, ToolMessage)
        ]

        # If there's already a response from executor (AIMessage with content but no tool calls), use it directly
        if (
            isinstance(last_message, AIMessage)
            and last_message.content
            and not has_tool_calls
        ):
            logger.info("Using existing response from executor (no tool calls needed)")
            return {"messages": []}

        # If no tool calls and no recent tool messages, pass through
        if not has_tool_calls and not recent_tool_messages:
            logger.info("No tool calls to format, passing through")
            return {"messages": []}

        # If we have tool calls or tool messages, format them
        if has_tool_calls or recent_tool_messages:
            system_msg = SystemMessage(
                content="""
                You are formatting tool outputs for the user. Look at the recent tool calls and results.
                
                1. Format any raw tool outputs naturally and conversationally
                2. Provide a coherent response based on the tool results
                3. Consider the full conversation history for context
                
                Provide a natural, helpful response that incorporates the tool results.
                """
            )

            # Include conversation history for context
            final_messages = [system_msg] + messages
            final_response_obj = llm.invoke(final_messages)
            final_response = final_response_obj.content

            # Add confidence indicators based on evaluation results
            evaluation = state.get("evaluation", {})
            evaluations = evaluation.get("evaluations", [])

            if evaluations:
                # Get the most recent evaluation
                latest_eval = evaluations[-1] if evaluations else {}

                # Check if this was a low confidence response
                if (
                    evaluation.get("should_retry", False)
                    and state.get("retry_count", 0) >= config.max_retries
                ):
                    # Detailed confidence warning with scores
                    combined_score = latest_eval.get("combined_score", 0.0)
                    rag_score = latest_eval.get("rag_score", 0.0)
                    consistency_score = latest_eval.get("consistency_score", 0.0)
                    verifier_score = latest_eval.get("verifier_score", 0.0)

                    confidence_warning = f"\n\n⚠️ **Low Confidence Response** (Score: {combined_score:.2f}/1.0)"
                    confidence_warning += f"\n- Retrieval relevance: {rag_score:.2f}"
                    confidence_warning += (
                        f"\n- Source consistency: {consistency_score:.2f}"
                    )
                    confidence_warning += f"\n- LLM verification: {verifier_score:.2f}"
                    confidence_warning += (
                        "\n\nPlease verify this information independently."
                    )

                    final_response += confidence_warning
                elif latest_eval.get("combined_score", 1.0) < 0.85:
                    # Mild confidence indicator for borderline scores
                    combined_score = latest_eval.get("combined_score", 1.0)
                    confidence_note = f"\n\n💡 *Confidence: {combined_score:.2f}/1.0*"
                    final_response += confidence_note

            ai_message = AIMessage(content=final_response)
            logger.info("Response formatted successfully")
            return {"messages": [ai_message]}
        else:
            # No formatting needed
            logger.info("No formatting needed")
            return {"messages": []}

    except Exception as e:
        logger.error(f"Error in response_formatter node: {e}")
        return {"messages": []}
