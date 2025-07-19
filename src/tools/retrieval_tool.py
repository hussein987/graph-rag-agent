"""
Retrieval tools for accessing the knowledge base using different search strategies:
1. Vector search using FAISS
2. Local search using GraphRAG
3. Global search using GraphRAG
"""

from langchain_core.tools import tool
from langchain_core.documents import Document
from src.knowledge.vector_store_builder import VectorStoreBuilder
from src.knowledge.graph_retrieval import GraphRAGRetrieval
from typing import List
import threading
import asyncio

# Thread-local storage for retrieval results
_retrieval_cache = threading.local()


@tool
def vector_search(query: str) -> str:
    """
    Search and return relevant information using vector similarity search.

    This tool uses FAISS vector store for semantic search based on embeddings.

    Args:
        query: The search query to find relevant information

    Returns:
        Relevant information from the knowledge base using vector search
    """
    try:
        # Initialize vector store
        vector_builder = VectorStoreBuilder()
        vector_store = vector_builder.build_vector_store()

        if not vector_store:
            return "No vector store available. Please build the vector store first."

        # Get relevant documents using vector search
        docs_with_scores = vector_store.similarity_search_with_score(query, k=5)

        if not docs_with_scores:
            return "No relevant information found in the knowledge base."

        # Format the results
        result_parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            result_parts.append(
                f"Result {i} (Source: {source}, Score: {score:.3f}):\\n{content}"
            )

        formatted_result = "\\n\\n".join(result_parts)

        # Cache the retrieval results for evaluation
        _retrieval_cache.last_retrieval = {
            "query": query,
            "documents": [doc for doc, _ in docs_with_scores],
            "formatted_result": formatted_result,
        }

        return formatted_result

    except Exception as e:
        error_msg = f"Error in vector search: {str(e)}"
        # Cache error results
        _retrieval_cache.last_retrieval = {
            "query": query,
            "documents": [],
            "formatted_result": error_msg,
        }
        return error_msg


@tool
def local_search(query: str) -> str:
    """
    Search and return relevant information using GraphRAG local search.

    This tool uses GraphRAG's local search which combines vector search with
    graph traversal for more detailed, entity-focused responses.

    Args:
        query: The search query to find relevant information

    Returns:
        Relevant information from the knowledge base using local search
    """
    try:
        # Initialize GraphRAG retrieval
        graphrag = GraphRAGRetrieval()

        # Run local search asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(graphrag.local_search(query))
            response = getattr(result, "response", str(result))

            # Cache the retrieval results for evaluation
            _retrieval_cache.last_retrieval = {
                "query": query,
                "documents": [],  # GraphRAG doesn't return documents directly
                "formatted_result": response,
            }

            return response
        finally:
            loop.close()

    except Exception as e:
        error_msg = f"Error in local search: {str(e)}"
        # Cache error results
        _retrieval_cache.last_retrieval = {
            "query": query,
            "documents": [],
            "formatted_result": error_msg,
        }
        return error_msg


@tool
def global_search(query: str) -> str:
    """
    Search and return relevant information using GraphRAG global search.

    This tool uses GraphRAG's global search which provides high-level summaries
    and insights across the entire knowledge base.

    Args:
        query: The search query to find relevant information

    Returns:
        Relevant information from the knowledge base using global search
    """
    try:
        # Initialize GraphRAG retrieval
        graphrag = GraphRAGRetrieval()

        # Run global search asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(graphrag.global_search(query))
            response = getattr(result, "response", str(result))

            # Cache the retrieval results for evaluation
            _retrieval_cache.last_retrieval = {
                "query": query,
                "documents": [],  # GraphRAG doesn't return documents directly
                "formatted_result": response,
            }

            return response
        finally:
            loop.close()

    except Exception as e:
        error_msg = f"Error in global search: {str(e)}"
        # Cache error results
        _retrieval_cache.last_retrieval = {
            "query": query,
            "documents": [],
            "formatted_result": error_msg,
        }
        return error_msg


def get_last_retrieval():
    """Get the last retrieval results from cache."""
    return getattr(_retrieval_cache, "last_retrieval", None)


def create_retrieval_tools():
    """
    Create all three retrieval tools.

    Returns:
        List of retrieval tools that can be used by LLM agents
    """
    return [vector_search, local_search, global_search]
