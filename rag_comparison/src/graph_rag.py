#!/usr/bin/env python3
"""
GraphRAG Implementation using Official GraphRAG Library
Uses the working implementation from graph_retrieval.py with both local and global search
"""

import os
import sys
import time
import asyncio
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

# Add parent directories to path for accessing main project
parent_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(parent_dir)

# Import the working GraphRAG implementation
from src.knowledge.graph_retrieval import GraphRAGRetrieval


class GraphRAG:
    """GraphRAG implementation with both local and global search results."""

    # Class-level cache for GraphRAGRetrieval instance
    _cached_retriever = None

    def __init__(self):
        # Use cached retriever if available
        if GraphRAG._cached_retriever is None:
            GraphRAG._cached_retriever = GraphRAGRetrieval()
            print("✅ GraphRAG initialized successfully")
        else:
            print("✅ Using cached GraphRAG retriever")
        
        self.retriever = GraphRAG._cached_retriever
        # Cache the event loop for reuse
        self._event_loop = None

    def retrieve(self, query: str, k: int = 5, max_depth: int = 2) -> Dict[str, Any]:
        """
        Retrieve documents using local GraphRAG search.

        Args:
            query: Search query
            k: Number of documents to retrieve (kept for compatibility)
            max_depth: Maximum depth for graph traversal (unused)

        Returns:
            Dictionary with results from local search
        """
        start_time = time.time()

        try:
            print(f"🔍 Running GraphRAG local search for: {query}")

            # Use direct async approach like test file
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run local search
                local_result = loop.run_until_complete(self.retriever.local_search(query))

                # Convert to expected format
                documents = []

                # Add local search result
                if local_result and hasattr(local_result, "response"):
                    # Convert any non-serializable objects to strings
                    context_data = getattr(local_result, "context_data", {})
                    if hasattr(context_data, "to_dict"):
                        context_data = context_data.to_dict()
                    elif hasattr(context_data, "__dict__"):
                        context_data = context_data.__dict__
                    else:
                        context_data = str(context_data) if context_data else ""

                    local_doc = Document(
                        page_content=f"{local_result.response}",
                        metadata={
                            "source": "graphrag_local",
                            "search_type": "local",
                            "retrieval_method": "graphrag",
                            "context_data": context_data,
                            "completion_time": float(
                                getattr(local_result, "completion_time", 0)
                            ),
                            "llm_calls": int(getattr(local_result, "llm_calls", 0)),
                            "prompt_tokens": int(
                                getattr(local_result, "prompt_tokens", 0)
                            ),
                        },
                    )
                    documents.append(local_doc)

                retrieval_time = time.time() - start_time

                # Convert results to JSON-serializable format
                def make_serializable(obj):
                    """Convert object to JSON-serializable format"""
                    if hasattr(obj, "to_dict"):
                        return obj.to_dict()
                    elif hasattr(obj, "__dict__"):
                        return {
                            k: make_serializable(v) for k, v in obj.__dict__.items()
                        }
                    elif isinstance(obj, (list, tuple)):
                        return [make_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {k: make_serializable(v) for k, v in obj.items()}
                    else:
                        return str(obj) if obj is not None else ""

                return {
                    "documents": documents,
                    "method": "graph_rag_local",
                    "query": query,
                    "retrieval_time": retrieval_time,
                    "total_documents": len(documents),
                    "strategy": "GraphRAG Local Search",
                    "search_type": "local",
                    "local_result": make_serializable(local_result),
                    "graph_traversal_success": True,
                }

            finally:
                loop.close()
                asyncio.set_event_loop(None)

        except Exception as e:
            print(f"❌ GraphRAG failed: {e}")
            return {
                "documents": [],
                "method": "graph_rag_local",
                "query": query,
                "error": str(e),
                "retrieval_time": time.time() - start_time,
                "total_documents": 0,
                "scores": [],
                "strategy": "GraphRAG failed",
                "graph_traversal_success": False,
            }

    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the GraphRAG system."""
        return {
            "method": "graph_rag_local",
            "retrieval_strategy": "GraphRAG Local Search",
            "entities": len(self.retriever.entities) if self.retriever.entities else 0,
            "relationships": (
                len(self.retriever.relationships) if self.retriever.relationships else 0
            ),
            "community_reports": (
                len(self.retriever.reports) if self.retriever.reports else 0
            ),
            "text_units": (
                len(self.retriever.text_units) if self.retriever.text_units else 0
            ),
            "communities": (
                len(self.retriever.communities) if self.retriever.communities else 0
            ),
            "data_source": "GraphRAG using official library",
            "chat_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "search_types": ["local"],
        }

    def get_graph_traversal_info(self, query: str) -> Dict[str, Any]:
        """Get graph traversal information for a query."""
        try:
            # Get basic info about the graph structure
            entities_count = (
                len(self.retriever.entities) if self.retriever.entities else 0
            )
            relationships_count = (
                len(self.retriever.relationships) if self.retriever.relationships else 0
            )

            # Return format expected by the comparison tool
            return {
                "query": query,
                "total_entities": entities_count,
                "total_relationships": relationships_count,
                "method": "graphrag_local",
                "search_types": ["local"],
                "supports_combined_search": False,
            }

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "total_entities": 0,
                "total_relationships": 0,
                "method": "graphrag_local",
                "search_types": ["local"],
                "supports_combined_search": False,
            }


def create_graph_rag() -> GraphRAG:
    """Factory function to create a GraphRAG instance."""
    return GraphRAG()


if __name__ == "__main__":
    # Test the GraphRAG implementation
    rag = create_graph_rag()

    test_query = "What are the main countries in Europe?"
    print(f"Testing GraphRAG with query: '{test_query}'")

    results = rag.retrieve(test_query)
    print(f"Retrieved {results['total_documents']} documents")
    print(f"Retrieval time: {results['retrieval_time']:.3f}s")
    print(f"Method: {results.get('strategy', 'N/A')}")

    for i, doc in enumerate(results["documents"]):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Search Type: {doc.metadata.get('search_type', 'unknown')}")
        print(f"Score: {doc.metadata.get('similarity_score', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")

    # Show graph traversal info
    print(f"\n--- Graph Traversal Info ---")
    traversal_info = rag.get_graph_traversal_info(test_query)
    print(f"Total entities: {traversal_info['total_entities']}")
    print(f"Total relationships: {traversal_info['total_relationships']}")
    print(f"Search types: {traversal_info['search_types']}")
    print(f"Supports combined search: {traversal_info['supports_combined_search']}")