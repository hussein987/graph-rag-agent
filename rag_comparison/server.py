#!/usr/bin/env python3
"""
Flask Web Server for RAG Comparison Tool
Serves the comparison UI and provides API endpoints for running comparisons
"""

import os
import sys
import json
import time
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory,
)
from flask_cors import CORS
import asyncio
import threading

# Add parent directories to path for accessing main project
parent_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(parent_dir)

from rag_comparison.src.comparison_tool import RAGComparator

app = Flask(__name__)
CORS(app)

# Global instances
comparator = None
_event_loop = None
_loop_thread = None


def init_comparator():
    """Initialize the RAG comparator."""
    global comparator
    if comparator is None:
        try:
            comparator = RAGComparator()
        except Exception as e:
            print(f"Error initializing comparator: {e}")
            comparator = None
    return comparator


def get_event_loop():
    """Get or create a persistent event loop in a background thread."""
    global _event_loop, _loop_thread
    
    if _event_loop is None or _event_loop.is_closed():
        def run_loop():
            global _event_loop
            _event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_event_loop)
            _event_loop.run_forever()
        
        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()
        
        # Wait for loop to be ready
        while _event_loop is None:
            time.sleep(0.01)
    
    return _event_loop


def run_async(coro):
    """Run async function in the persistent event loop."""
    loop = get_event_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=120)  # 2 minute timeout


@app.route("/")
def index():
    """Serve the React app."""
    try:
        return send_from_directory("build", "index.html")
    except FileNotFoundError:
        # Fallback to development mode message
        return """
        <h1>RAG Comparison Tool</h1>
        <p>Please build the React app first:</p>
        <pre>cd rag_comparison && npm install && npm run build</pre>
        <p>Then restart the server.</p>
        <p>For development, run <code>npm start</code> in the rag_comparison directory.</p>
        """


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files from React build."""
    return send_from_directory("build/static", filename)




@app.route("/api/traditional-rag", methods=["POST"])
def traditional_rag():
    """API endpoint for traditional RAG retrieval."""
    try:
        data = request.get_json()
        query = data.get("query", "")
        k = data.get("k", 5)

        if not query:
            return jsonify({"error": "Query is required"}), 400

        comp = init_comparator()
        if comp is None:
            return jsonify({"error": "Failed to initialize comparator"}), 500

        # Run traditional RAG only
        try:
            result = comp.traditional_rag.retrieve(query, k=k)

            # Convert to JSON-serializable format
            def make_json_serializable(obj):
                import numpy as np

                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif hasattr(obj, "page_content"):
                    return {
                        "page_content": obj.page_content,
                        "metadata": make_json_serializable(obj.metadata),
                    }
                else:
                    return obj

            response = {
                "method": result.get("method"),
                "retrieval_time": float(result.get("retrieval_time", 0)),
                "total_documents": result.get("total_documents", 0),
                "strategy": result.get("strategy", "Unknown"),
                "documents": [
                    make_json_serializable(doc) for doc in result.get("documents", [])
                ],
                "query": query,
                "timestamp": time.time(),
            }

            return jsonify(response)

        except Exception as e:
            return (
                jsonify(
                    {
                        "error": str(e),
                        "method": "traditional_rag",
                        "retrieval_time": 0,
                        "total_documents": 0,
                        "strategy": "Error - Traditional RAG failed",
                        "documents": [],
                        "query": query,
                        "timestamp": time.time(),
                    }
                ),
                500,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/local-search", methods=["POST"])
def local_search():
    """API endpoint for GraphRAG local search."""
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        comp = init_comparator()
        if comp is None:
            return jsonify({"error": "Failed to initialize comparator"}), 500

        # Run local search only
        try:
            retriever = comp.graph_rag.retriever
            
            # Measure actual execution time
            start_time = time.time()
            # Use persistent event loop
            local_result = run_async(retriever.local_search(query))
            end_time = time.time()
            actual_retrieval_time = end_time - start_time

            if local_result and hasattr(local_result, "response"):
                document = {
                    "page_content": local_result.response,
                    "metadata": {
                        "source": "graphrag_local",
                        "search_type": "local",
                        "retrieval_method": "graphrag",
                        "completion_time": float(
                            getattr(local_result, "completion_time", 0)
                        ),
                        "llm_calls": int(getattr(local_result, "llm_calls", 0)),
                        "prompt_tokens": int(
                            getattr(local_result, "prompt_tokens", 0)
                        ),
                    },
                }

                response = {
                    "method": "graph_rag_local",
                    "retrieval_time": actual_retrieval_time,
                    "total_documents": 1,
                    "strategy": "GraphRAG Local Search",
                    "documents": [document],
                    "search_type": "local",
                    "query": query,
                    "timestamp": time.time(),
                }

                return jsonify(response)
            else:
                return (
                    jsonify(
                        {
                            "error": "No local search result",
                            "method": "graph_rag_local",
                            "retrieval_time": 0,
                            "total_documents": 0,
                            "strategy": "Error - Local search failed",
                            "documents": [],
                            "query": query,
                            "timestamp": time.time(),
                        }
                    ),
                    500,
                )

        except Exception as e:
            return (
                jsonify(
                    {
                        "error": str(e),
                        "method": "graph_rag_local",
                        "retrieval_time": 0,
                        "total_documents": 0,
                        "strategy": "Error - Local search failed",
                        "documents": [],
                        "query": query,
                        "timestamp": time.time(),
                    }
                ),
                500,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/global-search", methods=["POST"])
def global_search():
    """API endpoint for GraphRAG global search."""
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        comp = init_comparator()
        if comp is None:
            return jsonify({"error": "Failed to initialize comparator"}), 500

        # Run global search only
        try:
            retriever = comp.graph_rag.retriever
            
            # Measure actual execution time
            start_time = time.time()
            # Use persistent event loop
            global_result = run_async(retriever.global_search(query))
            end_time = time.time()
            actual_retrieval_time = end_time - start_time

            if global_result and hasattr(global_result, "response"):
                document = {
                    "page_content": global_result.response,
                    "metadata": {
                        "source": "graphrag_global",
                        "search_type": "global",
                        "retrieval_method": "graphrag",
                        "completion_time": float(
                            getattr(global_result, "completion_time", 0)
                        ),
                        "llm_calls": int(getattr(global_result, "llm_calls", 0)),
                        "prompt_tokens": int(
                            getattr(global_result, "prompt_tokens", 0)
                        ),
                    },
                }

                response = {
                    "method": "graph_rag_global",
                    "retrieval_time": actual_retrieval_time,
                    "total_documents": 1,
                    "strategy": "GraphRAG Global Search",
                    "documents": [document],
                    "search_type": "global",
                    "query": query,
                    "timestamp": time.time(),
                }

                return jsonify(response)
            else:
                return (
                    jsonify(
                        {
                            "error": "No global search result",
                            "method": "graph_rag_global",
                            "retrieval_time": 0,
                            "total_documents": 0,
                            "strategy": "Error - Global search failed",
                            "documents": [],
                            "query": query,
                            "timestamp": time.time(),
                        }
                    ),
                    500,
                )

        except Exception as e:
            return (
                jsonify(
                    {
                        "error": str(e),
                        "method": "graph_rag_global",
                        "retrieval_time": 0,
                        "total_documents": 0,
                        "strategy": "Error - Global search failed",
                        "documents": [],
                        "query": query,
                        "timestamp": time.time(),
                    }
                ),
                500,
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/api/system_info", methods=["GET"])
def system_info():
    """API endpoint for getting system information."""
    try:
        comp = init_comparator()
        if comp is None:
            return jsonify({"error": "Failed to initialize comparator"}), 500

        traditional_info = comp.traditional_rag.get_document_info()
        graph_info = comp.graph_rag.get_document_info()

        return jsonify({"traditional_rag": traditional_info, "graph_rag": graph_info})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/graph_info", methods=["POST"])
def graph_info():
    """API endpoint for getting graph traversal information."""
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        comp = init_comparator()
        if comp is None:
            return jsonify({"error": "Failed to initialize comparator"}), 500

        traversal_info = comp.graph_rag.get_graph_traversal_info(query)

        return jsonify(traversal_info)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


# For backward compatibility, keep the main section
if __name__ == "__main__":
    print("🚀 Starting RAG Comparison Server...")
    print("🔧 Initializing comparator...")

    # Initialize comparator at startup
    comp = init_comparator()
    if comp:
        print("✅ Comparator initialized successfully!")
    else:
        print(
            "⚠️  Warning: Comparator initialization failed. Some features may not work."
        )

    print("🌐 Server starting at http://localhost:5000")
    print("📊 Access the comparison UI at: http://localhost:5000")

    app.run(debug=True, host="0.0.0.0", port=5000)
