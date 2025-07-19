#!/usr/bin/env python3
"""
RAG Comparison Tool
Compares Traditional RAG vs GraphRAG approaches side by side
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Add parent directories to path for accessing main project
parent_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(parent_dir)

from .traditional_rag import create_traditional_rag
from .graph_rag import create_graph_rag


@dataclass
class ComparisonResult:
    """Results from comparing Traditional RAG vs GraphRAG."""

    query: str
    traditional_results: Dict[str, Any]
    graph_results: Dict[str, Any]
    comparison_metrics: Dict[str, Any]
    timestamp: float


class RAGComparator:
    """Compares Traditional RAG vs GraphRAG approaches."""

    def __init__(self):
        print("🔧 Initializing RAG Comparator...")
        self.traditional_rag = create_traditional_rag()
        self.graph_rag = create_graph_rag()
        print("✅ RAG Comparator initialized!")

    def compare_retrieval(self, query: str, k: int = 5) -> ComparisonResult:
        """
        Compare Traditional RAG vs GraphRAG for a given query.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            ComparisonResult with detailed comparison
        """
        print(f"🔍 Comparing retrieval approaches for query: '{query}'")

        # Get results from both approaches
        traditional_results = self.traditional_rag.retrieve(query, k=k)
        graph_results = self.graph_rag.retrieve(query, k=k)

        # Calculate comparison metrics
        comparison_metrics = self._calculate_comparison_metrics(
            traditional_results, graph_results
        )

        return ComparisonResult(
            query=query,
            traditional_results=traditional_results,
            graph_results=graph_results,
            comparison_metrics=comparison_metrics,
            timestamp=time.time(),
        )

    def _calculate_comparison_metrics(
        self, traditional: Dict, graph: Dict
    ) -> Dict[str, Any]:
        """Calculate metrics comparing the two approaches."""
        metrics = {}

        # Performance metrics
        metrics["retrieval_time"] = {
            "traditional": traditional.get("retrieval_time", 0),
            "graph": graph.get("retrieval_time", 0),
            "speedup": self._calculate_speedup(
                traditional.get("retrieval_time", 0), graph.get("retrieval_time", 0)
            ),
        }

        # Document overlap
        traditional_docs = traditional.get("documents", [])
        graph_docs = graph.get("documents", [])

        metrics["document_overlap"] = self._calculate_document_overlap(
            traditional_docs, graph_docs
        )

        # Score distributions
        traditional_scores = traditional.get("scores", [])
        graph_scores = graph.get("scores", [])

        metrics["score_distribution"] = {
            "traditional": {
                "mean": (
                    sum(traditional_scores) / len(traditional_scores)
                    if traditional_scores
                    else 0
                ),
                "min": min(traditional_scores) if traditional_scores else 0,
                "max": max(traditional_scores) if traditional_scores else 0,
                "scores": traditional_scores,
            },
            "graph": {
                "mean": sum(graph_scores) / len(graph_scores) if graph_scores else 0,
                "min": min(graph_scores) if graph_scores else 0,
                "max": max(graph_scores) if graph_scores else 0,
                "scores": graph_scores,
            },
        }

        # Strategy information
        metrics["strategies"] = {
            "traditional": traditional.get("strategy", "Unknown"),
            "graph": graph.get("strategy", "Unknown"),
        }

        # Graph-specific metrics
        if graph.get("graph_traversal_success"):
            metrics["graph_traversal"] = {
                "success": True,
                "edges_used": graph.get("edges_used", []),
                "graph_stats": graph.get("graph_stats", {}),
            }
        else:
            metrics["graph_traversal"] = {
                "success": False,
                "fallback_reason": graph.get("fallback_reason", "Unknown"),
            }

        return metrics

    def _calculate_speedup(self, time1: float, time2: float) -> str:
        """Calculate speedup between two times."""
        if time1 == 0 or time2 == 0:
            return "N/A"

        if time1 < time2:
            return f"{time2/time1:.2f}x faster (traditional)"
        else:
            return f"{time1/time2:.2f}x faster (graph)"

    def _calculate_document_overlap(self, docs1: List, docs2: List) -> Dict[str, Any]:
        """Calculate overlap between two document lists."""
        if not docs1 or not docs2:
            return {
                "overlap_count": 0,
                "overlap_percentage": 0,
                "unique_to_traditional": 0,
                "unique_to_graph": 0,
            }

        # Use document content for comparison (simplified)
        set1 = set(
            doc.page_content[:100] for doc in docs1
        )  # First 100 chars as identifier
        set2 = set(doc.page_content[:100] for doc in docs2)

        overlap = len(set1.intersection(set2))
        unique_to_traditional = len(set1 - set2)
        unique_to_graph = len(set2 - set1)

        total_unique = len(set1.union(set2))
        overlap_percentage = (overlap / total_unique * 100) if total_unique > 0 else 0

        return {
            "overlap_count": overlap,
            "overlap_percentage": overlap_percentage,
            "unique_to_traditional": unique_to_traditional,
            "unique_to_graph": unique_to_graph,
        }

    def batch_compare(self, queries: List[str], k: int = 5) -> List[ComparisonResult]:
        """Compare multiple queries."""
        results = []
        for query in queries:
            result = self.compare_retrieval(query, k=k)
            results.append(result)
        return results

    def generate_comparison_report(
        self, results: List[ComparisonResult]
    ) -> Dict[str, Any]:
        """Generate a comprehensive comparison report."""
        if not results:
            return {"error": "No results to analyze"}

        # Aggregate metrics
        total_traditional_time = sum(
            r.comparison_metrics["retrieval_time"]["traditional"] for r in results
        )
        total_graph_time = sum(
            r.comparison_metrics["retrieval_time"]["graph"] for r in results
        )

        avg_overlap = sum(
            r.comparison_metrics["document_overlap"]["overlap_percentage"]
            for r in results
        ) / len(results)

        graph_success_rate = (
            sum(
                1 for r in results if r.comparison_metrics["graph_traversal"]["success"]
            )
            / len(results)
            * 100
        )

        return {
            "summary": {
                "total_queries": len(results),
                "avg_traditional_time": total_traditional_time / len(results),
                "avg_graph_time": total_graph_time / len(results),
                "avg_document_overlap": avg_overlap,
                "graph_success_rate": graph_success_rate,
            },
            "detailed_results": [
                {
                    "query": r.query,
                    "traditional_time": r.comparison_metrics["retrieval_time"][
                        "traditional"
                    ],
                    "graph_time": r.comparison_metrics["retrieval_time"]["graph"],
                    "overlap_percentage": r.comparison_metrics["document_overlap"][
                        "overlap_percentage"
                    ],
                    "graph_success": r.comparison_metrics["graph_traversal"]["success"],
                }
                for r in results
            ],
        }

    def save_results(self, results: List[ComparisonResult], filename: str):
        """Save comparison results to JSON file."""
        serializable_results = []
        for result in results:
            serializable_result = {
                "query": result.query,
                "timestamp": result.timestamp,
                "traditional_results": {
                    "method": result.traditional_results.get("method"),
                    "retrieval_time": result.traditional_results.get("retrieval_time"),
                    "total_documents": result.traditional_results.get(
                        "total_documents"
                    ),
                    "strategy": result.traditional_results.get("strategy"),
                    "scores": result.traditional_results.get("scores", []),
                },
                "graph_results": {
                    "method": result.graph_results.get("method"),
                    "retrieval_time": result.graph_results.get("retrieval_time"),
                    "total_documents": result.graph_results.get("total_documents"),
                    "strategy": result.graph_results.get("strategy"),
                    "scores": result.graph_results.get("scores", []),
                    "graph_traversal_success": result.graph_results.get(
                        "graph_traversal_success"
                    ),
                },
                "comparison_metrics": result.comparison_metrics,
            }
            serializable_results.append(serializable_result)

        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)

        print(f"📊 Results saved to {filename}")


def run_sample_comparison():
    """Run a sample comparison with predefined queries."""
    comparator = RAGComparator()

    # Sample queries for comparison
    sample_queries = [
        "Finnish sauna culture and traditions",
        "Helsinki travel guide recommendations",
        "What is the history of Finnish architecture?",
        "Traditional Finnish cuisine and food",
        "Finnish education system overview",
    ]

    print("🚀 Running sample comparison...")
    results = comparator.batch_compare(sample_queries, k=5)

    # Generate report
    report = comparator.generate_comparison_report(results)

    print("\n📊 Comparison Report:")
    print(f"Total queries: {report['summary']['total_queries']}")
    print(
        f"Average Traditional RAG time: {report['summary']['avg_traditional_time']:.3f}s"
    )
    print(f"Average Graph RAG time: {report['summary']['avg_graph_time']:.3f}s")
    print(f"Average document overlap: {report['summary']['avg_document_overlap']:.1f}%")
    print(
        f"Graph traversal success rate: {report['summary']['graph_success_rate']:.1f}%"
    )

    # Save results
    comparator.save_results(results, "rag_comparison/results/comparison_results.json")

    return results, report


if __name__ == "__main__":
    # Create results directory
    os.makedirs("rag_comparison/results", exist_ok=True)

    # Run sample comparison
    results, report = run_sample_comparison()

    print("\n✅ Comparison complete! Check the results directory for detailed output.")
