#!/usr/bin/env python3
"""
Knowledge Graph Builder Module

This module provides classes for building and loading knowledge graphs:
- GraphRAGLoader: Loads pre-built GraphRAG knowledge graphs
- GraphRAGBuilder: CLI tool for building GraphRAG knowledge graphs
"""

import os
import json
import shutil
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import networkx as nx
import tempfile
import subprocess

from src.utils.config import config


class GraphRAGLoader:
    """
    Loads pre-built GraphRAG knowledge graphs from JSON files.
    The graph is built separately using the GraphRAG CLI (GraphRAGBuilder).
    """

    def __init__(self, graph_file: str = "data/graphrag_knowledge_graph.json"):
        self.graph_file = graph_file
        self.graph = None

    def load_graph(self) -> nx.DiGraph:
        """Load GraphRAG knowledge graph from JSON file."""
        if not os.path.exists(self.graph_file):
            raise FileNotFoundError(
                f"GraphRAG knowledge graph not found at {self.graph_file}"
            )

        with open(self.graph_file, "r") as f:
            graph_data = json.load(f)

        self.graph = nx.DiGraph()

        # Add nodes
        for node_data in graph_data.get("nodes", []):
            node_id = node_data.get("id")
            if node_id:
                self.graph.add_node(
                    node_id, **{k: v for k, v in node_data.items() if k != "id"}
                )

        # Add edges
        for edge_data in graph_data.get("edges", []):
            source = edge_data.get("source")
            target = edge_data.get("target")
            if source and target:
                self.graph.add_edge(
                    source,
                    target,
                    **{
                        k: v
                        for k, v in edge_data.items()
                        if k not in ["source", "target"]
                    },
                )

        return self.graph


class GraphRAGBuilder:
    """
    Integration with Microsoft GraphRAG for building knowledge graphs.

    IMPORTANT: This is a CLI-only tool for building the knowledge graph.
    The agent uses the pre-built graph via GraphRAGLoader.
    Run this manually when you want to rebuild the knowledge graph.
    """

    def __init__(
        self, corpus_path: str = None, output_dir: str = None, temp_dir: str = None
    ):
        self.corpus_path = corpus_path or config.corpus_path
        self.output_dir = output_dir or "data/graphrag_output"
        self.temp_dir = temp_dir or None
        self.graph = None
        self.entities = None
        self.relationships = None
        self.communities = None
        self.community_reports = None

    def load_graphrag_outputs(self):
        """Load the generated GraphRAG outputs using the new flat structure."""
        output_dir = os.path.join(self.temp_dir, "output")

        if not os.path.exists(output_dir):
            raise FileNotFoundError(
                f"GraphRAG output directory not found: {output_dir}"
            )

        # Use new flat structure
        print("Loading GraphRAG outputs from flat structure...")

        entities_file = os.path.join(output_dir, "entities.parquet")
        relationships_file = os.path.join(output_dir, "relationships.parquet")
        communities_file = os.path.join(output_dir, "communities.parquet")
        reports_file = os.path.join(output_dir, "community_reports.parquet")

        # Load entities
        if os.path.exists(entities_file):
            self.entities = pd.read_parquet(entities_file)
            print(f"Loaded {len(self.entities)} entities")

        # Load relationships
        if os.path.exists(relationships_file):
            self.relationships = pd.read_parquet(relationships_file)
            print(f"Loaded {len(self.relationships)} relationships")

        # Load communities
        if os.path.exists(communities_file):
            self.communities = pd.read_parquet(communities_file)
            print(f"Loaded {len(self.communities)} communities")

        # Load community reports
        if os.path.exists(reports_file):
            self.community_reports = pd.read_parquet(reports_file)
            print(f"Loaded {len(self.community_reports)} community reports")

    def build_networkx_graph(self) -> nx.DiGraph:
        """Build NetworkX graph from GraphRAG outputs"""
        self.graph = nx.DiGraph()

        # Create a mapping from entity title to entity id
        title_to_id = {}
        if self.entities is not None:
            for _, entity in self.entities.iterrows():
                title_to_id[entity["title"]] = entity["id"]

        # Add entities as nodes
        if self.entities is not None:
            for _, entity in self.entities.iterrows():
                # Convert numpy types to Python native types for JSON serialization
                self.graph.add_node(
                    entity["id"],
                    label=entity["title"],
                    type=entity.get("type", "entity"),
                    description=entity.get("description", ""),
                    degree=(
                        int(entity.get("degree", 0))
                        if pd.notna(entity.get("degree", 0))
                        else 0
                    ),
                    community=(
                        int(entity.get("community", -1))
                        if pd.notna(entity.get("community", -1))
                        else -1
                    ),
                    level=(
                        int(entity.get("level", 0))
                        if pd.notna(entity.get("level", 0))
                        else 0
                    ),
                    x=(
                        float(entity.get("x", 0.0))
                        if pd.notna(entity.get("x", 0.0))
                        else 0.0
                    ),
                    y=(
                        float(entity.get("y", 0.0))
                        if pd.notna(entity.get("y", 0.0))
                        else 0.0
                    ),
                )

        # Add relationships as edges
        if self.relationships is not None:
            for _, rel in self.relationships.iterrows():
                source_id = title_to_id.get(rel["source"])
                target_id = title_to_id.get(rel["target"])

                if (
                    source_id
                    and target_id
                    and source_id in self.graph.nodes
                    and target_id in self.graph.nodes
                ):
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        relation=rel.get("description", ""),
                        weight=(
                            float(rel.get("weight", 1.0))
                            if pd.notna(rel.get("weight", 1.0))
                            else 1.0
                        ),
                        combined_degree=(
                            int(rel.get("combined_degree", 0))
                            if pd.notna(rel.get("combined_degree", 0))
                            else 0
                        ),
                    )

        print(
            f"Built NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges"
        )
        return self.graph

    def build_graph_from_existing_index(self, index_path: str) -> nx.DiGraph:
        """Build NetworkX graph from an existing GraphRAG index"""
        try:
            # Set the temp_dir to the existing index path
            self.temp_dir = index_path

            # Load outputs from existing index
            print(f"Loading GraphRAG outputs from {index_path}...")
            self.load_graphrag_outputs()

            # Build NetworkX graph
            print("Building NetworkX graph...")
            graph = self.build_networkx_graph()

            return graph

        except Exception as e:
            print(f"Error building graph from existing index: {e}")
            raise

    def save_graph(self, filepath: str = "data/graphrag_knowledge_graph.json"):
        """Save the GraphRAG knowledge graph to JSON"""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            raise ValueError("No graph to save. Build graph first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Enhanced graph data with community information
        graph_data = {
            "metadata": {
                "num_nodes": self.graph.number_of_nodes(),
                "num_edges": self.graph.number_of_edges(),
                "num_communities": (
                    len(self.communities) if self.communities is not None else 0
                ),
                "generation_method": "microsoft_graphrag",
                "has_communities": self.communities is not None,
            },
            "nodes": [
                {"id": node, **data} for node, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **data}
                for u, v, data in self.graph.edges(data=True)
            ],
            "communities": [],
            "community_reports": [],
        }

        # Convert community data to JSON-serializable format
        if self.communities is not None:
            for _, community in self.communities.iterrows():
                community_dict = {}
                for key, value in community.items():
                    try:
                        if pd.isna(value):
                            community_dict[key] = None
                        elif isinstance(value, pd.Timestamp):
                            community_dict[key] = str(value)
                        elif hasattr(value, "dtype") and "int" in str(value.dtype):
                            community_dict[key] = int(value)
                        elif hasattr(value, "dtype") and "float" in str(value.dtype):
                            community_dict[key] = float(value)
                        elif hasattr(value, "__iter__") and not isinstance(value, str):
                            # Handle numpy arrays and lists
                            community_dict[key] = (
                                list(value)
                                if hasattr(value, "__iter__")
                                else str(value)
                            )
                        else:
                            community_dict[key] = value
                    except (ValueError, TypeError):
                        # Fallback for problematic values
                        community_dict[key] = str(value)
                graph_data["communities"].append(community_dict)

        # Convert community reports to JSON-serializable format
        if self.community_reports is not None:
            for _, report in self.community_reports.iterrows():
                report_dict = {}
                for key, value in report.items():
                    try:
                        if pd.isna(value):
                            report_dict[key] = None
                        elif isinstance(value, pd.Timestamp):
                            report_dict[key] = str(value)
                        elif hasattr(value, "dtype") and "int" in str(value.dtype):
                            report_dict[key] = int(value)
                        elif hasattr(value, "dtype") and "float" in str(value.dtype):
                            report_dict[key] = float(value)
                        elif hasattr(value, "__iter__") and not isinstance(value, str):
                            # Handle numpy arrays and lists
                            report_dict[key] = (
                                list(value)
                                if hasattr(value, "__iter__")
                                else str(value)
                            )
                        else:
                            report_dict[key] = value
                    except (ValueError, TypeError):
                        # Fallback for problematic values
                        report_dict[key] = str(value)
                graph_data["community_reports"].append(report_dict)

        with open(filepath, "w") as f:
            json.dump(graph_data, f, indent=2)

        print(f"GraphRAG knowledge graph saved to {filepath}")

    def query_global(self, query: str) -> str:
        """Perform global search using GraphRAG"""
        if not self.temp_dir:
            raise ValueError("GraphRAG project not initialized")

        try:
            result = subprocess.run(
                [
                    "graphrag",
                    "query",
                    "--root",
                    self.temp_dir,
                    "--method",
                    "global",
                    "--query",
                    query,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout

        except subprocess.CalledProcessError as e:
            print(f"Error in global query: {e.stderr}")
            return f"Error: {e.stderr}"

    def query_local(self, query: str) -> str:
        """Perform local search using GraphRAG"""
        if not self.temp_dir:
            raise ValueError("GraphRAG project not initialized")

        try:
            result = subprocess.run(
                [
                    "graphrag",
                    "query",
                    "--root",
                    self.temp_dir,
                    "--method",
                    "local",
                    "--query",
                    query,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout

        except subprocess.CalledProcessError as e:
            print(f"Error in local query: {e.stderr}")
            return f"Error: {e.stderr}"


def main():
    """Test the GraphRAG builder"""
    builder = GraphRAGBuilder(temp_dir="graphrag")
    graph_json_path = "data/europe_knowledge_graph.json"

    try:
        # If the graph JSON already exists, don't build it again
        if os.path.exists(graph_json_path):
            print(f"Graph JSON already exists at {graph_json_path}. Skipping build.")
            # Optionally, you can load the graph for stats
            loader = GraphRAGLoader(graph_json_path)
            graph = loader.load_graph()
        else:
            # Check if we have an existing index to use
            index_path = "graphrag"  # Use the test index we created
            if os.path.exists(os.path.join(index_path, "output")):
                print("Using existing GraphRAG index...")
                graph = builder.build_graph_from_existing_index(index_path)
                # Save the graph after building
                builder.save_graph(graph_json_path)
            else:
                print(
                    "No existing index found. Please build GraphRAG index using CLI first."
                )
                return

        print("\n" + "=" * 50)
        print("GRAPHRAG KNOWLEDGE GRAPH STATISTICS")
        print("=" * 50)
        print(f"Nodes: {graph.number_of_nodes()}")
        print(f"Edges: {graph.number_of_edges()}")
        if builder.communities is not None:
            print(f"Communities: {len(builder.communities)}")
        print("=" * 50)

        # Test queries (only if we have the temp_dir for queries)
        if builder.temp_dir and os.path.exists(builder.temp_dir):
            print("\nTesting Global Query...")
            global_result = builder.query_global(
                "What are the main themes in the documents?"
            )
            print(f"Global Result: {global_result}...")

            print("\nTesting Local Query...")
            local_result = builder.query_local("What is Finland known for?")
            print(f"Local Result: {local_result}...")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
