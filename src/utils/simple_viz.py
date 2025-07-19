#!/usr/bin/env python3
import json
import networkx as nx
from pyvis.network import Network
import webbrowser
import os


def create_spring_graph():
    # Load the knowledge graph
    with open("data/graphrag_knowledge_graph.json") as f:
        data = json.load(f)

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    for node in data["nodes"]:
        G.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})

    # Add edges
    for edge in data["edges"]:
        G.add_edge(
            edge["source"],
            edge["target"],
            **{k: v for k, v in edge.items() if k not in ["source", "target"]},
        )

    # Create PyVis network
    net = Network(height="100vh", width="100%", directed=True, bgcolor="#f8f9fa")

    # Configure physics for spring layout
    net.set_options(
        """
    var options = {
        "physics": {
            "enabled": true,
            "stabilization": {"enabled": true, "iterations": 1000},
            "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.3,
                "springLength": 200,
                "springConstant": 0.3,
                "damping": 0.9,
                "avoidOverlap": 0.1
            }
        },
        "interaction": {
            "dragNodes": true,
            "dragView": true,
            "zoomView": true
        }
    }
    """
    )

    # Add nodes with colors
    colors = {
        "entity": "#3498db",
        "concept": "#e74c3c",
        "person": "#2ecc71",
        "place": "#f39c12",
        "organization": "#9b59b6",
        "event": "#1abc9c",
    }

    for node in G.nodes():
        node_data = G.nodes[node]
        size = max(20, G.degree(node) * 5 + 15)
        color = colors.get(node_data.get("type", "entity"), "#3498db")

        net.add_node(
            node,
            label=node_data.get("label", node),
            color=color,
            size=size,
            title=f"ID: {node}<br>Type: {node_data.get('type', 'entity')}<br>Degree: {G.degree(node)}",
        )

    # Add edges
    for edge in G.edges():
        edge_data = G.edges[edge]
        net.add_edge(
            edge[0],
            edge[1],
            title=f"{edge[0]} → {edge[1]}<br>{edge_data.get('relation', 'related')}",
        )

    # Save and show
    net.save_graph("spring_graph.html")
    print("Spring physics graph saved to spring_graph.html")

    # Open in browser
    webbrowser.open(f'file://{os.path.abspath("spring_graph.html")}')


if __name__ == "__main__":
    create_spring_graph()
