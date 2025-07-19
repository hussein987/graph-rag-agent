#!/usr/bin/env python3

import os
import sys
import asyncio
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.agents.graph_agent import create_agent
from src.knowledge.vector_store_builder import VectorStoreBuilder
from src.knowledge.graph_builder import GraphRAGLoader
from src.utils.config import config, setup_logging


def setup_knowledge_base():
    """Initialize vector store (GraphRAG knowledge graph is pre-built)."""
    print("🔧 Setting up knowledge base...")

    # Check if GraphRAG knowledge graph exists
    graph_loader = GraphRAGLoader()
    try:
        graph = graph_loader.load_graph()
        print(f"📊 Found GraphRAG knowledge graph:")
        print(f"   - {graph.number_of_nodes()} nodes")
        print(f"   - {graph.number_of_edges()} edges")
        print(f"   - Generated with Microsoft GraphRAG")
    except FileNotFoundError:
        print(
            "⚠️  GraphRAG knowledge graph not found at data/graphrag_knowledge_graph.json"
        )
        print("   Please run the GraphRAG builder first to create the knowledge graph")
        return None

    # Build vector store for semantic search
    print("🔍 Building vector store for semantic search...")
    vector_builder = VectorStoreBuilder()
    vector_store = vector_builder.build_vector_store()
    if vector_store:
        print(
            f"   - Vector store created with {len(vector_builder.documents)} documents"
        )
    else:
        print("   - No documents found in corpus directory")

    print("✅ Knowledge base setup complete!")
    return {"graph_loader": graph_loader, "vector_builder": vector_builder}


def print_response(result: Dict[str, Any]):
    """Pretty print the agent response."""
    print("\n" + "=" * 60)
    print("🤖 AGENT RESPONSE")
    print("=" * 60)
    print(result["messages"][-1].content)

    print("\n" + "-" * 40)
    print("📊 METADATA")
    print("-" * 40)
    print(
        f"Tools used: {', '.join(result['tools_used']) if result['tools_used'] else 'None'}"
    )
    print(f"Retry count: {result['retry_count']}")
    print(f"Confidence evaluations: {result['metadata']['total_evaluations']}")

    if result["metadata"]["low_confidence_warning"]:
        print("⚠️  WARNING: Low confidence response - verify information")

    # Show confidence scores if available
    if result["confidence_scores"]:
        print("\n📈 CONFIDENCE SCORES:")
        for i, eval_result in enumerate(result["confidence_scores"]):
            if "scores" in eval_result:
                scores = eval_result["scores"]
                print(f"  Evaluation {i+1}:")
                print(f"    Combined: {scores.get('combined_score', 'N/A'):.2f}")
                print(f"    RAG: {scores.get('rag_score', 'N/A'):.2f}")
                print(
                    f"    KG Consistency: {scores.get('kg_consistency_score', 'N/A'):.2f}"
                )
                print(
                    f"    Self-Consistency: {scores.get('self_consistency_score', 'N/A'):.2f}"
                )


def run_demo_queries():
    """Run several demo queries to showcase the agent."""

    # Setup
    kb_result = setup_knowledge_base()
    if kb_result is None:
        print("❌ Cannot run demo without knowledge base")
        return

    agent = create_agent()

    demo_queries = [
        "Plan a 24-hour trip to Helsinki including weather and top attractions",
        "What is the traditional Finnish culture like and what should I know before visiting?",
        "How much does accommodation cost in Helsinki and what are the transportation options?",
        "Tell me about Finnish sauna culture and design traditions",
    ]

    print("\n🚀 Starting Zero-Hallucination Agent Demo")
    print("=" * 60)

    # Use consistent thread ID for the demo to maintain conversation context
    thread_id = "demo_session"

    for i, query in enumerate(demo_queries, 1):
        print(f"\n🔥 DEMO QUERY {i}")
        print(f"Query: {query}")

        try:
            result = agent.run(query, thread_id=thread_id)
            print_response(result)

        except Exception as e:
            print(f"❌ Error processing query: {str(e)}")

        # Add separator between queries
        if i < len(demo_queries):
            print("\n" + "." * 60)
            input("Press Enter to continue to next query...")


def interactive_mode():
    """Run the agent in interactive mode."""
    kb_result = setup_knowledge_base()
    if kb_result is None:
        print("❌ Cannot run interactive mode without knowledge base")
        return

    agent = create_agent()

    print("\n🚀 Zero-Hallucination Agent - Interactive Mode")
    print("=" * 60)
    print("Ask me anything about Helsinki or Finnish culture!")
    print("Type 'quit' to exit, 'help' for more information.")
    print("💡 Your conversation history is preserved throughout the session.")
    print("=" * 60)

    # Use a consistent thread ID for the entire interactive session
    thread_id = "interactive_session"

    while True:
        try:
            query = input("\n💬 Your question: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break

            if query.lower() == "help":
                print(agent.get_graph_visualization())
                continue

            if not query:
                print("Please enter a question.")
                continue

            print("\n🤔 Thinking...")
            result = agent.run(query, thread_id=thread_id)
            print_response(result)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


async def async_demo():
    """Demonstrate async capabilities."""
    kb_result = setup_knowledge_base()
    if kb_result is None:
        print("❌ Cannot run async demo without knowledge base")
        return

    agent = create_agent()

    print("\n🚀 Async Demo - Processing Multiple Queries Concurrently")
    print("=" * 60)

    queries = [
        "What's the weather like in Helsinki?",
        "Tell me about Finnish design",
        "How do I get to Helsinki?",
    ]

    # Process all queries concurrently with different thread IDs
    tasks = [
        agent.arun(query, thread_id=f"async_{i}") for i, query in enumerate(queries)
    ]
    results = await asyncio.gather(*tasks)

    for query, result in zip(queries, results):
        print(f"\nQuery: {query}")
        print_response(result)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            run_demo_queries()
        elif sys.argv[1] == "interactive":
            interactive_mode()
        elif sys.argv[1] == "async":
            asyncio.run(async_demo())
        elif sys.argv[1] == "setup":
            setup_knowledge_base()
        else:
            print("Usage: python main.py [demo|interactive|async|setup]")
    else:
        # Default to interactive mode
        interactive_mode()


if __name__ == "__main__":
    # Setup logging first
    setup_logging()

    # Check if OpenAI API key is set
    if not config.openai_api_key or config.openai_api_key == "your_openai_api_key_here":
        print("❌ Error: Please set your OPENAI_API_KEY in a .env file")
        print("Copy .env.example to .env and add your API key")
        sys.exit(1)

    main()
