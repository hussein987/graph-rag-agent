# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

## Development Commands

### Setup and Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and optional WEATHER_API_KEY

# IMPORTANT: Knowledge graph is pre-built using GraphRAG
# If you need to rebuild it, use the GraphRAG builder CLI:
python src/knowledge/graphrag_builder.py

# Setup vector store (GraphRAG knowledge graph is loaded automatically)
python main.py setup
```

### Running the Agent
```bash
python main.py interactive  # Interactive mode (default)
python main.py demo         # Run pre-built demo queries
python main.py async        # Demonstrate async capabilities
python main.py setup        # Setup vector store (knowledge graph is pre-built)
```

### Testing
```bash
python tests/test_agent.py                    # Run basic agent tests
python tests/test_graphrag_integration.py     # Test GraphRAG integration
python -m pytest tests/ -v                   # Run all tests with pytest
```

### Environment Variables
Key configuration in `.env`:
- `OPENAI_API_KEY`: Required for LLM operations
- `WEATHER_API_KEY`: Optional for weather tool functionality
- `MODEL_NAME`: Defaults to "gpt-4"
- `KG_CONFIDENCE_THRESHOLD`: Retry threshold (default: 0.15)
- `CORPUS_PATH`: Knowledge base documents path (default: "data/corpus")

## Architecture

### LangGraph Workflow
The agent uses a state machine with these nodes:
1. **Planner**: Classifies queries and routes them appropriately
2. **Executor**: Automatically selects and calls appropriate tools (local_search, global_search, weather)
3. **Response Formatter**: Formats tool outputs into natural language responses
4. **Critic**: Evaluates output using 3-score ensemble evaluation system

### Key Components

**Agent State** (`src/agents/nodes.py:12-21`): TypedDict state schema for LangGraph 0.5 compatibility:
```python
class AgentState(TypedDict):
    input_text: str
    plan: List[Dict[str, Any]]
    retrievals: List[tuple]
    outputs: List[Dict[str, Any]]
    final_response: str
    retry_count: int
    evaluation: Dict[str, Any]
    tools_used: List[str]
```

**Knowledge Graph** (`data/graphrag_knowledge_graph.json`): Microsoft GraphRAG-generated knowledge graph with entities, relationships, communities, and hierarchical summaries. Pre-built using the GraphRAG pipeline and loaded via `GraphRAGLoader` class.

**Ensemble Scoring** (`src/evaluation/ensemble.py`): Three-signal confidence scoring:
- RAG Score (40% weight): Semantic similarity between response and context
- KG Consistency (40% weight): Percentage of facts verified in knowledge graph  
- Self-Consistency (20% weight): Agreement between multiple responses

**Tool System** (`src/tools/`): MCP-compatible tools (currently weather API). Tools are automatically called based on plan requirements.

### Retry Logic
- Combined confidence score < 0.15 triggers automatic retry
- Maximum 2 retries to prevent infinite loops
- Low confidence responses include warning labels

## Code Organization

- `src/agents/`: Main LangGraph agent and workflow nodes
- `src/knowledge/`: Knowledge graph building and retrieval
- `src/evaluation/`: Hallucination detection and confidence scoring
- `src/tools/`: External tools following MCP patterns
- `src/utils/`: Configuration and text processing utilities
- `data/corpus/`: Knowledge base documents (.txt files)
- `tests/`: Unit tests for core functionality

## Development Notes

### Adding New Tools
1. Create tool class in `src/tools/` following `WeatherTool` pattern
2. Register in `MCPAdapter` 
3. Update planner prompts to recognize new task types

### Expanding Knowledge Base
1. Add `.txt` documents to `data/corpus/`
2. Rebuild GraphRAG knowledge graph: `python src/knowledge/graphrag_builder.py`
3. Restart agent to load new graph: `python main.py setup`

**Note**: The knowledge graph is now built using Microsoft GraphRAG, not the old NetworkX approach. The GraphRAG builder handles entity extraction, community detection, and relationship mapping automatically.

### Modifying Confidence Thresholds
Adjust weights in `EnsembleScorer` initialization or environment variables:
- `KG_CONFIDENCE_THRESHOLD`: Overall retry threshold
- Scorer weights in `ensemble.py:100-104`

### LangGraph 0.5 Critical Implementation Details

**Node Functions Must Return Dictionaries**: 
```python
# ❌ Wrong (old LangGraph pattern):
def node(state: AgentState) -> AgentState:
    state.field = new_value
    return state

# ✅ Correct (LangGraph 0.5):
def node(state: AgentState) -> Dict[str, Any]:
    return {"field": new_value}
```

**State Access**:
```python
# Access state values with dictionary syntax
input_text = state["input_text"]
plan = state.get("plan", [])  # With default
```

**Imports**: Use `from langgraph.graph import StateGraph, END` (not deprecated decorator patterns)

### Common Issues
- "GraphRAG knowledge graph not found": Run `python src/knowledge/graphrag_builder.py` to build the knowledge graph first
- "No documents found": Add `.txt` files to `data/corpus/` and run setup
- API key errors: Ensure `.env` file exists with valid `OPENAI_API_KEY`
- Low confidence warnings: Increase corpus size or lower threshold
- LangGraph import errors: Ensure `langgraph>=0.5.0` and use proper imports
- "Field required" validation: Check `.env` file is properly loaded with required keys
- Vector store issues: Delete `data/faiss_index` directory to force rebuild