# RAG Comparison Tool 🔍

A comprehensive comparison tool for **Traditional RAG** vs **Graph-Enhanced RAG** approaches, featuring a beautiful web interface and detailed performance analysis.

## 🎯 Overview

This tool demonstrates the key differences between:

1. **Traditional RAG** - Pure vector similarity search
2. **Graph-Enhanced RAG** - Vector search + Graph traversal using LangChain's GraphRetriever

## 📊 Key Features

- **Side-by-side comparison** of both approaches
- **Performance metrics** including retrieval time, document overlap, and score distributions
- **Interactive web interface** with real-time charts and visualizations
- **Batch comparison** for testing multiple queries
- **Detailed document analysis** with similarity scores and metadata
- **Graph traversal insights** showing entity matching and community detection

## 🏗️ Architecture

```
Traditional RAG:
Query → Vector Embedding → Similarity Search → Top-K Documents

Graph-Enhanced RAG:
Query → Vector Embedding → Initial Documents → Graph Traversal → 
Connected Entities → Expanded Context → Ranked Results
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Flask for the web server
pip install flask flask-cors

# Ensure parent project dependencies are installed
cd .. && pip install -r requirements.txt
```

### 2. Start the Web Server

```bash
cd rag_comparison
python server.py
```

### 3. Access the Interface

Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Project Structure

```
rag_comparison/
├── src/
│   ├── traditional_rag.py     # Traditional RAG implementation
│   ├── graph_rag.py           # Graph-Enhanced RAG implementation
│   └── comparison_tool.py     # Comparison logic and metrics
├── comparison_ui.html         # Beautiful web interface
├── server.py                  # Flask web server
├── README.md                  # This file
└── results/                   # Generated comparison results
```

## 🔧 Components Explained

### Traditional RAG (`traditional_rag.py`)
- Uses only **FAISS vector similarity search**
- Straightforward embedding-based retrieval
- Fast but limited to surface-level similarity

### Graph-Enhanced RAG (`graph_rag.py`)
- Uses **LangChain's GraphRetriever**
- Combines vector search with graph traversal
- Leverages entity relationships and community detection
- Falls back to vector search if graph traversal fails

### Comparison Tool (`comparison_tool.py`)
- **Performance metrics**: Retrieval time, speedup calculations
- **Document overlap analysis**: How much content overlaps between approaches
- **Score distributions**: Similarity score statistics
- **Graph traversal success rate**: How often graph enhancement works

## 📈 Key Metrics

### Performance Metrics
- **Retrieval Time**: Speed comparison between approaches
- **Document Count**: Number of retrieved documents
- **Similarity Scores**: Mean, min, max similarity values

### Quality Metrics
- **Document Overlap**: Percentage of shared content
- **Graph Traversal Success**: Whether graph enhancement worked
- **Unique Content**: Documents found only by each approach

## 🎨 Web Interface Features

### Main Dashboard
- **Query Input**: Test with custom queries or predefined samples
- **Side-by-side Results**: Compare both approaches visually
- **Performance Charts**: Interactive visualizations using Chart.js

### Comparison Views
- **Document Lists**: Full content with metadata and scores
- **Metrics Cards**: Key performance indicators
- **Charts**: Speed comparison and score distributions

### Batch Analysis
- **Multiple Query Testing**: Run comparisons on query sets
- **Aggregate Reports**: Summary statistics across all queries
- **Export Results**: Save comparisons to JSON files

## 🔍 How GraphRetriever Works

The GraphRetriever combines two retrieval strategies:

1. **Initial Vector Search**: Find semantically similar documents
2. **Graph Traversal**: Explore connected entities using defined edges:
   - `("primary_type", "primary_type")` - Connect by entity types
   - `("primary_community", "primary_community")` - Connect by communities
   - `("entities", "entities")` - Connect by shared entities

### Strategy Configuration
```python
strategy = Eager(k=5, start_k=1, max_depth=2)
```
- Start with 1 document from vector search
- Expand to 5 total through graph traversal
- Maximum 2 hops in the graph

## 🧪 Testing Queries

### Sample Queries Included
1. **"Finnish sauna culture and traditions"**
2. **"Helsinki travel guide recommendations"**
3. **"Traditional Finnish cuisine"**
4. **"Finnish education system"**
5. **"Finnish architecture history"**

### What to Look For
- **Speed differences**: Traditional RAG is usually faster
- **Content diversity**: Graph RAG often finds more diverse content
- **Entity connections**: Graph RAG leverages entity relationships
- **Fallback behavior**: How Graph RAG handles failures

## 🎯 Expected Results

### Traditional RAG
- ✅ **Faster retrieval** (simple vector search)
- ✅ **Consistent performance** (no graph dependencies)
- ❌ **Limited to surface similarity** (no entity relationships)

### Graph-Enhanced RAG
- ✅ **Richer context** (entity relationships)
- ✅ **Better semantic understanding** (graph traversal)
- ✅ **Community-based connections** (related concepts)
- ❌ **Slower retrieval** (additional graph processing)
- ❌ **Graph dependency** (fallback to vector search if graph fails)

## 📊 API Endpoints

### Single Comparison
```bash
POST /api/compare
{
  "query": "Finnish sauna culture",
  "k": 5
}
```

### Batch Comparison
```bash
POST /api/batch_compare
{
  "queries": ["query1", "query2"],
  "k": 5
}
```

### System Information
```bash
GET /api/system_info
```

### Graph Traversal Info
```bash
POST /api/graph_info
{
  "query": "Finnish sauna culture"
}
```

## 🔧 Configuration

### Customizing Retrieval
Edit `graph_rag.py` to modify:
- **Edge definitions**: Change relationship types
- **Strategy parameters**: Adjust k, start_k, max_depth
- **Graph metadata**: Modify entity extraction logic

### UI Customization
Edit `comparison_ui.html` to:
- Change visual styling
- Add new metrics displays
- Modify chart configurations

## 🚦 Troubleshooting

### Common Issues

1. **Vector store not found**
   - Ensure FAISS index exists at `../data/faiss_index`
   - Run the main project setup first

2. **Graph not found**
   - Ensure knowledge graph exists at `../data/graphrag_knowledge_graph.json`
   - Run GraphRAG builder if needed

3. **API connection errors**
   - Check if Flask server is running
   - Verify API endpoints are accessible

### Performance Tips

1. **For faster testing**: Use smaller k values (e.g., k=3)
2. **For better graphs**: Ensure good entity extraction
3. **For reliability**: Monitor fallback behavior

## 📝 Example Usage

```python
# Direct Python usage
from src.comparison_tool import RAGComparator

comparator = RAGComparator()
result = comparator.compare_retrieval("Finnish sauna culture", k=5)

print(f"Traditional time: {result.traditional_results['retrieval_time']:.3f}s")
print(f"Graph time: {result.graph_results['retrieval_time']:.3f}s")
print(f"Overlap: {result.comparison_metrics['document_overlap']['overlap_percentage']:.1f}%")
```

## 🎯 Learning Objectives

This tool demonstrates:
1. **RAG approach trade-offs** - Speed vs. quality
2. **Graph-enhanced retrieval** - When and why it helps
3. **Performance evaluation** - Metrics that matter
4. **Fallback strategies** - Handling failures gracefully
5. **Web interface design** - Making complex comparisons accessible

## 📚 Further Reading

- [LangChain GraphRetriever Documentation](https://python.langchain.com/docs/integrations/retrievers/graph)
- [Microsoft GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [RAG Evaluation Metrics](https://docs.ragas.io/en/latest/)

---

**Built with ❤️ for better RAG understanding** 