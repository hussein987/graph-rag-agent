#!/usr/bin/env python3
"""
Traditional RAG Implementation
Pure vector similarity search without graph traversal
"""

import os
import sys
import time
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add parent directories to path for accessing main project
parent_dir = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(parent_dir)
from src.utils.config import config


class TraditionalRAG:
    """Traditional RAG using only vector similarity search."""

    def __init__(
        self,
        vector_store_path: str = "data/faiss_index",
        corpus_path: str = "data/corpus_europe",
    ):
        self.vector_store_path = vector_store_path
        self.corpus_path = corpus_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=config.openai_api_key)
        self.vector_store = None
        self.load_or_create_vector_store()

    def load_or_create_vector_store(self):
        """Load existing vector store or create new one from corpus."""
        if os.path.exists(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                print(
                    f"✅ Loaded traditional RAG vector store from {self.vector_store_path}"
                )
                return
            except Exception as e:
                print(f"❌ Error loading vector store: {e}")
                print("🔄 Will create new vector store...")

        # Create new vector store
        print(f"🔄 Creating new vector store from {self.corpus_path}")
        self.create_vector_store()

    def create_vector_store(self):
        """Create a new vector store from the corpus."""
        if not os.path.exists(self.corpus_path):
            print(f"❌ Corpus directory not found at {self.corpus_path}")
            self.vector_store = None
            return

        try:
            # Load documents from corpus
            documents = []
            text_files = [f for f in os.listdir(self.corpus_path) if f.endswith(".txt")]

            if not text_files:
                print(f"❌ No .txt files found in {self.corpus_path}")
                self.vector_store = None
                return

            print(f"📄 Loading {len(text_files)} documents from corpus...")

            for filename in text_files:
                file_path = os.path.join(self.corpus_path, filename)
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = filename
                        documents.append(doc)
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
                    continue

            if not documents:
                print("❌ No documents loaded from corpus")
                self.vector_store = None
                return

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )

            print(f"✂️ Splitting {len(documents)} documents into chunks...")
            splits = text_splitter.split_documents(documents)
            print(f"📑 Created {len(splits)} document chunks")

            # Create vector store
            print("🔄 Creating FAISS vector store...")
            self.vector_store = FAISS.from_documents(splits, self.embeddings)

            # Save vector store
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)

            print(f"✅ Created and saved vector store to {self.vector_store_path}")

        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            self.vector_store = None

    def retrieve(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Retrieve documents using traditional vector similarity search.

        Args:
            query: Search query
            k: Number of documents to retrieve

        Returns:
            Dictionary with results and metadata
        """
        if not self.vector_store:
            return {
                "documents": [],
                "method": "traditional_rag",
                "query": query,
                "error": "Vector store not available",
                "retrieval_time": 0,
                "total_documents": 0,
            }

        start_time = time.time()

        try:
            # Simple vector similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k
            )

            # Convert to Document objects with metadata
            documents = []
            for doc, score in docs_with_scores:
                doc.metadata["similarity_score"] = score
                doc.metadata["retrieval_method"] = "vector_similarity"
                documents.append(doc)

            # Generate comprehensive answer using LLM
            print("🤖 Generating comprehensive answer using LLM...")
            llm_answer = self.generate_answer(query, documents)

            # Create a document for the LLM-generated answer
            answer_doc = Document(
                page_content=llm_answer,
                metadata={
                    "source": "traditional_rag_llm_answer",
                    "search_type": "llm_synthesis",
                    "retrieval_method": "traditional_rag",
                    "is_generated_answer": True,
                },
            )

            # Add the answer document at the beginning
            documents.insert(0, answer_doc)

            retrieval_time = time.time() - start_time

            return {
                "documents": documents,
                "method": "traditional_rag",
                "query": query,
                "retrieval_time": retrieval_time,
                "total_documents": len(documents),
                "scores": [score for _, score in docs_with_scores],
                "strategy": "Vector similarity search + LLM synthesis",
                "llm_answer": llm_answer,
            }

        except Exception as e:
            print(f"❌ Traditional RAG retrieval failed: {e}")
            return {
                "documents": [],
                "method": "traditional_rag",
                "query": query,
                "error": str(e),
                "retrieval_time": time.time() - start_time,
                "total_documents": 0,
                "scores": [],
                "strategy": "Failed - Traditional RAG error",
            }

    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the vector store."""
        if not self.vector_store:
            return {"error": "Vector store not available"}

        # Get document count (approximate)
        try:
            # Sample a few documents to get info
            sample_docs = self.vector_store.similarity_search("sample", k=3)
            sources = set()
            for doc in sample_docs:
                if "source" in doc.metadata:
                    sources.add(doc.metadata["source"])

            return {
                "method": "traditional_rag",
                "vector_store_type": "FAISS",
                "embedding_model": "text-embedding-ada-002",
                "sample_sources": list(sources),
                "retrieval_strategy": "Vector similarity search only",
            }
        except:
            return {
                "method": "traditional_rag",
                "vector_store_type": "FAISS",
                "embedding_model": "text-embedding-ada-002",
                "retrieval_strategy": "Vector similarity search only",
            }

    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate a comprehensive answer using LLM and retrieved documents."""
        if not documents:
            return "No relevant documents found to answer the question."

        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(
                f"Document {i} (Source: {source}):\n{doc.page_content}\n"
            )

        context = "\n".join(context_parts)

        # Create prompt for LLM
        prompt = f"""Based on the following retrieved documents, provide a comprehensive and well-structured answer to the question. Use markdown formatting for better readability.

Question: {query}

Context from retrieved documents:
{context}

Please provide a detailed answer that:
1. Synthesizes information from the provided documents
2. Uses proper markdown formatting (headers, lists, etc.)
3. Cites which documents support different points
4. Is well-organized and easy to read
5. Does not contain any information that is not in the retrieved documents


IMPORTANT: ONLY USE THE INFORMATION FROM THE RETRIEVED DOCUMENTS TO ANSWER THE QUESTION. DO NOT MAKE UP ANY INFORMATION OR USE YOUR OWN KNOWLEDGE.

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"


def create_traditional_rag() -> TraditionalRAG:
    """Factory function to create a traditional RAG instance."""
    return TraditionalRAG()


if __name__ == "__main__":
    # Test the traditional RAG implementation
    rag = create_traditional_rag()

    test_query = "What are the main countries in Europe?"
    print(f"Testing traditional RAG with query: '{test_query}'")

    results = rag.retrieve(test_query)
    print(f"Retrieved {results['total_documents']} documents")
    print(f"Retrieval time: {results['retrieval_time']:.3f}s")

    for i, doc in enumerate(results["documents"]):
        print(f"\n--- Document {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Score: {doc.metadata.get('similarity_score', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")
