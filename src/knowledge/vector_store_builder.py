#!/usr/bin/env python3
"""
Vector Store Builder Module

This module provides the VectorStoreBuilder class for building FAISS vector stores
from text documents for semantic search.
"""

import os
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.utils.config import config


class VectorStoreBuilder:
    """
    Builds FAISS vector stores from text documents.
    Used by the GraphRetriever for semantic search.
    """

    def __init__(
        self, corpus_path: str = None, vector_store_path: str = "data/faiss_index"
    ):
        self.corpus_path = corpus_path or config.corpus_path
        self.vector_store_path = vector_store_path
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
        self.vector_store = None
        self.documents = []

    def load_documents(self) -> List[Document]:
        """Load documents from corpus directory."""
        if not os.path.exists(self.corpus_path):
            os.makedirs(self.corpus_path)
            return []

        loader = DirectoryLoader(
            self.corpus_path, glob="**/*.txt", loader_cls=TextLoader
        )
        documents = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        self.documents = text_splitter.split_documents(documents)
        return self.documents

    def build_vector_store(self) -> FAISS:
        """Build FAISS vector store for semantic search."""
        # Check if vector store already exists on disk
        if os.path.exists(self.vector_store_path):
            print(f"Loading existing FAISS vector store from {self.vector_store_path}")
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return self.vector_store
            except Exception as e:
                print(f"Error loading existing vector store: {e}")
                print("Rebuilding vector store...")

        # Load documents if not already loaded
        if not self.documents:
            self.load_documents()

        if self.documents:
            print(
                f"Building new FAISS vector store with {len(self.documents)} documents"
            )
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)

            # Save to disk
            print(f"Saving FAISS vector store to {self.vector_store_path}")
            os.makedirs(os.path.dirname(self.vector_store_path), exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
        else:
            print("No documents found to build vector store")

        return self.vector_store
