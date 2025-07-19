#!/usr/bin/env python3
"""
GraphRAG Retrieval System
Working GraphRAG implementation using official GraphRAG library
"""

import os
import pandas as pd
import asyncio
import tiktoken
from typing import List, Dict, Any
import time

# --- Monkey‑patch aiolimiter's AsyncLimiter for fnllm's rate limiter ---
from aiolimiter import AsyncLimiter
import asyncio as _asyncio

# Provide a _loop property that returns the running loop
AsyncLimiter._loop = property(lambda self: _asyncio.get_running_loop())
# Provide a no-op _wake_next so fnllm's update_limiter calls succeed
AsyncLimiter._wake_next = lambda self: None

# --- Import GraphRAG components ---
from graphrag.query.indexer_adapters import (
    read_indexer_communities,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.structured_search.global_search.search import GlobalSearch
from graphrag.query.structured_search.global_search.community_context import (
    GlobalCommunityContext,
)
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.language_model.manager import ModelManager
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.enums import ModelType


class GraphRAGRetrieval:
    """Working GraphRAG implementation using official GraphRAG library."""

    def __init__(self, input_dir: str = "graphrag/output"):
        self.input_dir = input_dir
        self.lancedb_uri = f"{input_dir}/lancedb"

        # Will hold our loaded data
        self.communities = None
        self.entities = None
        self.relationships = None
        self.reports = None
        self.text_units = None

        # Vector store & models
        self.description_store = None
        self.chat_model = None
        self.text_embedder = None
        self.token_encoder = None

        # Cache for search engines
        self._local_search_engine = None
        self._global_search_engine = None

        self._setup_components()

    def _setup_components(self):
        """Setup GraphRAG components."""
        try:
            # ---------- Load Parquet data ----------
            entity_df = pd.read_parquet(f"{self.input_dir}/entities.parquet")
            relationship_df = pd.read_parquet(f"{self.input_dir}/relationships.parquet")
            report_df = pd.read_parquet(f"{self.input_dir}/community_reports.parquet")
            text_unit_df = pd.read_parquet(f"{self.input_dir}/text_units.parquet")
            communities_df = pd.read_parquet(f"{self.input_dir}/communities.parquet")

            print(f"✅ Loaded GraphRAG data:")
            print(f"  - Entities: {len(entity_df)} rows")
            print(f"  - Relationships: {len(relationship_df)} rows")
            print(f"  - Community reports: {len(report_df)} rows")
            print(f"  - Text units: {len(text_unit_df)} rows")
            print(f"  - Communities: {len(communities_df)} rows")

            self.communities = read_indexer_communities(communities_df, report_df)
            self.entities = read_indexer_entities(
                entity_df, communities_df, community_level=0
            )
            self.relationships = read_indexer_relationships(relationship_df)
            self.reports = read_indexer_reports(
                report_df, communities_df, community_level=0
            )
            self.text_units = read_indexer_text_units(text_unit_df)

            # ---------- Connect to LanceDB ----------
            self.description_store = LanceDBVectorStore(
                collection_name="default-entity-description"
            )
            self.description_store.connect(db_uri=self.lancedb_uri)
            print(f"✅ Connected to LanceDB at: {self.lancedb_uri}")

            # ---------- Initialize Chat Model ----------
            chat_cfg = LanguageModelConfig(
                api_key=os.environ.get("OPENAI_API_KEY"),
                type=ModelType.OpenAIChat,
                model="gpt-4o-mini",
                max_retries=1,
            )
            self.chat_model = ModelManager().get_or_create_chat_model(
                name="graphrag_chat", model_type=ModelType.OpenAIChat, config=chat_cfg
            )

            # ---------- Initialize Embedding Model ----------
            embed_cfg = LanguageModelConfig(
                api_key=os.environ.get("OPENAI_API_KEY"),
                type=ModelType.OpenAIEmbedding,
                model="text-embedding-3-small",
                max_retries=1,
            )
            self.text_embedder = ModelManager().get_or_create_embedding_model(
                name="graphrag_embedding",
                model_type=ModelType.OpenAIEmbedding,
                config=embed_cfg,
            )

            # ---------- Token Encoder ----------
            self.token_encoder = tiktoken.encoding_for_model("gpt-4o-mini")

            print("✅ GraphRAG components initialized successfully")

        except Exception as e:
            print(f"❌ Error setting up GraphRAG components: {e}")
            raise

    def _get_local_search(self) -> LocalSearch:
        """Get or create local search engine."""
        if self._local_search_engine is None:
            print("🔧 Creating local search engine (cached for reuse)...")
            ctx = LocalSearchMixedContext(
                community_reports=self.reports,
                text_units=self.text_units,
                entities=self.entities,
                relationships=self.relationships,
                covariates=None,
                entity_text_embeddings=self.description_store,
                embedding_vectorstore_key=EntityVectorStoreKey.ID,
                text_embedder=self.text_embedder,
                token_encoder=self.token_encoder,
            )
            local_params = {
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 0,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 5,
                "top_k_relationships": 3,
                "include_entity_rank": False,
                "include_relationship_weight": False,
                "include_community_rank": False,
                "return_candidate_context": False,
                "embedding_vectorstore_key": EntityVectorStoreKey.ID,
                "max_tokens": 12_000,
            }
            model_params = {"max_tokens": 2_000, "temperature": 0.0}

            self._local_search_engine = LocalSearch(
                model=self.chat_model,
                context_builder=ctx,
                token_encoder=self.token_encoder,
                model_params=model_params,
                context_builder_params=local_params,
                response_type="multiple paragraphs",
            )
            print(f"✅ Local search engine created and cached")
        return self._local_search_engine

    def _get_global_search(self) -> GlobalSearch:
        """Get or create global search engine."""
        if self._global_search_engine is None:
            print("🔧 Creating global search engine (cached for reuse)...")
            ctx = GlobalCommunityContext(
                community_reports=self.reports,
                communities=self.communities,
                entities=self.entities,
                token_encoder=self.token_encoder,
            )
            global_params = {
                "use_community_summary": True,
                "shuffle_data": True,
                "include_community_rank": True,
                "min_community_rank": 0,
                "community_rank_name": "rank",
                "include_community_weight": True,
                "community_weight_name": "occurrence weight",
                "normalize_community_weight": True,
                "max_tokens": 12_000,
                "context_name": "Reports",
            }
            map_params = {
                "max_tokens": 1_000,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            }
            reduce_params = {"max_tokens": 2_000, "temperature": 0.0}

            self._global_search_engine = GlobalSearch(
                model=self.chat_model,
                context_builder=ctx,
                token_encoder=self.token_encoder,
                max_data_tokens=1_000,
                map_llm_params=map_params,
                reduce_llm_params=reduce_params,
                allow_general_knowledge=False,
                json_mode=True,
                context_builder_params=global_params,
                concurrent_coroutines=16,
                response_type="multiple paragraphs",
            )
            print(f"✅ Global search engine created and cached")
        return self._global_search_engine

    async def local_search(self, query: str):
        """Perform local search."""
        try:
            print(f"🔍 Starting local search for: {query}")
            import time

            start = time.time()
            search_engine = self._get_local_search()
            engine_ready = time.time()
            print(f"⏱️ Search engine ready in {engine_ready - start:.2f}s")
            result = await search_engine.search(query)
            end = time.time()
            print(
                f"⏱️ Local search completed in {end - start:.2f}s (engine: {engine_ready - start:.2f}s, search: {end - engine_ready:.2f}s)"
            )
            return result
        except Exception as e:
            print(f"❌ Error in local search: {e}")
            # Return a fallback response structure
            return type(
                "obj", (object,), {"response": f"Error in local search: {str(e)}"}
            )

    async def global_search(self, query: str):
        """Perform global search."""
        try:
            search_engine = self._get_global_search()
            result = await search_engine.search(query)
            return result
        except Exception as e:
            print(f"❌ Error in global search: {e}")
            # Return a fallback response structure
            return type(
                "obj", (object,), {"response": f"Error in global search: {str(e)}"}
            )

    def get_document_info(self) -> Dict[str, Any]:
        """Get information about the GraphRAG system"""
        return {
            "method": "working_graph_rag",
            "retrieval_strategy": "Working GraphRAG using official GraphRAG library",
            "entities": len(self.entities) if self.entities else 0,
            "relationships": len(self.relationships) if self.relationships else 0,
            "community_reports": len(self.reports) if self.reports else 0,
            "text_units": len(self.text_units) if self.text_units else 0,
            "communities": len(self.communities) if self.communities else 0,
            "data_source": "Working GraphRAG using official GraphRAG library",
            "chat_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
            "search_types": ["local", "global"],
        }

    def get_graph_traversal_info(self, query: str) -> Dict[str, Any]:
        """Get graph traversal information for a query"""
        try:
            # Return format expected by the comparison tool
            return {
                "query": query,
                "total_entities": len(self.entities) if self.entities else 0,
                "total_relationships": (
                    len(self.relationships) if self.relationships else 0
                ),
                "method": "working_graphrag",
            }

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "total_entities": 0,
                "total_relationships": 0,
                "method": "working_graphrag",
            }


async def main():
    """Test local and global search and print the response"""
    graphrag = GraphRAGRetrieval()

    # Test local search
    start_time = time.time()
    local_result = await graphrag.local_search("What is the capital of France?")
    end_time = time.time()
    print("Local search response:", getattr(local_result, "response", local_result))
    print(f"Local search time: {end_time - start_time} seconds")

    # Test global search
    start_time = time.time()
    global_result = await graphrag.global_search("What is the capital of France?")
    end_time = time.time()
    print("Global search response:", getattr(global_result, "response", global_result))
    print(f"Global search time: {end_time - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(main())
