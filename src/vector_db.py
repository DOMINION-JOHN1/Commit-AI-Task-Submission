"""
Vector Database Module

ChromaDB-based vector storage with embedding generation.
Supports similarity search and metadata filtering.

ENHANCEMENT: Hybrid search combining BM25 + Vector similarity
Research shows 15-25% accuracy improvement over vector-only search.
"""

from pathlib import Path
from typing import Optional
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

from .config import get_config
from .logging_utils import get_logger
from .models import PaperSource, QueryResult, TextChunk


logger = get_logger()


class VectorDatabase:
    """
    ChromaDB-based vector database for storing and querying text chunks.
    
    Features:
    - Automatic embedding generation using SentenceTransformers
    - Persistent storage to disk
    - Metadata filtering support
    - Similarity search with configurable top-k
    """
    
    # Cross-Encoder model for Stage 2 reranking
    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection (uses config default if not provided)
        """
        self.config = get_config().vector_db
        self.collection_name = collection_name or self.config.collection_name
        
        self._client = None
        self._collection = None
        self._embedding_function = None
        self._reranker = None
    
    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                # Create persistent client
                persist_dir = Path(self.config.persist_directory)
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"Initializing ChromaDB", persist_dir=str(persist_dir))
                
                self._client = chromadb.PersistentClient(
                    path=str(persist_dir),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required. Install with: pip install chromadb"
                )
        return self._client
    
    @property
    def embedding_function(self):
        """Lazy-load embedding function."""
        if self._embedding_function is None:
            try:
                from chromadb.utils import embedding_functions
                
                logger.info(
                    f"Loading embedding function",
                    model=self.config.embedding_model
                )
                
                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.config.embedding_model
                )
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedding_function
    
    @property
    def collection(self):
        """Get or create the collection."""
        if self._collection is None:
            logger.info(f"Getting collection", name=self.collection_name)
            
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
        return self._collection
    
    def add_chunks(self, chunks: list[TextChunk]) -> int:
        """
        Add text chunks to the vector database.
        
        Args:
            chunks: List of TextChunk objects to add
        
        Returns:
            Number of chunks added
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0
        
        with logger.timed_operation("add_chunks", chunk_count=len(chunks)):
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "paper_id": chunk.paper_id,
                    "paper_title": chunk.paper_title,
                    "source": chunk.source.value,
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section or "unknown"
                }
                for chunk in chunks
            ]
            
            # Upsert to handle duplicates gracefully
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added chunks to vector database", count=len(chunks))
            return len(chunks)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_filter: Optional[PaperSource] = None,
        paper_id_filter: Optional[str] = None
    ) -> list[QueryResult]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query text
            top_k: Number of results to return (uses config default if not provided)
            source_filter: Optional filter by paper source
            paper_id_filter: Optional filter by specific paper ID
        
        Returns:
            List of QueryResult objects sorted by relevance
        """
        top_k = top_k or self.config.default_top_k
        
        # Build where clause for filtering
        where = None
        where_conditions = []
        
        if source_filter:
            where_conditions.append({"source": source_filter.value})
        
        if paper_id_filter:
            where_conditions.append({"paper_id": paper_id_filter})
        
        if len(where_conditions) == 1:
            where = where_conditions[0]
        elif len(where_conditions) > 1:
            where = {"$and": where_conditions}
        
        with logger.timed_operation("vector_search", query_length=len(query), top_k=top_k):
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        
        # Convert to QueryResult objects
        query_results = []
        
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                chunk = TextChunk(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    paper_id=metadata["paper_id"],
                    paper_title=metadata["paper_title"],
                    source=PaperSource(metadata["source"]),
                    chunk_index=metadata["chunk_index"],
                    section=metadata.get("section")
                )
                
                # ChromaDB returns distances, convert to similarity score
                # For cosine distance: similarity = 1 - distance
                distance = results["distances"][0][i]
                similarity = 1 - distance
                
                query_results.append(QueryResult(
                    chunk=chunk,
                    score=similarity,
                    distance_metric="cosine"
                ))
        
        logger.info(
            f"Search complete",
            query_preview=query[:50],
            results_found=len(query_results)
        )
        
        return query_results
    
    def hybrid_search(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: float = 0.5,
        source_filter: Optional[PaperSource] = None,
        paper_id_filter: Optional[str] = None
    ) -> list[QueryResult]:
        """
        Hybrid search combining BM25 keyword search with vector similarity.
        
        Research shows this improves precision by 15-25% over vector-only search.
        Source: Pinecone hybrid search benchmarks
        
        Args:
            query: Search query text
            top_k: Number of results to return
            alpha: Weight for vector scores (0-1). (1-alpha) for BM25
                   0.5 = equal weight, 0.7 = favor vector, 0.3 = favor keyword
            source_filter: Optional filter by paper source
            paper_id_filter: Optional filter by specific paper ID
        
        Returns:
            List of QueryResult objects sorted by hybrid score
        """
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, falling back to vector search")
            return self.search(query, top_k, source_filter, paper_id_filter)
        
        top_k = top_k or self.config.default_top_k
        
        with logger.timed_operation("hybrid_search", query_length=len(query), top_k=top_k):
            # Get all documents for BM25 (with filters)
            where = self._build_where_clause(source_filter, paper_id_filter)
            
            all_results = self.collection.get(
                where=where,
                include=["documents", "metadatas"]
            )
            
            if not all_results or not all_results["ids"]:
                return []
            
            # Build BM25 index
            documents = all_results["documents"]
            tokenized_docs = [doc.lower().split() for doc in documents]
            bm25 = BM25Okapi(tokenized_docs)
            
            # Get BM25 scores
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores to 0-1
            if bm25_scores.max() > 0:
                bm25_scores = bm25_scores / bm25_scores.max()
            
            # Get vector scores
            vector_results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k * 3, len(documents)),  # Get more for reranking
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Build vector score mapping
            vector_scores = {}
            for i, doc_id in enumerate(vector_results["ids"][0]):
                distance = vector_results["distances"][0][i]
                similarity = 1 - distance  # Convert distance to similarity
                vector_scores[doc_id] = similarity
            
            # Combine scores
            hybrid_scores = []
            for i, doc_id in enumerate(all_results["ids"]):
                # BM25 score (normalized)
                bm25_score = bm25_scores[i]
                
                # Vector score (0 if not in top vector results)
                vector_score = vector_scores.get(doc_id, 0.0)
                
                # Hybrid score: weighted combination
                hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score
                
                hybrid_scores.append((doc_id, hybrid_score, i))
            
            # Sort by hybrid score
            hybrid_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Build QueryResult objects for top-k
            query_results = []
            for doc_id, score, idx in hybrid_scores[:top_k]:
                metadata = all_results["metadatas"][idx]
                
                chunk = TextChunk(
                    chunk_id=doc_id,
                    text=all_results["documents"][idx],
                    paper_id=metadata["paper_id"],
                    paper_title=metadata["paper_title"],
                    source=PaperSource(metadata["source"]),
                    chunk_index=metadata["chunk_index"],
                    section=metadata.get("section")
                )
                
                query_results.append(QueryResult(
                    chunk=chunk,
                    score=score,
                    distance_metric="hybrid_bm25_vector"
                ))
            
            logger.info(
                f"Hybrid search complete",
                query_preview=query[:50],
                results=len(query_results),
                alpha=alpha
            )
            
            return query_results
    
    @property
    def reranker(self) -> Optional["CrossEncoder"]:
        """Lazy-load the Cross-Encoder reranker model."""
        if self._reranker is None and RERANKER_AVAILABLE:
            logger.info(f"Loading Cross-Encoder reranker", model=self.RERANKER_MODEL)
            self._reranker = CrossEncoder(self.RERANKER_MODEL)
        return self._reranker

    def search_with_rerank(
        self,
        query: str,
        top_k: Optional[int] = None,
        retrieve_k: int = 25,
        alpha: float = 0.5,
        source_filter: Optional[PaperSource] = None,
        paper_id_filter: Optional[str] = None
    ) -> list[QueryResult]:
        """
        Two-stage retrieval: Hybrid Search followed by Cross-Encoder Reranking.

        Stage 1 (Hybrid Search): Fast BM25 + Vector retrieval of 'retrieve_k' candidates.
        Stage 2 (Reranking): Cross-Encoder scores each (query, chunk) pair together
                             for higher precision, then returns top 'top_k'.

        Research shows cross-encoder reranking eliminates false positives that
        look similar mathematically but don't actually answer the question.

        Args:
            query: Search query text
            top_k: Number of final results to return after reranking
            retrieve_k: Number of candidates to retrieve in Stage 1 (should be >> top_k)
            alpha: BM25 vs vector weight for Stage 1 hybrid search
            source_filter: Optional filter by paper source
            paper_id_filter: Optional filter by specific paper ID

        Returns:
            List of QueryResult objects sorted by rerank score (highest first)
        """
        top_k = top_k or self.config.default_top_k

        if not RERANKER_AVAILABLE:
            logger.warning(
                "Cross-Encoder not available, falling back to hybrid search",
                hint="pip install sentence-transformers"
            )
            return self.hybrid_search(query, top_k, alpha, source_filter, paper_id_filter)

        with logger.timed_operation("search_with_rerank", query_length=len(query), top_k=top_k, retrieve_k=retrieve_k):
            # Stage 1: Broad candidate retrieval
            candidates = self.hybrid_search(
                query,
                top_k=retrieve_k,
                alpha=alpha,
                source_filter=source_filter,
                paper_id_filter=paper_id_filter
            )

            if not candidates:
                return []

            logger.info(
                f"Reranking candidates",
                candidates=len(candidates),
                query_preview=query[:50]
            )

            # Stage 2: Cross-Encoder scoring — processes (query, doc) pairs TOGETHER
            pairs = [[query, result.chunk.text] for result in candidates]
            rerank_scores = self.reranker.predict(pairs)

            # Attach rerank scores and sort descending
            reranked = [
                QueryResult(
                    chunk=result.chunk,
                    score=float(score),
                    distance_metric="cross_encoder_rerank"
                )
                for result, score in zip(candidates, rerank_scores)
            ]
            reranked.sort(key=lambda x: x.score, reverse=True)

            top_results = reranked[:top_k]

            logger.info(
                f"Reranking complete",
                final_results=len(top_results),
                top_score=round(top_results[0].score, 4) if top_results else None
            )

            return top_results

    def _build_where_clause(
        self,
        source_filter: Optional[PaperSource],
        paper_id_filter: Optional[str]
    ) -> Optional[dict]:
        """Build ChromaDB where clause from filters."""
        where_conditions = []
        
        if source_filter:
            where_conditions.append({"source": source_filter.value})
        
        if paper_id_filter:
            where_conditions.append({"paper_id": paper_id_filter})
        
        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}
        
        return None
    
    def get_stats(self) -> dict:
        """Get statistics about the vector database."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "persist_directory": self.config.persist_directory,
            "embedding_model": self.config.embedding_model
        }
    
    def delete_by_paper_id(self, paper_id: str) -> int:
        """
        Delete all chunks from a specific paper.
        
        Args:
            paper_id: ID of the paper whose chunks should be deleted
        
        Returns:
            Number of chunks deleted
        """
        # Get all chunk IDs for this paper
        results = self.collection.get(
            where={"paper_id": paper_id},
            include=[]
        )
        
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted chunks", paper_id=paper_id, count=len(results["ids"]))
            return len(results["ids"])
        
        return 0
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        logger.warning("Clearing entire collection", name=self.collection_name)
        
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self._collection = None
        
        # Recreate
        _ = self.collection
        
        logger.info("Collection cleared and recreated")


# Module-level instance for convenience
_db_instance: Optional[VectorDatabase] = None


def get_vector_db(collection_name: Optional[str] = None) -> VectorDatabase:
    """Get or create the vector database instance."""
    global _db_instance
    
    if _db_instance is None or (collection_name and collection_name != _db_instance.collection_name):
        _db_instance = VectorDatabase(collection_name)
    
    return _db_instance
