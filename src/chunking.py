"""
Text Chunking Module

Implements multiple chunking strategies for optimal RAG performance.
Semantic chunking is prioritized for scientific text.
"""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Optional

from .config import get_config
from .logging_utils import get_logger
from .models import PaperSource, ScientificPaper, TextChunk


logger = get_logger()


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    def __init__(self):
        self.config = get_config().chunking
    
    @abstractmethod
    def chunk(self, paper: ScientificPaper) -> list[TextChunk]:
        """Split a paper into chunks."""
        pass
    
    def _generate_chunk_id(self, paper_id: str, chunk_index: int, text: str) -> str:
        """Generate a unique, deterministic chunk ID."""
        content = f"{paper_id}:{chunk_index}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class RecursiveChunker(BaseChunker):
    """
    Recursive text chunker with overlap.
    
    This chunker attempts to split on natural boundaries:
    1. Paragraphs (double newlines)
    2. Sentences (periods)
    3. Words (spaces)
    
    Each chunk maintains overlap with the previous chunk to preserve context.
    """
    
    # Separators in order of preference (most to least natural)
    SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "? ",    # Questions
        "! ",    # Exclamations
        "; ",    # Clauses
        ", ",    # Phrases
        " ",     # Words
    ]
    
    def _split_text(self, text: str, separator: str) -> list[str]:
        """Split text by separator, keeping the separator with the preceding chunk."""
        if not separator:
            return list(text)  # Character-level split
        
        parts = text.split(separator)
        # Rejoin with separator (keeps it at end of each chunk except last)
        result = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                result.append(part + separator)
            else:
                result.append(part)
        return [p for p in result if p.strip()]
    
    def _recursive_split(
        self, 
        text: str, 
        separators: list[str],
        chunk_size: int
    ) -> list[str]:
        """Recursively split text until all chunks are under the size limit."""
        
        if not text.strip():
            return []
        
        # Base case: text is already small enough
        if len(text) <= chunk_size:
            return [text.strip()]
        
        # Try to split using the current separator
        if not separators:
            # No more separators - force split at chunk_size
            chunks = []
            for i in range(0, len(text), chunk_size):
                chunks.append(text[i:i + chunk_size].strip())
            return [c for c in chunks if c]
        
        separator = separators[0]
        splits = self._split_text(text, separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # If adding this split would exceed chunk size
            if len(current_chunk) + len(split) > chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # If this single split is too large, recursively split it
                if len(split) > chunk_size:
                    sub_chunks = self._recursive_split(
                        split, 
                        separators[1:], 
                        chunk_size
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
            else:
                current_chunk += split
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _add_overlap(self, chunks: list[str], overlap: int) -> list[str]:
        """Add overlap between chunks for context continuity."""
        if len(chunks) <= 1 or overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]
            
            # Get the last 'overlap' characters from previous chunk
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) >= overlap else prev_chunk
            
            # Prepend to current chunk
            overlapped_chunks.append(overlap_text + current_chunk)
        
        return overlapped_chunks
    
    def chunk(self, paper: ScientificPaper) -> list[TextChunk]:
        """
        Chunk a paper using recursive splitting.
        
        Args:
            paper: ScientificPaper to chunk
        
        Returns:
            List of TextChunk objects
        """
        logger.debug(f"Chunking paper", paper_id=paper.paper_id, strategy="recursive")
        
        # Combine title, abstract, and full text if available
        text_parts = [
            f"Title: {paper.title}",
            f"Abstract: {paper.abstract}"
        ]
        if paper.full_text:
            text_parts.append(paper.full_text)
        
        full_text = "\n\n".join(text_parts)
        
        # Perform recursive splitting
        raw_chunks = self._recursive_split(
            full_text,
            self.SEPARATORS,
            self.config.chunk_size
        )
        
        # Add overlap
        overlapped_chunks = self._add_overlap(raw_chunks, self.config.chunk_overlap)
        
        # Filter out chunks that are too small
        filtered_chunks = [
            c for c in overlapped_chunks 
            if len(c) >= self.config.min_chunk_size
        ]
        
        # Create TextChunk objects
        text_chunks = []
        char_position = 0
        
        for i, chunk_text in enumerate(filtered_chunks):
            chunk = TextChunk(
                chunk_id=self._generate_chunk_id(paper.paper_id, i, chunk_text),
                text=chunk_text,
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source=paper.source,
                chunk_index=i,
                start_char=char_position,
                end_char=char_position + len(chunk_text)
            )
            text_chunks.append(chunk)
            char_position += len(chunk_text) - self.config.chunk_overlap
        
        logger.info(
            f"Chunking complete",
            paper_id=paper.paper_id,
            chunks_created=len(text_chunks)
        )
        
        return text_chunks


class SemanticChunker(BaseChunker):
    """
    Semantic-aware text chunker using embeddings.
    
    This chunker uses sentence embeddings to detect semantic
    boundaries and create more coherent chunks.
    
    Note: Requires sentence-transformers to be installed.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.embedding_model_name = embedding_model
        self._model = None
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model", model=self.embedding_model_name)
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for semantic chunking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex patterns."""
        # Pattern for sentence boundaries
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _compute_similarity(self, emb1, emb2) -> float:
        """Compute cosine similarity between two embeddings."""
        import numpy as np
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
    
    def chunk(self, paper: ScientificPaper) -> list[TextChunk]:
        """
        Chunk a paper using semantic similarity.
        
        The algorithm:
        1. Split into sentences
        2. Compute embeddings for each sentence
        3. Compare consecutive sentences
        4. Create a new chunk when similarity drops below threshold
        
        Args:
            paper: ScientificPaper to chunk
        
        Returns:
            List of TextChunk objects
        """
        logger.debug(f"Chunking paper", paper_id=paper.paper_id, strategy="semantic")
        
        # Combine text
        text_parts = [
            f"Title: {paper.title}",
            f"Abstract: {paper.abstract}"
        ]
        if paper.full_text:
            text_parts.append(paper.full_text)
        
        full_text = "\n\n".join(text_parts)
        
        # Split into sentences
        sentences = self._split_into_sentences(full_text)
        
        if not sentences:
            return []
        
        # Get embeddings for all sentences
        with logger.timed_operation("compute_embeddings", sentence_count=len(sentences)):
            embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Find semantic boundaries
        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_start = 0
        
        for i in range(1, len(sentences)):
            # Compute similarity with previous sentence
            similarity = self._compute_similarity(embeddings[i], embeddings[i - 1])
            
            # Check if we should start a new chunk
            current_chunk_text = " ".join(current_chunk_sentences)
            
            should_split = (
                similarity < self.config.semantic_threshold or
                len(current_chunk_text) >= self.config.chunk_size
            )
            
            if should_split and len(current_chunk_text) >= self.config.min_chunk_size:
                # Save current chunk
                chunks.append(current_chunk_text)
                current_chunk_sentences = [sentences[i]]
                current_chunk_start = i
            else:
                current_chunk_sentences.append(sentences[i])
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            final_chunk = " ".join(current_chunk_sentences)
            if len(final_chunk) >= self.config.min_chunk_size:
                chunks.append(final_chunk)
        
        # Create TextChunk objects
        text_chunks = []
        char_position = 0
        
        for i, chunk_text in enumerate(chunks):
            chunk = TextChunk(
                chunk_id=self._generate_chunk_id(paper.paper_id, i, chunk_text),
                text=chunk_text,
                paper_id=paper.paper_id,
                paper_title=paper.title,
                source=paper.source,
                chunk_index=i,
                start_char=char_position,
                end_char=char_position + len(chunk_text)
            )
            text_chunks.append(chunk)
            char_position += len(chunk_text)
        
        logger.info(
            f"Semantic chunking complete",
            paper_id=paper.paper_id,
            chunks_created=len(text_chunks)
        )
        
        return text_chunks


class HybridChunker(BaseChunker):
    """
    Hybrid chunker combining recursive and semantic approaches.
    
    Strategy:
    1. First, use recursive chunking to create initial chunks
    2. Then, use semantic analysis to merge or split chunks as needed
    
    This provides the best of both worlds: respecting document structure
    while also maintaining semantic coherence.
    """
    
    def __init__(self, use_semantic: bool = True):
        super().__init__()
        self.recursive_chunker = RecursiveChunker()
        self.use_semantic = use_semantic
        
        if use_semantic:
            try:
                self.semantic_chunker = SemanticChunker()
            except ImportError:
                logger.warning(
                    "sentence-transformers not available, "
                    "falling back to recursive chunking only"
                )
                self.use_semantic = False
    
    def chunk(self, paper: ScientificPaper) -> list[TextChunk]:
        """
        Chunk a paper using hybrid approach.
        
        Args:
            paper: ScientificPaper to chunk
        
        Returns:
            List of TextChunk objects
        """
        # For simplicity in this implementation, we use semantic if available,
        # otherwise fall back to recursive
        if self.use_semantic:
            try:
                return self.semantic_chunker.chunk(paper)
            except Exception as e:
                logger.warning(
                    f"Semantic chunking failed, falling back to recursive",
                    error=str(e)
                )
        
        return self.recursive_chunker.chunk(paper)


def get_chunker(strategy: str = "hybrid") -> BaseChunker:
    """
    Factory function to get a chunker by strategy name.
    
    Args:
        strategy: One of "recursive", "semantic", or "hybrid"
    
    Returns:
        Appropriate chunker instance
    """
    strategies = {
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
        "hybrid": HybridChunker,
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                        f"Choose from: {list(strategies.keys())}")
    
    return strategies[strategy]()
