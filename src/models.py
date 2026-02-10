"""
Scientific Paper Data Models

Pydantic models for type-safe data handling and validation.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, HttpUrl


class PaperSource(str, Enum):
    """Source of the scientific paper."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"


class Author(BaseModel):
    """Author information."""
    name: str
    affiliation: Optional[str] = None


class ScientificPaper(BaseModel):
    """
    Represents a scientific paper with metadata.
    
    This is the primary data structure for papers fetched from ArXiv/PubMed.
    """
    
    # Unique identifiers
    paper_id: str = Field(..., description="Unique identifier (ArXiv ID or PMID)")
    source: PaperSource = Field(..., description="Paper source (arxiv/pubmed)")
    
    # Core metadata
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    authors: list[Author] = Field(default_factory=list, description="List of authors")
    
    # Publication info
    published_date: Optional[datetime] = Field(None, description="Publication date")
    journal: Optional[str] = Field(None, description="Journal name (PubMed)")
    categories: list[str] = Field(default_factory=list, description="Subject categories")
    
    # Links
    url: Optional[HttpUrl] = Field(None, description="URL to the paper")
    pdf_url: Optional[HttpUrl] = Field(None, description="Direct PDF link")
    doi: Optional[str] = Field(None, description="Digital Object Identifier")
    
    # Full text (if available)
    full_text: Optional[str] = Field(None, description="Full paper text if available")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


class TextChunk(BaseModel):
    """
    A chunk of text with metadata for vector storage.
    """
    
    chunk_id: str = Field(..., description="Unique chunk identifier")
    text: str = Field(..., description="The text content")
    
    # Source tracking
    paper_id: str = Field(..., description="Parent paper ID")
    paper_title: str = Field(..., description="Parent paper title")
    source: PaperSource = Field(..., description="Original source")
    
    # Position metadata
    chunk_index: int = Field(..., description="Position in the document")
    start_char: int = Field(0, description="Starting character position")
    end_char: int = Field(0, description="Ending character position")
    
    # Semantic metadata
    section: Optional[str] = Field(None, description="Document section (intro, methods, etc.)")


class QueryResult(BaseModel):
    """
    Result from a vector database query.
    """
    
    chunk: TextChunk = Field(..., description="The matched text chunk")
    score: float = Field(..., description="Similarity score (0-1)")
    
    # Distance metric info
    distance_metric: str = Field("cosine", description="Distance metric used")


class SubQuery(BaseModel):
    """
    A decomposed sub-query from a complex question.
    """
    
    query_text: str = Field(..., description="The sub-query text")
    reasoning: str = Field(..., description="Why this sub-query is needed")
    order: int = Field(..., description="Execution order")
    depends_on: list[int] = Field(default_factory=list, description="Dependencies on other sub-queries")


class RAGResponse(BaseModel):
    """
    Complete response from the RAG system.
    """
    
    # User query
    original_query: str = Field(..., description="Original user question")
    
    # Decomposition (if applicable)
    sub_queries: list[SubQuery] = Field(default_factory=list, description="Decomposed sub-queries")
    
    # Retrieved context
    retrieved_chunks: list[QueryResult] = Field(default_factory=list, description="Retrieved text chunks")
    
    # Generated answer
    answer: str = Field(..., description="Generated answer")
    
    # Metadata
    sources_used: list[str] = Field(default_factory=list, description="Paper IDs used in answer")
    confidence: float = Field(0.0, description="Confidence score (0-1)")
    processing_time_ms: float = Field(0.0, description="Total processing time")
