"""
Configuration Management Module

Centralized configuration with environment variable support and validation.
Production-ready with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ScraperConfig:
    """Configuration for web scraping and API calls."""
    
    # Rate limiting (requests per second)
    arxiv_rate_limit: float = 0.5  # ArXiv recommends slow crawling
    pubmed_rate_limit: float = 3.0  # NCBI allows 3/sec without API key, 10/sec with
    
    # API keys
    ncbi_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("NCBI_API_KEY")
    )
    
    # Request settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.5
    
    # User agent for ethical scraping
    user_agent: str = "ScientificRAG/1.0 (Research Project; contact@example.com)"


@dataclass
class ChunkingConfig:
    """Configuration for text chunking strategies."""
    
    # Chunk size in characters (optimized for scientific text)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Semantic chunking threshold (cosine similarity)
    semantic_threshold: float = 0.75
    
    # Minimum chunk size to avoid tiny fragments
    min_chunk_size: int = 100


@dataclass
class VectorDBConfig:
    """Configuration for vector database."""
    
    # Persistence directory
    persist_directory: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    )
    
    # Collection name
    collection_name: str = "scientific_papers"
    
    # Embedding model
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Search parameters
    default_top_k: int = 5


@dataclass
class LLMConfig:
    """Configuration for Large Language Model."""
    
    # API key
    api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    
    # Model selection
    model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    
    # Generation parameters
    temperature: float = 0.1  # Low for factual accuracy
    max_tokens: int = 2000
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    
    level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Log file settings
    log_dir: str = "./logs"
    log_file: str = "rag_system.log"
    
    # Structured logging
    json_format: bool = True


@dataclass
class AppConfig:
    """Main application configuration aggregating all sub-configs."""
    
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []
        
        if not self.llm.api_key:
            issues.append("ERROR: OPENAI_API_KEY not set. LLM operations will fail.")
        
        if not self.scraper.ncbi_api_key:
            issues.append(
                "WARNING: NCBI_API_KEY not set. PubMed rate limit will be 3 req/sec."
            )
        
        # Ensure directories exist
        Path(self.vector_db.persist_directory).mkdir(parents=True, exist_ok=True)
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        
        return issues


# Global configuration instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config
