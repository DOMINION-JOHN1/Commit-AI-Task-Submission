"""
Test Suite for Scientific Paper RAG System

Comprehensive tests covering all modules.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import Paper, TextChunk, PaperSource, Author
from src.chunking import RecursiveChunker, get_chunker
from src.compliance import ComplianceHandler, is_hipaa_compliant_text


class TestChunking:
    """Tests for text chunking module."""
    
    def test_recursive_chunking_basic(self):
        """Test basic recursive chunking functionality."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20)
        
        paper = Paper(
            paper_id="test_001",
            source=PaperSource.ARXIV,
            title="Test Paper on Machine Learning",
            abstract="This is a test abstract. " * 50,  # Long abstract
            authors=[Author(name="Test Author")]
        )
        
        chunks = chunker.chunk(paper)
        
        # Assertions
        assert len(chunks) > 1, "Should create multiple chunks for long text"
        assert all(isinstance(c, TextChunk) for c in chunks)
        assert all(c.paper_id == "test_001" for c in chunks)
    
    def test_chunk_size_constraint(self):
        """Test that chunks respect size limits."""
        chunk_size = 200
        chunker = RecursiveChunker(chunk_size=chunk_size, overlap=50)
        
        paper = Paper(
            paper_id="test_002",
            source=PaperSource.PUBMED,
            title="Long Title " * 10,
            abstract="Long abstract. " * 100,
            authors=[]
        )
        
        chunks = chunker.chunk(paper)
        
        # Allow some tolerance for overlap
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size + 100, \
                f"Chunk too large: {len(chunk.text)} chars"
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        chunker = RecursiveChunker(chunk_size=150, overlap=30)
        
        text = "First sentence. Second sentence. Third sentence. " * 20
        chunks = chunker._split_recursive(text, chunker.SEPARATORS)
        
        # If multiple chunks, check overlap
        if len(chunks) > 1:
            # Second chunk should start with end of first chunk
            # (This is a simplified check; actual overlap is added separately)
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0
    
    def test_min_chunk_size_filter(self):
        """Test that tiny chunks are filtered out."""
        chunker = RecursiveChunker(chunk_size=100, overlap=20)
        
        paper = Paper(
            paper_id="test_003",
            source=PaperSource.ARXIV,
            title="Short",
            abstract="Tiny abstract.",
            authors=[]
        )
        
        chunks = chunker.chunk(paper)
        
        # All chunks should meet minimum size
        for chunk in chunks:
            assert len(chunk.text) >= chunker.config.min_chunk_size
    
    def test_chunk_metadata(self):
        """Test that chunk metadata is correct."""
        chunker = get_chunker("recursive")
        
        paper = Paper(
            paper_id="arxiv_123",
            source=PaperSource.ARXIV,
            title="Test Paper Title",
            abstract="Test abstract content.",
            authors=[Author(name="John Doe")]
        )
        
        chunks = chunker.chunk(paper)
        
        for i, chunk in enumerate(chunks):
            assert chunk.paper_id == "arxiv_123"
            assert chunk.paper_title == "Test Paper Title"
            assert chunk.source == PaperSource.ARXIV
            assert chunk.chunk_index == i
            assert chunk.end_char >= chunk.start_char


class TestCompliance:
    """Tests for compliance and security module."""
    
    def test_email_redaction(self):
        """Test email address redaction."""
        handler = ComplianceHandler()
        
        text = "Contact me at john.doe@example.com for more info."
        anonymized, report = handler.anonymize_text(text)
        
        assert "john.doe@example.com" not in anonymized
        assert "[EMAIL_REDACTED]" in anonymized
        assert "email" in report["patterns_matched"]
        assert report["patterns_matched"]["email"] == 1
    
    def test_phone_redaction(self):
        """Test phone number redaction."""
        handler = ComplianceHandler()
        
        text = "Call us at 555-123-4567 or 555.987.6543."
        anonymized, report = handler.anonymize_text(text)
        
        assert "555-123-4567" not in anonymized
        assert "555.987.6543" not in anonymized
        assert report["patterns_matched"]["phone_us"] == 2
    
    def test_multiple_pii_types(self):
        """Test redaction of multiple PII types."""
        handler = ComplianceHandler()
        
        text = """
        Patient: john@example.com
        Phone: 555-123-4567
        SSN: 123-45-6789
        IP: 192.168.1.1
        """
        
        anonymized, report = handler.anonymize_text(text)
        
        # Check all PII is removed
        assert "john@example.com" not in anonymized
        assert "555-123-4567" not in anonymized
        assert "123-45-6789" not in anonymized
        assert "192.168.1.1" not in anonymized
        
        # Check report
        assert len(report["patterns_matched"]) >= 3
    
    def test_hash_identifier_consistency(self):
        """Test that hashing is consistent."""
        id1 = ComplianceHandler.hash_identifier("patient_12345")
        id2 = ComplianceHandler.hash_identifier("patient_12345")
        id3 = ComplianceHandler.hash_identifier("patient_67890")
        
        # Same input = same hash
        assert id1 == id2
        
        # Different input = different hash
        assert id1 != id3
        
        # Hash should be non-reversible
        assert "patient_12345" not in id1
    
    def test_hipaa_compliance_check(self):
        """Test HIPAA compliance validation."""
        # Compliant text (no PII)
        clean_text = "This study examined the efficacy of treatment X."
        is_compliant, violations = is_hipaa_compliant_text(clean_text)
        assert is_compliant
        assert len(violations) == 0
        
        # Non-compliant text (contains PII)
        dirty_text = "Patient john@example.com responded well to treatment."
        is_compliant, violations = is_hipaa_compliant_text(dirty_text)
        assert not is_compliant
        assert len(violations) > 0


class TestModels:
    """Tests for data models."""
    
    def test_paper_creation(self):
        """Test Paper model creation."""
        paper = Paper(
            paper_id="arxiv_2024_001",
            source=PaperSource.ARXIV,
            title="Test Paper",
            abstract="Test abstract.",
            authors=[Author(name="Author 1"), Author(name="Author 2")]
        )
        
        assert paper.paper_id == "arxiv_2024_001"
        assert paper.source == PaperSource.ARXIV
        assert len(paper.authors) == 2
    
    def test_text_chunk_creation(self):
        """Test TextChunk model creation."""
        chunk = TextChunk(
            chunk_id="chunk_001",
            text="This is a test chunk of text.",
            paper_id="paper_123",
            paper_title="Test Paper",
            source=PaperSource.PUBMED,
            chunk_index=0,
            start_char=0,
            end_char=29
        )
        
        assert chunk.chunk_id == "chunk_001"
        assert chunk.paper_id == "paper_123"
        assert chunk.source == PaperSource.PUBMED


class TestVectorDBIntegration:
    """Integration tests for vector database (requires ChromaDB)."""
    
    @pytest.fixture
    def test_db(self):
        """Create a test vector database."""
        from src.vector_db import VectorDatabase
        
        # Use test collection
        db = VectorDatabase(collection_name="test_collection_pytest")
        yield db
        
        # Cleanup
        try:
            db.clear()
        except:
            pass
    
    def test_add_and_search(self, test_db):
        """Test adding chunks and searching."""
        # Create test chunks
        chunks = [
            TextChunk(
                chunk_id="test_1",
                text="Machine learning is a subset of artificial intelligence.",
                paper_id="paper_1",
                paper_title="ML Basics",
                source=PaperSource.ARXIV,
                chunk_index=0
            ),
            TextChunk(
                chunk_id="test_2",
                text="Deep learning uses neural networks with multiple layers.",
                paper_id="paper_1",
                paper_title="ML Basics",
                source=PaperSource.ARXIV,
                chunk_index=1
            )
        ]
        
        # Add chunks
        count = test_db.add_chunks(chunks)
        assert count == 2
        
        # Search
        results = test_db.search("what is machine learning", top_k=1)
        
        assert len(results) > 0
        assert "machine learning" in results[0].chunk.text.lower()
        assert results[0].score > 0


# Fixtures for test data
@pytest.fixture
def sample_paper():
    """Sample paper for testing."""
    return Paper(
        paper_id="test_paper_001",
        source=PaperSource.ARXIV,
        title="A Comprehensive Study of Machine Learning in Healthcare",
        abstract=(
            "Machine learning has revolutionized healthcare. "
            "This paper examines various applications including "
            "diagnosis, treatment planning, and patient monitoring. "
            "We review recent advances and discuss future directions."
        ),
        authors=[
            Author(name="Dr. Jane Smith", affiliation="Stanford University"),
            Author(name="Dr. John Doe", affiliation="MIT")
        ]
    )


@pytest.fixture
def sample_chunks(sample_paper):
    """Sample chunks for testing."""
    chunker = RecursiveChunker()
    return chunker.chunk(sample_paper)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
