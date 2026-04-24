"""
Task 1: Data Acquisition, Chunking, and Vector Database Setup
Implementation including PDF extraction, hybrid search, and cross-encoder reranking.
"""

import os
import hashlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import requests
import chromadb
from chromadb.utils import embedding_functions

# Optional dependency imports
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False


@dataclass
class Config:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    arxiv_rate_limit: float = 0.33
    collection_name: str = "scientific_papers_v4"
    chroma_path: str = "./chroma_db"

config = Config()


@dataclass
class Paper:
    paper_id: str
    source: str
    title: str
    abstract: str
    authors: List[str]
    url: str = ""
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    paper_id: str
    paper_title: str
    source: str
    chunk_index: int


class PDFExtractor:
    """Handles PDF text extraction using PyMuPDF."""
    @staticmethod
    def extract_from_url(pdf_url: str) -> Optional[str]:
        if not PDF_AVAILABLE:
            return None
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
            # Clean up whitespace and null bytes
            return re.sub(r'\n{3,}', '\n\n', text.replace("\x00", ""))
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return None


class ArXivScraper:
    """Scraper for the ArXiv Atom API."""
    def search(self, query: str, max_results: int = 5, extract_pdfs: bool = False) -> List[Paper]:
        params = {
            "search_query": f"all:{query}",
            "max_results": max_results,
            "sortBy": "relevance"
        }
        res = requests.get("http://export.arxiv.org/api/query", params=params)
        root = ET.fromstring(res.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        
        papers = []
        for entry in root.findall("atom:entry", ns):
            p_id = entry.find("atom:id", ns).text.split("/")[-1]
            pdf = next((l.get("href") for l in entry.findall("atom:link", ns) 
                       if l.get("title") == "pdf"), None)
            
            full_text = None
            if extract_pdfs and pdf:
                full_text = PDFExtractor.extract_from_url(pdf)
                
            papers.append(Paper(
                paper_id=p_id,
                source="arxiv",
                title=entry.find("atom:title", ns).text.strip(),
                abstract=entry.find("atom:summary", ns).text.strip(),
                authors=[],
                url=f"https://arxiv.org/abs/{p_id}",
                pdf_url=pdf,
                full_text=full_text
            ))
        return papers


class PubMedScraper:
    """Scraper for PubMed E-utilities."""
    def search(self, query: str, max_results: int = 5) -> List[Paper]:
        # Search for IDs
        search_res = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "xml"}
        )
        ids = [i.text for i in ET.fromstring(search_res.text).findall(".//Id")]
        if not ids:
            return []
            
        # Fetch paper details
        fetch_res = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        )
        
        papers = []
        for article in ET.fromstring(fetch_res.text).findall(".//PubmedArticle"):
            pmid = article.find(".//PMID").text
            abstract_node = article.find(".//AbstractText")
            abstract_text = abstract_node.text if abstract_node is not None else ""
            
            papers.append(Paper(
                paper_id=pmid,
                source="pubmed",
                title=article.find(".//ArticleTitle").text,
                abstract=abstract_text,
                authors=[],
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            ))
        return papers


class VectorDB:
    """ChromaDB implementation with hybrid retrieval and cross-encoder reranking."""
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.chroma_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name, 
            embedding_function=self.embedding_fn
        )
        
        if RERANKER_AVAILABLE:
            print(f"Loading reranker model: {config.reranker_model}")
            self.reranker = CrossEncoder(config.reranker_model)

    def add_chunks(self, chunks: List[TextChunk]):
        if not chunks:
            return
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{
                "paper_id": c.paper_id,
                "paper_title": c.paper_title,
                "source": c.source
            } for c in chunks]
        )

    def hybrid_search(self, query: str, top_k: int = 25, alpha: float = 0.5) -> List[Dict]:
        """Hybrid search combining vector and BM25 scores."""
        all_res = self.collection.get(include=["documents", "metadatas"])
        docs = all_res["documents"]
        if not docs:
            return []
            
        # BM25 Sparse Search
        bm25 = BM25Okapi([d.lower().split() for d in docs])
        bm25_scores = bm25.get_scores(query.lower().split())
        if bm25_scores.max() > 0:
            bm25_scores /= bm25_scores.max()
            
        # Vector Dense Search
        vec_res = self.collection.query(query_texts=[query], n_results=len(docs))
        vec_map = {d_id: 1 - dist for d_id, dist in zip(vec_res["ids"][0], vec_res["distances"][0])}
        
        scored_results = []
        for i, doc_id in enumerate(all_res["ids"]):
            score = (alpha * vec_map.get(doc_id, 0.0)) + ((1 - alpha) * bm25_scores[i])
            scored_results.append({
                "id": doc_id,
                "text": docs[i],
                "metadata": all_res["metadatas"][i],
                "score": score
            })
        
        scored_results.sort(key=lambda x: x["score"], reverse=True)
        return scored_results[:top_k]

    def search_with_rerank(self, query: str, top_k: int = 5, retrieve_k: int = 25) -> List[Dict]:
        """Stage 2 retrieval using a cross-encoder to refine results."""
        candidates = self.hybrid_search(query, top_k=retrieve_k)
        if not candidates or not RERANKER_AVAILABLE:
            return candidates[:top_k]

        pairs = [[query, cand["text"]] for cand in candidates]
        rerank_scores = self.reranker.predict(pairs)
        
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(rerank_scores[i])
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]


def run_ingestion_pipeline():
    print("Initializing ingestion pipeline...")
    arxiv = ArXivScraper()
    pubmed = PubMedScraper()
    db = VectorDB()
    
    search_query = "machine learning for medical imaging"
    print(f"Searching APIs for query: {search_query}")
    
    papers = (arxiv.search(search_query, max_results=2, extract_pdfs=True) + 
              pubmed.search(search_query, max_results=2))
    
    chunks_to_index = []
    for p in papers:
        # Prioritize full text if available
        content = p.full_text if p.full_text else p.abstract
        # Simplified sliding window chunking
        p_chunks = [content[i:i+1000] for i in range(0, len(content), 800)]
        
        for i, text in enumerate(p_chunks):
            chunk_id = hashlib.md5(f"{p.paper_id}_{i}".encode()).hexdigest()[:12]
            chunks_to_index.append(TextChunk(
                chunk_id=chunk_id,
                text=text,
                paper_id=p.paper_id,
                paper_title=p.title,
                source=p.source,
                chunk_index=i
            ))
    
    print(f"Indexing {len(chunks_to_index)} chunks...")
    db.add_chunks(chunks_to_index)
    
    # Test retrieval
    verification_query = "deep learning in radiology"
    results = db.search_with_rerank(verification_query, top_k=3)
    
    print(f"\nVerification Results for: '{verification_query}'")
    for i, res in enumerate(results, 1):
        score = res.get('rerank_score', res.get('score'))
        print(f"[{i}] Score: {score:.3f} | Source: {res['metadata']['source']}")
        print(f"    Text: {res['text'][:120]}...\n")

    return db


if __name__ == "__main__":
    db = run_ingestion_pipeline()
