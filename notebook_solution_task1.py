"""
TASK 1: Mining Web Pages, Text Extraction, Chunking & Vector Database
======================================================================
Enhanced with PDF full-text extraction, Hybrid Search, and CROSS-ENCODER RERANKING.

Copy this code into your Jupyter notebook cell for Task 1.
"""

# ============================================================================
# INSTALLATION (Run this once)
# ============================================================================
# !pip install arxiv biopython chromadb sentence-transformers PyMuPDF rank-bm25 tenacity requests langchain-openai langchain-community langchain-core pydantic

import os
import hashlib
import time
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
from xml.etree import ElementTree as ET
import requests

# Vector DB & Embeddings
import chromadb
from chromadb.utils import embedding_functions

# Enhancement: Cross-Encoder Reranking
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False

# Enhancement: PDF Extraction
try:
    import fitz  # PyMuPDF
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Enhancement: Hybrid Search
try:
    from rank_bm25 import BM25Okapi
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
@dataclass
class Config:
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    # Reranker model (highly accurate for MS-MARCO datasets)
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ARXIV_RATE_LIMIT: float = 0.33 
    COLLECTION_NAME: str = "scientific_papers_v3"
    CHROMA_PATH: str = "./chroma_db"

config = Config()

# ============================================================================
# DATA MODELS
# ============================================================================
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

# ============================================================================
# PDF EXTRACTOR
# ============================================================================
class PDFExtractor:
    @staticmethod
    def extract_from_url(pdf_url: str) -> Optional[str]:
        if not PDF_AVAILABLE: return None
        try:
            response = requests.get(pdf_url, timeout=30); response.raise_for_status()
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = "\n".join([page.get_text("text") for page in doc])
            doc.close()
            return re.sub(r'\n{3,}', '\n\n', text.replace("\x00", ""))
        except: return None

# ============================================================================
# SCRAPERS
# ============================================================================
class ArXivScraper:
    def search(self, query: str, max_results: int = 5, extract_pdfs: bool = False) -> List[Paper]:
        params = {"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"}
        res = requests.get("http://export.arxiv.org/api/query", params=params)
        root = ET.fromstring(res.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        for entry in root.findall("atom:entry", ns):
            p_id = entry.find("atom:id", ns).text.split("/")[-1]
            pdf = next((l.get("href") for l in entry.findall("atom:link", ns) if l.get("title")=="pdf"), None)
            full = PDFExtractor.extract_from_url(pdf) if extract_pdfs and pdf else None
            papers.append(Paper(p_id, "arxiv", entry.find("atom:title", ns).text.strip(), entry.find("atom:summary", ns).text.strip(), [], f"https://arxiv.org/abs/{p_id}", pdf, full))
        return papers

class PubMedScraper:
    def search(self, query: str, max_results: int = 5) -> List[Paper]:
        res = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={"db":"pubmed", "term":query, "retmax":max_results, "retmode":"xml"})
        ids = [i.text for i in ET.fromstring(res.text).findall(".//Id")]
        if not ids: return []
        res = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params={"db":"pubmed", "id":",".join(ids), "retmode":"xml"})
        papers = []
        for a in ET.fromstring(res.text).findall(".//PubmedArticle"):
            pmid = a.find(".//PMID").text
            papers.append(Paper(pmid, "pubmed", a.find(".//ArticleTitle").text, a.find(".//AbstractText").text if a.find(".//AbstractText") is not None else "", [], f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
        return papers

# ============================================================================
# VECTOR DATABASE WITH RERANKING
# ============================================================================
class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL)
        self.collection = self.client.get_or_create_collection(name=config.COLLECTION_NAME, embedding_function=self.embedding_fn)
        # Load the Reranker model
        if RERANKER_AVAILABLE:
            print(f"📦 Loading Reranker: {config.RERANKER_MODEL}...")
            self.reranker = CrossEncoder(config.RERANKER_MODEL)

    def add_chunks(self, chunks: List[TextChunk]):
        if not chunks: return
        self.collection.upsert(ids=[c.chunk_id for c in chunks], documents=[c.text for c in chunks], metadatas=[{"paper_id": c.paper_id, "paper_title": c.paper_title, "source": c.source} for c in chunks])

    def hybrid_search(self, query: str, top_k: int = 25, alpha: float = 0.5) -> List[Dict]:
        """Stage 1 Retrieval: Fast filtering of candidates."""
        all_res = self.collection.get(include=["documents", "metadatas"])
        docs = all_res["documents"]
        if not docs: return []
        
        # BM25
        bm25 = BM25Okapi([d.lower().split() for d in docs])
        bm25_scores = bm25.get_scores(query.lower().split())
        if bm25_scores.max() > 0: bm25_scores /= bm25_scores.max()
        
        # Vector
        vec_res = self.collection.query(query_texts=[query], n_results=len(docs))
        vec_map = {d_id: 1 - dist for d_id, dist in zip(vec_res["ids"][0], vec_res["distances"][0])}
        
        scored = []
        for i, doc_id in enumerate(all_res["ids"]):
            score = alpha * vec_map.get(doc_id, 0.0) + (1 - alpha) * bm25_scores[i]
            scored.append({"id": doc_id, "text": docs[i], "metadata": all_res["metadatas"][i], "score": score})
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def search_with_rerank(self, query: str, top_k: int = 5, retrieve_k: int = 25) -> List[Dict]:
        """
        Stage 2: Cross-Encoder Reranking
        1. Retrieve 'retrieve_k' candidates using Hybrid Search
        2. Rerank them using the Cross-Encoder for higher precision
        """
        # 1. Broad retrieval
        candidates = self.hybrid_search(query, top_k=retrieve_k)
        if not candidates or not RERANKER_AVAILABLE:
            return candidates[:top_k]

        print(f"🔄 Reranking {len(candidates)} candidates for query: '{query[:40]}...'")
        
        # 2. Prepare pairs for reranking: [ [query, doc1], [query, doc2], ... ]
        pairs = [[query, cand["text"]] for cand in candidates]
        
        # 3. Predict scores (Higher is better)
        rerank_scores = self.reranker.predict(pairs)
        
        # 4. Attach and Sort
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(rerank_scores[i])
            cand["type"] = "reranked"
            
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

# ============================================================================
# DEMO
# ============================================================================
def run_task1_demo():
    print("=" * 60)
    print("🚀 TASK 1: Hybrid Search + Cross-Encoder Reranking")
    print("=" * 60)
    
    arxiv = ArXivScraper(); pubmed = PubMedScraper(); db = VectorDB()
    
    # 1. Search & Ingest
    query = "breast cancer detection AI"
    print(f"\n� Mining & Indexing: '{query}'")
    all_papers = arxiv.search(query, max_results=2, extract_pdfs=True) + pubmed.search(query, max_results=2)
    
    chunks = []
    for p in all_papers:
        text = p.full_text if p.full_text else p.abstract
        p_chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
        chunks.extend([TextChunk(hashlib.md5(c.encode()).hexdigest()[:12], c, p.paper_id, p.title, p.source, i) for i, c in enumerate(p_chunks)])
    
    db.add_chunks(chunks)
    
    # 2. Reranked Search
    print(f"\n🎯 Performing Reranked Search...")
    results = db.search_with_rerank("machine learning for oncology", top_k=3, retrieve_k=10)
    
    for i, res in enumerate(results, 1):
        print(f"\n   [{i}] (Rerank Score: {res['rerank_score']:.3f}) | Source: {res['metadata']['source']}")
        print(f"   Text: {res['text'][:140]}...")

    return db

if __name__ == "__main__":
    db = run_task1_demo()
