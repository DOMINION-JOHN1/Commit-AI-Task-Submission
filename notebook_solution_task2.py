"""
TASK 2: Complex Question Handling with Query Decomposition
===========================================================
Enhanced with LangChain integration and Two-Stage Retrieval (Reranked).

Copy this code into your Jupyter notebook cell for Task 2.
"""

# ============================================================================
# INSTALLATION (Run this once)
# ============================================================================
# !pip install langchain-openai langchain-core langchain-community openai pydantic

import os
import json
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# LangChain Imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================================
# CONFIGURATION
# ============================================================================
# ⚠️ SET YOUR API KEY HERE
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# ============================================================================
# DATA MODELS
# ============================================================================

class SubQuery(BaseModel):
    query: str = Field(description="A simpler, focused sub-question derived from the complex question.")
    reasoning: str = Field(description="Brief explanation of why this sub-query is necessary to answer the main question.")
    order: int = Field(description="The sequential order in which this sub-query should be answered (1, 2, ...).")

class DecomposedQuery(BaseModel):
    is_complex: bool = Field(description="Whether the original question requires decomposition.")
    sub_queries: List[SubQuery] = Field(description="A list of 2-4 sub-queries to solve the complex question.")

# ============================================================================
# QUERY DECOMPOSER (LangChain Implementation)
# ============================================================================

class QueryDecomposer:
    """Uses LangChain to break down complex queries into manageable sub-queries."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=DecomposedQuery)
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are a specialized RAG Query Optimizer. 
            Your task is to analyze a complex scientific question and break it down into focused sub-questions.
            
            Based on the 'Chain-of-Thought' (CoT) research, complex questions benefit from multi-step reasoning.
            
            Complex Question: {question}
            
            Instructions:
            1. Determine if the question is complex (requires comparing multiple concepts, multi-part answers, or specific technical details).
            2. If complex, generate 2-3 logical sub-queries that cover foundational concepts first, then relational ones.
            3. Each sub-query must be specific and answerable independently.
            4. If the question is simple, set is_complex to false and sub_queries to empty.
            
            {format_instructions}
            """
        )
        
        # Define the chain
        self.chain = self.prompt | self.llm | self.parser

    def decompose(self, question: str) -> Dict[str, Any]:
        """
        Executes the LangChain decomposition.
        """
        print(f"🧠 Analyzing complexity of: '{question}'")
        try:
            return self.chain.invoke({
                "question": question, 
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            print(f"❌ Decomposition failed: {e}")
            return {"is_complex": False, "sub_queries": []}

# ============================================================================
# RAG PIPELINE (Synthesis & Multi-Hop Retrieval with RERANKING)
# ============================================================================

class RAGPipeline:
    """Full RAG Pipeline including search, decomposition, and synthesis."""
    
    def __init__(self, vector_db, api_key: Optional[str] = None):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            
        self.db = vector_db
        self.decomposer = QueryDecomposer()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        self.synthesis_prompt = ChatPromptTemplate.from_template(
            """You are a Scientific Knowledge Assistant.
            Answer the user's question precisely using the provided context.
            
            Question: {question}
            
            Context:
            {context}
            
            Instructions:
            1. Cite specific sources (Paper Titles) where possible.
            2. Be technical and accurate.
            3. If information is missing, state what's unknown.
            4. If multiple papers are used, synthesize the findings.
            
            Answer:"""
        )

    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """Helper to format retrieval results into a clean text block."""
        context_blocks = []
        seen_texts = set()
        
        for res in results:
            text = res['text']
            if text not in seen_texts:
                seen_texts.add(text)
                title = res['metadata'].get('paper_title', 'Unknown Paper')
                # Include rerank score if available for transparency
                score_info = f" [Relevance: {res.get('rerank_score', res.get('score', 0)):.2f}]" if 'rerank_score' in res else ""
                context_blocks.append(f"Source [{title}]{score_info}:\n{text}")
        
        return "\n\n---\n\n".join(context_blocks)

    def query(self, question: str, top_k: int = 4, use_reranking: bool = True) -> Dict[str, Any]:
        """
        Main entry point for querying.
        1. Decompose if complex
        2. Retrieve with Two-Stage Reranking (NEW)
        3. Synthesize
        """
        start_time = time.time()
        
        # Step 1: Decompose
        decomp_result = self.decomposer.decompose(question)
        is_complex = decomp_result.get("is_complex", False)
        sub_queries = decomp_result.get("sub_queries", [])
        
        retrieved_results = []
        
        # Step 2: Retrieve using Reranking
        if is_complex and sub_queries:
            print(f"🔀 Multi-hop retrieval for {len(sub_queries)} sub-queries...")
            for sq in sub_queries:
                print(f"   - Querying: {sq['query']}")
                # Use the NEW reranked search from Task 1
                if use_reranking:
                    res = self.db.search_with_rerank(sq['query'], top_k=top_k, retrieve_k=15)
                else:
                    res = self.db.hybrid_search(sq['query'], top_k=top_k)
                retrieved_results.extend(res)
        else:
            print(f"📝 Simple retrieval with {'reranking' if use_reranking else 'hybrid search'}...")
            if use_reranking:
                retrieved_results = self.db.search_with_rerank(question, top_k=top_k, retrieve_k=15)
            else:
                retrieved_results = self.db.hybrid_search(question, top_k=top_k)
            
        # Step 3: Synthesis
        print("💡 Synthesizing final answer...")
        context = self._format_context(retrieved_results)
        
        chain = self.synthesis_prompt | self.llm
        answer_response = chain.invoke({"question": question, "context": context})
        answer_text = answer_response.content
        
        elapsed = time.time() - start_time
        
        # Extract unique sources
        sources = list(set([r['metadata'].get('paper_title', 'Unknown') for r in retrieved_results]))
        
        return {
            "answer": answer_text,
            "sub_queries": sub_queries,
            "retrieved_count": len(retrieved_results),
            "unique_sources": len(sources),
            "is_complex": is_complex,
            "latency": round(elapsed, 2),
            "reranking_used": use_reranking
        }

# ============================================================================
# MAIN EXECUTION - TASK 2 DEMO
# ============================================================================

def run_task2_demo(db):
    print("=" * 60)
    print("🚀 RUNNING TASK 2: Complex Query + Reranked Retrieval")
    print("=" * 60)
    
    # 1. Initialize Pipeline (Uses 'db' from Task 1)
    rag = RAGPipeline(db)
    
    # 2. Define a Complex Question
    complex_q = (
        "How does the effectiveness of transformer-based models compare to "
        "traditional CNNs for automated lesion detection in MRI scans?"
    )
    
    # 3. Process with Reranking
    result = rag.query(complex_q, use_reranking=True)
    
    # 4. Display Results
    print("\n" + "-" * 40)
    print(f"❓ QUESTION: {complex_q}")
    print("-" * 40)
    
    if result["is_complex"]:
        print("\n🔀 SUB-QUERIES GENERATED:")
        for i, sq in enumerate(result["sub_queries"], 1):
            print(f"   {i}. {sq['query']}")
            print(f"      Reason: {sq['reasoning']}")
    
    print(f"\n📄 FINAL ANSWER:")
    print(result["answer"])
    
    print(f"\n📊 METRICS:")
    print(f"   - Processing Time: {result['latency']}s")
    print(f"   - Context Chunks: {result['retrieved_count']}")
    print(f"   - Unique Sources: {result['unique_sources']}")
    print(f"   - Reranking: {'✅ Enabled' if result['reranking_used'] else '❌ Disabled'}")
    
    print("\n" + "=" * 60)
    print("✅ TASK 2 COMPLETE")
    print("=" * 60)
    
    return rag

# Execute (Note: requires 'db' variable from Task 1 cell)
if __name__ == "__main__":
    # Check if DB exists from Task 1
    if 'db' in globals():
        run_task2_demo(db)
    else:
        print("⚠️ ERROR: 'db' not found. Please run Task 1 cell first.")
