"""
Task 2: Advanced Query Intelligence
Handles complex scientific questions via structural decomposition and synthesis.
"""

import os
import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Set OpenAI key if not present in environment
# os.environ["OPENAI_API_KEY"] = "your-key-here"


class SubQuery(BaseModel):
    query: str = Field(description="Sub-question focused on a specific component of the main query.")
    reasoning: str = Field(description="Explanation of why this sub-query is relevant.")
    order: int = Field(description="Sequence index for multi-step reasoning.")


class DecomposedQuery(BaseModel):
    is_complex: bool = Field(description="Whether the query requires multi-step processing.")
    sub_queries: List[SubQuery] = Field(description="List of targeted sub-queries.")


class QueryDecomposer:
    """Decomposes complex requests into atomic sub-questions."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser = JsonOutputParser(pydantic_object=DecomposedQuery)
        
        self.prompt = ChatPromptTemplate.from_template(
            """Translate the following complex scientific question into focused sub-queries.
            
            Original Question: {question}
            
            Guidelines:
            1. If the question involves comparisons, definitions, and applications, create distinct sub-queries for each.
            2. For simple factual lookups, set is_complex to false.
            3. Ensure sub-queries are independent yet collectively exhaustive.
            
            {format_instructions}
            """
        )
        self.chain = self.prompt | self.llm | self.parser

    def decompose(self, question: str) -> Dict[str, Any]:
        try:
            return self.chain.invoke({
                "question": question, 
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            print(f"Decomposition error: {e}")
            return {"is_complex": False, "sub_queries": []}


class RAGPipeline:
    """Complete RAG pipeline orchestrating retrieval, decomposition, and synthesis."""
    def __init__(self, vector_db):
        self.db = vector_db
        self.decomposer = QueryDecomposer()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        self.synthesis_prompt = ChatPromptTemplate.from_template(
            """You are a research assistant. Synthesize a technical briefing based on the context provided.
            
            User Question: {question}
            
            Supporting Context:
            {context}
            
            Requirements:
            1. Use formal, technical language.
            2. Cite specific source papers where relevant.
            3. If the context is insufficient, specify the limitations.
            
            Briefing:"""
        )

    def _prepare_context(self, retrieved_results: List[Dict]) -> str:
        """Formats multiple retrieval results into a cohesive context block."""
        formatted = []
        for res in retrieved_results:
            title = res['metadata'].get('paper_title', 'Unknown Source')
            score = res.get('rerank_score', res.get('score', 0))
            formatted.append(f"Source [{title}] (Relevance: {score:.2f}):\n{res['text']}")
        return "\n\n---\n\n".join(formatted)

    def run_query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        start_time = time.time()
        
        # Decomposition step
        decomp = self.decomposer.decompose(question)
        is_complex = decomp.get("is_complex", False)
        sub_queries = decomp.get("sub_queries", [])
        
        all_results = []
        
        # Targeted retrieval based on complexity
        if is_complex and sub_queries:
            for sq in sub_queries:
                # Using reranked search from Task 1
                res = self.db.search_with_rerank(sq['query'], top_k=top_k)
                all_results.extend(res)
        else:
            all_results = self.db.search_with_rerank(question, top_k=top_k * 2)
            
        # Deduplication of retrieved chunks
        unique_results = {r['id']: r for r in all_results}.values()
        context = self._prepare_context(list(unique_results))
        
        # Answer synthesis
        chain = self.synthesis_prompt | self.llm
        synthesis_res = chain.invoke({"question": question, "context": context})
        
        return {
            "answer": synthesis_res.content,
            "sub_queries": sub_queries,
            "latency": round(time.time() - start_time, 2),
            "sources_count": len(unique_results)
        }


def run_task2_demo(db_instance):
    print("Initializing Task 2 workflow...")
    pipeline = RAGPipeline(db_instance)
    
    complex_question = (
        "Compare the diagnostic accuracy of transformer models versus traditional "
        "CNNs for identifying early-stage lung lesions in CT scans."
    )
    
    print(f"Processing query: {complex_question}")
    result = pipeline.run_query(complex_question)
    
    print(f"\nLatency: {result['latency']}s")
    print(f"Referenced {result['sources_count']} unique context chunks.")
    
    if result["sub_queries"]:
        print("\nSub-queries utilized:")
        for i, sq in enumerate(result["sub_queries"], 1):
            print(f"  {i}. {sq['query']}")
            
    print(f"\nSynthesized Answer:\n{result['answer']}")
    return pipeline


if __name__ == "__main__":
    # Note: Requires VectorDB instance 'db' from Task 1
    if 'db' in globals():
        run_task2_demo(db)
    else:
        print("Error: Global VectorDB instance 'db' not found.")
