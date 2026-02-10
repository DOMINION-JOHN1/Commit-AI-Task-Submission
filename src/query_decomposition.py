"""
Query Decomposition Module

Handles complex questions by breaking them into sub-queries,
executing each against the vector database, and composing answers.

This is the core of Task 2 - handling complex, multi-hop questions.
"""

import time
from typing import Optional

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_config
from .logging_utils import get_logger
from .models import QueryResult, RAGResponse, SubQuery
from .vector_db import VectorDatabase, get_vector_db


logger = get_logger()


# Prompt templates
DECOMPOSITION_PROMPT = """You are an expert at breaking down complex scientific questions into simpler sub-questions.

Given a complex question, decompose it into 2-5 simpler, focused sub-questions that together will help answer the original question.

For each sub-question, provide:
1. The sub-question text
2. Brief reasoning for why this sub-question is needed
3. The order in which it should be answered (1, 2, 3, etc.)

Original Question: {question}

Respond in the following JSON format:
{{
    "sub_queries": [
        {{
            "query_text": "the sub-question",
            "reasoning": "why this is needed",
            "order": 1,
            "depends_on": []
        }},
        ...
    ]
}}

Only output valid JSON. No additional text."""

SYNTHESIS_PROMPT = """You are a scientific research assistant. Your task is to synthesize information from multiple sources to answer a complex question.

Original Question: {question}

Retrieved Information:
{context}

Instructions:
1. Carefully analyze all the retrieved information
2. Synthesize a comprehensive answer to the original question
3. Only use information from the provided context
4. If the context doesn't contain enough information to fully answer, acknowledge the limitations
5. Cite the source papers when making claims

Provide your answer:"""

SIMPLE_RAG_PROMPT = """You are a scientific research assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Answer based only on the provided context
2. If the context doesn't contain relevant information, say so
3. Be concise but comprehensive
4. Cite sources when possible

Answer:"""


class QueryDecomposer:
    """
    Handles decomposition of complex queries into simpler sub-queries.
    
    Uses an LLM to intelligently break down multi-faceted questions.
    """
    
    def __init__(self):
        self.config = get_config().llm
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Please set it in your .env file."
                )
            self._client = OpenAI(api_key=self.config.api_key)
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, prompt: str) -> str:
        """Make a retry-enabled LLM call."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content
    
    def is_complex_query(self, query: str) -> bool:
        """
        Determine if a query is complex enough to require decomposition.
        
        Heuristics:
        - Contains multiple question words (what, how, why, etc.)
        - Contains conjunctions suggesting multiple parts (and, or, also)
        - Is relatively long (> 100 characters)
        - Contains comparison words (compare, difference, versus)
        """
        query_lower = query.lower()
        
        # Question words
        question_words = ["what", "how", "why", "when", "where", "which", "who"]
        question_count = sum(1 for w in question_words if w in query_lower)
        
        # Complexity indicators
        complexity_indicators = [
            "and", "also", "additionally", "furthermore",
            "compare", "difference", "versus", "vs",
            "relationship between", "how does",
            "multiple", "various", "several"
        ]
        has_complexity = any(ind in query_lower for ind in complexity_indicators)
        
        # Length check
        is_long = len(query) > 100
        
        return (question_count >= 2) or has_complexity or is_long
    
    def decompose(self, query: str) -> list[SubQuery]:
        """
        Decompose a complex query into sub-queries.
        
        Args:
            query: The complex question to decompose
        
        Returns:
            List of SubQuery objects
        """
        logger.info(f"Decomposing query", query_preview=query[:50])
        
        with logger.timed_operation("query_decomposition"):
            prompt = DECOMPOSITION_PROMPT.format(question=query)
            response = self._call_llm(prompt)
            
            # Parse JSON response
            import json
            try:
                # Clean up response if needed
                response = response.strip()
                if response.startswith("```json"):
                    response = response[7:]
                if response.startswith("```"):
                    response = response[3:]
                if response.endswith("```"):
                    response = response[:-3]
                
                data = json.loads(response)
                sub_queries = [
                    SubQuery(
                        query_text=sq["query_text"],
                        reasoning=sq["reasoning"],
                        order=sq["order"],
                        depends_on=sq.get("depends_on", [])
                    )
                    for sq in data["sub_queries"]
                ]
                
                # Sort by order
                sub_queries.sort(key=lambda x: x.order)
                
                logger.info(
                    f"Query decomposed",
                    num_sub_queries=len(sub_queries)
                )
                
                return sub_queries
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse decomposition response", error=str(e))
                # Return original query as single sub-query
                return [SubQuery(
                    query_text=query,
                    reasoning="Original query (decomposition failed)",
                    order=1,
                    depends_on=[]
                )]


class RAGPipeline:
    """
    Complete RAG pipeline with query decomposition support.
    
    This class orchestrates:
    1. Query complexity analysis
    2. Query decomposition (for complex queries)
    3. Vector database retrieval
    4. Answer synthesis using LLM
    """
    
    def __init__(self, vector_db: Optional[VectorDatabase] = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_db: VectorDatabase instance (creates new if not provided)
        """
        self.config = get_config().llm
        self.vector_db = vector_db or get_vector_db()
        self.decomposer = QueryDecomposer()
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Please set it in your .env file."
                )
            self._client = OpenAI(api_key=self.config.api_key)
        return self._client
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, prompt: str) -> str:
        """Make a retry-enabled LLM call."""
        response = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content
    
    def _format_context(self, results: list[QueryResult]) -> str:
        """Format retrieved results as context for the LLM."""
        context_parts = []
        
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            context_parts.append(
                f"[Source {i}: {chunk.paper_title} ({chunk.source.value})]:\n"
                f"{chunk.text}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _simple_query(self, query: str, top_k: int = 5) -> RAGResponse:
        """
        Handle a simple query with standard RAG.
        
        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
        
        Returns:
            RAGResponse with the answer
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        results = self.vector_db.search(query, top_k=top_k)
        
        if not results:
            return RAGResponse(
                original_query=query,
                sub_queries=[],
                retrieved_chunks=results,
                answer="I couldn't find any relevant information in the database to answer your question.",
                sources_used=[],
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Format context and generate answer
        context = self._format_context(results)
        prompt = SIMPLE_RAG_PROMPT.format(context=context, question=query)
        
        answer = self._call_llm(prompt)
        
        # Extract unique paper IDs
        sources = list(set(r.chunk.paper_id for r in results))
        
        # Calculate confidence based on retrieval scores
        avg_score = sum(r.score for r in results) / len(results) if results else 0
        
        return RAGResponse(
            original_query=query,
            sub_queries=[],
            retrieved_chunks=results,
            answer=answer,
            sources_used=sources,
            confidence=avg_score,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def _complex_query(self, query: str, top_k: int = 5) -> RAGResponse:
        """
        Handle a complex query with decomposition.
        
        Args:
            query: The complex question
            top_k: Number of chunks to retrieve per sub-query
        
        Returns:
            RAGResponse with synthesized answer
        """
        start_time = time.time()
        
        # Decompose the query
        sub_queries = self.decomposer.decompose(query)
        
        # Execute each sub-query
        all_results: list[QueryResult] = []
        sub_query_results: dict[int, list[QueryResult]] = {}
        
        for sq in sub_queries:
            logger.debug(f"Executing sub-query", order=sq.order, query=sq.query_text[:50])
            
            results = self.vector_db.search(sq.query_text, top_k=top_k)
            sub_query_results[sq.order] = results
            all_results.extend(results)
        
        if not all_results:
            return RAGResponse(
                original_query=query,
                sub_queries=sub_queries,
                retrieved_chunks=[],
                answer="I couldn't find relevant information to answer any part of your question.",
                sources_used=[],
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
        
        # Deduplicate results by chunk_id
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.chunk.chunk_id not in seen_ids:
                seen_ids.add(result.chunk.chunk_id)
                unique_results.append(result)
        
        # Sort by score (highest first)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Take top results for synthesis
        top_results = unique_results[:top_k * 2]  # Allow more context for complex queries
        
        # Build enhanced context with sub-query information
        context_parts = []
        for sq in sub_queries:
            sq_results = sub_query_results.get(sq.order, [])
            if sq_results:
                context_parts.append(f"\n=== Information for: {sq.query_text} ===")
                for result in sq_results[:top_k]:
                    context_parts.append(
                        f"[{result.chunk.paper_title}]:\n{result.chunk.text}"
                    )
        
        context = "\n\n".join(context_parts)
        
        # Synthesize final answer
        prompt = SYNTHESIS_PROMPT.format(question=query, context=context)
        answer = self._call_llm(prompt)
        
        # Extract unique sources
        sources = list(set(r.chunk.paper_id for r in top_results))
        
        # Calculate confidence
        avg_score = sum(r.score for r in top_results) / len(top_results) if top_results else 0
        
        return RAGResponse(
            original_query=query,
            sub_queries=sub_queries,
            retrieved_chunks=top_results,
            answer=answer,
            sources_used=sources,
            confidence=avg_score,
            processing_time_ms=(time.time() - start_time) * 1000
        )
    
    def query(
        self,
        question: str,
        force_decomposition: bool = False,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Automatically determines if the query is complex and needs
        decomposition, or if it can be handled directly.
        
        Args:
            question: The user's question
            force_decomposition: Force query decomposition even for simple queries
            top_k: Number of chunks to retrieve
        
        Returns:
            RAGResponse with the complete response
        """
        logger.info(f"Processing query", query_preview=question[:50])
        
        with logger.timed_operation("rag_query", query_length=len(question)):
            # Determine complexity
            is_complex = force_decomposition or self.decomposer.is_complex_query(question)
            
            if is_complex:
                logger.info("Query identified as complex, using decomposition")
                return self._complex_query(question, top_k)
            else:
                logger.info("Query identified as simple, using direct retrieval")
                return self._simple_query(question, top_k)


def create_rag_pipeline(vector_db: Optional[VectorDatabase] = None) -> RAGPipeline:
    """Factory function to create a RAG pipeline."""
    return RAGPipeline(vector_db)
