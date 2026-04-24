"""
Task 4: Quality Evaluation (LLM-as-Judge)
Implements RAGAS-style metrics for faithfulness, relevancy, and retrieval precision.
"""

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# os.environ["OPENAI_API_KEY"] = "your-key-here"


class StatementEvaluation(BaseModel):
    statement: str = Field(description="Individual claim extracted from the answer.")
    supported: bool = Field(description="Whether the context supports this specific claim.")
    reasoning: str = Field(description="Evidence from the context justifying the support status.")


class FaithfulnessReport(BaseModel):
    score: float = Field(description="Ratio of supported statements to total statements (0.0 - 1.0).")
    findings: List[StatementEvaluation] = Field(description="Detailed verification results.")


class RelevancyReport(BaseModel):
    score: float = Field(description="Degree to which the answer addresses the unique intent of the query.")
    feedback: str = Field(description="Qualitative assessment of query-answer alignment.")


class RAGEvaluator:
    """Evaluates RAG outputs for structural and factual integrity."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser_f = JsonOutputParser(pydantic_object=FaithfulnessReport)
        self.parser_r = JsonOutputParser(pydantic_object=RelevancyReport)

    def measure_faithfulness(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Calculates faithfulness to detect hallucinations."""
        prompt = ChatPromptTemplate.from_template(
            """Evaluate if the answer provided is grounded in the context.
            
            1. Extract all verifiable claims from the answer.
            2. Cross-reference each claim against the context block.
            
            Query: {query}
            Context: {context}
            Answer: {answer}
            
            {format_instructions}
            """
        )
        chain = prompt | self.llm | self.parser_f
        try:
            return chain.invoke({
                "query": query, "context": context, "answer": answer,
                "format_instructions": self.parser_f.get_format_instructions()
            })
        except Exception as e:
            print(f"Faithfulness eval error: {e}")
            return {"score": 0.0, "findings": []}

    def measure_relevancy(self, query: str, answer: str) -> Dict[str, Any]:
        """Evaluates relevancy to ensure the user's intent was met."""
        prompt = ChatPromptTemplate.from_template(
            """Assess the relevance of the answer to the provided query.
            Assign a score from 0.0 (unrelated) to 1.0 (perfectly addresses intent).
            
            Query: {query}
            Answer: {answer}
            
            {format_instructions}
            """
        )
        chain = prompt | self.llm | self.parser_r
        try:
            return chain.invoke({
                "query": query, "answer": answer,
                "format_instructions": self.parser_r.get_format_instructions()
            })
        except Exception as e:
            print(f"Relevancy eval error: {e}")
            return {"score": 0.0, "feedback": "Evaluation failed"}


class PerformanceMetrics:
    """Utility for classical retrieval metrics (MRR, Hit Rate)."""
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in ground_truth_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def hit_rate(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
        return 1.0 if any(doc_id in ground_truth_ids for doc_id in retrieved_ids) else 0.0


def run_evaluation_demo():
    print("Initiating Task 4 quality assessment...")
    
    # Mock data for demonstration
    query = "What is the primary driver of climate change?"
    answer = "The primary driver of climate change is anthropogenic greenhouse gas emissions, specifically CO2."
    context = "Climate change is largely driven by human activities that release greenhouse gases into the atmosphere."
    
    evaluator = RAGEvaluator()
    
    # 1. LLM-as-Judge results
    faith_res = evaluator.measure_faithfulness(query, answer, context)
    rel_res = evaluator.measure_relevancy(query, answer)
    
    print("\nLLM-Based Evaluation Results:")
    print(f"  Faithfulness Score: {faith_res['score']}")
    print(f"  Relevancy Score:    {rel_res['score']}")
    
    # 2. Traditional metrics results
    retrieved_docs = ["node_122", "node_45", "node_90"]
    relevant_docs = ["node_45", "node_77"]
    
    mrr = PerformanceMetrics.mean_reciprocal_rank(retrieved_docs, relevant_docs)
    hit = PerformanceMetrics.hit_rate(retrieved_docs, relevant_docs)
    
    print("\nRetrieval Performance:")
    print(f"  MRR:      {mrr:.2f}")
    print(f"  Hit Rate: {hit:.0f}")


if __name__ == "__main__":
    if os.environ.get("OPENAI_API_KEY"):
        run_evaluation_demo()
    else:
        print("Note: Skipping demo. OPENAI_API_KEY environment variable not set.")
