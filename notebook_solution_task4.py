"""
TASK 4: RAG Evaluation - Measuring Model Accuracy
==================================================
Implementing RAGAS-style metrics and LLM-as-judge Evaluation.

Copy this code into your Jupyter notebook cell for Task 4.
"""

# ============================================================================
# INSTALLATION (Run this once)
# ============================================================================
# !pip install langchain-openai langchain-core pydantic numpy pandas

import os
import json
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

# LangChain for LLM-as-judge
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION
# ============================================================================
# ⚠️ SET YOUR API KEY HERE
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# ============================================================================
# SUGGESTED EVALUATION STRATEGY
# ============================================================================

EVALUATION_EXPLANATION = """
HOW TO MEASURE RAG MODEL ACCURACY
=================================

THE CHALLENGE
-------------
Traditional NLP metrics (BLEU, ROUGE) don't work well for RAG because:
- They rely on exact text matching
- They don't measure factual accuracy
- They ignore whether the answer actually addresses the question

RAGAS FRAMEWORK APPROACH
------------------------
RAGAS (Retrieval Augmented Generation Assessment) provides specialized metrics:

1. FAITHFULNESS (Most Important)
   What it measures: Is the answer factually consistent with the retrieved context?
   Why it matters: Prevents hallucination - the #1 concern with LLMs.
   How to calculate:
   1. Extract all claims from the generated answer
   2. For each claim, check if it's supported by the context
   3. Score = (supported claims) / (total claims)

2. ANSWER RELEVANCY
   What it measures: Does the answer actually address the question?
   Why it matters: Ensures the system provides useful responses.
   How to calculate:
   1. Generate hypothetical questions the answer could address
   2. Compare these with the original question
   3. High similarity = high relevancy

3. CONTEXT RELEVANCY
   What it measures: How much of the retrieved context is useful?
   Why it matters: Evaluates retrieval quality (signal vs noise).
   How to calculate:
   1. Analyze each retrieved chunk
   2. Determine what percentage is relevant to the question
   3. Higher ratio = better retrieval

4. CONTEXT RECALL (with ground truth)
   What it measures: Does the context contain all info needed for the answer?
   Why it matters: Ensures the retrieval captures complete information.
   Requires: Ground truth answers for comparison.
"""

# ============================================================================
# EVALUATION MODELS (Pydantic)
# ============================================================================

class FaithfulnessScore(BaseModel):
    statement: str = Field(description="The claim or statement extracted from the answer.")
    supported_by_context: bool = Field(description="Whether the statement is supported by the provided context.")
    reasoning: str = Field(description="Explanation for the support decision.")

class FaithfulnessEvaluation(BaseModel):
    overall_score: float = Field(description="Score between 0 and 1 (supported statements / total statements).")
    evaluations: List[FaithfulnessScore] = Field(description="Detailed breakdown of each statement.")

class RelevancyEvaluation(BaseModel):
    score: float = Field(description="Score between 0 and 1 indicating how relevant the answer is to the question.")
    reasoning: str = Field(description="Explanation for the relevancy score.")

# ============================================================================
# RAG EVALUATOR (LLM-as-judge)
# ============================================================================

class RAGEvaluator:
    """
    Implements core metrics from the RAGAS framework using GPT-4o-mini as the judge.
    1. Faithfulness: Is the answer factually grounded in the context? (Hallucination check)
    2. Relevancy: Does the answer address the question?
    3. Context Quality: How useful were the retrieved chunks?
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.parser_f = JsonOutputParser(pydantic_object=FaithfulnessEvaluation)
        self.parser_r = JsonOutputParser(pydantic_object=RelevancyEvaluation)

    def evaluate_faithfulness(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Check for hallucinations. 
        Higher score = More grounded in context.
        """
        prompt = ChatPromptTemplate.from_template(
            """Evaluate the faithfulness of the answer based on the context.
            1. Extract the main claims from the answer.
            2. For each claim, check if it can be verified using ONLY the provided context.
            
            Question: {question}
            Context: {context}
            Answer: {answer}
            
            {format_instructions}
            """
        )
        
        chain = prompt | self.llm | self.parser_f
        try:
            return chain.invoke({
                "question": question,
                "context": context,
                "answer": answer,
                "format_instructions": self.parser_f.get_format_instructions()
            })
        except Exception as e:
            return {"overall_score": 0.0, "evaluations": [], "error": str(e)}

    def evaluate_relevancy(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Check if the answer addresses the question directly.
        """
        prompt = ChatPromptTemplate.from_template(
            """Rate the relevancy of the answer to the question on a scale from 0.0 to 1.0. 
            0.0 means completely off-topic. 1.0 means perfectly addresses the query.
            
            Question: {question}
            Answer: {answer}
            
            {format_instructions}
            """
        )
        
        chain = prompt | self.llm | self.parser_r
        try:
            return chain.invoke({
                "question": question,
                "answer": answer,
                "format_instructions": self.parser_r.get_format_instructions()
            })
        except Exception as e:
            return {"score": 0.0, "reasoning": "Error occurred", "error": str(e)}

# ============================================================================
# RETRIEVAL METRICS (Classical)
# ============================================================================

class RetrievalMetrics:
    """Calculates non-LLM metrics for the retrieval component."""
    
    @staticmethod
    def calc_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        Calculates Mean Reciprocal Rank (MRR).
        1.0 if the first result is relevant. 0.5 if the second is relevant, etc.
        """
        for i, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / i
        return 0.0

    @staticmethod
    def calc_hit_rate(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """Percentage of queries where at least one relevant document was found."""
        for doc_id in retrieved_ids:
            if doc_id in relevant_ids:
                return 1.0
        return 0.0

# ============================================================================
# DEMONSTRATION WORKFLOW
# ============================================================================

def run_task4_demo():
    print("=" * 60)
    print("🚀 RUNNING TASK 4: RAG Evaluation & LLM-as-judge")
    print("=" * 60)

    print(EVALUATION_EXPLANATION)
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION EXAMPLE")
    print("=" * 60)

    # 1. Mock Data for Evaluation
    sample_q = "How effective are mRNA vaccines against different variants?"
    sample_context = """
    A study in NEJM showed that mRNA-1273 was 94% effective against early variants. 
    However, efficacy dropped to 88% for the Delta variant. Omicron further reduced 
    neutralizing activity but protection against severe disease remained high.
    """
    
    # CASE A: Faithful Answer
    faithful_a = "mRNA vaccines show high overall efficacy (94%), though it decreases slightly for Delta (88%)."
    
    # CASE B: Hallucination (Unsupported claim)
    hallucinated_a = "mRNA vaccines are 100% effective against Omicron and prevent all transmission."

    evaluator = RAGEvaluator()

    # 2. Evaluate Faithfulness
    print(f"\n1. Judging Faithfulness (Hallucination Detection)...")
    
    res_f = evaluator.evaluate_faithfulness(sample_q, faithful_a, sample_context)
    print(f"   ✅ Faithful Answer Score: {res_f['overall_score']}")
    
    res_h = evaluator.evaluate_faithfulness(sample_q, hallucinated_a, sample_context)
    print(f"   ❌ Hallucinated Answer Score: {res_h['overall_score']}")
    for eval_item in res_h['evaluations']:
        if not eval_item['supported_by_context']:
            print(f"      - Hallucination Detected: '{eval_item['statement']}'")
            print(f"        Reason: {eval_item['reasoning']}")

    # 3. Evaluate Relevancy
    print(f"\n2. Judging Answer Relevancy...")
    res_rel = evaluator.evaluate_relevancy(sample_q, faithful_a)
    print(f"   ✅ Relevancy Score: {res_rel['score']}")
    print(f"      Reasoning: {res_rel['reasoning']}")

    # 4. Retrieval Metrics (Classical)
    print(f"\n3. Classical Retrieval Metrics (MRR/Hit Rate)...")
    retrieved = ["doc_9", "doc_1", "doc_4"] # System retrieved these
    relevant = ["doc_1", "doc_2"]           # Ground truth relevant papers
    
    mrr = RetrievalMetrics.calc_mrr(retrieved, relevant)
    hit = RetrievalMetrics.calc_hit_rate(retrieved, relevant)
    
    print(f"   - MRR: {mrr} (Relevant document was at Rank 2)")
    print(f"   - Hit Rate: {hit} (Found at least one relevant doc)")

    print("\n" + "=" * 60)
    print("✅ TASK 4 COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    # Note: Requires OPENAI_API_KEY
    if os.environ.get("OPENAI_API_KEY"):
        run_task4_demo()
    else:
        print("⚠️  To run the LLM-as-judge demo, please set OPENAI_API_KEY.")
