"""
RAG Evaluation Module

Implements multiple evaluation strategies for measuring RAG system quality.
Based on RAGAS and custom metrics.

This addresses Task 4 - Model Accuracy Evaluation
"""

from dataclasses import dataclass, field
from typing import Optional
import json

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import get_config
from .logging_utils import get_logger
from .models import RAGResponse


logger = get_logger()


@dataclass
class EvaluationResult:
    """Results from evaluating a RAG response."""
    
    # Core RAGAS-style metrics (0-1 scale)
    faithfulness: float = 0.0  # Is the answer supported by context?
    answer_relevancy: float = 0.0  # Does the answer address the question?
    context_relevancy: float = 0.0  # Is the retrieved context relevant?
    context_recall: float = 0.0  # Does context cover all needed info?
    
    # Custom metrics
    groundedness: float = 0.0  # Are claims grounded in sources?
    completeness: float = 0.0  # Does the answer fully address the question?
    
    # Aggregate score
    overall_score: float = 0.0
    
    # Details
    explanation: str = ""
    claims_verified: int = 0
    claims_total: int = 0
    
    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
            "context_recall": self.context_recall,
            "groundedness": self.groundedness,
            "completeness": self.completeness,
            "overall_score": self.overall_score,
            "explanation": self.explanation,
            "claims_verified": self.claims_verified,
            "claims_total": self.claims_total
        }


# Evaluation prompts
FAITHFULNESS_PROMPT = """You are evaluating the faithfulness of an AI-generated answer.

Faithfulness measures whether the answer is factually consistent with the provided context.
An answer is faithful if ALL claims in it can be verified from the context.

Context:
{context}

Answer:
{answer}

Instructions:
1. Identify all factual claims in the answer
2. For each claim, check if it is supported by the context
3. Calculate the ratio of supported claims to total claims

Respond in JSON format:
{{
    "claims": [
        {{"claim": "the claim text", "supported": true/false, "evidence": "quote from context or 'not found'"}}
    ],
    "faithfulness_score": 0.0 to 1.0,
    "explanation": "brief explanation"
}}

Only output valid JSON."""

ANSWER_RELEVANCY_PROMPT = """You are evaluating how relevant an answer is to the question asked.

Answer relevancy measures whether the answer directly addresses what was asked.

Question: {question}

Answer: {answer}

Instructions:
1. Determine if the answer addresses the main question
2. Check if all parts of the question are answered
3. Identify any irrelevant or tangential information

Respond in JSON format:
{{
    "addresses_question": true/false,
    "parts_addressed": ["list of question parts that were answered"],
    "parts_missing": ["list of question parts not answered"],
    "relevancy_score": 0.0 to 1.0,
    "explanation": "brief explanation"
}}

Only output valid JSON."""

CONTEXT_RELEVANCY_PROMPT = """You are evaluating the relevancy of retrieved context for answering a question.

Context relevancy measures the "signal-to-noise ratio" - how much of the context is actually useful.

Question: {question}

Retrieved Context:
{context}

Instructions:
1. Identify which parts of the context are relevant to answering the question
2. Calculate what percentage of the context is useful
3. Note any important missing information

Respond in JSON format:
{{
    "relevant_portions": ["list of relevant information found"],
    "irrelevant_portions": ["list of irrelevant information"],
    "context_relevancy_score": 0.0 to 1.0,
    "missing_information": ["what would have been helpful but is missing"],
    "explanation": "brief explanation"
}}

Only output valid JSON."""


class RAGEvaluator:
    """
    Evaluates RAG system responses using multiple metrics.
    
    Implements RAGAS-style evaluation with:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Relevancy: Is the retrieved context useful?
    - Custom metrics for comprehensive evaluation
    """
    
    def __init__(self):
        self.config = get_config().llm
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            if not self.config.api_key:
                raise ValueError("OPENAI_API_KEY not set")
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
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response with cleanup."""
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse evaluation JSON, returning defaults")
            return {}
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str
    ) -> tuple[float, int, int, str]:
        """
        Evaluate faithfulness of the answer to the context.
        
        Returns:
            Tuple of (score, claims_verified, total_claims, explanation)
        """
        with logger.timed_operation("evaluate_faithfulness"):
            prompt = FAITHFULNESS_PROMPT.format(
                context=context,
                answer=answer
            )
            
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)
            
            score = data.get("faithfulness_score", 0.0)
            claims = data.get("claims", [])
            verified = sum(1 for c in claims if c.get("supported", False))
            total = len(claims)
            explanation = data.get("explanation", "")
            
            return score, verified, total, explanation
    
    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> tuple[float, str]:
        """
        Evaluate how relevant the answer is to the question.
        
        Returns:
            Tuple of (score, explanation)
        """
        with logger.timed_operation("evaluate_answer_relevancy"):
            prompt = ANSWER_RELEVANCY_PROMPT.format(
                question=question,
                answer=answer
            )
            
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)
            
            score = data.get("relevancy_score", 0.0)
            explanation = data.get("explanation", "")
            
            return score, explanation
    
    def evaluate_context_relevancy(
        self,
        question: str,
        context: str
    ) -> tuple[float, str]:
        """
        Evaluate how relevant the retrieved context is.
        
        Returns:
            Tuple of (score, explanation)
        """
        with logger.timed_operation("evaluate_context_relevancy"):
            prompt = CONTEXT_RELEVANCY_PROMPT.format(
                question=question,
                context=context
            )
            
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)
            
            score = data.get("context_relevancy_score", 0.0)
            explanation = data.get("explanation", "")
            
            return score, explanation
    
    def evaluate(
        self,
        rag_response: RAGResponse,
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Perform comprehensive evaluation of a RAG response.
        
        Args:
            rag_response: The RAGResponse to evaluate
            ground_truth: Optional ground truth answer for comparison
        
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(
            "Starting RAG evaluation",
            query_preview=rag_response.original_query[:50]
        )
        
        # Format context from retrieved chunks
        context = "\n\n".join([
            f"[{r.chunk.paper_title}]: {r.chunk.text}"
            for r in rag_response.retrieved_chunks
        ])
        
        if not context:
            return EvaluationResult(
                explanation="No context was retrieved for evaluation"
            )
        
        # Evaluate faithfulness
        faithfulness, verified, total, faith_explanation = self.evaluate_faithfulness(
            rag_response.answer,
            context
        )
        
        # Evaluate answer relevancy
        answer_relevancy, rel_explanation = self.evaluate_answer_relevancy(
            rag_response.original_query,
            rag_response.answer
        )
        
        # Evaluate context relevancy
        context_relevancy, ctx_explanation = self.evaluate_context_relevancy(
            rag_response.original_query,
            context
        )
        
        # Calculate groundedness (similar to faithfulness but stricter)
        groundedness = faithfulness * 0.8 + (verified / max(total, 1)) * 0.2
        
        # Estimate completeness from answer relevancy
        completeness = answer_relevancy
        
        # Calculate overall score (weighted average)
        overall_score = (
            faithfulness * 0.30 +      # Most important: factual accuracy
            answer_relevancy * 0.25 +   # Important: actually answering
            context_relevancy * 0.20 +  # Retrieval quality
            groundedness * 0.15 +       # Source attribution
            completeness * 0.10         # Complete answers
        )
        
        # Compile explanation
        explanation = f"""
Faithfulness ({faithfulness:.2f}): {faith_explanation}
Answer Relevancy ({answer_relevancy:.2f}): {rel_explanation}
Context Relevancy ({context_relevancy:.2f}): {ctx_explanation}
        """.strip()
        
        result = EvaluationResult(
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_relevancy=context_relevancy,
            context_recall=context_relevancy,  # Using as proxy
            groundedness=groundedness,
            completeness=completeness,
            overall_score=overall_score,
            explanation=explanation,
            claims_verified=verified,
            claims_total=total
        )
        
        logger.info(
            "Evaluation complete",
            overall_score=overall_score,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy
        )
        
        return result


@dataclass
class EvaluationDataset:
    """A dataset for evaluating RAG systems."""
    
    questions: list[str] = field(default_factory=list)
    ground_truths: list[str] = field(default_factory=list)  # Optional
    contexts: list[str] = field(default_factory=list)  # Optional golden contexts
    
    def add_example(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        context: Optional[str] = None
    ):
        """Add an evaluation example."""
        self.questions.append(question)
        self.ground_truths.append(ground_truth or "")
        self.contexts.append(context or "")
    
    def __len__(self) -> int:
        return len(self.questions)


class BatchEvaluator:
    """
    Evaluates a RAG system on a batch of questions.
    
    Useful for systematic evaluation and regression testing.
    """
    
    def __init__(self, evaluator: Optional[RAGEvaluator] = None):
        self.evaluator = evaluator or RAGEvaluator()
    
    def evaluate_batch(
        self,
        rag_responses: list[RAGResponse],
        ground_truths: Optional[list[str]] = None
    ) -> dict:
        """
        Evaluate a batch of RAG responses.
        
        Args:
            rag_responses: List of RAGResponse objects
            ground_truths: Optional list of ground truth answers
        
        Returns:
            Dictionary with aggregate metrics and individual results
        """
        logger.info(f"Starting batch evaluation", size=len(rag_responses))
        
        results = []
        for i, response in enumerate(rag_responses):
            ground_truth = ground_truths[i] if ground_truths else None
            result = self.evaluator.evaluate(response, ground_truth)
            results.append(result)
        
        # Calculate aggregate metrics
        if results:
            avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
            avg_relevancy = sum(r.answer_relevancy for r in results) / len(results)
            avg_context_rel = sum(r.context_relevancy for r in results) / len(results)
            avg_overall = sum(r.overall_score for r in results) / len(results)
        else:
            avg_faithfulness = avg_relevancy = avg_context_rel = avg_overall = 0.0
        
        return {
            "aggregate": {
                "faithfulness": avg_faithfulness,
                "answer_relevancy": avg_relevancy,
                "context_relevancy": avg_context_rel,
                "overall_score": avg_overall,
                "total_evaluated": len(results)
            },
            "individual_results": [r.to_dict() for r in results]
        }


def create_evaluator() -> RAGEvaluator:
    """Factory function to create a RAG evaluator."""
    return RAGEvaluator()


def create_sample_evaluation_dataset() -> EvaluationDataset:
    """
    Create a sample evaluation dataset for testing.
    
    In production, this would be replaced with a curated
    dataset of questions and expected answers.
    """
    dataset = EvaluationDataset()
    
    # Sample scientific questions
    sample_questions = [
        "What are the primary mechanisms of action for mRNA vaccines?",
        "How does CRISPR-Cas9 achieve targeted gene editing?",
        "What are the main differences between supervised and unsupervised learning?",
        "What causes antibiotic resistance in bacteria?",
        "How do neural networks learn from data?",
    ]
    
    for q in sample_questions:
        dataset.add_example(q)
    
    return dataset
