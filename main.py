#!/usr/bin/env python3
"""
Scientific Paper RAG System - Main Entry Point

This script demonstrates the complete RAG pipeline:
1. Mining papers from ArXiv and PubMed
2. Chunking and indexing into vector database
3. Querying with automatic complexity detection
4. Evaluating response quality

Usage:
    python main.py --demo           # Run full demonstration
    python main.py --query "..."    # Query the system
    python main.py --ingest         # Only ingest papers
    python main.py --evaluate       # Run evaluation
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config
from src.logging_utils import get_logger
from src.scrapers import UnifiedScraper
from src.chunking import get_chunker
from src.vector_db import get_vector_db
from src.query_decomposition import create_rag_pipeline
from src.evaluation import create_evaluator, create_sample_evaluation_dataset


logger = get_logger()


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def ingest_papers(
    search_query: str = "machine learning healthcare",
    max_per_source: int = 3
) -> int:
    """
    Ingest papers from ArXiv and PubMed into the vector database.
    
    Args:
        search_query: Query to search for papers
        max_per_source: Maximum papers per source
    
    Returns:
        Total number of chunks indexed
    """
    print_header("TASK 1: Mining and Indexing Scientific Papers")
    
    # Initialize components
    scraper = UnifiedScraper()
    chunker = get_chunker("recursive")  # Use recursive for reliability
    vector_db = get_vector_db()
    
    # Step 1: Mine papers
    print(f"🔍 Searching for papers: '{search_query}'")
    print(f"   Max per source: {max_per_source}")
    
    papers = scraper.search_all(search_query, max_per_source=max_per_source)
    
    print(f"\n📄 Found {len(papers)} papers:")
    for i, paper in enumerate(papers, 1):
        print(f"   {i}. [{paper.source.value}] {paper.title[:60]}...")
    
    # Step 2: Chunk papers
    print(f"\n✂️  Chunking papers...")
    all_chunks = []
    
    for paper in papers:
        chunks = chunker.chunk(paper)
        all_chunks.extend(chunks)
        print(f"   - {paper.paper_id}: {len(chunks)} chunks")
    
    # Step 3: Index chunks
    print(f"\n📥 Indexing {len(all_chunks)} chunks into vector database...")
    indexed = vector_db.add_chunks(all_chunks)
    
    # Print stats
    stats = vector_db.get_stats()
    print(f"\n✅ Indexing complete!")
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Persist directory: {stats['persist_directory']}")
    
    return indexed


def query_system(question: str, force_decomposition: bool = False) -> dict:
    """
    Query the RAG system.
    
    Args:
        question: The user's question
        force_decomposition: Force query decomposition
    
    Returns:
        RAGResponse as dictionary
    """
    print_header("TASK 2: Querying the RAG System")
    
    rag_pipeline = create_rag_pipeline()
    
    print(f"❓ Question: {question}")
    
    if force_decomposition:
        print("   (Forcing query decomposition)")
    
    # Execute query
    response = rag_pipeline.query(
        question,
        force_decomposition=force_decomposition
    )
    
    # Display results
    print(f"\n📊 Query Analysis:")
    print(f"   - Complexity: {'Complex' if response.sub_queries else 'Simple'}")
    print(f"   - Processing time: {response.processing_time_ms:.0f}ms")
    print(f"   - Sources used: {len(response.sources_used)}")
    print(f"   - Confidence: {response.confidence:.2%}")
    
    if response.sub_queries:
        print(f"\n🔀 Sub-queries generated:")
        for sq in response.sub_queries:
            print(f"   {sq.order}. {sq.query_text}")
            print(f"      Reason: {sq.reasoning}")
    
    print(f"\n📝 Answer:")
    print("-" * 40)
    print(response.answer)
    print("-" * 40)
    
    if response.retrieved_chunks:
        print(f"\n📚 Sources:")
        seen_papers = set()
        for result in response.retrieved_chunks[:5]:
            if result.chunk.paper_id not in seen_papers:
                seen_papers.add(result.chunk.paper_id)
                print(f"   - {result.chunk.paper_title}")
                print(f"     Score: {result.score:.3f} | ID: {result.chunk.paper_id}")
    
    return response.model_dump()


def evaluate_system(num_questions: int = 3) -> dict:
    """
    Evaluate the RAG system on sample questions.
    
    Args:
        num_questions: Number of questions to evaluate
    
    Returns:
        Evaluation results
    """
    print_header("TASK 4: Evaluating RAG System Quality")
    
    rag_pipeline = create_rag_pipeline()
    evaluator = create_evaluator()
    
    # Sample questions for evaluation
    questions = [
        "What is machine learning and how is it used in healthcare?",
        "What are the benefits and risks of using AI in medical diagnosis?",
        "How can deep learning improve drug discovery processes?",
    ][:num_questions]
    
    print(f"🧪 Evaluating {len(questions)} questions...\n")
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"Question {i}: {question[:50]}...")
        
        # Get RAG response
        response = rag_pipeline.query(question)
        
        # Evaluate
        eval_result = evaluator.evaluate(response)
        
        print(f"   Faithfulness:     {eval_result.faithfulness:.2f}")
        print(f"   Answer Relevancy: {eval_result.answer_relevancy:.2f}")
        print(f"   Context Relevancy:{eval_result.context_relevancy:.2f}")
        print(f"   Overall Score:    {eval_result.overall_score:.2f}")
        print()
        
        results.append({
            "question": question,
            "evaluation": eval_result.to_dict()
        })
    
    # Aggregate results
    if results:
        avg_score = sum(r["evaluation"]["overall_score"] for r in results) / len(results)
        avg_faith = sum(r["evaluation"]["faithfulness"] for r in results) / len(results)
        
        print("=" * 40)
        print("📊 AGGREGATE RESULTS")
        print("=" * 40)
        print(f"   Average Overall Score: {avg_score:.2f}")
        print(f"   Average Faithfulness:  {avg_faith:.2f}")
        print(f"   Questions Evaluated:   {len(results)}")
    
    return {"results": results}


def run_demo():
    """Run the complete demonstration."""
    print_header("SCIENTIFIC PAPER RAG SYSTEM - FULL DEMO")
    
    # Validate configuration
    config = get_config()
    issues = config.validate()
    
    if issues:
        print("⚠️  Configuration issues:")
        for issue in issues:
            print(f"   {issue}")
        
        if any("ERROR" in issue for issue in issues):
            print("\n❌ Cannot proceed without fixing errors.")
            print("   Please set OPENAI_API_KEY in your .env file")
            return
    
    print("✅ Configuration validated\n")
    
    # Task 1: Ingest papers
    try:
        ingest_papers(
            search_query="artificial intelligence medicine",
            max_per_source=2
        )
    except Exception as e:
        logger.error(f"Ingestion failed", error=str(e), exc_info=True)
        print(f"❌ Ingestion failed: {e}")
        return
    
    # Task 2: Simple query
    try:
        query_system("What is machine learning?")
    except Exception as e:
        logger.error(f"Simple query failed", error=str(e), exc_info=True)
        print(f"❌ Simple query failed: {e}")
    
    # Task 2: Complex query with decomposition
    try:
        query_system(
            "What are the differences between deep learning and traditional "
            "machine learning, and how are they each applied in medical "
            "image analysis?",
            force_decomposition=True
        )
    except Exception as e:
        logger.error(f"Complex query failed", error=str(e), exc_info=True)
        print(f"❌ Complex query failed: {e}")
    
    # Task 4: Evaluation
    try:
        evaluate_system(num_questions=2)
    except Exception as e:
        logger.error(f"Evaluation failed", error=str(e), exc_info=True)
        print(f"❌ Evaluation failed: {e}")
    
    print_header("DEMO COMPLETE")
    print("All tasks demonstrated successfully!")
    print("\nFor production hardening (Task 3), see the code comments and")
    print("the structured logging throughout the codebase.")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Scientific Paper RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --demo
    python main.py --ingest --search "COVID-19 vaccines"
    python main.py --query "What are mRNA vaccines?"
    python main.py --evaluate
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the full demonstration"
    )
    
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Ingest papers into vector database"
    )
    
    parser.add_argument(
        "--search",
        type=str,
        default="machine learning healthcare",
        help="Search query for paper ingestion"
    )
    
    parser.add_argument(
        "--max-papers",
        type=int,
        default=3,
        help="Maximum papers per source"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run against the system"
    )
    
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Force query decomposition"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on sample questions"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Default to demo if no arguments
    if not any([args.demo, args.ingest, args.query, args.evaluate]):
        args.demo = True
    
    try:
        if args.demo:
            run_demo()
        
        elif args.ingest:
            result = ingest_papers(args.search, args.max_papers)
            if args.json:
                print(json.dumps({"chunks_indexed": result}))
        
        elif args.query:
            result = query_system(args.query, args.decompose)
            if args.json:
                print(json.dumps(result, indent=2))
        
        elif args.evaluate:
            result = evaluate_system()
            if args.json:
                print(json.dumps(result, indent=2))
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unhandled exception", error=str(e), exc_info=True)
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
