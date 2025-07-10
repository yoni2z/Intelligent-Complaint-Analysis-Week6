import pandas as pd
from rag_pipeline import Retriever, Generator, RAGPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the RAG pipeline with a sample question."""
    try:
        # Initialize components
        retriever = Retriever('vector_store/sampled_faiss_index.idx', 'vector_store/sampled_chunk_metadata.pkl')
        generator = Generator()
        rag = RAGPipeline(retriever, generator)

        # Test question
        question = "Why are people unhappy with BNPL?"
        answer, contexts = rag.answer(question)

        # Print results
        print("Question:", question)
        print("Answer:", answer)
        print("\nRetrieved Contexts:")
        for ctx in contexts:
            print(f"[{ctx['product']} - {ctx['complaint_id']}]: {ctx['text']}")
    except Exception as e:
        logger.error(f"Error testing pipeline: {e}")
        raise

def evaluate_pipeline():
    """Evaluate the RAG pipeline with a set of questions."""
    try:
        # Initialize pipeline
        retriever = Retriever('vector_store/sampled_faiss_index.idx', 'vector_store/sampled_chunk_metadata.pkl')
        generator = Generator()
        rag = RAGPipeline(retriever, generator)

        # Test questions
        questions = [
            "What are the common issues with Credit Cards?",
            "Why are customers complaining about Money Transfers?",
            "Are there any fraud-related complaints for Personal Loans?",
            "What problems do users face with BNPL?",
            "How satisfied are customers with Savings Accounts?"
        ]

        # Evaluate
        results = []
        for question in questions:
            answer, contexts = rag.answer(question)
            # Manual quality score (1-5) and comments
            quality_score = 3  # Placeholder (replace with manual review)
            comments = "TBD after manual review"  # Placeholder
            results.append({
                'Question': question,
                'Generated Answer': answer,
                'Retrieved Sources': "\n".join([f"[{c['product']} - {c['complaint_id']}]: {c['text']}" for c in contexts[:2]]),
                'Quality Score': quality_score,
                'Comments': comments
            })

        # Save to DataFrame
        eval_df = pd.DataFrame(results)
        eval_df.to_markdown('reports/evaluation_table.md')
        logger.info("Evaluation table saved to 'reports/evaluation_table.md'")
        return eval_df
    except Exception as e:
        logger.error(f"Error evaluating pipeline: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting pipeline test")
    test_pipeline()
    logger.info("Starting pipeline evaluation")
    evaluate_pipeline()