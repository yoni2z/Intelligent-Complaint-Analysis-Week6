import gradio as gr
from rag_pipeline import Retriever, Generator, RAGPipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_pipeline():
    """Initialize the RAG pipeline with sampled vector store."""
    try:
        retriever = Retriever('vector_store/sampled_faiss_index.idx', 'vector_store/sampled_chunk_metadata.pkl')
        generator = Generator()
        pipeline = RAGPipeline(retriever, generator)
        logger.info("RAG pipeline initialized successfully")
        return pipeline
    except Exception as e:
        logger.error(f"Error initializing RAG pipeline: {e}")
        raise

def answer_question(question, pipeline):
    """Process a user question and return the answer with sources."""
    try:
        if not question.strip():
            return "Please enter a valid question."
        
        # Get answer and sources
        answer, contexts = pipeline.answer(question)
        
        # Format sources
        sources_text = "\n\n**Sources**:\n"
        for idx, ctx in enumerate(contexts, 1):
            sources_text += f"{idx}. [{ctx['product']} - {ctx['complaint_id']}]: {ctx['text']}\n"
        
        logger.info(f"Processed question: {question}")
        return answer + sources_text
    except Exception as e:
        logger.error(f"Error processing question '{question}': {e}")
        return f"Error: Unable to process the question. {str(e)}"

def main():
    """Create and launch the Gradio interface."""
    try:
        # Initialize pipeline
        pipeline = initialize_pipeline()
        
        # Create Gradio interface
        iface = gr.Interface(
            fn=lambda question: answer_question(question, pipeline),
            inputs=gr.Textbox(
                lines=2,
                placeholder="Type your question here (e.g., 'Why are people unhappy with BNPL?')"
            ),
            outputs="markdown",
            title="CrediTrust Complaint Analyzer",
            description="Ask questions about customer complaints to get insights.",
            submit_btn="Ask",
            clear_btn="Clear"
        )
        
        # Launch interface (remove share=True for local testing)
        iface.launch(share=False)  # Set to True only for public sharing
    except Exception as e:
        logger.error(f"Error launching Gradio interface: {e}")
        raise

if __name__ == "__main__":
    main()