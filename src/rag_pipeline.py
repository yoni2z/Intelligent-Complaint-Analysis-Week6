import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define prompt template
PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

class Retriever:
    def __init__(self, index_path, metadata_path, model_name='all-MiniLM-L6-v2'):
        """Initialize retriever with FAISS index and embedding model."""
        try:
            self.embedder = SentenceTransformer(model_name)
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index from {index_path} and metadata from {metadata_path}")
        except Exception as e:
            logger.error(f"Error initializing Retriever: {e}")
            raise

    def retrieve(self, query, k=5):
        """Retrieve top-k relevant chunks for a query."""
        try:
            # Embed query
            query_embedding = self.embedder.encode([query])[0]
            # Search index
            distances, indices = self.index.search(np.array([query_embedding]).astype(np.float32), k)
            # Fetch results
            results = [
                {
                    'text': self.metadata[i]['text'],
                    'complaint_id': self.metadata[i]['complaint_id'],
                    'product': self.metadata[i]['product'],
                    'distance': distances[0][j]
                }
                for j, i in enumerate(indices[0])
            ]
            logger.info(f"Retrieved {len(results)} chunks for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

class Generator:
    def __init__(self, model_name='distilgpt2'):
        """Initialize generator with a lightweight LLM."""
        try:
            self.llm = pipeline(
                'text-generation',
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                max_new_tokens=200
            )
            logger.info(f"Initialized LLM: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing Generator: {e}")
            raise

    def generate(self, question, contexts):
        """Generate an answer using retrieved contexts."""
        try:
            # Combine contexts
            context_text = "\n".join([f"[{c['product']} - {c['complaint_id']}]: {c['text']}" for c in contexts])
            # Format prompt
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
            # Generate response
            response = self.llm(prompt)[0]['generated_text']
            # Extract answer
            answer = response.split("Answer:")[-1].strip()
            logger.info(f"Generated answer for question: {question}")
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

class RAGPipeline:
    def __init__(self, retriever, generator):
        """Initialize RAG pipeline with retriever and generator."""
        self.retriever = retriever
        self.generator = generator
        logger.info("Initialized RAGPipeline")

    def answer(self, question):
        """Answer a question using the RAG pipeline."""
        try:
            # Retrieve relevant chunks
            contexts = self.retriever.retrieve(question)
            # Generate answer
            answer = self.generator.generate(question, contexts)
            return answer, contexts
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            raise