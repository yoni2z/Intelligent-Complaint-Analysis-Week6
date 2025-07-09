import pandas as pd
import numpy as np
import faiss
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_filtered_dataset(file_path, sample_size=5000):
    """Load and sample the filtered complaints dataset."""
    try:
        # Load full dataset
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} complaints from {file_path}")
        
        # Perform stratified sampling based on Product
        if len(df) > sample_size:
            df = df.groupby('Product', group_keys=False).apply(
                lambda x: x.sample(frac=sample_size/len(df), random_state=42)
            )
            logger.info(f"Sampled {len(df)} complaints, maintaining product distribution")
        else:
            logger.warning(f"Dataset size ({len(df)}) is smaller than sample_size ({sample_size}). Using full dataset.")
        
        # Log product distribution
        logger.info("Sampled product distribution:\n" + str(df['Product'].value_counts()))
        return df
    except FileNotFoundError:
        logger.error(f"Dataset file {file_path} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

def chunk_text(text, complaint_id, product, chunk_size=500, chunk_overlap=50):
    """Chunk text into smaller segments with metadata."""
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return [{"text": chunk, "complaint_id": complaint_id, "product": product} for chunk in chunks]
    except Exception as e:
        logger.error(f"Error chunking text for complaint {complaint_id}: {e}")
        return []

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """Generate embeddings for a list of texts."""
    try:
        embedder = SentenceTransformer(model_name)
        # Batch processing to reduce memory usage
        batch_size = 128
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = embedder.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        embeddings = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise

def create_faiss_index(embeddings, index_path, metadata, metadata_path):
    """Create and save a FAISS index with metadata."""
    try:
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))
        
        # Ensure vector_store directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        # Save index
        faiss.write_index(index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved chunk metadata to {metadata_path}")
    except Exception as e:
        logger.error(f"Error creating FAISS index: {e}")
        raise

def main():
    """Main function to perform chunking, embedding, and indexing on sampled data."""
    # File paths
    input_file = 'data/filtered_complaints.csv'
    index_path = 'vector_store/sampled_faiss_index.idx'
    metadata_path = 'vector_store/sampled_chunk_metadata.pkl'
    
    # Step 1: Load and sample filtered dataset
    df = load_filtered_dataset(input_file, sample_size=5000)
    
    # Step 2: Chunk texts
    chunks = []
    for _, row in df.iterrows():
        chunks.extend(chunk_text(
            text=row['cleaned_narrative'],
            complaint_id=row['Complaint ID'],
            product=row['Product']
        ))
    logger.info(f"Created {len(chunks)} chunks")
    
    # Step 3: Generate embeddings
    texts = [chunk['text'] for chunk in chunks]
    embeddings = generate_embeddings(texts)
    
    # Step 4: Create and save FAISS index
    create_faiss_index(embeddings, index_path, chunks, metadata_path)

if __name__ == "__main__":
    main()