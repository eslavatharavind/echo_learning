"""
Vector Database Builder for EchoLearn AI - This file manages the searchable text storage
Builds and manages FAISS vector index for document embeddings - The "brain's memory" for documents
"""

import faiss  # Import FAISS (Facebook AI Similarity Search) - the engine for fast text searching
import numpy as np  # Import numpy for handling large lists of numbers (vectors)
import pickle  # Import pickle for saving and loading Python objects to disk
from pathlib import Path  # Import Path for managing file locations
from typing import List, Dict, Optional  # Import types for organization
from sentence_transformers import SentenceTransformer  # Import tool to turn text into numbers (embeddings)
import logging  # Import logging for tracking progress

from config import Config  # Import project settings

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for the database builder


class VectorDBBuilder:  # Define a class for building and managing the document database
    """Build and manage FAISS vector database"""
    
    def __init__(self, embedding_model: str = None, embedding_dim: int = None):  # Initialize settings
        """
        Initialize Vector DB Builder
        """
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL  # Set the model name
        self.embedding_dim = embedding_dim or Config.EMBEDDING_DIMENSION  # Set the expected vector size
        self.db_path = Config.VECTOR_DB_PATH  # Set where the database will be saved
        
        # Lazy load embedding model (don't load it yet to save memory at startup)
        self.embedding_model = None
        
        # Initialize FAISS index placeholders
        self.index = None  # This will hold the actual searchable index
        self.documents = []  # List to store the original text chunks
        self.metadata = []   # List to store info about each chunk (like filename)
        
        logger.info(f"VectorDBBuilder initialized (Lazy loading model: {self.embedding_model_name})")  # Log finish

    def _get_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Update dimension based on actual model
            if self.embedding_dim is None:
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
                
        return self.embedding_model
    
    def build_index(self, chunks: List[Dict], rebuild: bool = False) -> int:  # Main function to build DB
        """
        Build FAISS index from text chunks
        """
        if not chunks:  # If no chunks were provided
            logger.warning("No chunks provided to build index")  # Log warning
            return 0  # Return zero
        
        # Extract texts and their tags from the chunk objects
        texts = [chunk["text"] for chunk in chunks]
        chunk_metadata = [chunk.get("metadata", {}) for chunk in chunks]
        
        logger.info(f"Building embeddings for {len(texts)} chunks...")  # Log progress
        
        # Generate embeddings (turn all text into lists of numbers)
        # Generate embeddings (turn all text into lists of numbers)
        model = self._get_model()
        embeddings = model.encode(
            texts,
            show_progress_bar=True,  # Show a loading bar in the terminal
            convert_to_numpy=True  # Ensure result is in a math-friendly format
        )
        
        # Create or update FAISS index logic
        if rebuild or self.index is None:  # If starting fresh or first time
            logger.info("Creating new FAISS index")  # Log action
            self.index = faiss.IndexFlatL2(self.embedding_dim)  # Create a basic "straight search" index
            self.documents = []  # Clear text list
            self.metadata = []  # Clear metadata list
        
        # Add the new number-lists (embeddings) to the search engine
        self.index.add(embeddings.astype('float32'))  # FAISS likes float32 numbers
        
        # Store the original text and its info so we can show it later
        self.documents.extend(texts)
        self.metadata.extend(chunk_metadata)
        
        logger.info(f"Index now contains {self.index.ntotal} documents")  # Log total count
        return self.index.ntotal  # Return total number of items indexed
    
    def add_documents(self, chunks: List[Dict]) -> int:  # Helper to just add to existing DB
        """
        Add documents to existing index
        """
        return self.build_index(chunks, rebuild=False)  # Calls build_index but keeps old data
    
    def save_index(self, path: Optional[str] = None) -> str:  # Save the DB to a file
        """
        Save FAISS index and metadata to disk
        """
        if self.index is None:  # If index doesn't exist yet
            raise ValueError("No index to save. Build index first.")  # Stop and error
        
        save_path = Path(path) if path else self.db_path  # Decide where to save
        save_path.mkdir(parents=True, exist_ok=True)  # Create folders if missing
        
        # Save the FAISS search engine part
        index_file = save_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_file))  # Save binary index file
        logger.info(f"Saved FAISS index to {index_file}")  # Log success
        
        # Save the actual text and metadata part using Pickle
        data_file = save_path / "documents.pkl"
        with open(data_file, 'wb') as f:  # Open binary file for writing
            pickle.dump({  # Save everything in a single dictionary
                'documents': self.documents,
                'metadata': self.metadata,
                'embedding_model': self.embedding_model_name,
                'embedding_dim': self.embedding_dim
            }, f)
        logger.info(f"Saved documents and metadata to {data_file}")  # Log success
        
        return str(save_path)  # Return the folder path
    
    def load_index(self, path: Optional[str] = None) -> bool:  # Load a saved DB from file
        """
        Load FAISS index and metadata from disk
        """
        load_path = Path(path) if path else self.db_path  # Decide where to load from
        
        index_file = load_path / "faiss_index.bin"  # Expected index filename
        data_file = load_path / "documents.pkl"  # Expected data filename
        
        if not index_file.exists() or not data_file.exists():  # If files are missing
            logger.warning(f"Index files not found at {load_path}")  # Log a warning
            return False  # Return failure
        
        try:  # Try loading folders
            # Load the FAISS search part
            self.index = faiss.read_index(str(index_file))  # Read the binary index
            logger.info(f"Loaded FAISS index with {self.index.ntotal} documents")  # Log success
            
            # Load the text and metadata part
            with open(data_file, 'rb') as f:  # Open binary file for reading
                data = pickle.load(f)  # Load the dictionary
                self.documents = data['documents']  # Restore text chunks
                self.metadata = data['metadata']  # Restore metadata
                
                # Double check that the embedding model is still the same as when we saved
                if data['embedding_model'] != self.embedding_model_name:
                    logger.warning(
                        f"Loaded index uses different embedding model: "
                        f"{data['embedding_model']} vs {self.embedding_model_name}"
                    )
            
            return True  # Return success
            
        except Exception as e:  # If reading fails
            logger.error(f"Failed to load index: {e}")  # Log error
            return False  # Return failure
    
    def get_stats(self) -> Dict:  # Function to see DB information
        """
        Get statistics about the vector database
        """
        if self.index is None:  # If DB is empty
            return {"status": "not_initialized"}  # Return empty status
        
        return {  # Return info report
            "status": "ready",  # Ready to search
            "num_documents": self.index.ntotal,  # Total pieces of text stored
            "embedding_model": self.embedding_model_name,  # Which AI model made them
            "embedding_dimension": self.embedding_dim,  # How big the math vectors are
            "index_size_mb": self.index.ntotal * self.embedding_dim * 4 / 1024 / 1024,  # Estimated memory usage
        }
    
    def clear_index(self):  # Function to wipe the DB memory
        """Clear the current index"""
        self.index = None  # Delete index
        self.documents = []  # Delete texts
        self.metadata = []  # Delete metadata
        logger.info("Index cleared")  # Log action


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    builder = VectorDBBuilder()  # Init builder
    
    # Create some fake test data
    sample_chunks = [
        {"text": "This is the first chunk about machine learning.", "metadata": {"source": "doc1"}},
        {"text": "This is the second chunk about Python programming.", "metadata": {"source": "doc1"}},
        {"text": "This chunk discusses neural networks.", "metadata": {"source": "doc2"}},
    ]
    
    # Build index with test data
    builder.build_index(sample_chunks)
    
    # Get and print stats
    stats = builder.get_stats()
    print("Vector DB Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
