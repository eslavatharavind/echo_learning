"""
Document Retriever for EchoLearn AI - This file handles searching for relevant info
Retrieves relevant document chunks from FAISS vector database - Like a librarian finding the right books
"""

import faiss  # Import FAISS for fast mathematical search
import numpy as np  # Import numpy for math operations
from typing import List, Dict, Optional  # Import types for organization
from sentence_transformers import SentenceTransformer  # Import AI tool to turn query into numbers
import logging  # Import logging for tracking activity

from config import Config  # Import project settings
from build_vector_db import VectorDBBuilder  # Import tool to manage the database

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for the retriever


class DocumentRetriever:  # Define a class specifically for finding document parts
    """Retrieve relevant documents from vector database"""
    
    def __init__(  # Initialize settings and load the database
        self,
        db_path: Optional[str] = None,
        embedding_model: Optional[str] = None,
        top_k: int = None
    ):
        """
        Initialize Document Retriever
        """
        self.db_path = db_path or Config.VECTOR_DB_PATH  # Use folder path from config
        self.embedding_model_name = embedding_model or Config.EMBEDDING_MODEL  # Use AI model from config
        self.top_k = top_k or Config.RETRIEVAL_TOP_K  # Set default number of results to find
        
        # Lazy load embedding model
        self.embedding_model = None
        
        # Load the saved vector database from disk
        self.db_builder = VectorDBBuilder(embedding_model=self.embedding_model_name)
        self.loaded = self.db_builder.load_index(self.db_path)  # Try to load the index files
        
        if self.loaded:  # If loading worked
            logger.info(f"Retriever initialized with {self.db_builder.index.ntotal} documents")
        else:  # If no database found
            logger.warning("No vector database found. Please build index first.")

    def _get_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model for retriever: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model
    
    def retrieve(  # Main function to search for answers
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Retrieve relevant documents for a query
        """
        if not self.loaded or self.db_builder.index is None:  # If database is not ready
            logger.error("Vector database not loaded")  # Log error
            return []  # Return nothing
        
        k = top_k or self.top_k  # Decide how many items to look for
        
        # Turn the user's text question into a list of numbers (embedding)
        model = self._get_model()
        query_embedding = model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')  # Convert to standard format
        
        # Use FAISS to mathematically find the most similar documents
        distances, indices = self.db_builder.index.search(query_embedding, k)
        
        # Process and format the search results
        results = []  # List for final results
        for idx, (distance, doc_idx) in enumerate(zip(distances[0], indices[0])):  # Loop through results
            # Skip if FAISS couldn't find a match or index is out of range
            if doc_idx == -1 or doc_idx >= len(self.db_builder.documents):
                continue
            
            # Apply a quality threshold (if result is too irrelevant, skip it)
            if score_threshold is not None and distance > score_threshold:
                continue
            
            # Package the result info
            result = {
                "text": self.db_builder.documents[doc_idx],  # The actual words found
                "metadata": self.db_builder.metadata[doc_idx],  # Extra info about the source
                "score": float(distance),  # How good the match is (lower is better)
                "rank": idx + 1  # 1st place, 2nd place, etc.
            }
            results.append(result)  # Add to results list
        
        logger.info(f"Retrieved {len(results)} documents for query: '{query[:50]}...'")  # Log success
        return results  # Return the list of matches
    
    def retrieve_with_context(  # Search and combine results into one big text block
        self,
        query: str,
        top_k: Optional[int] = None,
        include_surrounding: bool = True
    ) -> Dict:
        """
        Retrieve documents with additional context
        """
        results = self.retrieve(query, top_k)  # Get the raw matches first
        
        if not results:  # If nothing found
            return {
                "results": [],
                "context": "",
                "num_results": 0
            }
        
        # Combine all matches into one big "Context" text for the AI to read
        context_parts = []  # List for text chunks
        for i, result in enumerate(results, 1):  # Loop through matches
            source = result["metadata"].get("source", "Unknown")  # Get filename
            context_parts.append(f"[Source {i}: {source}]")  # Add source label
            context_parts.append(result["text"])  # Add actual text content
            context_parts.append("")  # Add an empty line for readability
        
        context = "\n".join(context_parts)  # Join everything into one string
        
        return {  # Return the final report
            "results": results,  # The original matches
            "context": context,  # The combined text block
            "num_results": len(results),  # total count
            "query": query  # The original search terms
        }
    
    def reload_index(self) -> bool:  # Refresh the DB (useful if new files were uploaded)
        """
        Reload the vector database from disk
        """
        logger.info("Reloading vector database...")
        self.loaded = self.db_builder.load_index(self.db_path)  # Reload from disk
        
        if self.loaded:  # If successful
            logger.info(f"Reloaded index with {self.db_builder.index.ntotal} documents")
        else:  # If failed
            logger.warning("Failed to reload index")
        
        return self.loaded  # Return status
    
    def is_ready(self) -> bool:  # Check if everything is working
        """Check if retriever is ready to use"""
        return self.loaded and self.db_builder.index is not None
    
    def get_stats(self) -> Dict:  # Get summary report of the retriever
        """Get retriever statistics"""
        if not self.is_ready():  # If not working
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "num_documents": self.db_builder.index.ntotal,  # Total documents searchable
            "embedding_model": self.embedding_model_name,  # Current AI tool used
            "top_k": self.top_k,  # Default search count
            "db_path": str(self.db_path)  # Folder location
        }


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    retriever = DocumentRetriever()  # Init retriever
    
    if retriever.is_ready():  # Check status
        # Try a test search
        results = retriever.retrieve("What is machine learning?", top_k=3)
        
        print(f"Found {len(results)} results:")
        for result in results:  # Loop and print matches
            print(f"\nScore: {result['score']:.4f}")
            print(f"Text: {result['text'][:100]}...")
    else:
        print("Retriever not ready. Build vector database first.")
