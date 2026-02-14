"""
Text Chunker for EchoLearn AI - This file splits documents into manageable pieces
Splits text into semantic chunks with overlap for RAG - Helps the AI find exact answers
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter  # Import tool to split text intelligently
from typing import List, Dict  # Import types for organization
import logging  # Import logging to track processing

from config import Config  # Import project settings

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for the chunker


class TextChunker:  # Define the class for breaking text into pieces
    """Chunk text into smaller pieces with overlap"""
    
    def __init__(  # Initialize the chunker settings
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Initialize Text Chunker
        """
        self.chunk_size = chunk_size or Config.CHUNK_SIZE  # Set how many characters per piece
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP  # Set how many characters overlap between pieces
        
        # Default separators prioritize semantic boundaries (natural breaks)
        self.separators = separators or [  # List of where it's okay to cut text
            "\n\n",  # Best: cut at double newlines (paragraphs)
            "\n",    # Good: cut at line breaks
            ". ",    # Okay: cut at end of sentences
            "? ",    # Cut at questions
            "! ",    # Cut at exclamations
            "; ",    # Cut at semicolons
            ", ",    # Cut at commas
            " ",     # Cut at words (last resort for size)
            ""       # Cut at characters (absolute fallback)
        ]
        
        # Create LangChain splitter (the actual engine that calculates cuts)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,  # Maximum length of one piece
            chunk_overlap=self.chunk_overlap,  # How much to repeat from previous piece
            separators=self.separators,  # Natural break points to prefer
            length_function=len,  # Use standard character count
        )
        
        logger.info(f"TextChunker initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")  # Log setup
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:  # Main function to split text
        """
        Split text into chunks
        """
        if not text or not text.strip():  # If text is empty
            logger.warning("Empty text provided to chunker")  # Log warning
            return []  # Return nothing
        
        # Split text using LangChain
        chunks = self.splitter.split_text(text)  # Run the splitting logic
        
        logger.info(f"Split text ({len(text)} chars) into {len(chunks)} chunks")  # Log results
        
        # Create chunk dictionaries with metadata (tags)
        result = []  # List for final result
        for idx, chunk_text in enumerate(chunks):  # Loop through every new piece
            chunk_dict = {  # Build info package for this piece
                "text": chunk_text,  # The actual words
                "chunk_index": idx,  # The order number (0, 1, 2...)
                "chunk_size": len(chunk_text),  # Length of this piece
                "metadata": metadata or {}  # Extra info like source filename
            }
            result.append(chunk_dict)  # Add to list
        
        return result  # Return all prepared pieces
    
    def chunk_with_context(self, text: str, metadata: Dict = None) -> List[Dict]:  # Split with extra context
        """
        Split text into chunks with additional context information
        """
        chunks = self.chunk(text, metadata)  # Do standard splitting first
        
        # Add context from surrounding chunks (helps AI understand continuity)
        for idx, chunk in enumerate(chunks):  # Loop through pieces
            # Previous chunk context (what came just before)
            if idx > 0:  # If not the very first piece
                prev_text = chunks[idx - 1]["text"]  # Look at the piece before this one
                # Take last 20 words from previous chunk
                prev_words = prev_text.split()[-20:]
                chunk["prev_context"] = " ".join(prev_words)  # Store it as "previous context"
            
            # Next chunk context (what comes just after)
            if idx < len(chunks) - 1:  # If not the very last piece
                next_text = chunks[idx + 1]["text"]  # Look at the piece after this one
                # Take first 20 words from next chunk
                next_words = next_text.split()[:20]
                chunk["next_context"] = " ".join(next_words)  # Store it as "next context"
        
        return chunks  # Return pieces with their neighbors' context attached
    
    def chunk_by_section(self, text: str, section_markers: List[str] = None) -> List[Dict]:  # Split by titles
        """
        Chunk text while preserving section boundaries
        """
        section_markers = section_markers or ["# ", "## ", "### ", "---"]  # Markers for headers
        
        # Split by sections first (looking for headers like # Introduction)
        sections = []  # List for sections
        current_section = []  # Temporary list for current section lines
        current_header = None  # Header name for current section
        
        for line in text.split('\n'):  # Look line by line
            # Check if line is a section header (starts with #)
            is_header = any(line.strip().startswith(marker) for marker in section_markers)
            
            if is_header:  # If we hit a new header
                # Save the section we just finished
                if current_section:
                    sections.append({
                        "header": current_header,
                        "text": '\n'.join(current_section)
                    })
                
                # Start fresh with the new header
                current_header = line.strip()
                current_section = [line]
            else:  # If it's just a normal line
                current_section.append(line)  # Add to current section
        
        # Add the very last section remaining
        if current_section:
            sections.append({
                "header": current_header,
                "text": '\n'.join(current_section)
            })
        
        # Chunk each section if it's too large to fit in one piece
        result = []  # Final list
        for section in sections:  # Look at every group we found
            section_text = section["text"]  # The words in that group
            
            if len(section_text) <= self.chunk_size:  # If it's small enough
                # Section fits in one piece
                result.append({
                    "text": section_text,
                    "header": section["header"],
                    "metadata": {"section_header": section["header"]}
                })
            else:  # If it's too big
                # Break this section into smaller pieces using standard chunker
                section_chunks = self.chunk(section_text, {"section_header": section["header"]})
                for chunk in section_chunks:  # Add the header name to every piece of this section
                    chunk["header"] = section["header"]
                result.extend(section_chunks)  # Add to final list
        
        logger.info(f"Split text into {len(result)} section-aware chunks")  # Log success
        return result  # Return pieces that know what chapter they belong to
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:  # Function to see how well we split
        """
        Get statistics about chunks
        """
        if not chunks:  # If no chunks
            return {"num_chunks": 0}  # Return zero
        
        chunk_sizes = [chunk["chunk_size"] for chunk in chunks]  # Make a list of all piece lengths
        
        return {  # Return summary data
            "num_chunks": len(chunks),  # Total pieces
            "total_chars": sum(chunk_sizes),  # Total characters
            "avg_chunk_size": sum(chunk_sizes) / len(chunks),  # Average pieces size
            "min_chunk_size": min(chunk_sizes),  # Smallest piece
            "max_chunk_size": max(chunk_sizes),  # Largest piece
        }


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)  # Init chunker
    
    # Create a long sample text to test splitting
    sample_text = """
    # Introduction
    This is the introduction section with some text.
    
    # Main Content
    This is the main content with much more text.
    It has multiple paragraphs.
    
    Each paragraph discusses different topics.
    """ * 10  # Repeat text 10 times
    
    chunks = chunker.chunk(sample_text)  # Run chunking
    stats = chunker.get_chunk_stats(chunks)  # Get stats
    
    print(f"Created {stats['num_chunks']} chunks")  # Print total pieces
    print(f"Average chunk size: {stats['avg_chunk_size']:.0f} characters")  # Print avg size
