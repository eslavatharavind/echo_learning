"""
Text Cleaner for EchoLearn AI - This file handles making text neat and tidy
Cleans and normalizes extracted text from documents - To help the AI read better
"""

import re  # Import re for finding and replacing patterns in text (Regular Expressions)
from typing import Optional  # Import Optional for variables that might be empty
import logging  # Import logging for tracking activity

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for this file


class TextCleaner:  # Define a class specifically for making text cleaner
    """Clean and normalize extracted text"""
    
    def __init__(  # Initialize settings for the cleaner
        self,
        remove_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        preserve_code_formatting: bool = True
    ):
        """
        Initialize Text Cleaner
        """
        self.remove_headers_footers = remove_headers_footers  # Should we remove page headers?
        self.normalize_whitespace = normalize_whitespace  # Should we fix extra spaces?
        self.preserve_code_formatting = preserve_code_formatting  # Should we leave Python code alone?
    
    def clean(self, text: str, is_code: bool = False) -> str:  # Main function to clean text
        """
        Clean and normalize text
        """
        if not text or not text.strip():  # If text is empty or just spaces
            return ""  # Return nothing
        
        logger.debug(f"Cleaning text ({len(text)} chars)")  # Log that we are starting
        
        # Preserve code blocks
        code_blocks = []  # List to store code that shouldn't be touched
        if self.preserve_code_formatting and not is_code:  # If it's a mix of text and code
            text, code_blocks = self._preserve_code_blocks(text)  # Temporarily hide the code
        
        # Apply cleaning steps
        text = self._normalize_unicode(text)  # Fix weird symbols and quotes
        
        if self.normalize_whitespace:  # If space cleaning is ON
            text = self._normalize_whitespace_func(text)  # Fix spaces and new lines
        
        if self.remove_headers_footers:  # If header removal is ON
            text = self._remove_headers_footers_func(text)  # Remove things that look like headers
        
        text = self._remove_special_characters(text)  # Remove invisible or junk characters
        text = self._fix_spacing(text)  # Fix spaces around dots and commas
        
        # Restore code blocks
        if code_blocks:  # If we hid some code blocks earlier
            text = self._restore_code_blocks(text, code_blocks)  # Put them back in their spots
        
        logger.debug(f"Cleaned text ({len(text)} chars)")  # Log that we finished
        return text.strip()  # Return the neat text with edges trimmed
    
    def _preserve_code_blocks(self, text: str) -> tuple[str, list]:  # Helper to hide code blocks
        """
        Extract and preserve code blocks
        """
        code_blocks = []  # List for the actual code content
        
        # Match code blocks (starting and ending with ```)
        pattern = r'```[\s\S]*?```'  # Pattern to find everything inside triple backticks
        
        def replacer(match):  # Function called for every code block found
            code_blocks.append(match.group(0))  # Save the code block to our list
            return f"<<<CODE_BLOCK_{len(code_blocks)-1}>>>"  # Replace code with a marker tag
        
        text = re.sub(pattern, replacer, text)  # Swap code for tags in the main text
        return text, code_blocks  # Return the tagged text and the list of code
    
    def _restore_code_blocks(self, text: str, code_blocks: list) -> str:  # Helper to put code back
        """Restore preserved code blocks"""
        for i, block in enumerate(code_blocks):  # Loop through each saved code block
            placeholder = f"<<<CODE_BLOCK_{i}>>>"  # Match its marker tag
            text = text.replace(placeholder, block)  # Swap tag back with the real code
        return text  # Return text with code restored
    
    def _normalize_unicode(self, text: str) -> str:  # Helper to fix weird symbols
        """Normalize unicode characters"""
        # Replace common fancy quotes with standard ones
        text = text.replace('"', '"').replace('"', '"')  # Fix fancy double quotes
        text = text.replace(''', "'").replace(''', "'")  # Fix fancy single quotes
        
        # Replace fancy dashes with simple ones
        text = text.replace('—', '-').replace('–', '-')  # Fix long dashes
        
        # Replace invisible or special spaces with standard ones
        text = text.replace('\xa0', ' ')  # Fix non-breaking space
        text = text.replace('\u200b', '')  # Remove zero-width (invisible) space
        
        return text  # Return text with standard symbols
    
    def _normalize_whitespace_func(self, text: str) -> str:  # Helper to fix spacing
        """Normalize whitespace"""
        # Replace multiple spaces with a single one
        text = re.sub(r' +', ' ', text)  # Matches 2 or more spaces
        
        # Replace 3 or more empty lines with just 2
        text = re.sub(r'\n{3,}', '\n\n', text)  # Keep it tidy but separate sections
        
        # Remove empty spaces at the end of every line
        lines = [line.rstrip() for line in text.split('\n')]  # Split into lines and trim right side
        text = '\n'.join(lines)  # Join lines back together
        
        return text  # Return text with clean spaces
    
    def _remove_headers_footers_func(self, text: str) -> str:  # Helper to remove page noise
        """
        Attempt to remove repeated headers/footers
        """
        lines = text.split('\n')  # Break text into individual lines
        
        if len(lines) < 10:  # If document is very short
            return text  # Don't bother cleaning headers
        
        # Find lines that appear many times (likely headers)
        line_counts = {}  # Dictionary to count appearances
        for line in lines:  # Look at every line
            stripped = line.strip()  # Clean spaces
            if len(stripped) > 5 and len(stripped) < 100:  # If line is medium length
                line_counts[stripped] = line_counts.get(stripped, 0) + 1  # Add to count
        
        # Find lines that appear more than 3 times
        repeated_lines = {line for line, count in line_counts.items() if count > 3}
        
        if repeated_lines:  # If we found any repeated lines
            logger.debug(f"Removing {len(repeated_lines)} repeated header/footer lines")  # Log it
            lines = [line for line in lines if line.strip() not in repeated_lines]  # Filter them out
        
        return '\n'.join(lines)  # Join lines back together
    
    def _remove_special_characters(self, text: str) -> str:  # Helper to remove junk
        """Remove problematic special characters"""
        # Remove weird "control" characters that can confuse software
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        return text  # Return text with no junk
    
    def _fix_spacing(self, text: str) -> str:  # Helper to fix punctuation spacing
        """Fix spacing issues"""
        # Remove space before a dot or comma ("hello ." becomes "hello.")
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        
        # Add space after a dot or comma ("hello.World" becomes "hello. World")
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        
        # Ensure there is exactly one space before a bracket
        text = re.sub(r'\s+([(\[])', r' \1', text)
        
        return text  # Return text with perfect punctuation spacing
    
    def clean_for_embedding(self, text: str) -> str:  # Specialized cleaning for AI search
        """
        Clean text specifically for embedding generation
        """
        text = self.clean(text)  # Do standard cleaning first
        
        # Remove redundant page markers (like "--- Page 1 ---")
        text = re.sub(r'---\s*Page\s+\d+\s*---', '', text)
        
        # Change triple exclamation points into one ("Wow!!!" becomes "Wow!")
        text = re.sub(r'[!?]{2,}', '!', text)
        
        # Change different shaped bullet points into simple dashes
        text = re.sub(r'[•●○■□▪▫]', '-', text)
        
        return text.strip()  # Return the ultra-clean text


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    cleaner = TextCleaner()  # Create cleaner
    
    # Try cleaning a messy sample text
    sample = """
    This is    some   text with     extra    spaces.
    
    
    
    And too many newlines.
    
    "Unicode quotes" and — dashes.
    
    ```python
    # This code should be preserved
    def   example():
        pass
    ```
    """
    
    cleaned = cleaner.clean(sample)  # Run the cleaning
    print("Cleaned text:")  # Print header
    print(cleaned)  # Print the neat result
