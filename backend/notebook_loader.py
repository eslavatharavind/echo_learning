"""
Jupyter Notebook Loader for EchoLearn AI - This file handles reading .ipynb files
Extracts markdown cells, code, and comments from .ipynb files - To teach AI from code notebooks
"""

import nbformat  # Import nbformat (the standard tool for opening Jupyter Notebook files)
from pathlib import Path  # Import Path for managing file locations
from typing import Optional, List, Dict  # Import typing for better code organization
import logging  # Import logging for tracking progress
import json  # Import json for handling data format

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for this notebook loader


class NotebookLoader:  # Define the class for reading notebooks
    """Load and extract content from Jupyter Notebook files"""
    
    def __init__(self, include_code: bool = True, include_outputs: bool = False):  # Initialize settings
        """
        Initialize Notebook loader
        """
        self.include_code = include_code  # Save setting: Should we read the Python code cells?
        self.include_outputs = include_outputs  # Save setting: Should we read the code execution results?
    
    def load(self, notebook_path: str) -> str:  # Main function to get text from a notebook
        """
        Load Jupyter Notebook and extract text content
        """
        notebook_path = Path(notebook_path)  # Convert input string to a real Path object
        
        if not notebook_path.exists():  # Check if file actually exists
            raise FileNotFoundError(f"Notebook file not found: {notebook_path}")  # If not, stop
        
        if not notebook_path.suffix.lower() == '.ipynb':  # Check if the file ends in .ipynb
            raise ValueError(f"File is not a Jupyter Notebook: {notebook_path}")  # If not, stop
        
        try:  # Try reading the file content
            logger.info(f"Loading notebook: {notebook_path.name}")  # Log the filename
            
            # Read notebook
            with open(notebook_path, 'r', encoding='utf-8') as f:  # Open the file for reading
                nb = nbformat.read(f, as_version=4)  # Read it into the nbformat object
            
            # Extract content from cells
            text_parts = []  # List to store text chunks from each notebook cell
            text_parts.append(f"# Jupyter Notebook: {notebook_path.name}\n")  # Add file title
            
            for idx, cell in enumerate(nb.cells, start=1):  # Loop through every cell in the notebook
                cell_text = self._process_cell(cell, idx)  # Ask a helper function to read the cell
                if cell_text:  # If cell had any useful text
                    text_parts.append(cell_text)  # Add it to our list
            
            result = "\n\n".join(text_parts)  # Combine all cell texts with extra space in between
            
            logger.info(f"Successfully extracted {len(result)} characters from {notebook_path.name}")  # Log success
            return result  # Return the full notebook text
            
        except Exception as e:  # If something goes wrong while reading
            logger.error(f"Failed to load notebook: {e}")  # Log the error
            raise  # Stop and show the error message
    
    def _process_cell(self, cell, idx: int) -> Optional[str]:  # Helper function to categorize cells
        """
        Process a single notebook cell
        """
        cell_type = cell.cell_type  # Get the type of the cell (markdown or code)
        
        if cell_type == "markdown":  # If it's a documentation cell
            return self._process_markdown_cell(cell, idx)  # Use markdown reader
        elif cell_type == "code" and self.include_code:  # If it's code and we want code
            return self._process_code_cell(cell, idx)  # Use code reader
        
        return None  # Return nothing for other cell types
    
    def _process_markdown_cell(self, cell, idx: int) -> str:  # Reader for text/documentation cells
        """Extract text from markdown cell"""
        content = cell.source.strip()  # Get raw text and remove extra spaces
        if content:  # If cell is not empty
            return f"## Markdown Cell {idx}\n\n{content}"  # Return it with a header
        return ""  # Return empty if cell was empty
    
    def _process_code_cell(self, cell, idx: int) -> str:  # Reader for Python code cells
        """Extract code and comments from code cell"""
        parts = []  # List for code-related text
        
        # Add code with explanation marker
        code = cell.source.strip()  # Get the raw code
        if code:  # If there is code
            # Extract comments as explanations
            comments = self._extract_comments(code)  # Look for # comments inside the code
            
            parts.append(f"## Code Cell {idx}")  # Add header
            
            if comments:  # If we found any comments
                parts.append("\n**Explanations from comments:**")  # Add a prompt for the AI
                for comment in comments:  # Loop through comments
                    parts.append(f"- {comment}")  # Add each comment as a bullet point
            
            parts.append("\n**Code:**")  # Label for the code block
            parts.append(f"```python\n{code}\n```")  # Add the code itself
        
        # Include outputs if requested
        if self.include_outputs and hasattr(cell, 'outputs') and cell.outputs:  # If we want results
            output_text = self._extract_outputs(cell.outputs)  # Read the execution results
            if output_text:  # If there were any results (like printouts)
                parts.append("\n**Output:**")  # Label
                parts.append(output_text)  # The result text
        
        return "\n".join(parts) if parts else ""  # Join everything together
    
    def _extract_comments(self, code: str) -> List[str]:  # Function to find helpful notes in code
        """
        Extract comments from Python code
        """
        comments = []  # List for found comments
        for line in code.split('\n'):  # Look at it line by line
            line = line.strip()  # Clean spaces
            # Single-line comments
            if line.startswith('#'):  # If line starts with a comment symbol
                comment = line.lstrip('#').strip()  # Remove the # and extra space
                if comment:  # If some text remained
                    comments.append(comment)  # Keep it
        
        # Extract docstrings (long comments inside triple quotes)
        if '"""' in code or "'''" in code:  # If triple quotes are present
            # Simple docstring extraction
            for delimiter in ['"""', "'''"]:  # Check both types of triple quotes
                if code.count(delimiter) >= 2:  # If there is a start and an end
                    parts = code.split(delimiter)  # Split the code by those quotes
                    for i in range(1, len(parts), 2):  # Every second part is inside quotes
                        docstring = parts[i].strip()  # Get the text inside
                        if docstring:  # If it's not empty
                            comments.append(f"Docstring: {docstring}")  # Keep it
        
        return comments  # Return all notes found
    
    def _extract_outputs(self, outputs: List) -> str:  # Function to read what the code printed out
        """
        Extract text from cell outputs
        """
        output_parts = []  # List for output text
        
        for output in outputs:  # Loop through all possible outputs
            if output.output_type == "stream":  # If it's simple printed text
                output_parts.append(output.text.strip())  # Add it
            elif output.output_type in ["execute_result", "display_data"]:  # If it's a final variable result
                if hasattr(output, 'data'):  # If it has data attached
                    if 'text/plain' in output.data:  # If it has a plain text version
                        output_parts.append(output.data['text/plain'].strip())  # Add it
            elif output.output_type == "error":  # If the code crashed
                # Include error messages as they might be educational
                error_text = f"Error: {output.ename}: {output.evalue}"  # Get error name and reason
                output_parts.append(error_text)  # Add it
        
        return "\n".join(output_parts)  # Join different outputs
    
    def get_metadata(self, notebook_path: str) -> Dict:  # Function to get file info
        """
        Extract notebook metadata
        """
        notebook_path = Path(notebook_path)  # Ensure Path object
        metadata = {  # Start info dictionary
            "filename": notebook_path.name,  # Name
            "size_bytes": notebook_path.stat().st_size,  # Size
        }
        
        try:  # Try opening the file for stats
            with open(notebook_path, 'r', encoding='utf-8') as f:  # Read file
                nb = nbformat.read(f, as_version=4)  # Parse it
            
            metadata.update({  # Add notebook specific stats
                "num_cells": len(nb.cells),  # Total cells
                "num_code_cells": sum(1 for cell in nb.cells if cell.cell_type == "code"),  # Code cells
                "num_markdown_cells": sum(1 for cell in nb.cells if cell.cell_type == "markdown"),  # Markdown cells
                "kernel": nb.metadata.get("kernelspec", {}).get("name", "unknown"),  # What language/kernel used
            })
        except Exception as e:  # If we can't read stats
            logger.warning(f"Could not extract metadata: {e}")  # Just log a warning
        
        return metadata  # Return info dictionary


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    loader = NotebookLoader(include_code=True, include_outputs=False)  # Init loader
    
    # Test with a notebook file
    # text = loader.load("sample.ipynb")
    # print(f"Extracted {len(text)} characters")
    # print(text[:500])
    
    print("NotebookLoader initialized successfully")  # Success message
