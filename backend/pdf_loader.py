"""
PDF Loader for EchoLearn AI - This file handles reading PDF documents
Extracts text from PDF files using PyMuPDF and pdfplumber - Two different tools for better results
"""

import fitz  # Import PyMuPDF (a powerful tool for reading and rendering PDFs)
import pdfplumber  # Import pdfplumber (another tool, good for complex layouts)
from pathlib import Path  # Import Path for handling folder/file locations
from typing import Optional  # Import Optional for variables that might be empty
import logging  # Import logging to track activity
import time  # Import time for measuring performance

logging.basicConfig(level=logging.INFO)  # Setup standard logging level
logger = logging.getLogger(__name__)  # Create a logger for this specific file


class PDFLoader:  # Define a class specifically for loading PDF content
    """Load and extract text from PDF files"""
    
    def __init__(self, use_fallback: bool = True):  # Initialize the loader
        """
        Initialize PDF loader
        """
        self.use_fallback = use_fallback  # Set whether to try a second tool if the first fails
    
    def load(self, pdf_path: str) -> str:  # Main function to get text from a PDF
        """
        Load PDF and extract all text, with OCR fallback for scanned documents
        """
        pdf_path = Path(pdf_path)  # Convert the string path into a Path object
        
        if not pdf_path.exists():  # If the file isn't where we expected
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")  # Stop and tell the user
        
        text = ""  # Start with an empty string for the text
        start_time = time.time()
        
        try:  # Try the fast method first
            logger.info(f"Loading PDF with PyMuPDF: {pdf_path.name}")  # Log which file we are opening
            text = self._load_with_pymupdf(pdf_path)  # Extract text using PyMuPDF
            
            # If PyMuPDF worked well, we skip pdfplumber (faster)
            if len(text.strip()) > 500:
                logger.info(f"PyMuPDF extraction successful ({len(text)} chars) in {time.time() - start_time:.2f}s")
                return text
                
        except Exception as e:  # If PyMuPDF has an error
            logger.error(f"PyMuPDF failed: {e}")  # Log the error
            
        # Try fallback if PyMuPDF failed or produced very little text
        if self.use_fallback and len(text.strip()) < 100:
            logger.info(f"Attempting fallback with pdfplumber...")
            text = self._load_with_pdfplumber(pdf_path)

        # If text is still too short, perform OCR (Optical Character Recognition)
        if len(text.strip()) < 100:  # If we got very little text (like in a picture-only PDF)
            logger.info("Minimal text found, starting OCR... (This may take several seconds)")  # Log that we are starting OCR
            ocr_start = time.time()
            ocr_text = self._load_with_ocr(pdf_path)  # Use OCR to "read" the images
            
            if ocr_text.startswith("ERROR_"):  # If OCR returned an error code
                raise ValueError(ocr_text)  # Stop and show the error
            
            text = ocr_text  # Use the OCR text instead
            logger.info(f"OCR completed in {time.time() - ocr_start:.2f}s")
            
            if not text or len(text.strip()) < 10:  # If still no text was found
                raise ValueError(f"Could not extract text from PDF even with OCR: {pdf_path.name}")  # Report failure
        
        total_time = time.time() - start_time
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path.name} in {total_time:.2f}s")  # Log success
        return text  # Return the final extracted text
    
    def _load_with_ocr(self, pdf_path: Path) -> str:  # Function for reading text from images
        """Extract text from scanned PDF using OCR without requiring Poppler"""
        try:  # Start OCR processing
            import pytesseract  # Import Tesseract (the actual eye for reading images)
            from PIL import Image  # Import PIL for image handling
            import io  # Import io for temporary data storage
            
            # Explicitly set Tesseract path if it exists in common Windows locations
            tesseract_paths = [  # List of where Tesseract might be installed on Windows
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Users\ARAVIND\AppData\Local\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
            
            for path in tesseract_paths:  # Check each likely location
                if Path(path).exists():  # If we found the Tesseract program
                    pytesseract.pytesseract.tesseract_cmd = path  # Tell Python where it is
                    break  # Stop looking
            
            logger.info(f"Opening PDF with PyMuPDF for OCR rendering: {pdf_path.name}")  # Log message
            text_parts = []  # List to hold text from each page
            
            with fitz.open(pdf_path) as doc:  # Open the PDF file
                for i, page in enumerate(doc, start=1):  # Loop through every page
                    logger.info(f"OCR-ing page {i}/{len(doc)}...")  # Log progress per page
                    
                    # Render page to image (pixmap) - no Poppler needed!
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Draw the page as a high-quality photo
                    img_data = pix.tobytes("png")  # Convert that photo to PNG data
                    img = Image.open(io.BytesIO(img_data))  # Open it as a PIL image
                    
                    try:  # Try to "read" the image
                        page_text = pytesseract.image_to_string(img)  # Tesseract reads the photo
                        if page_text.strip():  # If it found any words
                            text_parts.append(f"\n--- Page {i} (OCR) ---\n")  # Add a page header
                            text_parts.append(page_text)  # Add the words it found
                    except Exception as e:  # If reading fails
                        if "tesseract" in str(e).lower():  # If the failure is because Tesseract isn't installed
                            logger.error("Tesseract not found or not in PATH.")  # Log it
                            return "ERROR_TESSERACT_MISSING: Tesseract OCR is required for scanned PDFs. Please ensure it is installed."
                        raise e  # For other errors, stop the program
            
            return "\n".join(text_parts)  # Combine all page texts together
        except ImportError as e:  # If needed libraries are not installed
            logger.error(f"Required OCR libraries missing: {e}")  # Log the missing libs
            return f"ERROR_LIBS_MISSING: {str(e)}"  # Return library error message
        except Exception as e:  # For any other OCR error
            logger.error(f"OCR processing failed: {e}")  # Log failure
            return f"ERROR_OCR_GENERAL: {str(e)}"  # Return general error message

    def _load_with_pymupdf(self, pdf_path: Path) -> str:  # Internal function for fast text extraction
        """Extract text using PyMuPDF (fitz)"""
        text_parts = []  # List for text chunks
        with fitz.open(pdf_path) as doc:  # Open the document
            for page_num, page in enumerate(doc, start=1):  # Count and loop pages
                page_text = page.get_text()  # Ask PyMuPDF for all selectable text
                if page_text.strip():  # If page is not empty
                    text_parts.append(f"\n--- Page {page_num} ---\n")  # Add separator
                    text_parts.append(page_text)  # Add the text
        return "\n".join(text_parts)  # Joins everything with new lines

    def _load_with_pdfplumber(self, pdf_path: Path) -> str:  # Internal backup extraction function
        """Extract text using pdfplumber (fallback)"""
        text_parts = []  # List for text chunks
        with pdfplumber.open(pdf_path) as pdf:  # Open the file using the backup tool
            for page_num, page in enumerate(pdf.pages, start=1):  # Loop through pages
                page_text = page.extract_text()  # Extract the text
                if page_text and page_text.strip():  # If content found
                    text_parts.append(f"\n--- Page {page_num} ---\n")  # Add header
                    text_parts.append(page_text)  # Add content
        return "\n".join(text_parts)  # Join results together
    
    def get_metadata(self, pdf_path: str) -> dict:  # Function to get info about the file (not the text)
        """
        Extract PDF metadata
        """
        pdf_path = Path(pdf_path)  # Ensure path is a Path object
        metadata = {  # Start a dictionary for the info
            "filename": pdf_path.name,  # Store the name of the file
            "size_bytes": pdf_path.stat().st_size,  # Store the file size
        }
        
        try:  # Try to get specialized PDF info
            with fitz.open(pdf_path) as doc:  # Open the file
                metadata.update({  # Add more details
                    "num_pages": len(doc),  # Total number of pages
                    "title": doc.metadata.get("title", ""),  # Document title (if set)
                    "author": doc.metadata.get("author", ""),  # Author name (if set)
                    "subject": doc.metadata.get("subject", ""),  # Topic (if set)
                })
        except Exception as e:  # If we can't read metadata
            logger.warning(f"Could not extract metadata: {e}")  # Just log a warning
        
        return metadata  # Return the info dictionary


if __name__ == "__main__":  # Code here runs only if you launch this file directly
    # Example usage
    loader = PDFLoader()  # Create a loader
    
    # Test with a PDF file
    # text = loader.load("sample.pdf")
    # print(f"Extracted {len(text)} characters")
    # print(text[:500])
    
    print("PDFLoader initialized successfully")  # Just print success message
