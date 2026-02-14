"""
FastAPI Server for EchoLearn AI - This is the backend server file
Backend API with document upload and voice query endpoints - Built with FastAPI
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form  # Import FastAPI tools for building web APIs
from fastapi.middleware.cors import CORSMiddleware  # Import tool to allow different websites to talk to this API
from fastapi.responses import FileResponse, JSONResponse  # Import ways to send files or data back to user
from pathlib import Path  # Import Path for managing file and folder paths
from typing import Optional  # Import Optional for variables that might be empty
import shutil  # Import tools for copying files
import logging  # Import logging to record what the server is doing
import time  # Import time for measuring performance or delays
from datetime import datetime  # Import datetime for adding timestamps to logs

from config import Config  # Import our project settings
from pdf_loader import PDFLoader  # Import our tool to read PDF files
from notebook_loader import NotebookLoader  # Import our tool to read Jupyter Notebooks
from text_cleaner import TextCleaner  # Import our tool to clean up messy text
from chunker import TextChunker  # Import our tool to split big text into small pieces
from build_vector_db import VectorDBBuilder  # Import our tool to create a searchable text database
from tutor_agent import TutorAgent  # Import our AI Brain (the tutor agent)
from speech_to_text import SpeechToText  # Import our tool to turn voice into text
from text_to_speech import TextToSpeech  # Import our tool to turn text into voice

# Setup logging
logging.basicConfig(  # Configure how we record server messages
    level=logging.INFO,  # Set the detail level of logs (INFO is standard)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Set the format of each log line
    handlers=[  # Specify where to save the logs
        logging.FileHandler("server.log"),  # Save logs to a file named server.log
        logging.StreamHandler()  # Also print logs to the terminal screen
    ]
)
logger = logging.getLogger(__name__)  # Create a logger object for this specific file

# Initialize FastAPI app
app = FastAPI(  # Create the main FastAPI application object
    title="EchoLearn AI API",  # Give our API a name
    description="Voice tutor API with RAG for document-based learning",  # Describe what the API does
    version="1.0.0"  # Set the version number
)

# Configure CORS
app.add_middleware(  # Add a "security guard" layer to our app
    CORSMiddleware,  # Use CORS middleware
    allow_origins=Config.CORS_ORIGINS,  # Allow only trusted websites to connect
    allow_credentials=True,  # Allow passing of cookies or logins
    allow_methods=["*"],  # Allow all types of requests (GET, POST, etc.)
    allow_headers=["*"],  # Allow all types of information in request headers
)

# Global instances (holders for our tools)
vector_db_builder: Optional[VectorDBBuilder] = None  # Placeholder for the database creator
tutor_agent: Optional[TutorAgent] = None  # Placeholder for the AI tutor
stt_engine: Optional[SpeechToText] = None  # Placeholder for the voice-to-text tool
tts_engine: Optional[TextToSpeech] = None  # Placeholder for the text-to-voice tool

# Document processors (the workers that prepare our files)
pdf_loader = PDFLoader()  # Create the PDF reader worker
notebook_loader = NotebookLoader(include_code=True, include_outputs=False)  # Create the Notebook reader worker
text_cleaner = TextCleaner()  # Create the text cleaning worker
text_chunker = TextChunker()  # Create the text splitting worker


@app.on_event("startup")  # Tell FastAPI to run this function when the server starts
async def startup_event():  # Define the startup logic
    """Initialize services on startup"""
    global vector_db_builder, tutor_agent, stt_engine, tts_engine  # Tell Python we are using the global variables
    
    logger.info("Starting EchoLearn AI Server...")  # Log that server initialization began
    
    try:  # Start error checking
        # Validate configuration
        Config.validate_config()  # Check if all API keys and settings are correct
        Config.ensure_directories()  # Make sure needed folders like 'uploads' exist
        
        # Initialize vector database builder
        vector_db_builder = VectorDBBuilder()  # Set up the database tool
        
        # Try to load existing index
        if vector_db_builder.load_index():  # Check for an existing saved database
            logger.info("Loaded existing vector database")  # Log success if found
        else:  # If no database found
            logger.info("No existing vector database found")  # Log that we are starting fresh
        
        # Initialize tutor agent
        tutor_agent = TutorAgent(use_memory=True)  # Create the AI tutor with "memory" to remember conversation
        logger.info("Tutor agent initialized")  # Log success
        
        # Initialize speech engines
        stt_engine = SpeechToText()  # Create the tool for hearing user voice
        logger.info("Speech-to-text engine initialized")  # Log success
        
        tts_engine = TextToSpeech()  # Create the tool for speaking to user
        logger.info("Text-to-speech engine initialized")  # Log success
        
        logger.info("✓ EchoLearn AI Server started successfully")  # Log that everything is ready
        
    except Exception as e:  # If something broke during startup
        logger.error(f"Error during startup: {e}")  # Record the error in logs
        raise  # Stop the server because it can't run properly


@app.get("/")  # Define a response for visiting the root website address
async def root():  # Define the root function
    """Root endpoint"""
    return {  # Send back a simple greeting message
        "message": "EchoLearn AI Voice Tutor API",  # Branding message
        "version": "1.0.0",  # Version number
        "status": "running"  # Status confirmation
    }


@app.get("/health")  # Define an address for checking if server is healthy
async def health_check():  # Define the health check logic
    """Health check endpoint"""
    return {  # Send back a report card of all systems
        "status": "healthy",  # Overall status
        "timestamp": datetime.now().isoformat(),  # Current time
        "services": {  # Status of individual parts
            "vector_db": vector_db_builder is not None,  # Check if database tool is active
            "tutor_agent": tutor_agent is not None and tutor_agent.is_ready(),  # Check if AI brain is ready
            "stt": stt_engine is not None,  # Check if hearing tool is active
            "tts": tts_engine is not None,  # Check if speaking tool is active
        },
        "vector_db_stats": vector_db_builder.get_stats() if vector_db_builder else {}  # Show how many docs we have
    }


@app.post("/upload")  # Define an address for receiving new document files
async def upload_document(  # Define the file processing logic
    file: UploadFile = File(...),  # The actual file being sent
    rebuild_index: bool = Form(False)  # Choice to clear old files or just add new ones
):
    """
    Upload and process a PDF or Jupyter Notebook file
    """
    try:  # Start error checking
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()  # Get the file extension (like .pdf)
        if file_ext not in Config.ALLOWED_EXTENSIONS:  # Check if we support this file type
            raise HTTPException(  # If unsupported, tell the user why
                status_code=400,
                detail=f"Unsupported file type. Allowed: {Config.ALLOWED_EXTENSIONS}"
            )
        
        # Save uploaded file
        upload_path = Config.UPLOAD_DIR / file.filename  # Decide where to save the file
        with open(upload_path, "wb") as buffer:  # Open a new file on our computer
            shutil.copyfileobj(file.file, buffer)  # Copy the user's file into our folder
        
        logger.info(f"Uploaded file saved: {file.filename}")  # Log the save action
        
        # Load document based on type
        start_time = time.time()  # Record the start time for measuring speed
        
        if file_ext == ".pdf":  # If it's a PDF
            text = pdf_loader.load(str(upload_path))  # Use PDF reader to extract text
            metadata = pdf_loader.get_metadata(str(upload_path))  # Get extra info like title or author
        elif file_ext == ".ipynb":  # If it's a Notebook
            text = notebook_loader.load(str(upload_path))  # Use Notebook reader to extract text
            metadata = notebook_loader.get_metadata(str(upload_path))  # Get notebook metadata
        else:  # Double check for safety
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Clean text
        cleaned_text = text_cleaner.clean(text)  # Remove extra spaces and junk characters
        
        # Chunk text
        chunks = text_chunker.chunk(  # Split big text into small snippets
            cleaned_text,
            metadata={  # Tag each snippet with info about its origin
                "source": file.filename,
                "file_type": file_ext,
                "upload_time": datetime.now().isoformat()
            }
        )
        
        # Build/update vector index
        if rebuild_index:  # If user wants to start fresh
            logger.info("Rebuilding vector index from scratch")  # Log the action
        
        num_docs = vector_db_builder.build_index(chunks, rebuild=rebuild_index)  # Add snippets to search engine
        
        # Save index
        vector_db_builder.save_index()  # Save the search engine to computer memory
        
        # Reload tutor agent's retriever
        if tutor_agent:  # If AI Brain is active
            tutor_agent.retriever.reload_index()  # Tell it to look at the new files we just added
        
        processing_time = time.time() - start_time  # Calculate how long processing took
        
        return {  # Send report of the upload back to the user
            "status": "success",  # status tag
            "filename": file.filename,  # file name
            "file_type": file_ext,  # file type
            "metadata": metadata,  # extra info
            "num_chunks": len(chunks),  # number of pieces we broke it into
            "total_documents_in_index": num_docs,  # how many total pieces are in our storage
            "processing_time": round(processing_time, 2),  # how fast we worked
            "index_rebuilt": rebuild_index,  # whether we started fresh
            "greeting_audio": None  # No voice greeting
        }
        
    except Exception as e:  # If an error happened anywhere during upload
        logger.error(f"Error processing upload: {e}")  # Log what went wrong
        raise HTTPException(status_code=500, detail=str(e))  # Tell user about the error


@app.post("/ask")  # Define an address for handling questions
async def ask_question(  # Define the questioning logic
    audio: UploadFile = File(None),  # Optional voice recording from user
    text: str = Form(None),  # Optional text question from user
    use_retrieval: bool = Form(True),  # Should we search the documents for answer?
    return_audio: bool = Form(True)  # Should the tutor speak back?
):
    """
    Ask a question via audio or text
    """
    try:  # Start error checking
        question = None  # Placeholder for the final text question
        transcription_time = 0  # Placeholder for measurement
        
        # Get question from audio or text
        if audio:  # If user sent a voice clip
            # Save audio file
            audio_path = Config.UPLOAD_DIR / f"question_{int(time.time())}.wav"  # Decide file name
            with open(audio_path, "wb") as buffer:  # Open new file
                shutil.copyfileobj(audio.file, buffer)  # Save the recording
            
            # Transcribe audio
            logger.info("Transcribing audio question...")  # Log that we are "listening"
            start_time = time.time()  # Start timer
            stt_result = stt_engine.transcribe(str(audio_path))  # Use hearing tool to turn voice into text
            question = stt_result["transcript"]  # Get the text transcript
            transcription_time = time.time() - start_time  # Stop timer
            
            logger.info(f"Question transcribed: '{question}'")  # Log what we heard
            
        elif text:  # If user just typed the question
            question = text  # Directly use the text
        else:  # If user sent nothing
            raise HTTPException(  # Send an error back
                status_code=400,
                detail="Either 'audio' or 'text' parameter required"
            )
        
        if not question or not question.strip() or question == "...":  # If question is empty or junk
            return JSONResponse(  # Return a "sorry" message
                status_code=200,
                content={
                    "status": "warning",
                    "question": "",
                    "answer": "I didn't quite catch that. Could you please repeat your question?",
                    "audio_path": None,
                    "sources": [],
                    "num_sources": 0,
                    "used_retrieval": False,
                    "used_memory": False,
                    "timing": {"total_time": 0}
                }
            )
        
        # Get answer from tutor agent
        logger.info(f"Processing question with tutor agent...")  # Log that AI Brain is thinking
        start_time = time.time()  # Start thinking timer
        
        result = tutor_agent.ask(question, use_retrieval=use_retrieval)  # Ask the AI tutor for answer
        answer = result["answer"]  # Get the answer text
        
        agent_time = time.time() - start_time  # Stop thinking timer
        
        # Generate audio response if requested
        audio_path = None  # Placeholder for voice file path
        synthesis_time = 0  # Timer for speaking
        
        if return_audio:  # If user wants tutor to speak
            logger.info("Generating audio response...")  # Log that we are preparing voice
            start_time = time.time()  # Start timer
            try:
                audio_path = tts_engine.synthesize(answer, add_pauses=True)  # Turn answer text into voice
                synthesis_time = time.time() - start_time  # Stop timer
            except Exception as tts_err:  # If speaking failed
                logger.error(f"TTS Synthesis failed: {tts_err}")  # Log failure
                audio_path = None  # Clear path
        
        return {  # Send everything back to the user
            "status": "success",  # tag
            "question": question,  # user's question
            "answer": answer,  # tutor's answer
            "audio_path": audio_path,  # path to hear the voice
            "sources": result["sources"],  # parts of documents used
            "num_sources": result["num_sources"],  # how many sources
            "used_retrieval": result["used_retrieval"],  # did we search docs?
            "used_memory": result["used_memory"],  # did we remember past chat?
            "timing": {  # speed report card
                "transcription_time": round(transcription_time, 2),
                "agent_time": round(agent_time, 2),
                "synthesis_time": round(synthesis_time, 2),
                "total_time": round(transcription_time + agent_time + synthesis_time, 2)
            }
        }
        
    except Exception as e:  # Catch all backend errors
        logger.error(f"Error processing question: {e}")  # Log failure
        raise HTTPException(status_code=500, detail=str(e))  # Send error back


@app.get("/audio/{filename}")  # Define address for downloading voice clips
async def get_audio(filename: str):  # Define voice delivery logic
    """
    Retrieve generated audio file
    """
    audio_path = Config.AUDIO_OUTPUT_DIR / filename  # Find where the voice file is stored
    
    if not audio_path.exists():  # If file is missing
        raise HTTPException(status_code=404, detail="Audio file not found")  # Tell user
    
    return FileResponse(  # Send the actual audio file back to browser
        audio_path,
        media_type="audio/mpeg",  # Tell browser it's an MP3 style file
        filename=filename
    )


@app.post("/clear-memory")  # Define address for resetting conversation
async def clear_memory():  # Define resetting logic
    """Clear conversation memory"""
    try:
        if tutor_agent:  # If AI Brain is active
            tutor_agent.clear_memory()  # Wipe out the chat history
            return {"status": "success", "message": "Conversation memory cleared"}  # Log success
        else:
            raise HTTPException(status_code=500, detail="Tutor agent not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation-summary")  # Define address for getting a chat recap
async def get_conversation_summary():  # Define summary logic
    """Get conversation summary"""
    try:
        if tutor_agent:  # If active
            summary = tutor_agent.get_conversation_summary()  # Get a short version of chat
            return summary  # Send to user
        else:
            raise HTTPException(status_code=500, detail="Tutor agent not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear-index")  # Define address for deleting all stored documents
async def clear_index():  # Define deletion logic
    """Clear vector database index"""
    try:
        if vector_db_builder:  # If active
            vector_db_builder.clear_index()  # Wipe the searchable database
            return {"status": "success", "message": "Vector index cleared"}  # Success message
        else:
            raise HTTPException(status_code=500, detail="Vector DB not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":  # If we are starting the server directly
    import uvicorn  # Import uvicorn server runner
    
    logger.info(f"Starting server on {Config.SERVER_HOST}:{Config.SERVER_PORT}")  # Log startup address
    
    uvicorn.run(  # Start the web server
        "server:app",  # Point to this file and the app object
        host=Config.SERVER_HOST,  # Set address (localhost)
        port=Config.SERVER_PORT,  # Set port (8000)
        reload=False  # Do not auto-restart on changes (more stable)
    )
