"""
Configuration file for EchoLearn AI Voice Tutor - This is where all settings are kept
Supports multiple LLM and TTS providers with flexible API key management - Built with standard config patterns
"""

import os  # Import os for interacting with the operating system (like getting environment variables)
from pathlib import Path  # Import Path for managing file and folder paths easily
from dotenv import load_dotenv  # Import load_dotenv to read the .env file

# Load environment variables from .env file
load_dotenv()  # This picks up your secret API keys from the .env file


class Config:  # Define a class to hold all our settings in one place
    """Main configuration class for EchoLearn AI"""
    
    # ============ API Keys ============
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Get OpenAI key from secrets
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Get Groq key from secrets
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")  # Get HuggingFace key from secrets
    
    # ============ LLM Configuration ============
    # Options: "openai", "groq", "llama", "mistral"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Choose which AI brain company to use
    
    # Model names for different providers
    LLM_MODELS = {  # A dictionary mapping companies to their specific AI models
        "openai": os.getenv("OPENAI_MODEL", "gpt-4"),  # Default to GPT-4 for OpenAI
        "groq": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),  # Default to Llama-3.1 for Groq
        "llama": "meta-llama/Llama-2-7b-chat-hf",  # Local Llama model name
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2"  # Local Mistral model name
    }
    
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))  # Set how "creative" or "precise" the AI is
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))  # Set the maximum length of AI answers
    
    # ============ Embedding Configuration ============
    # Options: "openai", "sentence-transformers"
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")  # Choose tool for making text searchable
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Choose specific search-tool model
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))  # Set the size of the search-vector
    
    # ============ TTS Configuration ============
    # Options: "openai", "coqui", "gtts"
    TTS_PROVIDER = os.getenv("TTS_PROVIDER", "openai")  # Choose which voice company to use
    
    # TTS model names
    TTS_MODELS = {  # Map companies to their voice models
        "openai": os.getenv("OPENAI_TTS_MODEL", "tts-1"),  # Default OpenAI tts model
        "coqui": "tts_models/en/ljspeech/tacotron2-DDC",  # Local coqui model
        "gtts": "en"  # Google text-to-speech language code
    }
    
    # Voice options
    TTS_VOICE = os.getenv("TTS_VOICE", "alloy")  # For OpenAI: alloy, echo, fable, onyx, nova, shimmer
    TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))  # Set how fast the tutor speaks
    
    # ============ STT Configuration ============
    # Options: "faster-whisper", "openai-whisper"
    STT_PROVIDER = os.getenv("STT_PROVIDER", "faster-whisper")  # Choose tool for hearing user voice
    STT_MODEL = os.getenv("STT_MODEL", "base")  # tiny, base, small, medium, large (bigger is better but slower)
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")  # Set language to English
    
    # Silence detection settings
    STT_PAUSE_THRESHOLD = float(os.getenv("STT_PAUSE_THRESHOLD", "6.0"))  # Wait 6 seconds of silence to stop recording
    STT_ENERGY_THRESHOLD = float(os.getenv("STT_ENERGY_THRESHOLD", "0.5"))  # Set voice loudness sensitivity
    STT_MIN_SILENCE_DURATION_MS = int(os.getenv("STT_MIN_SILENCE_DURATION_MS", "1000"))  # Set minimum silence in milliseconds
    
    # ============ Vector Database Configuration ============
    # Options: "faiss", "chroma"
    VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "faiss")  # Choose database type
    VECTOR_DB_PATH = Path(os.getenv("VECTOR_DB_PATH", "./data/vector_db"))  # Set where to save document index
    
    # Retrieval parameters
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))  # Look at 3 best matches in docs (shorter context = faster)
    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))  # Only use matches better than 0.5
    
    # ============ Document Processing Configuration ============
    # Chunking parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Break documents into 500-word pieces
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))  # Make pieces overlap by 100 words (to keep context)
    
    # Upload settings
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))  # Where to put uploaded files
    ALLOWED_EXTENSIONS = [".pdf", ".ipynb"]  # Support only PDF and Notebooks
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))  # Limit file size to 50 MB
    
    # ============ Storage Paths ============
    DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))  # Primary data folder
    AUDIO_OUTPUT_DIR = Path(os.getenv("AUDIO_OUTPUT_DIR", "./data/audio_output"))  # Folder for voice clips
    LOGS_DIR = Path(os.getenv("LOGS_DIR", "./logs"))  # Folder for log files
    
    # ============ Server Configuration ============
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")  # Run server on all available network addresses
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))  # Run server on port 8000
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")  # Allowed frontends
    
    # ============ Memory Configuration ============
    MEMORY_MAX_TOKENS = int(os.getenv("MEMORY_MAX_TOKENS", "1000"))  # Limit how much chat history AI remembers
    
    # ============ Helper Methods ============
    @classmethod  # Define a class method (doesn't need an instance)
    def ensure_directories(cls):  # Function to create missing folders
        """Create necessary directories if they don't exist"""
        dirs = [  # List of folders to check
            cls.DATA_DIR,
            cls.UPLOAD_DIR,
            cls.VECTOR_DB_PATH,
            cls.AUDIO_OUTPUT_DIR,
            cls.LOGS_DIR
        ]
        for directory in dirs:  # Loop through each folder path
            directory.mkdir(parents=True, exist_ok=True)  # Create it if it's missing
    
    @classmethod  # Define a class method
    def validate_config(cls):  # Function to check for errors in settings
        """Validate configuration and check for required API keys"""
        errors = []  # Start with an empty list of errors
        
        # Check LLM provider API keys
        if cls.LLM_PROVIDER == "openai" and not cls.OPENAI_API_KEY:  # If using OpenAI but no key
            errors.append("OpenAI API key required for LLM_PROVIDER='openai'")  # Record error
        elif cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:  # If using Groq but no key
            errors.append("Groq API key required for LLM_PROVIDER='groq'")  # Record error
        
        # Check TTS provider API keys
        if cls.TTS_PROVIDER == "openai" and not cls.OPENAI_API_KEY:  # If using OpenAI voice but no key
            errors.append("OpenAI API key required for TTS_PROVIDER='openai'")  # Record error
        
        # Check embedding provider API keys
        if cls.EMBEDDING_PROVIDER == "openai" and not cls.OPENAI_API_KEY:  # If using OpenAI search but no key
            errors.append("OpenAI API key required for EMBEDDING_PROVIDER='openai'")  # Record error
        
        if errors:  # If any errors were found
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))  # Stop program and tell user
        
        return True  # Return True if all is well
    
    @classmethod  # Define a class method
    def get_llm_model(cls):  # Helper function to get the current AI brain model
        """Get the current LLM model name"""
        return cls.LLM_MODELS.get(cls.LLM_PROVIDER, cls.LLM_MODELS["openai"])  # Return model for chosen company
    
    @classmethod  # Define a class method
    def get_tts_model(cls):  # Helper function to get the current voice model
        """Get the current TTS model name"""
        return cls.TTS_MODELS.get(cls.TTS_PROVIDER, cls.TTS_MODELS["openai"])  # Return model for chosen company


# Initialize directories on import
Config.ensure_directories()  # Automatically create folders as soon as this file is loaded
