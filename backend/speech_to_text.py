"""
Speech-to-Text for EchoLearn AI - This file handles hearing and understanding user voice
Converts audio to text using Faster-Whisper - A very fast and accurate tool
"""

from faster_whisper import WhisperModel  # Import the Faster-Whisper tool
from pathlib import Path  # Import Path for managing file locations
from typing import Optional, Dict  # Import types for organization
import logging  # Import logging for tracking processing
import time  # Import time for measuring speed

from config import Config  # Import project settings

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for speech-to-text


class SpeechToText:  # Define the class for converting voice to text
    """Convert speech audio to text"""
    
    def __init__(  # Initialize the voice recognition engine
        self,
        model_size: str = None,
        language: str = None,
        device: str = "cpu"
    ):
        """
        Initialize Speech-to-Text
        """
        self.model_size = model_size or Config.STT_MODEL  # Select the model size (tiny, small, base, etc.)
        self.language = language or Config.STT_LANGUAGE  # Select the default language
        self.device = device  # Choose to run on CPU or Graphics Card
        
        logger.info(f"Loading Faster-Whisper model: {self.model_size}")  # Log the start of loading
        start_time = time.time()  # Record the start time
        
        # Initialize Faster-Whisper model (the "ears" of our AI)
        # compute_type: "int8" makes it run fast on your standard CPU
        compute_type = "int8" if device == "cpu" else "float16"
        
        self.model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type
        )
        
        load_time = time.time() - start_time  # Calculate how long loading took
        logger.info(f"Model loaded in {load_time:.2f} seconds")  # Log finish
    
    def transcribe(  # Main function to turn an audio file into text
        self,
        audio_path: str,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio file to text
        """
        audio_path = Path(audio_path)  # Ensure path is a Path object
        
        if not audio_path.exists():  # If the audio file is missing
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"Transcribing: {audio_path.name}")  # Log processing start
        start_time = time.time()  # Start the clock
        
        # Run the transcription process
        language = language or self.language  # Use choice language or default
        
        # This function "listens" to the entire file
        segments, info = self.model.transcribe(
            str(audio_path),  # Path must be a string
            language=language,  # Tell it what language to listen for
            beam_size=5,  # Complexity of search (bigger is better but slower)
            vad_filter=True,  # "Voice Activity Detection" - skip silence automatically
            vad_parameters=dict(min_silence_duration_ms=Config.STT_MIN_SILENCE_DURATION_MS)
        )
        
        # Collect the text from all the different parts of the recording
        transcript_parts = []
        all_segments = []
        
        for segment in segments:  # Loop through each sentence or chunk found
            transcript_parts.append(segment.text)  # Add just the text
            all_segments.append({  # Add full info for each part
                "start": segment.start,  # When silence ended
                "end": segment.end,  # When silence started again
                "text": segment.text.strip(),  # The words
                "confidence": getattr(segment, 'avg_logprob', 0.0)  # How sure the AI is
            })
        
        # Join all the sentences into one big paragraph
        transcript = " ".join(transcript_parts).strip()
        transcription_time = time.time() - start_time  # Calculate total work time
        
        # Calculate how "confident" the AI was about its hearing on average
        avg_confidence = (
            sum(s["confidence"] for s in all_segments) / len(all_segments)
            if all_segments else 0.0
        )
        
        logger.info(
            f"Transcription complete in {transcription_time:.2f}s: "
            f"'{transcript[:50]}...'"
        )
        
        return {  # Return the full report
            "transcript": transcript,  # The text user said
            "language": info.language,  # The language detected
            "language_probability": info.language_probability,  # Probability score of language
            "duration": info.duration,  # Total audio length
            "segments": all_segments,  # Detailed timestamps
            "avg_confidence": avg_confidence,  # Average certainty
            "transcription_time": transcription_time  # Processing speed
        }
    
    def transcribe_simple(self, audio_path: str) -> str:  # Helper to just get text back
        """
        Simple transcription that returns only the text
        """
        result = self.transcribe(audio_path)  # Do the heavy lifting
        return result["transcript"]  # Return only the transcript part
    
    def is_speech_detected(self, audio_path: str, min_duration: float = 0.5) -> bool:  # Check if someone spoke
        """
        Check if speech is detected in audio file
        """
        try:
            result = self.transcribe(audio_path)  # Try to "listen"
            
            # Check if there were any words and if it lasted long enough
            has_content = len(result["transcript"].strip()) > 0
            sufficient_duration = result["duration"] >= min_duration
            
            return has_content and sufficient_duration  # If both are true, someone definitely spoke
            
        except Exception as e:
            logger.error(f"Error checking for speech: {e}")
            return False
    
    def get_supported_formats(self) -> list:  # Show which files we can read
        """Get list of supported audio formats"""
        return [
            ".wav", ".mp3", ".flac", ".ogg", ".m4a",
            ".wma", ".aac", ".opus", ".webm"
        ]


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    try:
        stt = SpeechToText(model_size="base")  # Load the model
        
        print(f"Speech-to-Text initialized with {stt.model_size} model")
        print(f"Supported formats: {', '.join(stt.get_supported_formats())}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
