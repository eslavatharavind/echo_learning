"""
Text-to-Speech for EchoLearn AI - This file handles giving the AI a voice
Converts text to natural-sounding speech audio - So the AI "talks" to you
"""

from pathlib import Path  # Import Path for managing file and folder locations
from typing import Optional  # Import types for organization
import logging  # Import logging for tracking sound generation
import time  # Import time for naming files and measuring speed
from openai import OpenAI  # Import OpenAI client (works for their high-quality voices)

from config import Config  # Import project settings

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for text-to-speech


class TextToSpeech:  # Define the class for converting text into voice
    """Convert text to speech audio"""
    
    def __init__(  # Initialize settings and the voice engine
        self,
        provider: str = None,
        voice: str = None,
        speed: float = None
    ):
        """
        Initialize Text-to-Speech
        """
        self.provider = provider or Config.TTS_PROVIDER  # Choose the voice company (OpenAI, Google, etc.)
        self.voice = voice or Config.TTS_VOICE  # Select the specific voice (e.g., "Alloy" or "Nova")
        self.speed = speed or Config.TTS_SPEED  # Set how fast the AI talks
        self.output_dir = Config.AUDIO_OUTPUT_DIR  # Set where the audio clips will be saved
        
        # Ensure the folder for audio clips exists on your computer
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the specific tool for the chosen company
        self._initialize_tts()
        
        logger.info(f"TextToSpeech initialized with {self.provider} provider")
    
    def _initialize_tts(self):  # Internal function to connect to the voice provider
        """Initialize TTS provider"""
        if self.provider == "openai":  # If user chose OpenAI's premium voices
            if not Config.OPENAI_API_KEY:  # Check for API key
                raise ValueError("OpenAI API key required for OpenAI TTS")
            
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)  # Connect to OpenAI
            self.model = Config.get_tts_model()  # Use specific model from config
            
        elif self.provider == "gtts":  # If user chose Google's free voices
            # gTTS is simple and doesn't need a special connection setup
            from gtts import gTTS
            self.gtts = gTTS
            
        elif self.provider == "coqui":  # If user chose local computer voices (Advanced)
            # Coqui TTS runs entirely on your machine
            try:
                from TTS.api import TTS
                model_name = Config.get_tts_model()
                self.tts_model = TTS(model_name)  # Load the heavy voice model
                logger.info(f"Loaded Coqui TTS model: {model_name}")
            except ImportError:
                raise ImportError(
                    "Coqui TTS not installed. Install with: pip install TTS"
                )
        else:
            raise ValueError(f"Unsupported TTS provider: {self.provider}")
    
    def synthesize(  # Main function to turn text into an MP3 file
        self,
        text: str,
        output_filename: Optional[str] = None,
        add_pauses: bool = True
    ) -> str:
        """
        Convert text to speech and save as audio file
        """
        if not text or not text.strip():  # If there is no text to say
            raise ValueError("Text cannot be empty")
        
        logger.info(f"Synthesizing speech ({len(text)} chars) using {self.provider}")
        start_time = time.time()  # Start the timer
        
        # Make the speech sound more like a teacher (add pauses between sentences)
        if add_pauses:
            text = self._add_teaching_pauses(text)
        
        # Create a unique filename if none was provided (using current time)
        if output_filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"tts_{timestamp}.mp3"
        
        # Ensure the filename ends with .mp3
        if not output_filename.endswith('.mp3'):
            output_filename += '.mp3'
        
        output_path = self.output_dir / output_filename  # Full path to the file
        
        # Use the chosen provider to actually make the sound
        if self.provider == "openai":
            self._synthesize_openai(text, output_path)
        elif self.provider == "gtts":
            self._synthesize_gtts(text, output_path)
        elif self.provider == "coqui":
            self._synthesize_coqui(text, output_path)
        
        synthesis_time = time.time() - start_time  # Calculate how long it took
        logger.info(f"Speech synthesis complete in {synthesis_time:.2f}s: {output_path.name}")
        
        return str(output_path)  # Return the path to the finished MP3 file
    
    def _synthesize_openai(self, text: str, output_path: Path):  # Helper for OpenAI synthesis
        """Synthesize using OpenAI TTS"""
        response = self.client.audio.speech.create(  # Ask OpenAI to generate audio
            model=self.model,
            voice=self.voice,
            input=text,
            speed=self.speed
        )
        
        # Save the audio data sent back by OpenAI into our local file
        response.stream_to_file(str(output_path))
    
    def _synthesize_gtts(self, text: str, output_path: Path):  # Helper for Google synthesis
        """Synthesize using Google TTS (gTTS)"""
        tts = self.gtts(  # Create Google speech object
            text=text,
            lang=Config.STT_LANGUAGE,  # Read in chosen language
            slow=False if self.speed >= 1.0 else True  # Support basic slow/normal speed
        )
        tts.save(str(output_path))  # Save to local file
    
    def _synthesize_coqui(self, text: str, output_path: Path):  # Helper for local synthesis
        """Synthesize using Coqui TTS"""
        # Note: Coqui usually makes .wav files, which are larger than .mp3
        wav_path = output_path.with_suffix('.wav')
        self.tts_model.tts_to_file(text=text, file_path=str(wav_path))  # Generate WAV
        
        # If the user really wants an MP3, we try to convert it
        if output_path.suffix == '.mp3':
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_wav(str(wav_path))
                audio.export(str(output_path), format='mp3')
                wav_path.unlink()  # Delete the original large WAV file
            except ImportError:
                logger.warning("pydub not available for MP3 conversion, keeping WAV")
                return str(wav_path)
    
    def _add_teaching_pauses(self, text: str) -> str:  # Helper to make AI sound like a tutor
        """
        Add natural pauses to text for better teaching delivery
        """
        # Change single dots into triple dots (tells voice AI to pause longer)
        text = text.replace(". ", "... ")
        text = text.replace("! ", "... ")
        text = text.replace("? ", "... ")
        
        # Add a clear pause before starting an explanation
        text = text.replace(": ", ":... ")
        
        # Add a pause before saying "For example" (sounds more natural)
        text = text.replace("For example,", "... For example,")
        text = text.replace("For instance,", "... For instance,")
        
        return text  # Return the "paused" text
    
    def synthesize_streaming(self, text: str) -> str:  # Function for faster live playback
        """
        Synthesize speech optimized for streaming/real-time playback
        """
        return self.synthesize(text, add_pauses=True)  # Currently just uses standard synthesis
    
    def get_available_voices(self) -> list:  # Show which voice names are allowed
        """Get list of available voices for current provider"""
        if self.provider == "openai":
            return ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
        elif self.provider == "gtts":
            return ["default"]  # Google's free version doesn't have many options
        elif self.provider == "coqui":
            return ["default"]  # Depends on the downloaded model
        return []


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    try:
        tts = TextToSpeech()  # Init engine
        
        print(f"Text-to-Speech initialized with {tts.provider} provider")
        print(f"Available voices: {', '.join(tts.get_available_voices())}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print("Make sure to set your API keys in .env file if using OpenAI TTS")
