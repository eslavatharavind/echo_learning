"""
Conversation Memory for EchoLearn AI - This file helps the AI remember the chat
Manages chat history and context across multiple turns - Like a person remembering what you said first
"""

from typing import List, Dict, Optional  # Import types for organization
from datetime import datetime  # Import datetime for timestamping chat sessions
import logging  # Import logging for tracking memory activity

from config import Config  # Import project settings

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for the memory tool


class ConversationMemory:  # Define a class for managing chat history
    """Manage conversation history and context"""
    
    def __init__(self, max_tokens: int = None, max_turns: int = 5):  # Initialize memory settings
        """
        Initialize Conversation Memory
        """
        self.max_tokens = max_tokens or Config.MEMORY_MAX_TOKENS  # Set limit on total words (tokens) to remember
        self.max_turns = max_turns  # Set limit on number of back-and-forth messages to remember (shorter = faster)
        
        self.history: List[Dict] = []  # Create an empty list to store the chat turns
        self.current_session_id = self._generate_session_id()  # Create a unique ID for this chat session
        
        logger.info(f"ConversationMemory initialized: max_tokens={self.max_tokens}, max_turns={self.max_turns}")
    
    def add_interaction(  # Function to add a new question and answer to memory
        self,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add a user-assistant interaction to memory
        """
        interaction = {  # Package the data into a dictionary
            "timestamp": datetime.now().isoformat(),  # Record exactly when this happened
            "user": user_message,  # Store user message
            "assistant": assistant_response,  # Store AI reply
            "metadata": metadata or {}  # Store any extra info (like source names)
        }
        
        self.history.append(interaction)  # Add the package to our history list
        
        # Trim history if it's getting too long (to save costs and keep AI efficient)
        self._trim_history()
        
        logger.debug(f"Added interaction. History now has {len(self.history)} turns")
    
    def get_history(self, num_turns: Optional[int] = None) -> List[Dict]:  # Function to retrieve saved chat
        """
        Get conversation history
        """
        if num_turns is None:  # If user wants everything
            return self.history.copy()  # Return a copy of the full list
        
        return self.history[-num_turns:] if num_turns > 0 else []  # Return only the last few turns
    
    def get_formatted_history(  # Function to turn history into a neat string for the AI to read
        self,
        num_turns: Optional[int] = None,
        format: str = "text"
    ) -> str:
        """
        Get formatted conversation history
        """
        history = self.get_history(num_turns)  # Get the history data first
        
        if not history:  # If history is empty
            return ""  # Return nothing
        
        if format == "text":  # Choice 1: Plain text
            return self._format_as_text(history)
        elif format == "markdown":  # Choice 2: Markdown style (with bolding)
            return self._format_as_markdown(history)
        elif format == "chat":  # Choice 3: Chat style (Role: Message)
            return self._format_as_chat(history)
        else:  # Default to text
            return self._format_as_text(history)
    
    def _format_as_text(self, history: List[Dict]) -> str:  # Helper for text formatting
        """Format history as plain text"""
        lines = []  # List for lines of string
        for interaction in history:  # Loop through turns
            lines.append(f"Student: {interaction['user']}")  # Add student line
            lines.append(f"Tutor: {interaction['assistant']}")  # Add tutor line
            lines.append("")  # Add space line
        return "\n".join(lines).strip()  # Combine and return
    
    def _format_as_markdown(self, history: List[Dict]) -> str:  # Helper for markdown formatting
        """Format history as markdown"""
        lines = []  # List for lines
        for i, interaction in enumerate(history, 1):  # Loop with a counter
            lines.append(f"### Turn {i}")  # Add header
            lines.append(f"**Student**: {interaction['user']}")  # Bold student
            lines.append(f"**Tutor**: {interaction['assistant']}")  # Bold tutor
            lines.append("")  # Space
        return "\n".join(lines).strip()  # Combine
    
    def _format_as_chat(self, history: List[Dict]) -> str:  # Helper for AI-ready chat formatting
        """Format history for chat-style prompts"""
        messages = []  # List for messages
        for interaction in history:  # Loop turns
            messages.append(f"User: {interaction['user']}")  # Standard Role name
            messages.append(f"Assistant: {interaction['assistant']}")  # Standard Role name
        return "\n".join(messages)  # Combine
    
    def _trim_history(self):  # Internal helper to cut off old memories
        """Trim history to stay within limits"""
        # Trim by number of turns first (keep only the last X chats)
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]
            logger.debug(f"Trimmed history to {self.max_turns} turns")
        
        # Trim by approximate token count (AI models have a memory limit)
        total_chars = sum(
            len(h['user']) + len(h['assistant'])
            for h in self.history
        )
        approx_tokens = total_chars // 4  # Rough math: 4 characters per token
        
        while approx_tokens > self.max_tokens and len(self.history) > 1:  # While memory is too full
            # Remove oldest message until size is okay
            removed = self.history.pop(0)  # Remove very first item
            total_chars -= len(removed['user']) + len(removed['assistant'])  # Subtract its size
            approx_tokens = total_chars // 4  # Recalculate
            logger.debug(f"Trimmed oldest interaction (approx tokens: {approx_tokens})")
    
    def clear_history(self):  # Function to forget everything immediately
        """Clear all conversation history"""
        self.history = []  # Reset list to empty
        logger.info("Conversation history cleared")
    
    def get_last_question(self) -> Optional[str]:  # Helper to get only the latest question
        """Get the last user question"""
        if not self.history:  # If no history
            return None
        return self.history[-1]["user"]  # Return the user field of the last item
    
    def get_last_response(self) -> Optional[str]:  # Helper to get only the latest reply
        """Get the last assistant response"""
        if not self.history:  # If no history
            return None
        return self.history[-1]["assistant"]  # Return the assistant field of the last item
    
    def get_summary(self) -> Dict:  # Function to get stats about the current chat
        """
        Get a summary of the conversation
        """
        if not self.history:  # If empty
            return {
                "num_turns": 0,
                "session_id": self.current_session_id
            }
        
        # Calculate total characters spent
        total_user_chars = sum(len(h['user']) for h in self.history)
        total_assistant_chars = sum(len(h['assistant']) for h in self.history)
        
        return {  # Return summary report
            "num_turns": len(self.history),  # Total messages exchanged
            "session_id": self.current_session_id,  # Current chat ID
            "total_user_chars": total_user_chars,  # Total typing from user
            "total_assistant_chars": total_assistant_chars,  # Total typing from AI
            "first_question": self.history[0]["user"][:50] + "..." if self.history else None,  # Start of first question
            "last_question": self.get_last_question()  # Exact last question
        }
    
    def start_new_session(self):  # Reset everything for a fresh start
        """Start a new conversation session"""
        self.clear_history()  # Forget history
        self.current_session_id = self._generate_session_id()  # New ID
        logger.info(f"Started new session: {self.current_session_id}")
    
    @staticmethod
    def _generate_session_id() -> str:  # Static helper to make IDs based on time
        """Generate a unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")  # e.g. 20231027_153005


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    memory = ConversationMemory(max_tokens=500, max_turns=5)  # Init memory
    
    # Add some dummy interactions to test the memory
    memory.add_interaction(
        "What is machine learning?",
        "Machine learning is a way for computers to learn from examples without being explicitly programmed."
    )
    
    memory.add_interaction(
        "Can you give me an example?",
        "Sure! Think of how email filters learn to detect spam. They look at many emails you've marked as spam and learn patterns."
    )
    
    # Print out what the memory recorded
    print("=== Conversation History ===")
    print(memory.get_formatted_history(format="markdown"))
    
    # Print statistics
    print("\n=== Summary ===")
    summary = memory.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
