"""
Tutor Agent for EchoLearn AI - This is the "brain" of our application
Main RAG-based tutor that combines retrieval, LLM, and memory - It coordinates everything
"""

from typing import Optional, Dict  # Import types for organization
from openai import OpenAI  # Import OpenAI client (works for both GPT and Groq)
import logging  # Import logging for tracking the brain's thoughts

from config import Config  # Import our project settings
from retriever import DocumentRetriever  # Import the tool that finds relevant document parts
from prompt import TutorPrompts  # Import the instruction templates for the AI
from memory import ConversationMemory  # Import the tool that remembers past chat

logging.basicConfig(level=logging.INFO)  # Setup standard log reports
logger = logging.getLogger(__name__)  # Create a logger for the tutor agent


class TutorAgent:  # Define the main AI Brain class
    """RAG-based AI tutor agent"""
    
    def __init__(  # Initialize the brain with its tools
        self,
        use_memory: bool = True,
        retriever: Optional[DocumentRetriever] = None
    ):
        """
        Initialize Tutor Agent
        """
        self.use_memory = use_memory  # Save whether we should remember the chat
        
        # Initialize the retriever (the tool that looks through our PDFs)
        self.retriever = retriever or DocumentRetriever()
        
        # Initialize memory (the tool that remembers what we just said)
        self.memory = ConversationMemory() if use_memory else None
        
        # Initialize the AI Model (LLM) based on what the user chose (OpenAI or Groq)
        self.llm_provider = Config.LLM_PROVIDER
        self._initialize_llm()
        
        logger.info(f"TutorAgent initialized with {self.llm_provider} LLM")  # Log startup
    
    def _initialize_llm(self):  # Internal function to setup the connection to the AI company
        """Initialize the LLM based on configured provider"""
        if self.llm_provider == "openai":  # If using OpenAI (ChatGPT)
            if not Config.OPENAI_API_KEY:  # Check if key is missing
                raise ValueError("OpenAI API key not found in configuration")  # Error if so
            
            self.client = OpenAI(api_key=Config.OPENAI_API_KEY)  # Connect to OpenAI
            self.model = Config.get_llm_model()  # Use the model from config (e.g. gpt-4)
            
        elif self.llm_provider == "groq":  # If using Groq (super fast AI)
            if not Config.GROQ_API_KEY:  # Check if key is missing
                raise ValueError("Groq API key not found in configuration")  # Error if so
            
            # Groq uses the exact same software tool as OpenAI (OpenAI-compatible)
            self.client = OpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"  # Just point it to Groq's web address
            )
            self.model = Config.get_llm_model()  # Use the model from config (eg. llama-3.1)
            
        else:  # If the user chose something we don't support
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def ask(  # The main function to talk to the tutor
        self,
        question: str,
        use_retrieval: bool = True,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Ask a question to the tutor
        """
        logger.info(f"Processing question: '{question[:50]}...'")  # Log the start of the question
        
        # Retrieve relevant context (find the right page in the PDF)
        context = ""  # Start with no document info
        sources = []  # Start with no sources list
        
        if use_retrieval and self.retriever.is_ready():  # If search is ON and we have documents
            retrieval_result = self.retriever.retrieve_with_context(  # Search the database
                question,
                top_k=top_k
            )
            context = retrieval_result["context"]  # Get the combined text from the documents
            sources = retrieval_result["results"]  # Get a list of which chunks were found
            logger.info(f"Retrieved {len(sources)} relevant sources")  # Log finding info
        else:
            logger.info("Skipping retrieval (disabled or retriever not ready)")  # Log that we skip search
        
        # Get chat history (remember what we said 5 minutes ago)
        chat_history = ""
        if self.memory:  # If memory is ON
            chat_history = self.memory.get_formatted_history(  # Get the last few messages
                num_turns=3,  # Include last 3 turns for context
                format="chat"
            )
        
        # Choose the right instructions (prompt) for the AI brain
        if chat_history and context:  # Best case: we have both history AND document info
            user_prompt = TutorPrompts.format_conversational_prompt(
                question=question,
                context=context,
                chat_history=chat_history
            )
        elif chat_history:  # Second case: only history (no search results found)
            user_prompt = TutorPrompts.format_followup_prompt(
                question=question,
                chat_history=chat_history
            )
        elif context:  # Third case: found document info but no previous history
            user_prompt = TutorPrompts.format_rag_prompt(
                question=question,
                context=context
            )
        else:  # Fallback: just ask the question directly to the AI
            user_prompt = f"Please answer this question: {question}"
        
        # Generate the actually answer using the AI (GPT-4 or Llama-3)
        try:
            response = self._generate_response(user_prompt)  # Send instructions to the AI company
            logger.info(f"Generated response ({len(response)} chars)")  # Log when done
            
        except Exception as e:  # If the AI company is down or errors happened
            logger.error(f"Error generating response: {e}")  # Log the error
            response = "I apologize, but I encountered an error processing your question. Please try again."
        
        # Save this interaction to memory (so we remember it for the NEXT question)
        if self.memory:
            self.memory.add_interaction(
                user_message=question,
                assistant_response=response,
                metadata={"num_sources": len(sources)}
            )
        
        # Return a report card for this question
        return {
            "answer": response,  # The text answer
            "question": question,  # The user's question
            "sources": sources,  # Which document parts were used
            "num_sources": len(sources),  # How many sources
            "used_memory": chat_history != "",  # Did we use history?
            "used_retrieval": use_retrieval and len(sources) > 0  # Did we use documents?
        }
    
    def _generate_response(self, prompt: str) -> str:  # Internal helper to actually call the AI
        """
        Generate response using configured LLM
        """
        # Prepare the messages for the AI model
        messages = [
            {"role": "system", "content": TutorPrompts.get_system_prompt()},  # Give it its personality
            {"role": "user", "content": prompt}  # Give it the question and context
        ]
        
        # Send the request over the internet to OpenAI/Groq
        response = self.client.chat.completions.create(
            model=self.model,  # The model name
            messages=messages,  # The conversation contents
            temperature=Config.LLM_TEMPERATURE,  # How creative to be
            max_tokens=Config.LLM_MAX_TOKENS  # How long the answer can be
        )
        
        # Return only the text reply from the AI
        return response.choices[0].message.content.strip()
    
    def simplify_explanation(self, text: str) -> str:  # Special tool to make things easier
        """
        Simplify a complex explanation
        """
        prompt = TutorPrompts.format_simplify_prompt(text)  # Get "simplify" instructions
        
        messages = [
            {"role": "system", "content": TutorPrompts.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_examples(self, concept: str, num_examples: int = 2) -> str:  # Tools to give examples
        """
        Generate examples for a concept
        """
        # Get relevant document context first
        context = ""
        if self.retriever.is_ready():
            retrieval_result = self.retriever.retrieve_with_context(concept, top_k=3)
            context = retrieval_result["context"]
        
        prompt = TutorPrompts.format_example_prompt(concept, context, num_examples)  # Get "example" instructions
        
        messages = [
            {"role": "system", "content": TutorPrompts.get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE,
            max_tokens=Config.LLM_MAX_TOKENS
        )
        
        return response.choices[0].message.content.strip()
    
    def clear_memory(self):  # Function to forget the chat history
        """Clear conversation memory"""
        if self.memory:  # If memory is active
            self.memory.clear_history()  # Wipe history
            logger.info("Memory cleared")  # Log action
    
    def get_conversation_summary(self) -> Dict:  # Function to get a recap of what was discussed
        """Get conversation summary"""
        if not self.memory:  # If memory is OFF
            return {"memory_enabled": False}
        
        summary = self.memory.get_summary()  # Get summary from memory tool
        summary["memory_enabled"] = True
        return summary
    
    def is_ready(self) -> bool:  # Simple check if setup is complete
        """Check if tutor is ready to answer questions"""
        return self.client is not None  # If we have a connected client, we are ready


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    try:
        tutor = TutorAgent(use_memory=True)  # Init tutor
        
        if tutor.is_ready():  # Check health
            print("Tutor is ready!")  # Success info
        else:
            print("Tutor not ready. Check configuration.")
            
    except Exception as e:
        print(f"Error initializing tutor: {e}")
        print("Make sure to set your API keys in .env file")
