"""
Prompt Templates for EchoLearn AI Tutor - This file contains the "scripts" the AI reads
Defines the tutor personality and prompt structure - Tells the AI how to act and teach
"""

from typing import List, Dict  # Import types for better organization


class TutorPrompts:  # Define a class to hold all AI instruction templates
    """Prompt templates for the AI tutor"""
    
    # System prompt defining tutor personality (The "Core identity" of the AI)
    SYSTEM_PROMPT = """You are EchoLearn, a concise and friendly AI tutor. 
Your goal is to explain concepts in simple, high-level terms that are extremely easy to understand.

Teaching Principles:
- **Be Brief**: Give short, punchy answers (ideally 2-4 sentences). Avoid long technical details unless specifically asked.
- **Explain the "Why" and "What"**: Focus on the overall concept. Use active, simple language.
- **Voice-First**: Your answers will be spoken aloud, so keep them conversational and clear.
- **One Concept at a Time**: Don't overwhelm the student. Focus on the core answer to their question."""

    # RAG prompt template (Instructions for when the AI has document info)
    RAG_TEMPLATE = """Based on the following information from the study materials, please answer the student's question.

=== Study Materials ===
{context}

=== Student's Question ===
{question}

=== Instructions ===
Please provide a clear, educational response that:
1. Directly answers the question using information from the study materials
2. Explains concepts in simple, easy-to-understand language
3. Includes relevant examples or analogies when helpful
4. Breaks down complex ideas into digestible steps
5. Encourages further learning

If the study materials don't contain enough information to fully answer the question, acknowledge what you can answer and what requires more information.

Your Response:"""

    # Conversational prompt template (Instructions for remembering past chat + document info)
    CONVERSATIONAL_TEMPLATE = """Based on our previous conversation and the following study materials, please answer the student's question.

=== Previous Conversation ===
{chat_history}

=== Study Materials ===
{context}

=== Student's Question ===
{question}

=== Instructions ===
Provide a thoughtful response that:
1. Takes into account our previous discussion
2. Uses information from the study materials when relevant
3. Explains new concepts clearly and simply
4. Builds upon what we've already discussed
5. Encourages deeper understanding

Your Response:"""

    # Follow-up question prompt (Instructions for when no new documents are searched)
    FOLLOWUP_TEMPLATE = """Based on our conversation, please respond to this follow-up question.

=== Previous Conversation ===
{chat_history}

=== Student's Follow-up Question ===
{question}

=== Instructions ===
Provide a clear response that:
1. Relates to our previous discussion
2. Clarifies or expands on concepts we've covered
3. Uses simple language and helpful examples
4. Maintains context from earlier in the conversation

Your Response:"""

    # Example generation prompt (Instructions for making up examples)
    EXAMPLE_TEMPLATE = """Please provide {num_examples} practical example(s) to illustrate this concept:

=== Concept ===
{concept}

=== Study Materials Context ===
{context}

=== Instructions ===
Create example(s) that:
1. Are relevant and easy to understand
2. Use real-world scenarios when possible
3. Show the concept in action
4. Include code examples if the concept is technical
5. Start simple and build complexity if multiple examples

Your Examples:"""

    # Simplification prompt (Instructions for explaining things like a 5th grader)
    SIMPLIFY_TEMPLATE = """A student is struggling to understand this explanation. Please simplify it.

=== Original Explanation ===
{original_text}

=== Instructions ===
Rewrite this explanation to:
1. Use simpler words (5th-grade reading level when possible)
2. Break down complex sentences
3. Add relatable analogies or metaphors
4. Remove unnecessary technical details
5. Keep the core meaning intact

Simplified Explanation:"""

    @staticmethod  # Static methods don't need a class instance
    def format_rag_prompt(question: str, context: str) -> str:  # Helper to fill in the RAG script
        """
        Format a RAG prompt with question and context
        """
        return TutorPrompts.RAG_TEMPLATE.format(  # Use Python's .format() to plug in the strings
            question=question,
            context=context
        )
    
    @staticmethod
    def format_conversational_prompt(  # Helper to fill in the History + Context script
        question: str,
        context: str,
        chat_history: str
    ) -> str:
        """
        Format a conversational prompt with history
        """
        return TutorPrompts.CONVERSATIONAL_TEMPLATE.format(
            question=question,
            context=context,
            chat_history=chat_history
        )
    
    @staticmethod
    def format_followup_prompt(question: str, chat_history: str) -> str:  # Helper for simple chat
        """
        Format a follow-up prompt (no new context needed)
        """
        return TutorPrompts.FOLLOWUP_TEMPLATE.format(
            question=question,
            chat_history=chat_history
        )
    
    @staticmethod
    def format_example_prompt(  # Helper to fill in the "give examples" script
        concept: str,
        context: str,
        num_examples: int = 2
    ) -> str:
        """
        Format a prompt to generate examples
        """
        return TutorPrompts.EXAMPLE_TEMPLATE.format(
            concept=concept,
            context=context,
            num_examples=num_examples
        )
    
    @staticmethod
    def format_simplify_prompt(original_text: str) -> str:  # Helper to fill in the "make simple" script
        """
        Format a prompt to simplify text
        """
        return TutorPrompts.SIMPLIFY_TEMPLATE.format(
            original_text=original_text
        )
    
    @staticmethod
    def get_system_prompt() -> str:  # Helper to get the AI's core identity
        """Get the system prompt defining tutor personality"""
        return TutorPrompts.SYSTEM_PROMPT


if __name__ == "__main__":  # Code for manual testing
    # Example usage
    prompts = TutorPrompts()  # Create instance
    
    # Test filling in a RAG prompt
    sample_context = "Machine learning is a subset of AI that enables computers to learn from data."
    sample_question = "What is machine learning?"
    
    rag_prompt = prompts.format_rag_prompt(sample_question, sample_context)  # Run formatting
    print("=== RAG Prompt ===")  # Print header
    print(rag_prompt[:200] + "...")  # Print start of the result
    
    print("\n=== System Prompt ===")  # Print header
    print(prompts.get_system_prompt()[:200] + "...")  # Print start of personality script
