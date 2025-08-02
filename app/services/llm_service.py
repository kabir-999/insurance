import google.generativeai as genai
import asyncio
from app.core.config import GOOGLE_API_KEY
from dotenv import load_dotenv
load_dotenv()   
# Configure the generative AI model
genai.configure(api_key=GOOGLE_API_KEY)

async def get_answer_from_llm(question: str, context: str) -> str:
    """Generates an answer to a question based on the given context using an LLM."""
    if not context:
        return "I could not find any relevant information in the document to answer your question."

    # Construct a prompt that instructs the model to use only the provided context
    prompt = f"""
    You are analyzing an insurance policy document. Based exclusively on the following context from the insurance document, please provide a clear, detailed, and specific answer to the question.
    
    Important instructions:
    - Use ONLY the information provided in the context below
    - Be specific with amounts, percentages, and conditions mentioned
    - If the context contains relevant information, provide a comprehensive answer
    - Include specific details like coverage amounts, waiting periods, exclusions, etc.
    - Do not make assumptions or add information not in the context

    Context from Insurance Document:
    {context}

    Question: {question}

    Detailed Answer:
    """

    try:
        # Using AGGRESSIVE model config for sub-10s performance
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.05,  # Lower temperature for speed
                max_output_tokens=300,  # Reduced for speed while maintaining quality
                top_p=0.9,  # Higher for better quality
                top_k=50,  # More choices for quality
                candidate_count=1,  # Single candidate for maximum speed
            )
        )
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, model.generate_content, prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred during LLM generation: {e}")
        return "An error occurred while generating the answer."
