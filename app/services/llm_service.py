import google.generativeai as genai
from app.core.config import GOOGLE_API_KEY
from dotenv import load_dotenv
load_dotenv()   
# Configure the generative AI model
genai.configure(api_key=GOOGLE_API_KEY)

def get_answer_from_llm(question: str, context: list[str]) -> str:
    """Generates an answer to a question based on the given context using an LLM."""
    if not context:
        return "I could not find any relevant information in the document to answer your question."

    # Construct a prompt that instructs the model to use only the provided context
    prompt = f"""
    Based exclusively on the following context, please provide a clear and concise answer to the question.
    Do not use any information outside of the provided text.

    Context:
    {' '.join(context)}

    Question: {question}

    Answer:
    """

    try:
        # Using a model that is good for question answering
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"An error occurred during LLM generation: {e}")
        return "An error occurred while generating the answer."
