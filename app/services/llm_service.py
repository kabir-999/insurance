import asyncio
import google.generativeai as genai
import os
import traceback

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

async def get_answer_from_llm(question: str, context: str) -> str:
    """Gets an answer from Gemini 1.5 Flash using the provided question and context."""
    if not context:
        return "I could not find any relevant information in the document to answer your question."

    prompt = f"""
You are analyzing an insurance policy document. Based exclusively on the following context from the insurance document, please provide a clear, detailed, and specific answer to the question.

Important instructions:
- Only use the given context. If the answer is not present, say so.
- Do not make up information.
- If the context is insufficient, state that.

Context:
{context}

Question:
{question}
"""
    try:
        loop = asyncio.get_event_loop()
        def gemini_generate():
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
        answer = await loop.run_in_executor(None, gemini_generate)
        print(f"DEBUG: Gemini LLM answer: {answer}")
        return answer
    except Exception as e:
        print(f"An error occurred during Gemini LLM generation: {e}")
        tb = traceback.format_exc()
        return f"LLM ERROR: {e}\n{tb}"
