#!/usr/bin/env python3

import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def test_embedding_dimensions():
    test_text = "This is a test document for insurance claims."
    
    models_to_test = [
        "models/embedding-001",
        "models/text-embedding-004",
    ]
    
    for model in models_to_test:
        try:
            result = genai.embed_content(
                model=model,
                content=[test_text],
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            if result and 'embedding' in result:
                dimensions = len(result['embedding'][0])
                print(f"{model}: {dimensions} dimensions")
            else:
                print(f"{model}: No embedding returned")
                
        except Exception as e:
            print(f"{model}: Error - {e}")

if __name__ == "__main__":
    test_embedding_dimensions()
