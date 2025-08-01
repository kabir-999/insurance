#!/usr/bin/env python3

import requests
import json

# Test the local API
url = "http://localhost:8000/api/v1/hackrx/run"

payload = {
    "documents": "https://hackrx.blob.core.windows.net/assets/arogya%20sanjeevani%20policy%20%20national%20%20uin%20nichlip25041v0224.pdf",
    "questions": [
        "What diseases or treatments are excluded?",
        "What is the sum insured under this policy for cataract treatments?"
    ]
}

headers = {
    'Content-Type': 'application/json'
}

print("Sending request to local API...")
response = requests.post(url, headers=headers, data=json.dumps(payload))

print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
