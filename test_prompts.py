import json
import requests

# Load prompts from prompts.json
with open("prompts.json", "r") as file:
    prompts = json.load(file)

# Mistral API Endpoint
API_URL = "https://api.mistral.ai/v1/chat/completions"
API_KEY = "DJq34gDXZSrve2zGjR7EzWI9VgtHMiUu" 

# Function to get response from Mistral AI
def get_mistral_response(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "mistral-tiny",  
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }

    response = requests.post(API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error: {response.status_code}, {response.text}"

# Test all prompts
for key, prompt in prompts.items():
    print(f"Prompt [{key}]: {prompt}")
    response = get_mistral_response(prompt)
    print(f"Response [{key}]: {response}\n")
