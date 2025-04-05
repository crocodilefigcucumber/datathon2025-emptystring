import requests
import json

# Load the API key from the file
with open("./api_key.txt", "r") as file:
    api_key = file.read().strip()

url = "https://api.fireworks.ai/inference/v1/completions"
payload = {
    "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "max_tokens": 64,
    "top_p": 1,
    "top_k": 40,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "temperature": 0.1,
    "prompt": "Hello, how are you?",
}
headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

print("hi")
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
print("bye")

# Print the response for debugging
print(response.status_code)
print(response.json())
