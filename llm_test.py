import requests
import json

# Load API key
with open("./api_key.txt", "r") as file:
    api_key = file.read().strip()

# Load Wealth Summary from JSON file
with open("data/client_2/client_description.json", "r") as f:
    data = json.load(f)

wealth_summary = data.get("Wealth Summary", "")
if not wealth_summary:
    raise ValueError("Missing 'Wealth Summary' in client_description.json")

# System prompt describing the extraction format
system_prompt = """
You are a data extraction system. Parse the following text and output a JSON object in exactly this format:

{
  "aum": {
    "savings": <int>,
    "inheritance": <int>,
    "real_estate_value": <int>
  },
  "inheritance_details": {
    "relationship": <string>,
    "inheritance year": <int>,
    "profession": <string>
  },
  "real_estate_details": [
    {
      "property type": <string>,
      "property value": <int>,
      "property location": <string>
    }
  ]
}

Respond ONLY with valid JSON. Do NOT include markdown, comments, or explanations.
"""

# Payload without seed
payload = {
    "model": "accounts/fireworks/models/mixtral-8x7b-instruct",
    "prompt": f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{wealth_summary}<|im_end|>",
    "max_tokens": 512,
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 50,
    "stop": ["<|im_end|>"],
}

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

# Make the request
response = requests.post(
    "https://api.fireworks.ai/inference/v1/completions",
    headers=headers,
    data=json.dumps(payload),
)

# Handle errors or missing keys
if response.status_code != 200:
    print("API request failed.")
    print("Status code:", response.status_code)
    print("Response:", response.text)
    exit(1)

resp_json = response.json()
if "choices" not in resp_json:
    print("Invalid response format. 'choices' not found.")
    print(json.dumps(resp_json, indent=2))
    exit(1)

# Extract model output
raw_output = resp_json["choices"][0]["text"]

# Clean up markdown formatting if present
if "```" in raw_output:
    raw_output = raw_output.strip("```json").strip("```")

# Attempt to parse the output as JSON
try:
    parsed = json.loads(raw_output)
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print("JSON parsing failed:", e)
    print("Raw model output:\n", raw_output)
