import os
import json
from flash_prompt import send_full_discrepancy_request

# Set your Gemini API key here or load from a local .txt file
with open("gemini_api_key.txt", "r") as f:
    api_key = f.read().strip()

# System prompt for the Gemini model
discrepancy_system_prompt = """
You are a discrepancy checker. Compare the ground truth JSON with a free-form text. For each key in the JSON (e.g., "secondary_school", "higher_education"), return a single boolean: "true" if there is any discrepancy between the ground truth and the free-form text for that key, and "false" if there is none. For nested objects or arrays, do not break them down further; treat them as a single unit. For empty arrays, return "false".

Return only a JSON object with the same top-level keys as the ground truth and nothing else (DO NOT ADD ANY EXTRA KEYS, thank you). Do not include any additional text or explanations.
"""

# Example ground truth (structured data)
ground_truth_data = {
    "secondary_school": {
        "name": "Ålborg Katedralskole",
        "graduation_year": 2009
    },
    "higher_education": []
}

# Example free-form summary text
summary_text = "In 2009, Jørgensen graduated from Ålborg Katedralskole with a secondary school diploma."

# Send the prompt to Gemini and print the result
try:
    result = send_full_discrepancy_request(
        system_prompt=discrepancy_system_prompt,
        ground_truth=ground_truth_data,
        summary_text=summary_text,
        api_key=api_key
    )
    print("Discrepancy Results:")
    print(json.dumps(result, indent=2))
except Exception as e:
    print("An error occurred:", e)