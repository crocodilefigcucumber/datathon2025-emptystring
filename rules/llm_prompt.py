import requests
import json


def send_full_discrepancy_request(
    system_prompt: str,
    ground_truth: dict,
    summary_text: str,
    api_key: str,
    model: str = "accounts/fireworks/models/mixtral-8x7b-instruct",
    max_tokens: int = 128,
    temperature: int = 0,
    top_p: float = 1.0,
    top_k: int = 50,
    stop: list = ["<|im_end|>"],
) -> dict:
    """
    Sends a prompt to the LLM to check for discrepancies between the provided ground truth JSON
    and the free-form summary text. The LLM is instructed to return a JSON object with the same structure
    as the ground truth, where each value is replaced by a boolean indicating a discrepancy (true) or not (false).

    Args:
        system_prompt (str): The system prompt with instructions for discrepancy checking.
        ground_truth (dict): The ground truth JSON data.
        summary_text (str): The free-form text (e.g., wealth summary).
        api_key (str): API key for authentication.
        model (str): The model to be used.
        max_tokens (int): Maximum tokens allowed in the response.
        temperature (int): Temperature for controlling randomness.
        top_p (float): Top-p sampling parameter.
        top_k (int): Top-k sampling parameter.
        stop (list): List of stop tokens.

    Returns:
        dict: A JSON object with the same structure as ground_truth where each entry's value is a boolean.
    """
    prompt = (
        f"{system_prompt}"
        f"Ground truth JSON:\n{json.dumps(ground_truth, indent=2)}\n\n"
        f"Free-form text:\n{summary_text}\n"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop": stop,
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    response = requests.post(
        "https://api.fireworks.ai/inference/v1/completions",
        headers=headers,
        data=json.dumps(payload),
    )

    if response.status_code != 200:
        raise Exception(
            f"API request failed with status code {response.status_code}: {response.text}"
        )

    resp_json = response.json()
    if "choices" not in resp_json:
        raise ValueError("Invalid response format. 'choices' not found in response.")

    raw_output = resp_json["choices"][0]["text"].strip()

    # Remove markdown formatting if present
    if raw_output.startswith("```") and raw_output.endswith("```"):
        raw_output = raw_output.strip("```").strip()

    # Remove any text before the first '{' character
    json_start = raw_output.find("{")
    if json_start != -1:
        raw_output = raw_output[json_start:]

    try:
        parsed = json.loads(raw_output)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}\nRaw model output:\n{raw_output}")


if __name__ == "__main__":
    # Example ground truth JSON data
    ground_truth_data = {
        "secondary_school": {"name": "Ålborg Katedralskole", "graduation_year": 2022},
        "higher_education": [],
    }

    # Example free-form text
    summary_text = (
        "In 2009, Jørgensen graduated from Ålborg Katedralskole with a secondary school diploma.\n",
    )

    # System prompt instructing the LLM to fill out discrepancy info for each entry.
    discrepancy_system_prompt = """
You are a discrepancy checker. Compare the ground truth JSON with a free-form text. For each key in the JSON (e.g., "secondary_school", "higher_education"), return a single boolean: "true" if there is any discrepancy between the ground truth and the free-form text for that key, and "false" if there is none. For nested objects or arrays, do not break them down further; treat them as a single unit. For empty arrays, return "false".
Return only a JSON object with the same top-level keys as the ground truth and nothing else. Do not include any additional text or explanations.
    """

    # Load API key from file
    with open("./api_key.txt", "r") as file:
        api_key = file.read().strip()

    try:
        discrepancies = send_full_discrepancy_request(
            discrepancy_system_prompt, ground_truth_data, summary_text, api_key
        )
        print("Discrepancy JSON:")
        print(json.dumps(discrepancies, indent=2))
    except Exception as e:
        print("An error occurred:", e)
